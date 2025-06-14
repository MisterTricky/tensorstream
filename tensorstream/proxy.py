"""
Proxy Layer implementation for TensorStream.

Proxy layers replace original model layers and intercept forward passes
to ensure tensors are loaded just-in-time from storage.
"""

import weakref
from typing import Any, Dict, Optional, Callable, Tuple
import torch
import torch.nn as nn

from .exceptions import LayerError
from .orchestrator import OrchestrationEngine


class TensorStreamProxyLayer(nn.Module):
    """
    Proxy layer that replaces original model layers.
    
    This class acts as a transparent proxy for the original layer,
    intercepting forward passes to ensure weights are loaded from
    storage just-in-time.
    """
    
    def __init__(self, layer_id: str, original_layer: nn.Module, 
                 orchestrator: OrchestrationEngine, 
                 layer_metadata: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        self.layer_id = layer_id
        self.original_layer = original_layer
        self.orchestrator_ref = weakref.ref(orchestrator)
        self.layer_metadata = layer_metadata or {}
        
        # Store original state for restoration if needed
        self._original_state_dict = None
        self._weights_loaded = False
        self._forward_hooks = []
        
        # Register forward pre-hook
        self.register_forward_pre_hook(self._pre_forward_hook)
        
        # Copy relevant attributes from original layer
        self._copy_layer_attributes()
    
    def _copy_layer_attributes(self) -> None:
        """Copy important attributes from the original layer."""
        # Copy training state
        self.training = self.original_layer.training
        
        # Copy dtype if available (device is handled by the property)
        if hasattr(self.original_layer, 'weight') and self.original_layer.weight is not None:
            self.dtype = self.original_layer.weight.dtype
        
        # Copy other important attributes that might be accessed
        for attr in ['in_features', 'out_features', 'hidden_size', 'num_attention_heads']:
            if hasattr(self.original_layer, attr):
                setattr(self, attr, getattr(self.original_layer, attr))
    
    def _pre_forward_hook(self, module: nn.Module, input: Tuple[torch.Tensor, ...]) -> None:
        """
        Pre-forward hook that ensures weights are loaded.
        
        This hook is called before the forward pass and ensures that
        the layer's weights are loaded into VRAM.
        """
        orchestrator = self.orchestrator_ref()
        if orchestrator is None:
            raise LayerError(
                self.layer_id, 
                "forward", 
                "Orchestrator reference is no longer valid"
            )
        
        try:
            # Request the layer to be loaded with high priority
            tensor = orchestrator.request_layer(self.layer_id, priority=10)
            
            # If this is the first time loading, restore the state
            if not self._weights_loaded:
                self._restore_weights(tensor)
                self._weights_loaded = True
            
        except Exception as e:
            raise LayerError(
                self.layer_id,
                "forward",
                f"Failed to load layer weights: {e}"
            )
    
    def _restore_weights(self, tensor: torch.Tensor) -> None:
        """
        Restore the layer's weights from the loaded tensor.
        
        This method reconstructs the original layer's parameter structure
        from the flattened tensor data.
        """
        if self._original_state_dict is None:
            # If we don't have the original state dict, we need to reconstruct it
            # This is a complex operation that depends on the layer type
            self._reconstruct_state_dict(tensor)
        else:
            # Restore from saved state dict
            self.original_layer.load_state_dict(self._original_state_dict)
    
    def _reconstruct_state_dict(self, tensor: torch.Tensor) -> None:
        """
        Reconstruct the layer's state dict from a flattened tensor.
        
        This is a complex operation that requires knowledge of the original
        layer's parameter structure.
        """
        # This is a simplified implementation
        # In practice, this would need to handle various layer types differently
        
        if hasattr(self.original_layer, 'weight') and self.original_layer.weight is not None:
            # For layers with weight parameters
            weight_size = self.original_layer.weight.numel()
            
            if len(tensor) >= weight_size:
                # Reshape tensor to match weight shape
                weight_data = tensor[:weight_size].view(self.original_layer.weight.shape)
                self.original_layer.weight.data = weight_data.to(
                    device=self.original_layer.weight.device,
                    dtype=self.original_layer.weight.dtype
                )
                
                # Handle bias if present
                if (hasattr(self.original_layer, 'bias') and 
                    self.original_layer.bias is not None and
                    len(tensor) > weight_size):
                    
                    bias_size = self.original_layer.bias.numel()
                    if len(tensor) >= weight_size + bias_size:
                        bias_data = tensor[weight_size:weight_size + bias_size].view(
                            self.original_layer.bias.shape
                        )
                        self.original_layer.bias.data = bias_data.to(
                            device=self.original_layer.bias.device,
                            dtype=self.original_layer.bias.dtype
                        )
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass through the original layer.
        
        The pre-forward hook ensures weights are loaded before this is called.
        """
        try:
            return self.original_layer(*args, **kwargs)
        except Exception as e:
            raise LayerError(
                self.layer_id,
                "forward",
                f"Forward pass failed: {e}"
            )
    
    def train(self, mode: bool = True) -> 'TensorStreamProxyLayer':
        """Set training mode."""
        super().train(mode)
        self.original_layer.train(mode)
        return self
    
    def eval(self) -> 'TensorStreamProxyLayer':
        """Set evaluation mode."""
        return self.train(False)
    
    def to(self, *args, **kwargs) -> 'TensorStreamProxyLayer':
        """Move layer to device/dtype."""
        super().to(*args, **kwargs)
        self.original_layer.to(*args, **kwargs)
        return self
    
    def cuda(self, device: Optional[int] = None) -> 'TensorStreamProxyLayer':
        """Move layer to CUDA device."""
        super().cuda(device)
        self.original_layer.cuda(device)
        return self
    
    def cpu(self) -> 'TensorStreamProxyLayer':
        """Move layer to CPU."""
        super().cpu()
        self.original_layer.cpu()
        return self
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Get state dict from original layer."""
        return self.original_layer.state_dict(destination, prefix, keep_vars)
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict into original layer."""
        return self.original_layer.load_state_dict(state_dict, strict)
    
    def parameters(self, recurse=True):
        """Get parameters from original layer."""
        return self.original_layer.parameters(recurse)
    
    def named_parameters(self, prefix='', recurse=True):
        """Get named parameters from original layer."""
        return self.original_layer.named_parameters(prefix, recurse)
    
    def modules(self):
        """Get modules from original layer."""
        return self.original_layer.modules()
    
    def named_modules(self, memo=None, prefix='', remove_duplicate=True):
        """Get named modules from original layer."""
        return self.original_layer.named_modules(memo, prefix, remove_duplicate)
    
    def children(self):
        """Get child modules from original layer."""
        return self.original_layer.children()
    
    def named_children(self):
        """Get named child modules from original layer."""
        return self.original_layer.named_children()
    
    def buffers(self, recurse=True):
        """Get buffers from original layer."""
        return self.original_layer.buffers(recurse)
    
    def named_buffers(self, prefix='', recurse=True):
        """Get named buffers from original layer."""
        return self.original_layer.named_buffers(prefix, recurse)
    
    @property
    def device(self):
        """Get device of the original layer."""
        try:
            # Try to get device from parameters
            param = next(self.original_layer.parameters(), None)
            if param is not None:
                return param.device
        except (StopIteration, RuntimeError):
            pass
        
        try:
            # Try to get device from buffers
            buffer = next(self.original_layer.buffers(), None)
            if buffer is not None:
                return buffer.device
        except (StopIteration, RuntimeError):
            pass
        
        # Default to CPU if no parameters/buffers found
        return torch.device('cpu')
    
    def get_original_layer(self) -> nn.Module:
        """Get the original layer (for debugging/inspection)."""
        return self.original_layer
    
    def get_layer_metadata(self) -> Dict[str, Any]:
        """Get layer metadata."""
        orchestrator = self.orchestrator_ref()
        layer_info = orchestrator.get_layer_info(self.layer_id) if orchestrator else {}
        
        return {
            "layer_id": self.layer_id,
            "layer_type": type(self.original_layer).__name__,
            "weights_loaded": self._weights_loaded,
            "metadata": self.layer_metadata,
            "orchestrator_info": layer_info,
        }
    
    def __repr__(self) -> str:
        return (f"TensorStreamProxyLayer({self.layer_id}, "
                f"original={type(self.original_layer).__name__}, "
                f"loaded={self._weights_loaded})")


class LayerRegistry:
    """
    Registry for managing proxy layers and their relationships.
    
    This class helps track which layers have been replaced with proxies
    and provides utilities for managing the proxy layer lifecycle.
    """
    
    def __init__(self):
        self.proxy_layers: Dict[str, TensorStreamProxyLayer] = {}
        self.original_layers: Dict[str, nn.Module] = {}
        self.layer_hierarchy: Dict[str, str] = {}  # child -> parent mapping
    
    def register_proxy(self, layer_id: str, proxy_layer: TensorStreamProxyLayer, 
                      original_layer: nn.Module, parent_id: Optional[str] = None) -> None:
        """Register a proxy layer."""
        self.proxy_layers[layer_id] = proxy_layer
        self.original_layers[layer_id] = original_layer
        
        if parent_id:
            self.layer_hierarchy[layer_id] = parent_id
    
    def get_proxy(self, layer_id: str) -> Optional[TensorStreamProxyLayer]:
        """Get a proxy layer by ID."""
        return self.proxy_layers.get(layer_id)
    
    def get_original(self, layer_id: str) -> Optional[nn.Module]:
        """Get the original layer by ID."""
        return self.original_layers.get(layer_id)
    
    def get_all_proxies(self) -> Dict[str, TensorStreamProxyLayer]:
        """Get all registered proxy layers."""
        return self.proxy_layers.copy()
    
    def remove_proxy(self, layer_id: str) -> bool:
        """Remove a proxy layer from the registry."""
        if layer_id in self.proxy_layers:
            del self.proxy_layers[layer_id]
            del self.original_layers[layer_id]
            if layer_id in self.layer_hierarchy:
                del self.layer_hierarchy[layer_id]
            return True
        return False
    
    def get_layer_hierarchy(self) -> Dict[str, str]:
        """Get the layer hierarchy mapping."""
        return self.layer_hierarchy.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics."""
        loaded_count = sum(1 for proxy in self.proxy_layers.values() 
                          if proxy._weights_loaded)
        
        return {
            "total_proxies": len(self.proxy_layers),
            "loaded_proxies": loaded_count,
            "unloaded_proxies": len(self.proxy_layers) - loaded_count,
            "hierarchy_depth": len(set(self.layer_hierarchy.values())),
        }
