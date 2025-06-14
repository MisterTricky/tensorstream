"""
Main API for TensorStream.

Provides the primary user-facing interface for tensor streaming operations.
"""

import inspect
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import torch
import torch.nn as nn

from .config import Config, create_default_config
from .orchestrator import OrchestrationEngine
from .proxy import TensorStreamProxyLayer, LayerRegistry
from .io import save_to_ts, get_ts_file_info
from .exceptions import TensorStreamError, ConfigurationError


def offload(model: nn.Module, config: Union[Config, str, Path]) -> nn.Module:
    """
    Apply TensorStream offloading to a PyTorch model.
    
    This function analyzes the model, shards its layers to disk, and replaces
    them with proxy layers that load weights just-in-time during inference.
    
    Args:
        model: The PyTorch model to offload
        config: TensorStream configuration (Config object, or path to storage)
        
    Returns:
        The modified model with proxy layers
        
    Example:
        >>> import torch
        >>> import tensorstream
        >>> from transformers import AutoModelForCausalLM
        >>> 
        >>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
        >>> config = tensorstream.Config(storage_path="/tmp/tensorstream")
        >>> offloaded_model = tensorstream.offload(model, config)
        >>> 
        >>> # Use as normal
        >>> offloaded_model.to('cuda:0')
        >>> output = offloaded_model.generate(...)
    """
    # Normalize config
    if isinstance(config, (str, Path)):
        config = create_default_config(config)
    elif not isinstance(config, Config):
        raise ConfigurationError(
            "config",
            "Config must be a Config object or path string"
        )
    
    # Initialize orchestrator
    orchestrator = OrchestrationEngine(config)
    orchestrator.initialize()
    
    # Create layer registry
    registry = LayerRegistry()
    
    try:
        # Analyze and shard the model
        layer_info = _analyze_model(model, config)
        layer_order = _shard_model_layers(model, layer_info, config, orchestrator)
        
        # Set execution order
        orchestrator.set_layer_order(layer_order)
        
        # Replace layers with proxies
        _replace_layers_with_proxies(model, layer_info, orchestrator, registry)
        
        # Attach metadata to model
        model._tensorstream_config = config
        model._tensorstream_orchestrator = orchestrator
        model._tensorstream_registry = registry
        model._tensorstream_layer_order = layer_order
        
        # Add cleanup method
        def cleanup_tensorstream():
            """Cleanup TensorStream resources."""
            if hasattr(model, '_tensorstream_orchestrator'):
                model._tensorstream_orchestrator.cleanup()
        
        model.cleanup_tensorstream = cleanup_tensorstream
        
        return model
    
    except Exception as e:
        # Cleanup on failure
        orchestrator.cleanup()
        raise TensorStreamError(f"Failed to offload model: {e}")


def _analyze_model(model: nn.Module, config: Config) -> Dict[str, Any]:
    """
    Analyze the model to identify layers suitable for offloading.
    
    Args:
        model: The model to analyze
        config: Configuration object
        
    Returns:
        Dictionary containing layer analysis information
    """
    layer_info = {
        "offloadable_layers": {},
        "total_parameters": 0,
        "total_size_bytes": 0,
        "layer_hierarchy": {},
    }
    
    # Find all named modules
    for name, module in model.named_modules():
        if _is_offloadable_layer(module):
            # Calculate layer size
            param_count = sum(p.numel() for p in module.parameters())
            size_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
            
            layer_info["offloadable_layers"][name] = {
                "module": module,
                "param_count": param_count,
                "size_bytes": size_bytes,
                "dtype": next(module.parameters()).dtype if param_count > 0 else torch.float32,
                "device": next(module.parameters()).device if param_count > 0 else torch.device('cpu'),
            }
            
            layer_info["total_parameters"] += param_count
            layer_info["total_size_bytes"] += size_bytes
    
    if config.debug_mode:
        print(f"TensorStream: Found {len(layer_info['offloadable_layers'])} offloadable layers")
        print(f"TensorStream: Total parameters: {layer_info['total_parameters']:,}")
        print(f"TensorStream: Total size: {layer_info['total_size_bytes'] / (1024**3):.2f} GB")
    
    return layer_info


def _is_offloadable_layer(module: nn.Module) -> bool:
    """
    Determine if a module is suitable for offloading.
    
    Args:
        module: The module to check
        
    Returns:
        True if the module should be offloaded
    """
    # Check if module has parameters
    if not list(module.parameters()):
        return False
    
    # Check if it's a leaf module (no child modules with parameters)
    for child in module.children():
        if list(child.parameters()):
            return False
    
    # Skip very small layers (< 1MB)
    size_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
    if size_bytes < 1024 * 1024:  # 1MB threshold
        return False
    
    # Check for supported layer types
    supported_types = (
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.Embedding,
        nn.LayerNorm,
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
    )
    
    if isinstance(module, supported_types):
        return True
    
    # Check for transformer-style layers (common pattern)
    module_name = module.__class__.__name__.lower()
    transformer_keywords = ['attention', 'feedforward', 'mlp', 'linear', 'embed']
    
    if any(keyword in module_name for keyword in transformer_keywords):
        return True
    
    return False


def _shard_model_layers(model: nn.Module, layer_info: Dict[str, Any], 
                       config: Config, orchestrator: OrchestrationEngine) -> List[str]:
    """
    Shard model layers to disk and register with orchestrator.
    
    Args:
        model: The model to shard
        layer_info: Layer analysis information
        config: Configuration object
        orchestrator: Orchestration engine
        
    Returns:
        List of layer IDs in execution order
    """
    layer_order = []
    
    for layer_name, info in layer_info["offloadable_layers"].items():
        module = info["module"]
        
        # Create layer ID
        layer_id = f"layer_{layer_name.replace('.', '_')}"
        layer_order.append(layer_id)
        
        # Serialize layer parameters
        layer_path = config.storage_path / f"{layer_id}.ts"
        
        # Flatten all parameters into a single tensor
        param_tensors = []
        param_shapes = []
        param_names = []
        
        for param_name, param in module.named_parameters():
            param_tensors.append(param.detach().cpu().flatten())
            param_shapes.append(param.shape)
            param_names.append(param_name)
        
        if param_tensors:
            # Concatenate all parameters
            combined_tensor = torch.cat(param_tensors)
            
            # Save metadata
            metadata = {
                "layer_name": layer_name,
                "layer_type": module.__class__.__name__,
                "param_shapes": param_shapes,
                "param_names": param_names,
                "total_params": info["param_count"],
                "original_dtype": str(info["dtype"]),
                "tensorstream_version": "0.1.0",
            }
            
            # Save to disk
            save_to_ts(
                combined_tensor,
                layer_path,
                metadata=metadata,
                compress=config.compression_enabled,
                compression_level=config.compression_level,
                verify_checksum=config.verify_checksums
            )
            
            # Register with orchestrator
            orchestrator.register_layer(layer_id, layer_path, info["size_bytes"])
            
            if config.debug_mode:
                print(f"TensorStream: Sharded layer {layer_name} -> {layer_path}")
    
    return layer_order


def _replace_layers_with_proxies(model: nn.Module, layer_info: Dict[str, Any],
                               orchestrator: OrchestrationEngine, 
                               registry: LayerRegistry) -> None:
    """
    Replace model layers with TensorStream proxy layers.
    
    Args:
        model: The model to modify
        layer_info: Layer analysis information
        orchestrator: Orchestration engine
        registry: Layer registry
    """
    for layer_name, info in layer_info["offloadable_layers"].items():
        # Get the parent module and attribute name
        parent_module, attr_name = _get_parent_module_and_attr(model, layer_name)
        
        if parent_module is None:
            warnings.warn(f"Could not find parent for layer {layer_name}")
            continue
        
        # Get original module
        original_module = getattr(parent_module, attr_name)
        
        # Create layer ID
        layer_id = f"layer_{layer_name.replace('.', '_')}"
        
        # Create proxy layer
        proxy_layer = TensorStreamProxyLayer(
            layer_id=layer_id,
            original_layer=original_module,
            orchestrator=orchestrator,
            layer_metadata={
                "original_name": layer_name,
                "layer_type": original_module.__class__.__name__,
                "param_count": info["param_count"],
                "size_bytes": info["size_bytes"],
            }
        )
        
        # Replace the layer
        setattr(parent_module, attr_name, proxy_layer)
        
        # Register in registry
        registry.register_proxy(layer_id, proxy_layer, original_module)
        
        if hasattr(orchestrator.config, 'debug_mode') and orchestrator.config.debug_mode:
            print(f"TensorStream: Replaced {layer_name} with proxy layer")


def _get_parent_module_and_attr(model: nn.Module, layer_name: str) -> tuple:
    """
    Get the parent module and attribute name for a given layer path.
    
    Args:
        model: The root model
        layer_name: Dot-separated layer name (e.g., "encoder.layers.0.attention")
        
    Returns:
        Tuple of (parent_module, attribute_name)
    """
    if '.' not in layer_name:
        return model, layer_name
    
    parts = layer_name.split('.')
    parent_name = '.'.join(parts[:-1])
    attr_name = parts[-1]
    
    # Navigate to parent module
    parent_module = model
    for part in parts[:-1]:
        if not hasattr(parent_module, part):
            return None, None
        parent_module = getattr(parent_module, part)
    
    return parent_module, attr_name


def get_model_statistics(model: nn.Module) -> Dict[str, Any]:
    """
    Get statistics about a TensorStream-enabled model.
    
    Args:
        model: The TensorStream model
        
    Returns:
        Dictionary containing model statistics
    """
    if not hasattr(model, '_tensorstream_orchestrator'):
        raise TensorStreamError("Model is not TensorStream-enabled")
    
    orchestrator = model._tensorstream_orchestrator
    registry = model._tensorstream_registry
    
    stats = orchestrator.get_statistics()
    stats.update({
        "registry": registry.get_statistics(),
        "model_info": {
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "device": str(next(model.parameters()).device),
            "dtype": str(next(model.parameters()).dtype),
        }
    })
    
    return stats


def optimize_memory_usage(model: nn.Module, target_utilization: float = 0.8) -> Dict[str, Any]:
    """
    Optimize memory usage for a TensorStream model.
    
    Args:
        model: The TensorStream model
        target_utilization: Target VRAM utilization (0.0 to 1.0)
        
    Returns:
        Dictionary containing optimization results
    """
    if not hasattr(model, '_tensorstream_orchestrator'):
        raise TensorStreamError("Model is not TensorStream-enabled")
    
    orchestrator = model._tensorstream_orchestrator
    stats_before = orchestrator.get_statistics()
    
    # Calculate target budget
    current_budget = stats_before["vram_budget_bytes"]
    target_budget = int(current_budget * target_utilization)
    
    # Force eviction if over budget
    if stats_before["vram_usage_bytes"] > target_budget:
        bytes_to_free = stats_before["vram_usage_bytes"] - target_budget
        orchestrator._evict_layers(bytes_to_free)
    
    stats_after = orchestrator.get_statistics()
    
    return {
        "optimization_applied": True,
        "target_utilization": target_utilization,
        "before": {
            "vram_usage": stats_before["vram_usage_bytes"],
            "vram_utilization": stats_before["vram_utilization"],
        },
        "after": {
            "vram_usage": stats_after["vram_usage_bytes"],
            "vram_utilization": stats_after["vram_utilization"],
        },
        "bytes_freed": stats_before["vram_usage_bytes"] - stats_after["vram_usage_bytes"],
    }
