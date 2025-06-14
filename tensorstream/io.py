"""
I/O utilities for TensorStream.

Implements the .ts file format for efficient tensor serialization and
provides utilities for reading/writing tensor data with metadata.
"""

import hashlib
import struct
import warnings
import zlib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, BinaryIO

import numpy as np
import torch

from .exceptions import StorageError


# Magic number for .ts files
TS_MAGIC = b'TSTR'
TS_VERSION = 1

# Data type mappings
TORCH_TO_NUMPY_DTYPE = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.float16: np.float16,
    torch.bfloat16: np.float32,  # bfloat16 not directly supported in numpy
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.uint8: np.uint8,
    torch.bool: np.bool_,
}

NUMPY_TO_TORCH_DTYPE = {v: k for k, v in TORCH_TO_NUMPY_DTYPE.items()}


class TSFileHeader:
    """Header structure for .ts files."""
    
    def __init__(self):
        self.magic = TS_MAGIC
        self.version = TS_VERSION
        self.tensor_dtype = None
        self.tensor_shape = None
        self.compressed = False
        self.compression_level = 0
        self.checksum = None
        self.metadata = {}
        self.header_size = 0
        self.data_size = 0
        self.compressed_size = 0
    
    def to_bytes(self) -> bytes:
        """Serialize header to bytes."""
        # Prepare metadata
        metadata_str = str(self.metadata).encode('utf-8')
        
        # Create header data
        header_data = struct.pack(
            '<4sI',  # magic (4 bytes), version (4 bytes)
            self.magic,
            self.version
        )
        
        # Add tensor info
        dtype_str = str(self.tensor_dtype).encode('utf-8')
        shape_bytes = struct.pack('<I', len(self.tensor_shape))
        shape_bytes += struct.pack(f'<{len(self.tensor_shape)}q', *self.tensor_shape)
        
        header_data += struct.pack('<I', len(dtype_str)) + dtype_str
        header_data += shape_bytes
        
        # Add compression info
        header_data += struct.pack('<BI', self.compressed, self.compression_level)
        
        # Add sizes
        header_data += struct.pack('<QQQ', self.data_size, self.compressed_size, 0)  # Reserved
        
        # Add checksum
        if self.checksum:
            header_data += struct.pack('<32s', self.checksum)
        else:
            header_data += b'\x00' * 32
        
        # Add metadata
        header_data += struct.pack('<I', len(metadata_str)) + metadata_str
        
        # Update header size
        self.header_size = len(header_data) + 4  # +4 for header size field
        header_data = struct.pack('<I', self.header_size) + header_data
        
        return header_data
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'TSFileHeader':
        """Deserialize header from bytes."""
        header = cls()
        offset = 0
        
        # Read header size
        header.header_size = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        
        # Read magic and version
        header.magic, header.version = struct.unpack('<4sI', data[offset:offset+8])
        offset += 8
        
        if header.magic != TS_MAGIC:
            raise StorageError("read", "header", f"Invalid magic number: {header.magic}")
        
        if header.version != TS_VERSION:
            raise StorageError("read", "header", f"Unsupported version: {header.version}")
        
        # Read tensor dtype
        dtype_len = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        dtype_str = data[offset:offset+dtype_len].decode('utf-8')
        header.tensor_dtype = dtype_str
        offset += dtype_len
        
        # Read tensor shape
        shape_len = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        shape_data = struct.unpack(f'<{shape_len}q', data[offset:offset+shape_len*8])
        header.tensor_shape = list(shape_data)
        offset += shape_len * 8
        
        # Read compression info
        compressed_int, header.compression_level = struct.unpack('<BI', data[offset:offset+5])
        header.compressed = bool(compressed_int)
        offset += 5
        
        # Read sizes
        header.data_size, header.compressed_size, _ = struct.unpack('<QQQ', data[offset:offset+24])
        offset += 24
        
        # Read checksum
        header.checksum = data[offset:offset+32]
        if header.checksum == b'\x00' * 32:
            header.checksum = None
        offset += 32
        
        # Read metadata
        metadata_len = struct.unpack('<I', data[offset:offset+4])[0]
        offset += 4
        if metadata_len > 0:
            metadata_str = data[offset:offset+metadata_len].decode('utf-8')
            try:
                header.metadata = eval(metadata_str)  # Simple eval for basic types
            except:
                header.metadata = {}
        
        return header


def save_to_ts(tensor: torch.Tensor, path: Union[str, Path], 
               metadata: Optional[Dict[str, Any]] = None,
               compress: bool = False, compression_level: int = 6,
               verify_checksum: bool = True) -> None:
    """
    Save a PyTorch tensor to .ts format.
    
    Args:
        tensor: The tensor to save
        path: Output file path
        metadata: Optional metadata dictionary
        compress: Whether to compress the data
        compression_level: Compression level (1-9)
        verify_checksum: Whether to compute and store checksum
    """
    path = Path(path)
    
    try:
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert tensor to numpy for serialization
        if tensor.dtype == torch.bfloat16:
            # Handle bfloat16 specially
            numpy_tensor = tensor.float().cpu().numpy().astype(np.float32)
            stored_dtype = torch.bfloat16
        else:
            numpy_tensor = tensor.cpu().numpy()
            stored_dtype = tensor.dtype
        
        # Get raw bytes
        tensor_bytes = numpy_tensor.tobytes()
        
        # Compress if requested
        compressed_bytes = None
        if compress:
            compressed_bytes = zlib.compress(tensor_bytes, compression_level)
        
        # Compute checksum
        checksum = None
        if verify_checksum:
            checksum = hashlib.sha256(tensor_bytes).digest()
        
        # Create header
        header = TSFileHeader()
        header.tensor_dtype = str(stored_dtype)
        header.tensor_shape = list(tensor.shape)
        header.compressed = compress
        header.compression_level = compression_level if compress else 0
        header.metadata = metadata or {}
        header.data_size = len(tensor_bytes)
        header.compressed_size = len(compressed_bytes) if compressed_bytes else 0
        header.checksum = checksum
        
        # Write file
        with open(path, 'wb') as f:
            # Write header
            header_bytes = header.to_bytes()
            f.write(header_bytes)
            
            # Write data
            if compress and compressed_bytes:
                f.write(compressed_bytes)
            else:
                f.write(tensor_bytes)
    
    except Exception as e:
        raise StorageError("write", str(path), str(e))


def read_ts_header(path: Union[str, Path]) -> TSFileHeader:
    """
    Read header information from a .ts file.
    
    Args:
        path: Path to the .ts file
        
    Returns:
        TSFileHeader object containing file metadata
    """
    path = Path(path)
    
    if not path.exists():
        raise StorageError("read", str(path), "File does not exist")
    
    try:
        with open(path, 'rb') as f:
            # Read header size first
            header_size_bytes = f.read(4)
            if len(header_size_bytes) < 4:
                raise StorageError("read", str(path), "File too small to contain valid header")
            
            header_size = struct.unpack('<I', header_size_bytes)[0]
            
            # Read full header
            f.seek(0)
            header_bytes = f.read(header_size)
            
            if len(header_bytes) < header_size:
                raise StorageError("read", str(path), "Incomplete header")
            
            return TSFileHeader.from_bytes(header_bytes)
    
    except Exception as e:
        if isinstance(e, StorageError):
            raise
        raise StorageError("read", str(path), str(e))


def load_from_ts(path: Union[str, Path], device: Optional[torch.device] = None,
                 verify_checksum: bool = True) -> torch.Tensor:
    """
    Load a PyTorch tensor from .ts format.
    
    Args:
        path: Path to the .ts file
        device: Device to load tensor to
        verify_checksum: Whether to verify data integrity
        
    Returns:
        Loaded PyTorch tensor
    """
    path = Path(path)
    
    try:
        # Read header
        header = read_ts_header(path)
        
        # Read data
        with open(path, 'rb') as f:
            f.seek(header.header_size)
            
            if header.compressed:
                compressed_data = f.read(header.compressed_size)
                tensor_bytes = zlib.decompress(compressed_data)
            else:
                tensor_bytes = f.read(header.data_size)
        
        # Verify checksum
        if verify_checksum and header.checksum:
            computed_checksum = hashlib.sha256(tensor_bytes).digest()
            if computed_checksum != header.checksum:
                raise StorageError("read", str(path), "Checksum verification failed")
        
        # Parse dtype
        if header.tensor_dtype.startswith('torch.'):
            torch_dtype = getattr(torch, header.tensor_dtype.split('.')[-1])
        else:
            # Handle string representations
            torch_dtype = eval(header.tensor_dtype)
        
        # Convert bytes back to tensor
        if torch_dtype == torch.bfloat16:
            # Handle bfloat16 specially
            numpy_array = np.frombuffer(tensor_bytes, dtype=np.float32)
            tensor = torch.from_numpy(numpy_array).to(torch.bfloat16)
        else:
            numpy_dtype = TORCH_TO_NUMPY_DTYPE.get(torch_dtype)
            if numpy_dtype is None:
                raise StorageError("read", str(path), f"Unsupported dtype: {torch_dtype}")
            
            numpy_array = np.frombuffer(tensor_bytes, dtype=numpy_dtype)
            tensor = torch.from_numpy(numpy_array)
        
        # Reshape tensor
        tensor = tensor.reshape(header.tensor_shape)
        
        # Move to device if specified
        if device is not None:
            tensor = tensor.to(device)
        
        return tensor
    
    except Exception as e:
        if isinstance(e, StorageError):
            raise
        raise StorageError("read", str(path), str(e))


def get_ts_file_info(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about a .ts file without loading the tensor.
    
    Args:
        path: Path to the .ts file
        
    Returns:
        Dictionary containing file information
    """
    header = read_ts_header(path)
    
    return {
        "dtype": header.tensor_dtype,
        "shape": header.tensor_shape,
        "compressed": header.compressed,
        "compression_level": header.compression_level,
        "data_size": header.data_size,
        "compressed_size": header.compressed_size,
        "has_checksum": header.checksum is not None,
        "metadata": header.metadata,
        "file_size": Path(path).stat().st_size,
    }


def shard_tensor(tensor: torch.Tensor, num_shards: int) -> list[torch.Tensor]:
    """
    Split a tensor into multiple shards along the first dimension.
    
    Args:
        tensor: Tensor to shard
        num_shards: Number of shards to create
        
    Returns:
        List of tensor shards
    """
    if num_shards <= 1:
        return [tensor]
    
    dim_size = tensor.shape[0]
    shard_size = (dim_size + num_shards - 1) // num_shards  # Ceiling division
    
    shards = []
    for i in range(num_shards):
        start_idx = i * shard_size
        end_idx = min((i + 1) * shard_size, dim_size)
        
        if start_idx < dim_size:
            shard = tensor[start_idx:end_idx]
            shards.append(shard)
    
    return shards


def combine_shards(shards: list[torch.Tensor]) -> torch.Tensor:
    """
    Combine tensor shards back into a single tensor.
    
    Args:
        shards: List of tensor shards
        
    Returns:
        Combined tensor
    """
    if len(shards) == 1:
        return shards[0]
    
    return torch.cat(shards, dim=0)
