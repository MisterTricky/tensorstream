# TensorStream Examples

This directory contains examples demonstrating various TensorStream features and use cases.

## Quick Start

Before running the examples, ensure you have TensorStream installed:

```bash
# Install TensorStream
pip install tensorstream

# Install example dependencies
pip install transformers torch
```

## Examples Overview

### 1. Basic Usage (`basic_usage.py`)

**Purpose**: Demonstrates the simplest way to use TensorStream with a Hugging Face model.

**Features Covered**:
- Loading a transformer model
- Creating basic TensorStream configuration
- Applying offloading with one line of code
- Text generation with offloaded model
- Basic statistics monitoring

**Run**:
```bash
python basic_usage.py
```

**Expected Output**:
- Model loading and offloading process
- Generated text sample
- Memory usage statistics

### 2. Advanced Configuration (`advanced_config.py`)

**Purpose**: Shows advanced configuration options and performance tuning.

**Features Covered**:
- Advanced configuration parameters
- Backend selection (GPUDirect, CUDA, mmap)
- Prefetching strategies
- Memory pressure management
- Compression settings
- Performance monitoring
- Fallback handling

**Run**:
```bash
python advanced_config.py
```

**Expected Output**:
- System information
- Detailed configuration display
- Performance benchmarks
- Memory usage monitoring
- Text generation with different parameters

### 3. Performance Benchmarking (`benchmark.py`)

**Purpose**: Comprehensive performance benchmarking comparing standard PyTorch with TensorStream.

**Features Covered**:
- Loading time comparison
- Inference speed measurement
- Memory efficiency analysis
- Multiple configuration testing
- Results saving and reporting

**Run**:
```bash
python benchmark.py
```

**Expected Output**:
- Detailed performance metrics
- JSON report file with results
- Comparison summary

## System Requirements

### Minimum Requirements
- Python 3.8+
- PyTorch 1.12+
- 8GB RAM
- 5GB free disk space

### Recommended for Best Performance
- NVIDIA GPU with 4GB+ VRAM
- NVMe SSD storage
- CUDA 11.2+ with GPUDirect Storage
- 16GB+ RAM

## Configuration Examples

### Basic Configuration
```python
import tensorstream

config = tensorstream.Config(
    storage_path="/tmp/tensorstream",
    vram_budget_gb=4.0,
    backend="auto"
)
```

### High-Performance Configuration
```python
from tensorstream.config import BackendType, PrefetchStrategy

config = tensorstream.Config(
    storage_path="/fast/nvme/tensorstream",
    vram_budget_gb=8.0,
    backend=BackendType.GPUDIRECT,
    prefetch_strategy=PrefetchStrategy.ADAPTIVE,
    compression_enabled=True,
    num_io_threads=4
)
```

### Memory-Constrained Configuration
```python
config = tensorstream.Config(
    storage_path="/tmp/tensorstream",
    vram_budget_gb=2.0,
    backend=BackendType.MMAP,
    compression_enabled=True,
    compression_level=9,
    memory_pressure_mode="aggressive"
)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `vram_budget_gb`
   - Enable compression
   - Use more aggressive memory pressure mode

2. **Slow Performance**
   - Use faster storage (NVMe SSD)
   - Enable prefetching
   - Increase `num_io_threads`
   - Use GPUDirect backend if available

3. **Import Errors**
   - Install missing dependencies: `pip install transformers`
   - Update PyTorch: `pip install torch --upgrade`

4. **GPUDirect Not Available**
   - Falls back to CUDA or mmap backend automatically
   - Install CUDA Toolkit 11.2+ and GPUDirect Storage

### Debug Mode

Enable debug mode for detailed logging:

```python
config = tensorstream.Config(
    storage_path="/tmp/tensorstream",
    debug_mode=True
)
```

### Performance Tips

1. **Storage Location**
   - Use fastest available storage (NVMe > SATA SSD > HDD)
   - Avoid network storage for production

2. **Memory Budget**
   - Leave 1-2GB VRAM for PyTorch operations
   - Monitor memory usage during testing

3. **Backend Selection**
   - GPUDirect: Best performance with supported hardware
   - CUDA: Good performance, broader compatibility
   - mmap: Fallback option, works everywhere

4. **Prefetching**
   - `NONE`: Lowest memory usage
   - `NEXT_LAYER`: Good balance
   - `ADAPTIVE`: Best performance

## Model Compatibility

### Tested Models
- ✅ GPT-2 (all sizes)
- ✅ GPT-Neo/GPT-J
- ✅ LLaMA/LLaMA-2
- ✅ Mistral
- ✅ Falcon
- ✅ BERT family
- ✅ T5 family

### Model Size Guidelines
- **Small models (< 1GB)**: May not benefit from offloading
- **Medium models (1-7GB)**: Good candidates for offloading
- **Large models (7GB+)**: Excellent candidates, significant memory savings

## Contributing Examples

We welcome contributions of new examples! Please follow these guidelines:

1. **Add comprehensive docstrings**
2. **Include error handling**
3. **Add performance measurements**
4. **Test with different model sizes**
5. **Update this README**

### Example Template

```python
"""
Example Name - Brief Description

Detailed description of what this example demonstrates.
"""

import tensorstream
# ... other imports

def main():
    """Main example function."""
    
    print("🚀 Example Name")
    print("=" * 50)
    
    # Example code here
    
    print("✅ Example completed!")

if __name__ == "__main__":
    main()
```

## Additional Resources

- **Documentation**: https://tensorstream.readthedocs.io
- **GitHub Issues**: https://github.com/tensorstream/tensorstream/issues
- **Discussions**: https://github.com/tensorstream/tensorstream/discussions

## License

These examples are part of the TensorStream project and are licensed under the MIT License.
