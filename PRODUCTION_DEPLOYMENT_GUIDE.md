# TensorStream Production Deployment Guide

**Version:** 0.1.0  
**Status:** ✅ Production Ready  
**Date:** June 13, 2025

## 🚀 Quick Start

### Installation

```bash
# Production installation
pip install tensorstream[all]

# Or install from source
git clone https://github.com/your-org/tensorstream.git
cd tensorstream
pip install -e .
```

### Basic Usage

```python
import tensorstream
from transformers import AutoModelForCausalLM

# Load any large model
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# Apply TensorStream offloading
config = tensorstream.Config(storage_path="/fast/ssd/tensorstream")
offloaded_model = tensorstream.offload(model, config)

# Use normally - layers load just-in-time
offloaded_model.to('cuda:0')
output = offloaded_model.generate(input_ids, max_length=100)
```

## 📋 Production Checklist

### ✅ Pre-Deployment Validation

- [ ] **Hardware Requirements Met**
  - [ ] Sufficient VRAM (8GB+ recommended)
  - [ ] Fast storage (NVMe SSD recommended)
  - [ ] Adequate RAM (32GB+ for large models)

- [ ] **Software Dependencies**
  - [ ] Python 3.8-3.13 installed
  - [ ] PyTorch 2.0+ installed
  - [ ] CUDA 11.8+ (if using GPU backends)

- [ ] **Storage Configuration**
  - [ ] High-speed storage path configured
  - [ ] Sufficient free space (1.5-2x model size)
  - [ ] Proper permissions for storage path

- [ ] **Run Validation Suite**
  ```bash
  cd tensorstream
  python production_validation.py
  ```

### ✅ Configuration Optimization

#### Memory Budget Configuration
```python
config = tensorstream.Config(
    storage_path="/nvme/tensorstream",
    vram_budget_gb=12.0,  # Leave 20% headroom
    memory_pressure_mode="balanced"
)
```

#### Backend Selection
```python
# For maximum performance with supported hardware
config.backend = tensorstream.BackendType.GPUDIRECT

# For broad compatibility
config.backend = tensorstream.BackendType.CUDA

# For CPU-only systems
config.backend = tensorstream.BackendType.MMAP
```

#### Prefetching Strategy
```python
# For predictable access patterns
config.prefetch_strategy = tensorstream.PrefetchStrategy.NEXT_LAYER

# For dynamic workloads
config.prefetch_strategy = tensorstream.PrefetchStrategy.ADAPTIVE
```

## 🔧 Production Configuration

### High-Performance Setup

```python
config = tensorstream.Config(
    storage_path="/nvme/tensorstream",
    vram_budget_gb=16.0,
    backend=tensorstream.BackendType.GPUDIRECT,
    prefetch_strategy=tensorstream.PrefetchStrategy.ADAPTIVE,
    compression_enabled=True,
    compression_level=6,
    num_io_threads=8,
    memory_pressure_mode="aggressive",
    verify_checksums=True,
    debug_mode=False
)
```

### Memory-Constrained Setup

```python
config = tensorstream.Config(
    storage_path="/ssd/tensorstream",
    vram_budget_gb=6.0,
    backend=tensorstream.BackendType.CUDA,
    prefetch_strategy=tensorstream.PrefetchStrategy.NONE,
    compression_enabled=True,
    compression_level=9,
    num_io_threads=4,
    memory_pressure_mode="conservative",
    min_layer_size=4 * 1024 * 1024  # 4MB threshold
)
```

## 📊 Monitoring & Observability

### Performance Monitoring

```python
# Get runtime statistics
stats = tensorstream.get_model_statistics(offloaded_model)
print(f"VRAM usage: {stats['vram_usage_bytes'] / 1e9:.2f} GB")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
print(f"I/O throughput: {stats['io_throughput_mbps']:.1f} MB/s")
```

### Health Checks

```python
def health_check(model):
    """Perform health check on offloaded model."""
    try:
        # Test basic inference
        dummy_input = torch.randint(0, 1000, (1, 10))
        with torch.no_grad():
            output = model(dummy_input)
        
        # Check memory statistics
        stats = tensorstream.get_model_statistics(model)
        if stats['vram_utilization'] > 0.9:
            return False, "VRAM utilization too high"
        
        return True, "Healthy"
    except Exception as e:
        return False, f"Health check failed: {e}"
```

### Logging Configuration

```python
import logging

# Configure TensorStream logging
logging.getLogger('tensorstream').setLevel(logging.INFO)

# For debugging
config.debug_mode = True  # Enable detailed logging
```

## 🛡️ Error Handling & Recovery

### Graceful Degradation

```python
def robust_model_loading(model_path: str, config: tensorstream.Config):
    """Load model with graceful degradation."""
    try:
        # Try TensorStream offloading first
        model = AutoModelForCausalLM.from_pretrained(model_path)
        return tensorstream.offload(model, config)
    except Exception as e:
        logging.warning(f"TensorStream offloading failed: {e}")
        
        # Fallback to standard loading
        return AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
```

### Memory Pressure Handling

```python
def handle_memory_pressure(model):
    """Handle memory pressure situations."""
    try:
        # Optimize memory usage
        result = tensorstream.optimize_memory_usage(model, target_utilization=0.7)
        logging.info(f"Memory optimization: {result}")
    except Exception as e:
        logging.error(f"Memory optimization failed: {e}")
        
        # Force cleanup
        if hasattr(model, 'cleanup_tensorstream'):
            model.cleanup_tensorstream()
```

## 🔄 Deployment Patterns

### Container Deployment

```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-devel

# Install TensorStream
RUN pip install tensorstream[all]

# Create storage directory
RUN mkdir -p /app/tensorstream_storage

# Configure environment
ENV TENSORSTREAM_STORAGE_PATH=/app/tensorstream_storage
ENV TENSORSTREAM_VRAM_BUDGET=12.0

COPY app.py /app/
WORKDIR /app

CMD ["python", "app.py"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: tensorstream-inference
spec:
  replicas: 1
  selector:
    matchLabels:
      app: tensorstream-inference
  template:
    metadata:
      labels:
        app: tensorstream-inference
    spec:
      containers:
      - name: inference
        image: your-org/tensorstream-app:latest
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "32Gi"
          limits:
            nvidia.com/gpu: 1
            memory: "64Gi"
        volumeMounts:
        - name: fast-storage
          mountPath: /app/tensorstream_storage
        env:
        - name: TENSORSTREAM_STORAGE_PATH
          value: "/app/tensorstream_storage"
        - name: TENSORSTREAM_VRAM_BUDGET
          value: "12.0"
      volumes:
      - name: fast-storage
        hostPath:
          path: /nvme/tensorstream
```

### Cloud Deployment

```python
# AWS/GCP/Azure optimized configuration
config = tensorstream.Config(
    storage_path="/mnt/ssd/tensorstream",  # Use attached SSD
    vram_budget_gb=14.0,  # For A100/V100 instances
    backend=tensorstream.BackendType.CUDA,
    prefetch_strategy=tensorstream.PrefetchStrategy.ADAPTIVE,
    compression_enabled=True,
    compression_level=6,
    num_io_threads=6,
    memory_pressure_mode="balanced"
)
```

## 🔍 Troubleshooting

### Common Issues

#### Issue: "CUDA out of memory"
```python
# Solution: Reduce VRAM budget
config.vram_budget_gb = config.vram_budget_gb * 0.8
```

#### Issue: Slow inference on first run
```python
# Solution: Enable prefetching
config.prefetch_strategy = tensorstream.PrefetchStrategy.NEXT_LAYER
```

#### Issue: High storage I/O latency
```python
# Solution: Increase compression or use faster storage
config.compression_enabled = True
config.compression_level = 9
```

### Debug Mode

```python
# Enable comprehensive debugging
config.debug_mode = True
config.verify_checksums = True

# Monitor I/O operations
logging.getLogger('tensorstream.io').setLevel(logging.DEBUG)
```

## 🧪 Testing in Production

### Load Testing

```python
def load_test(model, num_requests=100):
    """Perform load testing on offloaded model."""
    import time
    import concurrent.futures
    
    def inference_task():
        input_ids = torch.randint(0, 1000, (1, 20))
        with torch.no_grad():
            return model(input_ids)
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(inference_task) for _ in range(num_requests)]
        results = [f.result() for f in futures]
    
    end_time = time.time()
    
    return {
        "requests": num_requests,
        "total_time": end_time - start_time,
        "requests_per_second": num_requests / (end_time - start_time),
        "success_rate": len(results) / num_requests
    }
```

### A/B Testing

```python
def ab_test_tensorstream(model_path, test_inputs):
    """Compare TensorStream vs standard loading."""
    
    # Test standard PyTorch
    model_standard = AutoModelForCausalLM.from_pretrained(model_path)
    
    start_time = time.time()
    with torch.no_grad():
        outputs_standard = [model_standard(inp) for inp in test_inputs]
    standard_time = time.time() - start_time
    
    # Test TensorStream
    config = tensorstream.create_default_config("/tmp/tensorstream")
    model_ts = tensorstream.offload(
        AutoModelForCausalLM.from_pretrained(model_path), 
        config
    )
    
    start_time = time.time()
    with torch.no_grad():
        outputs_ts = [model_ts(inp) for inp in test_inputs]
    ts_time = time.time() - start_time
    
    return {
        "standard_time": standard_time,
        "tensorstream_time": ts_time,
        "speedup": standard_time / ts_time,
        "outputs_match": all(
            torch.allclose(o1, o2, atol=1e-4) 
            for o1, o2 in zip(outputs_standard, outputs_ts)
        )
    }
```

## 📈 Performance Optimization

### Hardware-Specific Tuning

#### For NVIDIA A100/H100
```python
config = tensorstream.Config(
    vram_budget_gb=32.0,  # Use ~80% of 40GB
    backend=tensorstream.BackendType.GPUDIRECT,
    num_io_threads=16,
    compression_level=4  # Balance speed vs space
)
```

#### For RTX 4090
```python
config = tensorstream.Config(
    vram_budget_gb=18.0,  # Use ~75% of 24GB
    backend=tensorstream.BackendType.CUDA,
    num_io_threads=8,
    compression_level=6
)
```

#### For RTX 3080/3090
```python
config = tensorstream.Config(
    vram_budget_gb=8.0,   # Conservative for 10-12GB cards
    backend=tensorstream.BackendType.CUDA,
    num_io_threads=6,
    compression_level=8
)
```

### Storage Optimization

```python
# For NVMe SSDs
config.num_io_threads = 8
config.compression_level = 4  # Lower compression for speed

# For SATA SSDs
config.num_io_threads = 4
config.compression_level = 8  # Higher compression for bandwidth

# For network storage
config.num_io_threads = 2
config.compression_level = 9  # Maximum compression
```

## 🛠️ Maintenance

### Regular Maintenance Tasks

```python
def maintenance_routine(storage_path):
    """Perform regular maintenance on TensorStream storage."""
    
    # Clean up orphaned files
    tensorstream.cleanup_orphaned_files(storage_path)
    
    # Defragment storage if needed
    tensorstream.defragment_storage(storage_path)
    
    # Update checksums
    tensorstream.verify_storage_integrity(storage_path)
```

### Storage Cleanup

```python
def cleanup_old_models(storage_path, max_age_days=30):
    """Clean up old cached models."""
    import os
    import time
    
    current_time = time.time()
    cutoff_time = current_time - (max_age_days * 24 * 60 * 60)
    
    for root, dirs, files in os.walk(storage_path):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.getmtime(file_path) < cutoff_time:
                os.remove(file_path)
```

## 📞 Support & Documentation

### Getting Help

- **Documentation:** [https://tensorstream.readthedocs.io/](https://tensorstream.readthedocs.io/)
- **GitHub Issues:** [https://github.com/your-org/tensorstream/issues](https://github.com/your-org/tensorstream/issues)
- **Community Discord:** [https://discord.gg/tensorstream](https://discord.gg/tensorstream)

### Reporting Issues

When reporting issues, include:
- TensorStream version
- PyTorch version
- Hardware specifications
- Configuration used
- Full error traceback
- Minimal reproduction case

---

## ✅ Production Deployment Checklist

- [ ] Hardware requirements validated
- [ ] Dependencies installed and verified
- [ ] Storage configured and tested
- [ ] Configuration optimized for workload
- [ ] Validation suite passes 100%
- [ ] Monitoring and logging configured
- [ ] Error handling implemented
- [ ] Performance benchmarks completed
- [ ] Load testing performed
- [ ] Backup and recovery procedures tested
- [ ] Documentation updated
- [ ] Team training completed

**🚀 Ready for Production Deployment!**
