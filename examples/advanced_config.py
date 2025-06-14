"""
Advanced TensorStream Configuration Example

This example demonstrates advanced configuration options and
performance tuning for TensorStream.
"""

import torch
import tensorstream
from transformers import AutoModelForCausalLM, AutoTokenizer
from tensorstream.config import BackendType, PrefetchStrategy, MemoryPressureMode
import time
import psutil

def print_system_info():
    """Print system information."""
    print("🖥️  System Information:")
    print(f"   CPU cores: {psutil.cpu_count()}")
    print(f"   RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        print(f"   CUDA version: {torch.version.cuda}")
    else:
        print("   GPU: Not available")

def create_advanced_config():
    """Create advanced TensorStream configuration."""
    
    config = tensorstream.Config(
        # Storage configuration
        storage_path="/fast/nvme/tensorstream",  # Use fastest storage available
        
        # Memory management
        vram_budget_gb=6.0,                      # Conservative GPU memory budget
        memory_pressure_mode=MemoryPressureMode.BALANCED,  # Balance performance vs memory
        
        # Backend selection
        backend=BackendType.GPUDIRECT,           # Use GPUDirect for best performance
        
        # Prefetching strategy
        prefetch_strategy=PrefetchStrategy.ADAPTIVE,  # Smart prefetching
        
        # Compression settings
        compression_enabled=True,                # Enable compression to save space
        compression_level=6,                     # Balanced compression (1-9)
        
        # I/O configuration
        num_io_threads=4,                       # Parallel I/O threads
        
        # Debug and monitoring
        debug_mode=True,                        # Enable detailed logging
        
        # Performance tuning
        cache_size_gb=2.0,                      # Larger cache for better hit rates
        prefetch_buffer_size=3,                 # Number of layers to prefetch
    )
    
    return config

def benchmark_performance(model, tokenizer, device, num_iterations=5):
    """Benchmark model performance."""
    
    print(f"\n⏱️  Running performance benchmark ({num_iterations} iterations)...")
    
    prompt = "The future of machine learning and artificial intelligence"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    times = []
    
    # Warm up
    with torch.no_grad():
        model.generate(inputs.input_ids, max_length=30, do_sample=False)
    
    # Benchmark
    for i in range(num_iterations):
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        end_time = time.time()
        iteration_time = end_time - start_time
        times.append(iteration_time)
        
        tokens_generated = outputs.shape[1] - inputs.input_ids.shape[1]
        tokens_per_second = tokens_generated / iteration_time
        
        print(f"   Iteration {i+1}: {iteration_time:.2f}s ({tokens_per_second:.1f} tok/s)")
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\n📈 Performance Summary:")
    print(f"   Average time: {avg_time:.2f}s")
    print(f"   Min time: {min_time:.2f}s")
    print(f"   Max time: {max_time:.2f}s")
    print(f"   Std deviation: {(sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5:.2f}s")

def monitor_memory_usage(model):
    """Monitor memory usage during inference."""
    
    print("\n🧠 Memory Usage Monitoring:")
    
    # System memory
    system_memory = psutil.virtual_memory()
    print(f"   System RAM: {system_memory.used / (1024**3):.1f} / {system_memory.total / (1024**3):.1f} GB")
    
    # GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024**3)
        gpu_memory_cached = torch.cuda.memory_reserved() / (1024**3)
        print(f"   GPU memory allocated: {gpu_memory:.1f} GB")
        print(f"   GPU memory cached: {gpu_memory_cached:.1f} GB")
    
    # TensorStream statistics
    stats = tensorstream.get_model_statistics(model)
    print(f"   TensorStream cache hit rate: {stats.get('cache_hit_rate', 0):.1%}")
    print(f"   Layers offloaded: {stats.get('offloaded_layers', 0)}")
    print(f"   VRAM usage: {stats.get('vram_usage_gb', 0):.1f} GB")

def main():
    """Advanced TensorStream configuration example."""
    
    print("🔧 TensorStream Advanced Configuration Example")
    print("=" * 60)
    
    # Print system information
    print_system_info()
    
    # Model configuration
    model_name = "gpt2-medium"  # Larger model for better demonstration
    print(f"\n📦 Loading model: {model_name}")
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create advanced configuration
    print("\n⚙️  Creating advanced configuration...")
    config = create_advanced_config()
    
    # Print configuration details
    print("Configuration details:")
    for key, value in config.__dict__.items():
        if not key.startswith('_'):
            print(f"   {key}: {value}")
    
    # Apply TensorStream with advanced config
    print("\n🚀 Applying TensorStream with advanced configuration...")
    try:
        offloaded_model = tensorstream.offload(model, config)
        print("✅ TensorStream applied successfully!")
    except Exception as e:
        print(f"❌ Failed to apply TensorStream: {e}")
        print("Falling back to memory-mapped backend...")
        
        # Fallback configuration
        fallback_config = tensorstream.Config(
            storage_path="/tmp/tensorstream",
            vram_budget_gb=4.0,
            backend=BackendType.MMAP,
            debug_mode=True
        )
        offloaded_model = tensorstream.offload(model, fallback_config)
        print("✅ Fallback configuration applied!")
    
    # Move to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n📱 Moving model to device: {device}")
    offloaded_model.to(device)
    
    # Monitor initial memory usage
    monitor_memory_usage(offloaded_model)
    
    # Benchmark performance
    benchmark_performance(offloaded_model, tokenizer, device)
    
    # Monitor memory usage after benchmark
    monitor_memory_usage(offloaded_model)
    
    # Test different generation parameters
    print("\n🎯 Testing different generation parameters...")
    prompt = "Artificial intelligence will transform"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    generation_configs = [
        {"max_length": 30, "temperature": 0.7, "do_sample": True, "top_p": 0.9},
        {"max_length": 50, "temperature": 1.0, "do_sample": True, "top_k": 50},
        {"max_length": 40, "do_sample": False},  # Greedy decoding
    ]
    
    for i, gen_config in enumerate(generation_configs):
        print(f"\n   Configuration {i+1}: {gen_config}")
        start_time = time.time()
        
        with torch.no_grad():
            outputs = offloaded_model.generate(
                inputs.input_ids,
                pad_token_id=tokenizer.eos_token_id,
                **gen_config
            )
        
        generation_time = time.time() - start_time
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"   Time: {generation_time:.2f}s")
        print(f"   Output: {generated_text}")
    
    # Final statistics
    print("\n📊 Final TensorStream Statistics:")
    final_stats = tensorstream.get_model_statistics(offloaded_model)
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    offloaded_model.cleanup_tensorstream()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("✅ Advanced example completed successfully!")

if __name__ == "__main__":
    main()
