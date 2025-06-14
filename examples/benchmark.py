"""
Performance Benchmarking Example

This example demonstrates how to benchmark TensorStream performance
and compare it with standard PyTorch model loading.
"""

import torch
import tensorstream
import time
import psutil
import json
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from tensorstream.config import BackendType, PrefetchStrategy

class PerformanceBenchmark:
    """Performance benchmarking utility for TensorStream."""
    
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device
        self.results = {}
        
    def load_model_and_tokenizer(self) -> Tuple[torch.nn.Module, object]:
        """Load model and tokenizer."""
        print(f"Loading {self.model_name}...")
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory_info = {
            "system_ram_gb": psutil.virtual_memory().used / (1024**3),
            "system_ram_percent": psutil.virtual_memory().percent,
        }
        
        if torch.cuda.is_available():
            memory_info.update({
                "gpu_allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "gpu_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                "gpu_free_gb": (torch.cuda.get_device_properties(0).total_memory - 
                               torch.cuda.memory_allocated()) / (1024**3)
            })
        
        return memory_info
    
    def benchmark_loading_time(self, model, config=None) -> Dict[str, float]:
        """Benchmark model loading time."""
        print("Benchmarking loading time...")
        
        if config is None:
            # Standard PyTorch loading
            start_time = time.time()
            model.to(self.device)
            loading_time = time.time() - start_time
            method = "standard"
        else:
            # TensorStream loading
            start_time = time.time()
            offloaded_model = tensorstream.offload(model, config)
            offloaded_model.to(self.device)
            loading_time = time.time() - start_time
            method = "tensorstream"
            model = offloaded_model
        
        memory_after = self.get_memory_usage()
        
        return {
            "method": method,
            "loading_time_s": loading_time,
            "memory_after": memory_after
        }
    
    def benchmark_inference_speed(self, model, tokenizer, num_iterations: int = 10) -> Dict[str, float]:
        """Benchmark inference speed."""
        print(f"Benchmarking inference speed ({num_iterations} iterations)...")
        
        # Prepare input
        prompt = "The future of artificial intelligence and machine learning is"
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Warm-up run
        with torch.no_grad():
            model.generate(inputs.input_ids, max_length=30, do_sample=False)
        
        # Benchmark runs
        times = []
        token_counts = []
        
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
            token_counts.append(tokens_generated)
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        avg_tokens = sum(token_counts) / len(token_counts)
        tokens_per_second = avg_tokens / avg_time
        
        return {
            "avg_inference_time_s": avg_time,
            "min_inference_time_s": min(times),
            "max_inference_time_s": max(times),
            "std_inference_time_s": (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5,
            "avg_tokens_generated": avg_tokens,
            "tokens_per_second": tokens_per_second,
            "iterations": num_iterations
        }
    
    def benchmark_memory_efficiency(self, model, tokenizer) -> Dict[str, float]:
        """Benchmark memory efficiency during inference."""
        print("Benchmarking memory efficiency...")
        
        memory_before = self.get_memory_usage()
        
        # Run inference with memory monitoring
        prompt = "Test prompt for memory efficiency measurement"
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Peak memory measurement
        peak_memory = memory_before.copy()
        
        for _ in range(5):  # Multiple runs to find peak
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=100,
                    do_sample=True,
                    temperature=0.7
                )
            
            current_memory = self.get_memory_usage()
            for key in peak_memory:
                if key in current_memory:
                    peak_memory[key] = max(peak_memory[key], current_memory[key])
        
        memory_after = self.get_memory_usage()
        
        # Calculate memory efficiency metrics
        memory_increase = {}
        for key in memory_before:
            if key in memory_after:
                memory_increase[f"memory_increase_{key}"] = memory_after[key] - memory_before[key]
                memory_increase[f"peak_memory_{key}"] = peak_memory[key]
        
        return memory_increase
    
    def run_full_benchmark(self, configs: List[Dict]) -> Dict:
        """Run complete benchmark suite."""
        print(f"🏃 Running full benchmark for {self.model_name}")
        print("=" * 60)
        
        # Load model and tokenizer
        model, tokenizer = self.load_model_and_tokenizer()
        
        # Get model statistics
        total_params = sum(p.numel() for p in model.parameters())
        model_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
        
        results = {
            "model_name": self.model_name,
            "device": self.device,
            "total_parameters": total_params,
            "model_size_gb": model_size_gb,
            "benchmarks": {}
        }
        
        # Benchmark standard PyTorch
        print("\n📊 Benchmarking standard PyTorch...")
        try:
            standard_loading = self.benchmark_loading_time(model)
            standard_inference = self.benchmark_inference_speed(model, tokenizer)
            standard_memory = self.benchmark_memory_efficiency(model, tokenizer)
            
            results["benchmarks"]["standard"] = {
                "loading": standard_loading,
                "inference": standard_inference,
                "memory": standard_memory
            }
            
        except Exception as e:
            print(f"❌ Standard benchmark failed: {e}")
            results["benchmarks"]["standard"] = {"error": str(e)}
        
        # Move model back to CPU for TensorStream benchmarks
        model = model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Benchmark TensorStream with different configurations
        for i, config_dict in enumerate(configs):
            config_name = config_dict.get("name", f"config_{i}")
            print(f"\n📊 Benchmarking TensorStream ({config_name})...")
            
            try:
                # Create TensorStream config
                config = tensorstream.Config(**{k: v for k, v in config_dict.items() if k != "name"})
                
                # Fresh model copy for each benchmark
                model_copy = AutoModelForCausalLM.from_pretrained(self.model_name)
                
                ts_loading = self.benchmark_loading_time(model_copy, config)
                ts_inference = self.benchmark_inference_speed(model_copy, tokenizer)
                ts_memory = self.benchmark_memory_efficiency(model_copy, tokenizer)
                
                # Get TensorStream statistics
                if hasattr(model_copy, '_tensorstream_stats'):
                    ts_stats = tensorstream.get_model_statistics(model_copy)
                else:
                    ts_stats = {}
                
                results["benchmarks"][config_name] = {
                    "config": config_dict,
                    "loading": ts_loading,
                    "inference": ts_inference,
                    "memory": ts_memory,
                    "tensorstream_stats": ts_stats
                }
                
                # Cleanup
                if hasattr(model_copy, 'cleanup_tensorstream'):
                    model_copy.cleanup_tensorstream()
                
            except Exception as e:
                print(f"❌ TensorStream benchmark ({config_name}) failed: {e}")
                results["benchmarks"][config_name] = {"error": str(e)}
        
        return results
    
    def save_results(self, results: Dict, output_path: str):
        """Save benchmark results to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📁 Results saved to: {output_file}")
    
    def print_summary(self, results: Dict):
        """Print benchmark summary."""
        print("\n📈 BENCHMARK SUMMARY")
        print("=" * 60)
        
        print(f"Model: {results['model_name']}")
        print(f"Parameters: {results['total_parameters']:,}")
        print(f"Model size: {results['model_size_gb']:.2f} GB")
        print(f"Device: {results['device']}")
        
        print("\nPerformance Comparison:")
        print("-" * 40)
        
        for benchmark_name, benchmark_data in results["benchmarks"].items():
            if "error" in benchmark_data:
                print(f"{benchmark_name:15s}: ❌ {benchmark_data['error']}")
                continue
            
            loading_time = benchmark_data["loading"]["loading_time_s"]
            inference_speed = benchmark_data["inference"]["tokens_per_second"]
            
            print(f"{benchmark_name:15s}: Load={loading_time:.2f}s, Speed={inference_speed:.1f} tok/s")

def main():
    """Run performance benchmarking example."""
    
    print("⚡ TensorStream Performance Benchmarking")
    print("=" * 50)
    
    # Configuration
    model_name = "gpt2"  # Start with smaller model for demo
    
    # Define benchmark configurations
    configs = [
        {
            "name": "mmap_basic",
            "storage_path": "/tmp/tensorstream_benchmark",
            "vram_budget_gb": 2.0,
            "backend": BackendType.MMAP,
            "prefetch_strategy": PrefetchStrategy.NONE
        },
        {
            "name": "mmap_prefetch",
            "storage_path": "/tmp/tensorstream_benchmark",
            "vram_budget_gb": 2.0,
            "backend": BackendType.MMAP,
            "prefetch_strategy": PrefetchStrategy.NEXT_LAYER
        },
        {
            "name": "cuda_adaptive",
            "storage_path": "/tmp/tensorstream_benchmark",
            "vram_budget_gb": 4.0,
            "backend": BackendType.CUDA,
            "prefetch_strategy": PrefetchStrategy.ADAPTIVE,
            "compression_enabled": True
        }
    ]
    
    # Run benchmark
    benchmark = PerformanceBenchmark(model_name)
    results = benchmark.run_full_benchmark(configs)
    
    # Print summary
    benchmark.print_summary(results)
    
    # Save detailed results
    benchmark.save_results(results, f"benchmark_results_{model_name.replace('/', '_')}.json")
    
    print("\n✅ Benchmarking completed!")

if __name__ == "__main__":
    main()
