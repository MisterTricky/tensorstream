"""
Basic TensorStream Usage Example

This example demonstrates the simplest way to use TensorStream
with a Hugging Face transformer model.
"""

import torch
import tensorstream
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    """Basic example of using TensorStream with a transformer model."""
    
    print("🚀 TensorStream Basic Example")
    print("=" * 50)
    
    # 1. Choose a model (using a smaller model for demo)
    model_name = "gpt2"  # Change to "mistralai/Mistral-7B-v0.1" for larger model
    print(f"Loading model: {model_name}")
    
    # 2. Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Create TensorStream configuration
    config = tensorstream.Config(
        storage_path="/tmp/tensorstream",  # Where to store offloaded layers
        vram_budget_gb=4.0,                # Available GPU memory (adjust for your system)
        backend="auto"                     # Auto-select best backend
    )
    
    print(f"Configuration: {config}")
    
    # 4. Apply TensorStream offloading (the magic happens here!)
    print("\n📦 Applying TensorStream offloading...")
    offloaded_model = tensorstream.offload(model, config)
    
    # 5. Move model to GPU (if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    offloaded_model.to(device)
    
    # 6. Test the model with some input
    print("\n🔤 Generating text...")
    prompt = "The future of artificial intelligence is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate text
    with torch.no_grad():
        outputs = offloaded_model.generate(
            inputs.input_ids,
            max_length=50,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and print results
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")
    
    # 7. Check statistics
    print("\n📊 TensorStream Statistics:")
    stats = tensorstream.get_model_statistics(offloaded_model)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # 8. Cleanup
    print("\n🧹 Cleaning up...")
    offloaded_model.cleanup_tensorstream()
    
    print("✅ Example completed successfully!")

if __name__ == "__main__":
    main()
