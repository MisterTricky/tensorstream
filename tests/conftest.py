# Test configuration and fixtures
pytest_plugins = ["pytest_mock"]

# Test markers
import pytest

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "backend: Backend-specific tests")

@pytest.fixture
def temp_storage_dir(tmp_path):
    """Provide a temporary directory for storage tests."""
    storage_dir = tmp_path / "tensorstream_storage"
    storage_dir.mkdir()
    return storage_dir

@pytest.fixture
def sample_tensor():
    """Provide a sample tensor for testing."""
    import torch
    return torch.randn(100, 50, dtype=torch.float32)

@pytest.fixture
def large_tensor():
    """Provide a large tensor for testing memory operations."""
    import torch
    return torch.randn(1000, 1000, dtype=torch.float32)

@pytest.fixture
def mock_config(temp_storage_dir):
    """Provide a mock configuration for testing."""
    from tensorstream.config import Config
    return Config(
        storage_path=temp_storage_dir,
        vram_budget_gb=2.0,
        backend="mmap",
        debug_mode=True,
        num_io_threads=2,
    )

@pytest.fixture
def sample_model():
    """Provide a sample model for testing."""
    import torch
    import torch.nn as nn
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(100, 50)
            self.layer2 = nn.Linear(50, 25)
            self.layer3 = nn.Linear(25, 10)
            
        def forward(self, x):
            x = torch.relu(self.layer1(x))
            x = torch.relu(self.layer2(x))
            return self.layer3(x)
    
    return SimpleModel()

@pytest.fixture
def transformer_like_model():
    """Provide a transformer-like model for testing."""
    import torch
    import torch.nn as nn
    
    class AttentionLayer(nn.Module):
        def __init__(self, d_model, n_heads):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            
        def forward(self, x):
            # Simplified attention
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            return self.out_proj(v)
    
    class TransformerBlock(nn.Module):
        def __init__(self, d_model, n_heads, d_ff):
            super().__init__()
            self.attention = AttentionLayer(d_model, n_heads)
            self.norm1 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            )
            self.norm2 = nn.LayerNorm(d_model)
            
        def forward(self, x):
            x = x + self.attention(self.norm1(x))
            x = x + self.ffn(self.norm2(x))
            return x
    
    class TransformerModel(nn.Module):
        def __init__(self, d_model=512, n_heads=8, d_ff=2048, n_layers=6):
            super().__init__()
            self.embedding = nn.Embedding(1000, d_model)
            self.layers = nn.ModuleList([
                TransformerBlock(d_model, n_heads, d_ff) 
                for _ in range(n_layers)
            ])
            self.output = nn.Linear(d_model, 1000)
            
        def forward(self, x):
            x = self.embedding(x)
            for layer in self.layers:
                x = layer(x)
            return self.output(x)
    
    return TransformerModel(d_model=256, n_heads=4, d_ff=1024, n_layers=4)
