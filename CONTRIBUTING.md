# Contributing to TensorStream

Thank you for your interest in contributing to TensorStream! We welcome contributions of all kinds, from bug reports and feature requests to code contributions and documentation improvements.

## 🚀 Getting Started

### Prerequisites

- Python 3.8-3.11
- Git
- C++ compiler (GCC/Clang with C++14 support)
- CUDA Toolkit 11.0+ (optional, for GPU features)

### Development Setup

1. **Fork and Clone**
   ```bash
   # Fork the repository on GitHub, then:
   git clone https://github.com/YOUR_USERNAME/tensorstream.git
   cd tensorstream
   ```

2. **Set up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   
   # Install development dependencies
   pip install -r requirements-dev.txt
   
   # Install in development mode
   pip install -e .
   ```

3. **Verify Installation**
   ```bash
   # Run tests to ensure everything works
   make test
   
   # Check code formatting
   make format-check
   
   # Run linting
   make lint
   ```

## 🔧 Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b bugfix/issue-number
```

### 2. Make Changes

- Follow the coding standards (see below)
- Add tests for new functionality
- Update documentation as needed
- Ensure all tests pass

### 3. Test Your Changes

```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration

# Check code coverage
pytest --cov=tensorstream --cov-report=html
```

### 4. Format and Lint

```bash
# Format code
make format

# Check formatting
make format-check

# Run linting
make lint
```

### 5. Commit and Push

```bash
git add .
git commit -m "feat: add your feature description"
git push origin feature/your-feature-name
```

### 6. Create Pull Request

- Go to GitHub and create a pull request
- Fill out the pull request template
- Wait for review and address feedback

## 📋 Coding Standards

### Python Code Style

We follow PEP 8 with some modifications:

- **Line length**: 88 characters (Black default)
- **Quotes**: Double quotes for strings
- **Import organization**: isort with Black profile
- **Type hints**: Required for all public APIs

### Code Formatting

We use automated formatting tools:

```bash
# Format Python code
black tensorstream/ tests/

# Sort imports
isort tensorstream/ tests/

# Or use the Makefile
make format
```

### Linting

We use multiple linting tools:

```bash
# Check Python style
flake8 tensorstream/ tests/

# Type checking
mypy tensorstream/

# Or use the Makefile
make lint
```

### C++ Code Style

For C++/CUDA code:

- **Standard**: C++14
- **Naming**: snake_case for variables, PascalCase for classes
- **Indentation**: 4 spaces
- **Headers**: Include guards for .h files

## 🧪 Testing Guidelines

### Test Categories

1. **Unit Tests** (`tests/unit/`)
   - Test individual functions and classes
   - Fast execution (< 1 second per test)
   - No external dependencies

2. **Integration Tests** (`tests/integration/`)
   - Test component interactions
   - End-to-end workflows
   - May use external models/data

3. **Performance Tests**
   - Benchmark critical paths
   - Memory usage validation
   - Marked with `@pytest.mark.slow`

### Writing Tests

```python
import pytest
import torch
from tensorstream import Config, offload

class TestFeature:
    """Test suite for new feature."""
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample model for testing."""
        # ... setup code
        
    def test_basic_functionality(self, sample_model):
        """Test basic functionality works."""
        # Arrange
        config = Config(storage_path="/tmp/test")
        
        # Act
        result = offload(sample_model, config)
        
        # Assert
        assert result is not None
        
    @pytest.mark.gpu
    def test_gpu_functionality(self):
        """Test GPU-specific functionality."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        # ... test code
```

### Test Fixtures

Use the shared fixtures in `tests/conftest.py`:

- `temp_storage_dir`: Temporary directory for testing
- `sample_tensor`: Basic tensor for testing
- `mock_config`: Pre-configured test configuration
- `sample_model`: Simple PyTorch model

### Running Tests

```bash
# All tests
pytest

# Specific category
pytest tests/unit/
pytest tests/integration/

# With markers
pytest -m "not slow"  # Skip slow tests
pytest -m "gpu"       # GPU tests only

# With coverage
pytest --cov=tensorstream
```

## 📚 Documentation

### Docstring Style

We use Google-style docstrings:

```python
def offload(model: nn.Module, config: Config) -> nn.Module:
    """Apply TensorStream offloading to a PyTorch model.
    
    This function analyzes the model, shards its layers to disk, and replaces
    them with proxy layers that load weights just-in-time during inference.
    
    Args:
        model: The PyTorch model to offload.
        config: TensorStream configuration.
        
    Returns:
        The modified model with proxy layers.
        
    Raises:
        ConfigurationError: If the configuration is invalid.
        TensorStreamError: If offloading fails.
        
    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> config = Config(storage_path="/tmp/tensorstream")
        >>> offloaded_model = offload(model, config)
    """
```

### Building Documentation

```bash
# Install documentation dependencies
pip install -r requirements-docs.txt

# Build documentation
make docs

# Serve locally
make serve-docs

# Live reload during development
make docs-live
```

## 🐛 Bug Reports

### Before Reporting

1. Check existing issues
2. Update to the latest version
3. Test with minimal reproduction case

### Bug Report Template

```markdown
## Bug Description
Brief description of the issue.

## Environment
- TensorStream version: X.X.X
- Python version: X.X.X
- PyTorch version: X.X.X
- CUDA version: X.X (if applicable)
- Operating system: 

## Reproduction Steps
1. Step 1
2. Step 2
3. ...

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Minimal Code Example
```python
# Minimal code that reproduces the issue
```

## Additional Context
Any additional information, logs, or screenshots.
```

## 💡 Feature Requests

### Before Requesting

1. Check existing feature requests
2. Consider if it fits TensorStream's scope
3. Think about backward compatibility

### Feature Request Template

```markdown
## Feature Description
Clear description of the proposed feature.

## Use Case
Why is this feature needed? What problem does it solve?

## Proposed Implementation
Any ideas on how this could be implemented.

## Alternatives Considered
Other approaches you considered.

## Additional Context
Any other relevant information.
```

## 🔍 Code Review Process

### For Contributors

1. **Keep PRs focused**: One feature/fix per PR
2. **Write good commit messages**: Use conventional commits
3. **Add tests**: All new code should have tests
4. **Update documentation**: Keep docs in sync with code
5. **Respond to feedback**: Address review comments promptly

### For Reviewers

1. **Be constructive**: Provide helpful feedback
2. **Check functionality**: Verify the code works
3. **Review tests**: Ensure adequate test coverage
4. **Consider performance**: Watch for performance implications
5. **Approve when ready**: Don't block unnecessarily

### Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests are included and pass
- [ ] Documentation is updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance impact considered
- [ ] Security implications reviewed

## 📦 Release Process

### Versioning

We use [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Steps

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release PR
4. Tag release after merge
5. Automated CI builds and publishes

## 🏆 Recognition

Contributors are recognized in:

- `CONTRIBUTORS.md` file
- Release notes
- Project documentation
- Special contributor badges

## 📞 Getting Help

- **GitHub Discussions**: For questions and ideas
- **GitHub Issues**: For bugs and feature requests
- **Email**: maintainers@tensorstream.ai

## 📋 Commit Message Guidelines

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Build/tool changes

### Examples

```
feat(api): add support for custom backends
fix(orchestrator): resolve memory leak in prefetching
docs(readme): update installation instructions
test(integration): add tests for GPU backend
```

## 🎯 Areas We Need Help

- **Performance optimization**: Profile and optimize critical paths
- **Documentation**: Improve examples and tutorials
- **Testing**: Add more comprehensive tests
- **Platform support**: Windows and macOS compatibility
- **Backends**: Additional I/O backend implementations
- **Integration**: Support for more ML frameworks

Thank you for contributing to TensorStream! 🚀
