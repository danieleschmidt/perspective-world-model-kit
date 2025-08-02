# Installation Guide

This guide covers all installation methods for the Perspective World Model Kit (PWMK).

## Prerequisites

### System Requirements
- **Python**: 3.9 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: Minimum 8GB RAM (16GB+ recommended for training)
- **Storage**: 2GB free space (more for datasets and models)

### Optional Requirements
- **GPU**: CUDA-compatible GPU for accelerated training
- **Docker**: For containerized development
- **Unity**: For 3D environment development

## Installation Methods

### Method 1: PyPI Installation (Recommended)

```bash
# Install the latest stable version
pip install perspective-world-model-kit

# Install with optional dependencies
pip install perspective-world-model-kit[dev,unity,prolog]

# Verify installation
pwmk --version
```

### Method 2: Development Installation

```bash
# Clone the repository
git clone https://github.com/your-org/perspective-world-model-kit.git
cd perspective-world-model-kit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify installation
pytest tests/
```

### Method 3: Docker Installation

```bash
# Pull the latest image
docker pull your-org/pwmk:latest

# Run with Jupyter Lab
docker run -it -p 8888:8888 your-org/pwmk:latest jupyter lab

# Run interactive shell
docker run -it your-org/pwmk:latest bash

# Build from source
git clone https://github.com/your-org/perspective-world-model-kit.git
cd perspective-world-model-kit
docker build -t pwmk:local .
```

### Method 4: Conda Installation

```bash
# Create conda environment
conda create -n pwmk python=3.9
conda activate pwmk

# Install from conda-forge (when available)
conda install -c conda-forge perspective-world-model-kit

# Or install via pip in conda environment
pip install perspective-world-model-kit
```

## Optional Dependencies

### Unity ML-Agents Integration

For 3D environments and Unity integration:

```bash
pip install perspective-world-model-kit[unity]

# Download pre-built Unity environments
pwmk download-unity-envs

# Or build Unity environments from source
cd unity/
./build.sh
```

### Prolog Backend Support

For advanced symbolic reasoning:

```bash
pip install perspective-world-model-kit[prolog]

# Install SWI-Prolog system dependency
# Ubuntu/Debian:
sudo apt-get install swi-prolog

# macOS:
brew install swi-prolog

# Windows: Download from https://www.swi-prolog.org/
```

### Development Tools

For contributing to PWMK:

```bash
pip install perspective-world-model-kit[dev]

# Additional tools for documentation
pip install perspective-world-model-kit[docs]

# Testing dependencies
pip install perspective-world-model-kit[test]
```

## GPU Support

### CUDA Installation

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch (if not already installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU support
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"
```

### Apple Silicon (M1/M2) Support

```bash
# Install MPS-optimized PyTorch
pip install torch torchvision torchaudio

# Verify MPS support
python -c "
import torch
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
"
```

## Verification

### Quick Verification

```bash
# Check installation
pwmk --version
pwmk --help

# Run basic tests
python -c "
import pwmk
print(f'PWMK version: {pwmk.__version__}')
print('Installation successful!')
"
```

### Comprehensive Testing

```bash
# Run full test suite
pytest tests/ -v

# Run specific test categories
pytest tests/unit/ -m "not slow"
pytest tests/integration/ -m "integration"

# Run with coverage
pytest tests/ --cov=pwmk --cov-report=html
```

### Example Script

Create and run this test script:

```python
# test_installation.py
from pwmk import PerspectiveWorldModel, BeliefStore
from pwmk.envs import MultiAgentBoxWorld
import torch

def test_basic_functionality():
    """Test basic PWMK functionality."""
    
    # Test environment creation
    env = MultiAgentBoxWorld(
        num_agents=2,
        partial_observability=True
    )
    print("✓ Environment creation successful")
    
    # Test world model initialization
    world_model = PerspectiveWorldModel(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=64,
        num_agents=2
    )
    print("✓ World model initialization successful")
    
    # Test belief store
    belief_store = BeliefStore()
    belief_store.add_belief("agent_0", "test_belief")
    print("✓ Belief store functionality successful")
    
    # Test GPU availability (if available)
    if torch.cuda.is_available():
        world_model = world_model.cuda()
        print("✓ GPU support verified")
    
    print("\n🎉 All tests passed! PWMK is ready to use.")

if __name__ == "__main__":
    test_basic_functionality()
```

Run the test:
```bash
python test_installation.py
```

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# If you encounter import errors, check dependencies
pip install --upgrade pip
pip install perspective-world-model-kit --force-reinstall
```

#### CUDA Issues
```bash
# Check PyTorch CUDA installation
python -c "import torch; print(torch.__version__, torch.version.cuda)"

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Unity Environment Issues
```bash
# Download Unity environments manually
mkdir -p ~/.pwmk/envs
cd ~/.pwmk/envs
wget https://github.com/your-org/pwmk-unity-envs/releases/latest/download/unity-envs.zip
unzip unity-envs.zip
```

#### Permission Errors (Windows)
```cmd
# Run as administrator or use --user flag
pip install --user perspective-world-model-kit
```

### Getting Help

If you encounter issues:

1. **Check the logs**: Run with `--verbose` flag for detailed output
2. **Search existing issues**: [GitHub Issues](https://github.com/your-org/perspective-world-model-kit/issues)
3. **Join our community**: [Discord](https://discord.gg/your-org)
4. **Email support**: [pwmk@your-org.com](mailto:pwmk@your-org.com)

### Environment Information

Collect environment info for bug reports:

```bash
# Generate environment report
pwmk diagnose --output env_report.txt

# Or manually collect info
python -c "
import sys
import torch
import pwmk
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'PWMK: {pwmk.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

## Next Steps

After successful installation:

1. 🚀 Follow the [Quick Start Tutorial](quickstart.md)
2. 📚 Read the [Basic Concepts Guide](concepts.md)
3. 🔬 Explore [Example Notebooks](../examples/)
4. 🏗️ Check the [Architecture Overview](../ARCHITECTURE.md)

## Update Instructions

### Regular Updates
```bash
# Update to latest version
pip install --upgrade perspective-world-model-kit

# Update development installation
cd perspective-world-model-kit
git pull origin main
pip install -e ".[dev]" --upgrade
```

### Version Pinning

For reproducible research, pin to specific versions:

```bash
# Install specific version
pip install perspective-world-model-kit==0.1.0

# Or use requirements.txt
echo "perspective-world-model-kit==0.1.0" > requirements.txt
pip install -r requirements.txt
```