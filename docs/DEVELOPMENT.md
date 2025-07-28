# Development Guide

## Prerequisites

- Python 3.9+
- Git
- Optional: Unity 2022.3+ (for environment development)

## Setup

```bash
# Clone repository
git clone https://github.com/your-org/perspective-world-model-kit
cd perspective-world-model-kit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev,unity]"
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=pwmk tests/

# Run specific test
pytest tests/test_belief_reasoning.py -v
```

## Code Quality

- Format: `black .`
- Lint: `flake8`
- Type check: `mypy pwmk/`

See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.