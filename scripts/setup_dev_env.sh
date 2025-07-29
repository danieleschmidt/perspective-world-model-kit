#!/bin/bash
# Development environment setup script for PWMK
# This script sets up a complete development environment

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Main setup function
main() {
    log_info "Starting PWMK development environment setup..."
    
    # Check Python version
    if ! command_exists python3; then
        log_error "Python 3 is not installed. Please install Python 3.9+ first."
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log_info "Python version: $PYTHON_VERSION"
    
    # Validate Python version
    if [[ $(echo "$PYTHON_VERSION >= 3.9" | bc -l) -eq 0 ]]; then
        log_error "Python 3.9+ is required. Current version: $PYTHON_VERSION"
        exit 1
    fi
    
    # Create virtual environment
    if [ ! -d ".venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv .venv
        log_success "Virtual environment created"
    else
        log_warning "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    log_info "Activating virtual environment..."
    source .venv/bin/activate
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    # Install development dependencies
    log_info "Installing development dependencies..."
    pip install -e ".[dev,test,docs]"
    
    # Install additional development tools
    if [ -f "requirements/dev.in" ]; then
        log_info "Installing additional dev requirements..."
        pip install -r requirements/dev.in
    fi
    
    # Setup pre-commit hooks
    log_info "Setting up pre-commit hooks..."
    pre-commit install
    pre-commit install --hook-type commit-msg
    
    # Setup Git hooks
    log_info "Configuring Git hooks..."
    git config --local core.hooksPath .githooks
    
    # Create necessary directories
    log_info "Creating project directories..."
    mkdir -p logs
    mkdir -p data
    mkdir -p experiments
    mkdir -p notebooks
    mkdir -p scripts
    mkdir -p .vscode
    
    # Setup VS Code configuration (if VS Code is available)
    if command_exists code; then
        log_info "Setting up VS Code configuration..."
        setup_vscode_config
    fi
    
    # Setup Jupyter
    log_info "Setting up Jupyter configuration..."
    jupyter lab --generate-config || true
    
    # Install Unity ML-Agents (optional)
    read -p "Install Unity ML-Agents dependencies? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Installing Unity ML-Agents..."
        pip install -e ".[unity]"
    fi
    
    # Install Prolog dependencies (optional)
    read -p "Install Prolog/ASP dependencies? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Installing Prolog dependencies..."
        pip install -e ".[prolog]"
    fi
    
    # Setup environment variables
    setup_env_file
    
    # Run initial tests
    log_info "Running initial test suite..."
    if pytest tests/unit/ -x --tb=short; then
        log_success "Initial tests passed"
    else
        log_warning "Some tests failed - this is normal for initial setup"
    fi
    
    # Setup monitoring (optional)
    read -p "Setup development monitoring stack? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        setup_monitoring
    fi
    
    # Final setup validation
    log_info "Validating setup..."
    validate_setup
    
    log_success "Development environment setup complete!"
    echo
    echo "Next steps:"
    echo "1. Activate the environment: source .venv/bin/activate"
    echo "2. Run tests: make test"
    echo "3. Start coding: code ."
    echo "4. View documentation: make docs-serve"
    echo "5. Start Jupyter: make jupyter"
}

# Setup VS Code configuration
setup_vscode_config() {
    cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests"],
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        ".mypy_cache": true,
        ".coverage": true,
        "htmlcov": true
    }
}
EOF

    cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        },
        {
            "name": "Python: Test File",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["${file}", "-v"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        }
    ]
}
EOF

    log_success "VS Code configuration created"
}

# Setup environment file
setup_env_file() {
    if [ ! -f ".env" ]; then
        log_info "Creating .env file..."
        cat > .env << 'EOF'
# PWMK Development Environment Variables

# Development settings
PWMK_ENV=development
PWMK_DEBUG=true
PWMK_LOG_LEVEL=INFO

# Testing settings
PYTEST_CURRENT_TEST=""
TEST_ENV=local

# Research settings
WANDB_MODE=offline
WANDB_PROJECT=pwmk-dev

# GPU settings (uncomment if using GPU)
# CUDA_VISIBLE_DEVICES=0

# Unity settings (uncomment if using Unity)
# UNITY_LICENSE=""
# UNITY_PROJECT_PATH=""

# Database settings (for integration tests)
REDIS_URL=redis://localhost:6379/0
POSTGRES_URL=postgresql://localhost:5432/pwmk_test

# Monitoring settings
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
EOF
        log_success ".env file created"
    else
        log_warning ".env file already exists"
    fi
}

# Setup monitoring stack
setup_monitoring() {
    if command_exists docker-compose; then
        log_info "Starting monitoring stack..."
        docker-compose -f monitoring/docker-compose.yml up -d
        log_success "Monitoring stack started"
        echo "  - Grafana: http://localhost:3000"
        echo "  - Prometheus: http://localhost:9090"
    else
        log_warning "Docker Compose not found. Skipping monitoring setup."
    fi
}

# Validate setup
validate_setup() {
    local errors=0
    
    # Check Python packages
    if ! python -c "import pwmk" 2>/dev/null; then
        log_error "PWMK package not importable"
        ((errors++))
    fi
    
    # Check development tools
    local tools=("black" "isort" "flake8" "mypy" "pytest" "pre-commit")
    for tool in "${tools[@]}"; do
        if ! command_exists "$tool"; then
            log_error "$tool not found in PATH"
            ((errors++))
        fi
    done
    
    # Check pre-commit hooks
    if ! pre-commit run --all-files --dry-run >/dev/null 2>&1; then
        log_warning "Pre-commit hooks not properly configured"
    fi
    
    if [ $errors -eq 0 ]; then
        log_success "All validation checks passed"
    else
        log_error "$errors validation errors found"
        return 1
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
}

# Set trap for cleanup
trap cleanup EXIT

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi