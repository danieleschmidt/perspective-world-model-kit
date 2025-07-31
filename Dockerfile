# Multi-stage Dockerfile for Perspective World Model Kit (PWMK)
# Optimized for both development and production environments

# Base image with Python 3.11 and CUDA support for GPU acceleration
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    pkg-config \
    libffi-dev \
    libssl-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash pwmk
WORKDIR /home/pwmk/app
RUN chown -R pwmk:pwmk /home/pwmk

# Switch to non-root user
USER pwmk

# Install Python dependencies
COPY --chown=pwmk:pwmk requirements/ requirements/
COPY --chown=pwmk:pwmk pyproject.toml ./

# Upgrade pip
RUN python3.11 -m pip install --upgrade pip

# Development stage
FROM base AS development

# Install development dependencies
RUN python3.11 -m pip install -e ".[dev,unity,prolog,test]"

# Install pre-commit for development
RUN python3.11 -m pip install pre-commit

# Copy source code
COPY --chown=pwmk:pwmk . .

# Install package in editable mode
RUN python3.11 -m pip install -e .

# Set up pre-commit hooks
RUN pre-commit install

# Expose Jupyter port
EXPOSE 8888

# Default command for development
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Production stage
FROM base AS production

# Copy only necessary files
COPY --chown=pwmk:pwmk pyproject.toml README.md LICENSE ./
COPY --chown=pwmk:pwmk pwmk/ pwmk/

# Install production dependencies only
RUN python3.11 -m pip install .

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python3.11 -c "import pwmk; print('OK')" || exit 1

# Default command for production
CMD ["python3.11", "-m", "pwmk.cli"]

# Testing stage for CI/CD
FROM development AS testing

# Copy test files
COPY --chown=pwmk:pwmk tests/ tests/
COPY --chown=pwmk:pwmk pytest.ini tox.ini ./

# Run tests as default
CMD ["pytest", "tests/", "-v", "--cov=pwmk"]