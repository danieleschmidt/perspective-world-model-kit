#!/bin/bash
# Build script for PWMK Docker images
# Supports multiple targets and platforms

set -euo pipefail

# Default values
TARGET="development"
PLATFORM="linux/amd64"
TAG="latest"
PUSH=false
CACHE_FROM=""
BUILD_ARGS=""
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build PWMK Docker images with support for multiple targets and platforms.

OPTIONS:
    -t, --target TARGET         Build target (development, production, testing) [default: development]
    -p, --platform PLATFORM    Target platform [default: linux/amd64]
    --tag TAG                   Image tag [default: latest]
    --push                      Push image to registry after build
    --cache-from IMAGE          Use external cache source
    --build-arg ARG=VALUE       Pass build argument
    --multi-platform            Build for multiple platforms (amd64, arm64)
    --no-cache                  Disable build cache
    -v, --verbose               Enable verbose output
    -h, --help                  Show this help

EXAMPLES:
    $0                                      # Build development image
    $0 -t production --tag v1.0.0          # Build production image with tag
    $0 --multi-platform --push             # Build and push multi-platform image
    $0 -t testing --build-arg PYTHON_VERSION=3.11

TARGETS:
    development     Full development environment with all tools
    production      Minimal production image
    testing         Testing environment with test dependencies

EOF
}

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

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            TARGET="$2"
            shift 2
            ;;
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --cache-from)
            CACHE_FROM="$2"
            shift 2
            ;;
        --build-arg)
            BUILD_ARGS="$BUILD_ARGS --build-arg $2"
            shift 2
            ;;
        --multi-platform)
            PLATFORM="linux/amd64,linux/arm64"
            shift
            ;;
        --no-cache)
            BUILD_ARGS="$BUILD_ARGS --no-cache"
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate target
if [[ ! "$TARGET" =~ ^(development|production|testing)$ ]]; then
    log_error "Invalid target: $TARGET"
    log_error "Valid targets: development, production, testing"
    exit 1
fi

# Set image name based on target
case $TARGET in
    "development")
        IMAGE_NAME="pwmk:dev-$TAG"
        ;;
    "production")
        IMAGE_NAME="pwmk:prod-$TAG"
        ;;
    "testing")
        IMAGE_NAME="pwmk:test-$TAG"
        ;;
esac

# Build command construction
BUILD_CMD="docker build"

if [[ "$VERBOSE" == "true" ]]; then
    BUILD_CMD="$BUILD_CMD --progress=plain"
fi

if [[ -n "$CACHE_FROM" ]]; then
    BUILD_CMD="$BUILD_CMD --cache-from $CACHE_FROM"
fi

if [[ -n "$BUILD_ARGS" ]]; then
    BUILD_CMD="$BUILD_CMD $BUILD_ARGS"
fi

BUILD_CMD="$BUILD_CMD --target $TARGET"
BUILD_CMD="$BUILD_CMD --platform $PLATFORM"
BUILD_CMD="$BUILD_CMD -t $IMAGE_NAME"
BUILD_CMD="$BUILD_CMD ."

# Pre-build checks
log_info "Starting PWMK Docker build process..."
log_info "Target: $TARGET"
log_info "Platform: $PLATFORM" 
log_info "Image: $IMAGE_NAME"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    log_error "Docker is not running or not accessible"
    exit 1
fi

# Check if Dockerfile exists
if [[ ! -f "Dockerfile" ]]; then
    log_error "Dockerfile not found in current directory"
    exit 1
fi

# Build the image
log_info "Building Docker image..."
if [[ "$VERBOSE" == "true" ]]; then
    log_info "Build command: $BUILD_CMD"
fi

if eval "$BUILD_CMD"; then
    log_success "Image built successfully: $IMAGE_NAME"
else
    log_error "Build failed"
    exit 1
fi

# Push if requested
if [[ "$PUSH" == "true" ]]; then
    log_info "Pushing image to registry..."
    if docker push "$IMAGE_NAME"; then
        log_success "Image pushed successfully: $IMAGE_NAME"
    else
        log_error "Push failed"
        exit 1
    fi
fi

# Post-build operations
log_info "Build completed successfully!"

# Show image details
if [[ "$VERBOSE" == "true" ]]; then
    log_info "Image details:"
    docker images "$IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
fi

# Generate build report
BUILD_REPORT=$(cat << EOF
=== PWMK Build Report ===
Target: $TARGET
Platform: $PLATFORM
Image: $IMAGE_NAME
Build Time: $(date)
Docker Version: $(docker version --format '{{.Server.Version}}')
EOF
)

if [[ "$VERBOSE" == "true" ]]; then
    echo "$BUILD_REPORT"
fi

# Save build report
echo "$BUILD_REPORT" > "build-report-$(date +%Y%m%d-%H%M%S).txt"

log_success "Build process completed!"