#!/bin/bash

# PWMK Sentiment Analysis Deployment Script
# Supports Docker Compose, Kubernetes, and cloud deployments

set -e

# Default configuration
ENVIRONMENT="production"
DEPLOYMENT_TYPE="docker"
REGION="us-east-1"
NAMESPACE="pwmk-sentiment"
HELM_CHART_VERSION="latest"
SKIP_TESTS="false"
SKIP_BUILD="false"
BUILD_TARGET="production"

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

# Help function
show_help() {
    cat << EOF
PWMK Sentiment Analysis Deployment Script

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -e, --environment       Environment (development|staging|production) [default: production]
    -t, --type              Deployment type (docker|k8s|aws|gcp|azure) [default: docker]
    -r, --region            Cloud region [default: us-east-1]
    -n, --namespace         Kubernetes namespace [default: pwmk-sentiment]
    --helm-version          Helm chart version [default: latest]
    --skip-tests            Skip running tests before deployment
    --skip-build            Skip building Docker images
    --build-target          Docker build target (development|production|slim) [default: production]

EXAMPLES:
    # Deploy with Docker Compose (local development)
    $0 --type docker --environment development

    # Deploy to Kubernetes
    $0 --type k8s --environment production --namespace pwmk-sentiment

    # Deploy to AWS
    $0 --type aws --environment production --region us-east-1

    # Deploy to staging environment, skip tests
    $0 --type k8s --environment staging --skip-tests

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--type)
            DEPLOYMENT_TYPE="$2"
            shift 2
            ;;
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --helm-version)
            HELM_CHART_VERSION="$2"
            shift 2
            ;;
        --skip-tests)
            SKIP_TESTS="true"
            shift
            ;;
        --skip-build)
            SKIP_BUILD="true"
            shift
            ;;
        --build-target)
            BUILD_TARGET="$2"
            shift 2
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
    log_error "Invalid environment: $ENVIRONMENT"
    exit 1
fi

# Validate deployment type
if [[ ! "$DEPLOYMENT_TYPE" =~ ^(docker|k8s|aws|gcp|azure)$ ]]; then
    log_error "Invalid deployment type: $DEPLOYMENT_TYPE"
    exit 1
fi

log_info "Starting deployment with configuration:"
log_info "  Environment: $ENVIRONMENT"
log_info "  Deployment Type: $DEPLOYMENT_TYPE"
log_info "  Region: $REGION"
log_info "  Namespace: $NAMESPACE"
log_info "  Build Target: $BUILD_TARGET"

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if required commands exist
    local required_commands=("docker" "git")
    
    case $DEPLOYMENT_TYPE in
        k8s)
            required_commands+=("kubectl" "helm")
            ;;
        aws)
            required_commands+=("aws" "terraform" "kubectl")
            ;;
        gcp)
            required_commands+=("gcloud" "terraform" "kubectl")
            ;;
        azure)
            required_commands+=("az" "terraform" "kubectl")
            ;;
    esac

    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            log_error "$cmd is required but not installed"
            exit 1
        fi
    done

    # Check Docker is running
    if ! docker info &> /dev/null; then
        log_error "Docker is not running"
        exit 1
    fi

    log_success "Prerequisites check passed"
}

# Run tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_warning "Skipping tests"
        return
    fi

    log_info "Running tests..."

    # Build test image
    docker build \
        --target development \
        --tag pwmk-sentiment:test \
        --file docker/Dockerfile.sentiment \
        .

    # Run unit tests
    docker run --rm \
        --volume "$(pwd):/app" \
        --workdir /app \
        pwmk-sentiment:test \
        python -m pytest tests/unit/ -v --junitxml=test-results/unit-tests.xml

    # Run integration tests
    docker run --rm \
        --volume "$(pwd):/app" \
        --workdir /app \
        pwmk-sentiment:test \
        python -m pytest tests/integration/ -v --junitxml=test-results/integration-tests.xml

    # Run security tests
    docker run --rm \
        --volume "$(pwd):/app" \
        --workdir /app \
        pwmk-sentiment:test \
        python -m pytest tests/security/ -v --junitxml=test-results/security-tests.xml

    log_success "All tests passed"
}

# Build Docker images
build_images() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        log_warning "Skipping Docker build"
        return
    fi

    log_info "Building Docker images..."

    local image_tag="pwmk-sentiment:${ENVIRONMENT}-$(git rev-parse --short HEAD)"
    
    # Build main image
    docker build \
        --target "$BUILD_TARGET" \
        --tag "$image_tag" \
        --tag "pwmk-sentiment:${ENVIRONMENT}-latest" \
        --tag "pwmk-sentiment:latest" \
        --file docker/Dockerfile.sentiment \
        --build-arg ENVIRONMENT="$ENVIRONMENT" \
        .

    log_success "Docker images built successfully"
    log_info "Tagged as: $image_tag"
}

# Deploy with Docker Compose
deploy_docker() {
    log_info "Deploying with Docker Compose..."

    # Set environment variables
    export COMPOSE_PROJECT_NAME="pwmk-sentiment-${ENVIRONMENT}"
    export ENVIRONMENT="$ENVIRONMENT"
    export PWMK_REGION="$REGION"

    # Create necessary directories
    mkdir -p data/models data/logs

    # Deploy services
    if [[ "$ENVIRONMENT" == "development" ]]; then
        docker-compose -f docker-compose.sentiment.yml up -d
    else
        docker-compose -f docker-compose.sentiment.yml --profile scaling up -d
    fi

    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30

    # Health check
    if curl -f http://localhost:8000/health &> /dev/null; then
        log_success "Sentiment API is healthy"
    else
        log_error "Sentiment API health check failed"
        docker-compose -f docker-compose.sentiment.yml logs sentiment-api
        exit 1
    fi

    log_success "Docker deployment completed successfully"
    log_info "Services available at:"
    log_info "  Sentiment API: http://localhost:8000"
    log_info "  Grafana: http://localhost:3000 (admin/admin)"
    log_info "  Prometheus: http://localhost:9090"
}

# Deploy to Kubernetes
deploy_k8s() {
    log_info "Deploying to Kubernetes..."

    # Check if kubectl is configured
    if ! kubectl cluster-info &> /dev/null; then
        log_error "kubectl is not configured or cluster is not accessible"
        exit 1
    fi

    # Create namespace if it doesn't exist
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

    # Apply secrets
    log_info "Creating secrets..."
    kubectl create secret generic sentiment-secrets \
        --namespace="$NAMESPACE" \
        --from-literal=POSTGRES_PASSWORD="$(openssl rand -base64 32)" \
        --from-literal=REDIS_PASSWORD="$(openssl rand -base64 32)" \
        --from-literal=JWT_SECRET="$(openssl rand -base64 64)" \
        --dry-run=client -o yaml | kubectl apply -f -

    # Apply Kubernetes manifests
    log_info "Applying Kubernetes manifests..."
    kubectl apply -f k8s/sentiment-deployment.yaml

    # Wait for deployment to be ready
    log_info "Waiting for deployment to be ready..."
    kubectl wait --namespace="$NAMESPACE" \
        --for=condition=available \
        --timeout=600s \
        deployment/sentiment-api

    # Get service endpoint
    local service_ip
    if kubectl get service sentiment-api-service --namespace="$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' &> /dev/null; then
        service_ip=$(kubectl get service sentiment-api-service --namespace="$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        log_success "Kubernetes deployment completed successfully"
        log_info "Service available at: http://$service_ip"
    else
        log_success "Kubernetes deployment completed successfully"
        log_info "Use 'kubectl port-forward' to access the service locally"
        log_info "  kubectl port-forward --namespace=$NAMESPACE service/sentiment-api-service 8000:80"
    fi
}

# Deploy to AWS
deploy_aws() {
    log_info "Deploying to AWS..."

    # Check AWS CLI configuration
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS CLI is not configured"
        exit 1
    fi

    # Initialize Terraform
    log_info "Initializing Terraform..."
    cd terraform/
    terraform init

    # Plan Terraform changes
    log_info "Planning Terraform changes..."
    terraform plan \
        -var="environment=$ENVIRONMENT" \
        -var="aws_region=$REGION" \
        -out=tfplan

    # Apply Terraform changes
    log_info "Applying Terraform changes..."
    terraform apply tfplan

    # Get cluster name
    local cluster_name
    cluster_name=$(terraform output -raw cluster_name)

    # Update kubectl config
    log_info "Updating kubectl configuration..."
    aws eks --region "$REGION" update-kubeconfig --name "$cluster_name"

    # Deploy to EKS cluster
    cd ../
    DEPLOYMENT_TYPE="k8s" deploy_k8s

    log_success "AWS deployment completed successfully"
}

# Deploy to GCP
deploy_gcp() {
    log_info "Deploying to GCP..."
    
    # Check gcloud configuration
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
        log_error "gcloud is not authenticated"
        exit 1
    fi

    log_warning "GCP deployment not yet implemented"
    exit 1
}

# Deploy to Azure
deploy_azure() {
    log_info "Deploying to Azure..."
    
    # Check Azure CLI configuration
    if ! az account show &> /dev/null; then
        log_error "Azure CLI is not authenticated"
        exit 1
    fi

    log_warning "Azure deployment not yet implemented"
    exit 1
}

# Post-deployment verification
post_deployment_verification() {
    log_info "Running post-deployment verification..."

    case $DEPLOYMENT_TYPE in
        docker)
            # Test Docker deployment
            if curl -f http://localhost:8000/health &> /dev/null; then
                log_success "Health check passed"
            else
                log_error "Health check failed"
                return 1
            fi
            ;;
        k8s)
            # Test Kubernetes deployment
            local pod_name
            pod_name=$(kubectl get pods --namespace="$NAMESPACE" -l app=sentiment-api -o jsonpath='{.items[0].metadata.name}')
            
            if kubectl exec --namespace="$NAMESPACE" "$pod_name" -- curl -f http://localhost:8000/health &> /dev/null; then
                log_success "Health check passed"
            else
                log_error "Health check failed"
                return 1
            fi
            ;;
    esac

    log_success "Post-deployment verification completed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f tfplan
}

# Main deployment function
main() {
    # Set up trap for cleanup
    trap cleanup EXIT

    # Create results directory
    mkdir -p test-results

    # Run deployment steps
    check_prerequisites
    run_tests
    build_images

    # Deploy based on type
    case $DEPLOYMENT_TYPE in
        docker)
            deploy_docker
            ;;
        k8s)
            deploy_k8s
            ;;
        aws)
            deploy_aws
            ;;
        gcp)
            deploy_gcp
            ;;
        azure)
            deploy_azure
            ;;
    esac

    # Verify deployment
    post_deployment_verification

    log_success "Deployment completed successfully!"
}

# Run main function
main "$@"