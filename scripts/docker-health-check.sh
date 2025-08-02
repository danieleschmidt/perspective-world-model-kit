#!/bin/bash
# Docker health check script for PWMK services
# Provides comprehensive health monitoring for all containerized services

set -euo pipefail

# Configuration
TIMEOUT=30
VERBOSE=false
CHECK_ALL=false
SERVICES=()

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] [SERVICES...]

Perform health checks on PWMK Docker services.

OPTIONS:
    -a, --all               Check all services
    -t, --timeout SECONDS   Health check timeout [default: 30]
    -v, --verbose           Enable verbose output
    -h, --help              Show this help

SERVICES:
    app         Main PWMK application
    postgres    PostgreSQL database
    redis       Redis cache
    prometheus  Prometheus monitoring
    grafana     Grafana dashboards
    nginx       Nginx reverse proxy

EXAMPLES:
    $0 app postgres          # Check specific services
    $0 --all                 # Check all services
    $0 -v --timeout 60 app   # Verbose check with custom timeout

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -a|--all)
            CHECK_ALL=true
            shift
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
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
            SERVICES+=("$1")
            shift
            ;;
    esac
done

# Set default services if none specified and not checking all
if [[ "$CHECK_ALL" == "false" && ${#SERVICES[@]} -eq 0 ]]; then
    SERVICES=("app")
fi

# Health check functions
check_app_health() {
    local service_name="pwmk_app"
    local container_id
    
    log_info "Checking PWMK application health..."
    
    # Check if container is running
    if ! container_id=$(docker ps -q -f name="$service_name"); then
        log_error "Failed to get container ID for $service_name"
        return 1
    fi
    
    if [[ -z "$container_id" ]]; then
        log_error "Container $service_name is not running"
        return 1
    fi
    
    # Check application health endpoint
    if timeout "$TIMEOUT" docker exec "$container_id" python -c "import pwmk; print('OK')" > /dev/null 2>&1; then
        log_success "PWMK application is healthy"
        return 0
    else
        log_error "PWMK application health check failed"
        return 1
    fi
}

check_postgres_health() {
    local service_name="pwmk_postgres"
    local container_id
    
    log_info "Checking PostgreSQL health..."
    
    if ! container_id=$(docker ps -q -f name="$service_name"); then
        log_error "Failed to get container ID for $service_name"
        return 1
    fi
    
    if [[ -z "$container_id" ]]; then
        log_error "Container $service_name is not running"
        return 1
    fi
    
    # Check PostgreSQL connection
    if timeout "$TIMEOUT" docker exec "$container_id" pg_isready -U pwmk > /dev/null 2>&1; then
        log_success "PostgreSQL is healthy"
        
        if [[ "$VERBOSE" == "true" ]]; then
            local db_info
            db_info=$(docker exec "$container_id" psql -U pwmk -d pwmk -c "SELECT version();" -t | head -1 | xargs)
            log_info "Database version: $db_info"
        fi
        
        return 0
    else
        log_error "PostgreSQL health check failed"
        return 1
    fi
}

check_redis_health() {
    local service_name="pwmk_redis"
    local container_id
    
    log_info "Checking Redis health..."
    
    if ! container_id=$(docker ps -q -f name="$service_name"); then
        log_error "Failed to get container ID for $service_name"
        return 1
    fi
    
    if [[ -z "$container_id" ]]; then
        log_error "Container $service_name is not running"
        return 1
    fi
    
    # Check Redis ping
    if timeout "$TIMEOUT" docker exec "$container_id" redis-cli ping | grep -q "PONG"; then
        log_success "Redis is healthy"
        
        if [[ "$VERBOSE" == "true" ]]; then
            local redis_info
            redis_info=$(docker exec "$container_id" redis-cli info server | grep "redis_version" | cut -d: -f2 | tr -d '\r')
            log_info "Redis version: $redis_info"
        fi
        
        return 0
    else
        log_error "Redis health check failed"
        return 1
    fi
}

check_prometheus_health() {
    local service_name="pwmk_prometheus"
    local container_id
    
    log_info "Checking Prometheus health..."
    
    if ! container_id=$(docker ps -q -f name="$service_name"); then
        log_error "Failed to get container ID for $service_name"
        return 1
    fi
    
    if [[ -z "$container_id" ]]; then
        log_error "Container $service_name is not running"
        return 1
    fi
    
    # Check Prometheus health endpoint
    if timeout "$TIMEOUT" docker exec "$container_id" wget --quiet --tries=1 --spider http://localhost:9090/-/healthy 2>/dev/null; then
        log_success "Prometheus is healthy"
        return 0
    else
        log_error "Prometheus health check failed"
        return 1
    fi
}

check_grafana_health() {
    local service_name="pwmk_grafana"
    local container_id
    
    log_info "Checking Grafana health..."
    
    if ! container_id=$(docker ps -q -f name="$service_name"); then
        log_error "Failed to get container ID for $service_name"
        return 1
    fi
    
    if [[ -z "$container_id" ]]; then
        log_error "Container $service_name is not running"
        return 1
    fi
    
    # Check Grafana health endpoint
    if timeout "$TIMEOUT" docker exec "$container_id" curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
        log_success "Grafana is healthy"
        return 0
    else
        log_error "Grafana health check failed"
        return 1
    fi
}

check_nginx_health() {
    local service_name="pwmk_nginx"
    local container_id
    
    log_info "Checking Nginx health..."
    
    if ! container_id=$(docker ps -q -f name="$service_name"); then
        log_error "Failed to get container ID for $service_name"
        return 1
    fi
    
    if [[ -z "$container_id" ]]; then
        log_error "Container $service_name is not running"
        return 1
    fi
    
    # Check Nginx status
    if timeout "$TIMEOUT" docker exec "$container_id" nginx -t > /dev/null 2>&1; then
        log_success "Nginx is healthy"
        return 0
    else
        log_error "Nginx health check failed"
        return 1
    fi
}

# Get list of services to check
get_services_to_check() {
    if [[ "$CHECK_ALL" == "true" ]]; then
        echo "app postgres redis prometheus grafana nginx"
    else
        echo "${SERVICES[@]}"
    fi
}

# Main health check loop
main() {
    log_info "Starting PWMK health checks..."
    
    local services_to_check
    services_to_check=($(get_services_to_check))
    
    local failed_services=()
    local passed_services=()
    
    for service in "${services_to_check[@]}"; do
        case $service in
            "app")
                if check_app_health; then
                    passed_services+=("$service")
                else
                    failed_services+=("$service")
                fi
                ;;
            "postgres")
                if check_postgres_health; then
                    passed_services+=("$service")
                else
                    failed_services+=("$service")
                fi
                ;;
            "redis")
                if check_redis_health; then
                    passed_services+=("$service")
                else
                    failed_services+=("$service")
                fi
                ;;
            "prometheus")
                if check_prometheus_health; then
                    passed_services+=("$service")
                else
                    failed_services+=("$service")
                fi
                ;;
            "grafana")
                if check_grafana_health; then
                    passed_services+=("$service")
                else
                    failed_services+=("$service")
                fi
                ;;
            "nginx")
                if check_nginx_health; then
                    passed_services+=("$service")
                else
                    failed_services+=("$service")
                fi
                ;;
            *)
                log_warning "Unknown service: $service"
                failed_services+=("$service")
                ;;
        esac
        
        echo # Add spacing between checks
    done
    
    # Summary
    echo "=== Health Check Summary ==="
    
    if [[ ${#passed_services[@]} -gt 0 ]]; then
        log_success "Healthy services: ${passed_services[*]}"
    fi
    
    if [[ ${#failed_services[@]} -gt 0 ]]; then
        log_error "Failed services: ${failed_services[*]}"
        exit 1
    else
        log_success "All checked services are healthy!"
        exit 0
    fi
}

# Run main function
main "$@"