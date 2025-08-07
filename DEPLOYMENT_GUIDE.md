# PWMK Sentiment Analysis - Deployment Guide

This guide provides comprehensive instructions for deploying the PWMK Sentiment Analysis system across different environments and platforms.

## 🚀 Quick Start

### Prerequisites

- **Docker & Docker Compose**: For local development and container-based deployments
- **Python 3.9+**: For local development
- **Git**: For version control
- **kubectl**: For Kubernetes deployments  
- **Helm**: For Kubernetes package management
- **Terraform**: For infrastructure as code (cloud deployments)
- **Cloud CLI**: AWS CLI, gcloud, or Azure CLI depending on target platform

### 1. Clone Repository

```bash
git clone https://github.com/danieleschmidt/perspective-world-model-kit.git
cd perspective-world-model-kit
git checkout terragon/autonomous-sdlc-sentiment-analyzer
```

### 2. Local Development Deployment

```bash
# Quick Docker Compose deployment
./scripts/deploy.sh --type docker --environment development

# Or manually with Docker Compose
docker-compose -f docker-compose.sentiment.yml up -d
```

### 3. Production Deployment

```bash
# Kubernetes deployment
./scripts/deploy.sh --type k8s --environment production

# AWS deployment with Terraform
./scripts/deploy.sh --type aws --environment production --region us-east-1
```

## 📋 Deployment Options

### 🐳 Docker Compose (Recommended for Development)

**Use Case**: Local development, testing, small-scale deployments

**Services Included**:
- Sentiment Analysis API
- Redis (caching)
- PostgreSQL (compliance data)
- Prometheus (metrics)
- Grafana (monitoring)
- Nginx (reverse proxy)

**Commands**:
```bash
# Development deployment
./scripts/deploy.sh --type docker --environment development

# Production-like deployment with scaling
./scripts/deploy.sh --type docker --environment production

# Manual deployment
docker-compose -f docker-compose.sentiment.yml up -d

# With scaling profile (includes worker nodes)
docker-compose -f docker-compose.sentiment.yml --profile scaling up -d

# With logging profile (includes ELK stack)
docker-compose -f docker-compose.sentiment.yml --profile logging up -d
```

**Access Points**:
- Sentiment API: http://localhost:8000
- Grafana Dashboard: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- Kibana (with logging profile): http://localhost:5601

### ☸️ Kubernetes (Recommended for Production)

**Use Case**: Production deployments, high availability, auto-scaling

**Features**:
- Horizontal Pod Autoscaling (HPA)
- Network policies for security
- Pod disruption budgets
- Health checks and readiness probes
- Service monitoring with Prometheus
- Ingress with TLS termination

**Commands**:
```bash
# Deploy to existing Kubernetes cluster
./scripts/deploy.sh --type k8s --environment production --namespace pwmk-sentiment

# Custom namespace deployment
./scripts/deploy.sh --type k8s --environment staging --namespace pwmk-staging

# Manual deployment
kubectl apply -f k8s/sentiment-deployment.yaml

# Check deployment status
kubectl get pods -n pwmk-sentiment
kubectl get services -n pwmk-sentiment
kubectl get ingress -n pwmk-sentiment
```

**Scaling Configuration**:
- **Min Replicas**: 3
- **Max Replicas**: 20
- **CPU Target**: 70%
- **Memory Target**: 80%

### ☁️ Cloud Deployments

#### AWS (Production Ready)

**Infrastructure**:
- EKS cluster with managed node groups
- RDS PostgreSQL (Multi-AZ)
- ElastiCache Redis (Replication Group)
- Application Load Balancer
- S3 for model storage
- Secrets Manager for sensitive data
- CloudWatch for logging and monitoring

**Commands**:
```bash
# Full AWS deployment
./scripts/deploy.sh --type aws --environment production --region us-east-1

# Manual Terraform deployment
cd terraform/
terraform init
terraform plan -var="environment=production"
terraform apply
```

**Estimated Costs** (us-east-1, monthly):
- EKS Cluster: $73
- EC2 Instances (5x t3.large): $370
- RDS PostgreSQL (db.t3.medium): $68
- ElastiCache Redis (2x cache.t3.medium): $110
- Load Balancer: $23
- **Total**: ~$644/month

#### GCP (Coming Soon)

**Infrastructure** (Planned):
- GKE cluster
- Cloud SQL PostgreSQL
- Memorystore Redis
- Cloud Load Balancing
- Cloud Storage
- Secret Manager

#### Azure (Coming Soon)

**Infrastructure** (Planned):
- AKS cluster
- Azure Database for PostgreSQL
- Azure Cache for Redis
- Application Gateway
- Blob Storage
- Key Vault

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENVIRONMENT` | Deployment environment | `production` | Yes |
| `PWMK_REGION` | Deployment region | `us-east-1` | Yes |
| `PWMK_LOG_LEVEL` | Logging level | `INFO` | No |
| `PWMK_CACHE_ENABLED` | Enable Redis caching | `true` | No |
| `PWMK_MONITORING_ENABLED` | Enable monitoring | `true` | No |
| `PWMK_MODEL_CACHE_DIR` | Model storage directory | `/app/models` | No |
| `POSTGRES_PASSWORD` | PostgreSQL password | Generated | Yes |
| `REDIS_PASSWORD` | Redis password | Generated | Yes |
| `JWT_SECRET` | JWT signing secret | Generated | Yes |

### Regional Configuration

The system supports multi-region deployments with compliance-aware configurations:

| Region | Compliance | Data Residency | Languages |
|--------|------------|----------------|-----------|
| US East | CCPA | Optional | en, es |
| EU West | GDPR | Required | en, fr, de, es, it |
| EU Central | GDPR | Required | de, en, fr |
| Asia Pacific | PDPA-SG | Required | en, zh, ja |
| Brazil | LGPD | Required | pt, en |

### Scaling Configuration

#### Docker Compose Scaling
```bash
# Scale API service
docker-compose -f docker-compose.sentiment.yml up -d --scale sentiment-api=3

# Scale workers
docker-compose -f docker-compose.sentiment.yml up -d --scale sentiment-worker=5
```

#### Kubernetes Auto-scaling
```yaml
# HPA automatically scales based on:
# - CPU utilization (target: 70%)
# - Memory utilization (target: 80%)
# - Custom metrics (requests per second)

# Manual scaling
kubectl scale deployment sentiment-api --replicas=10 -n pwmk-sentiment
```

## 🔒 Security

### Network Security
- **Docker**: Internal bridge network with service discovery
- **Kubernetes**: Network policies restrict inter-pod communication
- **Cloud**: VPC with private subnets, security groups, NACLs

### Data Security
- **Encryption at Rest**: AES-256 for all stored data
- **Encryption in Transit**: TLS 1.3 for all communications
- **Secrets Management**: Kubernetes secrets, AWS Secrets Manager
- **Key Management**: AWS KMS, Azure Key Vault (planned)

### Compliance Features
- **GDPR Compliance**: Right to erasure, data portability, consent management
- **CCPA Compliance**: Data subject rights, opt-out mechanisms
- **Audit Logging**: All data processing activities logged
- **Data Retention**: Automatic cleanup based on regional policies

## 📊 Monitoring & Observability

### Metrics (Prometheus)
- Application metrics (request rate, latency, errors)
- System metrics (CPU, memory, disk, network)
- Business metrics (sentiment analysis accuracy, throughput)
- Custom metrics (model performance, cache hit rates)

### Dashboards (Grafana)
- **System Overview**: Infrastructure health and performance
- **Application Performance**: API metrics, error rates, latencies
- **Business Intelligence**: Sentiment analysis insights
- **Compliance Dashboard**: Data processing activities, retention status

### Logging
- **Structured Logging**: JSON format with correlation IDs
- **Log Aggregation**: ELK stack (Elasticsearch, Logstash, Kibana)
- **Log Retention**: Configurable based on compliance requirements
- **Alerting**: Slack, PagerDuty, email notifications

### Health Checks
- **Liveness Probes**: Container health monitoring
- **Readiness Probes**: Service availability checks
- **Health Endpoints**: `/health`, `/ready`, `/metrics`

## 🧪 Testing

### Test Types
```bash
# Unit tests
python -m pytest tests/unit/ -v

# Integration tests
python -m pytest tests/integration/ -v

# Performance tests
python -m pytest tests/performance/ -v --benchmark

# Security tests
python -m pytest tests/security/ -v

# End-to-end tests
python -m pytest tests/e2e/ -v
```

### Test Environments
- **Local**: Docker Compose with test data
- **Staging**: Kubernetes cluster with production-like setup
- **Production**: Blue-green deployments with canary testing

## 🚨 Troubleshooting

### Common Issues

#### 1. Service Not Starting
```bash
# Check logs
docker-compose -f docker-compose.sentiment.yml logs sentiment-api

# Kubernetes
kubectl logs -n pwmk-sentiment deployment/sentiment-api
kubectl describe pod -n pwmk-sentiment <pod-name>
```

#### 2. Database Connection Issues
```bash
# Test PostgreSQL connection
docker exec -it pwmk-postgres psql -U pwmk -d pwmk_sentiment

# Check environment variables
kubectl get secret sentiment-secrets -n pwmk-sentiment -o yaml
```

#### 3. Model Loading Issues
```bash
# Check model directory permissions
ls -la /app/models/

# Verify S3 access (AWS)
aws s3 ls s3://bucket-name/models/
```

#### 4. Memory Issues
```bash
# Check resource usage
docker stats
kubectl top pods -n pwmk-sentiment

# Increase memory limits in deployment configs
```

### Performance Tuning

#### Database Optimization
```sql
-- PostgreSQL settings
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
```

#### Redis Optimization
```bash
# Memory optimization
CONFIG SET maxmemory-policy allkeys-lru
CONFIG SET maxmemory 512mb
```

#### Application Tuning
```python
# Environment variables
PWMK_WORKER_CONCURRENCY=4
PWMK_BATCH_SIZE=32
PWMK_CACHE_TTL=3600
PWMK_MODEL_CACHE_SIZE=1000
```

## 📈 Capacity Planning

### Resource Requirements

#### Small Deployment (< 1000 requests/day)
- **API**: 1 replica, 1 CPU, 2GB RAM
- **Database**: db.t3.micro
- **Cache**: cache.t3.micro
- **Storage**: 10GB

#### Medium Deployment (< 100K requests/day)
- **API**: 3 replicas, 2 CPU, 4GB RAM each
- **Database**: db.t3.medium
- **Cache**: cache.t3.medium (2 nodes)
- **Storage**: 100GB

#### Large Deployment (> 1M requests/day)
- **API**: 10+ replicas, 4 CPU, 8GB RAM each
- **Database**: db.r5.large (Multi-AZ)
- **Cache**: cache.r5.large (3+ nodes)
- **Storage**: 1TB+

### Scaling Guidelines
- **CPU Usage**: Scale up at 70% average utilization
- **Memory Usage**: Scale up at 80% average utilization
- **Response Time**: Scale up if p95 > 2 seconds
- **Error Rate**: Scale up if error rate > 1%

## 🔄 Maintenance

### Regular Tasks
- **Weekly**: Security patch updates
- **Monthly**: Dependency updates, performance review
- **Quarterly**: Disaster recovery testing, compliance audit
- **Annually**: Major version upgrades, architecture review

### Backup Strategies
- **Database**: Daily automated backups, 7-day retention
- **Models**: Versioned storage with lifecycle policies
- **Configurations**: Git-based version control
- **Metrics**: Long-term storage for historical analysis

### Disaster Recovery
- **RTO**: Recovery Time Objective < 4 hours
- **RPO**: Recovery Point Objective < 1 hour
- **Multi-Region**: Automated failover for critical workloads
- **Testing**: Monthly DR drills

## 📞 Support

### Getting Help
1. **Documentation**: Check this guide and inline code documentation
2. **Issues**: Create GitHub issue with detailed reproduction steps
3. **Discussions**: Use GitHub Discussions for questions
4. **Emergency**: Follow on-call escalation procedures

### Reporting Issues
Include the following information:
- Environment details (Docker/K8s/Cloud)
- Error messages and stack traces
- Configuration files (redacted)
- Steps to reproduce
- Expected vs actual behavior

---

## 📄 License

This deployment guide is part of the PWMK project and is licensed under the Apache License 2.0.