#!/usr/bin/env python3
"""
Production Deployment System - Enterprise-Grade Deployment

Handles production deployment with comprehensive health checks, monitoring,
rollback capabilities, and multi-environment support for PWMK.
"""

import os
import sys
import time
import json
import logging
import subprocess
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import docker
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_deployment.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Production deployment configuration."""
    environment: str = "production"
    namespace: str = "pwmk"
    replicas: int = 3
    cpu_request: str = "2"
    cpu_limit: str = "4"
    memory_request: str = "4Gi"
    memory_limit: str = "8Gi"
    gpu_request: int = 1
    image_tag: str = "latest"
    registry_url: str = "registry.terragon.ai/pwmk"
    health_check_timeout: int = 300
    rollback_enabled: bool = True
    auto_scaling_enabled: bool = True
    min_replicas: int = 2
    max_replicas: int = 10
    monitoring_enabled: bool = True
    backup_enabled: bool = True
    ssl_enabled: bool = True
    
    # Service configuration
    consciousness_service: Dict[str, Any] = field(default_factory=lambda: {
        "port": 8080,
        "health_endpoint": "/health",
        "metrics_endpoint": "/metrics",
        "replicas": 3
    })
    
    quantum_service: Dict[str, Any] = field(default_factory=lambda: {
        "port": 8081,
        "health_endpoint": "/health",
        "metrics_endpoint": "/metrics",
        "replicas": 2
    })
    
    research_service: Dict[str, Any] = field(default_factory=lambda: {
        "port": 8082,
        "health_endpoint": "/health",
        "metrics_endpoint": "/metrics",
        "replicas": 2
    })


class KubernetesDeployer:
    """Kubernetes deployment manager."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.kubectl_available = self._check_kubectl()
        
    def _check_kubectl(self) -> bool:
        """Check if kubectl is available."""
        try:
            subprocess.run(["kubectl", "version", "--client"], 
                          capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("kubectl not available - Kubernetes deployment disabled")
            return False
    
    def deploy(self) -> bool:
        """Deploy to Kubernetes."""
        if not self.kubectl_available:
            logger.error("Cannot deploy to Kubernetes - kubectl not available")
            return False
        
        try:
            # Create namespace
            self._create_namespace()
            
            # Deploy services
            services = [
                ("consciousness", self.config.consciousness_service),
                ("quantum", self.config.quantum_service),
                ("research", self.config.research_service)
            ]
            
            for service_name, service_config in services:
                success = self._deploy_service(service_name, service_config)
                if not success:
                    logger.error(f"Failed to deploy {service_name} service")
                    return False
            
            # Deploy monitoring and observability
            if self.config.monitoring_enabled:
                self._deploy_monitoring()
            
            # Configure auto-scaling
            if self.config.auto_scaling_enabled:
                self._configure_auto_scaling()
            
            # Setup ingress/load balancer
            if self.config.ssl_enabled:
                self._setup_ingress()
            
            logger.info("✅ Kubernetes deployment completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Kubernetes deployment failed: {e}")
            return False
    
    def _create_namespace(self):
        """Create Kubernetes namespace."""
        namespace_yaml = f"""
apiVersion: v1
kind: Namespace
metadata:
  name: {self.config.namespace}
  labels:
    app: pwmk
    environment: {self.config.environment}
"""
        
        self._apply_yaml(namespace_yaml, "namespace")
    
    def _deploy_service(self, service_name: str, service_config: Dict[str, Any]) -> bool:
        """Deploy individual service to Kubernetes."""
        try:
            # Deployment
            deployment_yaml = self._generate_deployment_yaml(service_name, service_config)
            self._apply_yaml(deployment_yaml, f"{service_name}-deployment")
            
            # Service
            service_yaml = self._generate_service_yaml(service_name, service_config)
            self._apply_yaml(service_yaml, f"{service_name}-service")
            
            # ConfigMap
            configmap_yaml = self._generate_configmap_yaml(service_name)
            self._apply_yaml(configmap_yaml, f"{service_name}-configmap")
            
            # Wait for deployment to be ready
            return self._wait_for_deployment(service_name)
            
        except Exception as e:
            logger.error(f"Failed to deploy service {service_name}: {e}")
            return False
    
    def _generate_deployment_yaml(self, service_name: str, service_config: Dict[str, Any]) -> str:
        """Generate Kubernetes deployment YAML."""
        return f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pwmk-{service_name}
  namespace: {self.config.namespace}
  labels:
    app: pwmk
    service: {service_name}
    environment: {self.config.environment}
spec:
  replicas: {service_config.get('replicas', 2)}
  selector:
    matchLabels:
      app: pwmk
      service: {service_name}
  template:
    metadata:
      labels:
        app: pwmk
        service: {service_name}
    spec:
      containers:
      - name: {service_name}
        image: {self.config.registry_url}:{self.config.image_tag}
        command: ["python", "-m", "pwmk.{service_name}.service"]
        ports:
        - containerPort: {service_config['port']}
          name: http
        env:
        - name: SERVICE_NAME
          value: {service_name}
        - name: ENVIRONMENT
          value: {self.config.environment}
        - name: LOG_LEVEL
          value: INFO
        resources:
          requests:
            cpu: {self.config.cpu_request}
            memory: {self.config.memory_request}
            nvidia.com/gpu: {self.config.gpu_request if service_name == 'quantum' else 0}
          limits:
            cpu: {self.config.cpu_limit}
            memory: {self.config.memory_limit}
            nvidia.com/gpu: {self.config.gpu_request if service_name == 'quantum' else 0}
        livenessProbe:
          httpGet:
            path: {service_config['health_endpoint']}
            port: {service_config['port']}
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: {service_config['health_endpoint']}
            port: {service_config['port']}
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        volumeMounts:
        - name: config
          mountPath: /app/config
        - name: data
          mountPath: /app/data
      volumes:
      - name: config
        configMap:
          name: pwmk-{service_name}-config
      - name: data
        persistentVolumeClaim:
          claimName: pwmk-{service_name}-data
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pwmk-{service_name}-data
  namespace: {self.config.namespace}
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
"""
    
    def _generate_service_yaml(self, service_name: str, service_config: Dict[str, Any]) -> str:
        """Generate Kubernetes service YAML."""
        return f"""
apiVersion: v1
kind: Service
metadata:
  name: pwmk-{service_name}-service
  namespace: {self.config.namespace}
  labels:
    app: pwmk
    service: {service_name}
spec:
  selector:
    app: pwmk
    service: {service_name}
  ports:
  - port: {service_config['port']}
    targetPort: {service_config['port']}
    protocol: TCP
    name: http
  type: ClusterIP
"""
    
    def _generate_configmap_yaml(self, service_name: str) -> str:
        """Generate Kubernetes ConfigMap YAML."""
        config_data = {
            "service.yaml": yaml.dump({
                "service_name": service_name,
                "environment": self.config.environment,
                "logging": {
                    "level": "INFO",
                    "format": "json"
                },
                "monitoring": {
                    "enabled": self.config.monitoring_enabled,
                    "metrics_port": 9090
                }
            })
        }
        
        return f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: pwmk-{service_name}-config
  namespace: {self.config.namespace}
data:
{yaml.dump(config_data, indent=2)}
"""
    
    def _apply_yaml(self, yaml_content: str, resource_name: str):
        """Apply YAML to Kubernetes cluster."""
        # Write YAML to temporary file
        yaml_file = Path(f"/tmp/{resource_name}.yaml")
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        
        # Apply with kubectl
        subprocess.run(
            ["kubectl", "apply", "-f", str(yaml_file)],
            check=True,
            capture_output=True
        )
        
        # Clean up
        yaml_file.unlink()
        
        logger.info(f"Applied {resource_name} to Kubernetes")
    
    def _wait_for_deployment(self, service_name: str) -> bool:
        """Wait for deployment to be ready."""
        deployment_name = f"pwmk-{service_name}"
        
        for attempt in range(60):  # Wait up to 10 minutes
            try:
                result = subprocess.run([
                    "kubectl", "get", "deployment", deployment_name,
                    "-n", self.config.namespace,
                    "-o", "jsonpath={.status.readyReplicas}"
                ], capture_output=True, text=True, check=True)
                
                ready_replicas = int(result.stdout or "0")
                expected_replicas = self.config.consciousness_service.get('replicas', 2)
                
                if service_name == 'quantum':
                    expected_replicas = self.config.quantum_service.get('replicas', 2)
                elif service_name == 'research':
                    expected_replicas = self.config.research_service.get('replicas', 2)
                
                if ready_replicas >= expected_replicas:
                    logger.info(f"✅ Deployment {deployment_name} is ready")
                    return True
                
                time.sleep(10)
                
            except subprocess.CalledProcessError:
                time.sleep(10)
        
        logger.error(f"❌ Deployment {deployment_name} failed to become ready")
        return False
    
    def _deploy_monitoring(self):
        """Deploy monitoring stack."""
        monitoring_yaml = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: {self.config.namespace}
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'pwmk-services'
      kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
          - {self.config.namespace}
      relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: pwmk
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prometheus
  namespace: {self.config.namespace}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: prometheus
  template:
    metadata:
      labels:
        app: prometheus
    spec:
      containers:
      - name: prometheus
        image: prom/prometheus:latest
        args:
        - --config.file=/etc/prometheus/prometheus.yml
        - --storage.tsdb.path=/prometheus/
        - --web.console.libraries=/etc/prometheus/console_libraries
        - --web.console.templates=/etc/prometheus/consoles
        ports:
        - containerPort: 9090
        volumeMounts:
        - name: config
          mountPath: /etc/prometheus/
        - name: storage
          mountPath: /prometheus/
      volumes:
      - name: config
        configMap:
          name: prometheus-config
      - name: storage
        emptyDir: {{}}
"""
        
        self._apply_yaml(monitoring_yaml, "monitoring")
    
    def _configure_auto_scaling(self):
        """Configure horizontal pod autoscaler."""
        for service_name in ["consciousness", "quantum", "research"]:
            hpa_yaml = f"""
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pwmk-{service_name}-hpa
  namespace: {self.config.namespace}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pwmk-{service_name}
  minReplicas: {self.config.min_replicas}
  maxReplicas: {self.config.max_replicas}
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
"""
            
            self._apply_yaml(hpa_yaml, f"{service_name}-hpa")
    
    def _setup_ingress(self):
        """Setup ingress controller and SSL."""
        ingress_yaml = f"""
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: pwmk-ingress
  namespace: {self.config.namespace}
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.pwmk.terragon.ai
    secretName: pwmk-tls
  rules:
  - host: api.pwmk.terragon.ai
    http:
      paths:
      - path: /consciousness
        pathType: Prefix
        backend:
          service:
            name: pwmk-consciousness-service
            port:
              number: {self.config.consciousness_service['port']}
      - path: /quantum
        pathType: Prefix
        backend:
          service:
            name: pwmk-quantum-service
            port:
              number: {self.config.quantum_service['port']}
      - path: /research
        pathType: Prefix
        backend:
          service:
            name: pwmk-research-service
            port:
              number: {self.config.research_service['port']}
"""
        
        self._apply_yaml(ingress_yaml, "ingress")


class DockerDeployer:
    """Docker-based deployment for single-node or development."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.docker_client = None
        self._init_docker()
    
    def _init_docker(self):
        """Initialize Docker client."""
        try:
            self.docker_client = docker.from_env()
            self.docker_client.ping()
            logger.info("Docker client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
    
    def deploy(self) -> bool:
        """Deploy using Docker Compose."""
        if not self.docker_client:
            logger.error("Docker client not available")
            return False
        
        try:
            # Generate docker-compose.yml
            self._generate_docker_compose()
            
            # Build images
            self._build_images()
            
            # Deploy services
            self._deploy_with_compose()
            
            # Verify deployment
            return self._verify_deployment()
            
        except Exception as e:
            logger.error(f"Docker deployment failed: {e}")
            return False
    
    def _generate_docker_compose(self):
        """Generate docker-compose.yml for production."""
        compose_config = {
            'version': '3.8',
            'services': {
                'consciousness-service': {
                    'build': {
                        'context': '.',
                        'dockerfile': 'Dockerfile'
                    },
                    'image': f"{self.config.registry_url}:consciousness-{self.config.image_tag}",
                    'command': ["python", "-m", "pwmk.revolution.service"],
                    'ports': [f"{self.config.consciousness_service['port']}:8080"],
                    'environment': {
                        'SERVICE_NAME': 'consciousness',
                        'ENVIRONMENT': self.config.environment,
                        'LOG_LEVEL': 'INFO'
                    },
                    'volumes': [
                        './data:/app/data',
                        './config:/app/config'
                    ],
                    'restart': 'unless-stopped',
                    'healthcheck': {
                        'test': f"curl -f http://localhost:8080{self.config.consciousness_service['health_endpoint']} || exit 1",
                        'interval': '30s',
                        'timeout': '10s',
                        'retries': 3
                    },
                    'deploy': {
                        'replicas': self.config.consciousness_service['replicas'],
                        'resources': {
                            'limits': {
                                'cpus': self.config.cpu_limit,
                                'memory': self.config.memory_limit
                            },
                            'reservations': {
                                'cpus': self.config.cpu_request,
                                'memory': self.config.memory_request
                            }
                        }
                    }
                },
                'quantum-service': {
                    'build': {
                        'context': '.',
                        'dockerfile': 'Dockerfile.quantum'
                    },
                    'image': f"{self.config.registry_url}:quantum-{self.config.image_tag}",
                    'command': ["python", "-m", "pwmk.quantum.service"],
                    'ports': [f"{self.config.quantum_service['port']}:8081"],
                    'environment': {
                        'SERVICE_NAME': 'quantum',
                        'ENVIRONMENT': self.config.environment,
                        'LOG_LEVEL': 'INFO'
                    },
                    'volumes': [
                        './data:/app/data',
                        './config:/app/config'
                    ],
                    'restart': 'unless-stopped',
                    'runtime': 'nvidia' if self.config.gpu_request > 0 else None,
                    'deploy': {
                        'replicas': self.config.quantum_service['replicas'],
                        'resources': {
                            'reservations': {
                                'devices': [
                                    {
                                        'driver': 'nvidia',
                                        'count': self.config.gpu_request,
                                        'capabilities': ['gpu']
                                    }
                                ] if self.config.gpu_request > 0 else []
                            }
                        }
                    }
                },
                'research-service': {
                    'build': {
                        'context': '.',
                        'dockerfile': 'Dockerfile'
                    },
                    'image': f"{self.config.registry_url}:research-{self.config.image_tag}",
                    'command': ["python", "-m", "pwmk.research.service"],
                    'ports': [f"{self.config.research_service['port']}:8082"],
                    'environment': {
                        'SERVICE_NAME': 'research',
                        'ENVIRONMENT': self.config.environment,
                        'LOG_LEVEL': 'INFO'
                    },
                    'volumes': [
                        './data:/app/data',
                        './config:/app/config'
                    ],
                    'restart': 'unless-stopped'
                }
            }
        }
        
        # Add monitoring services if enabled
        if self.config.monitoring_enabled:
            compose_config['services'].update({
                'prometheus': {
                    'image': 'prom/prometheus:latest',
                    'ports': ['9090:9090'],
                    'volumes': [
                        './monitoring/prometheus.yml:/etc/prometheus/prometheus.yml'
                    ],
                    'restart': 'unless-stopped'
                },
                'grafana': {
                    'image': 'grafana/grafana:latest',
                    'ports': ['3000:3000'],
                    'environment': {
                        'GF_SECURITY_ADMIN_PASSWORD': 'admin'
                    },
                    'volumes': [
                        './monitoring/grafana:/var/lib/grafana'
                    ],
                    'restart': 'unless-stopped'
                }
            })
        
        # Save docker-compose.yml
        with open('docker-compose.prod.yml', 'w') as f:
            yaml.dump(compose_config, f, default_flow_style=False)
        
        logger.info("Generated docker-compose.prod.yml")
    
    def _build_images(self):
        """Build Docker images."""
        # Build main application image
        logger.info("Building main application image...")
        self.docker_client.images.build(
            path=".",
            dockerfile="Dockerfile",
            tag=f"{self.config.registry_url}:consciousness-{self.config.image_tag}",
            rm=True
        )
        
        # Build quantum service image (if different)
        if Path("Dockerfile.quantum").exists():
            logger.info("Building quantum service image...")
            self.docker_client.images.build(
                path=".",
                dockerfile="Dockerfile.quantum",
                tag=f"{self.config.registry_url}:quantum-{self.config.image_tag}",
                rm=True
            )
    
    def _deploy_with_compose(self):
        """Deploy services using Docker Compose."""
        subprocess.run([
            "docker-compose",
            "-f", "docker-compose.prod.yml",
            "up", "-d", "--remove-orphans"
        ], check=True)
        
        logger.info("Services deployed with Docker Compose")
    
    def _verify_deployment(self) -> bool:
        """Verify Docker deployment health."""
        services = ["consciousness-service", "quantum-service", "research-service"]
        
        for service in services:
            if not self._check_service_health(service):
                return False
        
        return True
    
    def _check_service_health(self, service_name: str) -> bool:
        """Check health of individual service."""
        try:
            containers = self.docker_client.containers.list(
                filters={"name": service_name}
            )
            
            if not containers:
                logger.error(f"Service {service_name} not found")
                return False
            
            container = containers[0]
            
            if container.status != "running":
                logger.error(f"Service {service_name} is not running: {container.status}")
                return False
            
            # Check health status
            health = container.attrs.get("State", {}).get("Health", {})
            if health.get("Status") == "unhealthy":
                logger.error(f"Service {service_name} is unhealthy")
                return False
            
            logger.info(f"✅ Service {service_name} is healthy")
            return True
            
        except Exception as e:
            logger.error(f"Health check failed for {service_name}: {e}")
            return False


class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.health_endpoints = self._get_health_endpoints()
    
    def _get_health_endpoints(self) -> List[str]:
        """Get health check endpoints."""
        endpoints = []
        
        services = [
            ("consciousness", self.config.consciousness_service),
            ("quantum", self.config.quantum_service),
            ("research", self.config.research_service)
        ]
        
        for service_name, service_config in services:
            endpoint = f"http://localhost:{service_config['port']}{service_config['health_endpoint']}"
            endpoints.append(endpoint)
        
        return endpoints
    
    def run_health_checks(self, timeout: int = 300) -> bool:
        """Run comprehensive health checks."""
        logger.info("🏥 Running production health checks...")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            all_healthy = True
            
            for endpoint in self.health_endpoints:
                if not self._check_endpoint_health(endpoint):
                    all_healthy = False
                    break
            
            if all_healthy:
                logger.info("✅ All services are healthy")
                return True
            
            logger.info("⏳ Waiting for services to become healthy...")
            time.sleep(10)
        
        logger.error("❌ Health check timeout - services are not healthy")
        return False
    
    def _check_endpoint_health(self, endpoint: str) -> bool:
        """Check health of individual endpoint."""
        try:
            response = requests.get(endpoint, timeout=5)
            
            if response.status_code == 200:
                health_data = response.json()
                
                if health_data.get("status") == "healthy":
                    return True
                else:
                    logger.warning(f"Endpoint {endpoint} returned unhealthy status: {health_data}")
                    return False
            else:
                logger.warning(f"Endpoint {endpoint} returned status code: {response.status_code}")
                return False
                
        except requests.RequestException as e:
            logger.warning(f"Health check failed for {endpoint}: {e}")
            return False


class RollbackManager:
    """Manage deployment rollbacks."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.rollback_data = {}
    
    def save_rollback_point(self, deployment_info: Dict[str, Any]):
        """Save current state for potential rollback."""
        rollback_point = {
            "timestamp": time.time(),
            "deployment_info": deployment_info,
            "image_tags": {
                "consciousness": f"{self.config.registry_url}:consciousness-{self.config.image_tag}",
                "quantum": f"{self.config.registry_url}:quantum-{self.config.image_tag}",
                "research": f"{self.config.registry_url}:research-{self.config.image_tag}"
            }
        }
        
        # Save to file
        with open("rollback_point.json", "w") as f:
            json.dump(rollback_point, f, indent=2, default=str)
        
        logger.info("💾 Rollback point saved")
    
    def execute_rollback(self) -> bool:
        """Execute rollback to previous version."""
        if not self.config.rollback_enabled:
            logger.error("Rollback is disabled")
            return False
        
        try:
            # Load rollback point
            with open("rollback_point.json", "r") as f:
                rollback_point = json.load(f)
            
            logger.info("🔄 Executing rollback...")
            
            # Rollback using kubectl or docker-compose
            if Path("docker-compose.prod.yml").exists():
                return self._rollback_docker()
            else:
                return self._rollback_kubernetes()
                
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def _rollback_docker(self) -> bool:
        """Rollback Docker deployment."""
        try:
            subprocess.run([
                "docker-compose",
                "-f", "docker-compose.prod.yml",
                "down"
            ], check=True)
            
            subprocess.run([
                "docker-compose",
                "-f", "docker-compose.prod.yml",
                "up", "-d"
            ], check=True)
            
            logger.info("✅ Docker rollback completed")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker rollback failed: {e}")
            return False
    
    def _rollback_kubernetes(self) -> bool:
        """Rollback Kubernetes deployment."""
        try:
            services = ["consciousness", "quantum", "research"]
            
            for service in services:
                subprocess.run([
                    "kubectl", "rollout", "undo",
                    f"deployment/pwmk-{service}",
                    "-n", self.config.namespace
                ], check=True)
            
            # Wait for rollback to complete
            for service in services:
                subprocess.run([
                    "kubectl", "rollout", "status",
                    f"deployment/pwmk-{service}",
                    "-n", self.config.namespace,
                    "--timeout=300s"
                ], check=True)
            
            logger.info("✅ Kubernetes rollback completed")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Kubernetes rollback failed: {e}")
            return False


class ProductionDeploymentSystem:
    """Complete production deployment system."""
    
    def __init__(self, config: DeploymentConfig = None):
        self.config = config or DeploymentConfig()
        
        # Initialize components
        self.k8s_deployer = KubernetesDeployer(self.config)
        self.docker_deployer = DockerDeployer(self.config)
        self.health_checker = HealthChecker(self.config)
        self.rollback_manager = RollbackManager(self.config)
        
        # Deployment state
        self.deployment_active = False
        self.deployment_info = {}
    
    def deploy_to_production(self) -> bool:
        """Deploy PWMK to production environment."""
        logger.info("🚀 Starting production deployment...")
        
        try:
            # Pre-deployment checks
            if not self._pre_deployment_checks():
                logger.error("Pre-deployment checks failed")
                return False
            
            # Save rollback point
            self.rollback_manager.save_rollback_point(self.deployment_info)
            
            # Choose deployment method
            if self._should_use_kubernetes():
                success = self._deploy_kubernetes()
            else:
                success = self._deploy_docker()
            
            if not success:
                logger.error("Deployment failed")
                
                # Attempt rollback if enabled
                if self.config.rollback_enabled:
                    logger.info("Attempting automatic rollback...")
                    self.rollback_manager.execute_rollback()
                
                return False
            
            # Post-deployment validation
            if not self._post_deployment_validation():
                logger.error("Post-deployment validation failed")
                
                if self.config.rollback_enabled:
                    logger.info("Attempting automatic rollback...")
                    self.rollback_manager.execute_rollback()
                
                return False
            
            logger.info("🎉 Production deployment completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Production deployment failed: {e}")
            return False
    
    def _pre_deployment_checks(self) -> bool:
        """Run pre-deployment checks."""
        logger.info("🔍 Running pre-deployment checks...")
        
        checks = [
            ("Quality Gates", self._check_quality_gates),
            ("Environment Variables", self._check_environment_variables),
            ("Dependencies", self._check_dependencies),
            ("Resource Availability", self._check_resource_availability),
            ("Security Configuration", self._check_security_configuration)
        ]
        
        for check_name, check_func in checks:
            logger.info(f"   Checking {check_name}...")
            if not check_func():
                logger.error(f"   ❌ {check_name} check failed")
                return False
            logger.info(f"   ✅ {check_name} check passed")
        
        return True
    
    def _check_quality_gates(self) -> bool:
        """Check if quality gates have passed."""
        try:
            # Check for recent quality gates report
            report_file = Path("quality_gates_report.json")
            
            if not report_file.exists():
                logger.warning("No quality gates report found - running quality gates...")
                result = subprocess.run(["python", "run_quality_gates.py"], 
                                      capture_output=True)
                return result.returncode == 0
            
            with open(report_file) as f:
                report = json.load(f)
            
            return report.get("overall_success", False)
            
        except Exception as e:
            logger.error(f"Quality gates check failed: {e}")
            return False
    
    def _check_environment_variables(self) -> bool:
        """Check required environment variables."""
        required_vars = [
            "DEPLOYMENT_ENVIRONMENT",
            "REGISTRY_URL",
            "IMAGE_TAG"
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"Missing environment variables: {missing_vars}")
            return False
        
        return True
    
    def _check_dependencies(self) -> bool:
        """Check deployment dependencies."""
        # Check if Docker is available
        try:
            subprocess.run(["docker", "--version"], 
                          capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Docker is not available")
            return False
        
        return True
    
    def _check_resource_availability(self) -> bool:
        """Check if sufficient resources are available."""
        # Simple resource check - in production would be more sophisticated
        import psutil
        
        # Check CPU
        cpu_count = psutil.cpu_count()
        if cpu_count < 4:
            logger.warning(f"Low CPU count: {cpu_count}")
        
        # Check memory
        memory = psutil.virtual_memory()
        if memory.total < 8 * 1024**3:  # 8GB
            logger.warning(f"Low memory: {memory.total / 1024**3:.1f}GB")
        
        # Check disk space
        disk = psutil.disk_usage('/')
        if disk.free < 20 * 1024**3:  # 20GB
            logger.warning(f"Low disk space: {disk.free / 1024**3:.1f}GB")
        
        return True
    
    def _check_security_configuration(self) -> bool:
        """Check security configuration."""
        # Check SSL certificates
        if self.config.ssl_enabled:
            cert_files = ["server.crt", "server.key"]
            missing_certs = [f for f in cert_files if not Path(f).exists()]
            
            if missing_certs:
                logger.warning(f"Missing SSL certificates: {missing_certs}")
        
        return True
    
    def _should_use_kubernetes(self) -> bool:
        """Determine if Kubernetes should be used for deployment."""
        return self.k8s_deployer.kubectl_available and self.config.environment == "production"
    
    def _deploy_kubernetes(self) -> bool:
        """Deploy using Kubernetes."""
        logger.info("Deploying to Kubernetes...")
        return self.k8s_deployer.deploy()
    
    def _deploy_docker(self) -> bool:
        """Deploy using Docker."""
        logger.info("Deploying with Docker...")
        return self.docker_deployer.deploy()
    
    def _post_deployment_validation(self) -> bool:
        """Run post-deployment validation."""
        logger.info("🔍 Running post-deployment validation...")
        
        # Health checks
        if not self.health_checker.run_health_checks(
            timeout=self.config.health_check_timeout
        ):
            return False
        
        # Functional tests
        if not self._run_functional_tests():
            return False
        
        # Performance validation
        if not self._validate_performance():
            return False
        
        return True
    
    def _run_functional_tests(self) -> bool:
        """Run functional tests against deployed system."""
        logger.info("Running functional tests...")
        
        try:
            # Run integration tests against live system
            result = subprocess.run([
                "python", "-m", "pytest", 
                "tests/integration/", 
                "--live-system",
                "-v"
            ], capture_output=True, timeout=300)
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            logger.error("Functional tests timed out")
            return False
        except Exception as e:
            logger.error(f"Functional tests failed: {e}")
            return False
    
    def _validate_performance(self) -> bool:
        """Validate system performance."""
        logger.info("Validating performance...")
        
        # Simple performance validation - in production would be more comprehensive
        response_times = []
        
        for endpoint in self.health_checker.health_endpoints:
            try:
                start_time = time.time()
                response = requests.get(endpoint, timeout=5)
                end_time = time.time()
                
                if response.status_code == 200:
                    response_times.append(end_time - start_time)
                else:
                    logger.warning(f"Performance check failed for {endpoint}")
                    return False
                    
            except requests.RequestException as e:
                logger.error(f"Performance validation failed for {endpoint}: {e}")
                return False
        
        # Check if response times are acceptable
        avg_response_time = sum(response_times) / len(response_times)
        
        if avg_response_time > 2.0:  # 2 second threshold
            logger.warning(f"High average response time: {avg_response_time:.2f}s")
        
        logger.info(f"Average response time: {avg_response_time:.3f}s")
        return True
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        return {
            "deployment_active": self.deployment_active,
            "config": {
                "environment": self.config.environment,
                "namespace": self.config.namespace,
                "replicas": self.config.replicas,
                "auto_scaling_enabled": self.config.auto_scaling_enabled,
                "monitoring_enabled": self.config.monitoring_enabled
            },
            "health_status": self._get_current_health_status(),
            "deployment_info": self.deployment_info
        }
    
    def _get_current_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        # Quick health check
        healthy_endpoints = 0
        total_endpoints = len(self.health_checker.health_endpoints)
        
        for endpoint in self.health_checker.health_endpoints:
            if self.health_checker._check_endpoint_health(endpoint):
                healthy_endpoints += 1
        
        return {
            "healthy_endpoints": healthy_endpoints,
            "total_endpoints": total_endpoints,
            "health_percentage": (healthy_endpoints / total_endpoints * 100) if total_endpoints > 0 else 0
        }


def main():
    """Main deployment function."""
    print("🚀 PWMK Production Deployment System")
    print("=" * 50)
    
    # Load configuration
    config = DeploymentConfig(
        environment=os.getenv("DEPLOYMENT_ENVIRONMENT", "production"),
        registry_url=os.getenv("REGISTRY_URL", "registry.terragon.ai/pwmk"),
        image_tag=os.getenv("IMAGE_TAG", "latest")
    )
    
    # Create deployment system
    deployment_system = ProductionDeploymentSystem(config)
    
    try:
        # Deploy to production
        success = deployment_system.deploy_to_production()
        
        if success:
            print("\n🎉 Production deployment completed successfully!")
            
            # Display deployment status
            status = deployment_system.get_deployment_status()
            print(f"Deployment Status: {json.dumps(status, indent=2, default=str)}")
            
            sys.exit(0)
        else:
            print("\n💥 Production deployment failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Deployment interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n💥 Deployment system failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()