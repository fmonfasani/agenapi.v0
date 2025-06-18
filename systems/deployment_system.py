import asyncio
import json
import yaml
import os
import subprocess
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
import tempfile
import shutil

from core.autonomous_agent_framework import AgentFramework
from core.security_system import SecurityManager
from core.persistence_system import PersistenceManager, PersistenceBackend
from interfaces.rest_api import FrameworkAPIServer

# Configure logging for the entire module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================================
# DEPLOYMENT MODELS
# ================================

class DeploymentEnvironment(Enum):
    """Deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class DeploymentStrategy(Enum):
    """Deployment strategies."""
    STANDALONE = "standalone"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    DOCKER_COMPOSE = "docker_compose"

@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    environment: DeploymentEnvironment
    strategy: DeploymentStrategy
    framework_config: Dict[str, Any]
    security_config: Dict[str, Any]
    persistence_config: Dict[str, Any]
    api_config: Dict[str, Any]
    agents_config: List[Dict[str, Any]]
    monitoring_config: Dict[str, Any]
    scaling_config: Dict[str, Any]
    
@dataclass
class DeploymentStatus:
    """Deployment status."""
    environment: str
    strategy: str
    status: str  # deploying, running, stopped, failed
    started_at: Optional[datetime]
    last_health_check: Optional[datetime]
    agents_count: int
    error_message: Optional[str] = None

# ================================
# DOCKER DEPLOYMENT
# ================================

class DockerDeployment:
    """Manages Docker deployments."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.container_name = f"agent-framework-{config.environment.value}"
        
    def generate_dockerfile(self) -> str:
        """Generates the Dockerfile content."""
        dockerfile = """
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/plugins

# Configure environment variables
ENV PYTHONPATH=/app
ENV FRAMEWORK_ENV={environment}
ENV LOG_LEVEL=INFO

# Expose ports
EXPOSE 8000 8080

# Default command
CMD ["python", "main.py"]
        """.format(environment=self.config.environment.value)
        
        return dockerfile.strip()
        
    def generate_requirements_txt(self) -> str:
        """Generates the requirements.txt file content."""
        requirements = """
aiohttp==3.9.1
aiohttp-cors==0.7.0
aiohttp-swagger==1.0.16
aiosqlite==0.19.0
pydantic==2.5.0
pyjwt==2.8.0
pyyaml==6.0.1
asyncio-mqtt==0.16.1
redis==5.0.1
psycopg2-binary==2.9.9
cryptography==41.0.8
prometheus-client==0.19.0
        """
        return requirements.strip()
        
    def generate_docker_compose(self) -> str:
        """Generates the docker-compose.yml content."""
        compose = {
            "version": "3.8",
            "services": {
                "agent-framework": {
                    "build": ".",
                    "container_name": self.container_name,
                    "ports": [
                        "8000:8000",  # API
                        "8080:8080"   # Dashboard
                    ],
                    "environment": {
                        "FRAMEWORK_ENV": self.config.environment.value,
                        "DATABASE_URL": "sqlite:///data/framework.db",
                        "JWT_SECRET": "${JWT_SECRET:-default_secret}",
                        "LOG_LEVEL": "INFO"
                    },
                    "volumes": [
                        "./data:/app/data",
                        "./logs:/app/logs",
                        "./plugins:/app/plugins"
                    ],
                    "restart": "unless-stopped",
                    "healthcheck": {
                        "test": ["CMD", "curl", "-f", "http://localhost:8000/api/health"],
                        "interval": "30s",
                        "timeout": "10s",
                        "retries": 3,
                        "start_period": "60s"
                    }
                },
                "redis": {
                    "image": "redis:7-alpine",
                    "container_name": f"{self.container_name}-redis",
                    "ports": ["6379:6379"],
                    "volumes": ["redis_data:/data"],
                    "restart": "unless-stopped"
                } if self.config.persistence_config.get("backend") == "redis" else None,
                "postgres": {
                    "image": "postgres:15-alpine",
                    "container_name": f"{self.container_name}-postgres",
                    "environment": {
                        "POSTGRES_DB": "agent_framework",
                        "POSTGRES_USER": "framework_user",
                        "POSTGRES_PASSWORD": "${POSTGRES_PASSWORD:-framework_pass}"
                    },
                    "ports": ["5432:5432"],
                    "volumes": ["postgres_data:/var/lib/postgresql/data"],
                    "restart": "unless-stopped"
                } if self.config.persistence_config.get("backend") == "postgresql" else None,
                "prometheus": {
                    "image": "prom/prometheus:latest",
                    "container_name": f"{self.container_name}-prometheus",
                    "ports": ["9090:9090"],
                    "volumes": [
                        "./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml"
                    ],
                    "restart": "unless-stopped"
                } if self.config.monitoring_config.get("prometheus") else None,
                "grafana": {
                    "image": "grafana/grafana:latest",
                    "container_name": f"{self.container_name}-grafana",
                    "ports": ["3000:3000"],
                    "environment": {
                        "GF_SECURITY_ADMIN_PASSWORD": "${GRAFANA_PASSWORD:-admin}"
                    },
                    "volumes": [
                        "grafana_data:/var/lib/grafana",
                        "./monitoring/grafana:/etc/grafana/provisioning"
                    ],
                    "restart": "unless-stopped"
                } if self.config.monitoring_config.get("grafana") else None
            },
            "volumes": {
                "redis_data": None,
                "postgres_data": None,
                "grafana_data": None
            },
            "networks": {
                "agent-network": {
                    "driver": "bridge"
                }
            }
        }
        
        # Clean up None services
        compose["services"] = {k: v for k, v in compose["services"].items() if v is not None}
        
        return yaml.dump(compose, default_flow_style=False)
        
    async def deploy(self, output_dir: str = "./deployment") -> bool:
        """Deploys the application using Docker."""
        try:
            deploy_path = Path(output_dir)
            deploy_path.mkdir(exist_ok=True)
            
            # Generate files
            with open(deploy_path / "Dockerfile", "w") as f:
                f.write(self.generate_dockerfile())
                
            with open(deploy_path / "requirements.txt", "w") as f:
                f.write(self.generate_requirements_txt())
                
            with open(deploy_path / "docker-compose.yml", "w") as f:
                f.write(self.generate_docker_compose())
                
            # Generate configuration file
            config_data = {
                "framework": self.config.framework_config,
                "security": self.config.security_config,
                "persistence": self.config.persistence_config,
                "api": self.config.api_config,
                "agents": self.config.agents_config
            }
            
            with open(deploy_path / "config.yaml", "w") as f:
                yaml.dump(config_data, f)
                
            # Generate deployment script
            deploy_script = self._generate_deploy_script()
            with open(deploy_path / "deploy.sh", "w") as f:
                f.write(deploy_script)
            os.chmod(deploy_path / "deploy.sh", 0o755)
            
            # Generate main.py
            main_py = self._generate_main_py()
            with open(deploy_path / "main.py", "w") as f:
                f.write(main_py)
                
            logger.info(f"Docker deployment files generated in {deploy_path}")
            return True
            
        except Exception as e:
            logger.error(f"Docker deployment failed: {e}")
            return False
            
    def _generate_deploy_script(self) -> str:
        """Generates the Docker deployment script."""
        script = f"""#!/bin/bash

# Deployment script for Agent Framework
# Environment: {self.config.environment.value}

set -e

echo "🚀 Deploying Agent Framework ({self.config.environment.value})"

# Verify Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is required but not installed"
    exit 1
fi

# Create necessary directories
mkdir -p data logs plugins monitoring/grafana monitoring/prometheus

# Generate Prometheus configuration if it doesn't exist
if [ ! -f "monitoring/prometheus.yml" ]; then
    cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'agent-framework'
    static_configs:
      - targets: ['agent-framework:8000']
    metrics_path: '/metrics'
EOF
fi

# Environment variables
export JWT_SECRET=${{JWT_SECRET:-$(openssl rand -hex 32)}}
export POSTGRES_PASSWORD=${{POSTGRES_PASSWORD:-framework_pass_$(openssl rand -hex 8)}}
export GRAFANA_PASSWORD=${{GRAFANA_PASSWORD:-admin}}

echo "🔧 Building and starting containers..."

# Build and start
docker-compose down --remove-orphans
docker-compose build
docker-compose up -d

echo "⏳ Waiting for services to be ready..."
sleep 30

# Health check
echo "🏥 Checking health..."
for i in {{1..10}}; do
    if curl -f http://localhost:8000/api/health > /dev/null 2>&1; then
        echo "✅ Framework is healthy"
        break
    fi
    echo "⏳ Waiting for framework to start (attempt $i/10)..."
    sleep 10
done

echo "📊 Service URLs:"
echo "   API: http://localhost:8000"
echo "   Dashboard: http://localhost:8080"
if docker-compose ps | grep prometheus > /dev/null; then
    echo "   Prometheus: http://localhost:9090"
fi
if docker-compose ps | grep grafana > /dev/null; then
    echo "   Grafana: http://localhost:3000 (admin:$GRAFANA_PASSWORD)"
fi

echo "🎉 Deployment completed!"
        """
        return script
        
    def _generate_main_py(self) -> str:
        """Generates the main.py file content for the container."""
        main_py = """
import asyncio
import logging
import os
import yaml
from pathlib import Path

# Import framework components
from autonomous_agent_framework import AgentFramework
from specialized_agents import ExtendedAgentFactory
from security_system import SecurityManager
from persistence_system import PersistenceFactory, PersistenceBackend
from rest_api import FrameworkAPIServer
from web_dashboard import DashboardServer

async def main():
    # Configure logging
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('/app/logs/framework.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Agent Framework...")
    
    # Load configuration
    config_path = Path("/app/config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "framework": {},
            "security": {"jwt_secret": os.getenv("JWT_SECRET", "default_secret")},
            "persistence": {"backend": "sqlite", "connection_string": "/app/data/framework.db"},
            "api": {"host": "0.0.0.0", "port": 8000},
            "agents": []
        }
    
    try:
        # Initialize framework
        framework = AgentFramework()
        await framework.start()
        logger.info("Framework started")
        
        # Configure security
        security_manager = SecurityManager(config.get("security", {}))
        logger.info("Security manager initialized")
        
        # Configure persistence
        persistence_config = config.get("persistence", {})
        if persistence_config.get("backend") == "sqlite":
            persistence_manager = PersistenceFactory.create_persistence_manager(
                backend=PersistenceBackend.SQLITE,
                connection_string=persistence_config.get("connection_string", "/app/data/framework.db")
            )
            await persistence_manager.initialize()
            logger.info("Persistence initialized")
        
        # Create configured agents
        agents_config = config.get("agents", [])
        for agent_config in agents_config:
            try:
                agent = ExtendedAgentFactory.create_agent(
                    agent_config["namespace"],
                    agent_config["name"],
                    framework
                )
                if agent_config.get("auto_start", True):
                    await agent.start()
                logger.info(f"Created agent: {agent_config['name']}")
            except Exception as e:
                logger.error(f"Failed to create agent {agent_config.get('name')}: {e}")
        
        # Start API server
        api_config = config.get("api", {})
        api_server = FrameworkAPIServer(
            framework,
            host=api_config.get("host", "0.0.0.0"),
            port=api_config.get("port", 8000)
        )
        api_runner = await api_server.start()
        logger.info(f"API server started on {api_config.get('host', '0.0.0.0')}:{api_config.get('port', 8000)}")
        
        # Start dashboard
        dashboard_server = DashboardServer(
            framework,
            host="0.0.0.0",
            port=8080
        )
        dashboard_runner = await dashboard_server.start()
        logger.info("Dashboard started on 0.0.0.0:8080")
        
        # Keep running
        logger.info("All services started successfully")
        while True:
            await asyncio.sleep(60)
            # Internal health check
            agents = framework.registry.list_all_agents()
            logger.debug(f"Health check: {len(agents)} agents active")
            
    except KeyboardInterrupt:
        logger.info("Shutdown signal received")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        logger.info("Shutting down...")
        await framework.stop()
        if 'api_runner' in locals():
            await api_runner.cleanup()
        if 'dashboard_runner' in locals():
            await dashboard_runner.cleanup()
        logger.info("Shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
        """
        return main_py.strip()

# ================================
# KUBERNETES DEPLOYMENT
# ================================

class KubernetesDeployment:
    """Manages Kubernetes deployments."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.namespace = f"agent-framework-{config.environment.value}"
        
    def generate_namespace(self) -> Dict[str, Any]:
        """Generates the Kubernetes namespace manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": self.namespace,
                "labels": {
                    "app": "agent-framework",
                    "environment": self.config.environment.value
                }
            }
        }
        
    def generate_configmap(self) -> Dict[str, Any]:
        """Generates the ConfigMap manifest."""
        config_data = {
            "framework": self.config.framework_config,
            "security": self.config.security_config,
            "persistence": self.config.persistence_config,
            "api": self.config.api_config,
            "agents": self.config.agents_config
        }
        
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "agent-framework-config",
                "namespace": self.namespace
            },
            "data": {
                "config.yaml": yaml.dump(config_data)
            }
        }
        
    def generate_secret(self) -> Dict[str, Any]:
        """Generates the Secret manifest."""
        import base64
        
        # In production, these values should be configured externally
        jwt_secret = self.config.security_config.get("jwt_secret", "change_in_production")
        db_password = "framework_db_password"
        
        return {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": "agent-framework-secrets",
                "namespace": self.namespace
            },
            "type": "Opaque",
            "data": {
                "jwt-secret": base64.b64encode(jwt_secret.encode()).decode(),
                "db-password": base64.b64encode(db_password.encode()).decode()
            }
        }
        
    def generate_deployment(self) -> Dict[str, Any]:
        """Generates the Deployment manifest."""
        replicas = self.config.scaling_config.get("replicas", 1)
        
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "agent-framework",
                "namespace": self.namespace,
                "labels": {
                    "app": "agent-framework",
                    "environment": self.config.environment.value
                }
            },
            "spec": {
                "replicas": replicas,
                "selector": {
                    "matchLabels": {
                        "app": "agent-framework"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "agent-framework"
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": "agent-framework",
                                "image": f"agent-framework:{self.config.environment.value}",
                                "ports": [
                                    {"containerPort": 8000, "name": "api"},
                                    {"containerPort": 8080, "name": "dashboard"}
                                ],
                                "env": [
                                    {
                                        "name": "FRAMEWORK_ENV",
                                        "value": self.config.environment.value
                                    },
                                    {
                                        "name": "JWT_SECRET",
                                        "valueFrom": {
                                            "secretKeyRef": {
                                                "name": "agent-framework-secrets",
                                                "key": "jwt-secret"
                                            }
                                        }
                                    }
                                ],
                                "volumeMounts": [
                                    {
                                        "name": "config",
                                        "mountPath": "/app/config.yaml",
                                        "subPath": "config.yaml"
                                    },
                                    {
                                        "name": "data",
                                        "mountPath": "/app/data"
                                    }
                                ],
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": "/api/health",
                                        "port": 8000
                                    },
                                    "initialDelaySeconds": 60,
                                    "periodSeconds": 30
                                },
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": "/api/health",
                                        "port": 8000
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10
                                },
                                "resources": {
                                    "requests": {
                                        "memory": "512Mi",
                                        "cpu": "250m"
                                    },
                                    "limits": {
                                        "memory": "1Gi",
                                        "cpu": "500m"
                                    }
                                }
                            }
                        ],
                        "volumes": [
                            {
                                "name": "config",
                                "configMap": {
                                    "name": "agent-framework-config"
                                }
                            },
                            {
                                "name": "data",
                                "persistentVolumeClaim": {
                                    "claimName": "agent-framework-data"
                                }
                            }
                        ]
                    }
                }
            }
        }
        
    def generate_service(self) -> Dict[str, Any]:
        """Generates the Service manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "agent-framework-service",
                "namespace": self.namespace
            },
            "spec": {
                "selector": {
                    "app": "agent-framework"
                },
                "ports": [
                    {
                        "name": "api",
                        "port": 8000,
                        "targetPort": 8000
                    },
                    {
                        "name": "dashboard",
                        "port": 8080,
                        "targetPort": 8080
                    }
                ],
                "type": "ClusterIP"
            }
        }
        
    def generate_ingress(self) -> Dict[str, Any]:
        """Generates the Ingress manifest."""
        domain = self.config.framework_config.get("domain", "agent-framework.local")
        
        return {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": "agent-framework-ingress",
                "namespace": self.namespace,
                "annotations": {
                    "nginx.ingress.kubernetes.io/rewrite-target": "/",
                    "cert-manager.io/cluster-issuer": "letsencrypt-prod"
                }
            },
            "spec": {
                "tls": [
                    {
                        "hosts": [domain],
                        "secretName": "agent-framework-tls"
                    }
                ],
                "rules": [
                    {
                        "host": domain,
                        "http": {
                            "paths": [
                                {
                                    "path": "/api",
                                    "pathType": "Prefix",
                                    "backend": {
                                        "service": {
                                            "name": "agent-framework-service",
                                            "port": {"number": 8000}
                                        }
                                    }
                                },
                                {
                                    "path": "/",
                                    "pathType": "Prefix",
                                    "backend": {
                                        "service": {
                                            "name": "agent-framework-service",
                                            "port": {"number": 8080}
                                        }
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }
        
    def generate_pvc(self) -> Dict[str, Any]:
        """Generates the PersistentVolumeClaim manifest."""
        return {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": "agent-framework-data",
                "namespace": self.namespace
            },
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "resources": {
                    "requests": {
                        "storage": "10Gi"
                    }
                }
            }
        }
        
    async def deploy(self, output_dir: str = "./k8s-deployment") -> bool:
        """Deploys the application using Kubernetes."""
        try:
            deploy_path = Path(output_dir)
            deploy_path.mkdir(exist_ok=True)
            
            # Generate manifests
            manifests = [
                ("namespace.yaml", self.generate_namespace()),
                ("configmap.yaml", self.generate_configmap()),
                ("secret.yaml", self.generate_secret()),
                ("pvc.yaml", self.generate_pvc()),
                ("deployment.yaml", self.generate_deployment()),
                ("service.yaml", self.generate_service()),
                ("ingress.yaml", self.generate_ingress())
            ]
            
            for filename, manifest in manifests:
                with open(deploy_path / filename, "w") as f:
                    yaml.dump(manifest, f, default_flow_style=False)
                    
            # Generate deployment script
            deploy_script = self._generate_k8s_deploy_script()
            with open(deploy_path / "deploy.sh", "w") as f:
                f.write(deploy_script)
            os.chmod(deploy_path / "deploy.sh", 0o755)
            
            logger.info(f"Kubernetes manifests generated in {deploy_path}")
            return True
            
        except Exception as e:
            logger.error(f"Kubernetes deployment failed: {e}")
            return False
            
    def _generate_k8s_deploy_script(self) -> str:
        """Generates the Kubernetes deployment script."""
        script = f"""#!/bin/bash

# Kubernetes deployment script for Agent Framework
# Environment: {self.config.environment.value}

set -e

echo "🚀 Deploying Agent Framework to Kubernetes ({self.config.environment.value})"

# Verify kubectl
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is required but not installed"
    exit 1
fi

# Verify cluster connection
if ! kubectl cluster-info > /dev/null 2>&1; then
    echo "❌ Not connected to a Kubernetes cluster"
    exit 1
fi

echo "🔧 Applying Kubernetes manifests..."

# Apply manifests in order
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml
kubectl apply -f pvc.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml

echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/agent-framework -n {self.namespace}

echo "✅ Deployment completed!"

# Show information
echo "📊 Service information:"
kubectl get pods,svc,ingress -n {self.namespace}

echo "🔍 To check logs:"
echo "kubectl logs -f deployment/agent-framework -n {self.namespace}"

echo "🌐 To access the service:"
echo "kubectl port-forward service/agent-framework-service 8000:8000 -n {self.namespace}"
        """
        return script

# ================================
# DEPLOYMENT ORCHESTRATOR
# ================================

class DeploymentOrchestrator:
    """Main deployment orchestrator."""
    
    def __init__(self):
        self.deployments: Dict[str, Any] = {}
        
    def create_deployment_config(
        self,
        environment: DeploymentEnvironment,
        strategy: DeploymentStrategy,
        **kwargs
    ) -> DeploymentConfig:
        """Creates a deployment configuration."""
        
        # Default configuration based on environment
        if environment == DeploymentEnvironment.DEVELOPMENT:
            framework_config = {
                "log_level": "DEBUG",
                "auto_save_interval": 30
            }
            security_config = {
                "jwt_secret": "dev_secret",
                "session_max_hours": 8
            }
            persistence_config = {
                "backend": "json",
                "connection_string": "./data"
            }
            scaling_config = {
                "replicas": 1,
                "auto_scaling": False
            }
        elif environment == DeploymentEnvironment.STAGING:
            framework_config = {
                "log_level": "INFO",
                "auto_save_interval": 60
            }
            security_config = {
                "jwt_secret": kwargs.get("jwt_secret", "staging_secret"),
                "session_max_hours": 12
            }
            persistence_config = {
                "backend": "sqlite",
                "connection_string": "./data/framework.db"
            }
            scaling_config = {
                "replicas": 2,
                "auto_scaling": False
            }
        else:  # PRODUCTION
            framework_config = {
                "log_level": "WARNING",
                "auto_save_interval": 300,
                "domain": kwargs.get("domain", "agent-framework.com")
            }
            security_config = {
                "jwt_secret": kwargs.get("jwt_secret", "CHANGE_IN_PRODUCTION"),
                "session_max_hours": 24,
                "enable_agent_authentication": True
            }
            persistence_config = {
                "backend": "postgresql",
                "connection_string": kwargs.get("db_url", "postgresql://user:pass@localhost/framework")
            }
            scaling_config = {
                "replicas": 3,"auto_scaling": True,
                "min_replicas": 3,
                "max_replicas": 10
            }

        # Override with any provided kwargs
        framework_config.update(kwargs.get("framework_config", {}))
        security_config.update(kwargs.get("security_config", {}))
        persistence_config.update(kwargs.get("persistence_config", {}))
        api_config = kwargs.get("api_config", {"host": "0.0.0.0", "port": 8000})
        agents_config = kwargs.get("agents_config", [])
        monitoring_config = kwargs.get("monitoring_config", {"prometheus": True, "grafana": True})
        scaling_config.update(kwargs.get("scaling_config", {}))

        return DeploymentConfig(
            environment=environment,
            strategy=strategy,
            framework_config=framework_config,
            security_config=security_config,
            persistence_config=persistence_config,
            api_config=api_config,
            agents_config=agents_config,
            monitoring_config=monitoring_config,
            scaling_config=scaling_config
        )

    async def deploy(self, config: DeploymentConfig, output_dir: str = "./deployments") -> DeploymentStatus:
        """Initiates a deployment based on the provided configuration."""
        deployment_key = f"{config.environment.value}-{config.strategy.value}"
        logger.info(f"Initiating deployment for {deployment_key}...")

        status = DeploymentStatus(
            environment=config.environment.value,
            strategy=config.strategy.value,
            status="deploying",
            started_at=datetime.now(),
            last_health_check=None,
            agents_count=len(config.agents_config)
        )

        try:
            if config.strategy == DeploymentStrategy.DOCKER_COMPOSE:
                docker_deployer = DockerDeployment(config)
                success = await docker_deployer.deploy(output_dir=os.path.join(output_dir, deployment_key))
            elif config.strategy == DeploymentStrategy.KUBERNETES:
                k8s_deployer = KubernetesDeployment(config)
                success = await k8s_deployer.deploy(output_dir=os.path.join(output_dir, deployment_key))
            else:
                logger.error(f"Unsupported deployment strategy: {config.strategy.value}")
                status.status = "failed"
                status.error_message = f"Unsupported strategy: {config.strategy.value}"
                self.deployments[deployment_key] = status
                return status

            if success:
                status.status = "running"
                logger.info(f"Deployment {deployment_key} completed successfully.")
            else:
                status.status = "failed"
                status.error_message = "Deployment script generation or execution failed."
                logger.error(f"Deployment {deployment_key} failed.")

        except Exception as e:
            logger.error(f"An unexpected error occurred during deployment {deployment_key}: {e}", exc_info=True)
            status.status = "failed"
            status.error_message = str(e)

        self.deployments[deployment_key] = status
        return status

    async def get_deployment_status(self, environment: DeploymentEnvironment, strategy: DeploymentStrategy) -> Optional[DeploymentStatus]:
        """Retrieves the status of a specific deployment."""
        deployment_key = f"{environment.value}-{strategy.value}"
        status = self.deployments.get(deployment_key)
        if status:
            logger.info(f"Retrieving status for {deployment_key}: {status.status}")
        else:
            logger.warning(f"No deployment found for {deployment_key}")
        return status

    async def undeploy(self, environment: DeploymentEnvironment, strategy: DeploymentStrategy, deployment_dir: str = "./deployments") -> bool:
        """Undeploys a specific deployment."""
        deployment_key = f"{environment.value}-{strategy.value}"
        logger.info(f"Initiating undeployment for {deployment_key}...")

        deploy_path = Path(deployment_dir) / deployment_key
        if not deploy_path.exists():
            logger.warning(f"Deployment directory not found for {deployment_key}. Nothing to undeploy.")
            return False

        try:
            if strategy == DeploymentStrategy.DOCKER_COMPOSE:
                command = ["docker-compose", "down", "--remove-orphans"]
                cwd = deploy_path
                log_message = "Docker Compose services stopped."
            elif strategy == DeploymentStrategy.KUBERNETES:
                command = ["kubectl", "delete", "-f", ".", "-n", f"agent-framework-{environment.value}"]
                cwd = deploy_path
                log_message = "Kubernetes resources deleted."
            else:
                logger.error(f"Undeployment not supported for strategy: {strategy.value}")
                return False

            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                logger.info(f"{log_message} for {deployment_key}")
                shutil.rmtree(deploy_path)
                logger.info(f"Cleaned up deployment directory: {deploy_path}")
                if deployment_key in self.deployments:
                    self.deployments[deployment_key].status = "stopped"
                    self.deployments[deployment_key].error_message = None
                return True
            else:
                error_output = stderr.decode().strip()
                logger.error(f"Undeployment of {deployment_key} failed: {error_output}")
                if deployment_key in self.deployments:
                    self.deployments[deployment_key].status = "failed"
                    self.deployments[deployment_key].error_message = f"Undeployment command failed: {error_output}"
                return False

        except FileNotFoundError:
            logger.error(f"Deployment tool not found for {strategy.value}. Make sure it's installed and in your PATH.")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during undeployment {deployment_key}: {e}", exc_info=True)
            if deployment_key in self.deployments:
                self.deployments[deployment_key].status = "failed"
                self.deployments[deployment_key].error_message = str(e)
            return False

# Example usage (for demonstration purposes)
async def main():
    orchestrator = DeploymentOrchestrator()

    # Create a Docker Compose deployment configuration for development
    dev_docker_config = orchestrator.create_deployment_config(
        environment=DeploymentEnvironment.DEVELOPMENT,
        strategy=DeploymentStrategy.DOCKER_COMPOSE,
        api_config={"host": "0.0.0.0", "port": 8001}, # Custom API port for development
        agents_config=[
            {"namespace": "default", "name": "data_processor_dev", "auto_start": True},
            {"namespace": "default", "name": "logger_agent_dev", "auto_start": False}
        ]
    )
    logger.info(f"Generated Dev Docker Config: {asdict(dev_docker_config)}")

    # Deploy the development environment with Docker Compose
    dev_docker_status = await orchestrator.deploy(dev_docker_config)
    logger.info(f"Dev Docker Deployment Status: {dev_docker_status.status}")

    # Create a Kubernetes deployment configuration for staging
    staging_k8s_config = orchestrator.create_deployment_config(
        environment=DeploymentEnvironment.STAGING,
        strategy=DeploymentStrategy.KUBERNETES,
        jwt_secret="super_secret_staging_key_123",
        persistence_config={"backend": "postgresql", "connection_string": "postgresql://staging_user:staging_pass@db-staging/framework"},
        scaling_config={"replicas": 3, "auto_scaling": True, "min_replicas": 2, "max_replicas": 5}
    )
    logger.info(f"Generated Staging K8s Config: {asdict(staging_k8s_config)}")

    # Deploy the staging environment to Kubernetes
    staging_k8s_status = await orchestrator.deploy(staging_k8s_config)
    logger.info(f"Staging K8s Deployment Status: {staging_k8s_status.status}")

    # Get status of the development deployment
    current_dev_status = await orchestrator.get_deployment_status(DeploymentEnvironment.DEVELOPMENT, DeploymentStrategy.DOCKER_COMPOSE)
    if current_dev_status:
        logger.info(f"Current Dev Docker Status: {current_dev_status.status}, Agents: {current_dev_status.agents_count}")

    # Simulate waiting for a bit
    await asyncio.sleep(5)

    # Undeploy the development environment
    undeploy_success = await orchestrator.undeploy(DeploymentEnvironment.DEVELOPMENT, DeploymentStrategy.DOCKER_COMPOSE)
    logger.info(f"Dev Docker Undeployment Success: {undeploy_success}")

    # Check status after undeployment
    current_dev_status_after_undeploy = await orchestrator.get_deployment_status(DeploymentEnvironment.DEVELOPMENT, DeploymentStrategy.DOCKER_COMPOSE)
    if current_dev_status_after_undeploy:
        logger.info(f"Dev Docker Status After Undeploy: {current_dev_status_after_undeploy.status}")

if __name__ == "__main__":
    asyncio.run(main())