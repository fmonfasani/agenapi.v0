"""
deployment_system.py - Sistema de deployment y orquestación para producción
"""

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

# ================================
# DEPLOYMENT MODELS
# ================================

class DeploymentEnvironment(Enum):
    """Entornos de deployment"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class DeploymentStrategy(Enum):
    """Estrategias de deployment"""
    STANDALONE = "standalone"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    DOCKER_COMPOSE = "docker_compose"

@dataclass
class DeploymentConfig:
    """Configuración de deployment"""
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
    """Estado del deployment"""
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
    """Gestión de deployment con Docker"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.container_name = f"agent-framework-{config.environment.value}"
        
    def generate_dockerfile(self) -> str:
        """Generar Dockerfile"""
        dockerfile = """
FROM python:3.11-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \\
    curl \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY . .

# Crear directorios necesarios
RUN mkdir -p /app/logs /app/data /app/plugins

# Configurar variables de entorno
ENV PYTHONPATH=/app
ENV FRAMEWORK_ENV={environment}
ENV LOG_LEVEL=INFO

# Exponer puertos
EXPOSE 8000 8080

# Comando por defecto
CMD ["python", "main.py"]
        """.format(environment=self.config.environment.value)
        
        return dockerfile.strip()
        
    def generate_requirements_txt(self) -> str:
        """Generar archivo requirements.txt"""
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
        """Generar docker-compose.yml"""
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
        
        # Limpiar servicios None
        compose["services"] = {k: v for k, v in compose["services"].items() if v is not None}
        
        return yaml.dump(compose, default_flow_style=False)
        
    async def deploy(self, output_dir: str = "./deployment") -> bool:
        """Realizar deployment con Docker"""
        try:
            deploy_path = Path(output_dir)
            deploy_path.mkdir(exist_ok=True)
            
            # Generar archivos
            with open(deploy_path / "Dockerfile", "w") as f:
                f.write(self.generate_dockerfile())
                
            with open(deploy_path / "requirements.txt", "w") as f:
                f.write(self.generate_requirements_txt())
                
            with open(deploy_path / "docker-compose.yml", "w") as f:
                f.write(self.generate_docker_compose())
                
            # Generar archivo de configuración
            config_data = {
                "framework": self.config.framework_config,
                "security": self.config.security_config,
                "persistence": self.config.persistence_config,
                "api": self.config.api_config,
                "agents": self.config.agents_config
            }
            
            with open(deploy_path / "config.yaml", "w") as f:
                yaml.dump(config_data, f)
                
            # Generar script de deployment
            deploy_script = self._generate_deploy_script()
            with open(deploy_path / "deploy.sh", "w") as f:
                f.write(deploy_script)
            os.chmod(deploy_path / "deploy.sh", 0o755)
            
            # Generar main.py
            main_py = self._generate_main_py()
            with open(deploy_path / "main.py", "w") as f:
                f.write(main_py)
                
            logging.info(f"Docker deployment files generated in {deploy_path}")
            return True
            
        except Exception as e:
            logging.error(f"Docker deployment failed: {e}")
            return False
            
    def _generate_deploy_script(self) -> str:
        """Generar script de deployment"""
        script = f"""#!/bin/bash

# Deployment script for Agent Framework
# Environment: {self.config.environment.value}

set -e

echo "🚀 Deploying Agent Framework ({self.config.environment.value})"

# Verificar Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is required but not installed"
    exit 1
fi

# Crear directorios necesarios
mkdir -p data logs plugins monitoring/grafana monitoring/prometheus

# Generar configuración de Prometheus si no existe
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

# Variables de entorno
export JWT_SECRET=${{JWT_SECRET:-$(openssl rand -hex 32)}}
export POSTGRES_PASSWORD=${{POSTGRES_PASSWORD:-framework_pass_$(openssl rand -hex 8)}}
export GRAFANA_PASSWORD=${{GRAFANA_PASSWORD:-admin}}

echo "🔧 Building and starting containers..."

# Build y start
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
echo "  API: http://localhost:8000"
echo "  Dashboard: http://localhost:8080"
if docker-compose ps | grep prometheus > /dev/null; then
    echo "  Prometheus: http://localhost:9090"
fi
if docker-compose ps | grep grafana > /dev/null; then
    echo "  Grafana: http://localhost:3000 (admin:$GRAFANA_PASSWORD)"
fi

echo "🎉 Deployment completed!"
        """
        return script
        
    def _generate_main_py(self) -> str:
        """Generar archivo main.py para el contenedor"""
        main_py = """
import asyncio
import logging
import os
import yaml
from pathlib import Path

# Importar componentes del framework
from autonomous_agent_framework import AgentFramework
from specialized_agents import ExtendedAgentFactory
from security_system import SecurityManager
from persistence_system import PersistenceFactory, PersistenceBackend
from rest_api import FrameworkAPIServer
from web_dashboard import DashboardServer

async def main():
    # Configurar logging
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
    
    # Cargar configuración
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
        # Inicializar framework
        framework = AgentFramework()
        await framework.start()
        logger.info("Framework started")
        
        # Configurar seguridad
        security_manager = SecurityManager(config.get("security", {}))
        logger.info("Security manager initialized")
        
        # Configurar persistencia
        persistence_config = config.get("persistence", {})
        if persistence_config.get("backend") == "sqlite":
            persistence_manager = PersistenceFactory.create_persistence_manager(
                backend=PersistenceBackend.SQLITE,
                connection_string=persistence_config.get("connection_string", "/app/data/framework.db")
            )
            await persistence_manager.initialize()
            logger.info("Persistence initialized")
        
        # Crear agentes configurados
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
        
        # Iniciar API server
        api_config = config.get("api", {})
        api_server = FrameworkAPIServer(
            framework,
            host=api_config.get("host", "0.0.0.0"),
            port=api_config.get("port", 8000)
        )
        api_runner = await api_server.start()
        logger.info(f"API server started on {api_config.get('host', '0.0.0.0')}:{api_config.get('port', 8000)}")
        
        # Iniciar dashboard
        dashboard_server = DashboardServer(
            framework,
            host="0.0.0.0",
            port=8080
        )
        dashboard_runner = await dashboard_server.start()
        logger.info("Dashboard started on 0.0.0.0:8080")
        
        # Mantener corriendo
        logger.info("All services started successfully")
        while True:
            await asyncio.sleep(60)
            # Health check interno
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
    """Gestión de deployment con Kubernetes"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.namespace = f"agent-framework-{config.environment.value}"
        
    def generate_namespace(self) -> Dict[str, Any]:
        """Generar namespace de Kubernetes"""
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
        """Generar ConfigMap"""
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
        """Generar Secret"""
        import base64
        
        # En producción, estos valores deberían ser configurados externamente
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
        """Generar Deployment"""
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
        """Generar Service"""
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
        """Generar Ingress"""
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
        """Generar PersistentVolumeClaim"""
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
        """Realizar deployment con Kubernetes"""
        try:
            deploy_path = Path(output_dir)
            deploy_path.mkdir(exist_ok=True)
            
            # Generar manifiestos
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
                    
            # Generar script de deployment
            deploy_script = self._generate_k8s_deploy_script()
            with open(deploy_path / "deploy.sh", "w") as f:
                f.write(deploy_script)
            os.chmod(deploy_path / "deploy.sh", 0o755)
            
            logging.info(f"Kubernetes manifests generated in {deploy_path}")
            return True
            
        except Exception as e:
            logging.error(f"Kubernetes deployment failed: {e}")
            return False
            
    def _generate_k8s_deploy_script(self) -> str:
        """Generar script de deployment para Kubernetes"""
        script = f"""#!/bin/bash

# Kubernetes deployment script for Agent Framework
# Environment: {self.config.environment.value}

set -e

echo "🚀 Deploying Agent Framework to Kubernetes ({self.config.environment.value})"

# Verificar kubectl
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is required but not installed"
    exit 1
fi

# Verificar conexión al cluster
if ! kubectl cluster-info > /dev/null 2>&1; then
    echo "❌ Not connected to a Kubernetes cluster"
    exit 1
fi

echo "🔧 Applying Kubernetes manifests..."

# Aplicar manifiestos en orden
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

# Mostrar información
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
    """Orquestador principal de deployments"""
    
    def __init__(self):
        self.deployments: Dict[str, Any] = {}
        
    def create_deployment_config(
        self,
        environment: DeploymentEnvironment,
        strategy: DeploymentStrategy,
        **kwargs
    ) -> DeploymentConfig:
        """Crear configuración de deployment"""
        
        # Configuración por defecto basada en el entorno
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
                "replicas": 3,
                "auto_scaling": True,
                "min_replicas": 2,
                "max_replicas": 10,
                "target_cpu": 70
            }
            
        # Configuración de API
        api_config = {
            "host": "0.0.0.0",
            "port": 8000,
            "enable_cors": True
        }
        
        # Agentes por defecto
        agents_config = [
            {"namespace": "agent.planning.strategist", "name": "strategist", "auto_start": True},
            {"namespace": "agent.build.code.generator", "name": "generator", "auto_start": True},
            {"namespace": "agent.test.generator", "name": "tester", "auto_start": True},
            {"namespace": "agent.security.sentinel", "name": "sentinel", "auto_start": True}
        ]
        
        # Configuración de monitoreo
        monitoring_config = {
            "prometheus": environment != DeploymentEnvironment.DEVELOPMENT,
            "grafana": environment == DeploymentEnvironment.PRODUCTION,
            "health_check_interval": 30
        }
        
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
        
    async def deploy(self, config: DeploymentConfig, output_dir: str = None) -> bool:
        """Ejecutar deployment"""
        deployment_id = f"{config.environment.value}_{config.strategy.value}"
        
        if output_dir is None:
            output_dir = f"./deployment_{deployment_id}"
            
        try:
            # Crear deployer apropiado
            if config.strategy == DeploymentStrategy.DOCKER:
                deployer = DockerDeployment(config)
            elif config.strategy == DeploymentStrategy.DOCKER_COMPOSE:
                deployer = DockerDeployment(config)  # Mismo que Docker
            elif config.strategy == DeploymentStrategy.KUBERNETES:
                deployer = KubernetesDeployment(config)
            else:
                raise ValueError(f"Unsupported deployment strategy: {config.strategy}")
                
            # Ejecutar deployment
            success = await deployer.deploy(output_dir)
            
            if success:
                self.deployments[deployment_id] = {
                    "config": config,
                    "status": "deployed",
                    "deployed_at": datetime.now(),
                    "output_dir": output_dir
                }
                
            return success
            
        except Exception as e:
            logging.error(f"Deployment failed: {e}")
            return False
            
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Obtener estado de deployment"""
        return self.deployments.get(deployment_id)
        
    def list_deployments(self) -> List[Dict[str, Any]]:
        """Listar todos los deployments"""
        return list(self.deployments.values())

# ================================
# EXAMPLE USAGE
# ================================

async def deployment_demo():
    """Demo del sistema de deployment"""
    
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Deployment System Demo")
    print("="*60)
    
    orchestrator = DeploymentOrchestrator()
    
    # Demo 1: Development deployment con Docker
    print("\n1. Creating Development Deployment (Docker)...")
    dev_config = orchestrator.create_deployment_config(
        DeploymentEnvironment.DEVELOPMENT,
        DeploymentStrategy.DOCKER
    )
    
    success = await orchestrator.deploy(dev_config, "./deployment_dev")
    if success:
        print("✅ Development deployment files created")
        print("   Location: ./deployment_dev")
        print("   Run: cd deployment_dev && ./deploy.sh")
    
    # Demo 2: Production deployment con Kubernetes
    print("\n2. Creating Production Deployment (Kubernetes)...")
    prod_config = orchestrator.create_deployment_config(
        DeploymentEnvironment.PRODUCTION,
        DeploymentStrategy.KUBERNETES,
        domain="my-agent-framework.com",
        jwt_secret="super_secure_production_secret",
        db_url="postgresql://prod_user:prod_pass@postgres.cluster/framework"
    )
    
    success = await orchestrator.deploy(prod_config, "./deployment_prod")
    if success:
        print("✅ Production deployment manifests created")
        print("   Location: ./deployment_prod")
        print("   Run: cd deployment_prod && ./deploy.sh")
    
    # Demo 3: Listar deployments
    print("\n3. Deployment Status:")
    deployments = orchestrator.list_deployments()
    for i, deployment in enumerate(deployments, 1):
        config = deployment["config"]
        print(f"   {i}. {config.environment.value} ({config.strategy.value})")
        print(f"      Status: {deployment['status']}")
        print(f"      Created: {deployment['deployed_at'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"      Output: {deployment['output_dir']}")
    
    print("\n✅ Deployment demo completed")
    print("\n📋 Next steps:")
    print("   1. Review generated deployment files")
    print("   2. Customize configuration as needed")
    print("   3. Run deployment scripts")
    print("   4. Monitor deployment status")

if __name__ == "__main__":
    asyncio.run(deployment_demo())