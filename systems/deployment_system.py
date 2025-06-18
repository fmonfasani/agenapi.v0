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

# Importaciones actualizadas
from core.autonomous_agent_framework import AgentFramework # <-- CAMBIO AQUI
# Asumimos que SecurityManager y PersistenceManager están en core/
from core.security_system import SecurityManager # <-- CAMBIO AQUI (si se mueve)
from core.persistence_system import PersistenceManager, PersistenceBackend # <-- CAMBIO AQUI (si se mueve)
from interfaces.rest_api import FrameworkAPIServer # Se mantiene así, o se cambia si la estructura de interfaces cambia


# ================================\
# DEPLOYMENT MODELS
# ================================\

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
    """Estado de un deployment"""
    deployment_id: str
    config: DeploymentConfig
    status: str # "pending", "running", "completed", "failed"
    deployed_at: datetime
    last_updated: datetime
    output_dir: str # Directorio donde se generaron los artefactos
    errors: Optional[str] = None

# ================================\
# DEPLOYMENT ENGINE (Generates manifests, executes commands)
# ================================\

class DeploymentEngine:
    """
    Genera artefactos de deployment (Dockerfiles, Kubernetes YAMLs, etc.)
    y ejecuta comandos de deployment.
    """
    def __init__(self, base_output_dir: str = "./deployments"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("DeploymentEngine")
        self.logger.info(f"DeploymentEngine initialized. Output directory: {self.base_output_dir}")

    async def generate_manifests(self, config: DeploymentConfig, output_dir: Path) -> bool:
        """Genera archivos de configuración (YAML, Dockerfile) para el deployment."""
        self.logger.info(f"Generating manifests for {config.environment.value} with {config.strategy.value} to {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            if config.strategy == DeploymentStrategy.DOCKER:
                await self._generate_docker_manifests(config, output_dir)
            elif config.strategy == DeploymentStrategy.KUBERNETES:
                await self._generate_kubernetes_manifests(config, output_dir)
            elif config.strategy == DeploymentStrategy.DOCKER_COMPOSE:
                await self._generate_docker_compose_manifests(config, output_dir)
            elif config.strategy == DeploymentStrategy.STANDALONE:
                await self._generate_standalone_scripts(config, output_dir)
            else:
                raise ValueError(f"Unsupported deployment strategy: {config.strategy.value}")
            
            self.logger.info("Manifests generated successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to generate manifests: {e}")
            return False

    async def _generate_docker_manifests(self, config: DeploymentConfig, output_dir: Path):
        """Genera Dockerfile y scripts de construcción/ejecución."""
        dockerfile_content = f"""
FROM python:3.9-slim-buster
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "core/autonomous_agent_framework.py"] # Asume que el main está aquí
"""
        (output_dir / "Dockerfile").write_text(dockerfile_content)

        # Crear un script de construcción
        build_script_content = f"""
#!/bin/bash
echo "Building Docker image for {config.environment.value}..."
docker build -t agent-framework-{config.environment.value} .
echo "Docker image built: agent-framework-{config.environment.value}"
"""
        build_script_path = output_dir / "build.sh"
        build_script_path.write_text(build_script_content)
        os.chmod(build_script_path, 0o755)

        # Crear un script de ejecución
        run_script_content = f"""
#!/bin/bash
echo "Running Docker container for {config.environment.value}..."
docker run -d --name agent-framework-{config.environment.value} -p 8000:8000 -p 8080:8080 agent-framework-{config.environment.value}
echo "Container agent-framework-{config.environment.value} started."
"""
        run_script_path = output_dir / "run.sh"
        run_script_path.write_text(run_script_content)
        os.chmod(run_script_path, 0o755)

        # Escribir la configuración del framework
        with open(output_dir / "framework_config.yaml", "w") as f:
            yaml.dump(asdict(config), f)

    async def _generate_kubernetes_manifests(self, config: DeploymentConfig, output_dir: Path):
        """Genera archivos YAML para deployment en Kubernetes."""
        app_name = f"agent-framework-{config.environment.value}"
        
        # Deployment
        deployment_yaml = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": app_name},
            "spec": {
                "replicas": config.scaling_config.get("replicas", 1),
                "selector": {"matchLabels": {"app": app_name}},
                "template": {
                    "metadata": {"labels": {"app": app_name}},
                    "spec": {
                        "containers": [{
                            "name": "framework",
                            "image": f"agent-framework-{config.environment.value}:latest", # Asume que la imagen ya fue construida
                            "ports": [{"containerPort": 8000}, {"containerPort": 8080}],
                            "env": [
                                {"name": "JWT_SECRET", "value": config.security_config.get("jwt_secret", "")},
                                # Añadir otras variables de entorno desde config
                            ]
                        }]
                    }
                }
            }
        }
        (output_dir / "deployment.yaml").write_text(yaml.dump(deployment_yaml))

        # Service
        service_yaml = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": app_name},
            "spec": {
                "selector": {"app": app_name},
                "ports": [
                    {"protocol": "TCP", "port": 8000, "targetPort": 8000, "name": "api"},
                    {"protocol": "TCP", "port": 8080, "targetPort": 8080, "name": "dashboard"}
                ],
                "type": "LoadBalancer" if config.environment == DeploymentEnvironment.PRODUCTION else "ClusterIP"
            }
        }
        (output_dir / "service.yaml").write_text(yaml.dump(service_yaml))

        # Script de despliegue
        deploy_script_content = f"""
#!/bin/bash
echo "Deploying to Kubernetes for {config.environment.value}..."
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
echo "Kubernetes deployment for {app_name} applied."
"""
        deploy_script_path = output_dir / "deploy.sh"
        deploy_script_path.write_text(deploy_script_content)
        os.chmod(deploy_script_path, 0o755)

    async def _generate_docker_compose_manifests(self, config: DeploymentConfig, output_dir: Path):
        """Genera un archivo docker-compose.yaml."""
        compose_yaml = {
            "version": "3.8",
            "services": {
                "framework": {
                    "build": ".",
                    "ports": ["8000:8000", "8080:8080"],
                    "environment": {
                        "JWT_SECRET": config.security_config.get("jwt_secret", "")
                        # Añadir otras variables de entorno
                    },
                    "volumes": [
                        "./data:/app/data" # Persistencia de datos si la base de datos es SQLite local
                    ],
                    "restart": "always"
                }
            }
        }
        (output_dir / "docker-compose.yaml").write_text(yaml.dump(compose_yaml))

        # Crear un Dockerfile básico dentro del directorio de salida
        dockerfile_content = f"""
FROM python:3.9-slim-buster
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "core/autonomous_agent_framework.py"]
"""
        (output_dir / "Dockerfile").write_text(dockerfile_content)

        # Script para levantar
        up_script_content = f"""
#!/bin/bash
echo "Bringing up Docker Compose for {config.environment.value}..."
docker-compose -f docker-compose.yaml up -d
echo "Docker Compose services started."
"""
        up_script_path = output_dir / "up.sh"
        up_script_path.write_text(up_script_content)
        os.chmod(up_script_path, 0o755)

    async def _generate_standalone_scripts(self, config: DeploymentConfig, output_dir: Path):
        """Genera scripts para ejecución standalone."""
        # Un simple script de inicio
        start_script_content = f"""
#!/bin/bash
echo "Starting Agent Framework for {config.environment.value} in standalone mode..."
python core/autonomous_agent_framework.py
"""
        start_script_path = output_dir / "start.sh"
        start_script_path.write_text(start_script_content)
        os.chmod(start_script_path, 0o755)

        # Copiar requirements.txt y otros archivos esenciales si es necesario
        shutil.copy("requirements.txt", output_dir / "requirements.txt")
        # Asegurarse de copiar los módulos del core para que el script pueda ejecutarse
        shutil.copytree("core", output_dir / "core", dirs_exist_ok=True)
        shutil.copytree("agents", output_dir / "agents", dirs_exist_ok=True) # Copiar agentes si son necesarios para el runtime
        shutil.copytree("interfaces", output_dir / "interfaces", dirs_exist_ok=True) # Copiar interfaces

        # Escribir la configuración del framework
        with open(output_dir / "framework_config.yaml", "w") as f:
            yaml.dump(asdict(config), f)


# ================================\
# DEPLOYMENT ORCHESTRATOR
# ================================\

class DeploymentOrchestrator:
    """
    Orquesta el proceso de deployment, utilizando el DeploymentEngine.
    Mantiene un registro de los deployments.
    """
    def __init__(self, base_output_dir: str = "./deployments"):
        self.engine = DeploymentEngine(base_output_dir)
        self.deployments: Dict[str, DeploymentStatus] = {} # deployment_id -> DeploymentStatus
        self.logger = logging.getLogger("DeploymentOrchestrator")
        self.logger.info("DeploymentOrchestrator initialized.")

    async def create_deployment_config(self, environment: DeploymentEnvironment, strategy: DeploymentStrategy, **kwargs) -> DeploymentConfig:
        """Crea una configuración de deployment con parámetros por defecto y sobrescritos."""
        
        # Configuraciones por defecto (simplificadas)
        default_framework_config = {"log_level": "INFO"}
        default_security_config = {"jwt_secret": "default_dev_secret", "enable_rbac": False}
        default_persistence_config = {"backend": PersistenceBackend.SQLITE.value, "connection_string": f"framework_{environment.value}.db"}
        default_api_config = {"port": 8000}
        default_agents_config = [] # Se pueden especificar agentes a desplegar
        default_monitoring_config = {"enable_monitoring": True}
        default_scaling_config = {"replicas": 1}

        # Sobrescribir con kwargs específicos del entorno o pasados
        framework_config = kwargs.pop("framework_config", {}).copy()
        framework_config.update(default_framework_config)

        security_config = kwargs.pop("security_config", {}).copy()
        security_config.update(default_security_config)
        if environment == DeploymentEnvironment.PRODUCTION:
            security_config["jwt_secret"] = kwargs.get("jwt_secret", "PROD_SECRET_REQUIRED") # Asegurar secreto en prod
            security_config["enable_rbac"] = True

        persistence_config = kwargs.pop("persistence_config", {}).copy()
        persistence_config.update(default_persistence_config)
        
        api_config = kwargs.pop("api_config", {}).copy()
        api_config.update(default_api_config)

        agents_config = kwargs.pop("agents_config", []).copy()
        monitoring_config = kwargs.pop("monitoring_config", {}).copy()
        monitoring_config.update(default_monitoring_config)
        scaling_config = kwargs.pop("scaling_config", {}).copy()
        scaling_config.update(default_scaling_config)

        # Añadir cualquier otro setting pasado como kwargs que no sea una config de componente
        # Esto permite pasar parámetros directos para ser usados en la generación de manifiestos
        # Por ejemplo, 'domain' para ingress, 'db_url' para bases de datos externas
        framework_config.update({k: v for k, v in kwargs.items() if k not in ["jwt_secret", "db_url"]}) # Evitar reescribir secrets

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

    async def deploy(self, config: DeploymentConfig, output_dir: str) -> bool:
        """Inicia un proceso de deployment."""
        deployment_id = f"deploy_{config.environment.value}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        output_path = Path(output_dir)
        
        status = DeploymentStatus(
            deployment_id=deployment_id,
            config=config,
            status="pending",
            deployed_at=datetime.now(),
            last_updated=datetime.now(),
            output_dir=str(output_path)
        )
        self.deployments[deployment_id] = status
        self.logger.info(f"Starting deployment {deployment_id} to {config.environment.value} with strategy {config.strategy.value}.")

        try:
            # Generar manifiestos
            manifest_success = await self.engine.generate_manifests(config, output_path)
            if not manifest_success:
                status.status = "failed"
                status.errors = "Manifest generation failed."
                self.logger.error(f"Deployment {deployment_id} failed: Manifest generation failed.")
                return False

            status.status = "manifests_generated"
            self.logger.info(f"Manifests for deployment {deployment_id} generated.")

            # Ejecutar comandos de deployment (simulado o real con subprocess)
            if config.strategy == DeploymentStrategy.DOCKER:
                self.logger.info("Simulating Docker build and run...")
                # En un entorno real, ejecutar los scripts generados (build.sh, run.sh)
                # await self._execute_shell_command(f"cd {output_path} && ./build.sh && ./run.sh")
                pass
            elif config.strategy == DeploymentStrategy.KUBERNETES:
                self.logger.info("Simulating Kubernetes deployment with kubectl apply...")
                # await self._execute_shell_command(f"cd {output_path} && kubectl apply -f .")
                pass
            elif config.strategy == DeploymentStrategy.DOCKER_COMPOSE:
                self.logger.info("Simulating Docker Compose deployment...")
                # await self._execute_shell_command(f"cd {output_path} && docker-compose up -d")
                pass
            elif config.strategy == DeploymentStrategy.STANDALONE:
                self.logger.info("Simulating standalone deployment setup...")
                # No hay un "comando de despliegue" per se, solo se generan los archivos para su ejecución manual.
                pass

            status.status = "completed"
            status.last_updated = datetime.now()
            self.logger.info(f"Deployment {deployment_id} completed successfully.")
            return True

        except Exception as e:
            status.status = "failed"
            status.errors = str(e)
            status.last_updated = datetime.now()
            self.logger.error(f"Deployment {deployment_id} failed: {e}", exc_info=True)
            return False

    async def _execute_shell_command(self, command: str) -> Tuple[bool, str]:
        """Ejecuta un comando de shell y captura su salida."""
        self.logger.info(f"Executing shell command: {command}")
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            self.logger.error(f"Command failed with error: {stderr.decode()}")
            return False, stderr.decode()
        self.logger.debug(f"Command output: {stdout.decode()}")
        return True, stdout.decode()

    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentStatus]:
        """Obtiene el estado de un deployment específico."""
        return self.deployments.get(deployment_id)

    def list_deployments(self) -> List[DeploymentStatus]:
        """Lista todos los deployments registrados."""
        return list(self.deployments.values())
    
    async def cleanup_deployment_artifacts(self, deployment_id: str) -> bool:
        """Limpia los archivos generados para un deployment."""
        status = self.deployments.get(deployment_id)
        if not status:
            self.logger.warning(f"Deployment {deployment_id} not found for cleanup.")
            return False
        
        output_dir = Path(status.output_dir)
        if output_dir.exists() and output_dir.is_dir():
            try:
                shutil.rmtree(output_dir)
                self.logger.info(f"Cleaned up deployment artifacts for {deployment_id} at {output_dir}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to clean up deployment artifacts for {deployment_id}: {e}")
                return False
        self.logger.info(f"No artifacts found to cleanup for {deployment_id} at {output_dir}")
        return True

# ================================\
# DEMO
# ================================\

async def deployment_demo():
    """Ejemplo de uso del sistema de deployment."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("🚀 Starting Deployment System Demo")
    print("="*50)

    orchestrator = DeploymentOrchestrator()

    # Demo 1: Crear deployment de desarrollo (Docker)
    print("1. Creating Development Deployment (Docker)...\n")
    dev_config = await orchestrator.create_deployment_config(
        DeploymentEnvironment.DEVELOPMENT,
        DeploymentStrategy.DOCKER,
        framework_config={"log_level": "DEBUG"},
        api_config={"port": 8001},
        security_config={"jwt_secret": "dev_secret_key"}
    )
    
    success = await orchestrator.deploy(dev_config, "./deployment_dev")
    if success:
        print("✅ Development deployment manifests created")
        print("   Location: ./deployment_dev")
        print("   Run: cd deployment_dev && ./build.sh && ./run.sh")
    else:
        print("❌ Development deployment failed")

    await asyncio.sleep(1) # Pequeña pausa

    # Demo 2: Crear deployment de producción (Kubernetes)
    print("\n2. Creating Production Deployment (Kubernetes)...\n")
    prod_config = await orchestrator.create_deployment_config(
        DeploymentEnvironment.PRODUCTION,
        DeploymentStrategy.KUBERNETES,
        domain="my-agent-framework.com",
        jwt_secret="super_secure_production_secret", # Esto sobrescribe el default
        db_url="postgresql://prod_user:prod_pass@postgres.cluster/framework",
        scaling_config={"replicas": 3}
    )
    
    success = await orchestrator.deploy(prod_config, "./deployment_prod")
    if success:
        print("✅ Production deployment manifests created")
        print("   Location: ./deployment_prod")
        print("   Run: cd deployment_prod && ./deploy.sh")
    else:
        print("❌ Production deployment failed")

    await asyncio.sleep(1) # Pequeña pausa
    
    # Demo 3: Listar deployments
    print("\n3. Deployment Status:\n")
    deployments = orchestrator.list_deployments()
    for i, deployment in enumerate(deployments, 1):
        config = deployment.config
        print(f"   {i}. {config.environment.value} ({config.strategy.value})")
        print(f"      Status: {deployment.status}")
        print(f"      Created: {deployment.deployed_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"      Output: {deployment.output_dir}")
        if deployment.errors:
            print(f"      Errors: {deployment.errors}")
    
    # Demo 4: Limpiar artefactos de un deployment (ej. el de desarrollo)
    if deployments:
        dev_deployment_id = next((d.deployment_id for d in deployments if d.config.environment == DeploymentEnvironment.DEVELOPMENT), None)
        if dev_deployment_id:
            print(f"\n4. Cleaning up development deployment artifacts ({dev_deployment_id})...\n")
            cleanup_success = await orchestrator.cleanup_deployment_artifacts(dev_deployment_id)
            if cleanup_success:
                print(f"   ✅ Development deployment artifacts cleaned up.")
            else:
                print(f"   ❌ Failed to clean up development deployment artifacts.")
        
    print("\n✅ Deployment demo completed")
    print("\n📋 Next steps:")
    print("   1. Review generated deployment files in ./deployments/ directories.")
    print("   2. Customize configuration as needed.")
    print("   3. Run deployment scripts manually (e.g., ./deployment_dev/run.sh).")
    print("   4. Monitor deployment status.")


if __name__ == "__main__":
    asyncio.run(deployment_demo())