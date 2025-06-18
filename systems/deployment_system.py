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
from core.models import ResourceType # Importar de core.models

class DeploymentEnvironment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class DeploymentStrategy(Enum):
    STANDALONE = "standalone"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    DOCKER_COMPOSE = "docker_compose"

@dataclass
class DeploymentConfig:
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
    deployment_id: str
    config: DeploymentConfig
    status: str
    deployed_at: datetime
    output_dir: Path
    logs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

class DeploymentOrchestrator:
    def __init__(self, framework: AgentFramework):
        self.framework = framework
        self.logger = logging.getLogger("DeploymentOrchestrator")
        self.deployments: Dict[str, DeploymentStatus] = {}

    def create_deployment_config(self, 
                                 environment: DeploymentEnvironment, 
                                 strategy: DeploymentStrategy, 
                                 **kwargs) -> DeploymentConfig:
        
        framework_defaults = {
            "name": f"Framework-{environment.value}",
            "log_level": "INFO" if environment == DeploymentEnvironment.PRODUCTION else "DEBUG",
            "message_queue_size": 10000,
        }
        security_defaults = {
            "jwt_secret": "super_secret_key" if environment != DeploymentEnvironment.PRODUCTION else kwargs.get("jwt_secret", "CHANGE_THIS_IN_PROD"),
            "enable_authentication": True,
            "enable_authorization": True,
        }
        persistence_defaults = {
            "backend": PersistenceBackend.SQLITE.value,
            "connection_string": f"{environment.value}.db",
            "auto_save_interval": 300,
        }
        api_defaults = {
            "host": "0.0.0.0",
            "port": 8000 if environment != DeploymentEnvironment.PRODUCTION else 443,
            "enable_https": True if environment == DeploymentEnvironment.PRODUCTION else False,
            "swagger_path": "/docs",
        }
        agents_defaults = [
            {"namespace": "agent.planning.strategist", "name": "strategist", "auto_start": True},
            {"namespace": "agent.build.code.generator", "name": "code_generator", "auto_start": True},
        ]
        monitoring_defaults = {
            "enable_monitoring": True,
            "health_check_interval": 60,
            "metrics_collection": True,
            "alerting": {
                "enabled": True,
                "email_alerts": False,
                "webhook_alerts": [],
            }
        }
        scaling_defaults = {
            "min_agents": 1,
            "max_agents": 5,
            "cpu_threshold": 80,
        }

        # Override defaults with kwargs
        framework_config = {**framework_defaults, **kwargs.get("framework_config", {})}
        security_config = {**security_defaults, **kwargs.get("security_config", {})}
        persistence_config = {**persistence_defaults, **kwargs.get("persistence_config", {})}
        api_config = {**api_defaults, **kwargs.get("api_config", {})}
        agents_config = kwargs.get("agents_config", agents_defaults)
        monitoring_config = {**monitoring_defaults, **kwargs.get("monitoring_config", {})}
        scaling_config = {**scaling_defaults, **kwargs.get("scaling_config", {})}

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

    async def deploy(self, config: DeploymentConfig, output_dir: Union[str, Path]) -> bool:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        deployment_id = f"deploy-{config.environment.value}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        status = DeploymentStatus(
            deployment_id=deployment_id,
            config=config,
            status="PENDING",
            deployed_at=datetime.now(),
            output_dir=output_dir,
        )
        self.deployments[deployment_id] = status
        self.logger.info(f"Initiating deployment '{deployment_id}' for {config.environment.value} environment with {config.strategy.value} strategy.")

        try:
            if config.strategy == DeploymentStrategy.STANDALONE:
                await self._deploy_standalone(config, output_dir, status)
            elif config.strategy == DeploymentStrategy.DOCKER:
                await self._deploy_docker(config, output_dir, status)
            elif config.strategy == DeploymentStrategy.KUBERNETES:
                await self._deploy_kubernetes(config, output_dir, status)
            elif config.strategy == DeploymentStrategy.DOCKER_COMPOSE:
                await self._deploy_docker_compose(config, output_dir, status)
            else:
                raise ValueError(f"Unsupported deployment strategy: {config.strategy.value}")

            status.status = "COMPLETED"
            self.logger.info(f"Deployment '{deployment_id}' completed successfully.")
            return True
        except Exception as e:
            status.status = "FAILED"
            status.errors.append(str(e))
            self.logger.error(f"Deployment '{deployment_id}' failed: {e}", exc_info=True)
            return False

    async def _deploy_standalone(self, config: DeploymentConfig, output_dir: Path, status: DeploymentStatus):
        self.logger.info(f"Generating standalone deployment files in {output_dir}")
        
        main_script_content = self._generate_main_script(config)
        with open(output_dir / "run_framework.py", "w") as f:
            f.write(main_script_content)
        status.logs.append(f"Generated {output_dir / 'run_framework.py'}")

        requirements_content = self._generate_requirements(config)
        with open(output_dir / "requirements.txt", "w") as f:
            f.write(requirements_content)
        status.logs.append(f"Generated {output_dir / 'requirements.txt'}")

        run_script_content = f"""#!/bin/bash
echo "Installing dependencies..."
pip install -r requirements.txt
echo "Starting framework..."
python run_framework.py
"""
        with open(output_dir / "run.sh", "w") as f:
            f.write(run_script_content)
        os.chmod(output_dir / "run.sh", 0o755)
        status.logs.append(f"Generated {output_dir / 'run.sh'}")
        self.logger.info("Standalone deployment files generated.")

    async def _deploy_docker(self, config: DeploymentConfig, output_dir: Path, status: DeploymentStatus):
        self.logger.info(f"Generating Docker deployment files in {output_dir}")
        
        main_script_content = self._generate_main_script(config)
        with open(output_dir / "app.py", "w") as f:
            f.write(main_script_content)
        
        requirements_content = self._generate_requirements(config)
        with open(output_dir / "requirements.txt", "w") as f:
            f.write(requirements_content)

        dockerfile_content = f"""
FROM python:3.9-slim-buster
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "app.py"]
"""
        with open(output_dir / "Dockerfile", "w") as f:
            f.write(dockerfile_content)
        status.logs.append(f"Generated {output_dir / 'Dockerfile'}")

        build_script_content = f"""#!/bin/bash
IMAGE_NAME="agent-framework-{config.environment.value}"
TAG="latest"
echo "Building Docker image $IMAGE_NAME:$TAG..."
docker build -t $IMAGE_NAME:$TAG .
echo "Docker image built. To run: docker run -p {config.api_config['port']}:{config.api_config['port']} $IMAGE_NAME:$TAG"
"""
        with open(output_dir / "build_and_run.sh", "w") as f:
            f.write(build_script_content)
        os.chmod(output_dir / "build_and_run.sh", 0o755)
        status.logs.append(f"Generated {output_dir / 'build_and_run.sh'}")
        self.logger.info("Docker deployment files generated.")

    async def _deploy_kubernetes(self, config: DeploymentConfig, output_dir: Path, status: DeploymentStatus):
        self.logger.info(f"Generating Kubernetes deployment files in {output_dir}")
        
        # Similar a Docker, primero generar la imagen, luego los manifiestos
        await self._deploy_docker(config, output_dir, status) 

        app_name = f"agent-framework-{config.environment.value}"

        deployment_manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": app_name},
            "spec": {
                "replicas": config.scaling_config.get("min_agents", 1),
                "selector": {"matchLabels": {"app": app_name}},
                "template": {
                    "metadata": {"labels": {"app": app_name}},
                    "spec": {
                        "containers": [{
                            "name": "framework",
                            "image": f"{app_name}:latest", 
                            "ports": [{"containerPort": config.api_config['port']}],
                            "resources": {
                                "requests": {"cpu": "100m", "memory": "256Mi"},
                                "limits": {"cpu": "500m", "memory": "1Gi"}
                            },
                            "env": [
                                {"name": "FRAMEWORK_CONFIG", "value": json.dumps(config.framework_config)},
                                {"name": "SECURITY_CONFIG", "value": json.dumps(config.security_config)},
                                {"name": "PERSISTENCE_CONFIG", "value": json.dumps(config.persistence_config)},
                                {"name": "API_CONFIG", "value": json.dumps(config.api_config)},
                                {"name": "AGENTS_CONFIG", "value": json.dumps(config.agents_config)},
                                {"name": "MONITORING_CONFIG", "value": json.dumps(config.monitoring_config)},
                            ]
                        }]
                    }
                }
            }
        }
        with open(output_dir / "deployment.yaml", "w") as f:
            yaml.dump(deployment_manifest, f, indent=2)
        status.logs.append(f"Generated {output_dir / 'deployment.yaml'}")

        service_manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": app_name},
            "spec": {
                "selector": {"app": app_name},
                "ports": [{"protocol": "TCP", "port": config.api_config['port'], "targetPort": config.api_config['port']}],
                "type": "LoadBalancer" if config.environment == DeploymentEnvironment.PRODUCTION else "ClusterIP"
            }
        }
        with open(output_dir / "service.yaml", "w") as f:
            yaml.dump(service_manifest, f, indent=2)
        status.logs.append(f"Generated {output_dir / 'service.yaml'}")

        deploy_script_content = f"""#!/bin/bash
echo "Applying Kubernetes manifests..."
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
echo "Kubernetes deployment initiated for {app_name}."
echo "Monitor status with: kubectl get pods -l app={app_name}"
echo "Access service: kubectl get service {app_name}"
"""
        with open(output_dir / "deploy.sh", "w") as f:
            f.write(deploy_script_content)
        os.chmod(output_dir / "deploy.sh", 0o755)
        status.logs.append(f"Generated {output_dir / 'deploy.sh'}")
        self.logger.info("Kubernetes deployment files generated.")

    async def _deploy_docker_compose(self, config: DeploymentConfig, output_dir: Path, status: DeploymentStatus):
        self.logger.info(f"Generating Docker Compose deployment files in {output_dir}")

        main_script_content = self._generate_main_script(config)
        with open(output_dir / "app.py", "w") as f:
            f.write(main_script_content)
        
        requirements_content = self._generate_requirements(config)
        with open(output_dir / "requirements.txt", "w") as f:
            f.write(requirements_content)

        dockerfile_content = f"""
FROM python:3.9-slim-buster
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "app.py"]
"""
        with open(output_dir / "Dockerfile", "w") as f:
            f.write(dockerfile_content)
        
        service_name = f"framework-{config.environment.value}"
        docker_compose_content = {
            "version": "3.8",
            "services": {
                service_name: {
                    "build": ".",
                    "ports": [f"{config.api_config['port']}:{config.api_config['port']}"],
                    "environment": {
                        "FRAMEWORK_CONFIG": json.dumps(config.framework_config),
                        "SECURITY_CONFIG": json.dumps(config.security_config),
                        "PERSISTENCE_CONFIG": json.dumps(config.persistence_config),
                        "API_CONFIG": json.dumps(config.api_config),
                        "AGENTS_CONFIG": json.dumps(config.agents_config),
                        "MONITORING_CONFIG": json.dumps(config.monitoring_config),
                    },
                    "volumes": [
                        f"./data_{config.environment.value}:/app/data" # Persist data
                    ]
                }
            }
        }

        with open(output_dir / "docker-compose.yaml", "w") as f:
            yaml.dump(docker_compose_content, f, indent=2)
        status.logs.append(f"Generated {output_dir / 'docker-compose.yaml'}")

        run_script_content = f"""#!/bin/bash
echo "Building and running Docker Compose services..."
docker-compose up --build -d
echo "Services started. Use 'docker-compose logs -f' to see logs."
echo "To stop: docker-compose down"
"""
        with open(output_dir / "run_compose.sh", "w") as f:
            f.write(run_script_content)
        os.chmod(output_dir / "run_compose.sh", 0o755)
        status.logs.append(f"Generated {output_dir / 'run_compose.sh'}")
        self.logger.info("Docker Compose deployment files generated.")

    def _generate_main_script(self, config: DeploymentConfig) -> str:
        framework_config_str = json.dumps(config.framework_config)
        security_config_str = json.dumps(config.security_config)
        persistence_config_str = json.dumps(config.persistence_config)
        api_config_str = json.dumps(config.api_config)
        agents_config_str = json.dumps(config.agents_config)
        monitoring_config_str = json.dumps(config.monitoring_config)

        # Importaciones necesarias
        imports = """
import asyncio
import os
import json
import logging
from typing import Dict, Any, List
from aiohttp import web

from core.autonomous_agent_framework import AgentFramework, BaseAgent
from core.specialized_agents import StrategistAgent, WorkflowDesignerAgent, CodeGeneratorAgent, TestGeneratorAgent, BuildAgent
from core.security_system import SecurityManager
from core.persistence_system import PersistenceManager, PersistenceFactory, PersistenceBackend
from interfaces.rest_api import FrameworkAPIServer
from core.monitoring_system import MonitoringOrchestrator
"""

        # Agentes disponibles para carga (a mapear en la factoría)
        agent_class_map = {
            "agent.planning.strategist": "StrategistAgent",
            "agent.planning.workflow_designer": "WorkflowDesignerAgent",
            "agent.build.code.generator": "CodeGeneratorAgent",
            "agent.test.generator": "TestGeneratorAgent",
            "agent.build.builder": "BuildAgent",
        }
        
        agent_creation_lines = []
        for agent_cfg in config.agents_config:
            if agent_cfg.get("enabled", True):
                namespace = agent_cfg["namespace"]
                name = agent_cfg["name"]
                agent_class = agent_class_map.get(namespace)
                if agent_class:
                    agent_creation_lines.append(
                        f"    agent_instance = await framework.agent_factory.create_agent_instance(\n"
                        f"        '{namespace}', '{name}', {agent_class}, framework\n"
                        f"    )\n"
                        f"    if agent_instance: deployed_agents[agent_instance.id] = agent_instance\n"
                    )
                else:
                    self.logger.warning(f"Agent class for namespace '{namespace}' not found. Skipping.")


        script_content = f"""{imports}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DeploymentMain")

async def main():
    logger.info("Starting framework deployment...")

    framework_config = json.loads(os.getenv('FRAMEWORK_CONFIG', '{framework_config_str}'))
    security_config = json.loads(os.getenv('SECURITY_CONFIG', '{security_config_str}'))
    persistence_config = json.loads(os.getenv('PERSISTENCE_CONFIG', '{persistence_config_str}'))
    api_config = json.loads(os.getenv('API_CONFIG', '{api_config_str}'))
    agents_config = json.loads(os.getenv('AGENTS_CONFIG', '{agents_config_str}'))
    monitoring_config = json.loads(os.getenv('MONITORING_CONFIG', '{monitoring_config_str}'))

    framework = AgentFramework(config=framework_config)
    await framework.start()

    security_manager = SecurityManager(framework, security_config)
    
    persistence_manager = PersistenceManager(framework, persistence_config)
    await persistence_manager.initialize()

    api_server = FrameworkAPIServer(framework, security_manager, persistence_manager, host=api_config['host'], port=api_config['port'])
    runner = await api_server.start()

    monitoring_orchestrator = MonitoringOrchestrator(framework, monitoring_config)
    await monitoring_orchestrator.start_monitoring()

    logger.info("Deploying agents as per configuration...")
    deployed_agents: Dict[str, BaseAgent] = {{}}
{chr(10).join(agent_creation_lines)}
    logger.info(f"Deployed {{len(deployed_agents)}} agents.")

    logger.info(f"Framework and services are running. Access API at http://{{api_config['host']}}:{{api_config['port']}}")
    logger.info("Press Ctrl+C to stop...")

    try:
        while True:
            await asyncio.sleep(3600) # Keep running indefinitely
    except asyncio.CancelledError:
        logger.info("Application shutdown initiated.")
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Shutting down...")
    finally:
        logger.info("Stopping monitoring...")
        await monitoring_orchestrator.stop_monitoring()
        logger.info("Stopping API server...")
        await api_server.stop()
        logger.info("Stopping framework and agents...")
        await framework.stop()
        logger.info("Closing persistence...")
        await persistence_manager.close()
        logger.info("Application shut down cleanly.")

if __name__ == "__main__":
    asyncio.run(main())
"""
        return script_content

    def _generate_requirements(self, config: DeploymentConfig) -> str:
        requirements = [
            "aiohttp",
            "aiohttp-cors",
            "aiohttp-swagger",
            "PyYAML",
            "pyjwt",
            "psutil",
            "aiohttp",
            "aiosqlite",
            "python-dotenv",
            "asyncio",
            "logging",
            "json",
            "dataclasses",
            "datetime",
            "enum",
            "abc",
            "uuid",
            "weakref",
            "contextlib",
            "traceback",
            "statistics",
            "smtplib",
            "email",
            "paramiko", # for SSH backend example (if used)
            "boto3",    # for S3 backend example (if used)
            "azure-storage-blob", # for Azure backend example (if used)
        ]
        
        # Add specialized agent dependencies here if they are external packages
        # Example: if you had an agent that uses 'tensorflow'
        # if any(a.get("namespace").startswith("agent.ml") for a in config.agents_config):
        #     requirements.append("tensorflow")
            
        # Add persistence backend specific requirements
        if config.persistence_config["backend"] == PersistenceBackend.POSTGRESQL.value:
            requirements.append("asyncpg")
        elif config.persistence_config["backend"] == PersistenceBackend.REDIS.value:
            requirements.append("aioredis")

        return "\n".join(sorted(list(set(requirements))))

    def list_deployments(self) -> List[DeploymentStatus]:
        return list(self.deployments.values())

    async def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentStatus]:
        return self.deployments.get(deployment_id)

async def deployment_example():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger("DeploymentOrchestrator").setLevel(logging.DEBUG)

    print("🚀 Starting Deployment System Demo")
    print("="*50)

    framework_instance = AgentFramework() # Placeholder framework instance
    orchestrator = DeploymentOrchestrator(framework_instance)

    print("\n1. Creating Development Deployment (Standalone)...")
    dev_config = orchestrator.create_deployment_config(
        DeploymentEnvironment.DEVELOPMENT,
        DeploymentStrategy.STANDALONE,
        api_config={"port": 8001, "host": "127.0.0.1"},
        agents_config=[
            {"namespace": "agent.planning.strategist", "name": "dev_strategist", "auto_start": True}
        ]
    )
    success = await orchestrator.deploy(dev_config, "./deployment_dev")
    if success:
        print("✅ Development deployment files created")
        print("   Location: ./deployment_dev")
        print("   To run: cd deployment_dev && ./run.sh")

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
    
    print("\n3. Deployment Status:")
    deployments = orchestrator.list_deployments()
    for i, deployment in enumerate(deployments, 1):
        config = deployment.config
        print(f"   {i}. {config.environment.value} ({config.strategy.value})")
        print(f"      Status: {deployment.status}")
        print(f"      Created: {deployment.deployed_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"      Output: {deployment.output_dir}")
    
    print("\n✅ Deployment demo completed")
    print("\n📋 Next steps:")
    print("   1. Review generated deployment files")
    print("   2. Customize configuration as needed")
    print("   3. Run deployment scripts")
    print("   4. Monitor deployment status")

if __name__ == "__main__":
    asyncio.run(deployment_example())