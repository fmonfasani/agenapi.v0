"""
framework_config_utils.py - Configuración y utilidades del framework de agentes
"""

import json
import yaml
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from pathlib import Path

# ================================
# CONFIGURATION CLASSES
# ================================

@dataclass
class AgentConfig:
    """Configuración de un agente específico"""
    namespace: str
    name: str
    enabled: bool = True
    auto_start: bool = True
    max_concurrent_tasks: int = 10
    heartbeat_interval: int = 30
    restart_on_failure: bool = True
    custom_settings: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_settings is None:
            self.custom_settings = {}

@dataclass
class FrameworkConfig:
    """Configuración principal del framework"""
    # Framework settings
    name: str = "Autonomous Agent Framework"
    version: str = "1.0.0"
    
    # Message bus settings
    message_queue_size: int = 1000
    message_timeout: int = 30
    enable_message_persistence: bool = False
    
    # Resource manager settings
    max_resources: int = 10000
    resource_cleanup_interval: int = 3600
    
    # Monitoring settings
    enable_monitoring: bool = True
    health_check_interval: int = 60
    metrics_collection: bool = True
    log_level: str = "INFO"
    
    # Security settings
    enable_agent_authentication: bool = False
    max_agent_lifetime: int = 86400  # 24 hours
    
    # Agents configuration
    agents: List[AgentConfig] = None
    
    def __post_init__(self):
        if self.agents is None:
            self.agents = []

class ConfigFormat(Enum):
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"

# ================================
# CONFIGURATION MANAGER
# ================================

class ConfigManager:
    """Gestor de configuración del framework"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "agent_framework_config.yaml"
        self.config: FrameworkConfig = FrameworkConfig()
        
    def load_config(self, path: Optional[str] = None) -> FrameworkConfig:
        """Cargar configuración desde archivo"""
        config_file = path or self.config_path
        
        if not os.path.exists(config_file):
            logging.warning(f"Config file {config_file} not found, using defaults")
            return self.config
            
        try:
            with open(config_file, 'r') as f:
                if config_file.endswith('.json'):
                    data = json.load(f)
                elif config_file.endswith(('.yaml', '.yml')):
                    data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_file}")
                    
            # Convert agent configs
            if 'agents' in data:
                agents = []
                for agent_data in data['agents']:
                    agents.append(AgentConfig(**agent_data))
                data['agents'] = agents
                
            self.config = FrameworkConfig(**data)
            logging.info(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            
        return self.config
        
    def save_config(self, config: FrameworkConfig, path: Optional[str] = None, 
                   format: ConfigFormat = ConfigFormat.YAML):
        """Guardar configuración a archivo"""
        config_file = path or self.config_path
        
        # Convert to dict
        config_dict = asdict(config)
        
        try:
            with open(config_file, 'w') as f:
                if format == ConfigFormat.JSON:
                    json.dump(config_dict, f, indent=2)
                elif format == ConfigFormat.YAML:
                    yaml.dump(config_dict, f, default_flow_style=False)
                    
            logging.info(f"Configuration saved to {config_file}")
            
        except Exception as e:
            logging.error(f"Error saving config: {e}")
            
    def create_default_config(self) -> FrameworkConfig:
        """Crear configuración por defecto"""
        agents = [
            AgentConfig("agent.planning.strategist", "strategist"),
            AgentConfig("agent.planning.workflow", "workflow_designer"),
            AgentConfig("agent.build.code.generator", "code_generator"),
            AgentConfig("agent.build.ux.generator", "ux_generator"),
            AgentConfig("agent.test.generator", "test_generator"),
            AgentConfig("agent.security.sentinel", "security_sentinel"),
            AgentConfig("agent.monitor.progress", "progress_monitor"),
        ]
        
        config = FrameworkConfig(
            name="Default Agent Framework",
            agents=agents
        )
        
        return config
        
    def validate_config(self, config: FrameworkConfig) -> List[str]:
        """Validar configuración"""
        errors = []
        
        # Validar nombres únicos de agentes
        agent_names = [agent.name for agent in config.agents]
        if len(agent_names) != len(set(agent_names)):
            errors.append("Duplicate agent names found")
            
        # Validar namespaces
        from core.specialized_agents import ExtendedAgentFactory
        available_namespaces = ExtendedAgentFactory.list_available_namespaces()
        
        for agent in config.agents:
            if agent.namespace not in available_namespaces:
                errors.append(f"Unknown namespace: {agent.namespace}")
                
        # Validar configuraciones numéricas
        if config.message_queue_size <= 0:
            errors.append("message_queue_size must be positive")
            
        if config.health_check_interval <= 0:
            errors.append("health_check_interval must be positive")
            
        return errors

# ================================
# FRAMEWORK BUILDER
# ================================

class FrameworkBuilder:
    """Constructor del framework con configuración"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config()
        
    async def build_framework(self):
        """Construir framework con configuración"""
        from core.autonomous_agent_framework import AgentFramework
        from core.specialized_agents import ExtendedAgentFactory
        
        # Crear framework
        framework = AgentFramework()
        
        # Configurar logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Iniciar framework
        await framework.start()
        
        # Crear agentes configurados
        agents = {}
        for agent_config in self.config.agents:
            if agent_config.enabled:
                try:
                    agent = ExtendedAgentFactory.create_agent(
                        agent_config.namespace,
                        agent_config.name,
                        framework
                    )
                    
                    if agent_config.auto_start:
                        await agent.start()
                        
                    agents[agent_config.name] = agent
                    logging.info(f"Created agent: {agent_config.name}")
                    
                except Exception as e:
                    logging.error(f"Failed to create agent {agent_config.name}: {e}")
                    
        return framework, agents
        
    def create_sample_config(self, path: str = "sample_config.yaml"):
        """Crear configuración de ejemplo"""
        config = self.config_manager.create_default_config()
        self.config_manager.save_config(config, path)
        print(f"Sample configuration created at: {path}")

# ================================
# UTILITIES
# ================================

class AgentOrchestrator:
    """Orquestador de agentes para flujos complejos"""
    
    def __init__(self, framework, agents: Dict[str, Any]):
        self.framework = framework
        self.agents = agents
        
    async def execute_workflow(self, workflow_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecutar un workflow completo"""
        results = {}
        steps = workflow_definition.get("steps", [])
        
        for step in steps:
            step_id = step["id"]
            agent_name = step["agent"]
            action = step["action"]
            params = step.get("params", {})
            
            # Resolver dependencias
            if "dependencies" in step:
                for dep in step["dependencies"]:
                    if dep in results:
                        params.update(results[dep])
                        
            # Ejecutar paso
            try:
                if agent_name in self.agents:
                    agent = self.agents[agent_name]
                    result = await agent.execute_action(action, params)
                    results[step_id] = result
                    logging.info(f"Completed step {step_id}")
                else:
                    raise ValueError(f"Agent {agent_name} not found")
                    
            except Exception as e:
                logging.error(f"Error in step {step_id}: {e}")
                results[step_id] = {"error": str(e)}
                
        return results
        
    async def create_development_pipeline(self, project_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Crear pipeline de desarrollo automático"""
        
        pipeline = {
            "steps": [
                {
                    "id": "strategy",
                    "agent": "strategist",
                    "action": "define.strategy",
                    "params": {"requirements": project_spec}
                },
                {
                    "id": "workflow",
                    "agent": "workflow_designer", 
                    "action": "design.workflow",
                    "params": {"tasks": project_spec.get("tasks", [])},
                    "dependencies": ["strategy"]
                },
                {
                    "id": "code_generation",
                    "agent": "code_generator",
                    "action": "generate.component",
                    "params": {"specification": project_spec.get("code_spec", {})},
                    "dependencies": ["workflow"]
                },
                {
                    "id": "test_generation",
                    "agent": "test_generator",
                    "action": "generate.tests",
                    "params": {"test_framework": "pytest"},
                    "dependencies": ["code_generation"]
                },
                {
                    "id": "security_scan",
                    "agent": "security_sentinel",
                    "action": "scan.vulnerabilities",
                    "params": {"scan_type": "comprehensive"},
                    "dependencies": ["code_generation"]
                }
            ]
        }
        
        return await self.execute_workflow(pipeline)

class MetricsCollector:
    """Recolector de métricas del framework"""
    
    def __init__(self, framework):
        self.framework = framework
        self.metrics = {
            "agents_created": 0,
            "agents_terminated": 0,
            "messages_sent": 0,
            "resources_created": 0,
            "errors": 0
        }
        
    def record_agent_created(self):
        self.metrics["agents_created"] += 1
        
    def record_agent_terminated(self):
        self.metrics["agents_terminated"] += 1
        
    def record_message_sent(self):
        self.metrics["messages_sent"] += 1
        
    def record_resource_created(self):
        self.metrics["resources_created"] += 1
        
    def record_error(self):
        self.metrics["errors"] += 1
        
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas actuales"""
        active_agents = len(self.framework.registry.list_all_agents())
        
        return {
            **self.metrics,
            "active_agents": active_agents,
            "timestamp": __import__("time").time()
        }
        
    def export_metrics(self, format: str = "json") -> str:
        """Exportar métricas en formato específico"""
        metrics = self.get_metrics()
        
        if format == "json":
            return json.dumps(metrics, indent=2)
        elif format == "yaml":
            return yaml.dump(metrics)
        else:
            return str(metrics)

# ================================
# CLI UTILITIES
# ================================

class FrameworkCLI:
    """Interfaz de línea de comandos para el framework"""
    
    def __init__(self):
        self.framework = None
        self.agents = {}
        
    async def start_framework(self, config_path: Optional[str] = None):
        """Iniciar framework desde CLI"""
        builder = FrameworkBuilder(config_path)
        self.framework, self.agents = await builder.build_framework()
        print(f"Framework started with {len(self.agents)} agents")
        
    async def stop_framework(self):
        """Detener framework"""
        if self.framework:
            await self.framework.stop()
            print("Framework stopped")
            
    def list_agents(self):
        """Listar agentes activos"""
        if not self.agents:
            print("No agents active")
            return
            
        print("Active agents:")
        for name, agent in self.agents.items():
            print(f"  - {name} ({agent.namespace}) - {agent.status.value}")
            
    async def send_command(self, agent_name: str, action: str, params: str = "{}"):
        """Enviar comando a agente"""
        if agent_name not in self.agents:
            print(f"Agent {agent_name} not found")
            return
            
        try:
            params_dict = json.loads(params)
            agent = self.agents[agent_name]
            result = await agent.execute_action(action, params_dict)
            print(f"Result: {json.dumps(result, indent=2)}")
        except Exception as e:
            print(f"Error: {e}")
            
    def show_resources(self):
        """Mostrar recursos creados"""
        if not self.framework:
            print("Framework not started")
            return
            
        all_resources = []
        for agent in self.agents.values():
            agent_resources = self.framework.resource_manager.find_resources_by_owner(agent.id)
            all_resources.extend(agent_resources)
            
        if not all_resources:
            print("No resources found")
            return
            
        print("Resources:")
        for resource in all_resources:
            print(f"  - {resource.type.value}: {resource.name} (owner: {resource.owner_agent_id})")

# ================================
# EXAMPLE USAGE
# ================================

async def example_with_config():
    """Ejemplo usando configuración"""
    
    # Crear configuración de ejemplo
    builder = FrameworkBuilder()
    builder.create_sample_config("example_config.yaml")
    
    # Construir framework
    framework, agents = await builder.build_framework()
    
    # Crear orquestador
    orchestrator = AgentOrchestrator(framework, agents)
    
    # Ejecutar pipeline de desarrollo
    project_spec = {
        "goal": "Create user management system",
        "tasks": [
            {"name": "Design API", "type": "design"},
            {"name": "Implement backend", "type": "development"},
            {"name": "Create tests", "type": "testing"}
        ],
        "code_spec": {
            "name": "UserManager",
            "methods": [
                {"name": "create_user", "parameters": [{"name": "user_data", "type": "dict"}]},
                {"name": "update_user", "parameters": [{"name": "user_id", "type": "str"}]},
                {"name": "delete_user", "parameters": [{"name": "user_id", "type": "str"}]}
            ]
        }
    }
    
    results = await orchestrator.create_development_pipeline(project_spec)
    
    print("Pipeline completed:")
    for step_id, result in results.items():
        print(f"  {step_id}: {'✓' if 'error' not in result else '✗'}")
        
    # Mostrar métricas
    metrics = MetricsCollector(framework)
    print("\nMetrics:", metrics.export_metrics())
    
    await framework.stop()

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_with_config())