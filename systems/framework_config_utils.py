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

# ================================\
# CONFIGURATION CLASSES
# ================================\

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
    log_level: str = "INFO" # Default logging level

    # Security settings
    authentication_enabled: bool = True
    default_auth_method: str = "JWT_TOKEN"
    jwt_secret_key: str = "super_secret_jwt_key_please_change"
    api_key_header: str = "X-API-Key"

    # Persistence settings
    persistence_backend: str = "sqlite"
    persistence_connection_string: str = "framework.db"
    
    # Deployment settings
    deployment_environment: str = "development" # development, staging, production
    deployment_strategy: str = "standalone" # standalone, docker, kubernetes

    # Plugin settings
    plugin_paths: List[str] = field(default_factory=list)
    enabled_plugins: List[str] = field(default_factory=list)

    # Agent specific configurations (overrides for specific agents)
    agent_configs: List[AgentConfig] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convierte la configuración a un diccionario."""
        # Convert dataclass to dict, handling nested dataclasses and enums
        data = asdict(self)
        # Convert enums to their values if any were added to the config
        for key, value in data.items():
            if isinstance(value, Enum):
                data[key] = value.value
            elif isinstance(value, list):
                data[key] = [v.value if isinstance(v, Enum) else v for v in value]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Crea una instancia de FrameworkConfig desde un diccionario."""
        # Convert enum values back to enums if necessary
        # This example assumes simple string representations for enums, adjust if actual Enum objects are used
        if 'log_level' in data and isinstance(data['log_level'], str):
            data['log_level'] = data['log_level'].upper() # Ensure it's uppercase for logging.setLevel

        if 'agent_configs' in data:
            data['agent_configs'] = [AgentConfig(**ac) for ac in data['agent_configs']]

        return cls(**data)

    def save_to_yaml(self, file_path: Union[str, Path]):
        """Guarda la configuración a un archivo YAML."""
        with open(file_path, 'w') as f:
            yaml.dump(self.to_dict(), f, sort_keys=False, indent=2)
        logging.info(f"Configuration saved to {file_path}")

    @classmethod
    def load_from_yaml(cls, file_path: Union[str, Path]):
        """Carga la configuración desde un archivo YAML."""
        if not Path(file_path).exists():
            logging.warning(f"Configuration file not found: {file_path}. Using default configuration.")
            return cls()
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        logging.info(f"Configuration loaded from {file_path}")
        return cls.from_dict(data)

# ================================\
# FRAMEWORK BUILDER & ORCHESTRATOR
# (These would be moved elsewhere if they become complex, e.g., core/orchestration)
# ================================\

class FrameworkBuilder:
    """Clase para construir el framework y sus componentes a partir de una configuración."""
    
    def __init__(self, config: Optional[FrameworkConfig] = None):
        self.config = config if config else FrameworkConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    def create_sample_config(self, file_path: Union[str, Path] = "config/default_framework_config.yaml"):
        """Crea un archivo de configuración de ejemplo."""
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        sample_config = FrameworkConfig(
            name="MyAgentSystem",
            log_level="INFO",
            enable_message_persistence=True,
            authentication_enabled=True,
            jwt_secret_key="my_development_secret_key",
            persistence_backend="sqlite",
            persistence_connection_string="data/my_agent_system.db",
            agent_configs=[
                AgentConfig(namespace="agent.planning.strategist", name="strategist_v1", auto_start=True),
                AgentConfig(namespace="agent.build.code.generator", name="codegen_v1", auto_start=True),
                AgentConfig(namespace="agent.test.unit_tester", name="tester_v1", auto_start=False)
            ],
            plugin_paths=["plugins"],
            enabled_plugins=["github_plugin", "openai_plugin"]
        )
        sample_config.save_to_yaml(file_path)
        self.logger.info(f"Sample configuration created at {file_path}")

    async def build_framework(self) -> Tuple['AgentFramework', Dict[str, 'BaseAgent']]:
        """Construye una instancia del framework y los agentes basados en la configuración."""
        from core.autonomous_agent_framework import AgentFramework # Lazy import to avoid circular dependencies
        from core.security_system import SecurityManager
        from core.persistence_system import PersistenceFactory, PersistenceBackend
        from core.monitoring_system import MonitoringOrchestrator
        from systems.plugin_system import PluginManager, ExternalAPIPlugin # Assuming these imports are correct
        from core.specialized_agents import ExtendedAgentFactory # For creating specific agent types

        self.logger.info("Building Agent Framework from configuration...")

        framework = AgentFramework(
            message_queue_size=self.config.message_queue_size,
            message_timeout=self.config.message_timeout,
            log_level=self.config.log_level
        )

        # 1. Initialize Security System
        security_manager = SecurityManager(framework, self.config.jwt_secret_key, self.config.authentication_enabled)
        await security_manager.initialize()
        framework.security_manager = security_manager
        self.logger.info("Security System initialized.")

        # 2. Initialize Persistence System
        persistence_backend_enum = PersistenceBackend(self.config.persistence_backend.lower())
        persistence_manager = PersistenceFactory.create_persistence_manager(
            framework, persistence_backend_enum, self.config.persistence_connection_string
        )
        await persistence_manager.initialize()
        framework.persistence_manager = persistence_manager
        self.logger.info("Persistence System initialized.")

        # 3. Initialize Monitoring System
        if self.config.enable_monitoring:
            monitoring_orchestrator = MonitoringOrchestrator(framework)
            await monitoring_orchestrator.start_monitoring()
            framework.monitoring_orchestrator = monitoring_orchestrator
            self.logger.info("Monitoring System initialized and started.")
        else:
            self.logger.info("Monitoring System disabled by configuration.")

        # 4. Initialize Plugin System
        plugin_manager = PluginManager(framework, self.config.plugin_paths)
        await plugin_manager.load_plugins()
        framework.plugin_manager = plugin_manager
        self.logger.info(f"Plugin System initialized. Loaded {len(plugin_manager.list_plugins())} plugins.")
        
        # Enable configured plugins
        for plugin_name in self.config.enabled_plugins:
            plugin = plugin_manager.get_plugin(plugin_name)
            if plugin:
                await plugin_manager.enable_plugin(plugin_name, {"api_key": "YOUR_API_KEY"}) # Placeholder for actual plugin config
                self.logger.info(f"Plugin '{plugin_name}' enabled.")
            else:
                self.logger.warning(f"Configured plugin '{plugin_name}' not found.")

        await framework.start()
        self.logger.info("Agent Framework core started.")

        # 5. Create and Register Agents
        created_agents = {}
        for agent_cfg in self.config.agent_configs:
            if agent_cfg.enabled:
                try:
                    # ExtendedAgentFactory can handle creating agents from their namespace string
                    agent = ExtendedAgentFactory.create_agent(agent_cfg.namespace, agent_cfg.name, framework)
                    if agent:
                        # Apply custom settings from config to agent instance
                        if agent_cfg.custom_settings:
                            for key, value in agent_cfg.custom_settings.items():
                                setattr(agent, key, value) # Set attributes dynamically
                        
                        await framework.registry.register_agent(agent)
                        if agent_cfg.auto_start:
                            await agent.start()
                            self.logger.info(f"Agent '{agent.name}' ({agent.namespace}) started.")
                        else:
                            self.logger.info(f"Agent '{agent.name}' ({agent.namespace}) registered (auto_start=False).")
                        created_agents[agent.id] = agent
                    else:
                        self.logger.error(f"Could not create agent for namespace: {agent_cfg.namespace}")
                except Exception as e:
                    self.logger.error(f"Error creating agent {agent_cfg.name} ({agent_cfg.namespace}): {e}")
                    self.logger.debug(traceback.format_exc())

        self.logger.info(f"Finished building framework with {len(created_agents)} agents.")
        return framework, created_agents

# No es necesario incluir AgentOrchestrator y MetricsCollector aquí.
# AgentOrchestrator puede ir en un módulo de "core.orchestration" si se vuelve más complejo.
# MetricsCollector ya está en systems.monitoring_system.