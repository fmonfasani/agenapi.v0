# agentapi/config.py

import os
import yaml
import logging
from pathlib import Path
from typing import Optional

from agentapi.models.framework_models import FrameworkConfig, AgentConfig # Import config models

class ConfigLoader:
    _instance: Optional['ConfigLoader'] = None
    _config: Optional[FrameworkConfig] = None
    _config_path: Optional[Path] = None

    def __new__(cls, config_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._initialize(config_path)
        return cls._instance

    def _initialize(self, config_path: Optional[str]):
        self.logger = logging.getLogger("ConfigLoader")
        if config_path:
            self._config_path = Path(config_path)
        else:
            default_paths = [
                Path("./config/framework_config.yaml"),
                Path("./framework_config.yaml"),
                Path.home() / ".agent-framework" / "config.yaml"
            ]
            for p in default_paths:
                if p.exists():
                    self._config_path = p
                    break
            
            if not self._config_path:
                self.logger.warning("No config file found in default paths. Using default FrameworkConfig.")

        if self._config_path and self._config_path.exists():
            self._load_config_from_file()
        else:
            self.logger.info("Using default framework configuration.")
            ConfigLoader._config = FrameworkConfig() # Use default if no file found

    def _load_config_from_file(self):
        try:
            with open(self._config_path, 'r') as f:
                raw_config = yaml.safe_load(f)
            
            # Manually instantiate nested dataclasses from raw_config
            # This is a robust way to handle nested configurations
            security_config = raw_config.pop("security", {})
            persistence_config = raw_config.pop("persistence", {})
            monitoring_config = raw_config.pop("monitoring", {})
            backup_config = raw_config.pop("backup", {})
            deployment_config = raw_config.pop("deployment", {})
            plugins_config = raw_config.pop("plugins", {})
            agents_config_list = raw_config.pop("agents", [])

            ConfigLoader._config = FrameworkConfig(
                security=FrameworkConfig.__dataclass_fields__['security'].type(**security_config),
                persistence=FrameworkConfig.__dataclass_fields__['persistence'].type(**persistence_config),
                monitoring=FrameworkConfig.__dataclass_fields__['monitoring'].type(**monitoring_config),
                backup=FrameworkConfig.__dataclass_fields__['backup'].type(**backup_config),
                deployment=FrameworkConfig.__dataclass_fields__['deployment'].type(**deployment_config),
                plugins=FrameworkConfig.__dataclass_fields__['plugins'].type(**plugins_config),
                agents=[AgentConfig(**agent_data) for agent_data in agents_config_list],
                **raw_config # Remaining top-level fields
            )
            self.logger.info(f"Configuration loaded from {self._config_path}")
        except FileNotFoundError:
            self.logger.error(f"Config file not found at {self._config_path}. Using default configuration.")
            ConfigLoader._config = FrameworkConfig()
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing YAML config file {self._config_path}: {e}. Using default configuration.")
            ConfigLoader._config = FrameworkConfig()
        except Exception as e:
            self.logger.error(f"An unexpected error occurred loading config from {self._config_path}: {e}. Using default configuration.", exc_info=True)
            ConfigLoader._config = FrameworkConfig()

    @classmethod
    def get_config(cls) -> FrameworkConfig:
        if cls._config is None:
            # Ensure the loader is initialized even if get_config is called directly
            cls() 
        return cls._config

def create_sample_config_file(file_path: str = "framework_config.yaml"):
    sample_config = {
        "name": "MyAgentFramework",
        "version": "1.1.0",
        "log_level": "DEBUG",
        "heartbeat_interval": 5,
        "message_queue_size": 2000,
        "security": {
            "enabled": True,
            "token_expiry_minutes": 120
        },
        "persistence": {
            "backend": "json",
            "connection_string": "data/framework_data.json"
        },
        "monitoring": {
            "enabled": True,
            "health_check_interval": 30
        },
        "backup": {
            "enabled": True,
            "storage_backend": "s3",
            "s3_bucket": "my-agent-backups"
        },
        "deployment": {
            "environment": "production"
        },
        "plugins": {
            "enabled": True,
            "plugin_paths": ["./plugins"]
        },
        "agents": [
            {
                "namespace": "agent.core",
                "name": "master_agent",
                "auto_start": True
            },
            {
                "namespace": "agent.dev",
                "name": "code_agent",
                "enabled": True,
                "max_concurrent_tasks": 5
            }
        ],
        "api_enabled": True,
        "api_port": 8001
    }

    try:
        os.makedirs(Path(file_path).parent, exist_ok=True)
        with open(file_path, 'w') as f:
            yaml.dump(sample_config, f, indent=2, sort_keys=False)
        print(f"Sample config file '{file_path}' created successfully.")
    except Exception as e:
        print(f"Error creating sample config file: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    sample_config_path = "framework_config.yaml"
    create_sample_config_file(sample_config_path)

    framework_config = ConfigLoader(config_path=sample_config_path).get_config()

    print("\nLoaded Framework Config:")
    print(f"Name: {framework_config.name}")
    print(f"Version: {framework_config.version}")
    print(f"Log Level: {framework_config.log_level}")
    print(f"Security Enabled: {framework_config.security.enabled}")
    print(f"Persistence Backend: {framework_config.persistence.backend}")
    if framework_config.agents:
        print(f"  First agent: {framework_config.agents[0].name} ({framework_config.agents[0].namespace})")

    # Clean up the sample file
    # os.remove(sample_config_path)
    # print(f"\nCleaned up {sample_config_path}")