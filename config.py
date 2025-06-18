# agentapi/config.py

import os
import yaml
import logging
from pathlib import Path
from typing import Optional

from agentapi.models.framework_models import FrameworkConfig

class ConfigLoader:
    """
    Clase para cargar y gestionar la configuración del framework.
    Carga desde un archivo YAML y proporciona el objeto FrameworkConfig.
    """
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
            # Buscar en directorio de trabajo actual y luego en el home del usuario
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
                self.logger.warning("No config file found at default paths. Using default FrameworkConfig.")

        self._config = self._load_config_from_file()

    def _load_config_from_file(self) -> FrameworkConfig:
        if self._config_path and self._config_path.exists():
            try:
                with open(self._config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                # Deserializar dict a FrameworkConfig, manejando sub-dataclases
                # Esto es una forma básica, para casos complejos se usaría `dataclasses_json` o similar
                framework_config = FrameworkConfig(**config_data)
                
                # Deserializar sub-configuraciones
                for key, cls in [
                    ("security", FrameworkConfig.__annotations__["security"]),
                    ("persistence", FrameworkConfig.__annotations__["persistence"]),
                    ("monitoring", FrameworkConfig.__annotations__["monitoring"]),
                    ("backup", FrameworkConfig.__annotations__["backup"]),
                    ("deployment", FrameworkConfig.__annotations__["deployment"]),
                    ("plugins", FrameworkConfig.__annotations__["plugins"]),
                ]:
                    if key in config_data and isinstance(config_data[key], dict):
                        setattr(framework_config, key, cls(**config_data[key]))

                # Deserializar lista de agentes
                if "agents" in config_data and isinstance(config_data["agents"], list):
                    framework_config.agents = [FrameworkConfig.__annotations__["agents"].__args__[0](**agent_data) 
                                                for agent_data in config_data["agents"]]

                self.logger.info(f"Configuration loaded from {self._config_path}")
                return framework_config
            except Exception as e:
                self.logger.error(f"Error loading configuration from {self._config_path}: {e}")
                self.logger.warning("Using default FrameworkConfig due to loading error.")
                return FrameworkConfig()
        else:
            self.logger.info("No configuration file specified or found. Using default FrameworkConfig.")
            return FrameworkConfig()

    @classmethod
    def load_framework_config(cls, config_path: Optional[str] = None) -> FrameworkConfig:
        """
        Carga la configuración del framework. 
        Si ya está cargada, devuelve la instancia existente.
        """
        if cls._config is None:
            # Crear una instancia para cargar la configuración
            _ = cls(config_path) 
        return cls._config

    @classmethod
    def get_config(cls) -> FrameworkConfig:
        """Obtiene la configuración actual del framework."""
        if cls._config is None:
            raise RuntimeError("Framework configuration has not been loaded yet. Call load_framework_config() first.")
        return cls._config

    @classmethod
    def save_framework_config(cls, config: FrameworkConfig, path: str) -> bool:
        """Guarda la configuración actual en un archivo YAML."""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config, f, default_flow_style=False)
            logging.getLogger("ConfigLoader").info(f"Configuration saved to {path}")
            return True
        except Exception as e:
            logging.getLogger("ConfigLoader").error(f"Error saving configuration to {path}: {e}")
            return False

# Funciones de utilidad para generación de configuración de ejemplo
def create_sample_config_file(output_path: str = "framework_config.yaml"):
    """
    Crea un archivo de configuración de ejemplo.
    """
    config = FrameworkConfig()
    
    # Añadir un agente de ejemplo
    config.agents.append(
        FrameworkConfig.__annotations__["agents"].__args__[0](
            namespace="agent.planning.strategist",
            name="default_strategist",
            auto_start=True
        )
    )
    config.agents.append(
        FrameworkConfig.__annotations__["agents"].__args__[0](
            namespace="agent.build.code.generator",
            name="default_code_generator",
            auto_start=True
        )
    )

    # Añadir configuración de seguridad de ejemplo
    config.security.admin_api_keys = {"admin": "a7b9c1d3e5f7a9b1c3d5e7f9a1b3c5d7"}
    config.security.jwt_secret = "your_strong_jwt_secret_here_12345"

    # Añadir configuración de persistencia de ejemplo
    config.persistence.backend = "sqlite"
    config.persistence.connection_string = "framework_data.db"
    config.persistence.auto_save_interval = 300

    # Añadir configuración de monitoreo de ejemplo
    config.monitoring.notification_channels = ["console"]
    config.monitoring.channel_settings = {
        "email": {"smtp_server": "smtp.example.com", "port": 587, "user": "user@example.com", "password": "pass"}
    }
    
    # Añadir configuración de backup de ejemplo
    config.backup.enabled = True
    config.backup.storage_backend = "local"
    config.backup.local_path = "./framework_backups"
    
    # Añadir configuración de plugins de ejemplo
    config.plugins.plugin_paths = ["./plugins", "./external_plugins"]
    config.plugins.active_plugins = {
        "example_plugin": {"enabled": True, "api_key": "some_api_key"}
    }
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False) # to_dict si usa dataclasses_json
        print(f"Sample configuration file created at: {output_path}")
    except Exception as e:
        print(f"Error creating sample config file: {e}")

if __name__ == "__main__":
    # Ejemplo de uso
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("Generating sample framework_config.yaml...")
    create_sample_config_file()

    print("\nLoading configuration...")
    # Cargar la configuración desde el archivo creado
    framework_config = ConfigLoader.load_framework_config(config_path="framework_config.yaml")

    print("\nLoaded Framework Config:")
    print(f"Name: {framework_config.name}")
    print(f"Version: {framework_config.version}")
    print(f"Log Level: {framework_config.log_level}")
    print(f"Message Queue Size: {framework_config.message_queue_size}")
    print(f"Security Enabled: {framework_config.security.enabled}")
    print(f"Persistence Backend: {framework_config.persistence.backend}")
    print(f"Monitoring Enabled: {framework_config.monitoring.enabled}")
    print(f"Backup Enabled: {framework_config.backup.enabled}")
    print(f"Number of agents in config: {len(framework_config.agents)}")
    if framework_config.agents:
        print(f"  First agent: {framework_config.agents[0].name} ({framework_config.agents[0].namespace})")

    # Puedes acceder a las sub-configuraciones directamente
    print(f"JWT Secret (first 10 chars): {framework_config.security.jwt_secret[:10]}...")
    print(f"Local Backup Path: {framework_config.backup.local_path}")