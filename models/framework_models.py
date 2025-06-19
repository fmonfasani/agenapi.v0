# agentapi/models/framework_models.py

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum

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
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.custom_settings is None:
            self.custom_settings = {}

@dataclass
class SecurityConfig:
    """Configuración del sistema de seguridad"""
    enabled: bool = True
    jwt_secret: str = "your_super_secret_jwt_key"
    token_expiry_minutes: int = 60
    default_security_level: str = "internal"
    admin_api_keys: Dict[str, str] = field(default_factory=dict)
    # Ejemplo: {"admin_user": "hashed_password"}

@dataclass
class PersistenceConfig:
    """Configuración del sistema de persistencia"""
    backend: str = "sqlite" # 'sqlite', 'json', 'memory', 'postgresql', 'redis'
    connection_string: str = "framework.db"
    auto_save_interval: int = 60  # segundos
    max_message_history: int = 1000
    enable_compression: bool = False
    backup_enabled: bool = True
    backup_interval: int = 3600  # segundos

@dataclass
class MonitoringConfig:
    """Configuración del sistema de monitoreo"""
    enabled: bool = True
    collection_interval_seconds: int = 5
    alert_thresholds: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    notification_channels: List[str] = field(default_factory=list) # 'email', 'slack', etc.

@dataclass
class BackupConfig:
    """Configuración del sistema de backup y recuperación"""
    enabled: bool = True
    backup_interval_seconds: int = 86400 # 24 horas
    backup_retention_days: int = 7
    storage_backend: str = "local" # 'local', 's3', 'azure', 'ssh', 'ftp'
    local_backup_dir: str = "./backups"
    # Añadir más campos para S3, Azure, SSH, FTP credenciales/config si se implementan

@dataclass
class DeploymentConfig:
    """Configuración del sistema de despliegue continuo"""
    enabled: bool = True
    default_environment: str = "development"
    # Configuración por entorno, e.g., directorios de despliegue, scripts
    environments: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PluginConfig:
    """Configuración del sistema de plugins"""
    enabled: bool = True
    plugin_dirs: List[str] = field(default_factory=lambda: ["./plugins"])
    auto_load: List[str] = field(default_factory=list) # Lista de nombres de plugins a cargar automáticamente

# --- NUEVA CLASE DE CONFIGURACIÓN COGNITIVA ---
@dataclass
class CognitiveConfig:
    """Configuración para las capacidades cognitivas (LLM y Memoria)."""
    enabled: bool = True
    llm_provider: str = "openai" # 'openai', 'anthropic', 'local'
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1000
    memory_config: Dict[str, Any] = field(default_factory=dict) # Configuración para AgentMemorySystem

@dataclass
class FrameworkConfig:
    """Configuración principal del framework"""
    # Framework settings
    name: str = "Autonomous Agent Framework"
    version: str = "1.0.0"
    logging_level: str = "INFO" # e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    
    # Message bus settings
    message_queue_size: int = 1000
    message_timeout: int = 30 # seconds for message processing timeout
    enable_message_persistence: bool = False

    # Resource manager settings
    max_resources: int = 10000
    resource_cleanup_interval: int = 3600 # seconds

    # Sub-configuraciones para los sistemas
    security: SecurityConfig = field(default_factory=SecurityConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    backup: BackupConfig = field(default_factory=BackupConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    plugins: PluginConfig = field(default_factory=PluginConfig)
    cognitive: CognitiveConfig = field(default_factory=CognitiveConfig) # <--- NUEVO

    # Configuración de agentes (lista de AgentConfig)
    agents: List[AgentConfig] = field(default_factory=list)

    # API / Web Dashboard settings
    api_enabled: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    dashboard_enabled: bool = True
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8080

@dataclass
class FrameworkMetrics:
    """Métricas operacionales del framework"""
    total_agents: int = 0
    active_agents: int = 0
    total_messages_processed: int = 0
    total_resources: int = 0
    # Añadir más métricas aquí según sea necesario