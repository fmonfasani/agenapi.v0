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
    custom_settings: Dict[str, Any] = None
    
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
    health_check_interval: int = 60
    metrics_collection_interval: int = 10
    alerting_enabled: bool = True
    notification_channels: List[str] = field(default_factory=list) # e.g., ["slack", "email"]
    # Configuración específica para cada canal (ej. webhook de Slack, credenciales de correo)
    channel_settings: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BackupConfig:
    """Configuración del sistema de backup y recuperación"""
    enabled: bool = True
    storage_backend: str = "local" # 'local', 's3', 'azure', 'ssh'
    local_path: str = "./backups"
    s3_bucket: Optional[str] = None
    azure_container: Optional[str] = None
    backup_interval_hours: int = 24
    retention_days: int = 7

@dataclass
class DeploymentConfig:
    """Configuración del sistema de despliegue"""
    environment: str = "development" # 'development', 'staging', 'production', 'testing'
    strategy: str = "standalone" # 'standalone', 'docker', 'kubernetes', 'docker_compose'
    output_directory: str = "./deployments"
    docker_image_name: str = "agent-framework"
    kubernetes_namespace: str = "agents"
    # Otras configuraciones específicas del despliegue

@dataclass
class PluginConfig:
    """Configuración para el sistema de plugins"""
    enabled: bool = True
    plugin_paths: List[str] = field(default_factory=list) # Directorios donde buscar plugins
    active_plugins: Dict[str, Dict[str, Any]] = field(default_factory=dict) # {plugin_name: config}

@dataclass
class FrameworkConfig:
    """Configuración principal del framework"""
    name: str = "Autonomous Agent Framework"
    version: str = "1.0.0"
    log_level: str = "INFO"
    heartbeat_interval: int = 10 # segundos

    # Message bus settings
    message_queue_size: int = 1000
    message_timeout: int = 30
    enable_message_persistence: bool = False # Si el bus de mensajes guarda mensajes

    # Resource manager settings
    max_resources: int = 10000
    resource_cleanup_interval: int = 3600 # segundos

    # Sub-configuraciones para los sistemas
    security: SecurityConfig = field(default_factory=SecurityConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    backup: BackupConfig = field(default_factory=BackupConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    plugins: PluginConfig = field(default_factory=PluginConfig)
    
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
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    network_io_mb: Dict[str, float] = field(default_factory=dict)
    uptime_seconds: float = 0.0
    last_backup_time: Optional[datetime] = None
    last_restore_time: Optional[datetime] = None
    active_alerts: int = 0
    system_health_status: str = "unknown"