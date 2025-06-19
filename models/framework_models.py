# agentapi/models/framework_models.py
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum

@dataclass
class AgentConfig:
    namespace: str
    name: str
    enabled: bool = True
    auto_start: bool = True
    max_concurrent_tasks: int = 10
    heartbeat_interval: int = 30
    restart_on_failure: bool = True
    custom_settings: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SecurityConfig:
    enabled: bool = True
    jwt_secret: str = "your_super_secret_jwt_key"
    token_expiry_minutes: int = 60
    default_security_level: str = "internal"
    admin_api_keys: Dict[str, str] = field(default_factory=dict)

@dataclass
class PersistenceConfig:
    backend: str = "sqlite"
    connection_string: str = "framework.db"
    auto_save_interval: int = 60
    max_message_history: int = 1000
    enable_compression: bool = False
    backup_enabled: bool = True
    backup_interval: int = 3600

@dataclass
class MonitoringConfig:
    enabled: bool = True
    health_check_interval: int = 60
    metrics_collection_interval: int = 10
    alerting_enabled: bool = True
    notification_channels: List[str] = field(default_factory=list)
    channel_settings: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BackupConfig:
    enabled: bool = True
    storage_backend: str = "local"
    local_path: str = "./backups"
    s3_bucket: Optional[str] = None
    azure_container: Optional[str] = None
    backup_interval_hours: int = 24
    retention_days: int = 7

@dataclass
class DeploymentConfig:
    environment: str = "development"
    strategy: str = "standalone"
    output_directory: str = "./deployments"
    docker_image_name: str = "agent-framework"
    kubernetes_namespace: str = "agents"

@dataclass
class PluginConfig:
    enabled: bool = True
    plugin_paths: List[str] = field(default_factory=list)
    active_plugins: Dict[str, Dict[str, Any]] = field(default_factory=dict)

@dataclass
class FrameworkConfig:
    name: str = "Autonomous Agent Framework"
    version: str = "1.0.0"
    log_level: str = "INFO"
    heartbeat_interval: int = 10
    message_queue_size: int = 1000
    message_timeout: int = 30
    enable_message_persistence: bool = False
    max_resources: int = 10000
    resource_cleanup_interval: int = 3600
    security: SecurityConfig = field(default_factory=SecurityConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    backup: BackupConfig = field(default_factory=BackupConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    plugins: PluginConfig = field(default_factory=PluginConfig)
    agents: List[AgentConfig] = field(default_factory=list)
    api_enabled: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    dashboard_enabled: bool = True
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8080

@dataclass
class FrameworkMetrics:
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