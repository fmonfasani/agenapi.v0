# agentapi/interfaces/backup_interfaces.py

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from agentapi.models.framework_models import BackupConfig
from agentapi.models.backup_models import BackupMetadata, BackupType, BackupStatus, RestorePoint

class IBackupRecoverySystem(ABC):
    """
    Interfaz abstracta para el sistema de Backup y Recuperación.
    Define las operaciones de backup, restauración y gestión de puntos de recuperación.
    """

    @abstractmethod
    async def initialize(self, config: BackupConfig) -> bool:
        """Inicializa el sistema de backup y recuperación con la configuración dada."""
        pass

    @abstractmethod
    async def perform_backup(self, backup_type: BackupType = BackupType.FULL) -> Optional[BackupMetadata]:
        """Realiza un backup del estado del framework."""
        pass

    @abstractmethod
    async def restore_from_backup(self, backup_id: str) -> bool:
        """Restaura el estado del framework desde un backup específico."""
        pass

    @abstractmethod
    async def disaster_recovery_plan(self, scenario: str) -> Dict[str, Any]:
        """Ejecuta un plan de recuperación ante desastres para un escenario dado."""
        pass

    @abstractmethod
    def get_backup_history(self, limit: int = 100) -> List[BackupMetadata]:
        """Obtiene el historial de backups realizados."""
        pass

    @abstractmethod
    def get_restore_points(self, limit: int = 100) -> List[RestorePoint]:
        """Obtiene los puntos de restauración disponibles."""
        pass

    @abstractmethod
    async def get_recovery_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual del sistema de recuperación."""
        pass

    @abstractmethod
    async def cleanup_old_backups(self, retention_days: int) -> int:
        """Elimina backups antiguos según la política de retención."""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Cierra cualquier conexión o recurso abierto por el sistema de backup."""
        pass