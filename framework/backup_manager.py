# framework/backup_manager.py

import asyncio
import logging
from typing import Optional
from datetime import datetime

from agentapi.models.backup_models import BackupConfig, BackupMetadata, BackupType, BackupStatus
from agentapi.models.general_models import Alert, AlertSeverity # For raising alerts

class BackupManager:
    def __init__(self, config: BackupConfig, monitoring_manager=None):
        self.config = config
        self.logger = logging.getLogger("BackupManager")
        self.monitoring_manager = monitoring_manager # Reference for raising alerts
        self._backup_schedule_task: Optional[asyncio.Task] = None # For potential scheduled backups

    async def initialize(self):
        """Initializes the BackupManager and starts any scheduled tasks."""
        if self.config.enabled and self.config.backup_interval_seconds > 0:
            self._backup_schedule_task = asyncio.create_task(self._scheduled_backup_loop())
            self.logger.info(f"Scheduled backups enabled every {self.config.backup_interval_seconds} seconds.")

    async def _scheduled_backup_loop(self):
        """Background task for scheduled backups."""
        while True:
            try:
                await asyncio.sleep(self.config.backup_interval_seconds)
                self.logger.info("Initiating scheduled backup...")
                await self.create_backup(BackupType.INCREMENTAL) # Or FULL, based on config logic
            except asyncio.CancelledError:
                self.logger.info("Scheduled backup loop cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error in scheduled backup loop: {e}", exc_info=True)


    async def create_backup(self, backup_type: BackupType = BackupType.FULL) -> Optional[BackupMetadata]:
        if not self.config.enabled:
            self.logger.info("Backup is disabled in configuration.")
            return None

        self.logger.info(f"Initiating {backup_type.value} backup...")
        metadata = BackupMetadata(
            backup_type=backup_type,
            status=BackupStatus.RUNNING,
            size_bytes=0,
            file_path=f"backup_{datetime.now().strftime('%Y%m%d%H%M%S')}.json",
            storage_backend=self.config.storage_backend
        )
        
        try:
            # Simulate backup process based on storage backend
            await asyncio.sleep(1) # Simulate I/O
            metadata.status = BackupStatus.COMPLETED
            metadata.completed_at = datetime.now()
            metadata.size_bytes = 1024 * 1024 # Dummy size
            self.logger.info(f"Backup {metadata.backup_id} completed successfully.")
            if self.monitoring_manager:
                await self.monitoring_manager.raise_alert(Alert(rule_name="BackupSuccess", severity=AlertSeverity.INFO, message=f"Backup {metadata.backup_id} completed."))
            return metadata
        except Exception as e:
            metadata.status = BackupStatus.FAILED
            self.logger.error(f"Backup {metadata.backup_id} failed: {e}", exc_info=True)
            if self.monitoring_manager:
                await self.monitoring_manager.raise_alert(Alert(rule_name="BackupFailure", severity=AlertSeverity.CRITICAL, message=f"Backup {metadata.backup_id} failed: {e}"))
            return None

    async def restore_from_backup(self, backup_id: str) -> bool:
        if not self.config.enabled:
            self.logger.info("Backup/Restore is disabled in configuration.")
            return False

        self.logger.info(f"Initiating restore from backup {backup_id}...")
        try:
            await asyncio.sleep(1) # Simulate I/O
            self.logger.info(f"Restore from backup {backup_id} completed.")
            if self.monitoring_manager:
                await self.monitoring_manager.raise_alert(Alert(rule_name="RestoreSuccess", severity=AlertSeverity.INFO, message=f"Restore from backup {backup_id} completed."))
            return True
        except Exception as e:
            self.logger.error(f"Restore from backup {backup_id} failed: {e}", exc_info=True)
            if self.monitoring_manager:
                await self.monitoring_manager.raise_alert(Alert(rule_name="RestoreFailure", severity=AlertSeverity.CRITICAL, message=f"Restore from backup {backup_id} failed: {e}"))
            return False

    async def shutdown(self):
        """Stops any scheduled backup tasks and performs cleanup."""
        self.logger.info("Shutting down BackupManager...")
        if self._backup_schedule_task:
            self._backup_schedule_task.cancel()
            try:
                await self._backup_schedule_task
            except asyncio.CancelledError:
                pass
            self.logger.info("Scheduled backup task cancelled.")
        self.logger.info("BackupManager shut down.")