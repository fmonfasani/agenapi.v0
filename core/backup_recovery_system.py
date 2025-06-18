import asyncio
import json
import gzip
import tarfile
import shutil
import os
import hashlib
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import tempfile
import boto3
from azure.storage.blob import BlobServiceClient
import paramiko

from core.autonomous_agent_framework import AgentFramework, BaseAgent
from core.persistence_system import PersistenceManager
from core.models import AgentResource, AgentStatus, AgentMessage # Importar de core.models

class BackupType(Enum):
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"

class BackupStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CORRUPTED = "corrupted"

class StorageBackend(Enum):
    LOCAL = "local"
    S3 = "s3"
    AZURE = "azure"
    SSH = "ssh"
    FTP = "ftp"

@dataclass
class BackupMetadata:
    backup_id: str
    backup_type: BackupType
    status: BackupStatus
    created_at: datetime
    completed_at: Optional[datetime]
    size_bytes: int
    checksum: str
    file_path: str
    storage_backend: StorageBackend
    retention_days: int
    parent_backup_id: Optional[str] = None
    restoration_info: Dict[str, Any] = field(default_factory=dict)
    metadata_version: str = "1.0"

class BackupEngine:
    def __init__(self, framework: AgentFramework, persistence_manager: PersistenceManager, config: Dict[str, Any]):
        self.framework = framework
        self.persistence_manager = persistence_manager
        self.config = config
        self.backup_dir = Path(config.get("backup_dir", "./backups"))
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger("BackupEngine")
        self.backup_history: List[BackupMetadata] = []
        self._load_backup_history()

    def _get_storage_backend(self, backend_type: StorageBackend):
        if backend_type == StorageBackend.LOCAL:
            return LocalStorageBackend()
        # Add other backends here
        raise ValueError(f"Unsupported storage backend: {backend_type}")

    def _load_backup_history(self):
        history_file = self.backup_dir / "backup_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
                    self.backup_history = [BackupMetadata(**item) for item in history_data]
                self.logger.info(f"Loaded {len(self.backup_history)} backup records from history.")
            except Exception as e:
                self.logger.error(f"Failed to load backup history: {e}")

    def _save_backup_history(self):
        history_file = self.backup_dir / "backup_history.json"
        try:
            with open(history_file, 'w') as f:
                json.dump([asdict(b) for b in self.backup_history], f, indent=4, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save backup history: {e}")

    async def create_full_backup(self) -> Optional[BackupMetadata]:
        backup_id = f"full_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        backup_path = self.backup_dir / f"{backup_id}.tar.gz"
        temp_dir = Path(tempfile.mkdtemp())
        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=BackupType.FULL,
            status=BackupStatus.RUNNING,
            created_at=datetime.now(),
            completed_at=None,
            size_bytes=0,
            checksum="",
            file_path=str(backup_path),
            storage_backend=StorageBackend.LOCAL,
            retention_days=self.config.get("retention_days", 7)
        )
        self.backup_history.append(metadata)
        self._save_backup_history()

        try:
            full_state = await self.persistence_manager.get_full_state()
            state_file = temp_dir / "framework_state.json"
            with open(state_file, 'w') as f:
                json.dump(full_state, f, indent=4)

            with tarfile.open(backup_path, "w:gz") as tar:
                tar.add(state_file, arcname="framework_state.json")
            
            size_bytes = backup_path.stat().st_size
            file_checksum = self._calculate_checksum(backup_path)

            metadata.status = BackupStatus.COMPLETED
            metadata.completed_at = datetime.now()
            metadata.size_bytes = size_bytes
            metadata.checksum = file_checksum
            self.logger.info(f"Full backup {backup_id} completed. Size: {size_bytes} bytes.")
            return metadata
        except Exception as e:
            self.logger.error(f"Full backup {backup_id} failed: {e}", exc_info=True)
            metadata.status = BackupStatus.FAILED
            metadata.completed_at = datetime.now()
            return None
        finally:
            shutil.rmtree(temp_dir)
            self._save_backup_history()

    async def create_incremental_backup(self, parent_backup_id: str) -> Optional[BackupMetadata]:
        backup_id = f"inc_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        backup_path = self.backup_dir / f"{backup_id}.tar.gz"
        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=BackupType.INCREMENTAL,
            status=BackupStatus.RUNNING,
            created_at=datetime.now(),
            completed_at=None,
            size_bytes=0,
            checksum="",
            file_path=str(backup_path),
            storage_backend=StorageBackend.LOCAL,
            retention_days=self.config.get("retention_days", 7),
            parent_backup_id=parent_backup_id
        )
        self.backup_history.append(metadata)
        self._save_backup_history()

        try:
            diff_state = await self.persistence_manager.get_incremental_state(parent_backup_id)
            
            temp_dir = Path(tempfile.mkdtemp())
            diff_file = temp_dir / "incremental_state.json"
            with open(diff_file, 'w') as f:
                json.dump(diff_state, f, indent=4)

            with tarfile.open(backup_path, "w:gz") as tar:
                tar.add(diff_file, arcname="incremental_state.json")

            size_bytes = backup_path.stat().st_size
            file_checksum = self._calculate_checksum(backup_path)

            metadata.status = BackupStatus.COMPLETED
            metadata.completed_at = datetime.now()
            metadata.size_bytes = size_bytes
            metadata.checksum = file_checksum
            self.logger.info(f"Incremental backup {backup_id} completed. Size: {size_bytes} bytes.")
            return metadata
        except Exception as e:
            self.logger.error(f"Incremental backup {backup_id} failed: {e}", exc_info=True)
            metadata.status = BackupStatus.FAILED
            metadata.completed_at = datetime.now()
            return None
        finally:
            shutil.rmtree(temp_dir)
            self._save_backup_history()

    async def restore_from_backup(self, backup_id: str) -> bool:
        self.logger.info(f"Attempting to restore from backup: {backup_id}")
        backup_meta = next((b for b in self.backup_history if b.backup_id == backup_id), None)

        if not backup_meta:
            self.logger.error(f"Backup {backup_id} not found in history.")
            return False
        if backup_meta.status != BackupStatus.COMPLETED:
            self.logger.error(f"Backup {backup_id} is not in COMPLETED status. Current: {backup_meta.status.value}")
            return False
        
        backup_path = Path(backup_meta.file_path)
        if not backup_path.exists():
            self.logger.error(f"Backup file not found at {backup_path}. Possibly moved or deleted.")
            return False
        
        if self._calculate_checksum(backup_path) != backup_meta.checksum:
            self.logger.error(f"Checksum mismatch for backup {backup_id}. File may be corrupted.")
            backup_meta.status = BackupStatus.CORRUPTED
            self._save_backup_history()
            return False

        temp_restore_dir = Path(tempfile.mkdtemp())
        try:
            with tarfile.open(backup_path, "r:gz") as tar:
                tar.extractall(path=temp_restore_dir)
            
            state_file = temp_restore_dir / "framework_state.json"
            if not state_file.exists():
                self.logger.error(f"Could not find framework_state.json in backup {backup_id}.")
                return False

            with open(state_file, 'r') as f:
                restored_state = json.load(f)
            
            await self.persistence_manager.restore_full_state(self.framework, restored_state)
            self.logger.info(f"Successfully restored from backup {backup_id}.")
            return True
        except Exception as e:
            self.logger.error(f"Failed to restore from backup {backup_id}: {e}", exc_info=True)
            return False
        finally:
            shutil.rmtree(temp_restore_dir)
            self._save_backup_history()


    def get_backup_history(self) -> List[BackupMetadata]:
        return sorted(self.backup_history, key=lambda x: x.created_at, reverse=True)

    def _calculate_checksum(self, file_path: Path, block_size=65536):
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            buf = f.read(block_size)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(block_size)
        return hasher.hexdigest()

    async def cleanup_old_backups(self):
        self.logger.info("Starting cleanup of old backups...")
        current_time = datetime.now()
        backups_to_keep = []
        cleaned_count = 0

        for backup in self.backup_history:
            if backup.status == BackupStatus.COMPLETED and \
               (current_time - backup.created_at).days > backup.retention_days:
                try:
                    backup_path = Path(backup.file_path)
                    if backup_path.exists():
                        os.remove(backup_path)
                        self.logger.info(f"Deleted old backup file: {backup_path}")
                    else:
                        self.logger.warning(f"Backup file {backup_path} not found during cleanup, removing from history.")
                    cleaned_count += 1
                except Exception as e:
                    self.logger.error(f"Error deleting backup file {backup.file_path}: {e}")
                    backups_to_keep.append(backup) # Keep if file deletion failed
            else:
                backups_to_keep.append(backup)
        
        self.backup_history = backups_to_keep
        self._save_backup_history()
        self.logger.info(f"Cleanup completed. Removed {cleaned_count} old backups.")

class LocalStorageBackend:
    def upload_file(self, local_path: Path, remote_path: Path):
        shutil.copy(local_path, remote_path)
        return True

    def download_file(self, remote_path: Path, local_path: Path):
        shutil.copy(remote_path, local_path)
        return True

class DisasterRecoveryOrchestrator:
    def __init__(self, framework: AgentFramework, persistence_manager: PersistenceManager, config: Dict[str, Any]):
        self.framework = framework
        self.persistence_manager = persistence_manager
        self.config = config
        self.backup_engine = BackupEngine(framework, persistence_manager, config.get("backup_engine", {}))
        self.logger = logging.getLogger("DisasterRecoveryOrchestrator")
        self.recovery_status: Dict[str, Any] = {
            "last_recovery_attempt": None,
            "last_recovery_success": False,
            "recovery_plan_active": False,
            "issues": []
        }
        self.recovery_plans: Dict[str, Callable] = {
            "system_crash": self._recover_system_crash,
            "data_corruption": self._recover_data_corruption,
        }
        self._recovery_task: Optional[asyncio.Task] = None
        self._stop_recovery_event = asyncio.Event()

    async def start_monitoring_backups(self):
        if not self._recovery_task:
            self._stop_recovery_event.clear()
            self._recovery_task = asyncio.create_task(self._monitor_backups_loop())
            self.logger.info("Disaster Recovery Orchestrator started monitoring backups.")

    async def stop_monitoring_backups(self):
        if self._recovery_task:
            self._stop_recovery_event.set()
            self._recovery_task.cancel()
            try:
                await self._recovery_task
            except asyncio.CancelledError:
                self.logger.info("DR Orchestrator monitoring task cancelled.")
            self._recovery_task = None
            self.logger.info("Disaster Recovery Orchestrator stopped.")

    async def _monitor_backups_loop(self):
        interval = self.config.get("monitor_interval_seconds", 3600)
        self.logger.info(f"DR Orchestrator will run backup cleanup every {interval} seconds.")
        while not self._stop_recovery_event.is_set():
            try:
                await self.backup_engine.cleanup_old_backups()
            except Exception as e:
                self.logger.error(f"Error during scheduled backup cleanup: {e}")
            await asyncio.sleep(interval)

    async def trigger_full_backup(self) -> Optional[BackupMetadata]:
        self.logger.info("Triggering full backup...")
        return await self.backup_engine.create_full_backup()

    async def trigger_incremental_backup(self, parent_backup_id: str) -> Optional[BackupMetadata]:
        self.logger.info(f"Triggering incremental backup based on {parent_backup_id}...")
        return await self.backup_engine.create_incremental_backup(parent_backup_id)

    async def disaster_recovery_plan(self, plan_name: str, **kwargs) -> Dict[str, Any]:
        self.logger.warning(f"Initiating disaster recovery plan: {plan_name}")
        self.recovery_status["last_recovery_attempt"] = datetime.now()
        self.recovery_status["recovery_plan_active"] = True
        self.recovery_status["issues"] = []

        plan_handler = self.recovery_plans.get(plan_name)
        if not plan_handler:
            self.recovery_status["last_recovery_success"] = False
            error_msg = f"Unknown recovery plan: {plan_name}"
            self.recovery_status["issues"].append(error_msg)
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}

        try:
            result = await plan_handler(**kwargs)
            self.recovery_status["last_recovery_success"] = result.get("success", False)
            return result
        except Exception as e:
            self.recovery_status["last_recovery_success"] = False
            error_msg = f"Recovery plan '{plan_name}' failed with error: {e}"
            self.recovery_status["issues"].append(error_msg)
            self.logger.error(error_msg, exc_info=True)
            return {"success": False, "error": error_msg}
        finally:
            self.recovery_status["recovery_plan_active"] = False

    async def _recover_system_crash(self) -> Dict[str, Any]:
        self.logger.info("Executing 'system_crash' recovery plan.")
        
        await self.framework.stop()
        self.logger.info("Framework stopped to facilitate recovery.")

        latest_full_backup = next((b for b in self.backup_engine.get_backup_history() if b.backup_type == BackupType.FULL and b.status == BackupStatus.COMPLETED), None)
        if not latest_full_backup:
            return {"success": False, "error": "No completed full backup found for system crash recovery."}

        self.logger.info(f"Attempting to restore from latest full backup: {latest_full_backup.backup_id}")
        restore_success = await self.backup_engine.restore_from_backup(latest_full_backup.backup_id)

        if restore_success:
            self.logger.info("System state restored. Restarting framework and agents.")
            await self.framework.start()
            agents_info = self.framework.registry.get_agent_info_list()
            for agent_info in agents_info:
                # In a real scenario, you'd need to re-instantiate and start agents
                # based on their types/configurations stored in the backup.
                # For this demo, we assume persistence manager reloads enough state.
                agent = self.framework.registry.get_agent(agent_info.id)
                if agent and agent.status != AgentStatus.ACTIVE:
                    self.logger.info(f"Attempting to restart agent {agent_info.name} (ID: {agent_info.id})")
                    await agent.start()

            self.logger.info("Framework and agents restarted after recovery.")
            return {"success": True, "message": "System crash recovery completed successfully."}
        else:
            return {"success": False, "error": "Failed to restore from backup during system crash recovery."}

    async def _recover_data_corruption(self) -> Dict[str, Any]:
        self.logger.info("Executing 'data_corruption' recovery plan.")
        
        self.logger.warning("Data corruption recovery typically involves more granular restoration. For demo, using latest full backup.")
        latest_full_backup = next((b for b in self.backup_engine.get_backup_history() if b.backup_type == BackupType.FULL and b.status == BackupStatus.COMPLETED), None)
        if not latest_full_backup:
            return {"success": False, "error": "No completed full backup found for data corruption recovery."}
        
        self.logger.info(f"Attempting to restore from backup: {latest_full_backup.backup_id}")
        restore_success = await self.backup_engine.restore_from_backup(latest_full_backup.backup_id)

        if restore_success:
            return {"success": True, "message": "Data corruption recovery completed successfully by restoring full state."}
        else:
            return {"success": False, "error": "Failed to restore data from backup."}

    def get_recovery_status(self) -> Dict[str, Any]:
        return self.recovery_status

async def dr_example():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger("BackupEngine").setLevel(logging.DEBUG)
    logging.getLogger("DisasterRecoveryOrchestrator").setLevel(logging.DEBUG)
    
    print("🚀 Starting Disaster Recovery System Demo")
    print("="*50)

    framework = AgentFramework()
    await framework.start()

    persistence_config = {
        "backend": "json",
        "connection_string": "dr_demo_state.json",
        "auto_save_interval": 5
    }
    persistence_manager = PersistenceManager(framework, persistence_config)
    await persistence_manager.initialize()

    dr_config = {
        "backup_engine": {
            "backup_dir": "./dr_backups",
            "retention_days": 1
        },
        "monitor_interval_seconds": 10 
    }
    dr_orchestrator = DisasterRecoveryOrchestrator(framework, persistence_manager, dr_config)
    await dr_orchestrator.start_monitoring_backups()

    print("\n1. Simulating agent activity and initial state save...")
    from specialized_agents import StrategistAgent, CodeGeneratorAgent
    
    strategist = StrategistAgent(name="strategist_dr", framework=framework)
    code_gen = CodeGeneratorAgent(name="code_gen_dr", framework=framework)

    await strategist.initialize()
    await code_gen.initialize()

    await strategist.start()
    await code_gen.start()

    await asyncio.sleep(1) # Give agents a moment to register

    test_resource = AgentResource(
        type=ResourceType.CODE,
        name="initial_codebase",
        namespace="project.backend",
        data={"main.py": "print('Hello, world!')"},
        owner_agent_id=code_gen.id
    )
    await framework.resource_manager.create_resource(test_resource)
    print(f"   Created initial resource: {test_resource.name}")

    print("\n2. Performing full backup...")
    full_backup = await dr_orchestrator.trigger_full_backup()
    if full_backup:
        print(f"   ✅ Full backup created: {full_backup.backup_id}")
        print(f"   📁 Size: {full_backup.size_bytes} bytes")
    else:
        print("   ❌ Full backup failed.")
        await framework.stop()
        await persistence_manager.close()
        await dr_orchestrator.stop_monitoring_backups()
        return

    print("\n3. Simulating further agent activity (change in state)...")
    await code_gen.send_message(
        strategist.id,
        "action.report.status",
        {"status": "Code generation for feature X completed."}
    )
    # Modify resource
    await framework.resource_manager.update_resource(
        test_resource.id,
        {"main.py": "print('Hello, updated world!')\nprint('New line added.')"}
    )
    print(f"   Modified resource: {test_resource.name}")
    print(f"   Current agents: {len(framework.registry.list_all_agents())}")

    print("\n4. Performing incremental backup...")
    incremental_backup = await dr_orchestrator.trigger_incremental_backup(full_backup.backup_id)
    if incremental_backup:
        print(f"   ✅ Incremental backup created: {incremental_backup.backup_id}")
        print(f"   📁 Size: {incremental_backup.size_bytes} bytes")
    else:
        print("   ❌ Incremental backup failed.")

    print(f"\n5. Disaster recovery simulation...")
    
    print(f"   📉 Simulating system crash (stopping framework and clearing state)...")
    await framework.stop()
    # Simulate data loss by clearing persistence
    await persistence_manager.backend.cleanup(older_than_days=0) 
    framework = AgentFramework() # Re-initialize framework to simulate crash
    await framework.start() # Start minimal framework for DR orchestrator to interact with

    recovery_result = await dr_orchestrator.disaster_recovery_plan("system_crash")
    
    if recovery_result["success"]:
        print(f"   ✅ Recovery successful")
        print(f"   📊 Current state: {len(framework.registry.list_all_agents())} agents")
        resource_after_recovery = await framework.resource_manager.get_resource(test_resource.id)
        if resource_after_recovery:
            print(f"   ✅ Resource '{resource_after_recovery.name}' recovered. Content start: {str(resource_after_recovery.data)[:30]}...")
        else:
            print(f"   ❌ Resource '{test_resource.name}' NOT recovered.")
    else:
        print(f"   ❌ Recovery failed: {recovery_result['error']}")

    print(f"\n6. Recovery system status:")
    status = dr_orchestrator.get_recovery_status()
    
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    print(f"\n7. Backup history:")
    backup_history = dr_orchestrator.backup_engine.get_backup_history()
    
    for i, backup in enumerate(backup_history[:5], 1):
        print(f"   {i}. {backup.backup_id} ({backup.backup_type.value}) - {backup.status.value}")
        print(f"      Created: {backup.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"      Size: {backup.size_bytes} bytes")
        print(f"      Parent: {backup.parent_backup_id if backup.parent_backup_id else 'N/A'}")

    finally:
        await dr_orchestrator.stop_monitoring_backups()
        await framework.stop()
        await persistence_manager.close()
        print("\n👋 Disaster Recovery Demo completed.")

if __name__ == "__main__":
    asyncio.run(dr_example())