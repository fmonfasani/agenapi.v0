import asyncio
import json
import tarfile
import shutil
import os
import hashlib
import logging
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import tempfile
import boto3
from azure.storage.blob import BlobServiceClient
import paramiko

from core.autonomous_agent_framework import AgentFramework, BaseAgent, AgentResource
from core.persistence_system import PersistenceManager

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
    parent_backup_id: Optional[str] = None
    restoration_info: Optional[Dict[str, Any]] = None

@dataclass
class BackupConfig:
    enabled: bool = True
    backup_interval_seconds: int = 3600
    backup_type: BackupType = BackupType.FULL
    storage_backend: StorageBackend = StorageBackend.LOCAL
    local_backup_dir: str = "./backups"
    retention_days: int = 7
    s3_config: Optional[Dict[str, Any]] = None
    azure_config: Optional[Dict[str, Any]] = None
    ssh_config: Optional[Dict[str, Any]] = None

class BackupEngine:
    def __init__(self, persistence_manager: PersistenceManager, config: Optional[BackupConfig] = None):
        self.persistence_manager = persistence_manager
        self.config = config or BackupConfig()
        self.logger = logging.getLogger("BackupEngine")
        self.backup_history: List[BackupMetadata] = []
        self._backup_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        Path(self.config.local_backup_dir).mkdir(parents=True, exist_ok=True)

    async def start(self):
        if self.config.enabled and not self._backup_task:
            self._stop_event.clear()
            self._backup_task = asyncio.create_task(self._backup_loop())
            self.logger.info("BackupEngine started.")

    async def stop(self):
        if self._backup_task:
            self._stop_event.set()
            self._backup_task.cancel()
            try:
                await self._backup_task
            except asyncio.CancelledError:
                pass
            self._backup_task = None
            self.logger.info("BackupEngine stopped.")

    async def _backup_loop(self):
        while not self._stop_event.is_set():
            try:
                await self.create_backup(self.config.backup_type)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error during scheduled backup: {e}")
            
            try:
                await asyncio.sleep(self.config.backup_interval_seconds)
            except asyncio.CancelledError:
                break

    async def create_backup(self, backup_type: BackupType = BackupType.FULL) -> Optional[BackupMetadata]:
        backup_id = f"backup_{datetime.now().strftime('%Y%m%d%H%M%S')}_{str(uuid.uuid4())[:8]}"
        temp_file = Path(tempfile.gettempdir()) / f"{backup_id}.tar.gz"
        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=backup_type,
            status=BackupStatus.RUNNING,
            created_at=datetime.now(),
            completed_at=None,
            size_bytes=0,
            checksum="",
            file_path="",
            storage_backend=self.config.storage_backend
        )
        self.backup_history.append(metadata)

        try:
            await self.persistence_manager.save_full_state(self.persistence_manager.framework)
            db_file_path = self.persistence_manager.config.connection_string
            await self._compress_data([db_file_path], temp_file)
            dest_path = await self._upload_to_storage(temp_file, backup_id)

            metadata.status = BackupStatus.COMPLETED
            metadata.completed_at = datetime.now()
            metadata.size_bytes = temp_file.stat().st_size
            metadata.checksum = self._calculate_checksum(temp_file)
            metadata.file_path = dest_path
            self.logger.info(f"Backup {backup_id} completed successfully to {dest_path}.")
            return metadata

        except Exception as e:
            self.logger.error(f"Failed to create backup {backup_id}: {e}")
            metadata.status = BackupStatus.FAILED
            metadata.completed_at = datetime.now()
            return None
        finally:
            if temp_file.exists():
                os.remove(temp_file)
            await self._apply_retention_policy()

    async def _compress_data(self, files_to_compress: List[str], output_path: Path):
        with tarfile.open(output_path, "w:gz") as tar:
            for file_path in files_to_compress:
                if Path(file_path).exists():
                    tar.add(file_path, arcname=Path(file_path).name)
                else:
                    self.logger.warning(f"File not found for backup: {file_path}")

    async def _upload_to_storage(self, file_path: Path, backup_id: str) -> str:
        if self.config.storage_backend == StorageBackend.LOCAL:
            dest_path = Path(self.config.local_backup_dir) / file_path.name
            shutil.copy(file_path, dest_path)
            return str(dest_path.resolve())
        elif self.config.storage_backend == StorageBackend.S3:
            s3_config = self.config.s3_config
            if not s3_config:
                raise ValueError("S3 configuration missing.")
            s3 = boto3.client('s3', region_name=s3_config["region"])
            bucket_name = s3_config["bucket_name"]
            s3_key = f"backups/{file_path.name}"
            s3.upload_file(str(file_path), bucket_name, s3_key)
            return f"s3://{bucket_name}/{s3_key}"
        elif self.config.storage_backend == StorageBackend.AZURE:
            azure_config = self.config.azure_config
            if not azure_config:
                raise ValueError("Azure configuration missing.")
            blob_service_client = BlobServiceClient.from_connection_string(azure_config["connection_string"])
            container_client = blob_service_client.get_container_client(azure_config["container_name"])
            blob_name = f"backups/{file_path.name}"
            with open(file_path, "rb") as data:
                container_client.upload_blob(name=blob_name, data=data, overwrite=True)
            return f"azure://{azure_config['container_name']}/{blob_name}"
        elif self.config.storage_backend == StorageBackend.SSH:
            ssh_config = self.config.ssh_config
            if not ssh_config:
                raise ValueError("SSH configuration missing.")
            transport = paramiko.Transport((ssh_config["host"], ssh_config.get("port", 22)))
            transport.connect(
                username=ssh_config["username"], 
                password=ssh_config.get("password"), 
                pkey=paramiko.RSAKey.from_private_key_file(ssh_config["key_file"])
            )
            sftp = paramiko.SFTPClient.from_transport(transport)
            remote_path = f"{ssh_config.get('remote_path', '/backups')}/{file_path.name}"
            sftp.put(str(file_path), remote_path)
            sftp.close()
            transport.close()
            return f"ssh://{ssh_config['host']}{remote_path}"
        else:
            raise ValueError(f"Unsupported storage backend: {self.config.storage_backend.value}")

    def _calculate_checksum(self, file_path: Path, algorithm: str = "sha256") -> str:
        hasher = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    async def restore_backup(self, backup_id: str) -> bool:
        metadata = next((b for b in self.backup_history if b.backup_id == backup_id), None)
        if not metadata:
            self.logger.error(f"Backup with ID {backup_id} not found in history.")
            return False
        if metadata.status != BackupStatus.COMPLETED:
            self.logger.warning(f"Backup {backup_id} is not in COMPLETED status. Cannot restore.")
            return False

        self.logger.info(f"Restoring backup {backup_id} from {metadata.file_path}...")
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            downloaded_path = await self._download_from_storage(metadata.file_path, temp_dir)
            if not downloaded_path:
                return False
            
            if self._calculate_checksum(downloaded_path) != metadata.checksum:
                self.logger.error(f"Checksum mismatch for backup {backup_id}. Possible corruption.")
                metadata.status = BackupStatus.CORRUPTED
                return False

            await self._decompress_data(downloaded_path, temp_dir)
            db_file_name = Path(self.persistence_manager.config.connection_string).name
            restored_db_path = temp_dir / db_file_name

            if not restored_db_path.exists():
                self.logger.error(f"Restored database file {restored_db_path} not found in temporary directory.")
                return False

            current_db_path = Path(self.persistence_manager.config.connection_string)
            await self.persistence_manager.close()
            
            if current_db_path.exists():
                shutil.move(current_db_path, str(current_db_path) + ".old")

            shutil.move(restored_db_path, current_db_path)
            await self.persistence_manager.initialize()
            
            if not await self.persistence_manager.load_full_state(self.persistence_manager.framework):
                self.logger.error(f"Failed to load full state into framework after restoring DB.")
                return False

            self.logger.info(f"Backup {backup_id} restored successfully.")
            return True

        except Exception as e:
            self.logger.error(f"Error during restoration of backup {backup_id}: {e}")
            return False
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    async def _download_from_storage(self, source_path: str, dest_dir: Path) -> Optional[Path]:
        file_name = Path(source_path).name
        dest_path = dest_dir / file_name

        if source_path.startswith("s3://"):
            s3_config = self.config.s3_config
            if not s3_config:
                raise ValueError("S3 configuration missing.")
            s3 = boto3.client('s3', region_name=s3_config["region"])
            bucket_name = source_path.split("//")[1].split("/")[0]
            s3_key = "/".join(source_path.split("/")[3:])
            s3.download_file(bucket_name, s3_key, str(dest_path))
        elif source_path.startswith("azure://"):
            azure_config = self.config.azure_config
            if not azure_config:
                raise ValueError("Azure configuration missing.")
            blob_service_client = BlobServiceClient.from_connection_string(azure_config["connection_string"])
            container_name = source_path.split("//")[1].split("/")[0]
            blob_name = "/".join(source_path.split("/")[3:])
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            with open(dest_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
        elif source_path.startswith("ssh://"):
            ssh_config = self.config.ssh_config
            if not ssh_config:
                raise ValueError("SSH configuration missing.")
            transport = paramiko.Transport((ssh_config["host"], ssh_config.get("port", 22)))
            transport.connect(
                username=ssh_config["username"], 
                password=ssh_config.get("password"), 
                pkey=paramiko.RSAKey.from_private_key_file(ssh_config["key_file"])
            )
            sftp = paramiko.SFTPClient.from_transport(transport)
            remote_path = "/".join(source_path.split("/")[2:])
            sftp.get(remote_path, str(dest_path))
            sftp.close()
            transport.close()
        elif Path(source_path).exists():
            shutil.copy(source_path, dest_path)
        else:
            self.logger.error(f"Unsupported or invalid source path for download: {source_path}")
            return None
        
        return dest_path

    async def _decompress_data(self, compressed_file_path: Path, output_dir: Path):
        with tarfile.open(compressed_file_path, "r:gz") as tar:
            tar.extractall(path=output_dir)

    def get_backup_history(self) -> List[BackupMetadata]:
        return sorted(self.backup_history, key=lambda b: b.created_at, reverse=True)

    async def _apply_retention_policy(self):
        if self.config.retention_days <= 0:
            return

        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
        backups_to_delete = [
            b for b in self.backup_history 
            if b.created_at < cutoff_date and b.status == BackupStatus.COMPLETED
        ]
        
        for backup in backups_to_delete:
            try:
                await self._delete_from_storage(backup.file_path, backup.storage_backend)
                self.backup_history.remove(backup)
            except Exception as e:
                self.logger.error(f"Failed to delete expired backup {backup.backup_id}: {e}")

    async def _delete_from_storage(self, path: str, backend: StorageBackend) -> bool:
        try:
            if backend == StorageBackend.LOCAL:
                if Path(path).exists():
                    os.remove(path)
            elif backend == StorageBackend.S3:
                s3_config = self.config.s3_config
                if not s3_config:
                    raise ValueError("S3 configuration missing.")
                s3 = boto3.client('s3', region_name=s3_config["region"])
                bucket_name = path.split("//")[1].split("/")[0]
                s3_key = "/".join(path.split("/")[3:])
                s3.delete_object(Bucket=bucket_name, Key=s3_key)
            elif backend == StorageBackend.AZURE:
                azure_config = self.config.azure_config
                if not azure_config:
                    raise ValueError("Azure configuration missing.")
                blob_service_client = BlobServiceClient.from_connection_string(azure_config["connection_string"])
                container_name = path.split("//")[1].split("/")[0]
                blob_name = "/".join(path.split("/")[3:])
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
                blob_client.delete_blob()
            elif backend == StorageBackend.SSH:
                ssh_config = self.config.ssh_config
                if not ssh_config:
                    raise ValueError("SSH configuration missing.")
                transport = paramiko.Transport((ssh_config["host"], ssh_config.get("port", 22)))
                transport.connect(
                    username=ssh_config["username"], 
                    password=ssh_config.get("password"), 
                    pkey=paramiko.RSAKey.from_private_key_file(ssh_config["key_file"])
                )
                sftp = paramiko.SFTPClient.from_transport(transport)
                remote_path = "/".join(path.split("/")[2:])
                sftp.remove(remote_path)
                sftp.close()
                transport.close()
            else:
                self.logger.warning(f"Unsupported storage backend for deletion: {backend.value}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error deleting backup from storage {path}: {e}")
            return False

class DisasterRecoveryOrchestrator:
    def __init__(self, framework: AgentFramework, persistence_manager: PersistenceManager, config: Optional[BackupConfig] = None):
        self.framework = framework
        self.persistence_manager = persistence_manager
        self.config = config or BackupConfig()
        self.backup_engine = BackupEngine(persistence_manager, self.config)
        self.logger = logging.getLogger("DROrchestrator")
        self._recovery_status: Dict[str, Any] = {"last_recovery_attempt": None, "status": "idle", "error": None}

    async def start(self):
        await self.backup_engine.start()

    async def stop(self):
        await self.backup_engine.stop()

    async def disaster_recovery_plan(self, scenario: str, backup_id: Optional[str] = None) -> Dict[str, Any]:
        self.logger.warning(f"Initiating disaster recovery plan for scenario: {scenario}")
        self._recovery_status = {"last_recovery_attempt": datetime.now().isoformat(), "status": "running", "error": None}
        
        try:
            await self.framework.stop()
            
            if not backup_id:
                latest_backup = next(
                    (b for b in self.backup_engine.get_backup_history() 
                     if b.status == BackupStatus.COMPLETED and b.backup_type == BackupType.FULL), None
                )
                if not latest_backup:
                    self._recovery_status["status"] = "failed"
                    self._recovery_status["error"] = "No suitable backup found for recovery."
                    self.logger.error(self._recovery_status["error"])
                    return {"success": False, "error": self._recovery_status["error"]}
                backup_id = latest_backup.backup_id

            restore_success = await self.backup_engine.restore_backup(backup_id)
            if not restore_success:
                self._recovery_status["status"] = "failed"
                self._recovery_status["error"] = f"Failed to restore backup {backup_id}."
                self.logger.error(self._recovery_status["error"])
                return {"success": False, "error": self._recovery_status["error"]}

            await self.framework.start()
            self._recovery_status["status"] = "completed"
            return {"success": True, "details": f"Restored from backup {backup_id}."}

        except Exception as e:
            self._recovery_status["status"] = "failed"
            self._recovery_status["error"] = str(e)
            self.logger.critical(f"Critical error during disaster recovery for scenario '{scenario}': {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def get_recovery_status(self) -> Dict[str, Any]:
        return self._recovery_status

async def backup_recovery_demo():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("🚀 Starting Backup & Recovery System Demo")
    print("="*50)

    backup_config = BackupConfig(
        enabled=True,
        backup_interval_seconds=10,
        backup_type=BackupType.FULL,
        local_backup_dir="./demo_backups",
        retention_days=0
    )

    if Path(backup_config.local_backup_dir).exists():
        shutil.rmtree(backup_config.local_backup_dir)
        print(f"🗑️ Cleaned up previous backup directory: {backup_config.local_backup_dir}")
    Path(backup_config.local_backup_dir).mkdir(parents=True, exist_ok=True)

    framework = AgentFramework()
    
    from core.persistence_system import PersistenceConfig
    persistence_config_dr = PersistenceConfig(
        backend="sqlite",
        connection_string="./demo_framework_dr.db",
        auto_save_interval=0
    )
    
    if Path(persistence_config_dr.connection_string).exists():
        Path(persistence_config_dr.connection_string).unlink()
        print(f"🗑️ Cleaned up previous DB: {persistence_config_dr.connection_string}")

    persistence_manager = PersistenceManager(framework, persistence_config_dr)
    await persistence_manager.initialize()

    dr_orchestrator = DisasterRecoveryOrchestrator(framework, persistence_manager, backup_config)
    await dr_orchestrator.start()

    print("\n⚙️ Backup & Recovery system initialized.")

    from core.models import AgentStatus, ResourceType

    class DemoAgent(BaseAgent):
        def __init__(self, agent_id: str, name: str, framework_instance: AgentFramework):
            super().__init__("demo.agent", name, framework_instance)
            self.id = agent_id
            self.status = AgentStatus.ACTIVE
            self.message_queue = asyncio.Queue()

        async def initialize(self) -> bool:
            return True
            
        async def process_message(self, message):
            await asyncio.sleep(0.01)
    
    await framework.start()

    agent1 = DemoAgent("dr-agent-001", "DR_Agent_A", framework)
    agent2 = DemoAgent("dr-agent-002", "DR_Agent_B", framework)
    
    await agent1.initialize()
    await agent2.initialize()
    await agent1.start()
    await agent2.start()
    await agent1.send_message(agent2.id, "ping", {"data": "hello"})
    
    initial_resource = AgentResource(
        type=ResourceType.DATA,
        name="important_data_1",
        namespace="data.sensitive",
        data={"value": 123, "description": "critical system data"},
        owner_agent_id=agent1.id
    )
    await framework.resource_manager.create_resource(initial_resource)

    print(f"   ✅ Initial state: {len(framework.registry.list_all_agents())} agents, {len(framework.resource_manager.list_all_resources())} resources.")
    
    print(f"\n2. Waiting for first scheduled full backup (every {backup_config.backup_interval_seconds} seconds)...")
    await asyncio.sleep(backup_config.backup_interval_seconds + 2)

    full_backups = [b for b in dr_orchestrator.backup_engine.get_backup_history() if b.backup_type == BackupType.FULL and b.status == BackupStatus.COMPLETED]
    if full_backups:
        latest_full_backup = full_backups[0]
        print(f"   ✅ Full backup created: {latest_full_backup.backup_id}")
        print(f"   📁 Size: {latest_full_backup.size_bytes} bytes")
    else:
        print("   ❌ No full backup created. Check logs.")
        await dr_orchestrator.stop()
        await framework.stop()
        await persistence_manager.close()
        return

    print("\n3. Simulating state changes (new agent, new resource)...")
    agent3 = DemoAgent("dr-agent-003", "DR_Agent_C", framework)
    await agent3.initialize()
    await agent3.start()

    new_resource = AgentResource(
        type=ResourceType.CODE,
        name="new_code_module",
        namespace="code.feature",
        data={"code_content": "print('New feature code')"},
        owner_agent_id=agent3.id
    )
    await framework.resource_manager.create_resource(new_resource)
    await agent1.send_message(agent3.id, "welcome", {"data": "new agent"})
    print(f"   ✅ Current state: {len(framework.registry.list_all_agents())} agents, {len(framework.resource_manager.list_all_resources())} resources.")

    print(f"\n4. Waiting for another scheduled backup...")
    await asyncio.sleep(backup_config.backup_interval_seconds + 2)
    
    incremental_backups = [b for b in dr_orchestrator.backup_engine.get_backup_history() if b.created_at > latest_full_backup.created_at and b.status == BackupStatus.COMPLETED]
    if incremental_backups:
        incremental_backup = incremental_backups[0]
        print(f"   ✅ Incremental backup created: {incremental_backup.backup_id}")
    else:
        incremental_backup = await dr_orchestrator.backup_engine.create_backup(BackupType.FULL)
        if incremental_backup:
            print(f"   ✅ Forcing another full backup: {incremental_backup.backup_id}")

    print(f"\n5. Disaster recovery simulation (restoring to initial state)...\n")
    print(f"   📉 Simulating system crash...\n")
    
    recovery_result = await dr_orchestrator.disaster_recovery_plan(
        "system_crash",
        backup_id=latest_full_backup.backup_id
    )
    
    if recovery_result["success"]:
        print(f"   ✅ Recovery successful")
        await asyncio.sleep(1)
        print(f"   📊 Current state: {len(framework.registry.list_all_agents())} agents, {len(framework.resource_manager.list_all_resources())} resources.")
        if not framework.registry.get_agent(agent3.id):
            print(f"   ✅ Agent {agent3.name} not found after recovery (as expected).")
        if not await framework.resource_manager.get_resource(new_resource.id):
            print(f"   ✅ Resource {new_resource.name} not found after recovery (as expected).")
    else:
        print(f"   ❌ Recovery failed: {recovery_result['error']}")
    
    print(f"\n6. Recovery system status:\n")
    status = dr_orchestrator.get_recovery_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    print(f"\n7. Backup history:\n")
    backup_history = dr_orchestrator.backup_engine.get_backup_history()
    
    for i, backup in enumerate(backup_history[:5], 1):
        print(f"   {i}. {backup.backup_id} ({backup.backup_type.value}) - {backup.status.value}")
        print(f"      Created: {backup.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"      File: {backup.file_path}")
        print(f"      Size: {backup.size_bytes} bytes")
        
    await dr_orchestrator.stop()
    await framework.stop()
    await persistence_manager.close()
    print("\nDemo finished.")

if __name__ == "__main__":
    asyncio.run(backup_recovery_demo())