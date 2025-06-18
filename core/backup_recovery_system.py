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

from core.autonomous_agent_framework import AgentFramework, BaseAgent, AgentResource
from core.persistence_system import PersistenceManager

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================================
# BACKUP MODELS
# ================================

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
    retention_policy: str
    associated_resources: List[str] = field(default_factory=list) # IDs de recursos/agentes asociados

    def to_dict(self):
        return {
            "backup_id": self.backup_id,
            "backup_type": self.backup_type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "file_path": self.file_path,
            "storage_backend": self.storage_backend.value,
            "retention_policy": self.retention_policy,
            "associated_resources": self.associated_resources
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            backup_id=data["backup_id"],
            backup_type=BackupType(data["backup_type"]),
            status=BackupStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            size_bytes=data["size_bytes"],
            checksum=data["checksum"],
            file_path=data["file_path"],
            storage_backend=StorageBackend(data["storage_backend"]),
            retention_policy=data["retention_policy"],
            associated_resources=data.get("associated_resources", [])
        )

# ================================
# STORAGE BACKENDS
# ================================

class StorageBackendInterface(ABC):
    @abstractmethod
    async def upload_file(self, local_path: Path, remote_path: str) -> bool:
        pass

    @abstractmethod
    async def download_file(self, remote_path: str, local_path: Path) -> bool:
        pass

    @abstractmethod
    async def delete_file(self, remote_path: str) -> bool:
        pass

    @abstractmethod
    async def list_files(self, prefix: str = "") -> List[str]:
        pass

class LocalStorageBackend(StorageBackendInterface):
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalStorageBackend initialized at {self.base_dir}")

    async def upload_file(self, local_path: Path, remote_path: str) -> bool:
        destination = self.base_dir / remote_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy(local_path, destination)
            logger.debug(f"Uploaded {local_path} to {destination}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to {destination}: {e}")
            return False

    async def download_file(self, remote_path: str, local_path: Path) -> bool:
        source = self.base_dir / remote_path
        try:
            shutil.copy(source, local_path)
            logger.debug(f"Downloaded {source} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download {source} to {local_path}: {e}")
            return False

    async def delete_file(self, remote_path: str) -> bool:
        target = self.base_dir / remote_path
        try:
            if target.exists():
                os.remove(target)
                logger.debug(f"Deleted {target}")
                return True
            logger.warning(f"File {target} not found for deletion.")
            return False
        except Exception as e:
            logger.error(f"Failed to delete {target}: {e}")
            return False
            
    async def list_files(self, prefix: str = "") -> List[str]:
        files = []
        for root, _, filenames in os.walk(self.base_dir / prefix):
            for filename in filenames:
                relative_path = Path(root) / filename
                files.append(str(relative_path.relative_to(self.base_dir)))
        return files

class S3StorageBackend(StorageBackendInterface):
    def __init__(self, bucket_name: str, region_name: str, aws_access_key_id: str, aws_secret_access_key: str):
        self.bucket_name = bucket_name
        self.s3 = boto3.client(
            's3',
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        logger.info(f"S3StorageBackend initialized for bucket {bucket_name} in {region_name}")

    async def upload_file(self, local_path: Path, remote_path: str) -> bool:
        try:
            await asyncio.to_thread(self.s3.upload_file, str(local_path), self.bucket_name, remote_path)
            logger.debug(f"Uploaded {local_path} to s3://{self.bucket_name}/{remote_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to S3: {e}")
            return False

    async def download_file(self, remote_path: str, local_path: Path) -> bool:
        try:
            await asyncio.to_thread(self.s3.download_file, self.bucket_name, remote_path, str(local_path))
            logger.debug(f"Downloaded s3://{self.bucket_name}/{remote_path} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download {remote_path} from S3: {e}")
            return False

    async def delete_file(self, remote_path: str) -> bool:
        try:
            await asyncio.to_thread(self.s3.delete_object, Bucket=self.bucket_name, Key=remote_path)
            logger.debug(f"Deleted s3://{self.bucket_name}/{remote_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {remote_path} from S3: {e}")
            return False
            
    async def list_files(self, prefix: str = "") -> List[str]:
        try:
            response = await asyncio.to_thread(self.s3.list_objects_v2, Bucket=self.bucket_name, Prefix=prefix)
            return [obj['Key'] for obj in response.get('Contents', [])]
        except Exception as e:
            logger.error(f"Failed to list files from S3 with prefix {prefix}: {e}")
            return []

class AzureBlobStorageBackend(StorageBackendInterface):
    def __init__(self, connection_string: str, container_name: str):
        self.connection_string = connection_string
        self.container_name = container_name
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)
        
        try:
            self.container_client.create_container()
            logger.info(f"Azure container {container_name} created or already exists.")
        except Exception as e:
            logger.warning(f"Could not create Azure container {container_name}: {e}")
        
        logger.info(f"AzureBlobStorageBackend initialized for container {container_name}")

    async def upload_file(self, local_path: Path, remote_path: str) -> bool:
        blob_client = self.container_client.get_blob_client(remote_path)
        try:
            with open(local_path, "rb") as data:
                await asyncio.to_thread(blob_client.upload_blob, data, overwrite=True)
            logger.debug(f"Uploaded {local_path} to Azure Blob {remote_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload {local_path} to Azure Blob: {e}")
            return False

    async def download_file(self, remote_path: str, local_path: Path) -> bool:
        blob_client = self.container_client.get_blob_client(remote_path)
        try:
            download_stream = await asyncio.to_thread(blob_client.download_blob)
            with open(local_path, "wb") as file:
                file.write(await asyncio.to_thread(download_stream.readall))
            logger.debug(f"Downloaded Azure Blob {remote_path} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download {remote_path} from Azure Blob: {e}")
            return False

    async def delete_file(self, remote_path: str) -> bool:
        blob_client = self.container_client.get_blob_client(remote_path)
        try:
            await asyncio.to_thread(blob_client.delete_blob)
            logger.debug(f"Deleted Azure Blob {remote_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {remote_path} from Azure Blob: {e}")
            return False
            
    async def list_files(self, prefix: str = "") -> List[str]:
        try:
            blobs = await asyncio.to_thread(self.container_client.list_blobs, name_starts_with=prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            logger.error(f"Failed to list files from Azure Blob with prefix {prefix}: {e}")
            return []

# ================================
# BACKUP ENGINE
# ================================

class BackupEngine:
    def __init__(self, persistence_manager: PersistenceManager, storage_backend: StorageBackendInterface, backup_dir: Path = Path("./backups")):
        self.persistence_manager = persistence_manager
        self.storage_backend = storage_backend
        self.backup_dir = backup_dir
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.backup_history: List[BackupMetadata] = []
        self._load_backup_history()
        logger.info(f"BackupEngine initialized with storage backend: {type(storage_backend).__name__}")

    def _generate_checksum(self, file_path: Path) -> str:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _compress_directory(self, source_dir: Path, output_filepath: Path) -> Path:
        output_filepath_tar = output_filepath.with_suffix(".tar")
        with tarfile.open(output_filepath_tar, "w") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))
        
        # Comprimir el tar con gzip
        with open(output_filepath_tar, 'rb') as f_in:
            with gzip.open(output_filepath, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        os.remove(output_filepath_tar) # Eliminar el archivo .tar sin comprimir
        return output_filepath

    def _decompress_and_extract_directory(self, compressed_filepath: Path, output_dir: Path):
        # Descomprimir gzip
        temp_tar_path = compressed_filepath.with_suffix(".tar")
        with gzip.open(compressed_filepath, 'rb') as f_in:
            with open(temp_tar_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Extraer tar
        with tarfile.open(temp_tar_path, "r") as tar:
            tar.extractall(path=output_dir.parent) # Extrae el contenido del tar, que es el directorio

        os.remove(temp_tar_path) # Eliminar el archivo .tar temporal

    def _load_backup_history(self):
        history_file = self.backup_dir / "backup_history.json"
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.backup_history = [BackupMetadata.from_dict(item) for item in data]
                logger.info(f"Loaded {len(self.backup_history)} backup records from history.")
            except Exception as e:
                logger.error(f"Failed to load backup history: {e}. Starting with empty history.")
                self.backup_history = []
        else:
            logger.info("No backup history file found. Starting with empty history.")

    def _save_backup_history(self):
        history_file = self.backup_dir / "backup_history.json"
        try:
            with open(history_file, 'w') as f:
                json.dump([asdict(meta) for meta in self.backup_history], f, indent=4)
            logger.debug("Backup history saved.")
        except Exception as e:
            logger.error(f"Failed to save backup history: {e}")

    async def perform_full_backup(self, framework: AgentFramework, retention_policy: str = "7-days") -> Optional[BackupMetadata]:
        backup_id = f"full_backup_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        temp_backup_path = self.backup_dir / f"{backup_id}.tar.gz"
        
        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=BackupType.FULL,
            status=BackupStatus.PENDING,
            created_at=datetime.now(),
            completed_at=None,
            size_bytes=0,
            checksum="",
            file_path=str(temp_backup_path),
            storage_backend=self.storage_backend.__class__.__name__, # Store class name for identification
            retention_policy=retention_policy,
            associated_resources=[agent.id for agent in framework.registry.list_all_agents()] + \
                                 [res.id for res in framework.resource_manager.list_all_resources()]
        )
        self.backup_history.append(metadata)
        self._save_backup_history()

        logger.info(f"Starting full backup {backup_id}...")
        try:
            # 1. Guardar el estado completo del framework a un directorio temporal
            temp_state_dir = Path(tempfile.mkdtemp(prefix="framework_backup_"))
            await self.persistence_manager.save_full_state(framework, base_path=temp_state_dir)
            logger.debug(f"Framework state saved to temporary directory: {temp_state_dir}")

            # 2. Comprimir el directorio
            compressed_file = self._compress_directory(temp_state_dir, temp_backup_path)
            
            # 3. Calcular checksum
            checksum = self._generate_checksum(compressed_file)
            
            # 4. Subir al backend de almacenamiento
            remote_path = f"full/{compressed_file.name}"
            upload_success = await self.storage_backend.upload_file(compressed_file, remote_path)

            if upload_success:
                metadata.status = BackupStatus.COMPLETED
                metadata.completed_at = datetime.now()
                metadata.size_bytes = compressed_file.stat().st_size
                metadata.checksum = checksum
                metadata.file_path = remote_path # Actualizar a la ruta remota
                logger.info(f"Full backup {backup_id} completed and uploaded to {remote_path}.")
            else:
                metadata.status = BackupStatus.FAILED
                logger.error(f"Full backup {backup_id} failed during upload.")
        except Exception as e:
            metadata.status = BackupStatus.FAILED
            logger.error(f"Full backup {backup_id} failed: {e}")
        finally:
            # Limpiar archivos temporales
            if temp_backup_path.exists():
                os.remove(temp_backup_path)
            if temp_state_dir.exists():
                shutil.rmtree(temp_state_dir)
            self._save_backup_history()
        
        return metadata

    async def perform_incremental_backup(self, framework: AgentFramework, base_backup_id: str, retention_policy: str = "30-days") -> Optional[BackupMetadata]:
        # Para un backup incremental real, necesitaríamos comparar el estado actual con el último backup completo
        # y solo guardar los cambios. Esto es una simulación simplificada.
        
        last_full_backup = next((b for b in self.backup_history if b.backup_id == base_backup_id and b.backup_type == BackupType.FULL and b.status == BackupStatus.COMPLETED), None)
        if not last_full_backup:
            logger.error(f"Base full backup {base_backup_id} not found or not completed for incremental backup.")
            return None

        backup_id = f"inc_backup_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        temp_backup_path = self.backup_dir / f"{backup_id}.tar.gz"

        metadata = BackupMetadata(
            backup_id=backup_id,
            backup_type=BackupType.INCREMENTAL,
            status=BackupStatus.PENDING,
            created_at=datetime.now(),
            completed_at=None,
            size_bytes=0,
            checksum="",
            file_path=str(temp_backup_path),
            storage_backend=self.storage_backend.__class__.__name__,
            retention_policy=retention_policy,
            associated_resources=[agent.id for agent in framework.registry.list_all_agents()] + \
                                 [res.id for res in framework.resource_manager.list_all_resources()]
        )
        self.backup_history.append(metadata)
        self._save_backup_history()

        logger.info(f"Starting incremental backup {backup_id} based on {base_backup_id}...")
        try:
            # Simulación: guardamos solo un subconjunto de datos o cambios recientes
            temp_incremental_dir = Path(tempfile.mkdtemp(prefix="framework_incremental_"))
            # Aquí se implementaría la lógica para guardar solo los cambios incrementales
            # Por simplicidad, guardaremos solo un archivo de "cambios"
            changes_file = temp_incremental_dir / "incremental_changes.json"
            with open(changes_file, "w") as f:
                json.dump({"timestamp": datetime.now().isoformat(), "description": "Simulated incremental changes"}, f)
            logger.debug(f"Simulated incremental changes saved to {changes_file}")

            compressed_file = self._compress_directory(temp_incremental_dir, temp_backup_path)
            checksum = self._generate_checksum(compressed_file)
            
            remote_path = f"incremental/{compressed_file.name}"
            upload_success = await self.storage_backend.upload_file(compressed_file, remote_path)

            if upload_success:
                metadata.status = BackupStatus.COMPLETED
                metadata.completed_at = datetime.now()
                metadata.size_bytes = compressed_file.stat().st_size
                metadata.checksum = checksum
                metadata.file_path = remote_path
                logger.info(f"Incremental backup {backup_id} completed and uploaded to {remote_path}.")
            else:
                metadata.status = BackupStatus.FAILED
                logger.error(f"Incremental backup {backup_id} failed during upload.")
        except Exception as e:
            metadata.status = BackupStatus.FAILED
            logger.error(f"Incremental backup {backup_id} failed: {e}")
        finally:
            if temp_backup_path.exists():
                os.remove(temp_backup_path)
            if temp_incremental_dir.exists():
                shutil.rmtree(temp_incremental_dir)
            self._save_backup_history()
        
        return metadata

    async def restore_backup(self, backup_id: str, framework: AgentFramework) -> bool:
        metadata = next((b for b in self.backup_history if b.backup_id == backup_id and b.status == BackupStatus.COMPLETED), None)
        if not metadata:
            logger.error(f"Backup {backup_id} not found or not completed.")
            return False

        logger.info(f"Starting restore from backup {backup_id}...")
        temp_download_path = self.backup_dir / Path(metadata.file_path).name
        temp_restore_dir = Path(tempfile.mkdtemp(prefix="framework_restore_"))

        try:
            # 1. Descargar el archivo de backup
            download_success = await self.storage_backend.download_file(metadata.file_path, temp_download_path)
            if not download_success:
                logger.error(f"Failed to download backup file {metadata.file_path}.")
                return False

            # 2. Verificar checksum
            downloaded_checksum = self._generate_checksum(temp_download_path)
            if downloaded_checksum != metadata.checksum:
                logger.error(f"Checksum mismatch for backup {backup_id}. File might be corrupted.")
                return False
            logger.debug("Checksum verified successfully.")

            # 3. Descomprimir y extraer
            self._decompress_and_extract_directory(temp_download_path, temp_restore_dir)
            logger.debug(f"Backup extracted to {temp_restore_dir.parent}") # _decompress_and_extract_directory extrae al parent

            # 4. Cargar el estado restaurado en el framework
            # Asumimos que el directorio restaurado contiene el archivo framework_state.json u otros archivos necesarios
            # para la PersistenceManager.
            restored_state_path = temp_restore_dir.parent / Path(temp_restore_dir).name # Ruta real donde se extrajo el contenido
            
            # Detener el framework para una restauración limpia si está corriendo
            if framework.is_running:
                await framework.stop()

            # Esto es clave: la PersistenceManager debe ser capaz de cargar el estado desde un path dado
            # Si el framework usa una base de datos, esto implicaría restaurar la base de datos desde los archivos.
            # Aquí, para la demo, simularemos la carga desde los archivos guardados en el temp_restore_dir.
            # NOTA: La implementación real de persistence_manager.load_full_state() necesitaría ser robusta para manejar esto.
            # Por ahora, simplemente apuntamos al path donde se espera que PersistenceManager busque los datos.
            
            # Para la demo, simularemos que PersistenceManager carga de este path
            # En un sistema real, esto podría implicar reemplazar la base de datos o recargar estados internos.
            success = await self.persistence_manager.load_full_state(framework, base_path=restored_state_path)
            
            if success:
                logger.info(f"Successfully restored framework state from backup {backup_id}.")
                await framework.start() # Reiniciar el framework
                return True
            else:
                logger.error(f"Failed to load framework state from restored data for backup {backup_id}.")
                return False

        except Exception as e:
            logger.error(f"Error during restore of backup {backup_id}: {e}")
            return False
        finally:
            if temp_download_path.exists():
                os.remove(temp_download_path)
            if temp_restore_dir.exists(): # Puede ser que rmtree falle si el path no es un directorio, manejar con cuidado
                try:
                    shutil.rmtree(temp_restore_dir)
                except OSError as e:
                    logger.warning(f"Error removing temporary restore directory {temp_restore_dir}: {e}")
            # El contenido del tar se extrae al parent del temp_restore_dir, así que hay que limpiar ese también
            if restored_state_path.exists() and restored_state_path.is_dir():
                try:
                    shutil.rmtree(restored_state_path)
                except OSError as e:
                    logger.warning(f"Error removing extracted restore directory {restored_state_path}: {e}")


    def get_backup_history(self) -> List[BackupMetadata]:
        return sorted(self.backup_history, key=lambda x: x.created_at, reverse=True)

    async def prune_old_backups(self, retention_days: int = 30) -> int:
        now = datetime.now()
        pruned_count = 0
        backups_to_keep = []
        
        for backup in self.backup_history:
            if backup.status == BackupStatus.COMPLETED and (now - backup.created_at) > timedelta(days=retention_days):
                logger.info(f"Pruning old backup: {backup.backup_id} (created {backup.created_at.strftime('%Y-%m-%d')})")
                try:
                    await self.storage_backend.delete_file(backup.file_path)
                    pruned_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete backup file {backup.file_path} from storage: {e}. Keeping record.")
                    backups_to_keep.append(backup)
            else:
                backups_to_keep.append(backup)
        
        self.backup_history = backups_to_keep
        self._save_backup_history()
        logger.info(f"Pruned {pruned_count} old backups.")
        return pruned_count

# ================================
# DISASTER RECOVERY ORCHESTRATOR
# ================================

class DisasterRecoveryOrchestrator:
    def __init__(self, framework: AgentFramework, persistence_manager: PersistenceManager, storage_backend: StorageBackendInterface):
        self.framework = framework
        self.persistence_manager = persistence_manager
        self.backup_engine = BackupEngine(persistence_manager, storage_backend)
        self.recovery_plans: Dict[str, Callable] = {
            "system_crash": self._recover_from_system_crash,
            "data_corruption": self._recover_from_data_corruption,
            "agent_failure": self._recover_agent_failure
        }
        self.last_recovery_attempt: Optional[datetime] = None
        self.recovery_success_count: int = 0
        self.recovery_fail_count: int = 0
        logger.info("DisasterRecoveryOrchestrator initialized.")

    async def _recover_from_system_crash(self) -> Dict[str, Any]:
        logger.info("Executing recovery plan: system_crash")
        # 1. Identificar el último backup completo exitoso
        latest_full_backup = next((b for b in self.backup_engine.get_backup_history() if b.backup_type == BackupType.FULL and b.status == BackupStatus.COMPLETED), None)
        
        if not latest_full_backup:
            logger.error("No completed full backup found for system crash recovery.")
            return {"success": False, "error": "No completed full backup available."}

        logger.info(f"Attempting to restore from latest full backup: {latest_full_backup.backup_id}")
        # 2. Restaurar el framework desde este backup
        restore_success = await self.backup_engine.restore_backup(latest_full_backup.backup_id, self.framework)
        
        if restore_success:
            logger.info("System crash recovery completed successfully.")
            return {"success": True, "details": f"Restored from backup {latest_full_backup.backup_id}"}
        else:
            logger.error("System crash recovery failed.")
            return {"success": False, "error": "Failed to restore from backup."}

    async def _recover_from_data_corruption(self) -> Dict[str, Any]:
        logger.info("Executing recovery plan: data_corruption")
        # Simula encontrar corrupción en recursos o mensajes
        corrupted_resources = self.framework.resource_manager.find_resources_by_type(ResourceType.DATA)[:1]
        if corrupted_resources:
            logger.warning(f"Simulating detection of data corruption in resource: {corrupted_resources[0].name}")
            # Intentar restaurar solo los recursos afectados, o un backup más reciente si es posible
            latest_backup_with_resources = next((
                b for b in self.backup_engine.get_backup_history() 
                if b.status == BackupStatus.COMPLETED and any(res_id in b.associated_resources for res_id in [res.id for res in corrupted_resources])
            ), None)
            
            if latest_backup_with_resources:
                logger.info(f"Attempting to restore from backup {latest_backup_with_resources.backup_id} due to data corruption.")
                restore_success = await self.backup_engine.restore_backup(latest_backup_with_resources.backup_id, self.framework)
                if restore_success:
                    logger.info("Data corruption recovery completed successfully.")
                    return {"success": True, "details": f"Restored from backup {latest_backup_with_resources.backup_id}"}
            logger.error("Data corruption recovery failed: Could not find suitable backup or restore failed.")
            return {"success": False, "error": "No suitable backup found or restore failed for data corruption."}
        else:
            logger.info("No corrupted resources simulated. No specific data corruption recovery performed.")
            return {"success": True, "details": "No data corruption detected/simulated."}

    async def _recover_agent_failure(self) -> Dict[str, Any]:
        logger.info("Executing recovery plan: agent_failure")
        # Simular un agente fallido
        failed_agent = next((a for a in self.framework.registry.list_all_agents() if a.status == BaseAgent.AgentStatus.ERROR), None)
        if not failed_agent:
            # Si no hay agentes en ERROR, elegimos uno al azar para simular
            all_agents = self.framework.registry.list_all_agents()
            if all_agents:
                failed_agent = all_agents[0]
                failed_agent.status = BaseAgent.AgentStatus.ERROR
                logger.warning(f"Simulating failure of agent: {failed_agent.name} ({failed_agent.id})")
            else:
                logger.info("No agents to simulate failure for.")
                return {"success": True, "details": "No agent failure simulated."}

        logger.info(f"Attempting to recover failed agent: {failed_agent.name} ({failed_agent.id})")
        # Estrategia: reiniciar el agente y recargar su estado si es posible
        try:
            await failed_agent.shutdown()
            self.framework.registry.unregister_agent(failed_agent.id)
            
            # Para la recuperación de agente, la idea es recrearlo y recargar su último estado conocido
            # Esto requeriría que PersistenceManager pueda cargar el estado de un agente específico
            # Aquí, lo simplificamos creando una nueva instancia del mismo tipo
            
            # Buscar la clase original del agente
            original_agent_class = self.framework.agent_factory._agent_classes.get(failed_agent.namespace)
            if original_agent_class:
                new_agent = await self.framework.agent_factory.create_agent(failed_agent.namespace, failed_agent.name, original_agent_class)
                if new_agent:
                    # Intentar cargar el estado persistido para el agente por su ID anterior o por su nombre/namespace
                    # Esto es un placeholder; la PersistenceManager necesita esta capacidad
                    # await self.persistence_manager.load_agent_state(new_agent) 
                    logger.info(f"Agent {failed_agent.name} recovered successfully by recreation.")
                    return {"success": True, "details": f"Agent {failed_agent.name} ({failed_agent.id}) recovered."}
                else:
                    logger.error(f"Failed to recreate agent {failed_agent.name}.")
            else:
                logger.error(f"Agent class for namespace {failed_agent.namespace} not found to recreate agent {failed_agent.name}.")

        except Exception as e:
            logger.error(f"Error during agent failure recovery for {failed_agent.name}: {e}")
            
        return {"success": False, "error": f"Failed to recover agent {failed_agent.name}."}


    async def disaster_recovery_plan(self, scenario: str) -> Dict[str, Any]:
        self.last_recovery_attempt = datetime.now()
        recovery_func = self.recovery_plans.get(scenario)
        if recovery_func:
            logger.info(f"Initiating disaster recovery for scenario: {scenario}")
            result = await recovery_func()
            if result["success"]:
                self.recovery_success_count += 1
                logger.info(f"Disaster recovery for {scenario} finished successfully.")
            else:
                self.recovery_fail_count += 1
                logger.error(f"Disaster recovery for {scenario} failed.")
            return result
        else:
            logger.error(f"Unknown disaster recovery scenario: {scenario}")
            self.recovery_fail_count += 1
            return {"success": False, "error": f"Unknown scenario: {scenario}"}

    def get_recovery_status(self) -> Dict[str, Any]:
        return {
            "last_recovery_attempt": self.last_recovery_attempt.isoformat() if self.last_recovery_attempt else "N/A",
            "successful_recoveries": self.recovery_success_count,
            "failed_recoveries": self.recovery_fail_count,
            "total_backups": len(self.backup_engine.get_backup_history()),
            "last_full_backup": self.backup_engine.get_backup_history()[0].created_at.isoformat() if self.backup_engine.get_backup_history() else "N/A"
        }

# ================================
# DEMO USAGE
# ================================

async def demo_backup_recovery():
    # Inicializar el framework de agentes
    framework = AgentFramework("BackupRecoveryDemoFramework")
    await framework.start()

    # Inicializar el sistema de persistencia (usando un backend de archivo JSON para la demo)
    persistence_config = type('PersistenceConfig', (), {
        'backend': 'json', # Usaremos un directorio para simular la persistencia
        'connection_string': 'demo_persistence_data', # Esto será un directorio
        'auto_save_interval': 60,
        'max_message_history': 1000,
        'enable_compression': False,
        'backup_enabled': False,
        'backup_interval': 3600
    })()
    
    # Asegúrate de que el connection_string sea un Path para PersistenceManager en esta demo
    persistence_config.connection_string = Path(persistence_config.connection_string)

    persistence_manager = PersistenceManager(framework)
    await persistence_manager.initialize(persistence_config)

    # Inicializar el backend de almacenamiento (local para la demo)
    local_storage_path = Path("./demo_backups_storage")
    storage_backend = LocalStorageBackend(local_storage_path)

    # Inicializar el orquestador de DR
    dr_orchestrator = DisasterRecoveryOrchestrator(framework, persistence_manager, storage_backend)

    try:
        # Demo 1: Crear algunos agentes y recursos para simular un estado
        logger.info("\n1. Simulating initial system state with agents and resources...")
        
        class DemoAgent(BaseAgent):
            def __init__(self, name: str, framework: AgentFramework):
                super().__init__("agent.demo", name, framework)
            async def handle_message(self, message):
                logger.info(f"DemoAgent {self.name} received message: {message.action}")

        framework.agent_factory.register_agent_class("agent.demo", DemoAgent)
        agent1 = await framework.agent_factory.create_agent("agent.demo", "AgentAlpha")
        agent2 = await framework.agent_factory.create_agent("agent.demo", "AgentBeta")

        resource1 = AgentResource(name="config_file", namespace="resource.config", data={"key": "value"}, owner_agent_id=agent1.id)
        resource2 = AgentResource(name="user_data_db", namespace="resource.data.users", data=[{"id": 1}], owner_agent_id=agent2.id)
        await framework.resource_manager.create_resource(resource1)
        await framework.resource_manager.create_resource(resource2)
        
        # Simular algunos mensajes
        if agent1 and agent2:
            await agent1.send_message(agent2.id, "action.test", {"data": "hello"})
            await agent2.send_message(agent1.id, "action.response", {"status": "ok"})
            await asyncio.sleep(0.1) # Dar tiempo para que se procesen los mensajes
        
        logger.info("Initial state created.")
        logger.info(f"   📊 Current agents: {len(framework.registry.list_all_agents())}")
        logger.info(f"   📂 Current resources: {len(framework.resource_manager.list_all_resources())}")

        # Demo 2: Realizar un backup completo
        logger.info("\n2. Performing full backup...")
        full_backup = await dr_orchestrator.backup_engine.perform_full_backup(framework, retention_policy="30-days")
        if full_backup and full_backup.status == BackupStatus.COMPLETED:
            logger.info(f"   ✅ Full backup completed: {full_backup.backup_id}")
            logger.info(f"   📁 Size: {full_backup.size_bytes} bytes")
            logger.info(f"   ➕ Stored at: {full_backup.file_path}")
        else:
            logger.error("   ❌ Full backup failed.")

        # Demo 3: Simular cambios en el sistema
        logger.info("\n3. Simulating system changes (adding a new agent and resource)...")
        agent3 = await framework.agent_factory.create_agent("agent.demo", "AgentGamma")
        resource3 = AgentResource(name="log_data", namespace="resource.logs", data="log entry 1", owner_agent_id=agent3.id)
        await framework.resource_manager.create_resource(resource3)
        
        logger.info(f"   📊 Current agents: {len(framework.registry.list_all_agents())}")
        logger.info(f"   📂 Current resources: {len(framework.resource_manager.list_all_resources())}")

        # Demo 4: Realizar un backup incremental
        logger.info("\n4. Performing incremental backup...")
        if full_backup:
            incremental_backup = await dr_orchestrator.backup_engine.perform_incremental_backup(framework, full_backup.backup_id, retention_policy="7-days")
            if incremental_backup and incremental_backup.status == BackupStatus.COMPLETED:
                logger.info(f"   ✅ Incremental backup completed: {incremental_backup.backup_id}")
                logger.info(f"   📁 Size: {incremental_backup.size_bytes} bytes")
            else:
                logger.error("   ❌ Incremental backup failed.")
        else:
            logger.warning("Skipping incremental backup as full backup failed.")
            
        # Demo 5: Simular disaster recovery
        logger.info(f"\n5. Disaster recovery simulation...")
        
        # Simular fallo de sistema restaurando desde punto anterior
        logger.info(f"   📉 Simulating system crash...")
        
        recovery_result = await dr_orchestrator.disaster_recovery_plan("system_crash")
        
        if recovery_result["success"]:
            logger.info(f"   ✅ Recovery successful")
            logger.info(f"   📊 Current state: {len(framework.registry.list_all_agents())} agents")
            logger.info(f"   📂 Current resources: {len(framework.resource_manager.list_all_resources())} resources")
        else:
            logger.error(f"   ❌ Recovery failed: {recovery_result['error']}")
        
        # Demo 6: Estado del sistema
        logger.info(f"\n6. Recovery system status:")
        status = dr_orchestrator.get_recovery_status()
        
        for key, value in status.items():
            logger.info(f"   {key}: {value}")
        
        # Demo 7: Historial de backups
        logger.info(f"\n7. Backup history:")
        backup_history = dr_orchestrator.backup_engine.get_backup_history()
        
        for i, backup in enumerate(backup_history[:5], 1): # Mostrar los 5 más recientes
            logger.info(f"   {i}. {backup.backup_id} ({backup.backup_type.value}) - {backup.status.value}")
            logger.info(f"      Created: {backup.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"      Path: {backup.file_path}")
            logger.info(f"      Size: {backup.size_bytes} bytes")
            logger.info(f"      Checksum: {backup.checksum[:10]}...")
            
        # Demo 8: Podar backups antiguos (simulado)
        logger.info("\n8. Pruning old backups (simulated, setting retention to 0 days for demo to prune all)")
        pruned_count = await dr_orchestrator.backup_engine.prune_old_backups(retention_days=0)
        logger.info(f"   Pruned {pruned_count} backups.")
        logger.info(f"   Total backups after pruning: {len(dr_orchestrator.backup_engine.get_backup_history())}")

    except Exception as e:
        logger.error(f"An error occurred during the demo: {e}", exc_info=True)
    finally:
        logger.info("\nCleaning up demo directories...")
        await framework.stop()
        await persistence_manager.close()
        
        if local_storage_path.exists():
            shutil.rmtree(local_storage_path)
            logger.info(f"Removed {local_storage_path}")
        
        # persistence_config.connection_string es el directorio de persistencia para la demo
        if persistence_config.connection_string.exists():
            shutil.rmtree(persistence_config.connection_string)
            logger.info(f"Removed {persistence_config.connection_string}")
            
        logger.info("Demo cleanup complete.")

if __name__ == "__main__":
    asyncio.run(demo_backup_recovery())