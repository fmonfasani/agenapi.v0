"""
backup_recovery_system.py - Sistema de backup y recuperación ante desastres
"""

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
import boto3 # Si se usa AWS S3
from azure.storage.blob import BlobServiceClient # Si se usa Azure Blob Storage
import paramiko # Si se usa SSH/SFTP

# Importaciones actualizadas
from core.autonomous_agent_framework import AgentFramework, BaseAgent, AgentResource # <-- CAMBIO AQUI

# Asumimos que PersistenceManager se extraerá a su propio módulo `core/persistence_system.py`
from core.persistence_system import PersistenceManager 

# ================================\
# BACKUP MODELS
# ================================\

class BackupType(Enum):
    """Tipos de backup"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"

class BackupStatus(Enum):
    """Estados de backup"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CORRUPTED = "corrupted"

class StorageBackend(Enum):
    """Backends de almacenamiento"""
    LOCAL = "local"
    S3 = "s3"
    AZURE = "azure"
    SSH = "ssh"
    FTP = "ftp" # Aunque FTP es menos seguro para backups

@dataclass
class BackupMetadata:
    """Metadatos de backup"""
    backup_id: str
    backup_type: BackupType
    status: BackupStatus
    created_at: datetime
    completed_at: Optional[datetime]
    size_bytes: int
    checksum: str
    file_path: str # Ruta donde se almacenó el backup
    storage_backend: StorageBackend
    parent_backup_id: Optional[str] = None # Para backups incrementales/diferenciales
    restoration_info: Optional[Dict[str, Any]] = None # Información para restaurar

@dataclass
class BackupConfig:
    """Configuración del sistema de backup"""
    enabled: bool = True
    backup_interval_seconds: int = 3600 # Cada hora
    backup_type: BackupType = BackupType.FULL
    storage_backend: StorageBackend = StorageBackend.LOCAL
    local_backup_dir: str = "./backups"
    retention_days: int = 7
    s3_config: Optional[Dict[str, Any]] = None # {"bucket_name": "...", "region": "..."}
    azure_config: Optional[Dict[str, Any]] = None # {"connection_string": "...", "container_name": "..."}
    ssh_config: Optional[Dict[str, Any]] = None # {"host": "...", "port": ..., "username": "...", "key_file": "..."}


# ================================\
# BACKUP ENGINE
# ================================\

class BackupEngine:
    """
    Gestiona la creación y restauración de backups del estado del framework.
    Interactúa con el PersistenceManager para obtener los datos.
    """
    def __init__(self, persistence_manager: PersistenceManager, config: Optional[BackupConfig] = None):
        self.persistence_manager = persistence_manager
        self.config = config or BackupConfig()
        self.logger = logging.getLogger("BackupEngine")
        self.backup_history: List[BackupMetadata] = []
        self._backup_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self.logger.info("BackupEngine initialized.")

        # Asegurar el directorio de backups local
        Path(self.config.local_backup_dir).mkdir(parents=True, exist_ok=True)

    async def start(self):
        """Inicia el motor de backups."""
        if self.config.enabled and not self._backup_task:
            self._stop_event.clear()
            self._backup_task = asyncio.create_task(self._backup_loop())
            self.logger.info("BackupEngine started.")

    async def stop(self):
        """Detiene el motor de backups."""
        if self._backup_task:
            self._stop_event.set()
            self._backup_task.cancel()
            try:
                await self._backup_task
            except asyncio.CancelledError:
                self.logger.info("BackupEngine task cancelled.")
            self._backup_task = None
            self.logger.info("BackupEngine stopped.")

    async def _backup_loop(self):
        """Bucle principal para realizar backups periódicos."""
        while not self._stop_event.is_set():
            try:
                self.logger.info("Performing scheduled backup...")
                await self.create_backup(self.config.backup_type)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error during scheduled backup: {e}")
            
            try:
                await asyncio.sleep(self.config.backup_interval_seconds)
            except asyncio.CancelledError:
                break
        self.logger.info("Backup loop stopped.")


    async def create_backup(self, backup_type: BackupType = BackupType.FULL) -> Optional[BackupMetadata]:
        """Crea un backup del estado actual del framework."""
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
            # 1. Asegurarse de que el estado actual del framework esté persistido
            # Esto asume que el PersistenceManager guarda todo en el sistema de archivos (ej. SQLite DB file)
            # Para la demo, el PersistenceManager ya maneja el guardado en 'framework.db'
            await self.persistence_manager.save_full_state(self.persistence_manager.framework)
            db_file_path = self.persistence_manager.config.connection_string

            # 2. Comprimir la base de datos (o los archivos de estado)
            await self._compress_data([db_file_path], temp_file)

            # 3. Subir a almacenamiento
            dest_path = await self._upload_to_storage(temp_file, backup_id)

            # 4. Actualizar metadatos
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
            # Limpiar archivo temporal
            if temp_file.exists():
                os.remove(temp_file)
                self.logger.debug(f"Removed temporary backup file: {temp_file}")
            await self._apply_retention_policy() # Ejecutar política de retención

    async def _compress_data(self, files_to_compress: List[str], output_path: Path):
        """Comprime los archivos de datos en un archivo tar.gz."""
        self.logger.debug(f"Compressing files {files_to_compress} to {output_path}")
        with tarfile.open(output_path, "w:gz") as tar:
            for file_path in files_to_compress:
                if Path(file_path).exists():
                    tar.add(file_path, arcname=Path(file_path).name)
                else:
                    self.logger.warning(f"File not found for backup: {file_path}")
        self.logger.debug("Compression complete.")

    async def _upload_to_storage(self, file_path: Path, backup_id: str) -> str:
        """Sube el archivo de backup al backend de almacenamiento configurado."""
        dest_path_on_storage = ""
        if self.config.storage_backend == StorageBackend.LOCAL:
            dest_path = Path(self.config.local_backup_dir) / file_path.name
            shutil.copy(file_path, dest_path)
            dest_path_on_storage = str(dest_path.resolve())
            self.logger.info(f"Uploaded to local storage: {dest_path_on_storage}")
        elif self.config.storage_backend == StorageBackend.S3:
            s3_config = self.config.s3_config
            if not s3_config: raise ValueError("S3 configuration missing.")
            s3 = boto3.client('s3', region_name=s3_config["region"])
            bucket_name = s3_config["bucket_name"]
            s3_key = f"backups/{file_path.name}"
            s3.upload_file(str(file_path), bucket_name, s3_key)
            dest_path_on_storage = f"s3://{bucket_name}/{s3_key}"
            self.logger.info(f"Uploaded to S3: {dest_path_on_storage}")
        elif self.config.storage_backend == StorageBackend.AZURE:
            azure_config = self.config.azure_config
            if not azure_config: raise ValueError("Azure configuration missing.")
            blob_service_client = BlobServiceClient.from_connection_string(azure_config["connection_string"])
            container_client = blob_service_client.get_container_client(azure_config["container_name"])
            blob_name = f"backups/{file_path.name}"
            with open(file_path, "rb") as data:
                container_client.upload_blob(name=blob_name, data=data, overwrite=True)
            dest_path_on_storage = f"azure://{azure_config['container_name']}/{blob_name}"
            self.logger.info(f"Uploaded to Azure Blob Storage: {dest_path_on_storage}")
        elif self.config.storage_backend == StorageBackend.SSH:
            ssh_config = self.config.ssh_config
            if not ssh_config: raise ValueError("SSH configuration missing.")
            transport = paramiko.Transport((ssh_config["host"], ssh_config.get("port", 22)))
            transport.connect(username=ssh_config["username"], password=ssh_config.get("password"), pkey=paramiko.RSAKey.from_private_key_file(ssh_config["key_file"]))
            sftp = paramiko.SFTPClient.from_transport(transport)
            remote_path = f"{ssh_config.get('remote_path', '/backups')}/{file_path.name}"
            sftp.put(str(file_path), remote_path)
            sftp.close()
            transport.close()
            dest_path_on_storage = f"ssh://{ssh_config['host']}{remote_path}"
            self.logger.info(f"Uploaded via SSH: {dest_path_on_storage}")
        else:
            raise ValueError(f"Unsupported storage backend: {self.config.storage_backend.value}")
        return dest_path_on_storage

    def _calculate_checksum(self, file_path: Path, algorithm: str = "sha256") -> str:
        """Calcula el checksum de un archivo."""
        hasher = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    async def restore_backup(self, backup_id: str) -> bool:
        """Restaura un backup por su ID."""
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
            # 1. Descargar backup
            downloaded_path = await self._download_from_storage(metadata.file_path, temp_dir)
            if not downloaded_path: return False
            
            # 2. Verificar checksum
            if self._calculate_checksum(downloaded_path) != metadata.checksum:
                self.logger.error(f"Checksum mismatch for backup {backup_id}. Possible corruption.")
                metadata.status = BackupStatus.CORRUPTED
                return False

            # 3. Descomprimir
            await self._decompress_data(downloaded_path, temp_dir)

            # 4. Cargar datos en el PersistenceManager
            # Asumimos que el archivo de BD es el mismo nombre que el original
            db_file_name = Path(self.persistence_manager.config.connection_string).name
            restored_db_path = temp_dir / db_file_name

            if not restored_db_path.exists():
                self.logger.error(f"Restored database file {restored_db_path} not found in temporary directory.")
                return False

            # Parar el framework y el persistence manager para reemplazar el archivo de BD
            current_db_path = Path(self.persistence_manager.config.connection_string)
            
            # Asegurarse de cerrar el PersistenceManager antes de mover el archivo de DB
            await self.persistence_manager.close()
            
            if current_db_path.exists():
                self.logger.info(f"Backing up current DB file to {current_db_path}.old")
                shutil.move(current_db_path, str(current_db_path) + ".old")

            shutil.move(restored_db_path, current_db_path)
            self.logger.info(f"Database file replaced with backup {backup_id}.")

            # Reinicializar el PersistenceManager para que cargue la nueva DB
            await self.persistence_manager.initialize()
            
            # Finalmente, cargar el estado en el framework
            if not await self.persistence_manager.load_full_state(self.persistence_manager.framework):
                self.logger.error(f"Failed to load full state into framework after restoring DB.")
                return False

            self.logger.info(f"Backup {backup_id} restored successfully.")
            return True

        except Exception as e:
            self.logger.error(f"Error during restoration of backup {backup_id}: {e}")
            return False
        finally:
            # Limpiar directorio temporal
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                self.logger.debug(f"Removed temporary restoration directory: {temp_dir}")

    async def _download_from_storage(self, source_path: str, dest_dir: Path) -> Optional[Path]:
        """Descarga un archivo de backup desde el backend de almacenamiento."""
        file_name = Path(source_path).name # Asumimos que el nombre del archivo es el último componente de la ruta
        dest_path = dest_dir / file_name

        if source_path.startswith("s3://"):
            s3_config = self.config.s3_config
            if not s3_config: raise ValueError("S3 configuration missing.")
            s3 = boto3.client('s3', region_name=s3_config["region"])
            bucket_name = source_path.split("//")[1].split("/")[0]
            s3_key = "/".join(source_path.split("/")[3:]) # path/to/file.tar.gz
            s3.download_file(bucket_name, s3_key, str(dest_path))
            self.logger.info(f"Downloaded from S3: {source_path}")
        elif source_path.startswith("azure://"):
            azure_config = self.config.azure_config
            if not azure_config: raise ValueError("Azure configuration missing.")
            blob_service_client = BlobServiceClient.from_connection_string(azure_config["connection_string"])
            container_name = source_path.split("//")[1].split("/")[0]
            blob_name = "/".join(source_path.split("/")[3:])
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            with open(dest_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            self.logger.info(f"Downloaded from Azure Blob Storage: {source_path}")
        elif source_path.startswith("ssh://"):
            ssh_config = self.config.ssh_config
            if not ssh_config: raise ValueError("SSH configuration missing.")
            transport = paramiko.Transport((ssh_config["host"], ssh_config.get("port", 22)))
            transport.connect(username=ssh_config["username"], password=ssh_config.get("password"), pkey=paramiko.RSAKey.from_private_key_file(ssh_config["key_file"]))
            sftp = paramiko.SFTPClient.from_transport(transport)
            remote_path = "/".join(source_path.split("/")[2:]) # /backups/file.tar.gz
            sftp.get(remote_path, str(dest_path))
            sftp.close()
            transport.close()
            self.logger.info(f"Downloaded via SSH: {source_path}")
        elif Path(source_path).exists(): # Local file
            shutil.copy(source_path, dest_path)
            self.logger.info(f"Copied from local: {source_path}")
        else:
            self.logger.error(f"Unsupported or invalid source path for download: {source_path}")
            return None
        
        return dest_path

    async def _decompress_data(self, compressed_file_path: Path, output_dir: Path):
        """Descomprime un archivo tar.gz."""
        self.logger.debug(f"Decompressing {compressed_file_path} to {output_dir}")
        with tarfile.open(compressed_file_path, "r:gz") as tar:
            tar.extractall(path=output_dir)
        self.logger.debug("Decompression complete.")

    def get_backup_history(self) -> List[BackupMetadata]:
        """Retorna el historial de backups, ordenado por fecha de creación."""
        return sorted(self.backup_history, key=lambda b: b.created_at, reverse=True)

    async def _apply_retention_policy(self):
        """Aplica la política de retención de backups."""
        if self.config.retention_days <= 0:
            return # No aplicar retención si es 0 o negativo

        self.logger.info(f"Applying retention policy: retaining backups for {self.config.retention_days} days.")
        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
        
        backups_to_delete = [
            b for b in self.backup_history 
            if b.created_at < cutoff_date and b.status == BackupStatus.COMPLETED
        ]
        
        for backup in backups_to_delete:
            try:
                self.logger.info(f"Deleting expired backup: {backup.backup_id} (created: {backup.created_at})")
                await self._delete_from_storage(backup.file_path, backup.storage_backend)
                self.backup_history.remove(backup)
            except Exception as e:
                self.logger.error(f"Failed to delete expired backup {backup.backup_id}: {e}")

    async def _delete_from_storage(self, path: str, backend: StorageBackend) -> bool:
        """Elimina un archivo de backup del almacenamiento."""
        try:
            if backend == StorageBackend.LOCAL:
                if Path(path).exists():
                    os.remove(path)
                self.logger.info(f"Deleted local file: {path}")
            elif backend == StorageBackend.S3:
                s3_config = self.config.s3_config
                if not s3_config: raise ValueError("S3 configuration missing.")
                s3 = boto3.client('s3', region_name=s3_config["region"])
                bucket_name = path.split("//")[1].split("/")[0]
                s3_key = "/".join(path.split("/")[3:])
                s3.delete_object(Bucket=bucket_name, Key=s3_key)
                self.logger.info(f"Deleted from S3: {path}")
            elif backend == StorageBackend.AZURE:
                azure_config = self.config.azure_config
                if not azure_config: raise ValueError("Azure configuration missing.")
                blob_service_client = BlobServiceClient.from_connection_string(azure_config["connection_string"])
                container_name = path.split("//")[1].split("/")[0]
                blob_name = "/".join(path.split("/")[3:])
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
                blob_client.delete_blob()
                self.logger.info(f"Deleted from Azure Blob Storage: {path}")
            elif backend == StorageBackend.SSH:
                ssh_config = self.config.ssh_config
                if not ssh_config: raise ValueError("SSH configuration missing.")
                transport = paramiko.Transport((ssh_config["host"], ssh_config.get("port", 22)))
                transport.connect(username=ssh_config["username"], password=ssh_config.get("password"), pkey=paramiko.RSAKey.from_private_key_file(ssh_config["key_file"]))
                sftp = paramiko.SFTPClient.from_transport(transport)
                remote_path = "/".join(path.split("/")[2:])
                sftp.remove(remote_path)
                sftp.close()
                transport.close()
                self.logger.info(f"Deleted via SSH: {path}")
            else:
                self.logger.warning(f"Unsupported storage backend for deletion: {backend.value}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error deleting backup from storage {path}: {e}")
            return False


# ================================\
# DISASTER RECOVERY ORCHESTRATOR
# ================================\

class DisasterRecoveryOrchestrator:
    """
    Orquesta planes de recuperación ante desastres utilizando backups.
    """
    def __init__(self, framework: AgentFramework, persistence_manager: PersistenceManager, config: Optional[BackupConfig] = None):
        self.framework = framework
        self.persistence_manager = persistence_manager
        self.config = config or BackupConfig()
        self.backup_engine = BackupEngine(persistence_manager, self.config)
        self.logger = logging.getLogger("DROrchestrator")
        self._recovery_status: Dict[str, Any] = {"last_recovery_attempt": None, "status": "idle", "error": None}
        self.logger.info("DisasterRecoveryOrchestrator initialized.")

    async def start(self):
        """Inicia el orquestador de DR y el motor de backups."""
        self.logger.info("Starting DisasterRecoveryOrchestrator...")
        await self.backup_engine.start()
        self.logger.info("DisasterRecoveryOrchestrator started.")

    async def stop(self):
        """Detiene el orquestador de DR y el motor de backups."""
        self.logger.info("Stopping DisasterRecoveryOrchestrator...")
        await self.backup_engine.stop()
        self.logger.info("DisasterRecoveryOrchestrator stopped.")

    async def disaster_recovery_plan(self, scenario: str, backup_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Ejecuta un plan de recuperación ante desastres basado en un escenario.
        Esto simulará un proceso de recuperación completo.
        """
        self.logger.warning(f"Initiating disaster recovery plan for scenario: {scenario}")
        self._recovery_status = {"last_recovery_attempt": datetime.now().isoformat(), "status": "running", "error": None}
        
        try:
            # 1. Detener el framework (simula un fallo o parada controlada)
            self.logger.info("Stopping framework for recovery...")
            await self.framework.stop()
            
            # 2. Seleccionar backup
            if not backup_id:
                # Obtener el último backup completo exitoso
                latest_backup = next((b for b in self.backup_engine.get_backup_history() if b.status == BackupStatus.COMPLETED and b.backup_type == BackupType.FULL), None)
                if not latest_backup:
                    self._recovery_status["status"] = "failed"
                    self._recovery_status["error"] = "No suitable backup found for recovery."
                    self.logger.error(self._recovery_status["error"])
                    return {"success": False, "error": self._recovery_status["error"]}
                backup_id = latest_backup.backup_id
                self.logger.info(f"Using latest full backup for recovery: {backup_id}")
            else:
                self.logger.info(f"Using specified backup for recovery: {backup_id}")

            # 3. Restaurar el backup
            restore_success = await self.backup_engine.restore_backup(backup_id)
            if not restore_success:
                self._recovery_status["status"] = "failed"
                self._recovery_status["error"] = f"Failed to restore backup {backup_id}."
                self.logger.error(self._recovery_status["error"])
                return {"success": False, "error": self._recovery_status["error"]}

            # 4. Reiniciar el framework
            self.logger.info("Restarting framework after recovery...")
            await self.framework.start()

            # Opcional: Re-registrar agentes que puedan haberse perdido si no se maneja en load_full_state
            # (En el caso real, load_full_state debería manejar esto con la ayuda del AgentFactory)
            
            self._recovery_status["status"] = "completed"
            self.logger.info(f"Disaster recovery for scenario '{scenario}' completed successfully.")
            return {"success": True, "details": f"Restored from backup {backup_id}."}

        except Exception as e:
            self._recovery_status["status"] = "failed"
            self._recovery_status["error"] = str(e)
            self.logger.critical(f"Critical error during disaster recovery for scenario '{scenario}': {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def get_recovery_status(self) -> Dict[str, Any]:
        """Obtiene el estado actual del orquestador de recuperación."""
        return self._recovery_status

# ================================\
# DEMO
# ================================\

async def backup_recovery_demo():
    """Ejemplo de uso del sistema de backup y recuperación."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("🚀 Starting Backup & Recovery System Demo")
    print("="*50)

    # Configuración de la demo
    backup_config = BackupConfig(
        enabled=True,
        backup_interval_seconds=10, # Backup cada 10 segundos para la demo
        backup_type=BackupType.FULL,
        local_backup_dir="./demo_backups",
        retention_days=0 # No retener para la demo, borrar todo
    )

    # Limpiar directorio de backups anterior para una demo limpia
    if Path(backup_config.local_backup_dir).exists():
        shutil.rmtree(backup_config.local_backup_dir)
        print(f"🗑️ Cleaned up previous backup directory: {backup_config.local_backup_dir}")
    Path(backup_config.local_backup_dir).mkdir(parents=True, exist_ok=True)


    framework = AgentFramework()
    # No iniciar framework todavía, el DR Orchestrator lo hará
    
    # Inicializar PersistenceManager con una DB específica para la demo
    from core.persistence_system import PersistenceConfig # Importar PersistenceConfig
    persistence_config_dr = PersistenceConfig(
        backend="sqlite",
        connection_string="./demo_framework_dr.db",
        auto_save_interval=0 # Deshabilitar auto-save para que el DR lo controle
    )
    # Limpiar DB de la demo
    if Path(persistence_config_dr.connection_string).exists():
        Path(persistence_config_dr.connection_string).unlink()
        print(f"🗑️ Cleaned up previous DB: {persistence_config_dr.connection_string}")

    persistence_manager = PersistenceManager(framework, persistence_config_dr)
    await persistence_manager.initialize()


    dr_orchestrator = DisasterRecoveryOrchestrator(framework, persistence_manager, backup_config)
    await dr_orchestrator.start()

    print("\n⚙️ Backup & Recovery system initialized.")

    # Demo 1: Crear agentes y recursos
    print("1. Simulating initial system state with agents and resources...")
    from core.models import AgentStatus, ResourceType # Importar de core.models

    class DemoAgent(BaseAgent):
        def __init__(self, agent_id: str, name: str, framework_instance: AgentFramework):
            super().__init__("demo.agent", name, framework_instance)
            self.id = agent_id
            self.status = AgentStatus.ACTIVE
            self.message_queue = asyncio.Queue()

        async def initialize(self) -> bool:
            self.logger.info(f"DemoAgent {self.name} initialized.")
            return True
        async def process_message(self, message):
            self.logger.info(f"DemoAgent {self.name} processed message: {message.id}")
            await asyncio.sleep(0.01) # Simular trabajo
    
    # Iniciar el framework antes de crear agentes para que registry y resource_manager estén listos
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
    
    # Dar tiempo para que el primer backup programado se ejecute
    print(f"\n2. Waiting for first scheduled full backup (every {backup_config.backup_interval_seconds} seconds)...")
    await asyncio.sleep(backup_config.backup_interval_seconds + 2) # Esperar un poco más del intervalo

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

    # Demo 3: Simular cambios de estado después del backup
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
    print(f"   ✅ Current state: {len(framework.registry.list_all_agents())} agents, {len(framework.resource_manager.list_all_resources())} resources (more than initial).")

    # Demo 4: Crear un backup incremental (conceptualmente, para la demo solo full)
    print(f"\n4. Waiting for another scheduled backup (incremental for demo purposes)...")
    await asyncio.sleep(backup_config.backup_interval_seconds + 2)
    
    incremental_backups = [b for b in dr_orchestrator.backup_engine.get_backup_history() if b.created_at > latest_full_backup.created_at and b.status == BackupStatus.COMPLETED]
    if incremental_backups:
        incremental_backup = incremental_backups[0]
        print(f"   ✅ Incremental backup created: {incremental_backup.backup_id}")
        print(f"   📁 Size: {incremental_backup.size_bytes} bytes")
    else:
        print("   ⚠️ No incremental backup created (expected if interval is not hit or only full backups).")
        # For a full demonstration, we would manually trigger one
        incremental_backup = await dr_orchestrator.backup_engine.create_backup(BackupType.FULL) # Forcing another full if no incremental logic
        if incremental_backup:
            print(f"   ✅ Forcing another full backup: {incremental_backup.backup_id}")
            print(f"   📁 Size: {incremental_backup.size_bytes} bytes")


    # Demo 5: Simular disaster recovery
    print(f"\n5. Disaster recovery simulation (restoring to initial state)...\n")
    
    # Simular fallo de sistema restaurando desde punto anterior (el primer full backup)
    print(f"   📉 Simulating system crash (stopping framework, DB will be replaced)...\n")
    
    recovery_result = await dr_orchestrator.disaster_recovery_plan(
        "system_crash",
        backup_id=latest_full_backup.backup_id # Restaurar al estado del primer backup
    )
    
    if recovery_result["success"]:
        print(f"   ✅ Recovery successful")
        await asyncio.sleep(1) # Dar tiempo al framework para arrancar
        print(f"   📊 Current state: {len(framework.registry.list_all_agents())} agents, {len(framework.resource_manager.list_all_resources())} resources.")
        # Verificar que el agente y el recurso nuevos no estén después de la recuperación
        if not framework.registry.get_agent(agent3.id):
            print(f"   ✅ Agent {agent3.name} not found after recovery (as expected).")
        if not await framework.resource_manager.get_resource(new_resource.id):
            print(f"   ✅ Resource {new_resource.name} not found after recovery (as expected).")
    else:
        print(f"   ❌ Recovery failed: {recovery_result['error']}")
    
    # Demo 6: Estado del sistema
    print(f"\n6. Recovery system status:\n")
    status = dr_orchestrator.get_recovery_status()
    
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Demo 7: Historial de backups
    print(f"\n7. Backup history:\n")
    backup_history = dr_orchestrator.backup_engine.get_backup_history()
    
    for i, backup in enumerate(backup_history[:5], 1): # Mostrar los 5 más recientes
        print(f"   {i}. {backup.backup_id} ({backup.backup_type.value}) - {backup.status.value}")
        print(f"      Created: {backup.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"      File: {backup.file_path}")
        print(f"      Size: {backup.size_bytes} bytes")
        
    # Cleanup final
    await dr_orchestrator.stop()
    await framework.stop() # Asegurarse de que el framework esté parado
    await persistence_manager.close()

    print("\nDemo finished.")


if __name__ == "__main__":
    asyncio.run(backup_recovery_demo())