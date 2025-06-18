# agentapi/models/backup_models.py

import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

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
    """Backends de almacenamiento para backups"""
    LOCAL = "local"
    S3 = "s3"
    AZURE = "azure"
    SSH = "ssh"
    FTP = "ftp"
    # Añadir otros según sea necesario

@dataclass
class BackupMetadata:
    """Metadatos de un backup realizado."""
    backup_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    backup_type: BackupType
    status: BackupStatus
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    size_bytes: int = 0
    checksum: Optional[str] = None # Hash de verificación de integridad
    file_path: str = "" # Ruta o identificador en el backend de almacenamiento
    storage_backend: StorageBackend = StorageBackend.LOCAL
    parent_backup_id: Optional[str] = None # Para backups incrementales/diferenciales
    details: Dict[str, Any] = field(default_factory=dict) # Información adicional, ej. agentes/recursos incluidos

@dataclass
class RestorePoint:
    """Representa un punto de restauración disponible."""
    point_id: str
    timestamp: datetime
    description: str
    backup_ids: List[str] # IDs de los backups que componen este punto de restauración
    is_consistent: bool = True # Indica si el punto es consistente para restauración
    
@dataclass
class RecoveryPlan:
    """Define un plan para la recuperación ante desastres."""
    name: str
    description: str
    steps: List[Dict[str, Any]] # Pasos del plan, e.g., [{"action": "restore_db", "args": {"backup_id": "latest"}}]
    # Podría ser un modelo más complejo con sub-clases para los pasos
    last_executed: Optional[datetime] = None
    last_result: Optional[str] = None # "SUCCESS" or "FAILED"