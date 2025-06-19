# agentapi/models/backup_models.py
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

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
    backup_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    backup_type: BackupType
    status: BackupStatus
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    size_bytes: int = 0
    checksum: Optional[str] = None
    file_path: str = ""
    storage_backend: StorageBackend = StorageBackend.LOCAL
    parent_backup_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RestorePoint:
    point_id: str
    timestamp: datetime
    description: str
    backup_ids: List[str]
    is_consistent: bool = True
    
@dataclass
class RecoveryPlan:
    name: str
    description: str
    steps: List[Dict[str, Any]]
    last_executed: Optional[datetime] = None
    last_result: Optional[str] = None