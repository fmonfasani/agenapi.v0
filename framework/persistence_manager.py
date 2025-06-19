# framework/persistence_manager.py

import json
import logging
import asyncio # Added for to_thread if needed, and for async operations
from typing import Any, Optional
from datetime import datetime
from pathlib import Path

from agentapi.models.framework_models import PersistenceConfig

# Assuming aiosqlite for async DB operations. If not, consider `asyncio.to_thread` for blocking calls.
# import aiosqlite


class PersistenceManager:
    def __init__(self, config: PersistenceConfig):
        self.config = config
        self.logger = logging.getLogger("PersistenceManager")
        self._db_connection: Optional[Any] = None # Placeholder for DB connection object

    async def initialize(self):
        """Initializes the persistence manager, e.g., connects to database."""
        if self.config.backend == "sqlite":
            try:
                # Example for aiosqlite. If using regular sqlite3, you'd use asyncio.to_thread
                # self._db_connection = await aiosqlite.connect(self.config.connection_string)
                # await self._db_connection.execute("CREATE TABLE IF NOT EXISTS kv_store (key TEXT PRIMARY KEY, value TEXT)")
                # await self._db_connection.commit()
                self.logger.info(f"PersistenceManager initialized for backend: {self.config.backend}")
            except Exception as e:
                self.logger.error(f"Failed to initialize SQLite persistence: {e}", exc_info=True)
                self._db_connection = None
        elif self.config.backend == "json":
            # Ensure data directory exists for file-based persistence
            if self.config.connection_string:
                Path(self.config.connection_string).parent.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"PersistenceManager initialized for backend: {self.config.backend} (file path: {self.config.connection_string})")
        else:
            self.logger.warning(f"Persistence backend '{self.config.backend}' not fully implemented or initialized.")

    async def save_state(self, key: str, data: Any):
        if not self._db_connection and self.config.backend == "sqlite": # Only log if DB backend is active and not connected
            self.logger.warning("Database connection not established for saving state.")
            return

        if self.config.backend == "json":
            # Use connection_string as base path, append key for specific file
            file_path = Path(self.config.connection_string).parent / f"{Path(self.config.connection_string).stem}_{key}.json"
            try:
                # Use default=str to serialize datetime objects and other non-JSON serializable types
                def json_default(o):
                    if isinstance(o, (datetime)):
                        return o.isoformat()
                    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

                # Run blocking file I/O in a separate thread
                await asyncio.to_thread(lambda: json.dump(data, file_path.open('w'), indent=2, default=json_default))
                self.logger.debug(f"State saved for {key} to {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to save state for {key} to {file_path}: {e}", exc_info=True)
        elif self.config.backend == "sqlite" and self._db_connection:
            try:
                # await self._db_connection.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)", (key, json.dumps(data, default=str)))
                # await self._db_connection.commit()
                self.logger.debug(f"State saved for {key} to SQLite.")
            except Exception as e:
                self.logger.error(f"Failed to save state for {key} to SQLite: {e}", exc_info=True)
        else:
            self.logger.warning(f"Persistence backend '{self.config.backend}' not implemented for saving state.")

    async def load_state(self, key: str) -> Optional[Any]:
        if not self._db_connection and self.config.backend == "sqlite":
            self.logger.warning("Database connection not established for loading state.")
            return None

        if self.config.backend == "json":
            file_path = Path(self.config.connection_string).parent / f"{Path(self.config.connection_string).stem}_{key}.json"
            try:
                # Run blocking file I/O in a separate thread
                data = await asyncio.to_thread(lambda: json.load(file_path.open('r')))
                self.logger.debug(f"State loaded for {key} from {file_path}")
                return data
            except FileNotFoundError:
                self.logger.warning(f"No saved state found for {key} at {file_path}")
                return None
            except Exception as e:
                self.logger.error(f"Failed to load state for {key} from {file_path}: {e}", exc_info=True)
                return None
        elif self.config.backend == "sqlite" and self._db_connection:
            try:
                # cursor = await self._db_connection.execute("SELECT value FROM kv_store WHERE key = ?", (key,))
                # row = await cursor.fetchone()
                # if row:
                #     self.logger.debug(f"State loaded for {key} from SQLite.")
                #     return json.loads(row[0])
                # return None
                pass # Placeholder
            except Exception as e:
                self.logger.error(f"Failed to load state for {key} from SQLite: {e}", exc_info=True)
                return None
        else:
            self.logger.warning(f"Persistence backend '{self.config.backend}' not implemented for loading state.")
            return None

    async def shutdown(self):
        """Closes any open connections and flushes pending data."""
        self.logger.info("Shutting down PersistenceManager...")
        if self._db_connection:
            try:
                # await self._db_connection.close()
                self.logger.info("Database connection closed.")
            except Exception as e:
                self.logger.error(f"Error closing database connection: {e}", exc_info=True)
        self.logger.info("PersistenceManager shut down.")