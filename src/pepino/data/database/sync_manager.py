"""
Synchronous Database Manager
"""

import sqlite3
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SyncDatabaseManager:
    """
    This is a synchronous database manager.
    """

    def __init__(self, db_path: str):
        self.db_path = str(Path(db_path).resolve())
        self.pool = None
        self._connect()

    def _connect(self):
        """Connect to database - simple sync connection"""
        try:
            self.pool = sqlite3.connect(self.db_path)
            self.pool.row_factory = sqlite3.Row  # Enable dict-like access
            logger.info(f"Connected to sync database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def execute(self, query: str, params=None):
        """Execute query - simple and sync!"""
        if params is None:
            params = ()
        
        try:
            cursor = self.pool.execute(query, params)
            return cursor
        except Exception as e:
            logger.error(f"Query failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise

    def close(self):
        """Close database connection"""
        if self.pool:
            self.pool.close()
            logger.info("Database connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 