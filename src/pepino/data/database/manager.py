"""
Synchronous Database Manager 

This is the future sync database manager that will replace the async version.
Provides clean, simple database operations without async complexity.

Key benefits:
- No aiosqlite dependency
- Simple connection management
- Thread-safe operation
- Context manager support
- Comprehensive error handling
"""

import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .schema import SCHEMA_QUERIES

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Synchronous Database Manager 
    
    Provides thread-safe, synchronous database operations.
    This will replace the async DatabaseManager.
    
    Features:
    - Thread-safe connection pooling
    - Context manager support
    - Automatic schema management
    - Comprehensive error handling
    - Connection health monitoring
    """
    
    def __init__(self, db_path: Union[str, Path] = "discord_messages.db"):
        self.db_path = Path(db_path)
        self._local = threading.local()
        self._lock = threading.RLock()
        self._connection_count = 0
        self._initialized = False
        
        logger.info(f"DatabaseManager initialized with path: {self.db_path}")
    
    async def initialize(self):
        """Initialize the database with proper schema."""
        from .schema import init_database
        
        # Initialize the database with comprehensive schema (95 columns)
        init_database(str(self.db_path))
        self._initialized = True
    
    def execute_query(
        self, 
        query: str, 
        params: Optional[Tuple] = None,
        fetch_one: bool = False,
        fetch_all: bool = True
    ) -> Optional[Union[sqlite3.Row, List[sqlite3.Row]]]:
        """
        Execute a database query.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            fetch_one: Return only one row
            fetch_all: Return all rows (default)
            
        Returns:
            Query results or None
        """
        
        with self.get_connection() as conn:
            try:
                cursor = conn.execute(query, params or ())
                
                if fetch_one:
                    return cursor.fetchone()
                elif fetch_all:
                    return cursor.fetchall()
                else:
                    return None
                    
            except Exception as e:
                logger.error(f"Query execution failed: {query[:100]}... Error: {e}")
                raise
    
    def execute_many(
        self, 
        query: str, 
        params_list: List[Tuple]
    ) -> int:
        """
        Execute query with multiple parameter sets.
        
        Args:
            query: SQL query to execute
            params_list: List of parameter tuples
            
        Returns:
            Number of affected rows
        """
        
        with self.get_connection() as conn:
            try:
                cursor = conn.executemany(query, params_list)
                conn.commit()
                return cursor.rowcount
                
            except Exception as e:
                logger.error(f"Batch execution failed: {query[:100]}... Error: {e}")
                raise
    
    def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        
        try:
            # Test basic query
            result = self.execute_query("SELECT COUNT(*) as total FROM messages", fetch_one=True)
            total_messages = result['total'] if result else 0
            
            # Test connection
            with self.get_connection() as conn:
                conn.execute("SELECT 1")
            
            return {
                'status': 'healthy',
                'total_messages': total_messages,
                'db_path': str(self.db_path),
                'connection_count': self._connection_count
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'db_path': str(self.db_path)
            }
    
    def close_connections(self):
        """Close all thread-local connections."""
        
        with self._lock:
            if hasattr(self._local, 'connection') and self._local.connection:
                try:
                    self._local.connection.close()
                    self._connection_count -= 1
                    logger.debug(f"Closed database connection (remaining: {self._connection_count})")
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
                finally:
                    self._local.connection = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_connections()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = self._get_connection()
        try:
            yield conn
        except Exception as e:
            logger.error(f"Database operation failed: {e}")
            conn.rollback()
            raise
        # Do not close the connection here; it is reused for the thread

    def _get_connection(self):
        """Get thread-local database connection."""
        if not hasattr(self, '_local'):
            self._local = threading.local()
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            self._local.connection = conn
        return self._local.connection


# Global instance for easy access
db_manager = DatabaseManager() 