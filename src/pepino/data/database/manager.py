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
    
    def __init__(self, db_path: Union[str, Path] = "data/discord_messages.db"):
        self.db_path = Path(db_path)
        self._local = threading.local()
        self._lock = threading.RLock()
        self._connection_count = 0
        self._initialized = False
        
        logger.info(f"DatabaseManager initialized with path: {self.db_path}")
    
    def initialize(self):
        """Initialize the database with proper schema."""
        
        with self._lock:
            # Get or create connection
            conn = self._get_connection()
            
            # Initialize schema if this is the first connection
            if not self._initialized:
                self._initialize_schema(conn)
                self._initialized = True
        
        return conn
    
    def _initialize_schema(self, conn: sqlite3.Connection):
        """Initialize database schema if needed."""
        
        try:
            # Check if the messages table already exists and has data
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
            messages_table_exists = cursor.fetchone() is not None
            
            if messages_table_exists:
                # Check if table has data
                cursor = conn.execute("SELECT COUNT(*) FROM messages")
                message_count = cursor.fetchone()[0]
                
                if message_count > 0:
                    logger.info(f"Database already initialized with {message_count} messages - skipping schema initialization")
                    return
                    
                logger.info("Messages table exists but is empty - checking schema completeness")
            else:
                logger.info("Messages table does not exist - initializing database schema")
            
            # Check if we have all required tables
            from .schema import validate_schema
            if validate_schema(str(self.db_path)):
                logger.info("Database schema is valid - no initialization needed")
                return
            
            logger.info("Initializing missing database schema components...")
            
            # Only create missing tables without overwriting existing data
            
            # Create tables that don't exist
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = {row[0] for row in cursor.fetchall()}
            
            required_tables = {
                'messages': SCHEMA_QUERIES.get('messages', ''),
                'sync_logs': SCHEMA_QUERIES.get('sync_logs', ''),
                'users': SCHEMA_QUERIES.get('users', ''),
                'channels': SCHEMA_QUERIES.get('channels', '')
            }
            
            for table_name, create_sql in required_tables.items():
                if table_name not in existing_tables and create_sql:
                    logger.info(f"Creating missing table: {table_name}")
                    conn.execute(create_sql)
            
            conn.commit()
            logger.info("Database schema initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            conn.rollback()
            raise
    

    
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
    
    def execute_single(
        self, 
        query: str, 
        params: Optional[Tuple] = None
    ) -> Optional[sqlite3.Row]:
        """
        Execute a database query and return a single row.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            Single row as sqlite3.Row object or None if no results
        """
        
        with self.get_connection() as conn:
            try:
                cursor = conn.execute(query, params or ())
                return cursor.fetchone()
                    
            except Exception as e:
                logger.error(f"Single query execution failed: {query[:100]}... Error: {e}")
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
    
    def execute_script(self, script: str) -> None:
        """Execute a SQL script."""
        with self.get_connection() as conn:
            try:
                conn.executescript(script)
                conn.commit()
            except Exception as e:
                logger.error(f"Script execution failed: {e}")
                raise
    
    def backup_database(self, backup_path: str) -> None:
        """Create a backup of the database."""
        import shutil
        try:
            shutil.copy2(str(self.db_path), backup_path)
            logger.info(f"Database backed up to: {backup_path}")
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            raise
    
    def vacuum(self) -> None:
        """Vacuum the database to reclaim space."""
        with self.get_connection() as conn:
            try:
                conn.execute("VACUUM")
                logger.info("Database vacuum completed")
            except Exception as e:
                logger.error(f"Database vacuum failed: {e}")
                raise
    
    def analyze_tables(self) -> None:
        """Analyze tables for query optimization."""
        with self.get_connection() as conn:
            try:
                conn.execute("ANALYZE")
                logger.info("Database analyze completed")
            except Exception as e:
                logger.error(f"Database analyze failed: {e}")
                raise
    
    def get_table_names(self) -> List[str]:
        """Get list of table names in the database."""
        with self.get_connection() as conn:
            try:
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
                return [row[0] for row in cursor.fetchall()]
            except Exception as e:
                logger.error(f"Failed to get table names: {e}")
                raise
    
    def get_table_info(self, table_name: str) -> List[Dict[str, Any]]:
        """Get table information."""
        with self.get_connection() as conn:
            try:
                # Validate table name to prevent SQL injection
                if not table_name.replace('_', '').replace('-', '').isalnum():
                    raise ValueError(f"Invalid table name: {table_name}")
                cursor = conn.execute(f"PRAGMA table_info({table_name})")
                return [dict(row) for row in cursor.fetchall()]
            except Exception as e:
                logger.error(f"Failed to get table info for {table_name}: {e}")
                raise
    
    def get_row_count(self, table_name: str) -> int:
        """Get row count for a table."""
        with self.get_connection() as conn:
            try:
                # Validate table name to prevent SQL injection
                if not table_name.replace('_', '').replace('-', '').isalnum():
                    raise ValueError(f"Invalid table name: {table_name}")
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                return cursor.fetchone()[0]
            except Exception as e:
                logger.error(f"Failed to get row count for {table_name}: {e}")
                raise
    
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
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            self._local.connection = conn
            self._connection_count += 1
            logger.debug(f"Created new database connection (total: {self._connection_count})")
        return self._local.connection


 