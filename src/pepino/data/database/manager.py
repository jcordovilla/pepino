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
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            with self._lock:
                try:
                    # Create connection with optimizations
                    conn = sqlite3.connect(
                        str(self.db_path),
                        check_same_thread=False,
                        timeout=30.0
                    )
                    
                    # Configure connection
                    conn.row_factory = sqlite3.Row  # Dict-like row access
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL") 
                    conn.execute("PRAGMA cache_size=10000")
                    conn.execute("PRAGMA temp_store=MEMORY")
                    
                    self._local.connection = conn
                    self._connection_count += 1
                    
                    logger.debug(f"Created new database connection (total: {self._connection_count})")
                    
                    # Initialize schema if this is the first connection
                    if not self._initialized:
                        self._initialize_schema(conn)
                        self._initialized = True
                    
                except Exception as e:
                    logger.error(f"Failed to create database connection: {e}")
                    raise
        
        return self._local.connection
    
    def _initialize_schema(self, conn: sqlite3.Connection):
        """Initialize database schema if needed."""
        
        try:
            logger.info("Initializing database schema...")
            
            # Create tables
            for table_name, create_query in SCHEMA_QUERIES.items():
                try:
                    conn.execute(create_query)
                    logger.debug(f"Created/verified table: {table_name}")
                except sqlite3.OperationalError as e:
                    if "already exists" not in str(e):
                        logger.error(f"Failed to create table {table_name}: {e}")
                        raise
            
            # Create indexes for performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(channel_name)",
                "CREATE INDEX IF NOT EXISTS idx_messages_author ON messages(author_name)",
                "CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_messages_channel_timestamp ON messages(channel_name, timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_users_name ON users(username)",
                "CREATE INDEX IF NOT EXISTS idx_channels_name ON channels(name)"
            ]
            
            for index_query in indexes:
                try:
                    conn.execute(index_query)
                except sqlite3.OperationalError as e:
                    if "already exists" not in str(e):
                        logger.warning(f"Failed to create index: {e}")
            
            conn.commit()
            logger.info("Database schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            conn.rollback()
            raise
    
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
        finally:
            # Connection stays open for thread reuse
            pass
    
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
    
    def insert_message(self, message_data: Dict[str, Any]) -> bool:
        """Insert a single message into database."""
        
        query = """
        INSERT OR REPLACE INTO messages (
            message_id, channel_name, author_name, content, 
            timestamp, message_type, reply_to, thread_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            message_data.get('message_id'),
            message_data.get('channel_name'),
            message_data.get('author_name'),
            message_data.get('content'),
            message_data.get('timestamp'),
            message_data.get('message_type', 'default'),
            message_data.get('reply_to'),
            message_data.get('thread_id')
        )
        
        try:
            with self.get_connection() as conn:
                conn.execute(query, params)
                conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert message: {e}")
            return False
    
    def insert_messages_batch(self, messages: List[Dict[str, Any]]) -> int:
        """Insert multiple messages in a single transaction."""
        
        query = """
        INSERT OR REPLACE INTO messages (
            message_id, channel_name, author_name, content, 
            timestamp, message_type, reply_to, thread_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params_list = []
        for msg in messages:
            params = (
                msg.get('message_id'),
                msg.get('channel_name'),
                msg.get('author_name'),
                msg.get('content'),
                msg.get('timestamp'),
                msg.get('message_type', 'default'),
                msg.get('reply_to'),
                msg.get('thread_id')
            )
            params_list.append(params)
        
        try:
            return self.execute_many(query, params_list)
        except Exception as e:
            logger.error(f"Failed to insert message batch: {e}")
            return 0
    
    def get_messages_by_channel(
        self, 
        channel_name: str, 
        limit: Optional[int] = None,
        days: Optional[int] = None
    ) -> List[sqlite3.Row]:
        """Get messages for a specific channel."""
        
        query = "SELECT * FROM messages WHERE channel_name = ?"
        params = [channel_name]
        
        if days:
            query += " AND timestamp >= datetime('now', '-' || ? || ' days')"
            params.append(days)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        return self.execute_query(query, tuple(params))
    
    def get_messages_by_user(
        self, 
        username: str, 
        limit: Optional[int] = None,
        days: Optional[int] = None
    ) -> List[sqlite3.Row]:
        """Get messages for a specific user."""
        
        query = "SELECT * FROM messages WHERE author_name = ?"
        params = [username]
        
        if days:
            query += " AND timestamp >= datetime('now', '-' || ? || ' days')"
            params.append(days)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        return self.execute_query(query, tuple(params))
    
    def get_channel_statistics(self, channel_name: str, days: Optional[int] = None) -> Dict[str, Any]:
        """Get statistics for a channel."""
        
        base_query = "SELECT COUNT(*) as total_messages FROM messages WHERE channel_name = ?"
        params = [channel_name]
        
        if days:
            base_query += " AND timestamp >= datetime('now', '-' || ? || ' days')"
            params.append(days)
        
        total_result = self.execute_query(base_query, tuple(params), fetch_one=True)
        total_messages = total_result['total_messages'] if total_result else 0
        
        # Get unique users
        user_query = base_query.replace("COUNT(*) as total_messages", "COUNT(DISTINCT author_name) as unique_users")
        user_result = self.execute_query(user_query, tuple(params), fetch_one=True)
        unique_users = user_result['unique_users'] if user_result else 0
        
        return {
            'total_messages': total_messages,
            'unique_users': unique_users,
            'channel_name': channel_name
        }
    
    def get_user_statistics(self, username: str, days: Optional[int] = None) -> Dict[str, Any]:
        """Get statistics for a user."""
        
        base_query = "SELECT COUNT(*) as total_messages FROM messages WHERE author_name = ?"
        params = [username]
        
        if days:
            base_query += " AND timestamp >= datetime('now', '-' || ? || ' days')"
            params.append(days)
        
        total_result = self.execute_query(base_query, tuple(params), fetch_one=True)
        total_messages = total_result['total_messages'] if total_result else 0
        
        # Get unique channels
        channel_query = base_query.replace("COUNT(*) as total_messages", "COUNT(DISTINCT channel_name) as unique_channels")
        channel_result = self.execute_query(channel_query, tuple(params), fetch_one=True)
        unique_channels = channel_result['unique_channels'] if channel_result else 0
        
        return {
            'total_messages': total_messages,
            'unique_channels': unique_channels,
            'username': username
        }
    
    def get_available_channels(self) -> List[str]:
        """Get list of channels with messages."""
        
        query = "SELECT DISTINCT channel_name FROM messages ORDER BY channel_name"
        results = self.execute_query(query)
        
        return [row['channel_name'] for row in results] if results else []
    
    def get_available_users(self) -> List[str]:
        """Get list of users with messages."""
        
        query = "SELECT DISTINCT author_name FROM messages ORDER BY author_name"
        results = self.execute_query(query)
        
        return [row['author_name'] for row in results] if results else []
    
    def get_top_users(self, limit: int = 10, days: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get top users by message count."""
        
        query = """
        SELECT 
            author_name,
            COUNT(*) as message_count,
            COUNT(DISTINCT channel_name) as channels_active
        FROM messages 
        WHERE 1=1
        """
        
        params = []
        if days:
            query += " AND timestamp >= datetime('now', '-' || ? || ' days')"
            params.append(days)
        
        query += """
        GROUP BY author_name 
        ORDER BY message_count DESC 
        LIMIT ?
        """
        params.append(limit)
        
        results = self.execute_query(query, tuple(params))
        
        return [
            {
                'username': row['author_name'],
                'message_count': row['message_count'],
                'channels_active': row['channels_active']
            }
            for row in results
        ] if results else []
    
    def get_top_channels(self, limit: int = 10, days: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get top channels by message count."""
        
        query = """
        SELECT 
            channel_name,
            COUNT(*) as message_count,
            COUNT(DISTINCT author_name) as unique_users
        FROM messages 
        WHERE 1=1
        """
        
        params = []
        if days:
            query += " AND timestamp >= datetime('now', '-' || ? || ' days')"
            params.append(days)
        
        query += """
        GROUP BY channel_name 
        ORDER BY message_count DESC 
        LIMIT ?
        """
        params.append(limit)
        
        results = self.execute_query(query, tuple(params))
        
        return [
            {
                'channel_name': row['channel_name'],
                'message_count': row['message_count'],
                'unique_users': row['unique_users']
            }
            for row in results
        ] if results else []
    
    def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        
        try:
            # Test basic query
            result = self.execute_query("SELECT COUNT(*) as total FROM messages", fetch_one=True)
            total_messages = result['total_messages'] if result else 0
            
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


# Global instance for easy access
db_manager = DatabaseManager() 