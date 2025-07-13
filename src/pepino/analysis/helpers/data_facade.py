"""
Analysis data facade for centralized repository management.

Encapsulates all repository initialization and transactional behavior for the analysis layer.
Provides a clean interface for analyzers to access data without managing repositories directly.
"""

import logging
from contextlib import contextmanager
from typing import Optional, List, Dict

from ...config import Settings
from ...data.database.manager import DatabaseManager
from ...data.repositories import (
    ChannelRepository,
    MessageRepository,
    UserRepository,
    DatabaseRepository,
)

logger = logging.getLogger(__name__)


class AnalysisDataFacade:
    """
    Data facade for analysis operations.
    
    Encapsulates all repository initialization and transactional behavior,
    providing a single point of access for all data operations in analysis.
    
    Features:
    - Centralized repository management
    - Transaction support
    - Connection pooling
    - Error handling and logging
    - Consistent configuration
    - Base filter management for dependency injection and testability
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None, base_filter: Optional[str] = None):
        """
        Initialize the analysis data facade.
        
        Args:
            db_manager: Optional database manager. If None, creates a new one.
            base_filter: Optional base filter for data queries. If None, uses Settings default.
        """
        self.settings = Settings()
        
        if db_manager is None:
            self.db_manager = DatabaseManager(self.settings.database_sqlite_path)
            self._owns_db_manager = True
        else:
            self.db_manager = db_manager
            self._owns_db_manager = False
            
        # Handle base filter - use provided or fallback to settings
        if base_filter is None:
            self.base_filter = self.settings.analysis_base_filter_sql
        else:
            self.base_filter = base_filter
        
        # Initialize repositories
        self._user_repository = None
        self._channel_repository = None
        self._message_repository = None
        self._database_repository = None
        
        logger.info("AnalysisDataFacade initialized with dependency injection support")
    
    @property
    def user_repository(self) -> UserRepository:
        """Get user repository instance with configured base filter."""
        if self._user_repository is None:
            self._user_repository = UserRepository(self.db_manager)
            # Override the base filter with our configured one
            self._user_repository.base_filter = self.base_filter.strip()
        return self._user_repository
    
    @property
    def channel_repository(self) -> ChannelRepository:
        """Get channel repository instance with configured base filter."""
        if self._channel_repository is None:
            self._channel_repository = ChannelRepository(self.db_manager)
            # Override the base filter with our configured one
            self._channel_repository.base_filter = self.base_filter.strip()
        return self._channel_repository
    
    @property
    def message_repository(self) -> MessageRepository:
        """Get message repository instance with configured base filter."""
        if self._message_repository is None:
            self._message_repository = MessageRepository(self.db_manager)
            # Override the base filter with our configured one
            self._message_repository.base_filter = self.base_filter.strip()
        return self._message_repository
    
    @property
    def database_repository(self) -> DatabaseRepository:
        """Get database repository instance."""
        if self._database_repository is None:
            self._database_repository = DatabaseRepository(self.db_manager)
        return self._database_repository
    
    @contextmanager
    def transaction(self):
        """
        Context manager for database transactions.
        
        Provides transaction isolation for analysis operations that
        require consistency across multiple repository calls.
        """
        with self.db_manager.get_connection() as conn:
            try:
                conn.execute("BEGIN")
                yield self
                conn.execute("COMMIT")
            except Exception as e:
                conn.execute("ROLLBACK")
                logger.error(f"Transaction rolled back due to error: {e}")
                raise
    
    def get_cross_channel_summary(self, days_back: int = 7, limit: int = 10) -> List[Dict]:
        """
        Get cross-channel summary analysis for all channels (humans only).
        
        Args:
            days_back: Number of days to look back
            limit: Maximum number of channels to return
            
        Returns:
            List of channel summary dictionaries with name, message_count, etc.
        """
        try:
            # Get all available channels
            all_channels = self.channel_repository.get_available_channels()
            channels_summary = []
            
            for channel_name in all_channels[:20]:  # Limit initial processing to top 20
                try:
                    # Get message count for this channel in the time period (humans only)
                    channel_messages = self.message_repository.get_channel_messages(
                        channel_name, 
                        days_back=days_back
                    )
                    
                    if channel_messages:
                        # Filter out bot messages
                        human_messages = [
                            msg for msg in channel_messages 
                            if not msg.get('author_is_bot', False)
                        ]
                        
                        message_count = len(human_messages)
                        
                        if message_count > 0:
                            # Calculate average message length for human messages only
                            total_length = sum(len(msg.get('content', '')) for msg in human_messages)
                            avg_length = total_length / message_count if message_count > 0 else 0
                            
                            # Get unique human users
                            unique_human_users = len(set(
                                msg.get('username') for msg in human_messages 
                                if msg.get('username') and not msg.get('author_is_bot', False)
                            ))
                            
                            channels_summary.append({
                                'name': channel_name,
                                'message_count': message_count,
                                'avg_length': round(avg_length, 1),
                                'unique_users': unique_human_users
                            })
                except Exception as e:
                    logger.debug(f"Could not get data for channel {channel_name}: {e}")
                    continue
            
            # Sort by message count (most active first) and limit results
            channels_summary.sort(key=lambda x: x['message_count'], reverse=True)
            return channels_summary[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get cross-channel summary: {e}")
            return []
    
    def close(self):
        """Close the data facade and clean up resources."""
        if self._owns_db_manager and self.db_manager:
            self.db_manager.close_connections()
            logger.info("AnalysisDataFacade closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience function for getting a data facade instance
def get_analysis_data_facade(db_manager: Optional[DatabaseManager] = None, base_filter: Optional[str] = None) -> AnalysisDataFacade:
    """
    Get an analysis data facade instance.
    
    Args:
        db_manager: Optional database manager. If None, creates a new one.
        base_filter: Optional base filter for data queries. If None, uses Settings default.
        
    Returns:
        AnalysisDataFacade instance
    """
    return AnalysisDataFacade(db_manager, base_filter)


# Context manager for transactional analysis operations
@contextmanager
def analysis_transaction(db_manager: Optional[DatabaseManager] = None, base_filter: Optional[str] = None):
    """
    Context manager for analysis operations requiring transaction support.
    
    Args:
        db_manager: Optional database manager. If None, creates a new one.
        base_filter: Optional base filter for data queries. If None, uses Settings default.
        
    Yields:
        AnalysisDataFacade instance within a transaction context
    """
    with get_analysis_data_facade(db_manager, base_filter) as facade:
        with facade.transaction():
            yield facade 