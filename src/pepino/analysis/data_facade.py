"""
Analysis data facade for centralized repository management.

Encapsulates all repository initialization and transactional behavior for the analysis layer.
Provides a clean interface for analyzers to access data without managing repositories directly.
"""

import logging
from contextlib import contextmanager
from typing import Optional

from ..data.config import Settings
from ..data.database.manager import DatabaseManager
from ..data.repositories import (
    ChannelRepository,
    EmbeddingRepository,
    MessageRepository,
    UserRepository,
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
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize the analysis data facade.
        
        Args:
            db_manager: Optional database manager. If None, creates a new one.
        """
        self.settings = Settings()
        
        if db_manager is None:
            self.db_manager = DatabaseManager(self.settings.db_path)
            self._owns_db_manager = True
        else:
            self.db_manager = db_manager
            self._owns_db_manager = False
        
        # Initialize repositories
        self._user_repository = None
        self._channel_repository = None
        self._message_repository = None
        self._embedding_repository = None
        
        logger.info("AnalysisDataFacade initialized")
    
    @property
    def user_repository(self) -> UserRepository:
        """Get user repository instance."""
        if self._user_repository is None:
            self._user_repository = UserRepository(self.db_manager)
        return self._user_repository
    
    @property
    def channel_repository(self) -> ChannelRepository:
        """Get channel repository instance."""
        if self._channel_repository is None:
            self._channel_repository = ChannelRepository(self.db_manager)
        return self._channel_repository
    
    @property
    def message_repository(self) -> MessageRepository:
        """Get message repository instance."""
        if self._message_repository is None:
            self._message_repository = MessageRepository(self.db_manager)
        return self._message_repository
    
    @property
    def embedding_repository(self) -> EmbeddingRepository:
        """Get embedding repository instance."""
        if self._embedding_repository is None:
            self._embedding_repository = EmbeddingRepository(self.db_manager)
        return self._embedding_repository
    
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
def get_analysis_data_facade(db_manager: Optional[DatabaseManager] = None) -> AnalysisDataFacade:
    """
    Get an analysis data facade instance.
    
    Args:
        db_manager: Optional database manager. If None, creates a new one.
        
    Returns:
        AnalysisDataFacade instance
    """
    return AnalysisDataFacade(db_manager)


# Context manager for transactional analysis operations
@contextmanager
def analysis_transaction(db_manager: Optional[DatabaseManager] = None):
    """
    Context manager for analysis operations requiring transaction support.
    
    Args:
        db_manager: Optional database manager. If None, creates a new one.
        
    Yields:
        AnalysisDataFacade instance within a transaction context
    """
    with get_analysis_data_facade(db_manager) as facade:
        with facade.transaction():
            yield facade 