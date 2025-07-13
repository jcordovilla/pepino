"""
Unified Analysis Service

Orchestrates all specialized analysis services while maintaining the same public interface
as the original AnalysisService. This provides a clean facade over the modularized services.
"""

import logging
from typing import Dict, Any, Optional, Literal
from contextlib import contextmanager
from datetime import datetime

from pepino.config import Settings
from .base_service import BaseAnalysisService, OutputFormat
from .channel_analysis_service import ChannelAnalysisService
from .user_analysis_service import UserAnalysisService
from .topic_analysis_service import TopicAnalysisService
from .temporal_analysis_service import TemporalAnalysisService
from .server_analysis_service import ServerAnalysisService
from .database_analysis_service import DatabaseAnalysisService

logger = logging.getLogger(__name__)


class UnifiedAnalysisService:
    """
    Unified analysis service that orchestrates specialized services.
    
    Maintains the same public interface as the original AnalysisService while
    delegating to specialized services for better modularity and maintainability.
    """
    
    def __init__(self, db_path: Optional[str] = None, base_filter: Optional[str] = None):
        """
        Initialize unified analysis service with specialized services.
        
        Args:
            db_path: Optional database path (uses settings default if None)
            base_filter: Optional base filter for data queries
        """
        self.settings = Settings()
        self.db_path = db_path or self.settings.database_sqlite_path
        self.base_filter = base_filter or self.settings.analysis_base_filter_sql
        
        # Initialize specialized services
        self._channel_service = None
        self._user_service = None
        self._topic_service = None
        self._temporal_service = None
        self._server_service = None
        self._database_service = None
        
        logger.debug(f"UnifiedAnalysisService initialized for {self.db_path}")
    
    @property
    def channel_service(self) -> ChannelAnalysisService:
        """Get channel analysis service (lazy initialization)."""
        if self._channel_service is None:
            self._channel_service = ChannelAnalysisService(self.db_path, self.base_filter)
        return self._channel_service
    
    @property
    def user_service(self) -> UserAnalysisService:
        """Get user analysis service (lazy initialization)."""
        if self._user_service is None:
            self._user_service = UserAnalysisService(self.db_path, self.base_filter)
        return self._user_service
    
    @property
    def topic_service(self) -> TopicAnalysisService:
        """Get topic analysis service (lazy initialization)."""
        if self._topic_service is None:
            self._topic_service = TopicAnalysisService(self.db_path, self.base_filter)
        return self._topic_service
    
    @property
    def temporal_service(self) -> TemporalAnalysisService:
        """Get temporal analysis service (lazy initialization)."""
        if self._temporal_service is None:
            self._temporal_service = TemporalAnalysisService(self.db_path, self.base_filter)
        return self._temporal_service
    
    @property
    def server_service(self) -> ServerAnalysisService:
        """Get server analysis service (lazy initialization)."""
        if self._server_service is None:
            self._server_service = ServerAnalysisService(self.db_path, self.base_filter)
        return self._server_service
    
    @property
    def database_service(self) -> DatabaseAnalysisService:
        """Get database analysis service (lazy initialization)."""
        if self._database_service is None:
            self._database_service = DatabaseAnalysisService(self.db_path, self.base_filter)
        return self._database_service
    
    @property
    def template_engine(self):
        """Expose the template engine from the channel service for compatibility."""
        return self.channel_service.template_engine

    # Channel Analysis Methods
    def pulsecheck(self, channel_name: Optional[str] = None, days_back: int = 7, 
                   end_date: Optional[datetime] = None, output_format: OutputFormat = "cli") -> str:
        """Delegate to channel service."""
        return self.channel_service.pulsecheck(channel_name, days_back, end_date, output_format)
    
    def top_channels(self, limit: int = 5, days_back: int = 7, 
                    end_date: Optional[datetime] = None, output_format: OutputFormat = "cli") -> str:
        """Delegate to channel service."""
        return self.channel_service.top_channels(limit, days_back, end_date, output_format)
    
    def list_channels(self, output_format: OutputFormat = "cli") -> str:
        """Delegate to channel service."""
        return self.channel_service.list_channels(output_format)
    
    # User Analysis Methods
    def top_contributors(self, channel_name: Optional[str] = None, limit: int = 10, 
                        days_back: int = 30, end_date: Optional[datetime] = None, 
                        output_format: OutputFormat = "cli") -> str:
        """Delegate to user service."""
        return self.user_service.top_contributors(channel_name, limit, days_back, end_date, output_format)
    
    def detailed_user_analysis(self, username: str, days_back: int = 30, 
                              output_format: OutputFormat = "cli") -> str:
        """Delegate to user service."""
        return self.user_service.detailed_user_analysis(username, days_back, output_format)
    
    # Topic Analysis Methods
    def detailed_topic_analysis(self, channel_name: Optional[str] = None, n_topics: int = 10, 
                               days_back: Optional[int] = None, output_format: OutputFormat = "cli") -> str:
        """Delegate to topic service."""
        return self.topic_service.detailed_topic_analysis(channel_name, n_topics, days_back, output_format)
    
    # Temporal Analysis Methods
    def detailed_temporal_analysis(self, channel_name: Optional[str] = None, 
                                  days_back: Optional[int] = None, granularity: str = "daily", 
                                  output_format: OutputFormat = "cli") -> str:
        """Delegate to temporal service."""
        return self.temporal_service.detailed_temporal_analysis(channel_name, days_back, granularity, output_format)
    
    def activity_trends_analysis(self, channel_name: Optional[str] = None, 
                                days_back: Optional[int] = None, output_format: OutputFormat = "discord") -> str:
        """Delegate to temporal service."""
        return self.temporal_service.activity_trends_analysis(channel_name, days_back, output_format)
    
    # Server Analysis Methods
    def server_overview_analysis(self, days_back: Optional[int] = None, 
                                output_format: OutputFormat = "discord") -> str:
        """Delegate to server service."""
        return self.server_service.server_overview_analysis(days_back, output_format)
    
    # Database Analysis Methods
    def database_stats(self, output_format: OutputFormat = "cli") -> str:
        """Delegate to database service."""
        return self.database_service.database_stats(output_format)
    
    # Template Rendering Methods
    def render_template(self, template_name: str, output_format: OutputFormat = "cli", **kwargs) -> str:
        """Render a template with the specified format."""
        # Use the channel service's template engine for rendering
        return self.channel_service.render_template(template_name, output_format, **kwargs)
    
    def close(self):
        """Close all specialized services and clean up resources."""
        if self._channel_service:
            self._channel_service.close()
        if self._user_service:
            self._user_service.close()
        if self._topic_service:
            self._topic_service.close()
        if self._temporal_service:
            self._temporal_service.close()
        if self._server_service:
            self._server_service.close()
        if self._database_service:
            self._database_service.close()
        logger.debug("UnifiedAnalysisService closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience function for common operations (maintains backward compatibility)
@contextmanager
def analysis_service(db_path: Optional[str] = None, base_filter: Optional[str] = None):
    """
    Context manager for analysis operations.
    
    Args:
        db_path: Optional database path (uses settings default if None)
        base_filter: Optional base filter for data queries (uses settings default if None)
        
    Yields:
        UnifiedAnalysisService instance
    """
    # Use settings for defaults if not provided
    settings = Settings()
    actual_db_path = db_path or settings.database_sqlite_path
    actual_base_filter = base_filter or settings.analysis_base_filter_sql
    service = UnifiedAnalysisService(db_path=actual_db_path, base_filter=actual_base_filter)
    try:
        yield service
    finally:
        service.close() 