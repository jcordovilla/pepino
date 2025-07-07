"""
Analysis package for Discord data analysis.

Primary Interface:
    AnalysisService: Main service for all analysis operations
    analysis_service: Context manager for convenient usage

Core Components:
    - AnalysisService: Orchestrates all analysis operations
    - AnalysisDataFacade: Centralized repository management
    - Individual Analyzers: Specialized analysis components

The service layer provides a clean interface while hiding internal complexity.
Individual analyzers are available for advanced use cases.
"""

# Primary service interface (recommended usage)
from .service import AnalysisService, analysis_service

# Data facade for advanced usage
from .helpers.data_facade import AnalysisDataFacade, get_analysis_data_facade, analysis_transaction

# Core analyzers (for advanced usage)
from .helpers.user_analyzer import UserAnalyzer
from .helpers.channel_analyzer import ChannelAnalyzer
from .helpers.topic_analyzer import TopicAnalyzer
from .helpers.temporal_analyzer import TemporalAnalyzer
from .helpers.database_analyzer import DatabaseAnalyzer

# Response models (for type hints and validation)
from .models import (
    AnalysisErrorResponse,
    AnalysisResponse,
    ChannelAnalysisResponse,
    ChannelInfo,
    ChannelStatistics,
    TemporalAnalysisResponse,
    TemporalDataPoint,
    TemporalPatterns,
    TopicAnalysisResponse,
    TopicItem,
    TopUserInChannel,
    UserAnalysisResponse,
    UserInfo,
    UserStatistics,
)

# Template system
from .templates import TemplateEngine

__all__ = [
    # Primary service interface (recommended)
    "AnalysisService",
    "analysis_service",
    # Data facade
    "AnalysisDataFacade",
    "get_analysis_data_facade", 
    "analysis_transaction",
    # Core analyzers (advanced usage)
    "UserAnalyzer",
    "ChannelAnalyzer",
    "TopicAnalyzer",
    "TemporalAnalyzer",
    "DatabaseAnalyzer",
    # Response models
    "AnalysisResponse",
    "AnalysisErrorResponse",
    "UserAnalysisResponse",
    "ChannelAnalysisResponse",
    "TopicAnalysisResponse",
    "TemporalAnalysisResponse",
    "DatabaseAnalysisResponse",
    "UserInfo",
    "UserStatistics",
    "ChannelInfo",
    "ChannelStatistics",
    "TopUserInChannel",
    "TopicItem",
    "TemporalDataPoint",
    "TemporalPatterns",
    "DatabaseInfo",
    "TableStatistics",
    "DatabaseSummary",
    # Template system
    "TemplateEngine",
]
