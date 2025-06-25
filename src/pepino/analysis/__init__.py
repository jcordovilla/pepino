"""
Analysis package for Discord data analysis.

Provides analyzer instances and data facade for centralized repository management:
- UserAnalyzer: User activity and statistics
- ChannelAnalyzer: Channel activity and insights  
- TopicAnalyzer: Topic and keyword analysis
- TemporalAnalyzer: Activity patterns over time
- AnalysisDataFacade: Centralized repository and transaction management

The data facade encapsulates all repository initialization and transactional behavior,
providing a clean interface for analyzers without direct repository management.
"""

from .channel_analyzer import ChannelAnalyzer
from .data_facade import AnalysisDataFacade, get_analysis_data_facade, analysis_transaction
# Removed imports with external dependencies for now
# from .conversation_analyzer import ConversationService
# from .nlp_analyzer import NLPService
# from .similarity_analyzer import SimilarityService

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
from .temporal_analyzer import TemporalAnalyzer
from .topic_analyzer import TopicAnalyzer

# Core analyzers
from .user_analyzer import UserAnalyzer

__all__ = [
    # Core analyzers
    "UserAnalyzer",
    "ChannelAnalyzer",
    "TopicAnalyzer",
    "TemporalAnalyzer",
    # Data facade
    "AnalysisDataFacade",
    "get_analysis_data_facade",
    "analysis_transaction",
    # Models
    "AnalysisResponse",
    "AnalysisErrorResponse",
    "UserAnalysisResponse",
    "ChannelAnalysisResponse",
    "TopicAnalysisResponse",
    "TemporalAnalysisResponse",
    "UserInfo",
    "UserStatistics",
    "ChannelInfo",
    "ChannelStatistics",
    "TopUserInChannel",
    "TopicItem",
    "TemporalDataPoint",
    "TemporalPatterns",
]
