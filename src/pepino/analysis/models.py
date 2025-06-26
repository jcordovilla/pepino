"""
Analysis response models for analyzer outputs.
"""

from typing import List, Literal, Optional, Union

from pydantic import BaseModel


# Base activity models for V2 analyzers
class UserActivity(BaseModel):
    """User activity data for channel analysis."""
    
    username: str
    display_name: Optional[str] = None
    message_count: int
    avg_message_length: float = 0.0
    first_message_date: Optional[str] = None
    last_message_date: Optional[str] = None


class ChannelActivity(BaseModel):
    """Channel activity data for user analysis."""
    
    channel_name: str
    message_count: int
    first_message_date: Optional[str] = None
    last_message_date: Optional[str] = None


# V2 Local statistics models  
class LocalChannelStatistics(BaseModel):
    """Basic channel statistics for V2 analyzers."""
    
    total_messages: int
    unique_users: int
    messages_per_day: float
    avg_message_length: float = 0.0
    first_message_date: Optional[str] = None
    last_message_date: Optional[str] = None
    analysis_period_days: Optional[int] = None
    bot_messages: int = 0
    human_messages: int = 0
    unique_human_users: int = 0


class LocalUserStatistics(BaseModel):
    """Basic user statistics for V2 analyzers."""
    
    total_messages: int
    unique_channels: int
    messages_per_day: float
    first_message_date: Optional[str] = None
    last_message_date: Optional[str] = None
    analysis_period_days: Optional[int] = None


# Base response models
class AnalysisResponseBase(BaseModel):
    """Base class for all analysis responses."""

    success: bool
    error: Optional[str] = None
    plugin: Optional[str] = None
    capabilities_used: List[str] = []


class AnalysisErrorResponse(AnalysisResponseBase):
    """Error response for failed analysis."""

    success: bool = False
    error: str


# User Analysis Models
class UserInfo(BaseModel):
    """User information from analysis."""

    author_id: str
    display_name: str


class UserStatistics(BaseModel):
    """User statistics from repository."""

    author_id: str
    author_name: str
    message_count: int
    channels_active: int
    avg_message_length: float
    first_message_date: Optional[str] = None
    last_message_date: Optional[str] = None


class UserAnalysisResponse(AnalysisResponseBase):
    """Response from UserAnalyzer.analyze()."""

    success: bool = True
    plugin: str = "UserAnalyzer"
    user_info: UserInfo
    statistics: UserStatistics
    concepts: List[str] = []
    capabilities_used: List[str] = ["user_analysis", "user_stats"]


# Channel Analysis Models
class ChannelInfo(BaseModel):
    """Channel information from analysis."""

    channel_name: str


class ChannelStatistics(BaseModel):
    """Comprehensive channel statistics from repository."""

    total_messages: int
    unique_users: int
    avg_message_length: float
    first_message: Optional[str] = None
    last_message: Optional[str] = None
    active_days: int = 0
    bot_messages: int = 0
    human_messages: int = 0
    unique_human_users: int = 0


class EngagementMetrics(BaseModel):
    """Channel engagement metrics."""
    
    total_replies: int
    original_posts: int
    posts_with_reactions: int
    replies_per_post: float
    reaction_rate: float


class PeakActivityHour(BaseModel):
    """Peak activity hour data."""
    
    hour: str
    messages: int


class PeakActivityDay(BaseModel):
    """Peak activity day data."""
    
    day: str
    messages: int


class PeakActivity(BaseModel):
    """Peak activity analysis."""
    
    peak_hours: List[PeakActivityHour]
    peak_days: List[PeakActivityDay]


class RecentActivityItem(BaseModel):
    """Recent activity data point."""
    
    date: str
    messages: int


class HealthMetrics(BaseModel):
    """Channel health metrics."""
    
    weekly_active: int
    inactive_users: int
    total_channel_members: int = 0
    lurkers: int = 0
    participation_rate: float = 0.0


class TopUserInChannel(BaseModel):
    """Top user data for channel analysis."""

    author_id: str
    author_name: str
    display_name: str
    message_count: int
    avg_message_length: float


class DailyActivityData(BaseModel):
    """Daily activity data for chart generation."""
    
    dates: List[str]
    counts: List[int] 
    average: float


class ChannelAnalysisResponse(AnalysisResponseBase):
    """Comprehensive response from ChannelAnalyzer.analyze()."""

    success: bool = True
    plugin: str = "ChannelAnalyzer"
    channel_info: ChannelInfo
    statistics: ChannelStatistics
    top_users: List[TopUserInChannel] = []
    engagement_metrics: Optional[EngagementMetrics] = None
    peak_activity: Optional[PeakActivity] = None
    recent_activity: List[RecentActivityItem] = []
    health_metrics: Optional[HealthMetrics] = None
    top_topics: List[str] = []
    daily_activity_data: Optional[DailyActivityData] = None
    capabilities_used: List[str] = ["channel_analysis", "channel_stats", "engagement_analysis", "peak_activity", "health_metrics"]


# Topic Analysis Models
class TopicItem(BaseModel):
    """Individual topic with frequency and relevance."""

    topic: str
    frequency: int
    relevance_score: float


class TopicAnalysisResponse(AnalysisResponseBase):
    """Response from TopicAnalyzer.analyze()."""

    success: bool = True
    plugin: str = "TopicAnalyzer"
    topics: List[TopicItem]
    message_count: int
    capabilities_used: List[str] = ["topic_analysis", "word_frequency"]


# Temporal Analysis Models
class TemporalDataPoint(BaseModel):
    """Single temporal data point."""

    period: str
    message_count: int
    unique_users: int


class TemporalPatterns(BaseModel):
    """Analyzed temporal patterns."""

    total_messages: int
    avg_messages_per_period: float
    max_messages_in_period: int
    min_messages_in_period: int
    most_active_period: Optional[str] = None
    message_trend: Literal["increasing", "decreasing", "stable"]
    trend_percentage: float
    peak_user_count: int
    total_periods: int


class TemporalAnalysisResponse(AnalysisResponseBase):
    """Response from TemporalAnalyzer.analyze()."""

    success: bool = True
    plugin: str = "TemporalAnalyzer"
    temporal_data: List[TemporalDataPoint]
    patterns: TemporalPatterns
    capabilities_used: List[str] = ["temporal_analysis", "activity_patterns"]


# Union type for all possible analyzer responses
AnalysisResponse = Union[
    UserAnalysisResponse,
    ChannelAnalysisResponse,
    TopicAnalysisResponse,
    TemporalAnalysisResponse,
    AnalysisErrorResponse,
]
