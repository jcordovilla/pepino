"""
User Analyzer 

Synchronous user analysis using the data facade pattern for repository management.
Provides comprehensive user activity analysis with proper separation of concerns.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .data_facade import AnalysisDataFacade, get_analysis_data_facade
from ..models import UserAnalysisResponse, LocalUserStatistics, ChannelActivity
from ..models import UserInfo
from ..models import UserStatistics as ModelUserStatistics

logger = logging.getLogger(__name__)


class UserAnalyzer:
    """
    Synchronous User Analyzer 
    
    Analyzes user activity patterns, channel engagement, and messaging behavior
    using the data facade pattern for centralized repository management.
    
    Features:
    - Message count analysis through data facade layer
    - Channel activity distribution
    - Time-based activity patterns
    - User engagement metrics
    - Cross-channel behavior analysis
    
    All database operations are abstracted through the data facade for proper
    separation of concerns and dependency injection support.
    """
    
    def __init__(self, data_facade: Optional[AnalysisDataFacade] = None):
        """Initialize user analyzer with data facade."""
        if data_facade is None:
            self.data_facade = get_analysis_data_facade()
            self._owns_facade = True
        else:
            self.data_facade = data_facade
            self._owns_facade = False
            
        logger.info("UserAnalyzer initialized with data facade pattern")
    
    def analyze(
        self, 
        username: str,
        days: Optional[int] = None,
        include_patterns: bool = True
    ) -> Optional[UserAnalysisResponse]:
        """
        Analyze a user comprehensively using repository layer.
        
        Args:
            username: Name of the user to analyze
            days: Number of days to look back (None = all time)
            include_patterns: Whether to include time patterns analysis
            
        Returns:
            UserAnalysisResponse object with comprehensive results
        """
        
        try:
            logger.info(f"Starting user analysis for: {username}")
            
            # Get basic statistics through repository
            statistics = self._get_user_statistics_via_repository(username, days)
            if not statistics or statistics.total_messages == 0:
                logger.warning(f"No messages found for user: {username}")
                return None
            
            # Get channel activity through repository
            channel_activity = self._get_user_channel_activity_via_repository(username, days)
            
            # Get time patterns if requested
            time_patterns = {}
            if include_patterns:
                time_patterns = self._analyze_time_patterns_via_repository(username, days)
            
            # Create summary
            summary = self._create_user_summary(statistics, channel_activity, time_patterns)
            
            # Build result
            
            analysis = UserAnalysisResponse(
                user_info=UserInfo(
                    author_id=username,  # Use username as ID for now
                    display_name=username
                ),
                statistics=self._convert_to_user_statistics_model(statistics, username)
            )
            
            logger.info(f"User analysis completed for: {username}")
            return analysis
            
        except Exception as e:
            logger.error(f"User analysis failed for {username}: {e}")
            return None
    
    def _convert_to_user_statistics_model(self, stats: LocalUserStatistics, username: str):
        """Convert our LocalUserStatistics to the model expected by UserAnalysisResponse."""
        
        return ModelUserStatistics(
            author_id=username,
            author_name=username,
            message_count=stats.total_messages,
            channels_active=stats.unique_channels,
            avg_message_length=0.0,  # We don't calculate this yet
            first_message_date=stats.first_message_date,
            last_message_date=stats.last_message_date
        )
    
    def _get_user_statistics_via_repository(self, username: str, days: Optional[int]) -> Optional[LocalUserStatistics]:
        """Get basic user statistics using data facade pattern."""
        
        try:
            # Use data facade for repository access
            stats_data = self.data_facade.user_repository.get_user_message_statistics(username, days)
            
            if not stats_data or stats_data.get('total_messages', 0) == 0:
                return None
            
            # Calculate messages per day
            total_messages = stats_data['total_messages']
            if total_messages > 0 and stats_data['first_message'] and stats_data['last_message']:
                first_date = datetime.fromisoformat(stats_data['first_message'].replace('Z', '+00:00'))
                last_date = datetime.fromisoformat(stats_data['last_message'].replace('Z', '+00:00'))
                time_span = (last_date - first_date).days + 1
                messages_per_day = total_messages / max(time_span, 1)
            else:
                messages_per_day = 0
            
            return LocalUserStatistics(
                total_messages=stats_data['total_messages'],
                unique_channels=stats_data['unique_channels'],
                messages_per_day=round(messages_per_day, 2),
                first_message_date=stats_data['first_message'],
                last_message_date=stats_data['last_message'],
                analysis_period_days=days
            )
            
        except Exception as e:
            logger.error(f"Failed to get user statistics via repository: {e}")
            return None
    
    def _get_user_channel_activity_via_repository(
        self, 
        username: str, 
        days: Optional[int],
        limit: int = 20
    ) -> List[ChannelActivity]:
        """Get user's activity distribution across channels using data facade."""
        
        try:
            # Use data facade for repository access
            activity_data = self.data_facade.user_repository.get_user_channel_activity(username, days, limit)
            
            channel_activities = []
            for data in activity_data:
                activity = ChannelActivity(
                    channel_name=data['channel_name'],
                    message_count=data['message_count'],
                    first_message=data['first_message'],
                    last_message=data['last_message']
                )
                channel_activities.append(activity)
            
            return channel_activities
            
        except Exception as e:
            logger.error(f"Failed to get user channel activity via repository: {e}")
            return []

    def _analyze_time_patterns_via_repository(self, username: str, days: Optional[int]) -> Dict[str, any]:
        """Analyze user time patterns using data facade."""
        
        try:
            # Get hourly patterns through data facade
            hourly_data = self.data_facade.user_repository.get_user_hourly_patterns(username, days)
            
            # Process hourly data
            hourly_activity = {}
            peak_hour = 0
            max_messages = 0
            
            for entry in hourly_data:
                hour = entry['hour']
                count = entry['message_count']
                hourly_activity[hour] = count
                
                if count > max_messages:
                    max_messages = count
                    peak_hour = hour
            
            # Determine activity period
            if peak_hour >= 6 and peak_hour < 12:
                activity_period = "morning"
            elif peak_hour >= 12 and peak_hour < 18:
                activity_period = "afternoon"
            elif peak_hour >= 18 and peak_hour < 22:
                activity_period = "evening"
            else:
                activity_period = "night"
            
            return {
                'hourly_activity': hourly_activity,
                'peak_hour': peak_hour,
                'activity_period': activity_period,
                'total_active_hours': len(hourly_activity)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze time patterns via repository: {e}")
            return {}
    
    def _create_user_summary(
        self,
        statistics: LocalUserStatistics,
        channel_activity: List[ChannelActivity],
        time_patterns: Dict[str, any]
    ) -> Dict[str, any]:
        """Create summary of user analysis results."""
        
        # Channel diversity
        if statistics.unique_channels > 5:
            channel_diversity = "high"
        elif statistics.unique_channels > 2:
            channel_diversity = "medium"
        else:
            channel_diversity = "low"
        
        # Activity level
        if statistics.messages_per_day > 10:
            activity_level = "very active"
        elif statistics.messages_per_day > 5:
            activity_level = "active"
        elif statistics.messages_per_day > 1:
            activity_level = "moderate"
        else:
            activity_level = "low"
        
        # Most active channel
        most_active_channel = None
        if channel_activity:
            most_active_channel = channel_activity[0].channel_name
        
        # Time preference
        time_preference = time_patterns.get('activity_period', 'unknown')
        
        return {
            'activity_level': activity_level,
            'channel_diversity': channel_diversity,
            'most_active_channel': most_active_channel,
            'time_preference': time_preference,
            'total_channels': statistics.unique_channels,
            'avg_daily_messages': statistics.messages_per_day
        }

    def get_available_users(self) -> List[str]:
        """Get list of available users using data facade."""
        try:
            return self.data_facade.user_repository.get_available_users()
        except Exception as e:
            logger.error(f"Failed to get available users: {e}")
            return []

    def get_top_users(self, limit: int = 10, days: Optional[int] = None) -> List[Dict[str, any]]:
        """Get top users by message count using data facade."""
        try:
            return self.data_facade.user_repository.get_top_users_by_message_count(limit, days)
        except Exception as e:
            logger.error(f"Failed to get top users: {e}")
            return []

    def get_user_health(self, username: str) -> Dict[str, any]:
        """Get user engagement health metrics using repository."""
        
        try:
            # Get basic stats
            recent_stats = self.user_repository.get_user_message_statistics(username, days=7)
            overall_stats = self.user_repository.get_user_message_statistics(username, days=None)
            
            if not overall_stats or overall_stats.get('total_messages', 0) == 0:
                return {
                    'status': 'inactive',
                    'health_score': 0,
                    'recommendations': ['User has no message history']
                }
            
            # Calculate health metrics
            recent_messages = recent_stats.get('total_messages', 0)
            total_messages = overall_stats.get('total_messages', 0)
            channels = overall_stats.get('unique_channels', 0)
            
            # Health score calculation
            health_score = 0
            
            # Recent activity (30% weight)
            if recent_messages > 10:
                health_score += 30
            elif recent_messages > 5:
                health_score += 20
            elif recent_messages > 0:
                health_score += 10
            
            # Channel diversity (30% weight)
            if channels > 5:
                health_score += 30
            elif channels > 2:
                health_score += 20
            elif channels > 0:
                health_score += 10
            
            # Overall engagement (40% weight)
            if total_messages > 1000:
                health_score += 40
            elif total_messages > 100:
                health_score += 30
            elif total_messages > 10:
                health_score += 20
            else:
                health_score += 10
            
            # Determine status
            if health_score >= 80:
                status = 'excellent'
            elif health_score >= 60:
                status = 'good'
            elif health_score >= 40:
                status = 'fair'
            else:
                status = 'needs_attention'
            
            # Generate recommendations
            recommendations = []
            if recent_messages == 0:
                recommendations.append('Consider encouraging recent participation')
            if channels <= 2:
                recommendations.append('User could benefit from engaging in more channels')
            if total_messages < 50:
                recommendations.append('New user - consider welcoming initiatives')
            
            return {
                'status': status,
                'health_score': health_score,
                'recent_activity': recent_messages,
                'total_activity': total_messages,
                'channel_diversity': channels,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Failed to get user health for {username}: {e}")
            return {
                'status': 'error',
                'health_score': 0,
                'recommendations': [f'Error analyzing user: {e}']
            }

    def get_user_channel_comparison(
        self, 
        username: str, 
        days: Optional[int] = None
    ) -> Dict[str, any]:
        """Compare user's activity across different channels using repository."""
        
        try:
            channel_activity = self.user_repository.get_user_channel_activity(username, days, limit=50)
            
            if not channel_activity:
                return {
                    'total_channels': 0,
                    'channels': [],
                    'primary_channel': None,
                    'diversity_score': 0
                }
            
            total_messages = sum(activity['message_count'] for activity in channel_activity)
            
            # Calculate percentages and create comparison
            channels = []
            for activity in channel_activity:
                percentage = (activity['message_count'] / total_messages * 100) if total_messages > 0 else 0
                channels.append({
                    'name': activity['channel_name'],
                    'messages': activity['message_count'],
                    'percentage': round(percentage, 1),
                    'first_message': activity['first_message'],
                    'last_message': activity['last_message']
                })
            
            # Primary channel (highest activity)
            primary_channel = channels[0]['name'] if channels else None
            
            # Diversity score (1 - Herfindahl index)
            if total_messages > 0:
                herfindahl = sum((activity['message_count'] / total_messages) ** 2 for activity in channel_activity)
                diversity_score = round((1 - herfindahl) * 100, 1)
            else:
                diversity_score = 0
            
            return {
                'total_channels': len(channels),
                'channels': channels,
                'primary_channel': primary_channel,
                'diversity_score': diversity_score,
                'total_messages': total_messages
            }
            
        except Exception as e:
            logger.error(f"Failed to get user channel comparison for {username}: {e}")
            return {
                'total_channels': 0,
                'channels': [],
                'primary_channel': None,
                'diversity_score': 0
            } 