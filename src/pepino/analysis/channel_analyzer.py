"""
Channel Analyzer 

Synchronous channel analysis using the data facade pattern for repository management.
Provides comprehensive channel activity analysis with proper separation of concerns.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .data_facade import AnalysisDataFacade, get_analysis_data_facade
from .models import ChannelAnalysisResponse
from .models import LocalChannelStatistics, UserActivity
from .utils import safe_timestamp_to_iso

logger = logging.getLogger(__name__)


class ChannelAnalyzer:
    """
    Analyzes channel activity patterns, user engagement, and messaging statistics
    using the data facade pattern for centralized repository management.
    
    Features:
    - Message count analysis through data facade layer
    - User activity patterns
    - Time-based patterns
    - Content analysis
    - Engagement metrics
    
    All database operations are abstracted through the data facade for proper
    separation of concerns and dependency injection support.
    """
    
    def __init__(self, data_facade: Optional[AnalysisDataFacade] = None):
        """Initialize channel analyzer with data facade."""
        if data_facade is None:
            self.data_facade = get_analysis_data_facade()
            self._owns_facade = True
        else:
            self.data_facade = data_facade
            self._owns_facade = False
            
        logger.info("ChannelAnalyzer initialized with data facade pattern")
    
    def analyze(
        self, 
        channel_name: str,
        days: Optional[int] = None,
        include_patterns: bool = True
    ) -> Optional[ChannelAnalysisResponse]:
        """
        Analyze a channel comprehensively using repository layer.
        
        Args:
            channel_name: Name of the channel to analyze
            days: Number of days to look back (None = all time)
            include_patterns: Whether to include time patterns analysis
            
        Returns:
            ChannelAnalysisResponse object with comprehensive results
        """
        
        try:
            logger.info(f"Starting channel analysis for: {channel_name}")
            
            # Get basic statistics through repository
            statistics = self._get_channel_statistics_via_repository(channel_name, days)
            if not statistics or statistics.total_messages == 0:
                logger.warning(f"No messages found for channel: {channel_name}")
                return None
            
            # Get top users in channel through repository
            top_users = self._get_top_users_in_channel_via_repository(channel_name, days)
            
            # Get time patterns if requested
            time_patterns = {}
            if include_patterns:
                time_patterns = self._analyze_time_patterns_via_repository(channel_name, days)
            
            # Build result
            from .models import ChannelInfo
            
            analysis = ChannelAnalysisResponse(
                channel_info=ChannelInfo(channel_name=channel_name),
                statistics=self._convert_to_channel_statistics_model(statistics),
                top_users=self._convert_top_users_to_model(top_users),
                peak_activity=self._convert_time_patterns_to_peak_activity(time_patterns) if time_patterns else None,
                # Other fields will use defaults
            )
            
            logger.info(f"Channel analysis completed for: {channel_name}")
            return analysis
            
        except Exception as e:
            logger.error(f"Channel analysis failed for {channel_name}: {e}")
            return None
    
    # Removed local method - now using shared utility
    
    def _convert_to_channel_statistics_model(self, stats: LocalChannelStatistics):
        """Convert our LocalChannelStatistics to the model expected by ChannelAnalysisResponse."""
        from .models import ChannelStatistics as ModelChannelStatistics
        
        return ModelChannelStatistics(
            total_messages=stats.total_messages,
            unique_users=stats.unique_users,
            avg_message_length=stats.avg_message_length,
            first_message=safe_timestamp_to_iso(stats.first_message_date),
            last_message=safe_timestamp_to_iso(stats.last_message_date),
            active_days=0,  # We don't calculate this yet
            bot_messages=stats.bot_messages,
            human_messages=stats.human_messages,
            unique_human_users=stats.unique_human_users
        )
    
    def _convert_top_users_to_model(self, user_activities: List[UserActivity]):
        """Convert UserActivity list to TopUserInChannel model list."""
        from .models import TopUserInChannel
        
        converted_users = []
        for activity in user_activities:
            # Use display name if available, otherwise username
            display_name = activity.display_name or activity.username
            
            converted_user = TopUserInChannel(
                author_id="",  # We don't have author_id available
                author_name=activity.username,  # UserActivity uses 'username' field
                display_name=display_name,
                message_count=activity.message_count,
                avg_message_length=activity.avg_message_length or 0.0
            )
            converted_users.append(converted_user)
        
        return converted_users
    
    def _convert_time_patterns_to_peak_activity(self, time_patterns: Dict[str, any]):
        """Convert time patterns to PeakActivity model."""
        from .models import PeakActivity, PeakActivityHour, PeakActivityDay
        
        peak_hours = []
        hourly_activity = time_patterns.get('hourly_activity', {})
        # Get top 3 peak hours
        sorted_hours = sorted(hourly_activity.items(), key=lambda x: x[1], reverse=True)[:3]
        for hour, count in sorted_hours:
            peak_hours.append(PeakActivityHour(
                hour=f"{hour:02d}:00-{hour:02d}:59",
                messages=count
            ))
        
        peak_days = []
        daily_activity = time_patterns.get('daily_activity', {})
        # Get top 3 peak days
        sorted_days = sorted(daily_activity.items(), key=lambda x: x[1], reverse=True)[:3]
        for date, count in sorted_days:
            peak_days.append(PeakActivityDay(
                day=date,
                messages=count
            ))
        
        return PeakActivity(
            peak_hours=peak_hours,
            peak_days=peak_days
        )
    
    def _get_channel_statistics_via_repository(self, channel_name: str, days: Optional[int]) -> Optional[LocalChannelStatistics]:
        """Get basic channel statistics using data facade pattern."""
        
        try:
            # Validate input
            if not channel_name or not channel_name.strip():
                logger.warning("Empty or invalid channel name provided")
                return None
            
            # Use data facade for repository access
            stats_data = self.data_facade.channel_repository.get_channel_message_statistics(channel_name, days)
            
            if not stats_data or stats_data.get('total_messages', 0) == 0:
                logger.info(f"No messages found for channel: {channel_name}")
                return None
            
            # Calculate messages per day
            total_messages = stats_data['total_messages']
            if total_messages > 0 and stats_data['first_message'] and stats_data['last_message']:
                try:
                    first_date = datetime.fromisoformat(stats_data['first_message'].replace('Z', '+00:00'))
                    last_date = datetime.fromisoformat(stats_data['last_message'].replace('Z', '+00:00'))
                    time_span = (last_date - first_date).days + 1
                    messages_per_day = total_messages / max(time_span, 1)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid timestamp format for channel {channel_name}: {e}")
                    messages_per_day = 0
            else:
                messages_per_day = 0
            
            return LocalChannelStatistics(
                total_messages=stats_data['total_messages'],
                unique_users=stats_data['unique_users'],
                messages_per_day=round(messages_per_day, 2),
                avg_message_length=stats_data.get('avg_message_length', 0.0),
                first_message_date=stats_data['first_message'],
                last_message_date=stats_data['last_message'],
                analysis_period_days=days,
                bot_messages=stats_data.get('bot_messages', 0),
                human_messages=stats_data.get('human_messages', 0),
                unique_human_users=stats_data.get('unique_human_users', 0)
            )
            
        except Exception as e:
            logger.error(f"Failed to get channel statistics via repository for {channel_name}: {e}")
            return None
    
    def _get_top_users_in_channel_via_repository(
        self, 
        channel_name: str, 
        days: Optional[int],
        limit: int = 10
    ) -> List[UserActivity]:
        """Get top users by message count in this channel using data facade."""
        
        try:
            # Validate input
            if not channel_name or not channel_name.strip():
                logger.warning("Empty or invalid channel name provided for user activity")
                return []
            
            if limit <= 0:
                logger.warning(f"Invalid limit provided: {limit}, using default of 10")
                limit = 10
            
            # Use data facade for repository access
            user_data = self.data_facade.channel_repository.get_channel_user_activity(channel_name, days, limit)
            
            user_activities = []
            for data in user_data:
                try:
                    activity = UserActivity(
                        username=data['author_name'],  # Fixed: UserActivity uses 'username' field
                        display_name=data.get('author_display_name'),  # New field
                        message_count=data['message_count'],
                        avg_message_length=data.get('avg_message_length', 0.0),  # New field
                        first_message_date=data['first_message'],  # Fixed: model uses '_date' suffix
                        last_message_date=data['last_message']     # Fixed: model uses '_date' suffix
                    )
                    user_activities.append(activity)
                except (KeyError, TypeError) as e:
                    logger.warning(f"Skipping invalid user data for channel {channel_name}: {e}")
                    continue
            
            return user_activities
            
        except Exception as e:
            logger.error(f"Failed to get top users in channel via repository for {channel_name}: {e}")
            return []
    
    def _analyze_time_patterns_via_repository(self, channel_name: str, days: Optional[int]) -> Dict[str, any]:
        """Analyze channel time patterns using data facade."""
        
        try:
            # Get hourly patterns through data facade
            hourly_data = self.data_facade.channel_repository.get_channel_hourly_patterns(channel_name, days)
            
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
            
            # Get daily patterns for trends
            daily_data = self.data_facade.channel_repository.get_channel_daily_patterns(channel_name, days)
            daily_activity = {}
            for entry in daily_data:
                daily_activity[entry['date']] = entry['message_count']
            
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
                'daily_activity': daily_activity,
                'peak_hour': peak_hour,
                'activity_period': activity_period,
                'total_active_hours': len(hourly_activity),
                'total_active_days': len(daily_activity)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze time patterns via repository: {e}")
            return {}
    
    def _create_channel_summary(
        self,
        statistics: LocalChannelStatistics,
        top_users: List[UserActivity],
        time_patterns: Dict[str, any]
    ) -> Dict[str, any]:
        """Create summary of channel analysis results."""
        
        # User engagement level
        if statistics.unique_users > 50:
            engagement_level = "very high"
        elif statistics.unique_users > 20:
            engagement_level = "high"
        elif statistics.unique_users > 10:
            engagement_level = "moderate"
        elif statistics.unique_users > 5:
            engagement_level = "low"
        else:
            engagement_level = "very low"
        
        # Activity level
        if statistics.messages_per_day > 100:
            activity_level = "very active"
        elif statistics.messages_per_day > 50:
            activity_level = "active"
        elif statistics.messages_per_day > 10:
            activity_level = "moderate"
        elif statistics.messages_per_day > 1:
            activity_level = "low"
        else:
            activity_level = "very low"
        
        # Most active user
        most_active_user = None
        if top_users:
            most_active_user = top_users[0].author_name
        
        # Time preference
        time_preference = time_patterns.get('activity_period', 'unknown')
        
        return {
            'activity_level': activity_level,
            'engagement_level': engagement_level,
            'most_active_user': most_active_user,
            'time_preference': time_preference,
            'total_users': statistics.unique_users,
            'avg_daily_messages': statistics.messages_per_day
        }

    def get_available_channels(self) -> List[str]:
        """Get list of available channels using data facade."""
        try:
            return self.data_facade.channel_repository.get_available_channels()
        except Exception as e:
            logger.error(f"Failed to get available channels: {e}")
            return []

    def get_top_channels(self, limit: int = 10, days: Optional[int] = None) -> List[Dict[str, any]]:
        """Get top channels by message count using data facade."""
        try:
            return self.data_facade.channel_repository.get_top_channels_by_message_count(limit, days)
        except Exception as e:
            logger.error(f"Failed to get top channels: {e}")
            return []

    def get_channel_health(self, channel_name: str) -> Dict[str, any]:
        """Get channel engagement health metrics using data facade."""
        
        try:
            # Get basic stats
            recent_stats = self.data_facade.channel_repository.get_channel_message_statistics(channel_name, days=7)
            overall_stats = self.data_facade.channel_repository.get_channel_message_statistics(channel_name, days=None)
            
            if not overall_stats or overall_stats.get('total_messages', 0) == 0:
                return {
                    'status': 'inactive',
                    'health_score': 0,
                    'recommendations': ['Channel has no message history']
                }
            
            # Calculate health metrics
            recent_messages = recent_stats.get('total_messages', 0)
            total_messages = overall_stats.get('total_messages', 0)
            unique_users = overall_stats.get('unique_users', 0)
            
            # Health score calculation (out of 100)
            health_score = 0
            
            # Recent activity (40% weight)
            if recent_messages > 50:
                health_score += 40
            elif recent_messages > 20:
                health_score += 30
            elif recent_messages > 5:
                health_score += 20
            elif recent_messages > 0:
                health_score += 10
            
            # User engagement (30% weight)
            if unique_users > 20:
                health_score += 30
            elif unique_users > 10:
                health_score += 20
            elif unique_users > 5:
                health_score += 15
            elif unique_users > 0:
                health_score += 10
            
            # Overall activity (30% weight)
            if total_messages > 5000:
                health_score += 30
            elif total_messages > 1000:
                health_score += 25
            elif total_messages > 100:
                health_score += 20
            elif total_messages > 10:
                health_score += 15
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
                recommendations.append('Channel appears inactive - consider content to stimulate discussion')
            if unique_users <= 5:
                recommendations.append('Low user engagement - consider encouraging participation')
            if total_messages < 100:
                recommendations.append('New channel - consider establishing regular content')
            
            return {
                'status': status,
                'health_score': health_score,
                'recent_activity': recent_messages,
                'total_activity': total_messages,
                'user_engagement': unique_users,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Failed to get channel health for {channel_name}: {e}")
            return {
                'status': 'error',
                'health_score': 0,
                'recommendations': [f'Error analyzing channel: {e}']
            } 