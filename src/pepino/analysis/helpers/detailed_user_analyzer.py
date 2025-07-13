"""
Detailed User Analyzer

Provides comprehensive per-user analysis for the new system, adapted from the legacy analyzer but with unique naming and structure.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from .data_facade import get_analysis_data_facade
from ..models import UserInfo
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class DetailedUserStatistics(BaseModel):
    total_messages: int
    avg_message_length: float
    active_days: int
    first_message: Optional[str]
    last_message: Optional[str]
    unique_channels: int

class DetailedChannelActivity(BaseModel):
    channel_name: str
    message_count: int
    first_message_date: Optional[str]
    last_message_date: Optional[str]

class DetailedTimePatterns(BaseModel):
    hourly_activity: List[Dict[str, Any]]
    daily_activity: List[Dict[str, Any]]
    peak_activity_hour: Optional[str] = None

class DetailedUserAnalysisResponse(BaseModel):
    user_info: UserInfo
    statistics: DetailedUserStatistics
    channel_activity: List[DetailedChannelActivity]
    time_patterns: Optional[DetailedTimePatterns]
    summary: str  # Changed from Dict to str for formatted summary
    messages: List[Dict[str, Any]] = []  # Add messages for content analysis

class DetailedUserAnalyzer:
    def __init__(self, data_facade=None):
        if data_facade is None:
            self.data_facade = get_analysis_data_facade()
            self._owns_facade = True
        else:
            self.data_facade = data_facade
            self._owns_facade = False
        logger.info("DetailedUserAnalyzer initialized with data facade pattern")

    def analyze(self, username: str, days: Optional[int] = None, include_patterns: bool = True) -> Optional[DetailedUserAnalysisResponse]:
        try:
            logger.info(f"Starting detailed user analysis for: {username}")
            statistics = self._get_user_statistics(username, days)
            if not statistics or statistics.total_messages == 0:
                logger.warning(f"No messages found for user: {username}")
                return None
            channel_activity = self._get_user_channel_activity(username, days)
            time_patterns = None
            if include_patterns:
                time_patterns = self._analyze_time_patterns(username, days)
            summary = self._create_user_summary(statistics, channel_activity, time_patterns)
            
            # Get messages for content analysis
            messages = self.data_facade.message_repository.get_user_messages(username, days_back=days, limit=50)
            messages_data = []
            for msg in messages:
                messages_data.append({
                    'id': msg.id,
                    'content': msg.content,
                    'author': msg.author_name,
                    'timestamp': msg.timestamp.isoformat() if msg.timestamp else None
                })
            
            analysis = DetailedUserAnalysisResponse(
                user_info=UserInfo(author_id=username, display_name=username),
                statistics=statistics,
                channel_activity=channel_activity,
                time_patterns=time_patterns,
                summary=summary,
                messages=messages_data
            )
            logger.info(f"Detailed user analysis completed for: {username}")
            return analysis
        except Exception as e:
            logger.error(f"Detailed user analysis failed for {username}: {e}")
            return None

    def _get_user_statistics(self, username: str, days: Optional[int] = None) -> Optional[DetailedUserStatistics]:
        try:
            messages = self.data_facade.message_repository.get_user_messages(username, days_back=days)
            if not messages:
                return None
            total_messages = len(messages)
            avg_message_length = sum(len(msg.content or '') for msg in messages) / total_messages if total_messages > 0 else 0
            unique_channels = len(set(msg.channel_name for msg in messages if msg.channel_name))
            message_dates = set(str(msg.timestamp)[:10] for msg in messages if msg.timestamp)
            active_days = len(message_dates)
            timestamps = [msg.timestamp for msg in messages if msg.timestamp]
            first_message = min(timestamps).isoformat() if timestamps else None
            last_message = max(timestamps).isoformat() if timestamps else None
            return DetailedUserStatistics(
                total_messages=total_messages,
                avg_message_length=avg_message_length,
                active_days=active_days,
                first_message=first_message,
                last_message=last_message,
                unique_channels=unique_channels
            )
        except Exception as e:
            logger.error(f"Failed to get user statistics for {username}: {e}")
            return None

    def _get_user_channel_activity(self, username: str, days: Optional[int] = None) -> List[DetailedChannelActivity]:
        try:
            messages = self.data_facade.message_repository.get_user_messages(username, days_back=days)
            if not messages:
                return []
            channel_counts = {}
            channel_dates = {}
            for msg in messages:
                channel_name = msg.channel_name
                if channel_name:
                    if channel_name not in channel_counts:
                        channel_counts[channel_name] = 0
                        channel_dates[channel_name] = []
                    channel_counts[channel_name] += 1
                    if msg.timestamp:
                        channel_dates[channel_name].append(msg.timestamp)
            channel_activity = []
            for channel_name, count in channel_counts.items():
                dates = channel_dates[channel_name]
                first_message = min(dates).isoformat() if dates else None
                last_message = max(dates).isoformat() if dates else None
                channel_activity.append(DetailedChannelActivity(
                    channel_name=channel_name,
                    message_count=count,
                    first_message_date=first_message,
                    last_message_date=last_message
                ))
            channel_activity.sort(key=lambda x: x.message_count, reverse=True)
            return channel_activity
        except Exception as e:
            logger.error(f"Failed to get channel activity for {username}: {e}")
            return []

    def _analyze_time_patterns(self, username: str, days: Optional[int] = None) -> Optional[DetailedTimePatterns]:
        try:
            messages = self.data_facade.message_repository.get_user_messages(username, days_back=days)
            if not messages:
                return None
            hourly_activity = {}
            daily_activity = {}
            for msg in messages:
                timestamp = msg.timestamp
                if timestamp:
                    try:
                        dt = timestamp if isinstance(timestamp, datetime) else datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))
                        hour = dt.hour
                        day = dt.strftime('%A')
                        hourly_activity[hour] = hourly_activity.get(hour, 0) + 1
                        daily_activity[day] = daily_activity.get(day, 0) + 1
                    except:
                        continue
            hourly_list = [{'hour': f"{hour:02d}:00", 'message_count': count} for hour, count in sorted(hourly_activity.items(), key=lambda x: x[1], reverse=True)]
            daily_list = [{'day': day, 'message_count': count} for day, count in sorted(daily_activity.items(), key=lambda x: x[1], reverse=True)]
            peak_hour = None
            if hourly_list:
                peak_hour = hourly_list[0]['hour']
            return DetailedTimePatterns(hourly_activity=hourly_list, daily_activity=daily_list, peak_activity_hour=peak_hour)
        except Exception as e:
            logger.error(f"Failed to analyze time patterns for {username}: {e}")
            return None

    def _create_user_summary(self, statistics: DetailedUserStatistics, channel_activity: List[DetailedChannelActivity], time_patterns: Optional[DetailedTimePatterns]) -> str:
        try:
            summary = []
            summary.append(f"Total Messages: {statistics.total_messages}")
            summary.append(f"Average Message Length: {statistics.avg_message_length:.2f} characters")
            summary.append(f"Active Days: {statistics.active_days}")
            summary.append(f"First Message: {statistics.first_message}")
            summary.append(f"Last Message: {statistics.last_message}")
            summary.append(f"Unique Channels: {statistics.unique_channels}")

            if time_patterns:
                summary.append(f"Peak Activity Hour: {time_patterns.peak_activity_hour}")
                summary.append(f"Most Active Hour: {time_patterns.hourly_activity[0]['hour'] if time_patterns.hourly_activity else 'N/A'}")
                summary.append(f"Most Active Day: {time_patterns.daily_activity[0]['day'] if time_patterns.daily_activity else 'N/A'}")

            if statistics.total_messages > 1000:
                summary.append("Activity Level: Very Active")
            elif statistics.total_messages > 500:
                summary.append("Activity Level: Active")
            elif statistics.total_messages > 100:
                summary.append("Activity Level: Moderate")
            else:
                summary.append("Activity Level: Low")

            if statistics.total_messages > 0 and statistics.unique_channels > 0:
                summary.append(f"Engagement Score: {(statistics.total_messages / statistics.unique_channels) * 10:.2f}")
            else:
                summary.append("Engagement Score: 0")

            if channel_activity:
                summary.append("Primary Channels:")
                for i, ch in enumerate(channel_activity[:3]):
                    summary.append(f"  {i+1}. {ch.channel_name} ({ch.message_count} messages)")
            return "\n".join(summary)
        except Exception as e:
            logger.error(f"Failed to create user summary: {e}")
            return "Summary could not be generated."

    def __del__(self):
        if hasattr(self, '_owns_facade') and self._owns_facade and hasattr(self, 'data_facade'):
            try:
                self.data_facade.close()
            except:
                pass 