"""
Weekly User Analyzer for channel analysis.

Provides focused analysis of user participation, engagement, and activity patterns.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter, defaultdict
import math

from pepino.analysis.helpers.data_facade import AnalysisDataFacade

logger = logging.getLogger(__name__)

def to_utc(dt):
    if isinstance(dt, str):
        if dt.endswith("Z"):
            dt = dt.replace("Z", "+00:00")
        dt = datetime.fromisoformat(dt)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

class WeeklyUserAnalyzer:
    """Analyzer for user-specific patterns and metrics in weekly analysis."""
    
    def __init__(self, data_facade: AnalysisDataFacade):
        self.data_facade = data_facade
    
    def analyze_weekly_users(self, channel_name: str, days_back: int = 7, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Analyze user participation for the specified week period.
        
        Args:
            channel_name: Channel to analyze
            days_back: Number of days to look back (default 7 for weekly)
            end_date: End date for analysis (default: current date)
            
        Returns:
            Dictionary with user analysis results
        """
        try:
            # Use provided end date or default to current date
            if end_date is None:
                end_date = to_utc(datetime.now())
            else:
                end_date = to_utc(end_date)
            
            start_date = end_date - timedelta(days=days_back)
            
            # Get messages for the current period using date range
            current_messages = self.data_facade.message_repository.get_messages_by_date_range(
                channel_name, start_date, end_date, limit=10000
            )
            
            if not current_messages:
                return self._get_empty_user_analysis()
            
            # For trend analysis, we need to compare with the previous calendar week
            # Calculate the previous week relative to the end date
            end_date_naive = end_date.replace(tzinfo=None) if end_date.tzinfo else end_date
            current_weekday = end_date_naive.weekday()  # Monday=0, Sunday=6
            
            # Calculate the start of the current week (Monday)
            days_since_monday = current_weekday
            current_week_start = end_date_naive - timedelta(days=days_since_monday)
            current_week_start = current_week_start.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Calculate the previous week (7 days before current week start)
            prev_week_start = current_week_start - timedelta(days=7)
            prev_week_end = current_week_start
            
            # Make dates timezone-aware for comparison
            prev_week_start = to_utc(prev_week_start)
            prev_week_end = to_utc(prev_week_end)
            
            # Get messages for the previous week period
            # We need to get messages from a wider range to ensure we capture the previous week
            max_days_back = max(days_back + 7, 14)  # At least 14 days to ensure we get previous week
            prev_messages = self.data_facade.message_repository.get_messages_by_date_range(
                channel_name, prev_week_start, prev_week_end, limit=10000
            )
            
            # Analyze user patterns
            user_stats = self._analyze_user_statistics(current_messages, prev_messages)
            top_contributors = self._get_top_contributors(current_messages)
            participation_distribution = self._analyze_participation_distribution(current_messages)
            lost_interest_users = self._get_lost_interest_users(channel_name, days_back, end_date)
            
            return {
                'user_stats': user_stats,
                'top_contributors': top_contributors,
                'participation_distribution': participation_distribution,
                'lost_interest_users': lost_interest_users,
                'total_participants': len(set(self._get_user_id_or_name(m) for m in current_messages if not m.get('author_is_bot', False) and self._get_user_id_or_name(m) is not None))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing weekly users for {channel_name}: {e}")
            return self._get_empty_user_analysis()
    
    def analyze_weekly_users_all_channels(self, days_back: int = 7, end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Analyze user participation across all channels for the specified week period.
        
        Args:
            days_back: Number of days to look back (default 7 for weekly)
            end_date: End date for analysis (default: current date)
            
        Returns:
            Dictionary with user analysis results
        """
        try:
            # Use provided end date or default to current date
            if end_date is None:
                end_date = to_utc(datetime.now())
            else:
                end_date = to_utc(end_date)
            
            start_date = end_date - timedelta(days=days_back)
            
            # Get messages for the current period using date range across all channels
            current_messages = self.data_facade.message_repository.get_messages_by_date_range(
                None, start_date, end_date, limit=10000
            )
            
            if not current_messages:
                return self._get_empty_user_analysis()
            
            # For trend analysis, we need to compare with the previous calendar week
            # Calculate the previous week relative to the end date
            end_date_naive = end_date.replace(tzinfo=None) if end_date.tzinfo else end_date
            current_weekday = end_date_naive.weekday()  # Monday=0, Sunday=6
            
            # Calculate the start of the current week (Monday)
            days_since_monday = current_weekday
            current_week_start = end_date_naive - timedelta(days=days_since_monday)
            current_week_start = current_week_start.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Calculate the previous week (7 days before current week start)
            prev_week_start = current_week_start - timedelta(days=7)
            prev_week_end = current_week_start
            
            # Make dates timezone-aware for comparison
            prev_week_start = to_utc(prev_week_start)
            prev_week_end = to_utc(prev_week_end)
            
            # Get messages for the previous week period across all channels
            max_days_back = max(days_back + 7, 14)  # At least 14 days to ensure we get previous week
            prev_messages = self.data_facade.message_repository.get_messages_by_date_range(
                None, prev_week_start, prev_week_end, limit=10000
            )
            
            # Analyze user patterns
            user_stats = self._analyze_user_statistics(current_messages, prev_messages)
            top_contributors = self._get_top_contributors(current_messages)
            participation_distribution = self._analyze_participation_distribution(current_messages)
            lost_interest_users = self._get_lost_interest_users_all_channels(days_back, end_date)
            
            return {
                'user_stats': user_stats,
                'top_contributors': top_contributors,
                'participation_distribution': participation_distribution,
                'lost_interest_users': lost_interest_users,
                'total_participants': len(set(self._get_user_id_or_name(m) for m in current_messages if not m.get('author_is_bot', False) and self._get_user_id_or_name(m) is not None))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing weekly users for all channels: {e}")
            return self._get_empty_user_analysis()
    
    def _get_user_id_or_name(self, m):
        return m.get('author_id') or m.get('author_name')
    
    def _analyze_user_statistics(self, current_messages: List[Dict], prev_messages: List[Dict]) -> Dict[str, Any]:
        """Analyze user statistics and trends."""
        if not current_messages:
            return {}
        
        # Current period stats
        current_human_users = set(self._get_user_id_or_name(m) for m in current_messages if not m.get('author_is_bot', False) and self._get_user_id_or_name(m) is not None)
        current_bot_users = set(self._get_user_id_or_name(m) for m in current_messages if m.get('author_is_bot', False) and self._get_user_id_or_name(m) is not None)
        
        # Previous period stats
        prev_human_users = set(self._get_user_id_or_name(m) for m in prev_messages if not m.get('author_is_bot', False) and self._get_user_id_or_name(m) is not None)
        prev_bot_users = set(self._get_user_id_or_name(m) for m in prev_messages if m.get('author_is_bot', False) and self._get_user_id_or_name(m) is not None)
        
        # Calculate trends
        human_user_change = len(current_human_users) - len(prev_human_users)
        
        # Handle percentage change calculation
        if len(prev_human_users) == 0:
            if len(current_human_users) > 0:
                human_user_change_pct = float('inf')  # New activity
                trend_direction = "new_activity"
            else:
                human_user_change_pct = 0
                trend_direction = "unchanged"
        else:
            human_user_change_pct = (human_user_change / len(prev_human_users) * 100)
            # Determine trend direction - any change should be reflected
            if human_user_change_pct > 0:
                trend_direction = "increasing"
            elif human_user_change_pct < 0:
                trend_direction = "decreasing"
            else:
                trend_direction = "unchanged"
        
        return {
            'current_human_users': len(current_human_users),
            'current_bot_users': len(current_bot_users),
            'prev_human_users': len(prev_human_users),
            'prev_bot_users': len(prev_bot_users),
            'human_user_change': human_user_change,
            'human_user_change_pct': human_user_change_pct,
            'trend_direction': trend_direction
        }
    
    def _get_top_contributors(self, messages: List[Dict], top_n: int = 5) -> List[Dict]:
        """Get top contributors by message count."""
        if not messages:
            return []
        
        # Count messages per user (excluding bots)
        user_message_counts = Counter()
        user_names = {}
        
        for message in messages:
            if not message.get('author_is_bot', False) and self._get_user_id_or_name(message) is not None:
                user_id = self._get_user_id_or_name(message)
                user_message_counts[user_id] += 1
                # Use display name if available, otherwise fall back to author name
                display_name = message.get('author_display_name') or message.get('author_name', 'Unknown')
                user_names[user_id] = display_name
        
        # Get top contributors
        top_contributors = user_message_counts.most_common(top_n)
        
        return [
            {
                'author_id': author_id,
                'name': user_names.get(author_id, 'Unknown'),
                'message_count': count,
                'percentage': (count / len([m for m in messages if not m.get('author_is_bot', False)]) * 100) if messages else 0
            }
            for author_id, count in top_contributors
        ]
    
    def _analyze_participation_distribution(self, messages: List[Dict]) -> Dict[str, Any]:
        """Analyze how participation is distributed among users."""
        if not messages:
            return {'distribution': 'unknown', 'top_contributors_percentage': 0}
        
        human_messages = [m for m in messages if not m.get('author_is_bot', False)]
        if not human_messages:
            return {'distribution': 'unknown', 'top_contributors_percentage': 0}
        
        # Count messages per user
        user_message_counts = Counter()
        for message in human_messages:
            user_id = self._get_user_id_or_name(message)
            if user_id is not None:
                user_message_counts[user_id] += 1
        
        total_messages = len(human_messages)
        total_contributors = len(user_message_counts)
        
        # Calculate top 5% contributors (rounded up, at least 1)
        top_percent = 0.05
        top_n = max(1, math.ceil(total_contributors * top_percent))
        top_contributors = user_message_counts.most_common(top_n)
        top_contributors_message_count = sum(count for _, count in top_contributors)
        top_contributors_percentage = (top_contributors_message_count / total_messages * 100) if total_messages > 0 else 0
        
        # Determine distribution type based on top 5% share
        if top_contributors_percentage > 25:
            distribution = "Poorly"
        elif top_contributors_percentage > 10:
            distribution = "Moderately"
        else:
            distribution = "Well"
        
        return {
            'distribution': distribution,
            'top_contributors_percentage': top_contributors_percentage,
            'top_contributors_count': top_n,
            'total_contributors': total_contributors
        }
    
    def _get_lost_interest_users(self, channel_name: str, days_back: int = 7, end_date: Optional[datetime] = None) -> List[Dict]:
        """Get users who were inactive in the 30-day period ending at end_date."""
        try:
            # Use provided end date or default to current date
            if end_date is None:
                end_date = to_utc(datetime.now())
            else:
                end_date = to_utc(end_date)
            
            # Calculate the 30-day period ending at end_date
            thirty_days_before_end = end_date - timedelta(days=30)
            
            # Get all messages from the 30-day period ending at end_date
            historical_messages = self.data_facade.message_repository.get_messages_by_date_range(
                channel_name, thirty_days_before_end, end_date, limit=10000
            )
            
            # Get all users who were active in this 30-day period
            active_users = set()
            for message in historical_messages:
                user_id = self._get_user_id_or_name(message)
                if (
                    not message.get('author_is_bot', False)
                    and user_id is not None
                ):
                    active_users.add(user_id)
            
            # For now, we'll return users who were active but haven't been active recently
            # In a real implementation, you'd need to track all channel members
            # For this demo, we'll return users who were active in the period but with low activity
            lost_interest_users = []
            for user_id in active_users:
                # Count messages from this user in the 30-day period
                user_messages = [m for m in historical_messages if self._get_user_id_or_name(m) == user_id]
                if user_messages:
                    latest_message = max(user_messages, key=lambda x: to_utc(x['timestamp']))
                    # Consider users with very low activity as "lost interest"
                    if len(user_messages) <= 1:  # Only 1 or fewer messages in 30 days
                        lost_interest_users.append({
                            'author_id': user_id,
                            'name': latest_message.get('author_name', user_id),
                            'last_message_date': latest_message['timestamp'],
                            'days_inactive': 30
                        })
            
            return lost_interest_users[:10]  # Limit to top 10
            
        except Exception as e:
            logger.error(f"Error getting lost interest users: {e}")
            return []
    
    def _get_lost_interest_users_all_channels(self, days_back: int = 7, end_date: Optional[datetime] = None) -> List[Dict]:
        """Get users who were inactive across all channels in the 30-day period ending at end_date."""
        try:
            # Use provided end date or default to current date
            if end_date is None:
                end_date = to_utc(datetime.now())
            else:
                end_date = to_utc(end_date)
            
            # Calculate the 30-day period ending at end_date
            thirty_days_before_end = end_date - timedelta(days=30)
            
            # Get all messages from the 30-day period ending at end_date across all channels
            historical_messages = self.data_facade.message_repository.get_messages_by_date_range(
                None, thirty_days_before_end, end_date, limit=10000
            )
            
            # Get all users who were active in this 30-day period
            active_users = set()
            for message in historical_messages:
                user_id = self._get_user_id_or_name(message)
                if (
                    not message.get('author_is_bot', False)
                    and user_id is not None
                ):
                    active_users.add(user_id)
            
            # For now, we'll return users who were active but haven't been active recently
            # In a real implementation, you'd need to track all channel members
            # For this demo, we'll return users who were active in the period but with low activity
            lost_interest_users = []
            for user_id in active_users:
                # Count messages from this user in the 30-day period
                user_messages = [m for m in historical_messages if self._get_user_id_or_name(m) == user_id]
                if user_messages:
                    latest_message = max(user_messages, key=lambda x: to_utc(x['timestamp']))
                    # Consider users with very low activity as "lost interest"
                    if len(user_messages) <= 1:  # Only 1 or fewer messages in 30 days
                        lost_interest_users.append({
                            'author_id': user_id,
                            'name': latest_message.get('author_name', user_id),
                            'last_message_date': latest_message['timestamp'],
                            'days_inactive': 30
                        })
            
            return lost_interest_users[:10]  # Limit to top 10
            
        except Exception as e:
            logger.error(f"Error getting lost interest users across all channels: {e}")
            return []
    
    def _get_empty_user_analysis(self) -> Dict[str, Any]:
        """Return empty user analysis structure."""
        return {
            'user_stats': {
                'current_human_users': 0,
                'current_bot_users': 0,
                'prev_human_users': 0,
                'prev_bot_users': 0,
                'human_user_change': 0,
                'human_user_change_pct': 0,
                'trend_direction': 'unchanged'
            },
            'top_contributors': [],
            'participation_distribution': {
                'distribution': 'unknown',
                'top_contributors_percentage': 0,
                'top_contributors_count': 0,
                'total_contributors': 0
            },
            'lost_interest_users': [],
            'total_participants': 0
        } 