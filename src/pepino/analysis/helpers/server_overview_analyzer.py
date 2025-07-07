"""
Server Overview Analyzer

Provides comprehensive server overview analysis including statistics,
health metrics, engagement analysis, and activity patterns.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from ..models import ServerOverviewData

class ServerOverviewAnalyzer:
    """Analyzer for comprehensive server overview statistics."""
    
    def __init__(self, message_repo, user_repo, channel_repo):
        self.message_repo = message_repo
        self.user_repo = user_repo
        self.channel_repo = channel_repo
    
    def analyze(self, days: int = 30) -> ServerOverviewData:
        """Generate comprehensive server overview statistics."""
        if days is None or not isinstance(days, int):
            days = 30
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        messages = self.message_repo.get_messages_by_date_range(None, start_date, end_date)
        users = self.user_repo.get_user_list()
        channels = self.channel_repo.get_all_channels_as_dicts()

        # Filter out channels missing 'id' or 'name'
        channels = [ch for ch in channels if ch.get('id') and ch.get('name')]

        total_messages = len(messages)
        total_users = len(users)
        total_channels = len(channels)
        
        # Use author_id instead of user_id
        active_user_ids = set(msg['author_id'] for msg in messages if msg.get('author_id'))
        active_users = len(active_user_ids)
        
        messages_per_day = total_messages / days if days > 0 else 0
        messages_per_user = total_messages / active_users if active_users > 0 else 0
        
        channel_message_counts = {}
        for msg in messages:
            channel_id = msg.get('channel_id')
            if channel_id:
                channel_message_counts[channel_id] = channel_message_counts.get(channel_id, 0) + 1
        
        channel_id_to_name = {channel['id']: channel['name'] for channel in channels}
        top_channels = []
        for channel_id, count in sorted(channel_message_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            channel_name = channel_id_to_name.get(channel_id, f"Unknown-{channel_id}")
            top_channels.append({
                'name': channel_name,
                'message_count': count
            })
        
        # Use author_id and author_name for contributors
        user_message_counts = {}
        for msg in messages:
            author_id = msg.get('author_id')
            if author_id:
                user_message_counts[author_id] = user_message_counts.get(author_id, 0) + 1
        
        # Map author_id to author_name from user list
        user_id_to_name = {user['author_id']: user.get('author_display_name') or user.get('author_name', f"Unknown-{user['author_id']}") for user in users if 'author_id' in user}
        # Fallback mapping from messages
        msg_id_to_name = {}
        for msg in messages:
            author_id = msg.get('author_id')
            if author_id:
                name = msg.get('author_display_name') or msg.get('author_name') or f"Unknown-{author_id}"
                msg_id_to_name[author_id] = name
        top_contributors = []
        for author_id, count in sorted(user_message_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            username = user_id_to_name.get(author_id) or msg_id_to_name.get(author_id) or f"Unknown-{author_id}"
            top_contributors.append({
                'username': username,
                'message_count': count
            })
        
        daily_message_counts = {}
        for msg in messages:
            msg_date = datetime.fromisoformat(msg['timestamp'].replace('Z', '+00:00')).date()
            daily_message_counts[msg_date] = daily_message_counts.get(msg_date, 0) + 1
        
        activity_trends = []
        for i in range(7):
            date = (end_date - timedelta(days=i)).date()
            count = daily_message_counts.get(date, 0)
            activity_trends.append({
                'date': date.strftime('%Y-%m-%d'),
                'message_count': count
            })
        activity_trends.reverse()
        
        return ServerOverviewData(
            total_messages=total_messages,
            total_users=total_users,
            total_channels=total_channels,
            active_users=active_users,
            messages_per_day=messages_per_day,
            messages_per_user=messages_per_user,
            top_channels=top_channels,
            top_contributors=top_contributors,
            activity_trends=activity_trends,
            analysis_period_days=days
        ) 