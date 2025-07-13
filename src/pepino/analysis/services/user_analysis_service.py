"""
User Analysis Service

Handles all user-related analysis operations including top contributors,
detailed user analysis, and user activity patterns. Focuses specifically on user analysis domain.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from .base_service import BaseAnalysisService, OutputFormat

logger = logging.getLogger(__name__)


class UserAnalysisService(BaseAnalysisService):
    """
    Specialized service for user analysis operations.
    
    Handles:
    - Top contributors analysis
    - Detailed user analysis
    - User activity patterns
    - User engagement metrics
    """
    
    def _create_analyzers(self) -> Dict[str, Any]:
        """Create user-specific analyzers."""
        try:
            from pepino.analysis.helpers.user_analyzer import UserAnalyzer
            from pepino.analysis.helpers.message_analyzer import MessageAnalyzer
            from pepino.analysis.helpers.detailed_user_analyzer import DetailedUserAnalyzer
            
            return {
                'user': UserAnalyzer(self.data_facade),
                'message': MessageAnalyzer(self.data_facade),
                'detailed_user': DetailedUserAnalyzer(self.data_facade),
            }
        except ImportError as e:
            logger.warning(f"Could not load user analyzer classes: {e}")
            return {}
    
    def top_contributors(self, channel_name: Optional[str] = None, limit: int = 10, 
                        days_back: int = 30, end_date: Optional[datetime] = None, 
                        output_format: OutputFormat = "cli") -> str:
        """
        Generate top contributors analysis.
        
        Args:
            channel_name: Optional channel name (None for all channels)
            limit: Number of top users to show (default: 10)
            days_back: Number of days to look back (default: 30)
            end_date: End date for analysis (default: now)
            output_format: Output format ("cli" or "discord")
            
        Returns:
            Formatted analysis string
        """
        end_date = end_date or datetime.now()
        
        user_analyzer = self.analyzers.get('user')
        message_analyzer = self.analyzers.get('message')
        
        if not user_analyzer or not message_analyzer:
            return "❌ Analysis failed: Required analyzers not available"
        
        # Get contributors data using analyzer methods or data facade directly
        if channel_name:
            # Use data facade directly for channel-specific queries
            contributors = self.data_facade.user_repository.get_top_users(
                limit=limit, days_back=days_back, channel_name=channel_name
            )
            messages = self.data_facade.message_repository.get_channel_messages(
                channel_name, days_back=days_back, limit=1000
            )
        else:
            # Use analyzer method for general top users
            contributors = user_analyzer.get_top_users(limit=limit, days=days_back)
            messages = self.data_facade.message_repository.get_recent_messages(
                limit=10000, days_back=days_back
            )

        # Build enhanced contributors data
        enhanced_contributors = self._build_enhanced_contributors(contributors, messages, user_analyzer, days_back)

        period = {
            'start_date': (end_date - timedelta(days=days_back)).strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        }
        
        return self.render_template(
            "top_contributors",
            output_format=output_format,
            channel_name=channel_name,
            contributors=enhanced_contributors,
            period=period
        )
    
    def detailed_user_analysis(self, username: str, days_back: int = 30, 
                              output_format: OutputFormat = "cli") -> str:
        """
        Perform a detailed user analysis.
        
        Args:
            username: The username to analyze
            days_back: Number of days to look back
            output_format: "cli" or "discord"
            
        Returns:
            Rendered analysis string
        """
        detailed_user_analyzer = self.analyzers.get('detailed_user')
        
        if not detailed_user_analyzer:
            return "❌ Analysis failed: Detailed user analyzer not available"
        
        analysis = detailed_user_analyzer.analyze(username, days=days_back, include_patterns=True)
        if not analysis:
            return f"❌ No data found for user: {username}"

        # Get recent messages for content analysis
        messages = self.data_facade.message_repository.get_user_messages(username, days_back=days_back, limit=50)

        return self.render_template(
            "detailed_user_analysis",
            output_format=output_format,
            user_info=analysis.user_info,
            statistics=analysis.statistics,
            channel_activity=analysis.channel_activity,
            time_patterns=analysis.time_patterns,
            summary=analysis.summary,
            messages=messages
        )
    
    def _build_enhanced_contributors(self, contributors, messages, user_analyzer, days_back):
        """Build enhanced contributors data with display names and activity."""
        # Build a lookup: author_id -> most recent message (with display name)
        most_recent_message = {}
        for msg in messages:
            author_id = getattr(msg, 'author_id', None) or getattr(msg, 'author_name', None)
            if not author_id:
                continue
            if author_id not in most_recent_message or getattr(msg, 'timestamp', None) > getattr(most_recent_message[author_id], 'timestamp', None):
                most_recent_message[author_id] = msg
        
        enhanced_contributors = []
        for contributor in contributors:
            # Support both dicts and objects
            if isinstance(contributor, dict):
                author_id = contributor.get('author_id') or contributor.get('author_name', 'Unknown')
                display_name = contributor.get('display_name') or contributor.get('author_name', 'Unknown')
                message_count = contributor.get('message_count', 0)
                author_name = contributor.get('author_name', 'Unknown')
            else:
                author_id = getattr(contributor, 'author_id', None) or getattr(contributor, 'author_name', 'Unknown')
                display_name = getattr(contributor, 'display_name', None) or getattr(contributor, 'author_name', 'Unknown')
                message_count = getattr(contributor, 'message_count', 0)
                author_name = getattr(contributor, 'author_name', 'Unknown')
            if author_id in most_recent_message:
                display_name = getattr(most_recent_message[author_id], 'author_display_name', None) or getattr(most_recent_message[author_id], 'author_name', display_name)
            
            # Use author_name for channel activity lookup
            channel_activity = self.data_facade.user_repository.get_user_channel_activity(
                author_name, days_back, limit=10
            )
            
            # Get top messages with reply counts for this user
            top_messages = self._get_user_top_messages(display_name, days_back)
            
            enhanced_contributor = {
                'name': display_name,
                'message_count': message_count,
                'channel_activity': channel_activity,
                'top_messages': top_messages
            }
            enhanced_contributors.append(enhanced_contributor)
        
        return enhanced_contributors
    
    def _get_user_top_messages(self, username: str, days_back: int, limit: int = 3) -> List[Dict[str, Any]]:
        """Get top messages with reply counts for a specific user."""
        try:
            # Get user's messages from the last N days
            user_messages = self.data_facade.message_repository.get_user_messages(
                username, days_back=days_back, limit=1000
            )
            
            if not user_messages:
                return []
            
            # Get the date range from user's messages
            timestamps = [msg.timestamp for msg in user_messages if msg.timestamp]
            if not timestamps:
                return []
            
            start_date = min(timestamps)
            end_date = max(timestamps)
            
            # Get top commented messages for this user in the same time period
            # We'll filter by author name to get only this user's messages
            all_top_messages = self.data_facade.message_repository.get_top_commented_messages(
                None, start_date, end_date, top_n=50  # Get more to filter by user
            )
            
            # Filter to only include messages from this user
            user_top_messages = [
                msg for msg in all_top_messages 
                if msg.get('author', '').lower() == username.lower()
            ][:limit]
            
            return user_top_messages
            
        except Exception as e:
            logger.warning(f"Failed to get top messages for user {username}: {e}")
            return [] 