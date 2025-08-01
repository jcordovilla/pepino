"""
User repository for data access operations.
Provides centralized data access for user-related operations following the persistence facade pattern.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ...config import settings
from ..database.manager import DatabaseManager
from ..models.user import User
from ...logging_config import get_logger

logger = get_logger(__name__)


class UserRepository:
    """
    Repository for user data access.
    
    Implements the persistence facade pattern by centralizing all user-related SQL operations
    and providing a clean interface for analyzers and other components.
    """

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize user repository.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self.base_filter = settings.base_filter.strip()

    # Sync methods for the new architecture

    def get_user_message_statistics(
        self, 
        username: str, 
        days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive message statistics for a user.
        
        Note: This method doesn't apply base filter since we want to analyze 
        any specific user, even if they're normally filtered out.
        
        Args:
            username: User name to analyze
            days: Number of days to look back (None = all time)
            
        Returns:
            Dictionary with user message statistics
        """
        query = """
        SELECT 
            COUNT(*) as total_messages,
            COUNT(DISTINCT channel_name) as unique_channels,
            MIN(timestamp) as first_message,
            MAX(timestamp) as last_message,
            AVG(LENGTH(content)) as avg_message_length
        FROM messages 
        WHERE author_name = ?
        """
        
        params = [username]
        
        if days:
            query += " AND timestamp >= datetime('now', '-' || ? || ' days')"
            params.append(days)
        
        query += " AND content IS NOT NULL"
        
        result = self.db_manager.execute_query(query, tuple(params), fetch_one=True)
        
        if not result:
            return {}
        
        return {
            'total_messages': result['total_messages'] or 0,
            'unique_channels': result['unique_channels'] or 0,
            'first_message': result['first_message'],
            'last_message': result['last_message'],
            'avg_message_length': result['avg_message_length'] or 0.0
        }

    def get_user_channel_activity(
        self, 
        username: str, 
        days: Optional[int] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get user's activity distribution across channels.
        
        Note: This method doesn't apply base filter since we want to analyze 
        any specific user, even if they're normally filtered out.
        
        Args:
            username: User name to analyze
            days: Number of days to look back
            limit: Maximum number of channels to return
            
        Returns:
            List of channel activity dictionaries
        """
        query = """
        SELECT 
            channel_name,
            COUNT(*) as message_count,
            MIN(timestamp) as first_message,
            MAX(timestamp) as last_message
        FROM messages 
        WHERE author_name = ?
        """
        
        params = [username]
        
        if days:
            query += " AND timestamp >= datetime('now', '-' || ? || ' days')"
            params.append(days)
        
        query += """
        AND content IS NOT NULL
        GROUP BY channel_name 
        ORDER BY message_count DESC 
        LIMIT ?
        """
        params.append(limit)
        
        results = self.db_manager.execute_query(query, tuple(params))
        
        return [
            {
                'channel_name': row['channel_name'],
                'message_count': row['message_count'],
                'first_message': row['first_message'],
                'last_message': row['last_message']
            }
            for row in results
        ] if results else []

    def get_user_hourly_patterns(
        self, 
        username: str, 
        days: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get user's hourly activity patterns.
        
        Note: This method doesn't apply base filter since we want to analyze 
        any specific user, even if they're normally filtered out.
        
        Args:
            username: User name to analyze
            days: Number of days to look back
            
        Returns:
            List of hourly activity data
        """
        query = """
        SELECT 
            strftime('%H', timestamp) as hour,
            COUNT(*) as message_count
        FROM messages 
        WHERE author_name = ?
        """
        
        params = [username]
        
        if days:
            query += " AND timestamp >= datetime('now', '-' || ? || ' days')"
            params.append(days)
        
        query += """
        AND timestamp IS NOT NULL
        AND content IS NOT NULL
        GROUP BY strftime('%H', timestamp)
        ORDER BY hour
        """
        
        results = self.db_manager.execute_query(query, tuple(params))
        
        return [
            {
                'hour': int(row['hour']),
                'message_count': row['message_count']
            }
            for row in results
        ] if results else []

    def get_available_users(self, limit: Optional[int] = None) -> List[str]:
        """
        Get list of available users (humans only) with display names.
        
        Args:
            limit: Maximum number of users to return (None = all users)
            
        Returns:
            List of display names sorted alphabetically
        """
        query = """
        SELECT DISTINCT 
            author_name,
            COALESCE(author_display_name, author_name) as display_name
        FROM messages 
        WHERE content IS NOT NULL
        AND (author_is_bot = 0 OR author_is_bot IS NULL)
        ORDER BY display_name
        """
        
        params = []
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        
        results = self.db_manager.execute_query(query, tuple(params) if params else ())
        
        return [row['display_name'] for row in results] if results else []

    def get_top_users_by_message_count(
        self, 
        limit: int = 10, 
        days: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get top users by message count (humans only) with display names.
        
        Args:
            limit: Maximum number of users to return
            days: Number of days to look back
            
        Returns:
            List of user statistics dictionaries
        """
        query = """
        SELECT 
            author_name,
            COALESCE(author_display_name, author_name) as display_name,
            COUNT(*) as message_count,
            COUNT(DISTINCT channel_name) as unique_channels,
            MIN(timestamp) as first_message,
            MAX(timestamp) as last_message,
            AVG(LENGTH(content)) as avg_message_length
        FROM messages 
        WHERE content IS NOT NULL
        AND (author_is_bot = 0 OR author_is_bot IS NULL)
        """
        
        params = []
        
        if days:
            query += " AND timestamp >= datetime('now', '-' || ? || ' days')"
            params.append(days)
        
        query += """
        GROUP BY author_name, author_display_name
        ORDER BY message_count DESC
        LIMIT ?
        """
        params.append(limit)
        
        results = self.db_manager.execute_query(query, tuple(params))
        
        return [
            {
                'author_name': row['author_name'],
                'display_name': row['display_name'],
                'message_count': row['message_count'],
                'unique_channels': row['unique_channels'],
                'first_message': row['first_message'],
                'last_message': row['last_message'],
                'avg_message_length': row['avg_message_length'] or 0.0
            }
            for row in results
        ] if results else []

    def get_users_for_autocomplete(self, limit: int = 200) -> List[Dict[str, str]]:
        """
        Get list of users with display names for autocomplete.
        
        Args:
            limit: Maximum number of users to return (default: 200 for better autocomplete)
            
        Returns:
            List of dictionaries with author_name, display_name, and message_count
        """
        query = """
        SELECT 
            author_name,
            author_display_name,
            COUNT(*) as message_count
        FROM messages 
        WHERE content IS NOT NULL
        AND (author_is_bot = 0 OR author_is_bot IS NULL)
        AND author_name IS NOT NULL
        AND author_name != ''
        GROUP BY author_name, author_display_name
        ORDER BY message_count DESC, author_name
        LIMIT ?
        """
        
        results = self.db_manager.execute_query(query, (limit,))
        
        return [
            {
                'author_name': row['author_name'],
                'display_name': row['author_display_name'] or row['author_name'],
                'message_count': str(row['message_count'])
            }
            for row in results
        ] if results else []

    def get_user_enhanced_statistics(
        self, 
        username: str, 
        days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get enhanced user statistics including active days and detailed metrics.
        
        Args:
            username: User name to analyze
            days: Number of days to look back (None = all time)
            
        Returns:
            Dictionary with enhanced user statistics
        """
        query = """
        SELECT 
            COUNT(*) as total_messages,
            COUNT(DISTINCT channel_name) as unique_channels,
            MIN(timestamp) as first_message,
            MAX(timestamp) as last_message,
            AVG(LENGTH(content)) as avg_message_length,
            COUNT(DISTINCT DATE(timestamp)) as active_days
        FROM messages 
        WHERE author_name = ?
        """
        
        params = [username]
        
        if days:
            query += " AND timestamp >= datetime('now', '-' || ? || ' days')"
            params.append(days)
        
        query += " AND content IS NOT NULL"
        
        result = self.db_manager.execute_query(query, tuple(params), fetch_one=True)
        
        if not result:
            return {}
        
        return {
            'total_messages': result['total_messages'] or 0,
            'unique_channels': result['unique_channels'] or 0,
            'first_message': result['first_message'],
            'last_message': result['last_message'],
            'avg_message_length': result['avg_message_length'] or 0.0,
            'active_days': result['active_days'] or 0
        }

    def get_user_channel_activity_enhanced(
        self, 
        username: str, 
        days: Optional[int] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get enhanced user's activity distribution across channels with message length stats.
        
        Args:
            username: User name to analyze
            days: Number of days to look back
            limit: Maximum number of channels to return
            
        Returns:
            List of enhanced channel activity dictionaries
        """
        query = """
        SELECT 
            channel_name,
            COUNT(*) as message_count,
            MIN(timestamp) as first_message,
            MAX(timestamp) as last_message,
            AVG(LENGTH(content)) as avg_message_length
        FROM messages 
        WHERE author_name = ?
        """
        
        params = [username]
        
        if days:
            query += " AND timestamp >= datetime('now', '-' || ? || ' days')"
            params.append(days)
        
        query += """
        AND content IS NOT NULL
        GROUP BY channel_name 
        ORDER BY message_count DESC 
        LIMIT ?
        """
        params.append(limit)
        
        results = self.db_manager.execute_query(query, tuple(params))
        
        return [
            {
                'channel_name': row['channel_name'],
                'message_count': row['message_count'],
                'first_message': row['first_message'],
                'last_message': row['last_message'],
                'avg_message_length': row['avg_message_length'] or 0.0
            }
            for row in results
        ] if results else []

    def get_user_time_of_day_patterns(
        self, 
        username: str, 
        days: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Get user's activity patterns by time of day periods.
        
        Args:
            username: User name to analyze
            days: Number of days to look back
            
        Returns:
            Dictionary with time period activity counts
        """
        query = """
        SELECT 
            CASE 
                WHEN CAST(strftime('%H', timestamp) AS INTEGER) BETWEEN 0 AND 5 THEN 'Night (00-05)'
                WHEN CAST(strftime('%H', timestamp) AS INTEGER) BETWEEN 6 AND 11 THEN 'Morning (06-11)'
                WHEN CAST(strftime('%H', timestamp) AS INTEGER) BETWEEN 12 AND 17 THEN 'Afternoon (12-17)'
                WHEN CAST(strftime('%H', timestamp) AS INTEGER) BETWEEN 18 AND 23 THEN 'Evening (18-23)'
                ELSE 'Unknown'
            END as time_period,
            COUNT(*) as message_count
        FROM messages 
        WHERE author_name = ?
        """
        
        params = [username]
        
        if days:
            query += " AND timestamp >= datetime('now', '-' || ? || ' days')"
            params.append(days)
        
        query += """
        AND timestamp IS NOT NULL
        AND content IS NOT NULL
        GROUP BY time_period
        ORDER BY message_count DESC
        """
        
        results = self.db_manager.execute_query(query, tuple(params))
        
        return {
            row['time_period']: row['message_count']
            for row in results
        } if results else {}

    def get_user_content_sample(
        self, 
        username: str, 
        days: Optional[int] = None,
        limit: int = 100
    ) -> List[str]:
        """
        Get sample content from a user for semantic analysis.
        
        Args:
            username: User name to analyze
            days: Number of days to look back
            limit: Maximum number of messages to return
            
        Returns:
            List of message content strings
        """
        query = """
        SELECT content
        FROM messages 
        WHERE author_name = ?
        """
        
        params = [username]
        
        if days:
            query += " AND timestamp >= datetime('now', '-' || ? || ' days')"
            params.append(days)
        
        query += """
        AND content IS NOT NULL
        AND LENGTH(content) > 10
        ORDER BY RANDOM()
        LIMIT ?
        """
        params.append(limit)
        
        results = self.db_manager.execute_query(query, tuple(params))
        
        return [row['content'] for row in results] if results else []

    def get_user_temporal_data(
        self, username: str, days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get user's daily activity data for chart generation.
        
        Args:
            username: User name to analyze
            days: Number of days to look back (None = all time)
            
        Returns:
            Dictionary with daily activity data
        """
        query = """
        SELECT 
            DATE(timestamp) as date,
            COUNT(*) as message_count
        FROM messages 
        WHERE author_name = ?
        """
        
        params = [username]
        
        if days:
            query += " AND timestamp >= datetime('now', '-' || ? || ' days')"
            params.append(days)
        
        query += """
        AND timestamp IS NOT NULL
        AND content IS NOT NULL
        GROUP BY DATE(timestamp)
        ORDER BY date
        """
        
        results = self.db_manager.execute_query(query, tuple(params))
        
        activity_by_day = [
            {
                'date': row['date'],
                'message_count': row['message_count']
            }
            for row in results
        ] if results else []
        
        return {
            'activity_by_day': activity_by_day,
            'total_days': len(activity_by_day),
            'total_messages': sum(day['message_count'] for day in activity_by_day)
        }

    # Existing async methods (keep for compatibility)

    def get_users_by_channel(
        self, channel_name: str, limit: Optional[int] = None
    ) -> List[User]:
        """Get users who have posted in a specific channel."""
        query = f"""
            SELECT DISTINCT author_id, author_name, author_display_name
            FROM messages 
            WHERE channel_name = ? AND {self.base_filter}
            ORDER BY author_name
        """
        params = [channel_name]

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        rows = self.db_manager.execute_query(query, tuple(params))
        return [User.from_db_row(row) for row in rows]

    def get_user_message_count(
        self, author_id: str, channel_name: Optional[str] = None
    ) -> int:
        """Get total message count for a user."""
        query = f"""
            SELECT COUNT(*) as count
            FROM messages 
            WHERE author_id = ? AND {self.base_filter}
        """
        params = [author_id]

        if channel_name:
            query += " AND channel_name = ?"
            params.append(channel_name)

        row = self.db_manager.execute_query(query, tuple(params), fetch_one=True)
        return row["count"] if row else 0

    def get_user_activity_summary(
        self, author_id: str
    ) -> Dict[str, Any]:
        """Get comprehensive activity summary for a user."""
        query = f"""
            SELECT 
                COUNT(*) as total_messages,
                COUNT(DISTINCT channel_name) as channels_active,
                COUNT(DISTINCT DATE(timestamp)) as active_days,
                AVG(LENGTH(content)) as avg_message_length,
                MIN(timestamp) as first_message,
                MAX(timestamp) as last_message
            FROM messages 
            WHERE author_id = ? AND {self.base_filter}
            AND content IS NOT NULL
        """

        row = self.db_manager.execute_query(query, (author_id,), fetch_one=True)

        if not row:
            return {}

        return {
            "total_messages": row["total_messages"],
            "channels_active": row["channels_active"],
            "active_days": row["active_days"],
            "avg_message_length": row["avg_message_length"] or 0.0,
            "first_message": row["first_message"],
            "last_message": row["last_message"],
        }

    def get_top_users(
        self,
        limit: int = 10,
        days_back: Optional[int] = None,
        channel_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get top users by message count."""
        query = f"""
            SELECT 
                author_id,
                author_name,
                author_display_name,
                COUNT(*) as message_count,
                COUNT(DISTINCT channel_name) as channels_active,
                AVG(LENGTH(content)) as avg_message_length,
                MIN(timestamp) as first_message,
                MAX(timestamp) as last_message
            FROM messages 
            WHERE {self.base_filter}
            AND content IS NOT NULL
        """

        params = []

        if days_back:
            query += " AND timestamp >= datetime('now', '-' || ? || ' days')"
            params.append(days_back)

        if channel_name:
            query += " AND channel_name = ?"
            params.append(channel_name)

        query += """
            GROUP BY author_id, author_name, author_display_name
            ORDER BY message_count DESC
            LIMIT ?
        """
        params.append(limit)

        rows = self.db_manager.execute_query(query, tuple(params))
        return [
            {
                "author_id": row["author_id"],
                "author_name": row["author_name"],
                "author_display_name": row["author_display_name"],
                "message_count": row["message_count"],
                "channels_active": row["channels_active"],
                "avg_message_length": row["avg_message_length"] or 0.0,
                "first_message": row["first_message"],
                "last_message": row["last_message"],
            }
            for row in rows
        ]

    def get_user_daily_activity(
        self, author_id: str, days_back: int = 30
    ) -> List[tuple]:
        """Get daily activity data for a user."""
        from datetime import datetime, timedelta

        threshold_date = (datetime.now() - timedelta(days=days_back)).isoformat()

        query = f"""
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as message_count
            FROM messages
            WHERE author_id = ? AND {self.base_filter}
            AND content IS NOT NULL
            AND timestamp >= ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        """

        rows = self.db_manager.execute_query(query, (author_id, threshold_date))
        return [(row["date"], row["message_count"]) for row in rows]

    def get_user_statistics(self, limit: int = 10) -> List[User]:
        """Get user statistics ordered by message count."""
        query = f"""
            SELECT 
                author_id,
                author_name,
                author_display_name,
                COUNT(*) as message_count,
                COUNT(DISTINCT channel_name) as channels_active,
                AVG(LENGTH(content)) as avg_message_length,
                MIN(timestamp) as first_message,
                MAX(timestamp) as last_message
            FROM messages
            WHERE {self.base_filter}
            AND content IS NOT NULL
            GROUP BY author_id, author_name, author_display_name
            ORDER BY message_count DESC
            LIMIT ?
        """

        rows = self.db_manager.execute_query(query, (limit,))
        return [
            User(
                author_id=row["author_id"],
                author_name=row["author_name"],
                author_display_name=row["author_display_name"],
                message_count=row["message_count"],
                channels_active=row["channels_active"],
                avg_message_length=row["avg_message_length"] or 0.0,
                first_message=row["first_message"],
                last_message=row["last_message"],
            )
            for row in rows
        ]

    def get_user_by_name(self, user_name: str) -> Optional[User]:
        """Get a specific user by name (either author_name or display_name)."""
        query = f"""
            SELECT DISTINCT 
                author_id, 
                author_name, 
                author_display_name
            FROM messages 
            WHERE (author_name LIKE ? OR author_display_name LIKE ?) 
            AND {self.base_filter}
            LIMIT 1
        """

        row = self.db_manager.execute_query(
            query, (f"%{user_name}%", f"%{user_name}%"), fetch_one=True
        )

        return User.from_db_row(row) if row else None

    def get_top_users_in_channel(
        self, channel_name: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top users in a specific channel."""
        query = f"""
            SELECT 
                author_id,
                author_name,
                author_display_name,
                COUNT(*) as message_count,
                AVG(LENGTH(content)) as avg_message_length,
                MIN(timestamp) as first_message,
                MAX(timestamp) as last_message
            FROM messages 
            WHERE channel_name = ? AND {self.base_filter}
            AND content IS NOT NULL
            GROUP BY author_id, author_name, author_display_name
            ORDER BY message_count DESC
            LIMIT ?
        """

        results = self.db_manager.execute_query(query, (channel_name, limit))
        return [
            {
                "author_id": row["author_id"],
                "author_name": row["author_name"],
                "author_display_name": row["author_display_name"],
                "message_count": row["message_count"],
                "avg_message_length": row["avg_message_length"] or 0.0,
                "first_message": row["first_message"],
                "last_message": row["last_message"],
            }
            for row in results
        ]

    def get_user_list(self, base_filter: str = None) -> List[Dict[str, str]]:
        """Get list of all users, ordered by activity."""
        filter_clause = base_filter or self.base_filter
        query = f"""
            SELECT 
                author_name,
                author_display_name,
                COUNT(*) as message_count
            FROM messages 
            WHERE author_name IS NOT NULL 
                AND author_name != ''
                AND {filter_clause}
            GROUP BY author_name, author_display_name
            ORDER BY message_count DESC, author_name
        """
        rows = self.db_manager.execute_query(query)
        return [
            {
                "author_name": row["author_name"],
                "display_name": row["author_display_name"] or row["author_name"],
                "message_count": str(row["message_count"]),
            }
            for row in rows
            if row["author_name"]
        ]

    async def get_user_statistics_by_id(
        self, author_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get comprehensive user statistics by author ID."""
        query = f"""
            SELECT 
                COUNT(*) as total_messages,
                COUNT(DISTINCT channel_name) as channels_active,
                AVG(LENGTH(content)) as avg_message_length,
                COUNT(DISTINCT DATE(timestamp)) as active_days,
                MIN(timestamp) as first_message,
                MAX(timestamp) as last_message
            FROM messages
            WHERE author_id = ? AND {self.base_filter}
            AND content IS NOT NULL
        """

        result = await self.db_manager.execute_single(query, (author_id,))

        if result and result[0] > 0:
            return {
                "total_messages": result[0],
                "channels_active": result[1],
                "avg_message_length": float(result[2]) if result[2] else 0.0,
                "active_days": result[3],
                "first_message": result[4],
                "last_message": result[5],
            }
        return None

    async def get_user_content_sample(
        self, author_id: str, limit: int = 100
    ) -> List[str]:
        """Get sample content from a user for concept analysis."""
        query = f"""
            SELECT content
            FROM messages
            WHERE author_id = ? AND {self.base_filter}
            AND content IS NOT NULL
            AND LENGTH(content) > 20
            ORDER BY RANDOM()
            LIMIT ?
        """

        results = await self.db_manager.execute_query(query, (author_id, limit))
        return [row["content"] for row in results if row["content"]]

    async def find_user_by_name(self, user_name: str) -> Optional[Dict[str, Any]]:
        """Find user by name or display name (compatibility alias)."""
        user = await self.get_user_by_name(user_name)
        if user:
            return {
                "author_id": user.author_id,
                "display_name": user.author_display_name,
                "author_name": user.author_name,
            }
        return None
