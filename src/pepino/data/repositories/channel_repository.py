"""
Channel repository for data access operations.
Provides centralized data access for channel-related operations following the persistence facade pattern.
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from ..config import settings
from ..database.manager import DatabaseManager
from ..models.channel import Channel

logger = logging.getLogger(__name__)


class ChannelRepository:
    """
    Repository for channel data access.
    
    Implements the persistence facade pattern by centralizing all channel-related SQL operations
    and providing a clean interface for analyzers and other components.
    """

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize channel repository.
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self.base_filter = settings.base_filter.strip()

    # Sync methods for the new architecture

    def get_channel_message_statistics(
        self, 
        channel_name: str, 
        days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive message statistics for a channel.
        
        Args:
            channel_name: Channel name to analyze
            days: Number of days to look back (None = all time)
            
        Returns:
            Dictionary with channel message statistics
        """
        query = """
        SELECT 
            COUNT(*) as total_messages,
            COUNT(DISTINCT author_name) as unique_users,
            MIN(timestamp) as first_message,
            MAX(timestamp) as last_message,
            AVG(LENGTH(content)) as avg_message_length,
            COUNT(CASE WHEN author_is_bot = 1 THEN 1 END) as bot_messages,
            COUNT(CASE WHEN author_is_bot = 0 OR author_is_bot IS NULL THEN 1 END) as human_messages,
            COUNT(DISTINCT CASE WHEN author_is_bot = 0 OR author_is_bot IS NULL THEN author_name END) as unique_human_users
        FROM messages 
        WHERE channel_name = ?
        """
        
        params = [channel_name]
        
        if days:
            query += " AND timestamp >= datetime('now', '-' || ? || ' days')"
            params.append(days)
        
        query += " AND content IS NOT NULL"
        
        result = self.db_manager.execute_query(query, tuple(params), fetch_one=True)
        
        if not result:
            return {}
        
        return {
            'total_messages': result['total_messages'] or 0,
            'unique_users': result['unique_users'] or 0,
            'first_message': result['first_message'],
            'last_message': result['last_message'],
            'avg_message_length': result['avg_message_length'] or 0.0,
            'bot_messages': result['bot_messages'] or 0,
            'human_messages': result['human_messages'] or 0,
            'unique_human_users': result['unique_human_users'] or 0
        }

    def get_channel_user_activity(
        self, 
        channel_name: str, 
        days: Optional[int] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get top users by activity in a specific channel (humans only).
        
        Args:
            channel_name: Channel name to analyze
            days: Number of days to look back
            limit: Maximum number of users to return
            
        Returns:
            List of user activity dictionaries
        """
        query = """
        SELECT 
            author_name,
            author_display_name,
            COUNT(*) as message_count,
            MIN(timestamp) as first_message,
            MAX(timestamp) as last_message,
            AVG(LENGTH(content)) as avg_message_length
        FROM messages 
        WHERE channel_name = ?
        AND (author_is_bot = 0 OR author_is_bot IS NULL)
        """
        
        params = [channel_name]
        
        if days:
            query += " AND timestamp >= datetime('now', '-' || ? || ' days')"
            params.append(days)
        
        query += """
        AND content IS NOT NULL
        GROUP BY author_name, author_display_name
        ORDER BY message_count DESC 
        LIMIT ?
        """
        params.append(limit)
        
        results = self.db_manager.execute_query(query, tuple(params))
        
        return [
            {
                'author_name': row['author_name'],
                'author_display_name': row['author_display_name'],
                'message_count': row['message_count'],
                'first_message': row['first_message'],
                'last_message': row['last_message'],
                'avg_message_length': row['avg_message_length'] or 0.0
            }
            for row in results
        ] if results else []

    def get_channel_hourly_patterns(
        self, 
        channel_name: str, 
        days: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get channel's hourly activity patterns (humans only).
        
        Args:
            channel_name: Channel name to analyze
            days: Number of days to look back
            
        Returns:
            List of hourly activity data
        """
        query = """
        SELECT 
            strftime('%H', timestamp) as hour,
            COUNT(*) as message_count
        FROM messages 
        WHERE channel_name = ?
        AND (author_is_bot = 0 OR author_is_bot IS NULL)
        """
        
        params = [channel_name]
        
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

    def get_available_channels(self, limit: int = 500) -> List[str]:
        """
        Get list of available channels.
        
        Args:
            limit: Maximum number of channels to return (default: 500 for better autocomplete)
            
        Returns:
            List of channel names sorted alphabetically
        """
        query = """
        SELECT DISTINCT channel_name
        FROM messages 
        WHERE content IS NOT NULL
        ORDER BY channel_name
        LIMIT ?
        """
        
        results = self.db_manager.execute_query(query, (limit,))
        
        return [row['channel_name'] for row in results] if results else []

    def get_top_channels_by_message_count(
        self, 
        limit: int = 10, 
        days: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get top channels by message count.
        
        Args:
            limit: Maximum number of channels to return
            days: Number of days to look back
            
        Returns:
            List of channel statistics dictionaries
        """
        query = """
        SELECT 
            channel_name,
            COUNT(*) as message_count,
            COUNT(DISTINCT author_name) as unique_users,
            MIN(timestamp) as first_message,
            MAX(timestamp) as last_message
        FROM messages 
        WHERE content IS NOT NULL
        """
        
        params = []
        
        if days:
            query += " AND timestamp >= datetime('now', '-' || ? || ' days')"
            params.append(days)
        
        query += """
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
                'unique_users': row['unique_users'],
                'first_message': row['first_message'],
                'last_message': row['last_message']
            }
            for row in results
        ] if results else []

    def get_channel_daily_patterns(
        self, 
        channel_name: str, 
        days: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get channel's daily activity patterns.
        
        Args:
            channel_name: Channel name to analyze
            days: Number of days to look back
            
        Returns:
            List of daily activity data
        """
        query = """
        SELECT 
            DATE(timestamp) as date,
            COUNT(*) as message_count
        FROM messages 
        WHERE channel_name = ?
        """
        
        params = [channel_name]
        
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
        
        return [
            {
                'date': row['date'],
                'message_count': row['message_count']
            }
            for row in results
        ] if results else []

    # Existing async methods (keep for compatibility)

    async def get_all_channels(self) -> List[Channel]:
        """Get all channels from the database."""
        query = f"""
            SELECT DISTINCT channel_id, channel_name
            FROM messages 
            WHERE {self.base_filter}
            ORDER BY channel_name
        """

        rows = await self.db_manager.execute_query(query)
        return [Channel.from_db_row(row) for row in rows]

    async def get_channel_by_name(self, channel_name: str) -> Optional[Channel]:
        """Get a specific channel by name."""
        query = f"""
            SELECT DISTINCT channel_id, channel_name
            FROM messages 
            WHERE channel_name = ? AND {self.base_filter}
            LIMIT 1
        """

        rows = await self.db_manager.execute_query(query, (channel_name,))
        return Channel.from_db_row(rows[0]) if rows else None

    async def get_channel_statistics(
        self, channel_name: str, days_back: int = 30
    ) -> Dict[str, Any]:
        """Get detailed statistics for a channel."""
        query = f"""
            SELECT 
                COUNT(*) as total_messages,
                COUNT(DISTINCT author_id) as unique_users,
                COUNT(DISTINCT DATE(timestamp)) as active_days,
                AVG(LENGTH(content)) as avg_message_length,
                MIN(timestamp) as first_message,
                MAX(timestamp) as last_message
            FROM messages 
            WHERE channel_name = ? AND {self.base_filter}
            AND timestamp >= datetime('now', '-{days_back} days')
            AND content IS NOT NULL
        """

        row = await self.db_manager.execute_single(query, (channel_name,))

        if not row:
            return {}

        return {
            "total_messages": row[0],
            "unique_users": row[1],
            "active_days": row[2],
            "avg_message_length": row[3],
            "first_message": row[4],
            "last_message": row[5],
        }

    async def get_channel_daily_activity(
        self, channel_name: str, days_back: int = 30
    ) -> List[tuple]:
        """Get daily activity data for a channel."""
        from datetime import datetime, timedelta

        threshold_date = (datetime.now() - timedelta(days=days_back)).isoformat()

        query = f"""
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as message_count
            FROM messages
            WHERE channel_name = ? AND {self.base_filter}
            AND content IS NOT NULL
            AND timestamp >= ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        """

        rows = await self.db_manager.execute_query(query, (channel_name, threshold_date))
        return [(row["date"], row["message_count"]) for row in rows]

    async def get_top_channels(
        self, limit: int = 10, days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """Get top channels by message count."""
        query = f"""
            SELECT 
                channel_name,
                COUNT(*) as message_count,
                COUNT(DISTINCT author_id) as unique_users,
                AVG(LENGTH(content)) as avg_message_length,
                MIN(timestamp) as first_message,
                MAX(timestamp) as last_message
            FROM messages 
            WHERE {self.base_filter}
            AND timestamp >= datetime('now', '-{days_back} days')
            AND content IS NOT NULL
            GROUP BY channel_name
            ORDER BY message_count DESC
            LIMIT ?
        """

        rows = await self.db_manager.execute_query(query, (limit,))
        return [
            {
                "channel_name": row["channel_name"],
                "message_count": row["message_count"],
                "unique_users": row["unique_users"],
                "avg_message_length": row["avg_message_length"] or 0.0,
                "first_message": row["first_message"],
                "last_message": row["last_message"],
            }
            for row in rows
        ]

    async def save_channel_members(self, messages_data: Dict[str, Any]) -> None:
        """Save channel members data to database"""
        if not messages_data:
            return

        # Clear existing channel members data to avoid stale data
        await self.db_manager.execute_query("DELETE FROM channel_members")
        logger.info("Cleared existing channel members data")

        # Extract all channel members from the data
        all_members = []
        for guild_name, channels in messages_data.items():
            if "_channel_members" in channels:
                for channel_name, members in channels["_channel_members"].items():
                    all_members.extend(members)

        if not all_members:
            logger.info("No channel members to save")
            return

        logger.info(f"Saving {len(all_members)} channel member records to database...")

        # Process members in batches
        batch_size = 100
        for i in range(0, len(all_members), batch_size):
            batch = all_members[i : i + batch_size]

            try:
                query = """
                    INSERT OR REPLACE INTO channel_members (
                        channel_id, channel_name, guild_id, guild_name,
                        user_id, user_name, user_display_name, user_joined_at,
                        user_roles, is_bot, member_permissions, synced_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

                values = [
                    (
                        member.get("channel_id", ""),
                        member.get("channel_name", ""),
                        member.get("guild_id", ""),
                        member.get("guild_name", ""),
                        member.get("user_id", ""),
                        member.get("user_name", ""),
                        member.get("user_display_name", ""),
                        member.get("user_joined_at"),
                        member.get("user_roles", "[]"),
                        member.get("is_bot", False),
                        member.get("member_permissions", "{}"),
                        member.get("synced_at", datetime.now(timezone.utc).isoformat()),
                    )
                    for member in batch
                ]

                await self.db_manager.execute_many(query, values)
                logger.info(f"✅ Saved batch of {len(batch)} channel member records")

            except Exception as e:
                logger.error(f"Error saving channel members batch: {str(e)}")
                raise

    async def get_channel_statistics(self, limit: int = 10) -> List[Channel]:
        """Get channel statistics ordered by message count."""
        query = f"""
            SELECT 
                channel_id,
                channel_name,
                COUNT(*) as message_count,
                COUNT(DISTINCT author_id) as unique_users,
                AVG(LENGTH(content)) as avg_message_length,
                MIN(timestamp) as first_message,
                MAX(timestamp) as last_message
            FROM messages
            WHERE {self.base_filter}
            GROUP BY channel_id, channel_name
            ORDER BY message_count DESC
            LIMIT ?
        """

        rows = await self.db_manager.execute_query(query, (limit,))
        return [
            Channel(
                channel_id=row[0],
                channel_name=row[1],
                message_count=row[2],
                unique_users=row[3],
                avg_message_length=row[4] or 0.0,
                first_message=row[5],
                last_message=row[6],
            )
            for row in rows
        ]

    async def get_channel_list(self, base_filter: str = None) -> list:
        """Get list of all channel names, ordered by activity."""
        filter_clause = base_filter or self.base_filter
        query = f"""
            SELECT 
                channel_name,
                COUNT(*) as message_count
            FROM messages 
            WHERE channel_name IS NOT NULL 
                AND channel_name != ''
                AND {filter_clause}
            GROUP BY channel_name
            ORDER BY message_count DESC, channel_name
        """
        rows = await self.db_manager.execute_query(query)
        return [row["channel_name"] for row in rows if row["channel_name"]]

    def get_channel_id_by_name(self, channel_name: str) -> Optional[str]:
        """Get channel_id by channel_name."""
        query = f"""
            SELECT DISTINCT channel_id 
            FROM messages 
            WHERE channel_name = ? AND {self.base_filter}
            LIMIT 1
        """

        result = self.db_manager.execute_query(query, (channel_name,), fetch_one=True)
        return result["channel_id"] if result else None

    def get_distinct_channel_count(self) -> int:
        """Get count of distinct channels."""
        query = (
            f"SELECT COUNT(DISTINCT channel_id) FROM messages WHERE {self.base_filter}"
        )
        result = self.db_manager.execute_query(query, fetch_one=True)
        return result[0] if result else 0

    def get_channel_human_member_count(self, channel_name: str) -> int:
        """
        Get the total number of human members who have ever posted in a channel.
        
        Args:
            channel_name: Channel name to analyze
            
        Returns:
            Total number of unique human members
        """
        query = """
        SELECT COUNT(DISTINCT author_name) as total_human_members
        FROM messages 
        WHERE channel_name = ?
        AND (author_is_bot = 0 OR author_is_bot IS NULL)
        AND content IS NOT NULL
        """
        
        result = self.db_manager.execute_query(query, (channel_name,), fetch_one=True)
        
        return result['total_human_members'] if result else 0
