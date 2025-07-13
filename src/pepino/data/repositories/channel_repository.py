"""
Channel repository for channels data access operations.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from ...config import settings
from ..database.manager import DatabaseManager
from ..models.channel import Channel
from ...logging_config import get_logger

logger = get_logger(__name__)


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
        self.base_filter = settings.analysis_base_filter_sql.strip()

    def _build_base_conditions(
        self, 
        channel_name: Optional[str] = None, 
        days: Optional[int] = None,
        exclude_bots: bool = False,
        include_content_check: bool = True
    ) -> Tuple[str, List[Any]]:
        """
        Build common WHERE conditions for channel queries.
        
        Args:
            channel_name: Optional channel name filter
            days: Optional days back filter
            exclude_bots: Whether to exclude bot messages
            include_content_check: Whether to include content IS NOT NULL check
            
        Returns:
            Tuple of (WHERE clause, parameters list)
        """
        conditions = [self.base_filter]
        params = []
        
        if channel_name:
            conditions.append("channel_name = ?")
            params.append(channel_name)
            
        if days:
            conditions.append("timestamp >= datetime('now', '-' || ? || ' days')")
            params.append(days)
            
        if exclude_bots:
            conditions.append("(author_is_bot = 0 OR author_is_bot IS NULL)")
            
        if include_content_check:
            conditions.append("content IS NOT NULL")
            
        return " AND ".join(conditions), params

    def _build_channel_stats_select(self, include_bot_stats: bool = False) -> str:
        """
        Build common SELECT clause for channel statistics.
        
        Args:
            include_bot_stats: Whether to include bot-specific statistics
            
        Returns:
            SELECT clause string
        """
        base_select = """
            COUNT(*) as total_messages,
            COUNT(DISTINCT author_name) as unique_users,
            MIN(timestamp) as first_message,
            MAX(timestamp) as last_message,
            AVG(LENGTH(content)) as avg_message_length
        """
        
        if include_bot_stats:
            base_select += """,
            COUNT(CASE WHEN author_is_bot = 1 THEN 1 END) as bot_messages,
            COUNT(CASE WHEN author_is_bot = 0 OR author_is_bot IS NULL THEN 1 END) as human_messages,
            COUNT(DISTINCT CASE WHEN author_is_bot = 0 OR author_is_bot IS NULL THEN author_name END) as unique_human_users
        """
            
        return base_select

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
        # Build subquery conditions for date filtering
        date_condition = ""
        if days:
            date_condition = " AND timestamp >= datetime('now', '-' || ? || ' days')"
        
        # For accurate bot statistics, we need to count all messages first, then apply filters for human analysis
        query = f"""
        SELECT 
            COUNT(*) as total_messages_filtered,
            COUNT(DISTINCT author_name) as unique_users_filtered,
            MIN(timestamp) as first_message,
            MAX(timestamp) as last_message,
            AVG(LENGTH(content)) as avg_message_length,
            COUNT(CASE WHEN author_is_bot = 0 OR author_is_bot IS NULL THEN 1 END) as human_messages,
            COUNT(DISTINCT CASE WHEN author_is_bot = 0 OR author_is_bot IS NULL THEN author_name END) as unique_human_users,
            -- Get accurate bot counts without base filter interference
            (SELECT COUNT(*) FROM messages WHERE channel_name = ? AND author_is_bot = 1 AND content IS NOT NULL{date_condition}) as bot_messages,
            (SELECT COUNT(*) FROM messages WHERE channel_name = ? AND content IS NOT NULL{date_condition}) as total_messages_actual,
            (SELECT COUNT(DISTINCT author_name) FROM messages WHERE channel_name = ? AND content IS NOT NULL{date_condition}) as unique_users_actual
        FROM messages 
        WHERE {where_clause}
        """
        
        # Build parameters for the main query and subqueries
        params = []
        
        # Parameters for subqueries (bot_messages, total_messages_actual, unique_users_actual)
        if days:
            params.extend([channel_name, days, channel_name, days, channel_name, days])  # 3 subqueries with days
        else:
            params.extend([channel_name, channel_name, channel_name])  # 3 subqueries without days
        
        # Parameter for main query
        params.append(channel_name)
        
        if days:
            query += " AND timestamp >= datetime('now', '-' || ? || ' days')"
            params.append(days)
        
        query += " AND content IS NOT NULL"
        
        result = self.db_manager.execute_query(query, tuple(params), fetch_one=True)
        
        if not result:
            return {}
        
        return {
            'total_messages': result['total_messages_actual'] or 0,  # Use actual total (including all bots)
            'unique_users': result['unique_users_actual'] or 0,  # Use actual unique users (including all bots)
            'first_message': result['first_message'],
            'last_message': result['last_message'],
            'avg_message_length': result['avg_message_length'] or 0.0,
            'bot_messages': result['bot_messages'] or 0,  # Accurate bot count without base filter
            'human_messages': result['human_messages'] or 0,  # Human count with base filter applied
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
        where_clause, params = self._build_base_conditions(
            channel_name=channel_name, 
            days=days, 
            exclude_bots=True,
            include_content_check=True
        )
        
        query = f"""
        SELECT 
            author_name,
            author_display_name,
            COUNT(*) as message_count,
            MIN(timestamp) as first_message,
            MAX(timestamp) as last_message,
            AVG(LENGTH(content)) as avg_message_length
        FROM messages 
        WHERE {where_clause}
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
        where_clause, params = self._build_base_conditions(
            channel_name=channel_name, 
            days=days, 
            exclude_bots=True,
            include_content_check=True
        )
        
        query = f"""
        SELECT 
            strftime('%H', timestamp) as hour,
            COUNT(*) as message_count
        FROM messages 
        WHERE {where_clause}
        AND timestamp IS NOT NULL
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

    def get_available_channels(self, limit: Optional[int] = None) -> List[str]:
        """
        Get list of available channels.
        
        Args:
            limit: Maximum number of channels to return (None = all channels)
            
        Returns:
            List of channel names sorted alphabetically
        """
        where_clause, params = self._build_base_conditions(include_content_check=True)
        
        query = f"""
        SELECT DISTINCT channel_name
        FROM messages 
        WHERE {where_clause}
        ORDER BY channel_name
        """
        params.append(limit)
        
        params = []
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        
        results = self.db_manager.execute_query(query, tuple(params) if params else ())
        
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
        select_clause = self._build_channel_stats_select(include_bot_stats=False)
        where_clause, params = self._build_base_conditions(days=days, include_content_check=True)
        
        query = f"""
        SELECT 
            channel_name,
            {select_clause}
        FROM messages 
        WHERE {where_clause}
        GROUP BY channel_name
        ORDER BY total_messages DESC
        LIMIT ?
        """
        params.append(limit)
        
        results = self.db_manager.execute_query(query, tuple(params))
        
        return [
            {
                'channel_name': row['channel_name'],
                'message_count': row['total_messages'],
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
        where_clause, params = self._build_base_conditions(
            channel_name=channel_name, 
            days=days, 
            include_content_check=True
        )
        
        query = f"""
        SELECT 
            DATE(timestamp) as date,
            COUNT(*) as message_count
        FROM messages 
        WHERE {where_clause}
        AND timestamp IS NOT NULL
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

    def get_all_channels(self) -> List[Channel]:
        """Get all channels from the database."""
        where_clause, params = self._build_base_conditions()
        
        query = f"""
            SELECT DISTINCT channel_id, channel_name
            FROM messages 
            WHERE {where_clause}
            ORDER BY channel_name
        """

        rows = self.db_manager.execute_query(query, tuple(params))
        return [Channel.from_db_row(row) for row in rows]

    def get_all_channels_as_dicts(self) -> List[Dict[str, str]]:
        """
        Get all channels as a list of dicts with 'id' and 'name' keys.
        
        Returns:
            List of channel dictionaries with 'id' and 'name' keys
        """
        where_clause, params = self._build_base_conditions()
        
        query = f"""
            SELECT DISTINCT channel_id, channel_name
            FROM messages 
            WHERE {where_clause}
            ORDER BY channel_name
        """

        rows = self.db_manager.execute_query(query, tuple(params))
        return [{'id': row['channel_id'], 'name': row['channel_name']} for row in rows]

    def get_channel_by_name(self, channel_name: str) -> Optional[Channel]:
        """Get a specific channel by name."""
        where_clause, params = self._build_base_conditions(channel_name=channel_name)
        
        query = f"""
            SELECT DISTINCT channel_id, channel_name
            FROM messages 
            WHERE {where_clause}
            LIMIT 1
        """

        rows = self.db_manager.execute_query(query, tuple(params))
        return Channel.from_db_row(rows[0]) if rows else None

    def get_channel_statistics_by_name(
        self, channel_name: str, days_back: int = 30
    ) -> Dict[str, Any]:
        """
        Get detailed statistics for a specific channel by name.
        
        Args:
            channel_name: Channel name to analyze
            days_back: Number of days to look back
            
        Returns:
            Dictionary with channel statistics
        """
        where_clause, params = self._build_base_conditions(
            channel_name=channel_name, 
            days=days_back, 
            include_content_check=True
        )
        
        query = f"""
            SELECT 
                COUNT(*) as total_messages,
                COUNT(DISTINCT author_id) as unique_users,
                COUNT(DISTINCT DATE(timestamp)) as active_days,
                AVG(LENGTH(content)) as avg_message_length,
                MIN(timestamp) as first_message,
                MAX(timestamp) as last_message
            FROM messages 
            WHERE {where_clause}
        """

        row = self.db_manager.execute_single(query, tuple(params))

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

    def get_channel_daily_activity(
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

        rows = self.db_manager.execute_query(query, (channel_name, threshold_date))
        return [(row["date"], row["message_count"]) for row in rows]

    def get_top_channels(
        self, limit: int = 10, days_back: int = 30
    ) -> List[Channel]:
        """Get top channels by message count."""
        where_clause, params = self._build_base_conditions(days=days_back, include_content_check=True)
        
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
            WHERE {where_clause}
            GROUP BY channel_id, channel_name
            ORDER BY message_count DESC
            LIMIT ?
        """
        params.append(limit)

        rows = self.db_manager.execute_query(query, tuple(params))
        return [Channel.from_db_row(row) for row in rows]

    def save_channel_members_sync(self, messages_data: Dict[str, Any]) -> None:
        """Save channel members data to database (sync version)"""
        if not messages_data:
            return

        # Clear existing channel members data to avoid stale data
        self.db_manager.execute_query("DELETE FROM channel_members", fetch_all=False)
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

                self.db_manager.execute_many(query, values)
                logger.info(f"✅ Saved batch of {len(batch)} channel member records")

            except Exception as e:
                logger.error(f"Error saving channel members batch: {str(e)}")
                raise

    async def save_channel_members(self, messages_data: Dict[str, Any]) -> None:
        """Save channel members data to database"""
        if not messages_data:
            return

        # Clear existing channel members data to avoid stale data
        self.db_manager.execute_query("DELETE FROM channel_members", fetch_one=False, fetch_all=False)
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

                self.db_manager.execute_many(query, values)
                logger.info(f"✅ Saved batch of {len(batch)} channel member records")

            except Exception as e:
                logger.error(f"Error saving channel members batch: {str(e)}")
                raise

    def get_channel_statistics_by_limit(self, limit: int = 10) -> List[Channel]:
        """Get channel statistics ordered by message count."""
        where_clause, params = self._build_base_conditions()
        
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
            WHERE {where_clause}
            GROUP BY channel_id, channel_name
            ORDER BY message_count DESC
            LIMIT ?
        """
        params.append(limit)

        rows = self.db_manager.execute_query(query, tuple(params))
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

    def get_channel_list(self, base_filter: str = None) -> list:
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
        rows = self.db_manager.execute_query(query)
        return [row["channel_name"] for row in rows if row["channel_name"]]

    def get_channel_id_by_name(self, channel_name: str) -> Optional[str]:
        """Get channel ID by channel name."""
        query = "SELECT DISTINCT channel_id FROM messages WHERE channel_name = ? LIMIT 1"
        result = self.db_manager.execute_query(query, (channel_name,))
        return result[0]["channel_id"] if result else None

    def get_distinct_channel_count(self) -> int:
        """Get count of distinct channels."""
        where_clause, params = self._build_base_conditions()
        
        query = f"SELECT COUNT(DISTINCT channel_id) FROM messages WHERE {where_clause}"
        result = self.db_manager.execute_query(query, tuple(params), fetch_one=True)
        return result[0] if result else 0

    def get_channel_human_member_count(self, channel_name: str) -> int:
        """
        Get the total number of human members who have ever posted in a channel.
        
        Args:
            channel_name: Channel name to analyze
            
        Returns:
            Total number of unique human members
        """
        where_clause, params = self._build_base_conditions(
            channel_name=channel_name, 
            exclude_bots=True,
            include_content_check=True
        )
        
        query = f"""
        SELECT COUNT(DISTINCT author_name) as total_human_members
        FROM messages 
        WHERE {where_clause}
        """
        
        result = self.db_manager.execute_query(query, tuple(params), fetch_one=True)
        
        return result['total_human_members'] if result else 0

    def get_total_channel_count(self) -> int:
        """Get total count of channels."""
        query = f"SELECT COUNT(DISTINCT channel_name) FROM messages WHERE {self.base_filter}"
        result = self.db_manager.execute_query(query, fetch_one=True)
        return result["COUNT(DISTINCT channel_name)"] if result else 0

    def get_total_human_member_count(self) -> int:
        """Get total count of human members across all channels."""
        query = f"""
            SELECT COUNT(DISTINCT author_id) 
            FROM messages 
            WHERE {self.base_filter}
            AND (author_is_bot = 0 OR author_is_bot IS NULL)
        """
        result = self.db_manager.execute_query(query, fetch_one=True)
        return result["COUNT(DISTINCT author_id)"] if result else 0

    def get_all_channels_with_activity(self, days: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all channels with their message counts for a given time period."""
        where_clause, params = self._build_base_conditions(days=days, include_content_check=True)
        
        query = f"""
            SELECT 
                channel_name,
                COUNT(*) as message_count
            FROM messages 
            WHERE {where_clause}
            GROUP BY channel_name
            ORDER BY message_count DESC
        """
        
        results = self.db_manager.execute_query(query, tuple(params))
        
        return [
            {
                'channel_name': row['channel_name'],
                'message_count': row['message_count']
            }
            for row in results
        ] if results else []

    def get_channel_engagement_metrics(
        self, 
        channel_name: str, 
        days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get engagement metrics for a channel (replies, reactions, etc.).
        
        Args:
            channel_name: Channel name to analyze
            days: Number of days to look back
            
        Returns:
            Dictionary with engagement metrics
        """
        # Note: This is a simplified version since we don't have reply/reaction data in our schema
        # We'll estimate based on message patterns and threading
        
        query = f"""
        SELECT 
            COUNT(*) as total_messages,
            COUNT(CASE WHEN author_is_bot = 0 OR author_is_bot IS NULL THEN 1 END) as human_messages,
            COUNT(CASE WHEN (author_is_bot = 0 OR author_is_bot IS NULL) AND referenced_message_id IS NOT NULL THEN 1 END) as human_replies,
            COUNT(CASE WHEN (author_is_bot = 0 OR author_is_bot IS NULL) AND referenced_message_id IS NULL THEN 1 END) as human_original_posts,
            COUNT(CASE WHEN referenced_message_id IS NOT NULL THEN 1 END) as total_replies,
            COUNT(CASE WHEN referenced_message_id IS NULL THEN 1 END) as total_original_posts
        FROM messages 
        WHERE channel_name = ? AND {self.base_filter}
        """
        
        params = [channel_name]
        
        if days:
            query += " AND timestamp >= datetime('now', '-' || ? || ' days')"
            params.append(days)
        
        query += " AND content IS NOT NULL"
        
        result = self.db_manager.execute_query(query, tuple(params), fetch_one=True)
        
        if not result:
            return {}
        
        # Calculate engagement ratios
        human_messages = result['human_messages'] or 0
        human_replies = result['human_replies'] or 0
        human_original_posts = result['human_original_posts'] or 0
        
        human_replies_per_post = (
            human_replies / max(human_original_posts, 1) 
            if human_original_posts > 0 else 0
        )
        
        # Estimate reaction rate (simplified - we don't have actual reaction data)
        # Using a heuristic based on message engagement patterns
        estimated_reactions = min(human_messages * 0.1, human_messages)  # Assume 10% get reactions
        human_reaction_rate = estimated_reactions / max(human_messages, 1) if human_messages > 0 else 0
        
        return {
            'total_replies': result['total_replies'] or 0,
            'original_posts': result['total_original_posts'] or 0,
            'human_replies': human_replies,
            'human_original_posts': human_original_posts,
            'human_replies_per_post': round(human_replies_per_post, 2),
            'human_posts_with_reactions': int(estimated_reactions),
            'human_reaction_rate': round(human_reaction_rate * 100, 1),  # Convert to percentage
            'posts_with_reactions': int(estimated_reactions),
            'replies_per_post': round((result['total_replies'] or 0) / max(result['total_original_posts'] or 1, 1), 2),
            'reaction_rate': round(human_reaction_rate * 100, 1)
        }

    def get_channel_recent_activity(
        self, 
        channel_name: str, 
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Get recent daily activity for a channel.
        
        Args:
            channel_name: Channel name to analyze
            days: Number of recent days to get
            
        Returns:
            List of daily activity data
        """
        query = f"""
        SELECT 
            DATE(timestamp) as date,
            COUNT(*) as message_count,
            COUNT(DISTINCT author_name) as unique_users
        FROM messages 
        WHERE channel_name = ? AND {self.base_filter}
        AND timestamp >= datetime('now', '-' || ? || ' days')
        AND content IS NOT NULL
        GROUP BY DATE(timestamp)
        ORDER BY date DESC
        LIMIT ?
        """
        
        results = self.db_manager.execute_query(query, (channel_name, days, days))
        
        return [
            {
                'date': row['date'],
                'message_count': row['message_count'],
                'unique_users': row['unique_users']
            }
            for row in results
        ] if results else []

    def get_channel_weekly_breakdown(
        self, 
        channel_name: str, 
        days: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Get activity breakdown by day of week.
        
        Args:
            channel_name: Channel name to analyze
            days: Number of days to look back
            
        Returns:
            Dictionary with day-of-week activity counts
        """
        query = f"""
        SELECT 
            CASE strftime('%w', timestamp)
                WHEN '0' THEN 'sunday'
                WHEN '1' THEN 'monday'
                WHEN '2' THEN 'tuesday'
                WHEN '3' THEN 'wednesday'
                WHEN '4' THEN 'thursday'
                WHEN '5' THEN 'friday'
                WHEN '6' THEN 'saturday'
            END as day_name,
            COUNT(*) as message_count
        FROM messages 
        WHERE channel_name = ? AND {self.base_filter}
        AND (author_is_bot = 0 OR author_is_bot IS NULL)
        """
        
        params = [channel_name]
        
        if days:
            query += " AND timestamp >= datetime('now', '-' || ? || ' days')"
            params.append(days)
        
        query += """
        AND content IS NOT NULL
        GROUP BY strftime('%w', timestamp)
        ORDER BY strftime('%w', timestamp)
        """
        
        results = self.db_manager.execute_query(query, tuple(params))
        
        # Initialize all days to 0
        weekly_breakdown = {
            'monday': 0,
            'tuesday': 0,
            'wednesday': 0,
            'thursday': 0,
            'friday': 0,
            'saturday': 0,
            'sunday': 0
        }
        
        # Fill in actual data
        for row in results:
            if row['day_name']:
                weekly_breakdown[row['day_name']] = row['message_count']
        
        return weekly_breakdown

    def get_channel_health_metrics(
        self, 
        channel_name: str
    ) -> Dict[str, Any]:
        """
        Get comprehensive health metrics for a channel.
        
        Args:
            channel_name: Channel name to analyze
            
        Returns:
            Dictionary with health metrics
        """
        # Get recent activity (last 7 days)
        recent_query = f"""
        SELECT COUNT(DISTINCT author_name) as weekly_active_humans
        FROM messages 
        WHERE channel_name = ?
        AND timestamp >= datetime('now', '-7 days')
        AND (author_is_bot = 0 OR author_is_bot IS NULL)
        AND content IS NOT NULL
        """
        
        recent_result = self.db_manager.execute_query(recent_query, (channel_name,), fetch_one=True)
        weekly_active = recent_result['weekly_active_humans'] if recent_result else 0
        
        # Get all-time human contributors
        all_time_query = f"""
        SELECT COUNT(DISTINCT author_name) as total_human_contributors
        FROM messages 
        WHERE channel_name = ?
        AND (author_is_bot = 0 OR author_is_bot IS NULL)
        AND content IS NOT NULL
        """
        
        all_time_result = self.db_manager.execute_query(all_time_query, (channel_name,), fetch_one=True)
        total_human_contributors = all_time_result['total_human_contributors'] if all_time_result else 0
        
        # Calculate inactive users (posted before but not in last 7 days)
        inactive_humans = max(0, total_human_contributors - weekly_active)
        
        # Estimate total channel members (this would ideally come from Discord API)
        # For now, we'll estimate based on message history and assume some lurkers
        estimated_total_members = max(total_human_contributors * 1.4, total_human_contributors + 5)  # Rough estimate
        human_lurkers = max(0, int(estimated_total_members) - total_human_contributors)
        
        # Calculate participation rate
        human_participation_rate = (
            total_human_contributors / max(estimated_total_members, 1) * 100
            if estimated_total_members > 0 else 0
        )
        
        return {
            'weekly_active': weekly_active,
            'inactive_users': inactive_humans,
            'total_channel_members': int(estimated_total_members),
            'lurkers': human_lurkers,
            'participation_rate': round(human_participation_rate, 1),
            'human_members_who_posted': total_human_contributors,
            'recently_inactive_humans': inactive_humans,
            'human_lurkers': human_lurkers,
            'human_participation_rate': round(human_participation_rate, 1)
        }
