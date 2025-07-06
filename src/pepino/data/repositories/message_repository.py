"""
Message repository for data access operations.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ...config import Settings
from ..database.manager import DatabaseManager
from ..models.message import Message
from ...logging_config import get_logger

logger = get_logger(__name__)


class MessageRepository:
    """Repository for message data access."""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        # Use unified settings for base filter
        self.base_filter = Settings().base_filter.strip()

    def get_messages_by_channel(
        self,
        channel_name: str,
        limit: Optional[int] = None,
        min_length: Optional[int] = None,
    ) -> List[Message]:
        """Get messages from a specific channel."""
        query = f"""
            SELECT * FROM messages 
            WHERE channel_name = ? AND {self.base_filter}
            AND content IS NOT NULL
        """
        params = [channel_name]

        if min_length:
            query += " AND LENGTH(content) > ?"
            params.append(min_length)

        query += " ORDER BY timestamp DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        rows = self.db_manager.execute_query(query, tuple(params))
        return [Message.from_db_row(row) for row in rows]

    def get_messages_by_user(
        self,
        author_id: str,
        limit: Optional[int] = None,
        min_length: Optional[int] = None,
    ) -> List[Message]:
        """Get messages from a specific user."""
        query = f"""
            SELECT * FROM messages 
            WHERE author_id = ? AND {self.base_filter}
            AND content IS NOT NULL
        """
        params = [author_id]

        if min_length:
            query += " AND LENGTH(content) > ?"
            params.append(min_length)

        query += " ORDER BY timestamp DESC"

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        rows = self.db_manager.execute_query(query, tuple(params))
        return [Message.from_db_row(row) for row in rows]

    def get_messages_by_date_range(
        self, channel_name: Optional[str], start_date: datetime, end_date: datetime, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get messages within a date range for a specific channel or all channels if channel_name is None."""
        query = f"""
            SELECT id as message_id, content, author_name, author_display_name, channel_name, timestamp, author_is_bot
            FROM messages 
            WHERE timestamp BETWEEN ? AND ? 
            AND {self.base_filter}
            AND content IS NOT NULL AND content != ''
        """
        params = [start_date.isoformat(), end_date.isoformat()]
        
        if channel_name:
            query += " AND channel_name = ?"
            params.append(channel_name)

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        rows = self.db_manager.execute_query(query, tuple(params))
        return [
            {
                'id': row['message_id'],
                'content': row['content'],
                'author_name': row['author_name'],
                'author_display_name': row['author_display_name'],
                'channel_name': row['channel_name'],
                'timestamp': row['timestamp'],
                'author_is_bot': row['author_is_bot'],
            }
            for row in rows
        ]

    def get_message_statistics(
        self, channel_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get message statistics."""
        base_query = f"""
            SELECT 
                COUNT(*) as total_messages,
                COUNT(DISTINCT author_id) as unique_users,
                COUNT(CASE WHEN author_is_bot = 1 THEN 1 END) as bot_messages,
                COUNT(CASE WHEN author_is_bot = 0 OR author_is_bot IS NULL THEN 1 END) as human_messages,
                AVG(LENGTH(content)) as avg_message_length,
                MIN(timestamp) as first_message,
                MAX(timestamp) as last_message,
                COUNT(DISTINCT DATE(timestamp)) as active_days
            FROM messages 
            WHERE {self.base_filter}
            AND content IS NOT NULL
        """

        if channel_name:
            base_query += " AND channel_name = ?"
            row = self.db_manager.execute_single(base_query, (channel_name,))
        else:
            row = self.db_manager.execute_single(base_query)

        if not row:
            return {}

        return {
            "total_messages": row[0],
            "unique_users": row[1],
            "bot_messages": row[2],
            "human_messages": row[3],
            "avg_message_length": row[4],
            "first_message": row[5],
            "last_message": row[6],
            "active_days": row[7],
        }

    def get_hourly_activity(self, days: int = 30) -> Dict[int, int]:
        """Get hourly activity distribution."""
        query = f"""
            SELECT 
                CAST(strftime('%H', timestamp) AS INTEGER) as hour,
                COUNT(*) as message_count
            FROM messages
            WHERE {self.base_filter}
            AND timestamp >= datetime('now', '-{days} days')
            AND timestamp IS NOT NULL
            GROUP BY hour
            ORDER BY hour
        """

        rows = self.db_manager.execute_query(query)
        return {row["hour"]: row["message_count"] for row in rows}

    def get_daily_activity(self, days: int = 30) -> List[tuple]:
        """Get daily activity for the specified number of days."""
        query = f"""
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as message_count
            FROM messages
            WHERE {self.base_filter}
            AND timestamp >= datetime('now', '-{days} days')
            AND timestamp IS NOT NULL
            GROUP BY DATE(timestamp)
            ORDER BY date
        """

        return self.db_manager.execute_query(query)

    def get_recent_messages(
        self, hours: int = 24, base_filter: str = None
    ) -> List[Message]:
        """Get recent messages within the specified hours."""
        filter_clause = base_filter or self.base_filter
        query = f"""
            SELECT * FROM messages 
            WHERE timestamp >= datetime('now', '-{hours} hours')
            AND {filter_clause}
            AND content IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 1000
        """

        rows = self.db_manager.execute_query(query)
        return [Message.from_db_row(row) for row in rows]

    def get_messages_for_sentiment_analysis(
        self, channel_name: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get messages for sentiment analysis with simplified fields."""
        query = f"""
            SELECT id, content, author_name, channel_name, timestamp
            FROM messages 
            WHERE content IS NOT NULL AND content != '' AND {self.base_filter}
        """
        params = []

        if channel_name:
            query += " AND channel_name = ?"
            params.append(channel_name)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self.db_manager.execute_query(query, tuple(params))
        return [
            {
                "id": row["id"],
                "content": row["content"],
                "author_name": row["author_name"],
                "channel_name": row["channel_name"],
                "timestamp": row["timestamp"],
            }
            for row in rows
        ]

    def get_messages_for_topic_analysis(
        self, channel_name: Optional[str] = None, days_back: Optional[int] = None, limit: int = 1000
    ) -> List[str]:
        """Get message content for topic analysis."""
        from datetime import datetime, timedelta

        if channel_name:
            query = f"""
                SELECT content
                FROM messages
                WHERE channel_name = ? AND {self.base_filter}
                AND content IS NOT NULL
                AND LENGTH(content) > 10
            """
            params = [channel_name]
        else:
            query = f"""
                SELECT content
                FROM messages
                WHERE {self.base_filter}
                AND content IS NOT NULL
                AND LENGTH(content) > 10
            """
            params = []

        if days_back is not None:
            threshold_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            query += " AND timestamp >= ?"
            params.append(threshold_date)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        results = self.db_manager.execute_query(query, tuple(params))
        return [row["content"] for row in results if row["content"]]

    def get_temporal_analysis_data(
        self,
        channel_name: Optional[str] = None,
        user_name: Optional[str] = None,
        days_back: Optional[int] = None,
        granularity: str = "day",
    ) -> List[Dict[str, Any]]:
        """Get temporal data for analysis with complete time series."""
        from datetime import datetime, timedelta

        # Handle None days_back (all time)
        if days_back is not None:
            threshold_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        else:
            threshold_date = None

        # Build query based on granularity
        if granularity == "hour":
            date_format = "strftime('%Y-%m-%d %H:00:00', timestamp)"
            group_by = "strftime('%Y-%m-%d %H', timestamp)"
            # For hourly, we'll use a simpler approach since generating all hours is complex
            use_date_series = False
        elif granularity == "week":
            date_format = "strftime('%Y-W%W', timestamp)"
            group_by = "strftime('%Y-W%W', timestamp)"
            use_date_series = False
        else:  # day
            date_format = "DATE(timestamp)"
            group_by = "DATE(timestamp)"
            use_date_series = True

        if use_date_series:
            # Generate complete date series for daily granularity
            if threshold_date is not None:
                query = f"""
                    WITH RECURSIVE date_series AS (
                        SELECT DATE(?) as period
                        UNION ALL
                        SELECT DATE(period, '+1 day')
                        FROM date_series
                        WHERE period < DATE('now', '-1 day')
                    ),
                    message_counts AS (
                        SELECT 
                            {date_format} as period,
                            COUNT(*) as message_count,
                            COUNT(DISTINCT author_id) as unique_users,
                            AVG(LENGTH(content)) as avg_message_length
                        FROM messages
                        WHERE {self.base_filter} AND timestamp >= ? AND content IS NOT NULL
                        {f"AND channel_name = ?" if channel_name else ""}
                        {f"AND (author_name LIKE ? OR author_display_name LIKE ?)" if user_name else ""}
                        GROUP BY {group_by}
                    )
                    SELECT 
                        ds.period,
                        COALESCE(mc.message_count, 0) as message_count,
                        COALESCE(mc.unique_users, 0) as unique_users,
                        COALESCE(mc.avg_message_length, 0.0) as avg_message_length
                    FROM date_series ds
                    LEFT JOIN message_counts mc ON ds.period = mc.period
                    ORDER BY ds.period
                """
                
                params = [threshold_date, threshold_date]
                if channel_name:
                    params.append(channel_name)
                if user_name:
                    params.extend([f"%{user_name}%", f"%{user_name}%"])
            else:
                # All time query without date series
                query = f"""
                    SELECT 
                        {date_format} as period,
                        COUNT(*) as message_count,
                        COUNT(DISTINCT author_id) as unique_users,
                        AVG(LENGTH(content)) as avg_message_length
                    FROM messages
                    WHERE {self.base_filter} AND content IS NOT NULL
                    {f"AND channel_name = ?" if channel_name else ""}
                    {f"AND (author_name LIKE ? OR author_display_name LIKE ?)" if user_name else ""}
                    GROUP BY {group_by}
                    ORDER BY period
                """
                
                params = []
                if channel_name:
                    params.append(channel_name)
                if user_name:
                    params.extend([f"%{user_name}%", f"%{user_name}%"])
        else:
            # For hourly/weekly, use the original approach
            conditions = [self.base_filter, "content IS NOT NULL"]
            params = []

            if threshold_date is not None:
                conditions.append("timestamp >= ?")
                params.append(threshold_date)

            if channel_name:
                conditions.append("channel_name = ?")
                params.append(channel_name)

            if user_name:
                conditions.append("(author_name LIKE ? OR author_display_name LIKE ?)")
                params.extend([f"%{user_name}%", f"%{user_name}%"])

            where_clause = " AND ".join(conditions)

            query = f"""
                SELECT 
                    {date_format} as period,
                    COUNT(*) as message_count,
                    COUNT(DISTINCT author_id) as unique_users,
                    AVG(LENGTH(content)) as avg_message_length
                FROM messages
                WHERE {where_clause}
                GROUP BY {group_by}
                ORDER BY period
            """

        results = self.db_manager.execute_query(query, tuple(params))

        return [
            {
                "period": row["period"],
                "message_count": row["message_count"],
                "unique_users": row["unique_users"],
                "avg_message_length": float(row["avg_message_length"]) if row["avg_message_length"] else 0.0,
            }
            for row in results
        ]

    def get_conversation_threads(
        self, channel_id: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get conversation threads data."""
        channel_filter = f"AND channel_id = '{channel_id}'" if channel_id else ""

        query = f"""
            SELECT 
                thread_id,
                thread_name,
                COUNT(*) as message_count,
                COUNT(DISTINCT author_id) as unique_participants,
                MIN(timestamp) as start_time,
                MAX(timestamp) as end_time,
                AVG(LENGTH(content)) as avg_message_length
            FROM messages 
            WHERE thread_id IS NOT NULL {channel_filter}
            GROUP BY thread_id
            ORDER BY message_count DESC
            LIMIT ?
        """

        results = self.db_manager.execute_query(query, (limit,))

        return [
            {
                "thread_id": row["thread_id"],
                "thread_name": row["thread_name"],
                "message_count": row["message_count"],
                "unique_participants": row["unique_participants"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "avg_message_length": float(row["avg_message_length"]) if row["avg_message_length"] else 0.0,
            }
            for row in results
        ]

    def get_reply_chains_data(
        self, channel_id: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get reply chains data."""
        channel_filter = f"AND channel_id = '{channel_id}'" if channel_id else ""

        query = f"""
            SELECT 
                referenced_message_id,
                COUNT(*) as reply_count,
                COUNT(DISTINCT author_id) as unique_repliers
            FROM messages 
            WHERE referenced_message_id IS NOT NULL {channel_filter}
            GROUP BY referenced_message_id
            ORDER BY reply_count DESC
            LIMIT ?
        """

        results = self.db_manager.execute_query(query, (limit,))

        return [
            {"message_id": row["referenced_message_id"], "reply_count": row["reply_count"], "unique_repliers": row["unique_repliers"]}
            for row in results
        ]

    def get_message_by_id(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get a single message by ID."""
        query = """
            SELECT id, content, author_name, timestamp, referenced_message_id
            FROM messages 
            WHERE id = ?
        """

        row = self.db_manager.execute_query(query, (message_id,), fetch_one=True)
        if row:
            return {
                "message_id": row["id"],
                "content": row["content"],
                "author": row["author_name"],
                "timestamp": row["timestamp"],
                "referenced_message_id": row["referenced_message_id"],
            }
        return None

    def get_replies_to_message(
        self, message_id: str, limit: int = 1
    ) -> List[Dict[str, Any]]:
        """Get replies to a specific message."""
        query = """
            SELECT id, content, author_name, timestamp
            FROM messages 
            WHERE referenced_message_id = ?
            ORDER BY timestamp
            LIMIT ?
        """

        results = self.db_manager.execute_query(query, (message_id, limit))
        return [
            {
                "message_id": row["id"],
                "content": row["content"],
                "author": row["author_name"],
                "timestamp": row["timestamp"],
            }
            for row in results
        ]

    def get_engagement_data(
        self, channel_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get engagement data for messages."""
        channel_filter = f"AND channel_id = '{channel_id}'" if channel_id else ""

        query = f"""
            SELECT 
                id,
                content,
                author_name,
                timestamp,
                (SELECT COUNT(*) FROM messages m2 WHERE m2.referenced_message_id = m.id) as reply_count,
                (SELECT COUNT(*) > 0 FROM messages m3 WHERE m3.id = m.id AND m3.reactions IS NOT NULL) as has_reactions
            FROM messages m
            WHERE content IS NOT NULL {channel_filter}
            ORDER BY timestamp DESC
            LIMIT ?
        """

        results = self.db_manager.execute_query(query, (limit,))
        return [
            {
                "message_id": row["id"],
                "content": row["content"],
                "author": row["author_name"],
                "timestamp": row["timestamp"],
                "reply_count": row["reply_count"],
                "has_reactions": bool(row["has_reactions"]),
            }
            for row in results
        ]

    def get_conversation_flow_data(
        self, channel_id: str, time_window_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get conversation flow data for a specific time window."""
        from datetime import datetime, timedelta

        start_time = (datetime.now() - timedelta(hours=time_window_hours)).isoformat()

        query = """
            SELECT 
                id,
                content,
                author_name,
                timestamp,
                referenced_message_id
            FROM messages 
            WHERE channel_id = ? AND timestamp >= ?
            ORDER BY timestamp
        """

        results = self.db_manager.execute_query(query, (channel_id, start_time))
        return [
            {
                "message_id": row["id"],
                "content": row["content"],
                "author": row["author_name"],
                "timestamp": row["timestamp"],
                "referenced_message_id": row["referenced_message_id"],
            }
            for row in results
        ]

    def get_engagement_metrics(
        self, channel_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get overall engagement metrics."""
        channel_filter = f"AND channel_id = '{channel_id}'" if channel_id else ""

        # Get basic engagement stats
        query = f"""
            SELECT 
                COUNT(*) as total_messages,
                COUNT(CASE WHEN referenced_message_id IS NOT NULL THEN 1 END) as total_replies,
                COUNT(CASE WHEN referenced_message_id IS NULL THEN 1 END) as original_posts,
                COUNT(CASE WHEN reactions IS NOT NULL THEN 1 END) as posts_with_reactions
            FROM messages 
            WHERE content IS NOT NULL {channel_filter}
        """

        row = self.db_manager.execute_query(query, fetch_one=True)
        
        if not row:
            return {}

        total_messages = row["total_messages"]
        total_replies = row["total_replies"]
        original_posts = row["original_posts"]
        posts_with_reactions = row["posts_with_reactions"]

        return {
            "total_messages": total_messages,
            "total_replies": total_replies,
            "original_posts": original_posts,
            "posts_with_reactions": posts_with_reactions,
            "replies_per_post": total_replies / original_posts if original_posts > 0 else 0,
            "reaction_rate": (posts_with_reactions / total_messages * 100) if total_messages > 0 else 0,
        }

    def load_existing_data(self) -> Dict[str, Any]:
        """Load existing message data from database grouped by guild and channel"""
        query = """
            SELECT guild_name, channel_name, id, timestamp 
            FROM messages 
            ORDER BY guild_name, channel_name, timestamp
        """

        rows = self.db_manager.execute_query(query)

        data = {}
        for row in rows:
            # rows is a list of dicts, not tuples
            guild_name = row["guild_name"]
            channel_name = row["channel_name"]
            msg_id = row["id"]
            timestamp = row["timestamp"]

            if guild_name not in data:
                data[guild_name] = {}
            if channel_name not in data[guild_name]:
                data[guild_name][channel_name] = []
            data[guild_name][channel_name].append(
                {"id": msg_id, "timestamp": timestamp}
            )

        return data

    def insert_message(self, message_data: Dict[str, Any]) -> bool:
        """Insert a single message into the database using the comprehensive schema."""
        # Map message_data to the full column list (assume a helper _map_to_full_schema exists)
        columns, values = self._map_to_full_schema(message_data)
        placeholders = ', '.join(['?'] * len(columns))
        query = f"INSERT OR REPLACE INTO messages ({', '.join(columns)}) VALUES ({placeholders})"
        try:
            self.db_manager.execute_query(query, tuple(values), fetch_all=False)
            return True
        except Exception as e:
            logger.error(f"Failed to insert message: {e}")
            return False

    def insert_messages_batch(self, messages: List[Dict[str, Any]]) -> int:
        """Insert multiple messages in a single transaction using the comprehensive schema."""
        if not messages:
            return 0
        # Map all messages to the full column list
        columns, values_list = None, []
        for msg in messages:
            cols, vals = self._map_to_full_schema(msg)
            if columns is None:
                columns = cols
            values_list.append(tuple(vals))
        placeholders = ', '.join(['?'] * len(columns))
        query = f"INSERT OR REPLACE INTO messages ({', '.join(columns)}) VALUES ({placeholders})"
        try:
            return self.db_manager.execute_many(query, values_list)
        except Exception as e:
            logger.error(f"Failed to insert message batch: {e}")
            return 0

    def _map_to_full_schema(self, message_data: Dict[str, Any]):
        """Map message_data dict to the full 95-column schema. Returns (columns, values)."""
        # For brevity, only a subset of columns is shown. Expand as needed for full schema.
        columns = [
            "id", "content", "timestamp", "edited_timestamp", "jump_url",
            "author_id", "author_name", "author_display_name", "author_is_bot",
            "channel_id", "channel_name", "channel_type",
            "guild_id", "guild_name",
            "mentions", "mention_everyone", "mention_roles",
            "referenced_message_id", "attachments", "embeds", "reactions",
            "emoji_stats", "pinned", "flags", "nonce", "type", "is_system",
            "mentions_everyone", "has_reactions"
            # ... add all other columns as needed for the full schema ...
        ]
        values = [
            message_data.get("id", message_data.get("message_id", "")),
            message_data.get("content", ""),
            message_data.get("timestamp", ""),
            message_data.get("edited_timestamp"),
            message_data.get("jump_url", ""),
            message_data.get("author_id", ""),
            message_data.get("author_name", ""),
            message_data.get("author_display_name", ""),
            message_data.get("author_is_bot", False),
            message_data.get("channel_id", ""),
            message_data.get("channel_name", ""),
            message_data.get("channel_type", ""),
            message_data.get("guild_id", ""),
            message_data.get("guild_name", ""),
            message_data.get("mentions", json.dumps([])),
            message_data.get("mention_everyone", False),
            message_data.get("mention_roles", json.dumps([])),
            message_data.get("referenced_message_id"),
            message_data.get("attachments", json.dumps([])),
            message_data.get("embeds", json.dumps([])),
            message_data.get("reactions", json.dumps([])),
            message_data.get("emoji_stats", json.dumps({})),
            message_data.get("pinned", False),
            message_data.get("flags"),
            message_data.get("nonce"),
            message_data.get("type", ""),
            message_data.get("is_system", False),
            message_data.get("mentions_everyone", False),
            message_data.get("has_reactions", False),
            # ... add all other values as needed for the full schema ...
        ]
        return columns, values

    def clear_all_messages(self) -> None:
        """Clear all messages from the database"""
        try:
            self.db_manager.execute_query("DELETE FROM messages", fetch_all=False)
            self.db_manager.execute_query(
                'DELETE FROM sqlite_sequence WHERE name="messages"', fetch_all=False
            )
            logger.info("✅ Database cleared")
        except Exception as e:
            logger.error(f"❌ Error clearing database: {e}")
            raise

    def get_user_activity_data(
        self, author_id: str, days_back: int = 30
    ) -> List[tuple]:
        """Get daily activity data for a user."""
        thirty_days_ago = (datetime.now() - timedelta(days=days_back)).isoformat()

        query = f"""
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as message_count
            FROM messages
            WHERE author_id = ?
            AND {self.base_filter}
            AND timestamp >= ?
            AND content IS NOT NULL
            GROUP BY DATE(timestamp)
            ORDER BY date
        """

        rows = self.db_manager.execute_query(query, (author_id, thirty_days_ago))
        return [(row["date"], row["message_count"]) for row in rows]

    def get_channel_top_users_data(
        self, channel_name: str, limit: int = 8
    ) -> List[tuple]:
        """Get top users data for a channel."""
        query = f"""
            SELECT 
                COALESCE(author_display_name, author_name) as name,
                COUNT(*) as message_count
            FROM messages
            WHERE channel_name = ?
            AND {self.base_filter}
            AND content IS NOT NULL
            GROUP BY author_id, author_display_name, author_name
            ORDER BY message_count DESC
            LIMIT ?
        """

        rows = self.db_manager.execute_query(query, (channel_name, limit))
        return [(row["name"], row["message_count"]) for row in rows]

    def get_total_message_count(self) -> int:
        """Get total count of messages."""
        query = f"SELECT COUNT(*) FROM messages WHERE {self.base_filter}"
        result = self.db_manager.execute_query(query, fetch_one=True)
        return result[0] if result else 0

    def get_distinct_user_count(self) -> int:
        """Get count of distinct users."""
        query = f"SELECT COUNT(DISTINCT author_id) FROM messages WHERE {self.base_filter}"
        result = self.db_manager.execute_query(query, fetch_one=True)
        return result[0] if result else 0

    def get_user_messages(self, user_name: str, days_back: Optional[int] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get messages from a specific user."""
        from datetime import datetime, timedelta
        
        query = f"""
            SELECT id as message_id, content, author_name, channel_name, timestamp
            FROM messages 
            WHERE (author_name LIKE ? OR author_display_name LIKE ?)
            AND content IS NOT NULL AND content != ''
            AND {self.base_filter}
        """
        
        params = [f"%{user_name}%", f"%{user_name}%"]
        
        if days_back is not None:
            threshold_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            query += " AND timestamp >= ?"
            params.append(threshold_date)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self.db_manager.execute_query(query, tuple(params))
        
        return [
            {
                "message_id": row["message_id"],
                "content": row["content"],
                "author_name": row["author_name"],
                "channel_name": row["channel_name"],
                "timestamp": row["timestamp"],
            }
            for row in rows
        ]

    def get_channel_messages(self, channel_name: str, days_back: Optional[int] = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get messages from a specific channel."""
        from datetime import datetime, timedelta
        
        query = f"""
            SELECT id as message_id, content, author_name, channel_name, timestamp
            FROM messages 
            WHERE channel_name = ?
            AND content IS NOT NULL AND content != ''
            AND {self.base_filter}
        """
        
        params = [channel_name]
        
        if days_back is not None:
            threshold_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            query += " AND timestamp >= ?"
            params.append(threshold_date)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self.db_manager.execute_query(query, tuple(params))
        
        return [
            {
                "message_id": row["message_id"],
                "content": row["content"],
                "author_name": row["author_name"],
                "channel_name": row["channel_name"],
                "timestamp": row["timestamp"],
            }
            for row in rows
        ]

    def get_recent_messages(self, limit: int = 1000, days_back: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get recent messages."""
        from datetime import datetime, timedelta
        
        query = f"""
            SELECT id as message_id, content, author_name, channel_name, timestamp
            FROM messages 
            WHERE content IS NOT NULL AND content != ''
            AND {self.base_filter}
        """
        
        params = []
        
        if days_back is not None:
            threshold_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            query += " AND timestamp >= ?"
            params.append(threshold_date)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self.db_manager.execute_query(query, tuple(params))
        
        return [
            {
                "message_id": row["message_id"],
                "content": row["content"],
                "author_name": row["author_name"],
                "channel_name": row["channel_name"],
                "timestamp": row["timestamp"],
            }
            for row in rows
        ]

    def get_messages_without_embeddings(
        self, channel_filter: Optional[str] = None, max_messages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get messages that don't have embeddings yet."""
        query = f"""
            SELECT DISTINCT m.id as message_id, m.content
            FROM messages m
            LEFT JOIN embeddings e ON m.id = e.message_id
            WHERE e.message_id IS NULL 
            AND m.content IS NOT NULL 
            AND LENGTH(m.content) > 10
            AND {self.base_filter}
        """
        
        params = []
        if channel_filter:
            query += " AND m.channel_name = ?"
            params.append(channel_filter)
            
        query += " ORDER BY m.timestamp DESC"
        
        if max_messages:
            query += " LIMIT ?"
            params.append(max_messages)
            
        rows = self.db_manager.execute_query(query, params)
        return [
            {
                "message_id": row["message_id"],
                "content": row["content"],
            }
            for row in rows
        ]

    def get_all_channels(self) -> List[str]:
        """Get all unique channel names from the database."""
        query = f"""
            SELECT DISTINCT channel_name 
            FROM messages 
            WHERE {self.base_filter}
            AND channel_name IS NOT NULL
            ORDER BY channel_name
        """
        
        rows = self.db_manager.execute_query(query)
        return [row["channel_name"] for row in rows]

    def get_top_commented_messages(self, channel_name: Optional[str], start_date: datetime, end_date: datetime, top_n: int = 3) -> List[Dict[str, Any]]:
        """Get the top N most replied-to messages in a channel and date range, or globally if channel_name is None."""
        # Create a base filter that explicitly references the main table
        base_filter_clean = self.base_filter.replace('author_id', 'm.author_id').replace('author_name', 'm.author_name').replace('channel_name', 'm.channel_name')
        
        query = f'''
            SELECT m.id as message_id, m.content, COALESCE(m.author_display_name, m.author_name) as author, COUNT(r.id) as reply_count, m.timestamp, m.jump_url
            FROM messages m
            LEFT JOIN messages r ON r.referenced_message_id = m.id
            WHERE m.timestamp BETWEEN ? AND ?
              AND {base_filter_clean}
              AND m.content IS NOT NULL
        '''
        params = [start_date.isoformat(), end_date.isoformat()]
        if channel_name:
            query += " AND m.channel_name = ?"
            params.append(channel_name)
        query += " GROUP BY m.id, m.content, m.author_display_name, m.author_name, m.timestamp, m.jump_url"
        query += " ORDER BY reply_count DESC, m.timestamp DESC"
        query += " LIMIT ?"
        params.append(top_n)
        rows = self.db_manager.execute_query(query, tuple(params))
        return [
            {
                'message_id': row['message_id'],
                'content': row['content'],
                'author': row['author'],
                'reply_count': row['reply_count'],
                'timestamp': row['timestamp'],
                'jump_url': row['jump_url']
            }
            for row in rows if row['reply_count'] > 0
        ]
