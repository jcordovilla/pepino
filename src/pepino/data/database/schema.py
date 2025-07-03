"""
Centralized database schema for Discord message analysis.
This is the single source of truth for all database operations.
"""

import logging
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)


def init_database(db_path: str = "data/discord_messages.db") -> None:
    """Initialize the SQLite database with the complete schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create messages table with comprehensive Discord data
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            content TEXT,
            timestamp TEXT,
            edited_timestamp TEXT,
            jump_url TEXT,
            
            -- Author information
            author_id TEXT,
            author_name TEXT,
            author_discriminator TEXT,
            author_display_name TEXT,
            author_is_bot BOOLEAN,
            author_avatar_url TEXT,
            author_accent_color INTEGER,
            author_banner_url TEXT,
            author_color INTEGER,
            author_created_at TEXT,
            author_default_avatar_url TEXT,
            author_public_flags INTEGER,
            author_system BOOLEAN,
            author_verified BOOLEAN,
            
            -- User presence
            author_status TEXT,
            author_activity TEXT,
            author_desktop_status TEXT,
            author_mobile_status TEXT,
            author_web_status TEXT,
            
            -- Guild information
            guild_id TEXT,
            guild_name TEXT,
            guild_member_count INTEGER,
            guild_description TEXT,
            guild_icon_url TEXT,
            guild_banner_url TEXT,
            guild_splash_url TEXT,
            guild_discovery_splash_url TEXT,
            guild_features TEXT,
            guild_verification_level TEXT,
            guild_explicit_content_filter TEXT,
            guild_mfa_level TEXT,
            guild_premium_tier TEXT,
            guild_premium_subscription_count INTEGER,
            
            -- Channel information
            channel_id TEXT,
            channel_name TEXT,
            channel_type TEXT,
            channel_topic TEXT,
            channel_nsfw BOOLEAN,
            channel_position INTEGER,
            channel_slowmode_delay INTEGER,
            channel_category_id TEXT,
            channel_overwrites TEXT,
            
            -- Thread information
            thread_id TEXT,
            thread_name TEXT,
            thread_archived BOOLEAN,
            thread_auto_archive_duration INTEGER,
            thread_locked BOOLEAN,
            thread_member_count INTEGER,
            thread_message_count INTEGER,
            thread_owner_id TEXT,
            thread_parent_id TEXT,
            thread_slowmode_delay INTEGER,
            
            -- Mentions and references
            mentions TEXT,
            mention_everyone BOOLEAN,
            mention_roles TEXT,
            mention_channels TEXT,
            referenced_message_id TEXT,
            referenced_message TEXT,
            
            -- Attachments and embeds
            attachments TEXT,
            embeds TEXT,
            
            -- Reactions
            reactions TEXT,
            emoji_stats TEXT,
            
            -- Message metadata
            pinned BOOLEAN,
            flags INTEGER,
            nonce TEXT,
            type TEXT,
            is_system BOOLEAN,
            mentions_everyone BOOLEAN,
            has_reactions BOOLEAN,
            
            -- Components and interactions
            components TEXT,
            interaction TEXT,
            
            -- Stickers and role subscriptions
            stickers TEXT,
            role_subscription_data TEXT,
            
            -- Application information
            application_id TEXT,
            application TEXT,
            
            -- Activity information
            activity TEXT,
            
            -- Additional metadata
            position INTEGER,
            role_subscription_listing_id TEXT,
            webhook_id TEXT,
            tts BOOLEAN,
            suppress_embeds BOOLEAN,
            allowed_mentions TEXT,
            message_reference TEXT,
            
            -- Analysis-friendly fields (computed)
            has_attachments BOOLEAN DEFAULT FALSE,
            has_embeds BOOLEAN DEFAULT FALSE,
            has_stickers BOOLEAN DEFAULT FALSE,
            has_mentions BOOLEAN DEFAULT FALSE,
            has_reference BOOLEAN DEFAULT FALSE,
            is_webhook BOOLEAN DEFAULT FALSE,
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Create sync logs table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sync_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            guilds_synced TEXT,
            channels_skipped TEXT,
            errors TEXT,
            total_messages_synced INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Create analysis tables
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS message_embeddings (
            message_id TEXT PRIMARY KEY,
            embedding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (message_id) REFERENCES messages(id)
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS message_topics (
            message_id TEXT,
            topic_id INTEGER,
            confidence FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (message_id, topic_id),
            FOREIGN KEY (message_id) REFERENCES messages(id)
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS word_frequencies (
            word TEXT,
            channel_id TEXT,
            frequency INTEGER,
            last_updated TIMESTAMP,
            PRIMARY KEY (word, channel_id),
            FOREIGN KEY (channel_id) REFERENCES messages(channel_id)
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS user_statistics (
            user_id TEXT,
            channel_id TEXT,
            message_count INTEGER,
            avg_message_length FLOAT,
            active_hours TEXT,
            last_active TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, channel_id),
            FOREIGN KEY (user_id) REFERENCES messages(author_id),
            FOREIGN KEY (channel_id) REFERENCES messages(channel_id)
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_chains (
            chain_id INTEGER PRIMARY KEY,
            root_message_id TEXT,
            last_message_id TEXT,
            message_count INTEGER,
            created_at TIMESTAMP,
            FOREIGN KEY (root_message_id) REFERENCES messages(id),
            FOREIGN KEY (last_message_id) REFERENCES messages(id)
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS message_temporal_stats (
            channel_id TEXT,
            date TEXT,
            message_count INTEGER,
            active_users INTEGER,
            avg_message_length FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (channel_id, date),
            FOREIGN KEY (channel_id) REFERENCES messages(channel_id)
        )
    """
    )

    # Create channel members table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS channel_members (
            channel_id TEXT,
            channel_name TEXT,
            guild_id TEXT,
            guild_name TEXT,
            user_id TEXT,
            user_name TEXT,
            user_display_name TEXT,
            user_joined_at TEXT,
            user_roles TEXT,
            is_bot BOOLEAN,
            member_permissions TEXT,
            synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (channel_id, user_id)
        )
    """
    )

    # Create virtual table for full-text search
    cursor.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            content,
            author_name,
            channel_name,
            guild_name,
            content='messages',
            content_rowid='id'
        )
    """
    )

    # Create indexes for performance
    _create_indexes(cursor)

    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {db_path}")


def _create_indexes(cursor: sqlite3.Cursor) -> None:
    """Create database indexes for performance."""
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_message_timestamp ON messages(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_guild_channel ON messages(guild_name, channel_name)",
        "CREATE INDEX IF NOT EXISTS idx_author ON messages(author_id)",
        "CREATE INDEX IF NOT EXISTS idx_message_type ON messages(type)",
        "CREATE INDEX IF NOT EXISTS idx_message_flags ON messages(flags)",
        "CREATE INDEX IF NOT EXISTS idx_message_reference ON messages(referenced_message_id)",
        "CREATE INDEX IF NOT EXISTS idx_thread ON messages(thread_id)",
        "CREATE INDEX IF NOT EXISTS idx_application ON messages(application_id)",
        "CREATE INDEX IF NOT EXISTS idx_channel_members_channel ON channel_members(channel_id)",
        "CREATE INDEX IF NOT EXISTS idx_channel_members_user ON channel_members(user_id)",
        "CREATE INDEX IF NOT EXISTS idx_channel_members_guild ON channel_members(guild_id)",
        "CREATE INDEX IF NOT EXISTS idx_channel_members_sync ON channel_members(synced_at)",
    ]

    for index_sql in indexes:
        cursor.execute(index_sql)


def get_schema_version() -> str:
    """Get the current schema version."""
    return "1.0.0"


def validate_schema(db_path: str) -> bool:
    """Validate that the database has the correct schema."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if required tables exist
        required_tables = [
            "messages",
            "sync_logs",
            "message_embeddings",
            "message_topics",
            "word_frequencies",
            "user_statistics",
            "conversation_chains",
            "message_temporal_stats",
            "channel_members",
        ]

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = {row[0] for row in cursor.fetchall()}

        missing_tables = set(required_tables) - existing_tables

        conn.close()

        if missing_tables:
            logger.warning(f"Missing tables: {missing_tables}")
            return False

        return True

    except Exception as e:
        logger.error(f"Schema validation failed: {e}")
        return False


# V2 Compatibility: Schema Queries Dictionary
# This provides a simplified schema dictionary for V2 database manager
SCHEMA_QUERIES = {
    "messages": """
        CREATE TABLE IF NOT EXISTS messages (
            message_id TEXT PRIMARY KEY,
            channel_name TEXT,
            author_name TEXT,
            content TEXT,
            timestamp TEXT,
            message_type TEXT DEFAULT 'default',
            reply_to TEXT,
            thread_id TEXT
        )
    """,
    
    "users": """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            display_name TEXT,
            joined_at TEXT
        )
    """,
    
    "channels": """
        CREATE TABLE IF NOT EXISTS channels (
            name TEXT PRIMARY KEY,
            topic TEXT,
            created_at TEXT
        )
    """,
    
    "sync_logs": """
        CREATE TABLE IF NOT EXISTS sync_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            guilds_synced TEXT,
            channels_skipped TEXT,
            errors TEXT,
            total_messages_synced INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
}
