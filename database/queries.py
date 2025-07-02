"""
Database operations and schema management for Discord message analysis
"""
import sqlite3
from typing import List, Dict, Optional
import aiosqlite


def init_database_schema(cursor, conn):
    """Initialize database schema"""
    # Create messages table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            channel_id TEXT,
            author_id TEXT,
            content TEXT,
            timestamp TEXT,
            edited_timestamp TEXT,
            is_bot BOOLEAN,
            is_system BOOLEAN,
            is_webhook BOOLEAN,
            has_attachments BOOLEAN,
            has_embeds BOOLEAN,
            has_stickers BOOLEAN,
            has_mentions BOOLEAN,
            has_reactions BOOLEAN,
            has_reference BOOLEAN,
            referenced_message_id TEXT,
            thread_id TEXT,
            thread_archived BOOLEAN,
            thread_archived_at TEXT,
            thread_auto_archive_duration INTEGER,
            thread_locked BOOLEAN,
            thread_member_count INTEGER,
            thread_message_count INTEGER,
            thread_name TEXT,
            thread_owner_id TEXT,
            thread_parent_id TEXT,
            thread_total_message_sent INTEGER,
            thread_type TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    ''')
    
    # Create message_embeddings table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS message_embeddings (
            message_id TEXT PRIMARY KEY,
            embedding BLOB,
            created_at TEXT,
            FOREIGN KEY (message_id) REFERENCES messages(id)
        )
    ''')
    
    # Create word_frequencies table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS word_frequencies (
            word TEXT,
            frequency INTEGER,
            last_updated TEXT,
            PRIMARY KEY (word)
        )
    ''')
    
    # Create user_statistics table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_statistics (
            user_id TEXT,
            channel_id TEXT,
            message_count INTEGER,
            avg_message_length REAL,
            active_hours TEXT,
            last_active TEXT,
            PRIMARY KEY (user_id, channel_id)
        )
    ''')
    
    # Create conversation_chains table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation_chains (
            root_message_id TEXT,
            last_message_id TEXT,
            message_count INTEGER,
            created_at TEXT,
            PRIMARY KEY (root_message_id)
        )
    ''')
    
    # Create message_temporal_stats table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS message_temporal_stats (
            channel_id TEXT,
            date TEXT,
            message_count INTEGER,
            active_users INTEGER,
            avg_message_length REAL,
            PRIMARY KEY (channel_id, date)
        )
    ''')
    
    # Create indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(channel_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_author ON messages(author_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_reference ON messages(referenced_message_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id)')
    
    conn.commit()


async def get_available_channels(pool, base_filter: str = "") -> List[str]:
    """Get list of available channels in the database (ordered by activity)"""
    try:
        async with pool.execute("""
            SELECT 
                channel_name,
                COUNT(*) as message_count
            FROM messages 
            WHERE channel_name IS NOT NULL 
            AND channel_name != ''
            AND channel_name NOT LIKE '%test%'
            AND channel_name NOT LIKE '%playground%'
            AND channel_name NOT LIKE '%pg%'
            GROUP BY channel_name
            ORDER BY message_count DESC, channel_name
        """) as cursor:
            channels = await cursor.fetchall()
            return [channel[0] for channel in channels if channel[0]]
    except Exception as e:
        print(f"Error getting available channels: {str(e)}")
        return []


async def get_available_users(pool, base_filter: str = "") -> List[str]:
    """Get list of available users in the database (display names preferred)"""
    try:
        query = f"""
            SELECT DISTINCT 
                COALESCE(author_display_name, author_name) as display_name,
                COUNT(*) as message_count
            FROM messages 
            WHERE author_name IS NOT NULL 
            AND author_id != 'sesh'
            AND author_id != '1362434210895364327'
            AND author_name != 'sesh'
            AND LOWER(author_name) != 'pepe'
            AND LOWER(author_name) != 'pepino'
            AND COALESCE(author_display_name, author_name) IS NOT NULL
            {f'AND {base_filter}' if base_filter else ''}
            GROUP BY COALESCE(author_display_name, author_name)
            ORDER BY message_count DESC, display_name
        """
        async with pool.execute(query) as cursor:
            users = await cursor.fetchall()
            return [user[0] for user in users if user[0]]
    except Exception as e:
        print(f"Error getting available users: {str(e)}")
        return []


async def get_channel_name_mapping(pool, bot_guilds=None) -> Dict[str, str]:
    """Create a mapping of old database channel names to current Discord channel names"""
    try:
        # Get all channel names from database
        async with pool.execute("""
            SELECT DISTINCT channel_name
            FROM messages 
            WHERE channel_name IS NOT NULL 
            AND channel_name != ''
            ORDER BY channel_name
        """) as cursor:
            db_channels = await cursor.fetchall()
            db_channel_names = [ch[0] for ch in db_channels if ch[0]]
        
        # If bot guilds are provided, get current Discord channel names
        current_channels = {}
        if bot_guilds:
            for guild in bot_guilds:
                for channel in guild.channels:
                    if hasattr(channel, 'name'):
                        current_channels[channel.name] = channel.name
                        # Also map with common prefixes that might be used
                        current_channels[f"#{channel.name}"] = channel.name
                        current_channels[f"🏛{channel.name}"] = channel.name
                        current_channels[f"🦾{channel.name}"] = channel.name
                        current_channels[f"🏘{channel.name}"] = channel.name
        
        # Create mapping: old_name -> new_name
        channel_mapping = {}
        for db_name in db_channel_names:
            # First, try exact match
            if db_name in current_channels:
                channel_mapping[db_name] = current_channels[db_name]
                continue
            
            # Try without emoji prefixes
            clean_db_name = db_name
            for prefix in ['🏛', '🦾', '🏘', '#']:
                if clean_db_name.startswith(prefix):
                    clean_db_name = clean_db_name[len(prefix):]
                    break
            
            # Look for matches in current channels
            best_match = None
            for current_name in current_channels.values():
                if clean_db_name == current_name:
                    best_match = current_name
                    break
                elif clean_db_name.lower() == current_name.lower():
                    best_match = current_name
                    break
                elif clean_db_name in current_name or current_name in clean_db_name:
                    best_match = current_name
            
            if best_match:
                channel_mapping[db_name] = best_match
            else:
                # Keep original name if no match found
                channel_mapping[db_name] = db_name
        
        return channel_mapping
        
    except Exception as e:
        print(f"Error creating channel name mapping: {str(e)}")
        return {}


async def get_available_channels_with_mapping(pool, bot_guilds=None) -> List[str]:
    """Get available channels with current Discord names when possible"""
    try:
        # Get channel mapping
        channel_mapping = await get_channel_name_mapping(pool, bot_guilds)
        
        # Get channels ordered by activity from database
        async with pool.execute("""
            SELECT 
                channel_name,
                COUNT(*) as message_count
            FROM messages 
            WHERE channel_name IS NOT NULL 
            AND channel_name != ''
            AND channel_name NOT LIKE '%test%'
            AND channel_name NOT LIKE '%playground%'
            AND channel_name NOT LIKE '%pg%'
            GROUP BY channel_name
            ORDER BY message_count DESC, channel_name
        """) as cursor:
            channels = await cursor.fetchall()
        
        # Map to current names and deduplicate
        mapped_channels = []
        seen = set()
        
        for channel, count in channels:
            if channel:
                # Use mapped name if available, otherwise original
                current_name = channel_mapping.get(channel, channel)
                if current_name not in seen:
                    mapped_channels.append(current_name)
                    seen.add(current_name)
        
        return mapped_channels
        
    except Exception as e:
        print(f"Error getting available channels with mapping: {str(e)}")
        # Fallback to original method
        return await get_available_channels(pool)


def filter_boilerplate_phrases(topics):
    """Remove common template/boilerplate phrases from topic list."""
    stop_phrases = set([
        'quick summary', 'tip summary', 'link :*', 'motivational monday', 'action step',
        '2025 link', 'date :*', 'headline :*', 'summary :*', 'tip:', 'summary:',
        'link:', 'headline:', 'date:', 'quick summary :*', 'tip summary :*',
        'summary', 'tip', 'link', 'headline', 'date', 'action', 'step',
        'summary *', 'tip *', 'link *', 'headline *', 'date *',
    ])
    def is_boilerplate(topic):
        t = topic.lower().strip(' :*')
        return t in stop_phrases or t.endswith(':*') or t.endswith(':')
    return [t for t in topics if not is_boilerplate(t)]


async def get_all_channel_names(pool, base_filter: str = "") -> List[str]:
    """Get all distinct channel names from the database"""
    try:
        query = f"""
            SELECT DISTINCT channel_name 
            FROM messages 
            WHERE channel_name IS NOT NULL
            {f'AND {base_filter}' if base_filter else ''}
            ORDER BY channel_name
        """
        async with pool.execute(query) as cursor:
            channels = await cursor.fetchall()
            return [ch[0] for ch in channels if ch[0]]
    except Exception as e:
        print(f"Error getting all channel names: {str(e)}")
        return []


async def find_similar_channel_names(pool, channel_name: str, base_filter: str = "") -> List[tuple]:
    """Find similar channel names in the database"""
    try:
        query = f"""
            SELECT DISTINCT channel_name, COUNT(*) as msg_count
            FROM messages 
            WHERE {base_filter}
            AND LOWER(channel_name) LIKE ?
            GROUP BY channel_name
            ORDER BY msg_count DESC
            LIMIT 5
        """
        async with pool.execute(query, (f"%{channel_name.lower()}%",)) as cursor:
            return await cursor.fetchall()
    except Exception as e:
        print(f"Error finding similar channel names: {str(e)}")
        return []


async def get_selectable_channels_and_threads(pool) -> list:
    """Return all selectable channels and forum threads for autocomplete."""
    try:
        # Get all main channels (text and forum, no threads)
        async with pool.execute('''
            SELECT DISTINCT channel_name
            FROM messages
            WHERE channel_name IS NOT NULL AND channel_name != ''
            ORDER BY channel_name
        ''') as cursor:
            main_channels = [row[0] for row in await cursor.fetchall() if row[0]]
        print(f"[DEBUG] Found {len(main_channels)} main channels: {main_channels}")

        # Get all forum threads (with parent forum name and thread title)
        async with pool.execute('''
            SELECT DISTINCT channel_name, thread_id, thread_name
            FROM messages
            WHERE thread_id IS NOT NULL AND thread_id != '' AND thread_name IS NOT NULL AND thread_name != ''
            ORDER BY channel_name, thread_name
        ''') as cursor:
            thread_rows = await cursor.fetchall()
        print(f"[DEBUG] Found {len(thread_rows)} threads: {thread_rows}")

        # Build list: (label, channel_name, thread_id)
        results = []
        for ch in main_channels:
            results.append((ch, ch, None))
        for ch, tid, tname in thread_rows:
            label = f"{ch} / {tname}"
            results.append((label, ch, tid))
        print(f"[DEBUG] Final selectable list: {results}")
        return results
    except Exception as e:
        print(f"Error getting selectable channels and threads: {str(e)}")
        return []
