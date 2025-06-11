import os
import json
import asyncio
import re
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
import discord

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.messages = True

def extract_emojis(text):
    # Fixed regex for Unicode emojis to capture common emoji ranges
    unicode_emojis = re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000026FF\U00002700-\U000027BF]', text)
    custom_emojis = re.findall(r'<a?:\w+:\d+>', text)
    return {
        'unicode_emojis': unicode_emojis,
        'custom_emojis': custom_emojis
    }

class DiscordFetcher(discord.Client):
    def __init__(self, data_store, **kwargs):
        super().__init__(**kwargs)
        self.data_store = data_store
        self.new_data = {}
        self.sync_log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "guilds_synced": [],
            "channels_skipped": [],
            "errors": [],
            "total_messages_synced": 0
        }

    async def on_ready(self):
        print(f"✅ Logged in as: {self.user} (ID: {self.user.id})")

        for guild in self.guilds:
            print(f"\n📂 Guild: {guild.name} (ID: {guild.id})")
            self.new_data.setdefault(guild.name, {})
            self.sync_log_entry["guilds_synced"].append({
                "guild_name": guild.name,
                "guild_id": str(guild.id)
            })

            for channel in guild.text_channels:
                print(f"  📄 Channel: #{channel.name} (ID: {channel.id})")

                existing_msgs = self.data_store.get(guild.name, {}).get(channel.name, [])
                last_message_id = existing_msgs[-1]['id'] if existing_msgs else None

                new_messages = []
                try:
                    async for message in channel.history(
                        limit=None,
                        after=discord.Object(id=last_message_id) if last_message_id else None,
                        oldest_first=True
                    ):
                        new_messages.append({
                            'id': str(message.id),
                            'content': message.content.replace('\u2028', ' ').replace('\u2029', ' ').strip(),
                            'timestamp': message.created_at.isoformat(),
                            'edited_timestamp': message.edited_at.isoformat() if message.edited_at else None,
                            'jump_url': message.jump_url,
                            'author': {
                                'id': str(message.author.id),
                                'name': message.author.name,
                                'discriminator': message.author.discriminator,
                                'display_name': getattr(message.author, "display_name", message.author.name),
                                'is_bot': message.author.bot,
                                'avatar_url': str(message.author.avatar.url) if message.author.avatar else None,
                            },
                            'user_presence': {
                                'status': str(message.author.status),
                                'activity': str(message.author.activity.name) if message.author.activity else None,
                            } if isinstance(message.author, discord.Member) else None,
                            'guild': {
                                'id': str(guild.id),
                                'name': guild.name,
                                'member_count': guild.member_count,
                            } if guild else None,
                            'channel': {
                                'id': str(channel.id),
                                'name': channel.name,
                                'type': str(channel.type),
                                'topic': getattr(channel, 'topic', None),
                            },
                            'thread': {
                                'id': str(message.thread.id),
                                'name': message.thread.name
                            } if hasattr(message, "thread") and message.thread else None,
                            'mentions': [str(user.id) for user in message.mentions],
                            'mention_everyone': message.mention_everyone,
                            'mention_roles': [str(role.id) for role in message.role_mentions],
                            'referenced_message_id': str(message.reference.message_id) if message.reference else None,
                            'attachments': [
                                {
                                    'url': att.url,
                                    'filename': att.filename,
                                    'size': att.size,
                                    'content_type': att.content_type
                                } for att in message.attachments
                            ],
                            'embeds': [embed.to_dict() for embed in message.embeds],
                            'reactions': [
                                {
                                    'emoji': str(reaction.emoji),
                                    'count': reaction.count,
                                    'me': reaction.me
                                } for reaction in message.reactions
                            ],
                            'emoji_stats': extract_emojis(message.content),
                            'pinned': message.pinned,
                            'flags': message.flags.value if message.flags else None,
                            'nonce': message.nonce,
                            'type': str(message.type),
                            'is_system': message.is_system(),
                            'mentions_everyone': message.mention_everyone,
                            'has_reactions': bool(message.reactions)
                        })
                except discord.Forbidden:
                    print(f"    🚫 Skipped channel #{channel.name}: insufficient permissions")
                    self.sync_log_entry["channels_skipped"].append({
                        "guild_name": guild.name,
                        "channel_name": channel.name,
                        "channel_id": str(channel.id),
                        "reason": "Forbidden - missing read permissions"
                    })
                    continue
                except Exception as e:
                    print(f"    ❌ Error fetching channel #{channel.name}: {str(e)}")
                    self.sync_log_entry["errors"].append(str(e))
                    continue

                if new_messages:
                    print(f"    ✅ {len(new_messages)} new message(s)")
                    self.sync_log_entry["total_messages_synced"] += len(new_messages)
                    self.new_data[guild.name].setdefault(channel.name, []).extend(new_messages)
                else:
                    print(f"    📫 No new messages")

        await self.close()

def init_database(db_path='discord_messages.db'):
    """Initialize the SQLite database with required tables"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create messages table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            content TEXT,
            timestamp TEXT,
            edited_timestamp TEXT,
            jump_url TEXT,
            author_id TEXT,
            author_name TEXT,
            author_discriminator TEXT,
            author_display_name TEXT,
            author_is_bot BOOLEAN,
            author_avatar_url TEXT,
            author_status TEXT,
            author_activity TEXT,
            guild_id TEXT,
            guild_name TEXT,
            guild_member_count INTEGER,
            channel_id TEXT,
            channel_name TEXT,
            channel_type TEXT,
            channel_topic TEXT,
            thread_id TEXT,
            thread_name TEXT,
            mentions TEXT,
            mention_everyone BOOLEAN,
            mention_roles TEXT,
            referenced_message_id TEXT,
            attachments TEXT,
            embeds TEXT,
            reactions TEXT,
            emoji_stats TEXT,
            pinned BOOLEAN,
            flags INTEGER,
            nonce TEXT,
            type TEXT,
            is_system BOOLEAN,
            mentions_everyone BOOLEAN,
            has_reactions BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create sync logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sync_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            guilds_synced TEXT,
            channels_skipped TEXT,
            errors TEXT,
            total_messages_synced INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create index for faster queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_message_timestamp ON messages(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_guild_channel ON messages(guild_name, channel_name)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_author ON messages(author_id)')
    
    conn.commit()
    conn.close()

def load_existing_data(db_path='discord_messages.db'):
    """Load existing message data from SQLite database"""
    if not os.path.exists(db_path):
        return {}
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all messages grouped by guild and channel
    cursor.execute('''
        SELECT guild_name, channel_name, id, timestamp 
        FROM messages 
        ORDER BY guild_name, channel_name, timestamp
    ''')
    
    data = {}
    for row in cursor.fetchall():
        guild_name, channel_name, msg_id, timestamp = row
        if guild_name not in data:
            data[guild_name] = {}
        if channel_name not in data[guild_name]:
            data[guild_name][channel_name] = []
        data[guild_name][channel_name].append({'id': msg_id, 'timestamp': timestamp})
    
    conn.close()
    return data

def save_messages_to_db(messages_data, db_path='discord_messages.db'):
    """Save new messages to SQLite database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    for guild_name, channels in messages_data.items():
        for channel_name, messages in channels.items():
            for msg in messages:
                # Convert complex data to JSON strings for storage
                mentions_json = json.dumps(msg.get('mentions', []))
                mention_roles_json = json.dumps(msg.get('mention_roles', []))
                attachments_json = json.dumps(msg.get('attachments', []))
                embeds_json = json.dumps(msg.get('embeds', []))
                reactions_json = json.dumps(msg.get('reactions', []))
                emoji_stats_json = json.dumps(msg.get('emoji_stats', {}))
                
                # Extract nested data
                author = msg.get('author', {})
                user_presence = msg.get('user_presence', {})
                guild = msg.get('guild', {})
                channel = msg.get('channel', {})
                thread = msg.get('thread', {})
                
                cursor.execute('''
                    INSERT OR REPLACE INTO messages (
                        id, content, timestamp, edited_timestamp, jump_url,
                        author_id, author_name, author_discriminator, author_display_name,
                        author_is_bot, author_avatar_url, author_status, author_activity,
                        guild_id, guild_name, guild_member_count,
                        channel_id, channel_name, channel_type, channel_topic,
                        thread_id, thread_name,
                        mentions, mention_everyone, mention_roles, referenced_message_id,
                        attachments, embeds, reactions, emoji_stats,
                        pinned, flags, nonce, type, is_system, mentions_everyone, has_reactions
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    msg.get('id'),
                    msg.get('content'),
                    msg.get('timestamp'),
                    msg.get('edited_timestamp'),
                    msg.get('jump_url'),
                    author.get('id'),
                    author.get('name'),
                    author.get('discriminator'),
                    author.get('display_name'),
                    author.get('is_bot'),
                    author.get('avatar_url'),
                    user_presence.get('status') if user_presence else None,
                    user_presence.get('activity') if user_presence else None,
                    guild.get('id') if guild else None,
                    guild.get('name') if guild else guild_name,
                    guild.get('member_count') if guild else None,
                    channel.get('id') if channel else None,
                    channel.get('name') if channel else channel_name,
                    channel.get('type') if channel else None,
                    channel.get('topic') if channel else None,
                    thread.get('id') if thread else None,
                    thread.get('name') if thread else None,
                    mentions_json,
                    msg.get('mention_everyone'),
                    mention_roles_json,
                    msg.get('referenced_message_id'),
                    attachments_json,
                    embeds_json,
                    reactions_json,
                    emoji_stats_json,
                    msg.get('pinned'),
                    msg.get('flags'),
                    msg.get('nonce'),
                    msg.get('type'),
                    msg.get('is_system'),
                    msg.get('mentions_everyone'),
                    msg.get('has_reactions')
                ))
    
    conn.commit()
    conn.close()

def save_sync_log_to_db(sync_log_entry, db_path='discord_messages.db'):
    """Save sync log entry to SQLite database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO sync_logs (
            timestamp, guilds_synced, channels_skipped, errors, total_messages_synced
        ) VALUES (?, ?, ?, ?, ?)
    ''', (
        sync_log_entry['timestamp'],
        json.dumps(sync_log_entry['guilds_synced']),
        json.dumps(sync_log_entry['channels_skipped']),
        json.dumps(sync_log_entry['errors']),
        sync_log_entry['total_messages_synced']
    ))
    
    conn.commit()
    conn.close()

def load_existing_data_old(path='discord_messages.json'):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_combined_data(existing, new_data, path='discord_messages.json'):
    for guild, channels in new_data.items():
        existing.setdefault(guild, {})
        for channel, messages in channels.items():
            existing[guild].setdefault(channel, [])
            existing[guild][channel].extend(messages)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

def save_sync_log(sync_log_entry):
    os.makedirs("logs", exist_ok=True)
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    log_path = os.path.join("logs", f"sync_log_{date_str}.jsonl")
    with open(log_path, 'a', encoding='utf-8') as log_f:
        log_f.write(json.dumps(sync_log_entry) + '\n')

def update_discord_messages():
    print("🔌 Connecting to Discord...")
    print("🗄️ Initializing database...")
    init_database()
    
    data_store = load_existing_data()
    client = DiscordFetcher(data_store=data_store, intents=intents)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(client.start(DISCORD_TOKEN))

    if client.new_data:
        print("💾 Saving messages to database...")
        save_messages_to_db(client.new_data)
        save_sync_log_to_db(client.sync_log_entry)
        print("\n📂 Data updated in database.")
        print(f"📝 Total messages synced: {client.sync_log_entry['total_messages_synced']}")
    else:
        print("\n✅ No new messages found.")

if __name__ == "__main__":
    update_discord_messages()
    print("🔌 Disconnecting from Discord...")
