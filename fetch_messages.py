import os
import json
import asyncio
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
import discord
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True
intents.messages = True
intents.reactions = True  # Enable reactions intent to capture reaction data

def extract_emojis(text):
    # Fixed regex for Unicode emojis to capture common emoji ranges
    unicode_emojis = re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000026FF\U00002700-\U000027BF]', text)
    custom_emojis = re.findall(r'<a?:\w+:\d+>', text)
    return {
        'unicode_emojis': unicode_emojis,
        'custom_emojis': custom_emojis
    }

def extract_components(message):
    """Extract all components from a message (buttons, select menus, etc.)"""
    components = []
    if hasattr(message, 'components'):
        for component in message.components:
            if isinstance(component, discord.ui.Button):
                components.append({
                    'type': 'button',
                    'label': component.label,
                    'custom_id': component.custom_id,
                    'style': str(component.style),
                    'disabled': component.disabled,
                    'url': component.url
                })
            elif isinstance(component, discord.ui.Select):
                components.append({
                    'type': 'select',
                    'custom_id': component.custom_id,
                    'placeholder': component.placeholder,
                    'min_values': component.min_values,
                    'max_values': component.max_values,
                    'options': [
                        {
                            'label': option.label,
                            'value': option.value,
                            'description': option.description,
                            'default': option.default
                        } for option in component.options
                    ]
                })
    return components

def extract_interaction_data(message):
    """Extract interaction data if the message is a response to an interaction"""
    if hasattr(message, 'interaction_metadata'):  # Updated to use interaction_metadata
        metadata = message.interaction_metadata
        return {
            'id': str(metadata.id) if metadata and hasattr(metadata, 'id') else None,
            'type': str(metadata.type) if metadata and hasattr(metadata, 'type') else None,
            'name': metadata.name if metadata and hasattr(metadata, 'name') else None,
            'user_id': str(metadata.user_id) if metadata and hasattr(metadata, 'user_id') else None,
            'guild_id': str(metadata.guild_id) if metadata and hasattr(metadata, 'guild_id') else None,
            'channel_id': str(metadata.channel_id) if metadata and hasattr(metadata, 'channel_id') else None,
            'data': metadata.data if metadata and hasattr(metadata, 'data') else None
        }
    return None

def extract_sticker_data(message):
    """Extract sticker data from a message, robust to missing fields"""
    stickers = []
    if hasattr(message, 'stickers'):
        for sticker in message.stickers:
            stickers.append({
                'id': str(getattr(sticker, 'id', '')),
                'name': getattr(sticker, 'name', None),
                'format_type': str(getattr(sticker, 'format_type', None)),
                'url': getattr(sticker, 'url', None)
            })
    return stickers

def extract_role_subscription_data(message):
    """Extract role subscription data if present"""
    if hasattr(message, 'role_subscription_data'):
        return {
            'role_subscription_listing_id': str(message.role_subscription_data.role_subscription_listing_id),
            'tier_name': message.role_subscription_data.tier_name,
            'total_months_subscribed': message.role_subscription_data.total_months_subscribed,
            'is_renewal': message.role_subscription_data.is_renewal
        }
    return None

class DiscordFetcher(discord.Client):
    def __init__(self, data_store, **kwargs):
        super().__init__(**kwargs)
        self.data_store = data_store
        self.new_data = {}
        self.rate_limit_delay = 0.1  # Minimal delay between channels (100ms) to respect rate limits
        self.max_retries = 3  # Maximum number of retries for failed requests
        self.sync_log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "guilds_synced": [],
            "channels_skipped": [],
            "errors": [],
            "total_messages_synced": 0,
            "sync_start_time": datetime.now(timezone.utc).isoformat(),
            "sync_end_time": None,
            "sync_duration_seconds": None
        }

    async def on_ready(self):
        logger.info(f"✅ Logged in as: {self.user} (ID: {self.user.id})")
        sync_start_time = datetime.now(timezone.utc)

        for guild in self.guilds:
            logger.info(f"\n📂 Guild: {guild.name} (ID: {guild.id})")
            self.new_data.setdefault(guild.name, {})
            self.sync_log_entry["guilds_synced"].append({
                "guild_name": guild.name,
                "guild_id": str(guild.id),
                "sync_start_time": datetime.now(timezone.utc).isoformat()
            })

            accessible_channels = []
            inaccessible_channels = []
            for channel in guild.text_channels:
                perms = channel.permissions_for(guild.me)
                if perms.read_messages and perms.read_message_history:
                    accessible_channels.append(channel)
                else:
                    inaccessible_channels.append(channel)

            logger.info(f"    Accessible channels: {[f'#{c.name}' for c in accessible_channels]}")
            logger.info(f"    Inaccessible channels: {[f'#{c.name}' for c in inaccessible_channels]}")

            for channel in inaccessible_channels:
                logger.warning(f"  🚫 Skipped channel #{channel.name}: insufficient permissions (pre-check)")
                self.sync_log_entry["channels_skipped"].append({
                    "guild_name": guild.name,
                    "channel_name": channel.name,
                    "channel_id": str(channel.id),
                    "reason": "Forbidden - missing read permissions (pre-check)",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })

            for channel in accessible_channels:
                logger.info(f"  📄 Channel: #{channel.name} (ID: {channel.id})")
                
                try:
                    logger.info(f"    🆕 Full fetch from beginning of channel history")
                    
                    # Always fetch all messages (no incremental logic)
                    new_messages = await self.fetch_with_retry(channel, None)
                    
                    if new_messages:
                        logger.info(f"    ✅ {len(new_messages)} message(s) fetched")
                        self.sync_log_entry["total_messages_synced"] += len(new_messages)
                        self.new_data[guild.name].setdefault(channel.name, []).extend(new_messages)
                    else:
                        logger.info(f"    📫 No messages found")
                    
                    # Fetch channel members
                    logger.info(f"    👥 Fetching channel members...")
                    channel_members = await self._fetch_channel_members(channel)
                    
                    if channel_members:
                        logger.info(f"    ✅ {len(channel_members)} member(s) found")
                        # Store channel members in new_data for saving to DB
                        self.new_data[guild.name].setdefault('_channel_members', {})
                        self.new_data[guild.name]['_channel_members'][channel.name] = channel_members
                    else:
                        logger.info(f"    👥 No accessible members found")
                        
                except Exception as e:
                    logger.error(f"    ❌ Error fetching channel #{channel.name}: {str(e)}")
                    self.sync_log_entry["errors"].append({
                        "error": str(e),
                        "guild_name": guild.name,
                        "channel_name": channel.name,
                        "channel_id": str(channel.id),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    continue

            self.sync_log_entry["guilds_synced"][-1]["sync_end_time"] = datetime.now(timezone.utc).isoformat()

        sync_end_time = datetime.now(timezone.utc)
        self.sync_log_entry["sync_end_time"] = sync_end_time.isoformat()
        self.sync_log_entry["sync_duration_seconds"] = (sync_end_time - sync_start_time).total_seconds()

        await self.close()

    async def fetch_with_retry(self, channel, last_message_id=None):
        """Fetch messages with retry logic, only sleep after each channel."""
        for attempt in range(self.max_retries):
            try:
                messages = []
                message_count = 0
                last_progress_time = time.time()
                
                async for message in channel.history(
                    limit=None,
                    after=discord.Object(id=last_message_id) if last_message_id else None,
                    oldest_first=True
                ):
                    message_count += 1
                    current_time = time.time()
                    # Show progress every 100 messages or every 5 seconds
                    if message_count % 100 == 0 or (current_time - last_progress_time) >= 5:
                        logger.info(f"      Fetched {message_count} messages from #{channel.name}")
                        last_progress_time = current_time
                    messages.append(await self._convert_message_to_dict(message))
                # Only sleep after the channel is done
                await asyncio.sleep(self.rate_limit_delay)
                return messages
            except discord.Forbidden:
                logger.error(f"Insufficient permissions to fetch messages from #{channel.name}")
                raise
            except discord.HTTPException as e:
                if attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    logger.warning(f"HTTP error (attempt {attempt + 1}/{self.max_retries}): {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to fetch messages after {self.max_retries} attempts")
                    raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

    async def _fetch_channel_members(self, channel: discord.TextChannel) -> List[Dict[str, Any]]:
        """Fetch all members who have access to a specific channel"""
        try:
            members_data = []
            member_count = 0
            
            logger.info(f"      Fetching members for #{channel.name}")
            
            for member in channel.guild.members:
                # Check if member can read this channel
                perms = channel.permissions_for(member)
                if perms.read_messages:
                    member_count += 1
                    
                    # Get member roles
                    roles = []
                    for role in member.roles:
                        if role.name != "@everyone":  # Skip @everyone role
                            roles.append({
                                'id': str(role.id),
                                'name': role.name,
                                'color': role.color.value,
                                'position': role.position,
                                'permissions': role.permissions.value
                            })
                    
                    # Get member permissions for this channel
                    member_perms = {
                        'read_messages': perms.read_messages,
                        'send_messages': perms.send_messages,
                        'manage_messages': perms.manage_messages,
                        'read_message_history': perms.read_message_history,
                        'add_reactions': perms.add_reactions,
                        'attach_files': perms.attach_files,
                        'embed_links': perms.embed_links,
                        'mention_everyone': perms.mention_everyone
                    }
                    
                    member_data = {
                        'channel_id': str(channel.id),
                        'channel_name': channel.name,
                        'guild_id': str(channel.guild.id),
                        'guild_name': channel.guild.name,
                        'user_id': str(member.id),
                        'user_name': member.name,
                        'user_display_name': member.display_name,
                        'user_joined_at': member.joined_at.isoformat() if member.joined_at else None,
                        'user_roles': json.dumps(roles),
                        'is_bot': member.bot,
                        'member_permissions': json.dumps(member_perms),
                        'synced_at': datetime.now(timezone.utc).isoformat()
                    }
                    
                    members_data.append(member_data)
            
            logger.info(f"      Found {member_count} members with access to #{channel.name}")
            return members_data
            
        except Exception as e:
            logger.error(f"Error fetching members for #{channel.name}: {e}")
            return []

    async def _convert_message_to_dict(self, message: discord.Message) -> Dict[str, Any]:
        # Helper function to safely get role type
        def get_role_type(role):
            return str(getattr(role, 'type', 'role'))

        # Helper function to safely get overwrite data
        def get_overwrite_data(overwrite):
            try:
                return {
                    'id': str(getattr(overwrite, 'id', '')),
                    'type': get_role_type(overwrite),
                    'allow': getattr(getattr(overwrite, 'allow', None), 'value', None),
                    'deny': getattr(getattr(overwrite, 'deny', None), 'value', None)
                }
            except Exception as e:
                logger.warning(f"Error processing overwrite: {e}")
                return None

        # Helper for safe attachment dict
        def safe_attachment(att):
            return {
                'url': getattr(att, 'url', None),
                'filename': getattr(att, 'filename', None),
                'size': getattr(att, 'size', None),
                'content_type': getattr(att, 'content_type', None),
                'height': getattr(att, 'height', None),
                'width': getattr(att, 'width', None),
                'ephemeral': getattr(att, 'ephemeral', None),
                'description': getattr(att, 'description', None),
                # Only include if present
                **({'duration_secs': getattr(att, 'duration_secs', None)} if hasattr(att, 'duration_secs') else {}),
                **({'waveform': getattr(att, 'waveform', None)} if hasattr(att, 'waveform') else {}),
            }

        # Helper for safe referenced message
        def safe_referenced_message(ref):
            if not ref or not hasattr(ref, 'resolved') or not ref.resolved:
                return None
            resolved = ref.resolved
            return {
                'id': str(getattr(resolved, 'id', '')),
                'content': getattr(resolved, 'content', None),
                'author_id': str(getattr(getattr(resolved, 'author', None), 'id', '')) if getattr(resolved, 'author', None) else None,
                'channel_id': str(getattr(getattr(resolved, 'channel', None), 'id', '')) if getattr(resolved, 'channel', None) else None,
                'guild_id': str(getattr(getattr(resolved, 'guild', None), 'id', '')) if getattr(resolved, 'guild', None) else None,
            }

        # Helper for safe message reference
        def safe_message_reference(ref):
            if not ref:
                return None
            return {
                'message_id': str(getattr(ref, 'message_id', '')),
                'channel_id': str(getattr(ref, 'channel_id', '')),
                'guild_id': str(getattr(ref, 'guild_id', '')) if getattr(ref, 'guild_id', None) else None,
                'fail_if_not_exists': getattr(ref, 'fail_if_not_exists', None),
            }

        return {
            'id': str(message.id),
            'content': getattr(message, 'content', '').replace('\u2028', ' ').replace('\u2029', ' ').strip(),
            'timestamp': message.created_at.isoformat(),
            'edited_timestamp': message.edited_at.isoformat() if getattr(message, 'edited_at', None) else None,
            'jump_url': getattr(message, 'jump_url', None),
            'author': {
                'id': str(getattr(message.author, 'id', '')),
                'name': getattr(message.author, 'name', None),
                'discriminator': getattr(message.author, 'discriminator', None),
                'display_name': getattr(message.author, 'display_name', getattr(message.author, 'name', None)),
                'is_bot': getattr(message.author, 'bot', None),
                'avatar_url': str(getattr(getattr(message.author, 'avatar', None), 'url', '')) if getattr(message.author, 'avatar', None) else None,
                'accent_color': getattr(getattr(message.author, 'accent_color', None), 'value', None) if getattr(message.author, 'accent_color', None) else None,
                'banner_url': str(getattr(getattr(message.author, 'banner', None), 'url', '')) if getattr(message.author, 'banner', None) else None,
                'color': getattr(getattr(message.author, 'color', None), 'value', None) if getattr(message.author, 'color', None) else None,
                'created_at': message.author.created_at.isoformat() if getattr(message.author, 'created_at', None) else None,
                'default_avatar_url': str(getattr(getattr(message.author, 'default_avatar', None), 'url', '')) if getattr(message.author, 'default_avatar', None) else None,
                'public_flags': getattr(getattr(message.author, 'public_flags', None), 'value', None) if getattr(message.author, 'public_flags', None) else None,
                'system': getattr(message.author, 'system', None),
                'verified': getattr(message.author, 'verified', None),
            },
            'user_presence': {
                'status': str(getattr(message.author, 'status', '')),
                'activity': str(getattr(getattr(message.author, 'activity', None), 'name', '')) if getattr(message.author, 'activity', None) else None,
                'desktop_status': str(getattr(message.author, 'desktop_status', '')) if hasattr(message.author, 'desktop_status') else None,
                'mobile_status': str(getattr(message.author, 'mobile_status', '')) if hasattr(message.author, 'mobile_status') else None,
                'web_status': str(getattr(message.author, 'web_status', '')) if hasattr(message.author, 'web_status') else None,
            } if isinstance(message.author, discord.Member) else None,
            'guild': {
                'id': str(getattr(message.guild, 'id', '')),
                'name': getattr(message.guild, 'name', None),
                'member_count': getattr(message.guild, 'member_count', None),
                'description': getattr(message.guild, 'description', None),
                'icon_url': str(getattr(getattr(message.guild, 'icon', None), 'url', '')) if getattr(message.guild, 'icon', None) else None,
                'banner_url': str(getattr(getattr(message.guild, 'banner', None), 'url', '')) if getattr(message.guild, 'banner', None) else None,
                'splash_url': str(getattr(getattr(message.guild, 'splash', None), 'url', '')) if getattr(message.guild, 'splash', None) else None,
                'discovery_splash_url': str(getattr(getattr(message.guild, 'discovery_splash', None), 'url', '')) if getattr(message.guild, 'discovery_splash', None) else None,
                'features': getattr(message.guild, 'features', None),
                'verification_level': str(getattr(message.guild, 'verification_level', '')),
                'explicit_content_filter': str(getattr(message.guild, 'explicit_content_filter', '')),
                'mfa_level': str(getattr(message.guild, 'mfa_level', '')),
                'premium_tier': str(getattr(message.guild, 'premium_tier', '')),
                'premium_subscription_count': getattr(message.guild, 'premium_subscription_count', None),
            } if getattr(message, 'guild', None) else None,
            'channel': {
                'id': str(getattr(message.channel, 'id', '')),
                'name': getattr(message.channel, 'name', None),
                'type': str(getattr(message.channel, 'type', '')),
                'topic': getattr(message.channel, 'topic', None),
                'nsfw': getattr(message.channel, 'nsfw', None),
                'position': getattr(message.channel, 'position', None),
                'slowmode_delay': getattr(message.channel, 'slowmode_delay', None),
                'category_id': str(getattr(message.channel, 'category_id', '')) if getattr(message.channel, 'category_id', None) else None,
                'overwrites': [
                    overwrite_data for overwrite in getattr(message.channel, 'overwrites', [])
                    if (overwrite_data := get_overwrite_data(overwrite)) is not None
                ],
            },
            'thread': {
                'id': str(getattr(message.thread, 'id', '')),
                'name': getattr(message.thread, 'name', None),
                'archived': getattr(message.thread, 'archived', None),
                'auto_archive_duration': getattr(message.thread, 'auto_archive_duration', None),
                'locked': getattr(message.thread, 'locked', None),
                'member_count': getattr(message.thread, 'member_count', None),
                'message_count': getattr(message.thread, 'message_count', None),
                'owner_id': str(getattr(message.thread, 'owner_id', '')),
                'parent_id': str(getattr(message.thread, 'parent_id', '')),
                'slowmode_delay': getattr(message.thread, 'slowmode_delay', None),
            } if hasattr(message, "thread") and getattr(message, 'thread', None) else None,
            'mentions': [str(getattr(user, 'id', '')) for user in getattr(message, 'mentions', [])],
            'mention_everyone': getattr(message, 'mention_everyone', None),
            'mention_roles': [str(getattr(role, 'id', '')) for role in getattr(message, 'role_mentions', [])],
            'mention_channels': [
                {
                    'id': str(getattr(channel, 'id', '')),
                    'name': getattr(channel, 'name', None),
                    'guild_id': str(getattr(getattr(channel, 'guild', None), 'id', '')) if getattr(channel, 'guild', None) else None
                } for channel in getattr(message, 'channel_mentions', [])
            ],
            'referenced_message_id': str(getattr(getattr(message, 'reference', None), 'message_id', '')) if getattr(message, 'reference', None) else None,
            'referenced_message': safe_referenced_message(getattr(message, 'reference', None)),
            'attachments': [safe_attachment(att) for att in getattr(message, 'attachments', [])],
            'embeds': [embed.to_dict() for embed in getattr(message, 'embeds', [])],
            'reactions': [
                {
                    'emoji': str(getattr(reaction, 'emoji', '')),
                    'count': getattr(reaction, 'count', None),
                    'me': getattr(reaction, 'me', None),
                    'emoji_id': str(getattr(getattr(reaction, 'emoji', None), 'id', '')) if getattr(reaction, 'emoji', None) and hasattr(reaction.emoji, 'id') else None,
                    'emoji_name': getattr(getattr(reaction, 'emoji', None), 'name', None) if getattr(reaction, 'emoji', None) and hasattr(reaction.emoji, 'name') else None,
                    'emoji_animated': getattr(getattr(reaction, 'emoji', None), 'animated', None) if getattr(reaction, 'emoji', None) and hasattr(reaction.emoji, 'animated') else None,
                } for reaction in getattr(message, 'reactions', [])
            ],
            'emoji_stats': extract_emojis(getattr(message, 'content', '')),
            'pinned': getattr(message, 'pinned', None),
            'flags': getattr(getattr(message, 'flags', None), 'value', None) if getattr(message, 'flags', None) else None,
            'nonce': getattr(message, 'nonce', None),
            'type': str(getattr(message, 'type', '')),
            'is_system': message.is_system() if hasattr(message, 'is_system') else None,
            'mentions_everyone': getattr(message, 'mention_everyone', None),
            'has_reactions': bool(getattr(message, 'reactions', [])),
            'components': extract_components(message) if 'extract_components' in globals() else None,
            'interaction': extract_interaction_data(message),
            'stickers': extract_sticker_data(message),
            'role_subscription_data': extract_role_subscription_data(message),
            'application_id': str(getattr(message, 'application_id', '')) if getattr(message, 'application_id', None) else None,
            'application': {
                'id': str(getattr(getattr(message, 'application', None), 'id', '')),
                'name': getattr(getattr(message, 'application', None), 'name', None),
                'description': getattr(getattr(message, 'application', None), 'description', None),
                'icon_url': str(getattr(getattr(getattr(message, 'application', None), 'icon', None), 'url', '')) if getattr(getattr(message, 'application', None), 'icon', None) else None,
                'cover_image_url': str(getattr(getattr(getattr(message, 'application', None), 'cover_image', None), 'url', '')) if getattr(getattr(message, 'application', None), 'cover_image', None) else None,
                'bot_public': getattr(getattr(message, 'application', None), 'bot_public', None),
                'bot_require_code_grant': getattr(getattr(message, 'application', None), 'bot_require_code_grant', None),
                'terms_of_service_url': getattr(getattr(message, 'application', None), 'terms_of_service_url', None),
                'privacy_policy_url': getattr(getattr(message, 'application', None), 'privacy_policy_url', None),
            } if getattr(message, 'application', None) else None,
            'activity': {
                'type': str(getattr(getattr(message, 'activity', None), 'type', '')),
                'party_id': getattr(getattr(message, 'activity', None), 'party_id', None),
                'application_id': str(getattr(getattr(message, 'activity', None), 'application_id', '')),
                'name': getattr(getattr(message, 'activity', None), 'name', None),
                'state': getattr(getattr(message, 'activity', None), 'state', None),
                'details': getattr(getattr(message, 'activity', None), 'details', None),
                'timestamps': getattr(getattr(message, 'activity', None), 'timestamps', None),
                'assets': getattr(getattr(message, 'activity', None), 'assets', None),
                'sync_id': getattr(getattr(message, 'activity', None), 'sync_id', None),
                'session_id': getattr(getattr(message, 'activity', None), 'session_id', None),
                'flags': getattr(getattr(message, 'activity', None), 'flags', None),
            } if getattr(message, 'activity', None) else None,
            'position': getattr(message, 'position', None),
            'role_subscription_listing_id': str(getattr(message, 'role_subscription_listing_id', '')) if hasattr(message, 'role_subscription_listing_id') else None,
            'webhook_id': str(getattr(message, 'webhook_id', '')) if getattr(message, 'webhook_id', None) else None,
            'tts': getattr(message, 'tts', None),
            # Only include if present
            **({'suppress_embeds': getattr(message, 'suppress_embeds', None)} if hasattr(message, 'suppress_embeds') else {}),
            'allowed_mentions': getattr(getattr(message, 'allowed_mentions', None), 'to_dict', lambda: None)() if getattr(message, 'allowed_mentions', None) else None,
            'message_reference': safe_message_reference(getattr(message, 'reference', None)),
        }

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
    
    # Create all other tables first
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS message_embeddings (
            message_id TEXT PRIMARY KEY,
            embedding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (message_id) REFERENCES messages(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS message_topics (
            message_id TEXT,
            topic_id INTEGER,
            confidence FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (message_id, topic_id),
            FOREIGN KEY (message_id) REFERENCES messages(id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS word_frequencies (
            word TEXT,
            channel_id TEXT,
            frequency INTEGER,
            last_updated TIMESTAMP,
            PRIMARY KEY (word, channel_id),
            FOREIGN KEY (channel_id) REFERENCES messages(channel_id)
        )
    ''')
    
    cursor.execute('''
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
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation_chains (
            chain_id INTEGER PRIMARY KEY,
            root_message_id TEXT,
            last_message_id TEXT,
            message_count INTEGER,
            created_at TIMESTAMP,
            FOREIGN KEY (root_message_id) REFERENCES messages(id),
            FOREIGN KEY (last_message_id) REFERENCES messages(id)
        )
    ''')
    
    cursor.execute('''
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
    ''')
    
    # Create channel members table for tracking actual channel membership
    cursor.execute('''
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
    ''')
    
    # Create virtual table for full-text search
    cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
            content,
            author_name,
            channel_name,
            guild_name,
            content='messages',
            content_rowid='id'
        )
    ''')
    
    # Create indexes after all tables are created
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_message_timestamp ON messages(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_guild_channel ON messages(guild_name, channel_name)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_author ON messages(author_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_message_type ON messages(type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_message_flags ON messages(flags)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_message_reference ON messages(referenced_message_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_thread ON messages(thread_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_application ON messages(application_id)')
    
    # Create indexes for channel_members table
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_channel_members_channel ON channel_members(channel_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_channel_members_user ON channel_members(user_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_channel_members_guild ON channel_members(guild_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_channel_members_sync ON channel_members(synced_at)')
    
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
    """Save messages to database with proper error handling and batching"""
    if not messages_data:
        return
        
    try:
        # Initialize database connection
        conn = sqlite3.connect(db_path)
        
        # Flatten the nested structure to get all messages
        all_messages = []
        for guild_name, channels in messages_data.items():
            for channel_name, messages in channels.items():
                all_messages.extend(messages)
        
        if not all_messages:
            logger.info("No messages to save")
            return
            
        logger.info(f"Saving {len(all_messages)} messages to database...")
        
        # Process messages in batches
        batch_size = 100
        for i in range(0, len(all_messages), batch_size):
            batch = all_messages[i:i + batch_size]
            message_data = []
            
            for msg in batch:
                try:
                    if not isinstance(msg, dict):
                        logger.warning(f"Skipping invalid message format: {type(msg)}")
                        continue
                        
                    # Extract basic message data
                    message_data.append({
                        'message_id': str(msg.get('id', '')),
                        'channel_id': str(msg.get('channel', {}).get('id', '')),
                        'channel_name': msg.get('channel', {}).get('name', ''),
                        'channel_type': str(msg.get('channel', {}).get('type', '')),
                        'guild_id': str(msg.get('guild', {}).get('id', '')),
                        'guild_name': msg.get('guild', {}).get('name', ''),
                        'author_id': str(msg.get('author', {}).get('id', '')),
                        'author_name': msg.get('author', {}).get('name', ''),
                        'author_display_name': msg.get('author', {}).get('display_name', ''),
                        'content': msg.get('content', ''),
                        'timestamp': msg.get('timestamp', ''),
                        'edited_timestamp': msg.get('edited_timestamp'),
                        'jump_url': msg.get('jump_url', ''),
                        'is_bot': msg.get('author', {}).get('is_bot', False),
                        'has_attachments': bool(msg.get('attachments', [])),
                        'has_embeds': bool(msg.get('embeds', [])),
                        'has_reactions': bool(msg.get('reactions', [])),
                        'has_stickers': bool(msg.get('stickers', [])),
                        'reference_message_id': str(msg.get('referenced_message_id', '')) if msg.get('referenced_message_id') else None,
                        'reference_channel_id': None,  # Not available in current structure
                        'reference_guild_id': None,    # Not available in current structure
                        'author_status': msg.get('user_presence', {}).get('status') if msg.get('user_presence') else None,
                        'author_activities': json.dumps(msg.get('user_presence', {}).get('activity', [])) if msg.get('user_presence') else json.dumps([])
                    })
                except Exception as e:
                    logger.error(f"Error processing message {msg.get('id', 'unknown')}: {str(e)}")
                    continue
            
            if message_data:
                try:
                    # Use bulk insert for better performance with correct schema
                    conn.executemany('''
                        INSERT OR REPLACE INTO messages (
                            id, content, timestamp, edited_timestamp, jump_url,
                            author_id, author_name, author_display_name, author_is_bot,
                            channel_id, channel_name, channel_type,
                            guild_id, guild_name,
                            mentions, mention_everyone, mention_roles,
                            referenced_message_id, attachments, embeds, reactions,
                            emoji_stats, pinned, flags, nonce, type, is_system,
                            mentions_everyone, has_reactions
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', [(
                        msg.get('message_id', ''),
                        msg.get('content', ''),
                        msg.get('timestamp', ''),
                        msg.get('edited_timestamp'),
                        msg.get('jump_url', ''),
                        msg.get('author_id', ''),
                        msg.get('author_name', ''),
                        msg.get('author_display_name', ''),
                        msg.get('is_bot', False),
                        msg.get('channel_id', ''),
                        msg.get('channel_name', ''),
                        msg.get('channel_type', ''),
                        msg.get('guild_id', ''),
                        msg.get('guild_name', ''),
                        json.dumps([]), # mentions - placeholder
                        False, # mention_everyone - placeholder
                        json.dumps([]), # mention_roles - placeholder
                        msg.get('reference_message_id'),
                        json.dumps(msg.get('attachments', [])),
                        json.dumps(msg.get('embeds', [])),
                        json.dumps(msg.get('reactions', [])),
                        json.dumps({}), # emoji_stats - placeholder
                        False, # pinned - placeholder
                        None, # flags - placeholder
                        None, # nonce - placeholder
                        '', # type - placeholder
                        False, # is_system - placeholder
                        False, # mentions_everyone - placeholder
                        msg.get('has_reactions', False)
                    ) for msg in message_data])
                    
                    conn.commit()
                    logger.info(f"✅ Saved batch of {len(message_data)} messages")
                except Exception as e:
                    logger.error(f"Error saving batch: {str(e)}")
                    conn.rollback()
                    
    except Exception as e:
        logger.error(f"Error in save_messages_to_db: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'conn' in locals():
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

def save_channel_members_to_db(messages_data, db_path='discord_messages.db'):
    """Save channel members data to database"""
    if not messages_data:
        return
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Clear existing channel members data to avoid stale data
        cursor.execute('DELETE FROM channel_members')
        logger.info("Cleared existing channel members data")
        
        # Extract all channel members from the data
        all_members = []
        for guild_name, channels in messages_data.items():
            if '_channel_members' in channels:
                for channel_name, members in channels['_channel_members'].items():
                    all_members.extend(members)
        
        if not all_members:
            logger.info("No channel members to save")
            conn.close()
            return
            
        logger.info(f"Saving {len(all_members)} channel member records to database...")
        
        # Process members in batches
        batch_size = 100
        for i in range(0, len(all_members), batch_size):
            batch = all_members[i:i + batch_size]
            
            try:
                cursor.executemany('''
                    INSERT OR REPLACE INTO channel_members (
                        channel_id, channel_name, guild_id, guild_name,
                        user_id, user_name, user_display_name, user_joined_at,
                        user_roles, is_bot, member_permissions, synced_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', [(
                    member.get('channel_id', ''),
                    member.get('channel_name', ''),
                    member.get('guild_id', ''),
                    member.get('guild_name', ''),
                    member.get('user_id', ''),
                    member.get('user_name', ''),
                    member.get('user_display_name', ''),
                    member.get('user_joined_at'),
                    member.get('user_roles', '[]'),
                    member.get('is_bot', False),
                    member.get('member_permissions', '{}'),
                    member.get('synced_at', datetime.now(timezone.utc).isoformat())
                ) for member in batch])
                
                conn.commit()
                logger.info(f"✅ Saved batch of {len(batch)} channel member records")
                
            except Exception as e:
                logger.error(f"Error saving channel members batch: {str(e)}")
                conn.rollback()
                
    except Exception as e:
        logger.error(f"Error in save_channel_members_to_db: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'conn' in locals():
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
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
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
        save_messages_to_db(client.new_data, 'discord_messages.db')
        print("👥 Saving channel members to database...")
        save_channel_members_to_db(client.new_data, 'discord_messages.db')
        save_sync_log_to_db(client.sync_log_entry)
        print("\n📂 Data updated in database.")
        print(f"📝 Total messages synced: {client.sync_log_entry['total_messages_synced']}")
    else:
        print("\n✅ No new messages found.")

if __name__ == "__main__":
    print("🗑️ Full refresh mode - clearing database and fetch state...")
    
    # Always clear database
    try:
        conn = sqlite3.connect('discord_messages.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM messages')
        cursor.execute('DELETE FROM sqlite_sequence WHERE name="messages"')
        conn.commit()
        conn.close()
        print("✅ Database cleared")
    except Exception as e:
        print(f"❌ Error clearing database: {e}")
    
    # Always clear fetch state
    try:
        import os
        if os.path.exists('data/fetch_state.json'):
            os.remove('data/fetch_state.json')
        print("✅ Fetch state cleared")
    except Exception as e:
        print(f"❌ Error clearing fetch state: {e}")
    
    print("🚀 Starting full re-fetch from beginning...")
    update_discord_messages()
    print("🔌 Disconnecting from Discord...")
