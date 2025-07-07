"""
Main Discord client class for synchronizing message data.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import discord

from pepino.data_operations.discord_sync.extractors import MessageExtractor
from pepino.data_operations.discord_sync.models import SyncLogEntry, GuildSyncInfo, ChannelSkipInfo, SyncError
from pepino.logging_config import get_logger

logger = get_logger(__name__)


class DiscordClient(discord.Client):
    """Main Discord client for synchronizing message data"""

    def __init__(self, data_store: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.data_store = data_store
        self.new_data = {}
        self.rate_limit_delay = 0.1  # Minimal delay between channels (100ms)
        self.max_retries = 3  # Maximum number of retries for failed requests
        
        # Initialize sync log entry
        now = datetime.now(timezone.utc).isoformat()
        self.sync_log_entry = SyncLogEntry(timestamp=now, sync_start_time=now)
        
        # Log sync start
        logger.info("Discord sync operation started")

    async def on_ready(self):
        """Called when the bot is ready and connected to Discord"""
        logger.info(f"✅ Logged in as: {self.user} (ID: {self.user.id})")
        sync_start_time = datetime.now(timezone.utc)

        for guild in self.guilds:
            logger.info(f"\n📂 Guild: {guild.name} (ID: {guild.id})")
            self.new_data.setdefault(guild.name, {})
            
            # Add guild to sync log
            guild_sync = GuildSyncInfo(
                guild_name=guild.name,
                guild_id=str(guild.id),
                sync_start_time=datetime.now(timezone.utc).isoformat(),
            )
            self.sync_log_entry.guilds_synced.append(guild_sync)
            logger.info(f"Starting guild sync: {guild.name} (ID: {guild.id})")

            accessible_channels = []
            inaccessible_channels = []
            for channel in guild.text_channels:
                perms = channel.permissions_for(guild.me)
                if perms.read_messages and perms.read_message_history:
                    accessible_channels.append(channel)
                else:
                    inaccessible_channels.append(channel)

            logger.info(
                f"    Accessible channels: {[f'#{c.name}' for c in accessible_channels]}"
            )
            logger.info(
                f"    Inaccessible channels: {[f'#{c.name}' for c in inaccessible_channels]}"
            )

            for channel in inaccessible_channels:
                logger.warning(
                    f"  🚫 Skipped channel #{channel.name}: insufficient permissions (pre-check)"
                )
                
                # Add channel skip to sync log
                skip_info = ChannelSkipInfo(
                    guild_name=guild.name,
                    channel_name=channel.name,
                    channel_id=str(channel.id),
                    reason="Forbidden - missing read permissions (pre-check)",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                self.sync_log_entry.channels_skipped.append(skip_info)
                logger.warning(f"Skipped channel: #{channel.name} in {guild.name} - Forbidden - missing read permissions (pre-check)")

            for channel in accessible_channels:
                logger.info(f"  📄 Channel: #{channel.name} (ID: {channel.id})")

                try:
                    logger.info(f"    🆕 Full sync from beginning of channel history")

                    # Always sync all messages (no incremental logic)
                    new_messages = await self.sync_with_retry(channel, None)

                    # Flatten message dicts for DB compatibility
                    def flatten_message(msg):
                        import json
                        
                        # Copy top-level fields from nested dicts
                        flat = dict(msg)
                        if isinstance(msg.get("author"), dict):
                            for k, v in msg["author"].items():
                                flat[f"author_{k}"] = v
                            # Remove the original nested dict
                            del flat["author"]
                        if isinstance(msg.get("guild"), dict):
                            for k, v in msg["guild"].items():
                                flat[f"guild_{k}"] = v
                            # Remove the original nested dict
                            del flat["guild"]
                        if isinstance(msg.get("channel"), dict):
                            for k, v in msg["channel"].items():
                                flat[f"channel_{k}"] = v
                            # Remove the original nested dict
                            del flat["channel"]
                        if isinstance(msg.get("user_presence"), dict):
                            for k, v in msg["user_presence"].items():
                                flat[f"author_{k}"] = v
                            # Remove the original nested dict
                            del flat["user_presence"]
                        
                        # Convert list fields to JSON strings for database compatibility
                        list_fields = [
                            "mentions", "mention_roles", "mention_channels", 
                            "attachments", "embeds", "reactions", "components", 
                            "stickers", "emoji_stats", "guild_features", "channel_overwrites"
                        ]
                        for field in list_fields:
                            if field in flat and isinstance(flat[field], list):
                                flat[field] = json.dumps(flat[field])
                        
                        # Convert dict fields to JSON strings for database compatibility
                        dict_fields = [
                            "author_activity", "author_status", "author_desktop_status", 
                            "author_mobile_status", "author_web_status", "thread", 
                            "application", "activity", "interaction", 
                            "message_reference", "allowed_mentions", "emoji_stats"
                        ]
                        for field in dict_fields:
                            if field in flat and isinstance(flat[field], dict):
                                flat[field] = json.dumps(flat[field])
                        
                        return flat

                    # Apply flattening
                    flattened_messages = [flatten_message(m) for m in new_messages]
                    
                    if flattened_messages:
                        logger.info(f"    ✅ {len(flattened_messages)} message(s) synced")
                        self.sync_log_entry.total_messages_synced += len(flattened_messages)
                        logger.info(f"Synced {len(flattened_messages)} messages")
                        # Flatten before saving
                        self.new_data[guild.name].setdefault(channel.name, []).extend(flattened_messages)
                    else:
                        logger.info(f"    📫 No messages found")

                    # Sync channel members
                    logger.info(f"    👥 Syncing channel members...")
                    channel_members = await self._sync_channel_members(channel)

                    if channel_members:
                        logger.info(f"    ✅ {len(channel_members)} member(s) synced")
                        # Store channel members in new_data for saving to DB
                        self.new_data[guild.name].setdefault("_channel_members", {})
                        self.new_data[guild.name]["_channel_members"][
                            channel.name
                        ] = channel_members
                    else:
                        logger.info(f"    👥 No accessible members found")

                except Exception as e:
                    logger.error(
                        f"    ❌ Error syncing channel #{channel.name}: {str(e)}"
                    )
                    
                    # Add error to sync log
                    error_info = SyncError(
                        error=str(e),
                        guild_name=guild.name,
                        channel_name=channel.name,
                        channel_id=str(channel.id),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                    self.sync_log_entry.errors.append(error_info)
                    logger.error(f"Sync error in {guild.name} channel #{channel.name}: {str(e)}")
                    continue

            # Mark guild sync as complete
            for guild_sync in self.sync_log_entry.guilds_synced:
                if guild_sync.guild_name == guild.name:
                    guild_sync.sync_end_time = datetime.now(timezone.utc).isoformat()
                    break
            logger.info(f"Completed guild sync: {guild.name}")

        # Finalize sync log
        sync_end_time = datetime.now(timezone.utc)
        self.sync_log_entry.sync_end_time = sync_end_time.isoformat()

        # Calculate duration
        sync_start_time = datetime.fromisoformat(self.sync_log_entry.sync_start_time)
        self.sync_log_entry.sync_duration_seconds = (
            sync_end_time - sync_start_time
        ).total_seconds()
        
        # Log sync completion with summary
        total_messages = self.sync_log_entry.total_messages_synced
        duration = self.sync_log_entry.sync_duration_seconds
        logger.info(f"Discord sync operation completed - {total_messages} messages in {duration:.1f}s")

        await self.close()

    async def sync_with_retry(
        self, channel, last_message_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Sync messages with retry logic, only sleep after each channel."""
        for attempt in range(self.max_retries):
            try:
                messages = []
                message_count = 0
                last_progress_time = time.time()

                async for message in channel.history(
                    limit=None,
                    after=discord.Object(id=last_message_id) if last_message_id else None,
                    oldest_first=True,
                ):
                    message_count += 1
                    current_time = time.time()
                    # Show progress every 100 messages or every 5 seconds
                    if (
                        message_count % 100 == 0
                        or (current_time - last_progress_time) >= 5
                    ):
                        logger.info(
                            f"      Synced {message_count} messages from #{channel.name}"
                        )
                        last_progress_time = current_time
                    messages.append(MessageExtractor.extract_message_data(message))
                # Only sleep after the channel is done
                await asyncio.sleep(self.rate_limit_delay)
                return messages
            except discord.Forbidden:
                logger.error(
                    f"Insufficient permissions to sync messages from #{channel.name}"
                )
                raise
            except discord.HTTPException as e:
                if attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    logger.warning(
                        f"HTTP error (attempt {attempt + 1}/{self.max_retries}): {e}. Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed to sync messages after {self.max_retries} attempts"
                    )
                    raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise

    async def _sync_channel_members(
        self, channel: discord.TextChannel
    ) -> List[Dict[str, Any]]:
        """Sync all members who have access to a specific channel"""
        try:
            members_data = []
            member_count = 0

            logger.info(f"      Syncing members for #{channel.name}")

            for member in channel.guild.members:
                # Check if member can read this channel
                perms = channel.permissions_for(member)
                if perms.read_messages:
                    member_count += 1

                    # Get member roles
                    roles = []
                    for role in member.roles:
                        if role.name != "@everyone":  # Skip @everyone role
                            roles.append(
                                {
                                    "id": str(role.id),
                                    "name": role.name,
                                    "color": role.color.value,
                                    "position": role.position,
                                    "permissions": role.permissions.value,
                                }
                            )

                    # Get member permissions for this channel
                    member_perms = {
                        "read_messages": perms.read_messages,
                        "send_messages": perms.send_messages,
                        "manage_messages": perms.manage_messages,
                        "read_message_history": perms.read_message_history,
                        "add_reactions": perms.add_reactions,
                        "attach_files": perms.attach_files,
                        "embed_links": perms.embed_links,
                        "mention_everyone": perms.mention_everyone,
                    }

                    member_data = {
                        "channel_id": str(channel.id),
                        "channel_name": channel.name,
                        "guild_id": str(channel.guild.id),
                        "guild_name": channel.guild.name,
                        "user_id": str(member.id),
                        "user_name": member.name,
                        "user_display_name": member.display_name,
                        "user_joined_at": member.joined_at.isoformat()
                        if member.joined_at
                        else None,
                        "user_roles": json.dumps(roles),
                        "is_bot": member.bot,
                        "member_permissions": json.dumps(member_perms),
                        "synced_at": datetime.now(timezone.utc).isoformat(),
                    }

                    members_data.append(member_data)

            logger.info(
                f"      Found {member_count} members with access to #{channel.name}"
            )
            return members_data

        except Exception as e:
            logger.error(f"Error syncing members for #{channel.name}: {e}")
            return []

    def get_sync_log(self) -> "SyncLogEntry":
        """Get the sync log entry"""
        return self.sync_log_entry

    async def close(self):
        """Override close method to ensure proper cleanup of HTTP session"""
        try:
            # Close the discord client
            await super().close()
        except Exception as e:
            logger.warning(f"Error during discord client close: {e}")
        
        # Ensure HTTP session is properly closed
        try:
            if hasattr(self, 'http') and hasattr(self.http, 'session'):
                await self.http.session.close()
        except Exception as e:
            logger.warning(f"Error closing HTTP session: {e}")
        
        # Force cleanup of any remaining connections
        try:
            import asyncio
            # Give a moment for connections to close
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.warning(f"Error during connection cleanup: {e}")
