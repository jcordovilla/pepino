"""
Main Discord client class for synchronizing message data.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import discord

from ..data import SyncLogger
from ..extractors import MessageExtractor

logger = logging.getLogger(__name__)


class DiscordClient(discord.Client):
    """Main Discord client for synchronizing message data"""

    def __init__(self, data_store: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.data_store = data_store
        self.new_data = {}
        self.rate_limit_delay = 0.1  # Minimal delay between channels (100ms)
        self.max_retries = 3  # Maximum number of retries for failed requests
        self.sync_logger = SyncLogger()

    async def on_ready(self):
        """Called when the bot is ready and connected to Discord"""
        logger.info(f"✅ Logged in as: {self.user} (ID: {self.user.id})")
        sync_start_time = datetime.now(timezone.utc)

        for guild in self.guilds:
            logger.info(f"\n📂 Guild: {guild.name} (ID: {guild.id})")
            self.new_data.setdefault(guild.name, {})
            self.sync_logger.add_guild_sync(guild.name, str(guild.id))

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
                self.sync_logger.add_channel_skip(
                    guild.name,
                    channel.name,
                    str(channel.id),
                    "Forbidden - missing read permissions (pre-check)",
                )

            for channel in accessible_channels:
                logger.info(f"  📄 Channel: #{channel.name} (ID: {channel.id})")

                try:
                    logger.info(f"    🆕 Full sync from beginning of channel history")

                    # Always sync all messages (no incremental logic)
                    new_messages = await self.sync_with_retry(channel, None)

                    if new_messages:
                        logger.info(f"    ✅ {len(new_messages)} message(s) synced")
                        self.sync_logger.add_messages_synced(len(new_messages))
                        self.new_data[guild.name].setdefault(channel.name, []).extend(
                            new_messages
                        )
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
                    self.sync_logger.add_error(
                        str(e), guild.name, channel.name, str(channel.id)
                    )
                    continue

            self.sync_logger.complete_guild_sync(guild.name)

        sync_end_time = datetime.now(timezone.utc)
        self.sync_logger.finalize_sync()

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
                    after=discord.Object(id=last_message_id)
                    if last_message_id
                    else None,
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
        return self.sync_logger.get_log_entry()
