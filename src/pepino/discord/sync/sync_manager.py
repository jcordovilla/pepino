"""
Sync manager for orchestrating Discord data synchronization.
"""

import asyncio
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import discord

from pepino.config import Settings
from pepino.data.database.manager import DatabaseManager
from pepino.data.database.schema import init_database
from pepino.data.models.sync import SyncLogEntry
from pepino.data.repositories import MessageRepository, SyncRepository
from pepino.logging_config import get_logger

from ..data.persistence import (
    _load_existing_data_async,
    _save_channel_members_to_db_async,
    _save_messages_to_db_async,
    _save_sync_log_to_db_async,
)
from .discord_client import DiscordClient
from .models import FullSyncResult, IncrementalSyncResult

logger = get_logger(__name__)


class SyncManager:
    """Manages Discord data synchronization operations"""

    def __init__(self, db_path: Optional[str] = None):
        self.settings = Settings()
        self.db_path = db_path or self.settings.db_path
        self.discord_token = self.settings.discord_token

        # Configure Discord intents
        self.intents = discord.Intents.default()
        self.intents.message_content = self.settings.message_content_intent
        self.intents.guilds = True
        self.intents.messages = True
        self.intents.reactions = True
        self.intents.members = self.settings.members_intent

    def initialize_database(self) -> None:
        """Initialize the database schema"""
        logger.info("🗄️ Initializing database...")
        init_database(self.db_path)

    async def load_existing_data(self) -> Dict[str, Any]:
        """Load existing data from database"""
        return await _load_existing_data_async(self.db_path)

    async def get_last_sync_time(self) -> Optional[datetime]:
        """Get the timestamp of the last successful sync"""
        try:
            db_manager = DatabaseManager(self.db_path)
            sync_repo = SyncRepository(db_manager)

            try:
                last_sync = sync_repo.get_last_sync_log()
                if last_sync and last_sync.get("completed_at"):
                    return datetime.fromisoformat(
                        last_sync["completed_at"].replace("Z", "+00:00")
                    )
                return None
            finally:
                db_manager.close_connections()
        except Exception as e:
            logger.warning(f"Could not get last sync time: {e}")
            return None

    async def is_data_stale(self, max_age_hours: Optional[int] = None) -> bool:
        """Check if data is older than the staleness threshold"""
        max_age_hours = max_age_hours or self.settings.auto_sync_threshold_hours
        last_sync = await self.get_last_sync_time()

        if not last_sync:
            return True  # No sync found, data is stale

        threshold = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        return last_sync < threshold

    async def detect_new_activity(self) -> bool:
        """Detect if there might be new activity (simplified heuristic)"""
        # This is a simplified version - could be enhanced with Discord API checks
        # For now, we'll use data staleness as a proxy for new activity
        return await self.is_data_stale(1)  # Consider 1 hour as "recent activity"

    def _get_latest_message_ids(self) -> Dict[str, str]:
        """Get latest message IDs per channel for incremental sync."""
        try:
            db_manager = DatabaseManager(self.db_path)
            message_repo = MessageRepository(db_manager)
            try:
                return message_repo.get_latest_message_id_per_channel()
            finally:
                db_manager.close_connections()
        except Exception as e:
            logger.warning(f"Could not get latest message IDs: {e}")
            return {}

    async def perform_sync(
        self, incremental: bool = True
    ) -> tuple[Dict[str, Any], SyncLogEntry]:
        """Perform a Discord data sync.

        Args:
            incremental: If True, only fetch messages newer than the last synced message
                        per channel. If False, fetch all messages.
        """
        logger.info("🔌 Connecting to Discord...")

        # Initialize database
        self.initialize_database()

        # Load existing data
        data_store = await self.load_existing_data()

        # Get latest message IDs for incremental sync
        latest_message_ids = self._get_latest_message_ids() if incremental else {}
        if latest_message_ids:
            logger.info(
                f"📊 Found {len(latest_message_ids)} channels with existing messages (incremental sync)"
            )
        else:
            logger.info("📊 No existing messages found (full sync)")

        # Create and run client
        client = DiscordClient(
            data_store=data_store,
            latest_message_ids=latest_message_ids,
            intents=self.intents,
        )
        try:
            await client.start(self.discord_token)
            return client.new_data, client.get_sync_log()
        finally:
            # Ensure client is properly closed
            try:
                if not client.is_closed():
                    await client.close()
            except Exception as e:
                logger.warning(f"Error closing Discord client: {e}")
            finally:
                # Force cleanup of any remaining resources
                import gc

                gc.collect()

    async def run_incremental_sync(self, force: bool = False) -> IncrementalSyncResult:
        """Run an incremental sync operation.

        Only fetches messages newer than the last synced message per channel.
        This is much faster than a full sync when the database already has data.
        """
        start_time = time.time()
        logger.info("🔄 Starting incremental sync...")

        # Check if sync is needed
        if not force and not await self.is_data_stale():
            logger.info("✅ Data is fresh, no sync needed")
            return IncrementalSyncResult(
                sync_performed=False,
                reason="Data is fresh",
                duration=0,
                last_sync=await self.get_last_sync_time(),
            )

        try:
            # Perform the sync
            messages_data, sync_log = await self.perform_sync()

            # Save results
            await self.save_sync_results(messages_data, sync_log)

            duration = time.time() - start_time
            new_messages = sync_log.total_messages_synced
            updated_channels = len(sync_log.guilds_synced)

            logger.info(f"✅ Incremental sync completed in {duration:.1f}s")
            logger.info(
                f"📝 New messages: {new_messages}, Updated channels: {updated_channels}"
            )

            return IncrementalSyncResult(
                sync_performed=True,
                new_messages=new_messages,
                updated_channels=updated_channels,
                duration=duration,
                sync_log=sync_log,
                last_sync=datetime.now(timezone.utc),
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ Incremental sync failed after {duration:.1f}s: {e}")

            return IncrementalSyncResult(
                sync_performed=False, error=str(e), duration=duration
            )

    async def save_sync_results(
        self, messages_data: Dict[str, Any], sync_log: SyncLogEntry
    ) -> None:
        """Save sync results to database"""
        if messages_data:
            logger.info("💾 Saving messages to database...")
            await _save_messages_to_db_async(messages_data, self.db_path)

            logger.info("👥 Saving channel members to database...")
            await _save_channel_members_to_db_async(messages_data, self.db_path)

            # Save sync log synchronously
            db_manager = DatabaseManager(self.db_path)
            sync_repo = SyncRepository(db_manager)
            try:
                sync_repo.save_sync_log(sync_log)
            finally:
                db_manager.close_connections()

            logger.info("\n📂 Data updated in database.")
            logger.info(f"📝 Total messages synced: {sync_log.total_messages_synced}")
        else:
            logger.info("\n✅ No new messages found.")

    async def clear_database(self) -> None:
        """Clear the database for a fresh start using repository"""
        logger.info("🗑️ Clearing database for fresh start...")

        try:
            db_manager = DatabaseManager(self.db_path)
            message_repo = MessageRepository(db_manager)

            try:
                await message_repo.clear_all_messages()
            finally:
                db_manager.close_connections()

        except Exception as e:
            logger.error(f"❌ Error clearing database: {e}")

    def clear_sync_state(self) -> None:
        """Clear sync state files"""
        logger.info("🗑️ Clearing sync state...")

        try:
            if os.path.exists("data/sync_state.json"):
                os.remove("data/sync_state.json")
            logger.info("✅ Sync state cleared")
        except Exception as e:
            logger.error(f"❌ Error clearing sync state: {e}")

    async def run_full_sync(self, clear_existing: bool = True) -> FullSyncResult:
        """Run a complete sync operation (fetches all messages, ignoring existing data)"""
        start_time = time.time()

        if clear_existing:
            await self.clear_database()
            self.clear_sync_state()
            logger.info("🚀 Starting full re-sync from beginning...")

        try:
            # Perform sync with incremental=False to fetch all messages
            messages_data, sync_log = await self.perform_sync(incremental=False)

            # Save results
            await self.save_sync_results(messages_data, sync_log)

            duration = time.time() - start_time
            new_messages = sync_log.total_messages_synced

            logger.info(f"✅ Full sync completed in {duration:.1f}s")
            logger.info("🔌 Disconnecting from Discord...")

            return FullSyncResult(
                sync_performed=True,
                new_messages=new_messages,
                duration=duration,
                sync_log=sync_log,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ Full sync failed after {duration:.1f}s: {e}")

            return FullSyncResult(sync_performed=False, error=str(e), duration=duration)


def update_discord_messages() -> None:
    """Main function to update Discord messages (legacy compatibility)"""
    sync_manager = SyncManager()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(sync_manager.run_full_sync())
