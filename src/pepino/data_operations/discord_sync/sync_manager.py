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
from pepino.data.repositories import MessageRepository, SyncRepository, ChannelRepository, UserRepository
from pepino.logging_config import get_logger

from .discord_client import DiscordClient
from .models import FullSyncResult, IncrementalSyncResult

logger = get_logger(__name__)


class SyncManager:
    """Manages Discord data synchronization operations"""

    def __init__(self, db_path: Optional[str] = None):
        self.settings = Settings()
        self.db_path = db_path or self.settings.db_path
        self.discord_token = self.settings.discord_token
        self._db_manager = None

        # Configure Discord intents
        self.intents = discord.Intents.default()
        self.intents.message_content = self.settings.message_content_intent
        self.intents.guilds = True
        self.intents.messages = True
        self.intents.reactions = True
        self.intents.members = self.settings.members_intent

    @property
    def db_manager(self) -> DatabaseManager:
        """Get database manager instance (lazy initialization)."""
        if self._db_manager is None:
            self._db_manager = DatabaseManager(self.db_path)
        return self._db_manager

    def initialize_database(self) -> None:
        """Initialize the database schema"""
        logger.info("🗄️ Initializing database...")
        init_database(self.db_path)

    def load_existing_data(self) -> Dict[str, Any]:
        """Load existing data from database using repository directly"""
        try:
            message_repo = MessageRepository(self.db_manager)
            return message_repo.load_existing_data()
        except Exception as e:
            logger.warning(f"Could not load existing data: {e}")
            return {}

    def get_last_sync_time(self) -> Optional[datetime]:
        """Get the timestamp of the last successful sync using repository directly"""
        try:
            sync_repo = SyncRepository(self.db_manager)
            last_sync = sync_repo.get_last_sync_log()
            if last_sync and last_sync.get("completed_at"):
                return datetime.fromisoformat(
                    last_sync["completed_at"].replace("Z", "+00:00")
                )
            return None
        except Exception as e:
            logger.warning(f"Could not get last sync time: {e}")
            return None

    def is_data_stale(self, max_age_hours: Optional[int] = None) -> bool:
        """Check if data is older than the staleness threshold"""
        max_age_hours = max_age_hours or self.settings.auto_sync_threshold_hours
        last_sync = self.get_last_sync_time()

        if not last_sync:
            return True  # No sync found, data is stale

        threshold = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        return last_sync < threshold

    def detect_new_activity(self) -> bool:
        """Detect if there might be new activity (simplified heuristic)"""
        # This is a simplified version - could be enhanced with Discord API checks
        # For now, we'll use data staleness as a proxy for new activity
        return self.is_data_stale(1)  # Consider 1 hour as "recent activity"

    async def perform_sync(self) -> tuple[Dict[str, Any], SyncLogEntry]:
        """Perform a complete Discord data sync"""
        logger.info("🔌 Connecting to Discord...")

        # Check if Discord token is available
        if not self.discord_token:
            raise ValueError("Discord token not configured. Please set DISCORD_TOKEN environment variable.")

        # Initialize database
        self.initialize_database()

        # Load existing data
        data_store = self.load_existing_data()

        # Create and run client
        client = DiscordClient(data_store=data_store, intents=self.intents)
        try:
            await client.start(self.discord_token)
            return client.new_data, client.get_sync_log()
        finally:
            # Ensure proper cleanup
            await self._cleanup_client(client)

    async def _cleanup_client(self, client: DiscordClient):
        """Ensure proper cleanup of Discord client and HTTP session"""
        try:
            # Close the client (this will also close the HTTP session)
            await client.close()
        except Exception as e:
            logger.warning(f"Error during client cleanup: {e}")
        
        # Additional cleanup for any remaining aiohttp connections
        try:
            import asyncio
            # Force cleanup of any remaining tasks
            pending = asyncio.all_tasks()
            for task in pending:
                if not task.done() and task != asyncio.current_task():
                    task.cancel()
            
            # Give a moment for cleanup
            await asyncio.sleep(0.1)
        except Exception as e:
            logger.warning(f"Error during additional cleanup: {e}")

    async def run_incremental_sync(self, force: bool = False) -> IncrementalSyncResult:
        """Run an incremental sync operation"""
        start_time = time.time()
        logger.info("🔄 Starting incremental sync...")

        # Check if sync is needed
        if not force and not self.is_data_stale():
            logger.info("✅ Data is fresh, no sync needed")
            return IncrementalSyncResult(
                sync_performed=False,
                reason="Data is fresh",
                duration=0,
                last_sync=self.get_last_sync_time(),
            )

        try:
            # Perform the sync
            messages_data, sync_log = await self.perform_sync()

            # Save results using repositories directly
            self.save_sync_results(messages_data, sync_log)

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

    def save_sync_results(
        self, messages_data: Dict[str, Any], sync_log: SyncLogEntry
    ) -> None:
        """Save sync results to database using repositories directly"""
        if messages_data:
            logger.info("💾 Saving messages to database...")
            message_repo = MessageRepository(self.db_manager)
            message_repo.bulk_insert_messages(messages_data)

            logger.info("👥 Saving channel members to database...")
            channel_repo = ChannelRepository(self.db_manager)
            channel_repo.save_channel_members(messages_data)

            logger.info("📝 Saving sync log to database...")
            sync_repo = SyncRepository(self.db_manager)
            sync_repo.save_sync_log(sync_log.model_dump())

            logger.info("\n📂 Data updated in database.")
            logger.info(f"📝 Total messages synced: {sync_log.total_messages_synced}")
        else:
            logger.info("\n✅ No new messages found.")

    def clear_database(self) -> None:
        """Clear the database for a fresh start using repository directly"""
        logger.info("🗑️ Clearing database for fresh start...")

        try:
            # Clear messages
            message_repo = MessageRepository(self.db_manager)
            message_repo.clear_all_messages()

            # Clear channels
            channel_repo = ChannelRepository(self.db_manager)
            channel_repo.clear_all_channels()

            # Clear users
            user_repo = UserRepository(self.db_manager)
            user_repo.clear_all_users()

            # Clear sync logs
            sync_repo = SyncRepository(self.db_manager)
            sync_repo.clear_all_sync_logs()

            logger.info("✅ Database cleared successfully")

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
        """Run a complete sync operation"""
        start_time = time.time()

        if clear_existing:
            self.clear_database()
            self.clear_sync_state()
            logger.info("🚀 Starting full re-sync from beginning...")

        try:
            # Perform sync
            messages_data, sync_log = await self.perform_sync()

            # Save results using repositories directly
            self.save_sync_results(messages_data, sync_log)

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

    def close(self):
        """Close database connections and clean up resources."""
        if self._db_manager:
            self._db_manager.close_connections()
        self._db_manager = None
        logger.debug("SyncManager closed")


def update_discord_messages() -> None:
    """Main function to update Discord messages (legacy compatibility)"""
    sync_manager = SyncManager()
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(sync_manager.run_full_sync())
    finally:
        sync_manager.close()
