"""
Database persistence operations for Discord data.
"""

import os
from typing import Any, Dict

from pepino.data.config import settings
from pepino.data.database.manager import DatabaseManager
from pepino.data.models.sync import SyncLogEntry
from pepino.data.repositories import (
    ChannelRepository,
    MessageRepository,
    SyncRepository,
)
from pepino.logging_config import get_logger

logger = get_logger(__name__)


async def _load_existing_data_async(db_path: str = settings.db_path) -> Dict[str, Any]:
    """Load existing message data from SQLite database"""
    if not os.path.exists(db_path):
        return {}

    db_manager = DatabaseManager(db_path)
    message_repo = MessageRepository(db_manager)

    try:
        return await message_repo.load_existing_data()
    finally:
        await db_manager.close()


async def _save_messages_to_db_async(
    messages_data: Dict[str, Any], db_path: str = settings.db_path
) -> None:
    """Save messages to database using repository pattern"""
    if not messages_data:
        return

    db_manager = DatabaseManager(db_path)
    message_repo = MessageRepository(db_manager)

    try:
        await message_repo.bulk_insert_messages(messages_data)
    finally:
        await db_manager.close()


async def _save_channel_members_to_db_async(
    messages_data: Dict[str, Any], db_path: str = settings.db_path
) -> None:
    """Save channel members data to database using repository pattern"""
    if not messages_data:
        return

    db_manager = DatabaseManager(db_path)
    channel_repo = ChannelRepository(db_manager)

    try:
        await channel_repo.save_channel_members(messages_data)
    finally:
        await db_manager.close()


async def _save_sync_log_to_db_async(
    sync_log_entry: SyncLogEntry, db_path: str = settings.db_path
) -> None:
    """Save sync log entry to database using repository pattern"""
    db_manager = DatabaseManager(db_path)
    sync_repo = SyncRepository(db_manager)

    try:
        await sync_repo.save_sync_log(sync_log_entry)
    finally:
        await db_manager.close()


# Legacy synchronous functions for backwards compatibility
def load_existing_data_sync(db_path: str = settings.db_path) -> Dict[str, Any]:
    """Synchronous wrapper for load_existing_data"""
    import asyncio

    return asyncio.run(_load_existing_data_async(db_path))


def save_messages_to_db_sync(
    messages_data: Dict[str, Any], db_path: str = settings.db_path
) -> None:
    """Synchronous wrapper for save_messages_to_db"""
    import asyncio

    asyncio.run(_save_messages_to_db_async(messages_data, db_path))


def save_channel_members_to_db_sync(
    messages_data: Dict[str, Any], db_path: str = settings.db_path
) -> None:
    """Synchronous wrapper for save_channel_members_to_db"""
    import asyncio

    asyncio.run(_save_channel_members_to_db_async(messages_data, db_path))


def save_sync_log_to_db_sync(
    sync_log_entry: SyncLogEntry, db_path: str = settings.db_path
) -> None:
    """Synchronous wrapper for save_sync_log_to_db"""
    import asyncio

    asyncio.run(_save_sync_log_to_db_async(sync_log_entry, db_path))


# Maintain backwards compatibility with original function names
# Map the public API to the sync versions for backwards compatibility
load_existing_data = load_existing_data_sync
save_messages_to_db = save_messages_to_db_sync
save_channel_members_to_db = save_channel_members_to_db_sync
save_sync_log_to_db = save_sync_log_to_db_sync
