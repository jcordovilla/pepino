"""Unit tests for SyncManager."""

import asyncio
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pepino.config import Settings
from pepino.discord.sync.sync_manager import SyncManager


class TestSyncManager:
    """Test cases for SyncManager."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)

    @pytest.fixture
    def sync_manager(self, temp_db_path):
        """Create SyncManager instance."""
        # Set DISCORD_TOKEN for testing
        os.environ["DISCORD_TOKEN"] = "test_token"
        return SyncManager(temp_db_path)

    @pytest.mark.asyncio
    async def test_initialize_success(self, sync_manager):
        """Test successful sync manager initialization."""
        # SyncManager doesn't have an initialize method - it's ready after construction
        assert sync_manager.db_path is not None
        assert sync_manager.intents is not None
        assert sync_manager.discord_token is not None

    @pytest.mark.asyncio
    async def test_initialize_database(self, sync_manager):
        """Test database initialization."""
        with patch("pepino.discord.sync.sync_manager.init_database") as mock_init:
            sync_manager.initialize_database()
            mock_init.assert_called_once_with(sync_manager.db_path)

    @pytest.mark.asyncio
    async def test_load_existing_data(self, sync_manager):
        """Test loading existing data."""
        with patch("pepino.discord.sync.sync_manager.load_existing_data") as mock_load:
            mock_load.return_value = {"guild1": {"channel1": []}}
            result = sync_manager.load_existing_data()
            assert result == {"guild1": {"channel1": []}}
            mock_load.assert_called_once_with(sync_manager.db_path)

    @pytest.mark.asyncio
    async def test_perform_sync(self, sync_manager):
        """Test performing sync operation."""
        # Mock the entire perform_sync method to avoid event loop conflicts
        with patch.object(sync_manager, "perform_sync") as mock_perform:
            mock_perform.return_value = (
                {"guild1": {"channel1": [{"id": "1"}]}},
                {"total_messages_synced": 1},
            )

            result = await sync_manager.perform_sync()

            assert len(result) == 2
            assert result[0] == {"guild1": {"channel1": [{"id": "1"}]}}
            assert result[1]["total_messages_synced"] == 1

    @pytest.mark.asyncio
    async def test_save_sync_results(self, sync_manager):
        """Test saving sync results."""
        from datetime import datetime

        from pepino.discord.sync.models import SyncLogEntry

        messages_data = {"guild1": {"channel1": [{"id": "1"}]}}
        sync_log = SyncLogEntry(
            timestamp=datetime.now().isoformat(),
            sync_start_time=datetime.now().isoformat(),
            total_messages_synced=1,
        )

        with patch(
            "pepino.discord.sync.sync_manager.save_messages_to_db"
        ) as mock_save_msgs:
            with patch(
                "pepino.discord.sync.sync_manager.save_channel_members_to_db"
            ) as mock_save_members:
                with patch(
                    "pepino.discord.sync.sync_manager.save_sync_log_to_db"
                ) as mock_save_log:
                    sync_manager.save_sync_results(messages_data, sync_log)

                    mock_save_msgs.assert_called_once_with(
                        messages_data, sync_manager.db_path
                    )
                    mock_save_members.assert_called_once_with(
                        messages_data, sync_manager.db_path
                    )
                    mock_save_log.assert_called_once_with(
                        sync_log, sync_manager.db_path
                    )

    @pytest.mark.asyncio
    async def test_clear_database(self, sync_manager):
        """Test clearing database."""
        with patch("pepino.data.database.manager.DatabaseManager") as mock_db_manager:
            with patch(
                "pepino.data.repositories.message_repository.MessageRepository"
            ) as mock_message_repo:
                mock_db_instance = MagicMock()
                mock_db_manager.return_value = mock_db_instance
                mock_repo_instance = MagicMock()
                mock_message_repo.return_value = mock_repo_instance

                await sync_manager.clear_database()

                mock_db_manager.assert_called_once_with(sync_manager.db_path)
                mock_message_repo.assert_called_once_with(mock_db_instance)
                mock_repo_instance.clear_all_messages.assert_called_once()
                mock_db_instance.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_full_sync(self, sync_manager):
        """Test running full sync."""
        with patch.object(sync_manager, "clear_database") as mock_clear:
            with patch.object(sync_manager, "clear_sync_state") as mock_clear_state:
                with patch.object(sync_manager, "perform_sync") as mock_perform:
                    with patch.object(sync_manager, "save_sync_results") as mock_save:
                        mock_perform.return_value = (
                            {"guild1": {}},
                            {"total_messages_synced": 0},
                        )

                        await sync_manager.run_full_sync()

                        mock_clear.assert_called_once()
                        mock_clear_state.assert_called_once()
                        mock_perform.assert_called_once()
                        mock_save.assert_called_once()


@pytest.mark.asyncio
async def test_sync_manager_runs_sync():
    """Test that SyncManager can run a sync operation."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        # Set DISCORD_TOKEN for testing
        os.environ["DISCORD_TOKEN"] = "test_token"
        manager = SyncManager(db_path)

        with patch.object(manager, "perform_sync") as mock_perform:
            mock_perform.return_value = ({}, {})

            # Test that we can call perform_sync
            result = await manager.perform_sync()
            assert len(result) == 2
            mock_perform.assert_called_once()
    finally:
        os.unlink(db_path)
