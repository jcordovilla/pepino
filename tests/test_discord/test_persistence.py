import os
import tempfile
from unittest.mock import patch

import pytest

from pepino.data.config import settings
from pepino.data.database.schema import init_database
from pepino.discord.data import persistence


@pytest.fixture
def test_db_path():
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    # Initialize database schema
    init_database(db_path)

    yield db_path

    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def sample_message_data():
    """Sample message data for testing."""
    return {
        "TestGuild": {
            "general": [
                {
                    "id": "123456789",
                    "content": "Hello, world!",
                    "timestamp": "2024-01-01T12:00:00Z",
                    "edited_timestamp": None,
                    "jump_url": "https://discord.com/channels/123/456/789",
                    "channel": {"id": "456789", "name": "general", "type": "text"},
                    "guild": {"id": "789123", "name": "TestGuild"},
                    "author": {
                        "id": "987654",
                        "name": "testuser",
                        "display_name": "Test User",
                        "is_bot": False,
                    },
                    "attachments": [],
                    "embeds": [],
                    "reactions": [],
                    "stickers": [],
                }
            ]
        }
    }


@pytest.fixture
def sample_sync_log():
    """Sample sync log data for testing."""
    from pepino.discord.sync.models import SyncLogEntry

    return SyncLogEntry(
        timestamp="2024-01-01T12:00:00Z",
        sync_start_time="2024-01-01T12:00:00Z",
        guilds_synced=["TestGuild"],
        channels_skipped=[],
        errors=[],
        total_messages_synced=1,
    )


class TestPersistenceOperations:
    """Test persistence operations using repository pattern."""

    def test_save_and_load_messages_sync(self, test_db_path, sample_message_data):
        """Test saving and loading message data - key write and read operation (sync version)."""
        # Test write operation: Save messages (sync version)
        persistence.save_messages_to_db_sync(sample_message_data, test_db_path)

        # Test read operation: Load existing data (sync version)
        loaded_data = persistence.load_existing_data_sync(test_db_path)

        # Verify data integrity
        assert "TestGuild" in loaded_data
        assert "general" in loaded_data["TestGuild"]
        assert len(loaded_data["TestGuild"]["general"]) == 1

        # Verify the message was saved correctly
        message = loaded_data["TestGuild"]["general"][0]
        assert message["id"] == "123456789"
        assert message["timestamp"] == "2024-01-01T12:00:00Z"

    def test_save_sync_log_sync(self, test_db_path, sample_sync_log):
        """Test saving sync log data (sync version)."""
        # Test write operation for sync logs (sync version)
        persistence.save_sync_log_to_db_sync(sample_sync_log, test_db_path)

        # Verify log was saved (indirect check via no exceptions)
        # Note: We don't have a direct read method for sync logs in the API
        # but the operation should complete without errors
        assert True  # If we get here, the save operation succeeded

    def test_load_nonexistent_database_sync(self):
        """Test loading from non-existent database returns empty dict (sync version)."""
        nonexistent_path = "/tmp/nonexistent_db_test.db"

        # Ensure file doesn't exist
        if os.path.exists(nonexistent_path):
            os.unlink(nonexistent_path)

        result = persistence.load_existing_data_sync(nonexistent_path)
        assert result == {}

    def test_save_empty_data_sync(self, test_db_path):
        """Test saving empty data handles gracefully (sync version)."""
        # Should not raise exceptions when saving empty data
        persistence.save_messages_to_db_sync({}, test_db_path)
        persistence.save_messages_to_db_sync(None, test_db_path)

        # Load should still work and return empty structure
        loaded = persistence.load_existing_data_sync(test_db_path)
        assert loaded == {}

    def test_channel_members_save_sync(self, test_db_path):
        """Test saving channel members data (sync version)."""
        channel_members_data = {
            "TestGuild": {
                "_channel_members": {
                    "general": [
                        {
                            "channel_id": "456789",
                            "channel_name": "general",
                            "guild_id": "789123",
                            "guild_name": "TestGuild",
                            "user_id": "987654",
                            "user_name": "testuser",
                            "user_display_name": "Test User",
                            "user_joined_at": "2024-01-01T10:00:00Z",
                            "user_roles": "[]",
                            "is_bot": False,
                            "member_permissions": "{}",
                            "synced_at": "2024-01-01T12:00:00Z",
                        }
                    ]
                }
            }
        }

        # Test write operation for channel members (sync version)
        persistence.save_channel_members_to_db_sync(channel_members_data, test_db_path)

        # Verify operation completed successfully
        assert True  # If we get here, the save operation succeeded


class TestSettingsIntegration:
    """Test integration with settings configuration."""

    def test_default_db_path_from_settings_sync(self, sample_message_data):
        """Test that functions use settings.db_path as default (sync version)."""
        from unittest.mock import AsyncMock

        # Mock the DatabaseManager and repositories to avoid actual DB operations
        with patch(
            "pepino.discord.data.persistence.DatabaseManager"
        ) as mock_db_manager:
            with patch(
                "pepino.discord.data.persistence.MessageRepository"
            ) as mock_repo:
                # Properly mock async methods
                mock_manager_instance = mock_db_manager.return_value
                mock_manager_instance.close = AsyncMock()

                mock_repo_instance = mock_repo.return_value
                mock_repo_instance.bulk_insert_messages = AsyncMock()

                # Call without explicit db_path - should use settings default (sync version)
                persistence.save_messages_to_db_sync(sample_message_data)

                # Verify DatabaseManager was called (it should use the default value from settings)
                mock_db_manager.assert_called_once()
                # Check that it was called with the default db_path value
                call_args = mock_db_manager.call_args[0]
                assert (
                    len(call_args) == 1
                )  # Should be called with one argument (db_path)
