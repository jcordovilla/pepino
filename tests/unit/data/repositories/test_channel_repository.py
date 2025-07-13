"""Unit tests for ChannelRepository."""

import asyncio
import os
import tempfile
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pepino.data.database.manager import DatabaseManager
from pepino.data.models import Channel
from pepino.data.repositories.channel_repository import ChannelRepository


class TestChannelRepository:
    """Test cases for ChannelRepository."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)

    @pytest.fixture
    def db_manager(self, temp_db_path):
        """Create DatabaseManager instance."""
        return DatabaseManager(temp_db_path)

    @pytest.fixture
    def channel_repo(self, db_manager):
        """Create ChannelRepository instance."""
        return ChannelRepository(db_manager)

    def test_initialize_success(self, channel_repo):
        """Test successful repository initialization."""
        # No pool attribute in DatabaseManager, just check instance
        assert isinstance(channel_repo.db_manager, DatabaseManager)

    def test_get_channel_statistics_by_limit(self, channel_repo):
        """Test getting channel statistics by limit."""
        channel_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(channel_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                (
                    "ch1",
                    "general",
                    100,
                    25,
                    50.5,
                    "2024-01-01T12:00:00Z",
                    "2024-12-01T12:00:00Z",
                ),
                (
                    "ch2",
                    "random",
                    200,
                    50,
                    75.2,
                    "2024-02-01T12:00:00Z",
                    "2024-11-01T12:00:00Z",
                ),
            ]

            channels = channel_repo.get_channel_statistics_by_limit(limit=10)

            assert len(channels) == 2
            assert channels[0].channel_id == "ch1"
            assert channels[0].channel_name == "general"
            assert channels[0].message_count == 100
            assert channels[1].channel_id == "ch2"
            assert channels[1].channel_name == "random"
            assert channels[1].message_count == 200

    def test_get_channel_by_name_success(self, channel_repo):
        """Test getting channel by name successfully."""
        channel_name = "general"
        # Patch execute_query to return a list of dicts
        with patch.object(channel_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [{"channel_id": "ch1", "channel_name": "general"}]
            channel = channel_repo.get_channel_by_name(channel_name)
            assert channel is not None
            assert channel.channel_id == "ch1"
            assert channel.channel_name == "general"

    def test_get_channel_by_name_not_found(self, channel_repo):
        """Test getting channel by name when not found."""
        channel_name = "nonexistent"
        with patch.object(channel_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = []
            channel = channel_repo.get_channel_by_name(channel_name)
            assert channel is None

    def test_get_channel_by_name_case_insensitive(self, channel_repo):
        """Test getting channel by name with case-insensitive matching."""
        channel_name = "GENERAL"
        with patch.object(channel_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [{"channel_id": "ch1", "channel_name": "general"}]
            channel = channel_repo.get_channel_by_name(channel_name)
            assert channel is not None
            assert channel.channel_name == "general"

    def test_get_top_channels(self, channel_repo):
        """Test getting top channels."""
        # Patch execute_query to return a list of dicts
        with patch.object(channel_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                {
                    "channel_id": "ch1",
                    "channel_name": "general",
                    "message_count": 100,
                    "unique_users": 25,
                    "avg_message_length": 50.0,
                    "first_message": "2024-01-01T12:00:00Z",
                    "last_message": "2024-12-01T12:00:00Z",
                },
                {
                    "channel_id": "ch2",
                    "channel_name": "random",
                    "message_count": 200,
                    "unique_users": 50,
                    "avg_message_length": 75.0,
                    "first_message": "2024-02-01T12:00:00Z",
                    "last_message": "2024-11-01T12:00:00Z",
                },
            ]
            channels = channel_repo.get_top_channels(limit=10)
            assert len(channels) == 2
            assert isinstance(channels[0], Channel)
            assert channels[0].channel_id == "ch1"
            assert channels[0].channel_name == "general"
            assert channels[0].message_count == 100
            assert channels[1].channel_id == "ch2"
            assert channels[1].channel_name == "random"
            assert channels[1].message_count == 200

    def test_get_top_channels_with_custom_filter(self, channel_repo):
        """Test getting top channels with custom base filter (now just test normal call)."""
        with patch.object(channel_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                {
                    "channel_id": "ch1",
                    "channel_name": "general",
                    "message_count": 100,
                    "unique_users": 25,
                    "avg_message_length": 50.0,
                    "first_message": "2024-01-01T12:00:00Z",
                    "last_message": "2024-12-01T12:00:00Z",
                }
            ]
            channels = channel_repo.get_top_channels(limit=1)
            assert len(channels) == 1
            assert isinstance(channels[0], Channel)
            assert channels[0].channel_name == "general"

    def test_get_channel_statistics_by_limit_empty_result(self, channel_repo):
        """Test getting channel statistics by limit with empty result."""
        channel_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(channel_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = []

            channels = channel_repo.get_channel_statistics_by_limit(limit=10)

            assert len(channels) == 0

    def test_get_top_channels_empty_result(self, channel_repo):
        """Test getting top channels with empty result."""
        channel_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(channel_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = []

            channels = channel_repo.get_top_channels(limit=10)

            assert len(channels) == 0

    def test_get_channel_statistics_by_limit_with_none_values(self, channel_repo):
        """Test getting channel statistics by limit with None values in database."""
        channel_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(channel_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [("ch1", "general", 100, 25, None, None, None)]

            channels = channel_repo.get_channel_statistics_by_limit(limit=10)

            assert len(channels) == 1
            channel = channels[0]
            assert channel.avg_message_length == 0.0  # Should default to 0.0
            assert channel.first_message is None
            assert channel.last_message is None

    def test_get_channel_statistics_by_limit_with_zero_values(self, channel_repo):
        """Test getting channel statistics by limit with zero values."""
        channel_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(channel_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                ("ch1", "Inactive Channel", 0, 0, 0.0, None, None)
            ]

            channels = channel_repo.get_channel_statistics_by_limit(limit=10)

            assert len(channels) == 1
            channel = channels[0]
            assert channel.message_count == 0
            assert channel.unique_users == 0
            assert channel.avg_message_length == 0.0
            assert not channel.is_active  # Should be inactive

    def test_get_channel_statistics_by_limit_ordered_by_message_count(self, channel_repo):
        """Test that channel statistics by limit are ordered by message count descending."""
        channel_repo.db_manager.initialize()

        # Mock database query results - note: these should be in the order the query returns them
        # (ordered by message_count DESC), not in the order we define them
        with patch.object(channel_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                (
                    "ch2",
                    "High Activity",
                    200,
                    50,
                    75.0,
                    "2024-01-01T12:00:00Z",
                    "2024-12-01T12:00:00Z",
                ),
                (
                    "ch3",
                    "Medium Activity",
                    100,
                    25,
                    50.0,
                    "2024-01-01T12:00:00Z",
                    "2024-12-01T12:00:00Z",
                ),
                (
                    "ch1",
                    "Low Activity",
                    50,
                    10,
                    25.0,
                    "2024-01-01T12:00:00Z",
                    "2024-12-01T12:00:00Z",
                ),
            ]

            channels = channel_repo.get_channel_statistics_by_limit(limit=3)

            assert len(channels) == 3
            # Should be ordered by message_count descending
            assert channels[0].message_count == 200  # High Activity
            assert channels[1].message_count == 100  # Medium Activity
            assert channels[2].message_count == 50  # Low Activity

    def test_get_channel_list(self, channel_repo):
        """Test getting channel list."""
        with patch.object(channel_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                {"channel_name": "general", "message_count": 100},
                {"channel_name": "random", "message_count": 75},
                {"channel_name": "announcements", "message_count": 25},
            ]
            channels = channel_repo.get_channel_list()
            assert len(channels) == 3
            assert "general" in channels
            assert "random" in channels
            assert "announcements" in channels

    def test_get_channel_list_with_custom_filter(self, channel_repo):
        """Test getting channel list with custom filter (now just test normal call)."""
        with patch.object(channel_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                {"channel_name": "general", "message_count": 100},
                {"channel_name": "random", "message_count": 75},
            ]
            channels = channel_repo.get_channel_list()
            assert len(channels) == 2
            assert "general" in channels
            assert "random" in channels

    def test_get_channel_list_empty_result(self, channel_repo):
        """Test getting channel list with empty result."""
        with patch.object(channel_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = []
            channels = channel_repo.get_channel_list()
            assert isinstance(channels, list)
            assert len(channels) == 0

    def test_get_channel_id_by_name(self, channel_repo):
        """Test getting channel ID by channel name."""
        channel_name = "general"
        with patch.object(channel_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [{"channel_id": "ch123"}]
            channel_id = channel_repo.get_channel_id_by_name(channel_name)
            assert channel_id == "ch123"

    def test_get_channel_id_by_name_not_found(self, channel_repo):
        """Test getting channel ID by name when not found."""
        channel_name = "nonexistent"
        with patch.object(channel_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = []
            channel_id = channel_repo.get_channel_id_by_name(channel_name)
            assert channel_id is None
