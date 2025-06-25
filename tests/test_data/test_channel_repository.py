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

    @pytest.mark.asyncio
    async def test_initialize_success(self, channel_repo):
        """Test successful repository initialization."""
        await channel_repo.db_manager.initialize()

        assert channel_repo.db_manager.pool is not None

    @pytest.mark.asyncio
    async def test_get_channel_statistics(self, channel_repo):
        """Test getting channel statistics."""
        await channel_repo.db_manager.initialize()

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

            channels = await channel_repo.get_channel_statistics(limit=10)

            assert len(channels) == 2
            assert channels[0].channel_id == "ch1"
            assert channels[0].channel_name == "general"
            assert channels[0].message_count == 100
            assert channels[1].channel_id == "ch2"
            assert channels[1].channel_name == "random"
            assert channels[1].message_count == 200

    @pytest.mark.asyncio
    async def test_get_channel_by_name_success(self, channel_repo):
        """Test getting channel by name successfully."""
        await channel_repo.db_manager.initialize()

        channel_name = "general"

        with patch.object(channel_repo.db_manager, "execute_single") as mock_execute:
            mock_execute.return_value = ("ch1", "general")

            channel = await channel_repo.get_channel_by_name(channel_name)

            assert channel is not None
            assert channel.channel_id == "ch1"
            assert channel.channel_name == "general"

    @pytest.mark.asyncio
    async def test_get_channel_by_name_not_found(self, channel_repo):
        """Test getting channel by name when not found."""
        await channel_repo.db_manager.initialize()

        channel_name = "nonexistent"

        with patch.object(channel_repo.db_manager, "execute_single") as mock_execute:
            mock_execute.return_value = None

            channel = await channel_repo.get_channel_by_name(channel_name)

            assert channel is None

    @pytest.mark.asyncio
    async def test_get_channel_by_name_case_insensitive(self, channel_repo):
        """Test getting channel by name with case-insensitive matching."""
        await channel_repo.db_manager.initialize()

        channel_name = "GENERAL"

        with patch.object(channel_repo.db_manager, "execute_single") as mock_execute:
            mock_execute.return_value = ("ch1", "general")

            channel = await channel_repo.get_channel_by_name(channel_name)

            assert channel is not None
            assert channel.channel_name == "general"

    @pytest.mark.asyncio
    async def test_get_top_channels(self, channel_repo):
        """Test getting top channels."""
        await channel_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(channel_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                (
                    "ch1",
                    "general",
                    100,
                    25,
                    50.0,
                    "2024-01-01T12:00:00Z",
                    "2024-12-01T12:00:00Z",
                ),
                (
                    "ch2",
                    "random",
                    200,
                    50,
                    75.0,
                    "2024-02-01T12:00:00Z",
                    "2024-11-01T12:00:00Z",
                ),
            ]

            channels = await channel_repo.get_top_channels(limit=10)

            assert len(channels) == 2
            assert channels[0]["channel_id"] == "ch1"
            assert channels[0]["channel_name"] == "general"
            assert channels[0]["message_count"] == 100
            assert channels[1]["channel_id"] == "ch2"
            assert channels[1]["channel_name"] == "random"
            assert channels[1]["message_count"] == 200

    @pytest.mark.asyncio
    async def test_get_top_channels_with_custom_filter(self, channel_repo):
        """Test getting top channels with custom base filter."""
        await channel_repo.db_manager.initialize()

        custom_filter = "channel_name NOT LIKE '%test%'"

        with patch.object(channel_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                (
                    "ch1",
                    "general",
                    100,
                    25,
                    50.0,
                    "2024-01-01T12:00:00Z",
                    "2024-12-01T12:00:00Z",
                )
            ]

            channels = await channel_repo.get_top_channels(base_filter=custom_filter)

            assert len(channels) == 1
            assert channels[0]["channel_name"] == "general"

    @pytest.mark.asyncio
    async def test_get_channel_statistics_empty_result(self, channel_repo):
        """Test getting channel statistics with empty result."""
        await channel_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(channel_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = []

            channels = await channel_repo.get_channel_statistics(limit=10)

            assert len(channels) == 0

    @pytest.mark.asyncio
    async def test_get_top_channels_empty_result(self, channel_repo):
        """Test getting top channels with empty result."""
        await channel_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(channel_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = []

            channels = await channel_repo.get_top_channels(limit=10)

            assert len(channels) == 0

    @pytest.mark.asyncio
    async def test_get_channel_statistics_with_none_values(self, channel_repo):
        """Test getting channel statistics with None values in database."""
        await channel_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(channel_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [("ch1", "general", 100, 25, None, None, None)]

            channels = await channel_repo.get_channel_statistics(limit=10)

            assert len(channels) == 1
            channel = channels[0]
            assert channel.avg_message_length == 0.0  # Should default to 0.0
            assert channel.first_message is None
            assert channel.last_message is None

    @pytest.mark.asyncio
    async def test_get_channel_statistics_with_zero_values(self, channel_repo):
        """Test getting channel statistics with zero values."""
        await channel_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(channel_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                ("ch1", "Inactive Channel", 0, 0, 0.0, None, None)
            ]

            channels = await channel_repo.get_channel_statistics(limit=10)

            assert len(channels) == 1
            channel = channels[0]
            assert channel.message_count == 0
            assert channel.unique_users == 0
            assert channel.avg_message_length == 0.0
            assert not channel.is_active  # Should be inactive

    @pytest.mark.asyncio
    async def test_get_channel_statistics_ordered_by_message_count(self, channel_repo):
        """Test that channel statistics are ordered by message count descending."""
        await channel_repo.db_manager.initialize()

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

            channels = await channel_repo.get_channel_statistics(limit=3)

            assert len(channels) == 3
            # Should be ordered by message_count descending
            assert channels[0].message_count == 200  # High Activity
            assert channels[1].message_count == 100  # Medium Activity
            assert channels[2].message_count == 50  # Low Activity

    @pytest.mark.asyncio
    async def test_get_channel_statistics_database_error(self, channel_repo):
        """Test handling of database errors in channel statistics."""
        await channel_repo.db_manager.initialize()

        with patch.object(channel_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.side_effect = Exception("Database error")

            with pytest.raises(Exception, match="Database error"):
                await channel_repo.get_channel_statistics(limit=10)

    @pytest.mark.asyncio
    async def test_get_channel_by_name_database_error(self, channel_repo):
        """Test handling of database errors in get_channel_by_name."""
        await channel_repo.db_manager.initialize()

        with patch.object(channel_repo.db_manager, "execute_single") as mock_execute:
            mock_execute.side_effect = Exception("Database error")

            with pytest.raises(Exception, match="Database error"):
                await channel_repo.get_channel_by_name("general")

    @pytest.mark.asyncio
    async def test_get_top_channels_database_error(self, channel_repo):
        """Test handling of database errors in get_top_channels."""
        await channel_repo.db_manager.initialize()

        with patch.object(channel_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.side_effect = Exception("Database error")

            with pytest.raises(Exception, match="Database error"):
                await channel_repo.get_top_channels(limit=10)

    @pytest.mark.asyncio
    async def test_get_channel_list(self, channel_repo):
        """Test getting channel list."""
        await channel_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(channel_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                ("general", 100),
                ("random", 75),
                ("announcements", 25),
            ]

            channels = await channel_repo.get_channel_list()

            assert len(channels) == 3
            assert "general" in channels
            assert "random" in channels
            assert "announcements" in channels

    @pytest.mark.asyncio
    async def test_get_channel_list_with_custom_filter(self, channel_repo):
        """Test getting channel list with custom filter."""
        await channel_repo.db_manager.initialize()

        custom_filter = "channel_name NOT LIKE '%test%'"

        # Mock database query results
        with patch.object(channel_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [("general", 100), ("random", 75)]

            channels = await channel_repo.get_channel_list(base_filter=custom_filter)

            assert len(channels) == 2
            assert "general" in channels
            assert "random" in channels

    @pytest.mark.asyncio
    async def test_get_channel_list_empty_result(self, channel_repo):
        """Test getting channel list with empty result."""
        await channel_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(channel_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = []

            channels = await channel_repo.get_channel_list()

            assert isinstance(channels, list)
            assert len(channels) == 0

    @pytest.mark.asyncio
    async def test_get_channel_id_by_name(self, channel_repo):
        """Test getting channel ID by channel name."""
        await channel_repo.db_manager.initialize()

        channel_name = "general"

        with patch.object(channel_repo.db_manager, "execute_single") as mock_execute:
            mock_execute.return_value = ("ch123",)

            channel_id = await channel_repo.get_channel_id_by_name(channel_name)

            assert channel_id == "ch123"

    @pytest.mark.asyncio
    async def test_get_channel_id_by_name_not_found(self, channel_repo):
        """Test getting channel ID by name when not found."""
        await channel_repo.db_manager.initialize()

        channel_name = "nonexistent"

        with patch.object(channel_repo.db_manager, "execute_single") as mock_execute:
            mock_execute.return_value = None

            channel_id = await channel_repo.get_channel_id_by_name(channel_name)

            assert channel_id is None
