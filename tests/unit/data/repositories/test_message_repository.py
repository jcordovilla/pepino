"""Unit tests for MessageRepository."""

import asyncio
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pepino.data.database.manager import DatabaseManager
from pepino.data.models import Message
from pepino.data.repositories.message_repository import MessageRepository


class TestMessageRepository:
    """Test cases for MessageRepository."""

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
    def message_repo(self, db_manager):
        """Create MessageRepository instance."""
        return MessageRepository(db_manager)

    @pytest.mark.asyncio
    async def test_initialize_success(self, message_repo):
        """Test successful repository initialization."""
        await message_repo.db_manager.initialize()

        assert message_repo.db_manager.pool is not None

    @pytest.mark.asyncio
    async def test_get_messages_by_channel(self, message_repo):
        """Test getting messages by channel."""
        await message_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(message_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                {
                    "id": "msg1",
                    "channel_id": "ch1",
                    "channel_name": "general",
                    "author_id": "user1",
                    "author_name": "Alice",
                    "content": "Hello world",
                    "timestamp": "2024-01-01T12:00:00Z",
                }
            ]

            messages = await message_repo.get_messages_by_channel("general")

            assert len(messages) == 1
            assert messages[0].id == "msg1"
            assert messages[0].content == "Hello world"

    @pytest.mark.asyncio
    async def test_get_messages_by_user(self, message_repo):
        """Test getting messages by user."""
        await message_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(message_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                {
                    "id": "msg1",
                    "channel_id": "ch1",
                    "channel_name": "general",
                    "author_id": "user1",
                    "author_name": "Alice",
                    "content": "Hello world",
                    "timestamp": "2024-01-01T12:00:00Z",
                }
            ]

            messages = await message_repo.get_messages_by_user("user1")

            assert len(messages) == 1
            assert messages[0].author_id == "user1"
            assert messages[0].content == "Hello world"

    @pytest.mark.asyncio
    async def test_get_messages_by_date_range(self, message_repo):
        """Test getting messages within a date range."""
        await message_repo.db_manager.initialize()

        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)

        # Mock database query results
        with patch.object(message_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                {
                    "id": "msg1",
                    "channel_id": "ch1",
                    "channel_name": "general",
                    "author_id": "user1",
                    "author_name": "Alice",
                    "content": "Hello world",
                    "timestamp": "2024-01-15T12:00:00Z",
                }
            ]

            messages = await message_repo.get_messages_by_date_range(
                start_date, end_date
            )

            assert len(messages) == 1
            assert messages[0].id == "msg1"

    @pytest.mark.asyncio
    async def test_get_message_statistics(self, message_repo):
        """Test getting message statistics."""
        await message_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(message_repo.db_manager, "execute_single") as mock_execute:
            mock_execute.return_value = (
                100,
                25,
                10,
                90,
                50.5,
                "2024-01-01",
                "2024-12-01",
            )

            stats = await message_repo.get_message_statistics("general")

            assert stats["total_messages"] == 100
            assert stats["unique_users"] == 25
            assert stats["bot_messages"] == 10
            assert stats["human_messages"] == 90
            assert stats["avg_message_length"] == 50.5

    @pytest.mark.asyncio
    async def test_get_hourly_activity(self, message_repo):
        """Test getting hourly activity."""
        await message_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(message_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [(12, 50), (13, 75), (14, 25)]

            activity = await message_repo.get_hourly_activity(30)

            assert activity[12] == 50
            assert activity[13] == 75
            assert activity[14] == 25

    @pytest.mark.asyncio
    async def test_get_daily_activity(self, message_repo):
        """Test getting daily activity."""
        await message_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(message_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                ("2024-01-01", 100),
                ("2024-01-02", 150),
                ("2024-01-03", 75),
            ]

            activity = await message_repo.get_daily_activity(30)

            assert len(activity) == 3
            assert activity[0][0] == "2024-01-01"
            assert activity[0][1] == 100

    @pytest.mark.asyncio
    async def test_get_messages_by_channel_with_limit(self, message_repo):
        """Test getting messages by channel with limit."""
        await message_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(message_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                {
                    "id": f"msg{i}",
                    "channel_id": "ch1",
                    "channel_name": "general",
                    "author_id": "user1",
                    "author_name": "Alice",
                    "content": f"Message {i}",
                    "timestamp": "2024-01-01T12:00:00Z",
                }
                for i in range(1, 4)
            ]

            messages = await message_repo.get_messages_by_channel("general", limit=3)

            assert len(messages) == 3
            assert messages[0].id == "msg1"
            assert messages[1].id == "msg2"
            assert messages[2].id == "msg3"

    @pytest.mark.asyncio
    async def test_get_messages_by_channel_with_min_length(self, message_repo):
        """Test getting messages by channel with minimum length filter."""
        await message_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(message_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                {
                    "id": "msg1",
                    "channel_id": "ch1",
                    "channel_name": "general",
                    "author_id": "user1",
                    "author_name": "Alice",
                    "content": "This is a longer message that meets the minimum length requirement",
                    "timestamp": "2024-01-01T12:00:00Z",
                }
            ]

            messages = await message_repo.get_messages_by_channel(
                "general", min_length=10
            )

            assert len(messages) == 1
            assert len(messages[0].content) >= 10

    @pytest.mark.asyncio
    async def test_get_messages_by_user_with_limit(self, message_repo):
        """Test getting messages by user with limit."""
        await message_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(message_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                {
                    "id": f"msg{i}",
                    "channel_id": "ch1",
                    "channel_name": "general",
                    "author_id": "user1",
                    "author_name": "Alice",
                    "content": f"Message {i}",
                    "timestamp": "2024-01-01T12:00:00Z",
                }
                for i in range(1, 3)
            ]

            messages = await message_repo.get_messages_by_user("user1", limit=2)

            assert len(messages) == 2
            assert messages[0].author_id == "user1"
            assert messages[1].author_id == "user1"

    @pytest.mark.asyncio
    async def test_get_message_statistics_no_channel(self, message_repo):
        """Test getting message statistics without specifying channel."""
        await message_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(message_repo.db_manager, "execute_single") as mock_execute:
            mock_execute.return_value = (
                500,
                50,
                25,
                475,
                45.2,
                "2024-01-01",
                "2024-12-01",
            )

            stats = await message_repo.get_message_statistics()

            assert stats["total_messages"] == 500
            assert stats["unique_users"] == 50
            assert stats["bot_messages"] == 25
            assert stats["human_messages"] == 475
            assert stats["avg_message_length"] == 45.2

    @pytest.mark.asyncio
    async def test_get_hourly_activity_empty_result(self, message_repo):
        """Test getting hourly activity with empty result."""
        await message_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(message_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = []

            activity = await message_repo.get_hourly_activity(30)

            assert isinstance(activity, dict)
            assert len(activity) == 0

    @pytest.mark.asyncio
    async def test_get_daily_activity_empty_result(self, message_repo):
        """Test getting daily activity with empty result."""
        await message_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(message_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = []

            activity = await message_repo.get_daily_activity(30)

            assert isinstance(activity, list)
            assert len(activity) == 0

    @pytest.mark.asyncio
    async def test_database_error_handling(self, message_repo):
        """Test handling of database errors."""
        await message_repo.db_manager.initialize()

        with patch.object(message_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.side_effect = Exception("Database error")

            with pytest.raises(Exception, match="Database error"):
                await message_repo.get_messages_by_channel("general")
