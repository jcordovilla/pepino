"""Unit tests for MessageRepository."""

import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

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

    def test_initialize_success(self, message_repo):
        """Test successful repository initialization."""
        message_repo.db_manager.initialize()

    def test_insert_messages_batch(self, message_repo):
        """Test batch insert of messages."""
        message_repo.db_manager.initialize()
        messages = [
            {"id": "msg1", "channel_id": "ch1", "channel_name": "general", "author_id": "user1", "author_name": "Alice", "content": "Hello world", "timestamp": "2024-01-01T12:00:00Z"},
            {"id": "msg2", "channel_id": "ch1", "channel_name": "general", "author_id": "user2", "author_name": "Bob", "content": "Hi Alice", "timestamp": "2024-01-01T12:01:00Z"}
        ]
        # Patch the db_manager's execute_many method
        with patch.object(message_repo.db_manager, "execute_many", return_value=2) as mock_exec:
            result = message_repo.insert_messages_batch(messages)
            assert result == 2
            mock_exec.assert_called_once()

    def test_get_messages_by_channel(self, message_repo):
        """Test getting messages by channel."""
        message_repo.db_manager.initialize()
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
            messages = message_repo.get_messages_by_channel("general")
            assert len(messages) == 1
            assert isinstance(messages[0], Message)
            assert messages[0].id == "msg1"
            assert messages[0].content == "Hello world"

    def test_get_messages_by_user(self, message_repo):
        """Test getting messages by user."""
        message_repo.db_manager.initialize()
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
            messages = message_repo.get_messages_by_user("user1")
            assert len(messages) == 1
            assert isinstance(messages[0], Message)
            assert messages[0].author_id == "user1"
            assert messages[0].content == "Hello world"

    def test_get_messages_by_date_range(self, message_repo):
        """Test getting messages within a date range."""
        message_repo.db_manager.initialize()
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        with patch.object(message_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                {
                    "id": "msg1",
                    "channel_id": "ch1",
                    "channel_name": "general",
                    "author_id": "user1",
                    "author_name": "Alice",
                    "author_display_name": "Alice",
                    "author_is_bot": False,
                    "content": "Hello world",
                    "timestamp": "2024-01-15T12:00:00Z",
                }
            ]
            messages = message_repo.get_messages_by_date_range("general", start_date, end_date)
            assert len(messages) == 1
            assert isinstance(messages[0], Message)
            assert messages[0].id == "msg1"
            assert messages[0].content == "Hello world"

    def test_get_message_statistics(self, message_repo):
        """Test getting message statistics."""
        message_repo.db_manager.initialize()
        with patch.object(message_repo.db_manager, "execute_single") as mock_execute:
            mock_execute.return_value = (
                100, 25, 10, 90, 50.5, "2024-01-01", "2024-12-01", 365
            )
            stats = message_repo.get_message_statistics("general")
            assert stats["total_messages"] == 100
            assert stats["unique_users"] == 25
            assert stats["bot_messages"] == 10
            assert stats["human_messages"] == 90
            assert stats["avg_message_length"] == 50.5
            assert stats["active_days"] == 365

    def test_get_hourly_activity(self, message_repo):
        """Test getting hourly activity."""
        message_repo.db_manager.initialize()
        with patch.object(message_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                {"hour": 12, "message_count": 50},
                {"hour": 13, "message_count": 75},
                {"hour": 14, "message_count": 25},
            ]
            activity = message_repo.get_hourly_activity(30)
            assert activity[12] == 50
            assert activity[13] == 75
            assert activity[14] == 25

    def test_get_daily_activity(self, message_repo):
        """Test getting daily activity."""
        message_repo.db_manager.initialize()
        with patch.object(message_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                ("2024-01-01", 100),
                ("2024-01-02", 150),
                ("2024-01-03", 75),
            ]
            activity = message_repo.get_daily_activity(30)
            assert len(activity) == 3
            assert activity[0][0] == "2024-01-01"
            assert activity[0][1] == 100

    def test_get_messages_by_channel_with_limit(self, message_repo):
        """Test getting messages by channel with limit."""
        message_repo.db_manager.initialize()
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
            messages = message_repo.get_messages_by_channel("general", limit=3)
            assert len(messages) == 3
            assert all(isinstance(m, Message) for m in messages)
            assert messages[0].id == "msg1"
            assert messages[1].id == "msg2"
            assert messages[2].id == "msg3"

    def test_get_messages_by_channel_with_min_length(self, message_repo):
        """Test getting messages by channel with minimum length filter."""
        message_repo.db_manager.initialize()
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
            messages = message_repo.get_messages_by_channel("general", min_length=10)
            assert len(messages) == 1
            assert isinstance(messages[0], Message)
            assert len(messages[0].content) >= 10

    def test_get_messages_by_user_with_limit(self, message_repo):
        """Test getting messages by user with limit."""
        message_repo.db_manager.initialize()
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
            messages = message_repo.get_messages_by_user("user1", limit=2)
            assert len(messages) == 2
            assert all(isinstance(m, Message) for m in messages)
            assert messages[0].author_id == "user1"
            assert messages[1].author_id == "user1"

    def test_get_message_statistics_no_channel(self, message_repo):
        """Test getting message statistics without specifying channel."""
        message_repo.db_manager.initialize()
        with patch.object(message_repo.db_manager, "execute_single") as mock_execute:
            mock_execute.return_value = (
                500, 50, 25, 475, 45.2, "2024-01-01", "2024-12-01", 365
            )
            stats = message_repo.get_message_statistics()
            assert stats["total_messages"] == 500
            assert stats["unique_users"] == 50
            assert stats["bot_messages"] == 25
            assert stats["human_messages"] == 475
            assert stats["avg_message_length"] == 45.2
            assert stats["active_days"] == 365

    def test_get_hourly_activity_empty_result(self, message_repo):
        """Test getting hourly activity with empty result."""
        message_repo.db_manager.initialize()
        with patch.object(message_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = []
            activity = message_repo.get_hourly_activity(30)
            assert isinstance(activity, dict)
            assert len(activity) == 0

    def test_get_daily_activity_empty_result(self, message_repo):
        """Test getting daily activity with empty result."""
        message_repo.db_manager.initialize()
        with patch.object(message_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = []
            activity = message_repo.get_daily_activity(30)
            assert isinstance(activity, list)
            assert len(activity) == 0

    def test_database_error_handling(self, message_repo):
        """Test handling of database errors."""
        message_repo.db_manager.initialize()
        with patch.object(message_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.side_effect = Exception("Database error")
            with pytest.raises(Exception, match="Database error"):
                message_repo.get_messages_by_channel("general")
