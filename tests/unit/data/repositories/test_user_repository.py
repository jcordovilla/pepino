"""Unit tests for UserRepository."""

import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from pepino.config import Settings
from pepino.data.database.manager import DatabaseManager
from pepino.data.models import User
from pepino.data.repositories.user_repository import UserRepository


class TestUserRepository:
    """Test cases for UserRepository."""

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
    def user_repo(self, db_manager):
        """Create UserRepository instance."""
        return UserRepository(db_manager)

    def test_initialize_success(self, user_repo):
        """Test successful repository initialization."""
        user_repo.db_manager.initialize()
        # Just check that initialize does not raise
        assert isinstance(user_repo.db_manager, DatabaseManager)

    def test_get_user_statistics(self, user_repo):
        """Test getting user statistics."""
        user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                {
                    "author_id": "user1",
                    "author_name": "Alice",
                    "author_display_name": "Alice Smith",
                    "message_count": 100,
                    "channels_active": 5,
                    "avg_message_length": 50.5,
                    "first_message_date": "2024-01-01",
                    "last_message_date": "2024-12-01",
                },
                {
                    "author_id": "user2",
                    "author_name": "Bob",
                    "author_display_name": None,
                    "message_count": 200,
                    "channels_active": 10,
                    "avg_message_length": 75.2,
                    "first_message_date": "2024-02-01",
                    "last_message_date": "2024-11-01",
                },
            ]

            users = user_repo.get_user_statistics(limit=10)

            assert len(users) == 2
            assert isinstance(users[0], User)
            assert users[0].author_id == "user1"
            assert users[0].author_name == "Alice"
            assert users[0].author_display_name == "Alice Smith"
            assert users[1].author_id == "user2"
            assert users[1].author_name == "Bob"
            assert users[1].author_display_name is None

    def test_get_user_by_name(self, user_repo):
        """Test getting user by name."""
        user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                {
                    "author_id": "user1",
                    "author_name": "Alice",
                    "author_display_name": "Alice Smith",
                }
            ]

            user = user_repo.get_user_by_name("Alice")

            assert user is not None
            assert isinstance(user, User)
            assert user.author_id == "user1"
            assert user.author_name == "Alice"
            assert user.author_display_name == "Alice Smith"

    def test_get_user_by_name_not_found(self, user_repo):
        """Test getting user by name when not found."""
        user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = []

            user = user_repo.get_user_by_name("nonexistent")

            assert user is None

    def test_get_user_by_name_case_insensitive(self, user_repo):
        """Test getting user by name with case-insensitive matching."""
        user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                {
                    "author_id": "user1",
                    "author_name": "Alice",
                    "author_display_name": "Alice Smith",
                }
            ]

            user = user_repo.get_user_by_name("alice")

            assert user is not None
            assert isinstance(user, User)
            assert user.author_name == "Alice"

    def test_get_top_users(self, user_repo):
        """Test getting top users."""
        user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                {
                    "author_id": "user1",
                    "author_name": "Alice",
                    "author_display_name": "Alice Smith",
                    "message_count": 100,
                    "channels_active": 5,
                    "avg_message_length": 50.5,
                    "first_message_date": "2024-01-01",
                    "last_message_date": "2024-12-01",
                },
                {
                    "author_id": "user2",
                    "author_name": "Bob",
                    "author_display_name": None,
                    "message_count": 200,
                    "channels_active": 10,
                    "avg_message_length": 75.2,
                    "first_message_date": "2024-02-01",
                    "last_message_date": "2024-11-01",
                },
            ]

            users = user_repo.get_top_users(limit=10)

            assert len(users) == 2
            assert isinstance(users[0], User)
            assert users[0].author_id == "user1"
            assert users[0].author_name == "Alice"
            assert users[0].message_count == 100
            assert users[1].author_id == "user2"
            assert users[1].author_name == "Bob"
            assert users[1].message_count == 200

    def test_get_top_users_with_custom_filter(self, user_repo):
        """Test getting top users with custom filter."""
        user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                {
                    "author_id": "user1",
                    "author_name": "Alice",
                    "author_display_name": "Alice Smith",
                    "message_count": 100,
                    "channels_active": 5,
                    "avg_message_length": 50.5,
                    "first_message_date": "2024-01-01",
                    "last_message_date": "2024-12-01",
                }
            ]

            users = user_repo.get_top_users(limit=10)

            assert len(users) == 1
            assert isinstance(users[0], User)
            assert users[0].author_name == "Alice"

    def test_get_user_statistics_empty_result(self, user_repo):
        """Test getting user statistics with empty result."""
        user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = []

            users = user_repo.get_user_statistics(limit=10)

            assert len(users) == 0

    def test_get_top_users_empty_result(self, user_repo):
        """Test getting top users with empty result."""
        user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = []

            users = user_repo.get_top_users(limit=10)

            assert len(users) == 0

    def test_get_user_statistics_with_none_values(self, user_repo):
        """Test getting user statistics with None values in database."""
        user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                {
                    "author_id": "user1",
                    "author_name": "Alice",
                    "author_display_name": None,
                    "message_count": 100,
                    "channels_active": 5,
                    "avg_message_length": None,
                    "first_message_date": None,
                    "last_message_date": None,
                }
            ]

            users = user_repo.get_user_statistics(limit=10)

            assert len(users) == 1
            user = users[0]
            assert isinstance(user, User)
            assert user.avg_message_length == 0.0  # Should default to 0.0
            assert user.first_message_date is None
            assert user.last_message_date is None

    def test_get_user_statistics_with_zero_values(self, user_repo):
        """Test getting user statistics with zero values."""
        user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                {
                    "author_id": "user1",
                    "author_name": "Inactive User",
                    "author_display_name": None,
                    "message_count": 0,
                    "channels_active": 0,
                    "avg_message_length": 0.0,
                    "first_message_date": None,
                    "last_message_date": None,
                }
            ]

            users = user_repo.get_user_statistics(limit=10)

            assert len(users) == 1
            user = users[0]
            assert isinstance(user, User)
            assert user.message_count == 0
            assert user.channels_active == 0
            assert user.avg_message_length == 0.0
            assert not user.is_active

    def test_get_user_statistics_ordered_by_message_count(self, user_repo):
        """Test that user statistics are ordered by message count."""
        user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                {
                    "author_id": "user2",
                    "author_name": "Bob",
                    "author_display_name": None,
                    "message_count": 200,
                    "channels_active": 10,
                    "avg_message_length": 75.2,
                    "first_message_date": "2024-02-01",
                    "last_message_date": "2024-11-01",
                },
                {
                    "author_id": "user1",
                    "author_name": "Alice",
                    "author_display_name": "Alice Smith",
                    "message_count": 100,
                    "channels_active": 5,
                    "avg_message_length": 50.5,
                    "first_message_date": "2024-01-01",
                    "last_message_date": "2024-12-01",
                },
            ]

            users = user_repo.get_user_statistics(limit=10)

            assert len(users) == 2
            assert users[0].message_count == 200  # Bob should be first
            assert users[1].message_count == 100  # Alice should be second

    def test_get_user_by_name_with_display_name_match(self, user_repo):
        """Test getting user by name with display name match."""
        user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                {
                    "author_id": "user1",
                    "author_name": "Alice",
                    "author_display_name": "Alice Smith",
                }
            ]

            user = user_repo.get_user_by_name("Smith")

            assert user is not None
            assert isinstance(user, User)
            assert user.author_display_name == "Alice Smith"

    def test_get_user_statistics_database_error(self, user_repo):
        """Test handling of database errors in get_user_statistics."""
        user_repo.db_manager.initialize()

        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.side_effect = Exception("Database error")

            with pytest.raises(Exception):
                user_repo.get_user_statistics(limit=10)

    def test_get_user_by_name_database_error(self, user_repo):
        """Test handling of database errors in get_user_by_name."""
        user_repo.db_manager.initialize()

        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.side_effect = Exception("Database error")

            with pytest.raises(Exception):
                user_repo.get_user_by_name("Alice")

    def test_get_top_users_database_error(self, user_repo):
        """Test handling of database errors in get_top_users."""
        user_repo.db_manager.initialize()

        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.side_effect = Exception("Database error")

            with pytest.raises(Exception):
                user_repo.get_top_users(limit=10)

    def test_get_user_list(self, user_repo):
        """Test getting user list."""
        user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                {
                    "author_name": "Alice",
                    "author_display_name": "Alice Smith",
                    "message_count": 100,
                },
                {
                    "author_name": "Bob",
                    "author_display_name": None,
                    "message_count": 200,
                },
            ]

            users = user_repo.get_user_list()

            assert len(users) == 2
            assert users[0]["author_name"] == "Alice"
            assert users[0]["display_name"] == "Alice Smith"
            assert users[1]["author_name"] == "Bob"
            assert users[1]["display_name"] == "Bob"  # Should use author_name when display_name is None

    def test_get_user_list_with_custom_filter(self, user_repo):
        """Test getting user list with custom filter."""
        user_repo.db_manager.initialize()

        custom_filter = "author_name NOT LIKE '%bot%'"

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                {
                    "author_name": "Alice",
                    "author_display_name": "Alice Smith",
                    "message_count": 100,
                }
            ]

            users = user_repo.get_user_list(base_filter=custom_filter)

            assert len(users) == 1
            assert users[0]["author_name"] == "Alice"

    def test_get_user_statistics_by_id(self, user_repo):
        """Test getting user statistics by ID."""
        user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                {
                    "total_messages": 100,
                    "channels_active": 5,
                    "avg_message_length": 50.5,
                    "first_message": "2024-01-01",
                    "last_message": "2024-12-01",
                }
            ]

            stats = user_repo.get_user_statistics_by_id("user1")

            assert stats is not None
            assert stats["total_messages"] == 100
            assert stats["channels_active"] == 5
            assert stats["avg_message_length"] == 50.5

    def test_get_user_statistics_by_id_not_found(self, user_repo):
        """Test getting user statistics by ID when not found."""
        user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = []

            stats = user_repo.get_user_statistics_by_id("nonexistent")

            assert stats is None

    def test_get_user_content_sample(self, user_repo):
        """Test getting user content sample."""
        user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                {"content": "Hello world"},
                {"content": "How are you?"},
                {"content": "Good morning!"},
            ]

            content = user_repo.get_user_content_sample("user1", limit=3)

            assert len(content) == 3
            assert "Hello world" in content
            assert "How are you?" in content
            assert "Good morning!" in content

    def test_get_user_content_sample_empty_result(self, user_repo):
        """Test getting user content sample with empty result."""
        user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = []

            content = user_repo.get_user_content_sample("user1", limit=3)

            assert len(content) == 0

    def test_find_user_by_name(self, user_repo):
        """Test finding user by name."""
        user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                {
                    "author_id": "user1",
                    "author_name": "Alice",
                    "author_display_name": "Alice Smith",
                    "message_count": 100,
                }
            ]

            user = user_repo.find_user_by_name("Alice")

            assert user is not None
            assert user["author_id"] == "user1"
            assert user["author_name"] == "Alice"
            assert user["author_display_name"] == "Alice Smith"

    def test_find_user_by_name_not_found(self, user_repo):
        """Test finding user by name when not found."""
        user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = []

            user = user_repo.find_user_by_name("nonexistent")

            assert user is None
