"""Unit tests for UserRepository."""

import asyncio
import os
import tempfile
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

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

    @pytest.mark.asyncio
    async def test_initialize_success(self, user_repo):
        """Test successful repository initialization."""
        await user_repo.db_manager.initialize()

        assert user_repo.db_manager.pool is not None

    @pytest.mark.asyncio
    async def test_get_user_statistics(self, user_repo):
        """Test getting user statistics."""
        await user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                (
                    "user1",
                    "Alice",
                    "Alice Smith",
                    100,
                    5,
                    50.5,
                    "2024-01-01",
                    "2024-12-01",
                ),
                ("user2", "Bob", None, 200, 10, 75.2, "2024-02-01", "2024-11-01"),
            ]

            users = await user_repo.get_user_statistics(limit=10)

            assert len(users) == 2
            assert users[0].author_id == "user1"
            assert users[0].author_name == "Alice"
            assert users[0].author_display_name == "Alice Smith"
            assert users[1].author_id == "user2"
            assert users[1].author_name == "Bob"
            assert users[1].author_display_name is None

    @pytest.mark.asyncio
    async def test_get_user_by_name(self, user_repo):
        """Test getting user by name."""
        await user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_single") as mock_execute:
            mock_execute.return_value = ("user1", "Alice", "Alice Smith")

            user = await user_repo.get_user_by_name("Alice")

            assert user is not None
            assert user.author_id == "user1"
            assert user.author_name == "Alice"
            assert user.author_display_name == "Alice Smith"

    @pytest.mark.asyncio
    async def test_get_user_by_name_not_found(self, user_repo):
        """Test getting user by name when not found."""
        await user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_single") as mock_execute:
            mock_execute.return_value = None

            user = await user_repo.get_user_by_name("nonexistent")

            assert user is None

    @pytest.mark.asyncio
    async def test_get_user_by_name_case_insensitive(self, user_repo):
        """Test getting user by name with case-insensitive matching."""
        await user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_single") as mock_execute:
            mock_execute.return_value = ("user1", "Alice", "Alice Smith")

            user = await user_repo.get_user_by_name("alice")

            assert user is not None
            assert user.author_name == "Alice"

    @pytest.mark.asyncio
    async def test_get_top_users(self, user_repo):
        """Test getting top users."""
        await user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                (
                    "user1",
                    "Alice",
                    "Alice Smith",
                    100,
                    5,
                    50.5,
                    "2024-01-01",
                    "2024-12-01",
                ),
                ("user2", "Bob", None, 200, 10, 75.2, "2024-02-01", "2024-11-01"),
            ]

            users = await user_repo.get_top_users(limit=10)

            assert len(users) == 2
            assert users[0]["author_id"] == "user1"
            assert users[0]["author_name"] == "Alice"
            assert users[0]["message_count"] == 100
            assert users[1]["author_id"] == "user2"
            assert users[1]["author_name"] == "Bob"
            assert users[1]["message_count"] == 200

    @pytest.mark.asyncio
    async def test_get_top_users_with_custom_filter(self, user_repo):
        """Test getting top users with custom filter."""
        await user_repo.db_manager.initialize()

        custom_filter = "author_name NOT LIKE '%bot%'"

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                (
                    "user1",
                    "Alice",
                    "Alice Smith",
                    100,
                    5,
                    50.5,
                    "2024-01-01",
                    "2024-12-01",
                )
            ]

            users = await user_repo.get_top_users(base_filter=custom_filter)

            assert len(users) == 1
            assert users[0]["author_name"] == "Alice"

    @pytest.mark.asyncio
    async def test_get_user_statistics_empty_result(self, user_repo):
        """Test getting user statistics with empty result."""
        await user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = []

            users = await user_repo.get_user_statistics(limit=10)

            assert len(users) == 0

    @pytest.mark.asyncio
    async def test_get_top_users_empty_result(self, user_repo):
        """Test getting top users with empty result."""
        await user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = []

            users = await user_repo.get_top_users(limit=10)

            assert len(users) == 0

    @pytest.mark.asyncio
    async def test_get_user_statistics_with_none_values(self, user_repo):
        """Test getting user statistics with None values in database."""
        await user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                ("user1", "Alice", None, 100, 5, None, None, None)
            ]

            users = await user_repo.get_user_statistics(limit=10)

            assert len(users) == 1
            user = users[0]
            assert user.author_display_name is None
            assert user.avg_message_length == 0.0  # Should default to 0.0
            assert user.first_message_date is None
            assert user.last_message_date is None

    @pytest.mark.asyncio
    async def test_get_user_statistics_with_zero_values(self, user_repo):
        """Test getting user statistics with zero values."""
        await user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                ("user1", "Inactive User", None, 0, 0, 0.0, None, None)
            ]

            users = await user_repo.get_user_statistics(limit=10)

            assert len(users) == 1
            user = users[0]
            assert user.message_count == 0
            assert user.channels_active == 0
            assert user.avg_message_length == 0.0
            assert not user.is_active  # Should be inactive

    @pytest.mark.asyncio
    async def test_get_user_statistics_ordered_by_message_count(self, user_repo):
        """Test that user statistics are ordered by message count descending."""
        await user_repo.db_manager.initialize()

        # Mock database query results - note: these should be in the order the query returns them
        # (ordered by message_count DESC), not in the order we define them
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                (
                    "user2",
                    "High Activity",
                    None,
                    200,
                    10,
                    75.0,
                    "2024-01-01",
                    "2024-12-01",
                ),
                (
                    "user3",
                    "Medium Activity",
                    None,
                    100,
                    5,
                    50.0,
                    "2024-01-01",
                    "2024-12-01",
                ),
                (
                    "user1",
                    "Low Activity",
                    None,
                    50,
                    2,
                    25.0,
                    "2024-01-01",
                    "2024-12-01",
                ),
            ]

            users = await user_repo.get_user_statistics(limit=3)

            assert len(users) == 3
            # Should be ordered by message_count descending
            assert users[0].message_count == 200  # High Activity
            assert users[1].message_count == 100  # Medium Activity
            assert users[2].message_count == 50  # Low Activity

    @pytest.mark.asyncio
    async def test_get_user_by_name_with_display_name_match(self, user_repo):
        """Test getting user by name that matches display name."""
        await user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_single") as mock_execute:
            mock_execute.return_value = ("user1", "Alice", "Alice Smith")

            user = await user_repo.get_user_by_name("Smith")

            assert user is not None
            assert user.author_display_name == "Alice Smith"

    @pytest.mark.asyncio
    async def test_get_user_statistics_database_error(self, user_repo):
        """Test handling of database errors in user statistics."""
        await user_repo.db_manager.initialize()

        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.side_effect = Exception("Database error")

            with pytest.raises(Exception, match="Database error"):
                await user_repo.get_user_statistics(limit=10)

    @pytest.mark.asyncio
    async def test_get_user_by_name_database_error(self, user_repo):
        """Test handling of database errors in get_user_by_name."""
        await user_repo.db_manager.initialize()

        with patch.object(user_repo.db_manager, "execute_single") as mock_execute:
            mock_execute.side_effect = Exception("Database error")

            with pytest.raises(Exception, match="Database error"):
                await user_repo.get_user_by_name("Alice")

    @pytest.mark.asyncio
    async def test_get_top_users_database_error(self, user_repo):
        """Test handling of database errors in get_top_users."""
        await user_repo.db_manager.initialize()

        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.side_effect = Exception("Database error")

            with pytest.raises(Exception, match="Database error"):
                await user_repo.get_top_users(limit=10)

    @pytest.mark.asyncio
    async def test_get_user_list(self, user_repo):
        """Test getting user list."""
        await user_repo.db_manager.initialize()

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                ("user1", "Alice", "Alice Smith"),
                ("user2", "Bob", None),
                ("user3", "Charlie", "Charlie Brown"),
            ]

            users = await user_repo.get_user_list()

            assert len(users) == 3
            assert users[0]["author_id"] == "user1"
            assert users[0]["author_name"] == "Alice"
            assert users[0]["display_name"] == "Alice Smith"
            assert users[1]["author_id"] == "user2"
            assert users[1]["author_name"] == "Bob"
            assert users[1]["display_name"] is None

    @pytest.mark.asyncio
    async def test_get_user_list_with_custom_filter(self, user_repo):
        """Test getting user list with custom filter."""
        await user_repo.db_manager.initialize()

        custom_filter = "author_name NOT LIKE '%bot%'"

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                ("user1", "Alice", "Alice Smith"),
                ("user2", "Bob", None),
            ]

            users = await user_repo.get_user_list(base_filter=custom_filter)

            assert len(users) == 2
            assert users[0]["author_name"] == "Alice"
            assert users[1]["author_name"] == "Bob"

    @pytest.mark.asyncio
    async def test_get_user_statistics_by_id(self, user_repo):
        """Test getting user statistics by ID."""
        await user_repo.db_manager.initialize()

        author_id = "user123"

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_single") as mock_execute:
            mock_execute.return_value = (
                150,
                5,
                45.5,
                25,
                "2024-01-01T12:00:00Z",
                "2024-12-01T12:00:00Z",
            )

            stats = await user_repo.get_user_statistics_by_id(author_id)

            assert stats is not None
            assert stats["total_messages"] == 150
            assert stats["channels_active"] == 5
            assert stats["avg_message_length"] == 45.5
            assert stats["active_days"] == 25
            assert stats["first_message"] == "2024-01-01T12:00:00Z"
            assert stats["last_message"] == "2024-12-01T12:00:00Z"

    @pytest.mark.asyncio
    async def test_get_user_statistics_by_id_not_found(self, user_repo):
        """Test getting user statistics by ID when user not found."""
        await user_repo.db_manager.initialize()

        author_id = "nonexistent"

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_single") as mock_execute:
            mock_execute.return_value = (0, 0, None, 0, None, None)

            stats = await user_repo.get_user_statistics_by_id(author_id)

            assert stats is None

    @pytest.mark.asyncio
    async def test_get_user_content_sample(self, user_repo):
        """Test getting user content sample."""
        await user_repo.db_manager.initialize()

        author_id = "user123"

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = [
                ("This is a sample message content",),
                ("Another interesting message here",),
                ("Yet another message with good content",),
            ]

            content = await user_repo.get_user_content_sample(author_id, limit=100)

            assert len(content) == 3
            assert "This is a sample message content" in content
            assert "Another interesting message here" in content
            assert "Yet another message with good content" in content

    @pytest.mark.asyncio
    async def test_get_user_content_sample_empty_result(self, user_repo):
        """Test getting user content sample with empty result."""
        await user_repo.db_manager.initialize()

        author_id = "nonexistent"

        # Mock database query results
        with patch.object(user_repo.db_manager, "execute_query") as mock_execute:
            mock_execute.return_value = []

            content = await user_repo.get_user_content_sample(author_id, limit=100)

            assert isinstance(content, list)
            assert len(content) == 0

    @pytest.mark.asyncio
    async def test_find_user_by_name(self, user_repo):
        """Test finding user by name (compatibility alias)."""
        await user_repo.db_manager.initialize()

        user_name = "Alice"

        # Mock the underlying get_user_by_name method
        with patch.object(user_repo, "get_user_by_name") as mock_get_user:
            mock_user = MagicMock()
            mock_user.author_id = "user123"
            mock_user.author_name = "Alice"
            mock_user.author_display_name = "Alice Smith"
            mock_get_user.return_value = mock_user

            result = await user_repo.find_user_by_name(user_name)

            assert result is not None
            assert result["author_id"] == "user123"
            assert result["author_name"] == "Alice"
            assert result["display_name"] == "Alice Smith"

    @pytest.mark.asyncio
    async def test_find_user_by_name_not_found(self, user_repo):
        """Test finding user by name when not found."""
        await user_repo.db_manager.initialize()

        user_name = "nonexistent"

        # Mock the underlying get_user_by_name method
        with patch.object(user_repo, "get_user_by_name") as mock_get_user:
            mock_get_user.return_value = None

            result = await user_repo.find_user_by_name(user_name)

            assert result is None
