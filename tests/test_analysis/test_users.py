"""Unit tests for user analysis."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pepino.analysis.user_analyzer import UserAnalyzer


@pytest.mark.asyncio
async def test_user_analyzer_get_top_users():
    """Test UserAnalyzer get_top_users with mocked repository"""
    # Mock database manager
    mock_db_manager = MagicMock()

    # Create analyzer
    analyzer = UserAnalyzer(mock_db_manager)

    # Mock the user repository's get_top_users method
    mock_top_users = [
        {
            "author_id": "user1",
            "author_name": "Alice",
            "display_name": "Alice Smith",
            "message_count": 150,
            "channels_active": 5,
            "avg_message_length": 45.5,
        },
        {
            "author_id": "user2",
            "author_name": "Bob",
            "display_name": "Bob Jones",
            "message_count": 120,
            "channels_active": 3,
            "avg_message_length": 38.2,
        },
    ]

    with patch.object(
        analyzer.user_repo, "get_top_users", new_callable=AsyncMock
    ) as mock_get_top_users:
        mock_get_top_users.return_value = mock_top_users

        top_users = await analyzer.get_top_users(5)

        assert len(top_users) == 2
        assert top_users[0]["author_name"] == "Alice"
        assert top_users[0]["message_count"] == 150
        assert top_users[1]["author_name"] == "Bob"
        assert top_users[1]["message_count"] == 120

        # Verify the repository method was called correctly
        mock_get_top_users.assert_called_once_with(5, analyzer.base_filter)


@pytest.mark.asyncio
async def test_user_analyzer_analyze_by_name():
    """Test UserAnalyzer analyze method by name with mocked repositories"""
    mock_db_manager = MagicMock()
    analyzer = UserAnalyzer(mock_db_manager)

    # Mock user lookup and statistics
    mock_user = {
        "author_id": "user123",
        "author_name": "alice",
        "author_display_name": "Alice Smith",
    }
    mock_stats = {
        "author_id": "user123",
        "author_name": "alice",
        "message_count": 150,
        "channels_active": 8,
        "avg_message_length": 42.5,
        "first_message_date": "2024-01-01T10:00:00Z",
        "last_message_date": "2024-12-01T15:00:00Z",
    }

    # Mock concepts
    mock_concepts = ["technology", "programming", "python", "data"]

    with patch.object(
        analyzer.user_repo, "find_user_by_name", new_callable=AsyncMock
    ) as mock_find_user, patch.object(
        analyzer.user_repo, "get_user_statistics", new_callable=AsyncMock
    ) as mock_get_stats, patch.object(
        analyzer, "_get_user_concepts", new_callable=AsyncMock
    ) as mock_get_concepts:
        mock_find_user.return_value = mock_user
        mock_get_stats.return_value = mock_stats
        mock_get_concepts.return_value = mock_concepts

        result = await analyzer.analyze(user_name="alice")

        assert result.success is True
        assert result.user_info.author_id == "user123"
        assert result.user_info.display_name == "Alice Smith"
        assert result.statistics.message_count == 150
        assert result.statistics.channels_active == 8
        assert result.statistics.avg_message_length == 42.5
        assert result.concepts == mock_concepts

        # Verify repository methods were called correctly
        mock_find_user.assert_called_once_with("alice")
        mock_get_stats.assert_called_once_with("user123", analyzer.base_filter)
        mock_get_concepts.assert_called_once_with("user123")


@pytest.mark.asyncio
async def test_user_analyzer_analyze_with_concepts():
    """Test UserAnalyzer analyze method with concept analysis enabled"""
    mock_db_manager = MagicMock()
    analyzer = UserAnalyzer(mock_db_manager)

    # Mock user lookup and statistics
    mock_user = {
        "author_id": "user456",
        "author_name": "bob",
        "author_display_name": "Bob Jones",
    }
    mock_stats = {
        "author_id": "user456",
        "author_name": "bob",
        "message_count": 89,
        "channels_active": 4,
        "avg_message_length": 38.1,
    }
    mock_concepts = ["javascript", "react", "nodejs", "web development"]

    with patch.object(
        analyzer.user_repo, "find_user_by_name", new_callable=AsyncMock
    ) as mock_find_user, patch.object(
        analyzer.user_repo, "get_user_statistics", new_callable=AsyncMock
    ) as mock_get_stats, patch.object(
        analyzer, "_get_user_concepts", new_callable=AsyncMock
    ) as mock_get_concepts:
        mock_find_user.return_value = mock_user
        mock_get_stats.return_value = mock_stats
        mock_get_concepts.return_value = mock_concepts

        result = await analyzer.analyze(user_name="bob", include_concepts=True)

        assert result.success is True
        assert result.user_info.author_id == "user456"
        assert result.statistics.message_count == 89
        assert len(result.concepts) == 4
        assert "javascript" in result.concepts
        assert "react" in result.concepts

        mock_get_concepts.assert_called_once_with("user456")


@pytest.mark.asyncio
async def test_user_analyzer_analyze_user_not_found():
    """Test UserAnalyzer analyze method when user is not found"""
    mock_db_manager = MagicMock()
    analyzer = UserAnalyzer(mock_db_manager)

    with patch.object(
        analyzer.user_repo, "find_user_by_name", new_callable=AsyncMock
    ) as mock_find_user:
        mock_find_user.return_value = None

        result = await analyzer.analyze(user_name="nonexistent")

        assert result.success is False
        assert "User 'nonexistent' not found" in result.error

        mock_find_user.assert_called_once_with("nonexistent")


@pytest.mark.asyncio
async def test_user_analyzer_analyze_no_data():
    """Test UserAnalyzer analyze method when user has no data"""
    mock_db_manager = MagicMock()
    analyzer = UserAnalyzer(mock_db_manager)

    # Mock user lookup but no statistics
    mock_user = {
        "author_id": "user789",
        "author_name": "charlie",
        "author_display_name": "Charlie Brown",
    }

    with patch.object(
        analyzer.user_repo, "find_user_by_name", new_callable=AsyncMock
    ) as mock_find_user, patch.object(
        analyzer.user_repo, "get_user_statistics", new_callable=AsyncMock
    ) as mock_get_stats:
        mock_find_user.return_value = mock_user
        mock_get_stats.return_value = {"message_count": 0}

        result = await analyzer.analyze(user_name="charlie")

        assert result.success is False
        assert "No data found for user 'charlie'" in result.error

        mock_find_user.assert_called_once_with("charlie")
        mock_get_stats.assert_called_once_with("user789", analyzer.base_filter)


@pytest.mark.asyncio
async def test_user_analyzer_get_available_users():
    """Test UserAnalyzer get_available_users with mocked repository"""
    mock_db_manager = MagicMock()
    analyzer = UserAnalyzer(mock_db_manager)

    mock_user_list = [
        {"author_id": "user1", "author_name": "Alice", "display_name": "Alice Smith"},
        {"author_id": "user2", "author_name": "Bob", "display_name": None},
        {"author_id": "user3", "author_name": "Carol", "display_name": "Carol Brown"},
    ]

    with patch.object(
        analyzer.user_repo, "get_user_list", new_callable=AsyncMock
    ) as mock_get_user_list:
        mock_get_user_list.return_value = mock_user_list

        users = await analyzer.get_available_users()

        assert len(users) == 3
        assert "Alice Smith" in users  # display_name preferred
        assert "Bob" in users  # author_name when no display_name
        assert "Carol Brown" in users  # display_name preferred

        mock_get_user_list.assert_called_once_with(analyzer.base_filter)


@pytest.mark.asyncio
async def test_user_analyzer_missing_parameters():
    """Test UserAnalyzer analyze method with missing required parameters"""
    mock_db_manager = MagicMock()
    analyzer = UserAnalyzer(mock_db_manager)

    # Call without any parameters
    result = await analyzer.analyze()

    assert result.success is False
    assert "Either user_name or user_id is required" in result.error
