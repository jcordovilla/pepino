"""Unit tests for channel analysis."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pepino.analysis.channel_analyzer import ChannelAnalyzer


@pytest.mark.asyncio
async def test_channel_analyzer_analyze():
    """Test ChannelAnalyzer analyze method with mocked repositories"""
    mock_db_manager = MagicMock()
    analyzer = ChannelAnalyzer(mock_db_manager)

    # Mock message statistics
    mock_stats = {
        "total_messages": 500,
        "unique_users": 25,
        "bot_messages": 50,
        "human_messages": 450,
        "avg_message_length": 45.2,
        "first_message": "2024-01-01T12:00:00Z",
        "last_message": "2024-12-01T12:00:00Z",
    }

    # Mock top users
    mock_top_users = [
        {
            "author_id": "user1",
            "author_name": "Alice Smith",
            "message_count": 75,
            "avg_message_length": 48.5,
        },
        {
            "author_id": "user2",
            "author_name": "Bob Jones",
            "message_count": 62,
            "avg_message_length": 42.1,
        },
    ]

    with patch.object(
        analyzer.message_repo, "get_message_statistics", new_callable=AsyncMock
    ) as mock_get_stats, patch.object(
        analyzer.user_repo, "get_top_users_in_channel", new_callable=AsyncMock
    ) as mock_get_top_users:
        mock_get_stats.return_value = mock_stats
        mock_get_top_users.return_value = mock_top_users

        result = await analyzer.analyze(channel_name="general")

        assert result.success is True
        assert result.channel_info.channel_name == "general"
        assert result.statistics.total_messages == 500
        assert result.statistics.unique_users == 25
        assert len(result.top_users) == 2
        assert result.top_users[0].author_name == "Alice Smith"
        assert result.top_users[0].message_count == 75

        # Verify repository methods were called correctly
        mock_get_stats.assert_called_once_with("general")
        mock_get_top_users.assert_called_once_with("general", 10, analyzer.data_facade.base_filter)


@pytest.mark.asyncio
async def test_channel_analyzer_analyze_no_top_users():
    """Test ChannelAnalyzer analyze method without top users"""
    mock_db_manager = MagicMock()
    analyzer = ChannelAnalyzer(mock_db_manager)

    # Mock message statistics
    mock_stats = {
        "total_messages": 500,
        "unique_users": 25,
        "bot_messages": 50,
        "human_messages": 450,
        "avg_message_length": 45.2,
    }

    with patch.object(
        analyzer.message_repo, "get_message_statistics", new_callable=AsyncMock
    ) as mock_get_stats:
        mock_get_stats.return_value = mock_stats

        result = await analyzer.analyze(channel_name="general", include_top_users=False)

        assert result.success is True
        assert result.channel_info.channel_name == "general"
        assert result.statistics.total_messages == 500
        assert result.top_users == []  # No top users requested

        mock_get_stats.assert_called_once_with("general")


@pytest.mark.asyncio
async def test_channel_analyzer_analyze_custom_user_limit():
    """Test ChannelAnalyzer analyze method with custom user limit"""
    mock_db_manager = MagicMock()
    analyzer = ChannelAnalyzer(mock_db_manager)

    # Mock message statistics
    mock_stats = {"total_messages": 500, "unique_users": 25, "avg_message_length": 45.2}

    # Mock top users
    mock_top_users = [
        {
            "author_id": "user1",
            "author_name": "Alice",
            "message_count": 75,
            "avg_message_length": 48.5,
        }
    ]

    with patch.object(
        analyzer.message_repo, "get_message_statistics", new_callable=AsyncMock
    ) as mock_get_stats, patch.object(
        analyzer.user_repo, "get_top_users_in_channel", new_callable=AsyncMock
    ) as mock_get_top_users:
        mock_get_stats.return_value = mock_stats
        mock_get_top_users.return_value = mock_top_users

        result = await analyzer.analyze(channel_name="general", limit_users=5)

        assert result.success is True
        assert len(result.top_users) == 1

        # Verify custom limit was passed
        mock_get_top_users.assert_called_once_with("general", 5, analyzer.data_facade.base_filter)


@pytest.mark.asyncio
async def test_channel_analyzer_analyze_no_data():
    """Test ChannelAnalyzer analyze method when no data is found"""
    mock_db_manager = MagicMock()
    analyzer = ChannelAnalyzer(mock_db_manager)

    with patch.object(
        analyzer.message_repo, "get_message_statistics", new_callable=AsyncMock
    ) as mock_get_stats:
        # Return empty stats
        mock_get_stats.return_value = {"total_messages": 0}

        result = await analyzer.analyze(channel_name="nonexistent")

        assert result.success is False
        assert "No data found for channel 'nonexistent'" in result.error

        mock_get_stats.assert_called_once_with("nonexistent")


@pytest.mark.asyncio
async def test_channel_analyzer_analyze_missing_channel():
    """Test ChannelAnalyzer analyze method with missing channel parameter"""
    mock_db_manager = MagicMock()
    analyzer = ChannelAnalyzer(mock_db_manager)

    result = await analyzer.analyze()

    assert result.success is False
    assert "channel_name is required" in result.error


@pytest.mark.asyncio
async def test_channel_analyzer_get_available_channels():
    """Test ChannelAnalyzer get_available_channels with mocked repository"""
    mock_db_manager = MagicMock()
    analyzer = ChannelAnalyzer(mock_db_manager)

    mock_channels = ["general", "random", "announcements", "tech-talk"]

    with patch.object(
        analyzer.channel_repo, "get_channel_list", new_callable=AsyncMock
    ) as mock_get_channels:
        mock_get_channels.return_value = mock_channels

        channels = await analyzer.get_available_channels()

        assert isinstance(channels, list)
        assert len(channels) == 4
        assert "general" in channels
        assert "random" in channels
        assert "announcements" in channels
        assert "tech-talk" in channels

        mock_get_channels.assert_called_once_with(analyzer.data_facade.base_filter)


@pytest.mark.asyncio
async def test_channel_analyzer_get_top_channels():
    """Test ChannelAnalyzer get_top_channels with mocked repository"""
    mock_db_manager = MagicMock()
    analyzer = ChannelAnalyzer(mock_db_manager)

    mock_top_channels = [
        {
            "channel_id": "ch1",
            "channel_name": "general",
            "message_count": 1500,
            "unique_users": 50,
            "avg_message_length": 45.2,
        },
        {
            "channel_id": "ch2",
            "channel_name": "random",
            "message_count": 800,
            "unique_users": 30,
            "avg_message_length": 38.7,
        },
    ]

    with patch.object(
        analyzer.channel_repo, "get_top_channels", new_callable=AsyncMock
    ) as mock_get_top_channels:
        mock_get_top_channels.return_value = mock_top_channels

        channels = await analyzer.get_top_channels(limit=5)

        assert len(channels) == 2
        assert channels[0]["channel_name"] == "general"
        assert channels[0]["message_count"] == 1500
        assert channels[1]["channel_name"] == "random"
        assert channels[1]["message_count"] == 800

        mock_get_top_channels.assert_called_once_with(5, analyzer.data_facade.base_filter)


@pytest.mark.asyncio
async def test_channel_analyzer_repository_error():
    """Test ChannelAnalyzer when repository raises an exception"""
    mock_db_manager = MagicMock()
    analyzer = ChannelAnalyzer(mock_db_manager)

    with patch.object(
        analyzer.message_repo, "get_message_statistics", new_callable=AsyncMock
    ) as mock_get_stats:
        mock_get_stats.side_effect = Exception("Database connection error")

        result = await analyzer.analyze(channel_name="general")

        assert result.success is False
        assert "Analysis failed: Database connection error" in result.error
