"""Discord package test fixtures."""

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_discord_client():
    """Mock Discord client."""
    mock_client = AsyncMock()
    mock_guild = AsyncMock()
    mock_channel = AsyncMock()
    mock_message = AsyncMock()

    # Set up mock attributes
    mock_guild.channels = [mock_channel]
    mock_guild.members = []
    mock_client.guilds = [mock_guild]

    return mock_client


@pytest.fixture
def data_store():
    """Create test data store for Discord client."""
    return {"messages": {}, "users": {}, "channels": {}, "guilds": {}}


@pytest.fixture
def sample_discord_messages():
    """Sample Discord message data for testing."""
    return [
        {
            "id": "msg1",
            "content": "Hello world!",
            "author": {
                "id": "user1",
                "name": "Alice",
                "display_name": "Alice Smith",
                "bot": False,
            },
            "channel": {
                "id": "ch1",
                "name": "general",
                "guild": {"id": "guild1", "name": "Test Guild"},
            },
            "timestamp": "2024-06-01T12:00:00Z",
            "edited_timestamp": None,
            "reactions": [],
            "attachments": [],
            "embeds": [],
        },
        {
            "id": "msg2",
            "content": "I love data science!",
            "author": {
                "id": "user2",
                "name": "Bob",
                "display_name": "Bob Johnson",
                "bot": False,
            },
            "channel": {
                "id": "ch1",
                "name": "general",
                "guild": {"id": "guild1", "name": "Test Guild"},
            },
            "timestamp": "2024-06-01T13:00:00Z",
            "edited_timestamp": None,
            "reactions": [],
            "attachments": [],
            "embeds": [],
        },
    ]


@pytest.fixture
def sample_discord_channels():
    """Sample Discord channel data for testing."""
    return [
        {
            "id": "ch1",
            "name": "general",
            "type": "text",
            "guild": {"id": "guild1", "name": "Test Guild"},
            "permissions": {
                "read_messages": True,
                "send_messages": True,
                "read_message_history": True,
            },
        },
        {
            "id": "ch2",
            "name": "random",
            "type": "text",
            "guild": {"id": "guild1", "name": "Test Guild"},
            "permissions": {
                "read_messages": True,
                "send_messages": True,
                "read_message_history": True,
            },
        },
    ]


@pytest.fixture
def sample_discord_guilds():
    """Sample Discord guild data for testing."""
    return [
        {
            "id": "guild1",
            "name": "Test Guild",
            "member_count": 100,
            "channels": [],
            "members": [],
        }
    ]
