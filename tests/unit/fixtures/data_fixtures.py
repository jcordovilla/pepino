"""Data package test fixtures."""

from datetime import datetime, timezone

import numpy as np
import pytest


@pytest.fixture
def sample_messages():
    """Sample message data for testing."""
    return [
        {
            "id": "1",
            "author_id": "user1",
            "author_name": "Alice",
            "author_display_name": "Alice Smith",
            "channel_id": "ch1",
            "channel_name": "general",
            "content": "Hello world! AI and ML are cool.",
            "timestamp": "2024-06-01T12:00:00Z",
            "is_bot": False,
            "has_reactions": True,
            "has_reference": False,
        },
        {
            "id": "2",
            "author_id": "user2",
            "author_name": "Bob",
            "author_display_name": "Bob Johnson",
            "channel_id": "ch1",
            "channel_name": "general",
            "content": "I love data science and machine learning.",
            "timestamp": "2024-06-01T13:00:00Z",
            "is_bot": False,
            "has_reactions": False,
            "has_reference": True,
        },
        {
            "id": "3",
            "author_id": "user1",
            "author_name": "Alice",
            "author_display_name": "Alice Smith",
            "channel_id": "ch2",
            "channel_name": "random",
            "content": "Python programming is essential for data science.",
            "timestamp": "2024-06-02T10:00:00Z",
            "is_bot": False,
            "has_reactions": True,
            "has_reference": False,
        },
        {
            "id": "4",
            "author_id": "bot1",
            "author_name": "TestBot",
            "author_display_name": "TestBot",
            "channel_id": "ch1",
            "channel_name": "general",
            "content": "I am a bot message.",
            "timestamp": "2024-06-01T14:00:00Z",
            "is_bot": True,
            "has_reactions": False,
            "has_reference": False,
        },
    ]


@pytest.fixture
def sample_users():
    """Sample user data for testing."""
    return [
        {
            "id": "user1",
            "name": "Alice",
            "display_name": "Alice Smith",
            "is_bot": False,
            "created_at": "2024-01-01T00:00:00Z",
            "last_seen": "2024-06-02T10:00:00Z",
        },
        {
            "id": "user2",
            "name": "Bob",
            "display_name": "Bob Johnson",
            "is_bot": False,
            "created_at": "2024-01-15T00:00:00Z",
            "last_seen": "2024-06-01T13:00:00Z",
        },
        {
            "id": "bot1",
            "name": "TestBot",
            "display_name": "TestBot",
            "is_bot": True,
            "created_at": "2024-02-01T00:00:00Z",
            "last_seen": "2024-06-01T14:00:00Z",
        },
    ]


@pytest.fixture
def sample_channels():
    """Sample channel data for testing."""
    return [
        {
            "id": "ch1",
            "name": "general",
            "type": "text",
            "created_at": "2024-01-01T00:00:00Z",
            "last_message_at": "2024-06-01T14:00:00Z",
        },
        {
            "id": "ch2",
            "name": "random",
            "type": "text",
            "created_at": "2024-01-01T00:00:00Z",
            "last_message_at": "2024-06-02T10:00:00Z",
        },
        {
            "id": "ch3",
            "name": "announcements",
            "type": "text",
            "created_at": "2024-01-01T00:00:00Z",
            "last_message_at": None,
        },
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embedding data for testing."""
    return [
        np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        np.array([0.2, 0.3, 0.4, 0.5, 0.6]),
        np.array([0.3, 0.4, 0.5, 0.6, 0.7]),
        np.array([0.4, 0.5, 0.6, 0.7, 0.8]),
    ]


@pytest.fixture
async def populated_db_manager(
    test_db_manager, sample_messages, sample_users, sample_channels
):
    """Create database manager with populated test data."""
    # Insert sample data
    for message in sample_messages:
        await test_db_manager.execute(
            """
            INSERT INTO messages (
                id, channel_id, channel_name, author_id, author_name, 
                author_display_name, content, timestamp, is_bot, 
                has_reactions, has_reference
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                message["id"],
                message["channel_id"],
                message["channel_name"],
                message["author_id"],
                message["author_name"],
                message["author_display_name"],
                message["content"],
                message["timestamp"],
                message["is_bot"],
                message["has_reactions"],
                message["has_reference"],
            ),
        )

    for user in sample_users:
        await test_db_manager.execute(
            """
            INSERT INTO users (id, name, display_name, is_bot, created_at, last_seen)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                user["id"],
                user["name"],
                user["display_name"],
                user["is_bot"],
                user["created_at"],
                user["last_seen"],
            ),
        )

    for channel in sample_channels:
        await test_db_manager.execute(
            """
            INSERT INTO channels (id, name, type, created_at, last_message_at)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                channel["id"],
                channel["name"],
                channel["type"],
                channel["created_at"],
                channel["last_message_at"],
            ),
        )

    try:
        yield test_db_manager
    finally:
        # Cleanup is handled by the test_db_manager fixture
        pass
