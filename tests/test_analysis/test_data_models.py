"""Unit tests for data models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from pepino.data.models import Channel, Message, User


def test_message_creation():
    """Test Message model creation."""
    timestamp = datetime.now()
    message = Message(
        id="123",
        author_id="456",
        author_name="TestUser",
        channel_id="789",
        channel_name="test-channel",
        content="Hello, world!",
        timestamp=timestamp,
        author_is_bot=False,
        has_reactions=True,
        has_reference=False,
    )

    assert message.id == "123"
    assert message.author_id == "456"
    assert message.author_name == "TestUser"
    assert message.channel_id == "789"
    assert message.channel_name == "test-channel"
    assert message.content == "Hello, world!"
    assert message.timestamp == timestamp
    assert message.author_is_bot is False
    assert message.has_reactions is True
    assert message.has_reference is False


def test_message_optional_fields():
    """Test Message model with optional fields."""
    message = Message(
        id="123",
        author_id="456",
        author_name="TestUser",
        channel_id="789",
        channel_name="test-channel",
        content="Hello, world!",
        timestamp=datetime.now(),
    )

    # Optional fields should have default values
    assert message.author_is_bot is False
    assert message.has_reactions is False
    assert message.has_reference is False
    assert message.referenced_message_id is None
    assert message.author_display_name is None
    assert message.edited_timestamp is None


def test_message_validation():
    """Test Message model validation."""
    # Test that missing required fields are enforced
    with pytest.raises(ValidationError):
        Message(
            # id missing - should fail
            author_id="456",
            author_name="TestUser",
            channel_id="789",
            channel_name="test-channel",
            content="Hello, world!",
            timestamp=datetime.now(),
        )

    # Test that empty content is allowed (should be converted to empty string)
    message = Message(
        id="123",
        author_id="456",
        author_name="TestUser",
        channel_id="789",
        channel_name="test-channel",
        content="   ",  # Whitespace should be converted to empty string
        timestamp=datetime.now(),
    )
    assert message.content == ""


def test_message_bot_detection():
    """Test Message bot detection logic."""
    # Test bot message
    bot_message = Message(
        id="123",
        author_id="456",
        author_name="BotUser",
        channel_id="789",
        channel_name="test-channel",
        content="Hello, world!",
        timestamp=datetime.now(),
        author_is_bot=True,
    )

    assert bot_message.author_is_bot is True
    assert bot_message.is_human is False

    # Test human message
    human_message = Message(
        id="124",
        author_id="457",
        author_name="HumanUser",
        channel_id="789",
        channel_name="test-channel",
        content="Hello, world!",
        timestamp=datetime.now(),
        author_is_bot=False,
    )

    assert human_message.author_is_bot is False
    assert human_message.is_human is True


def test_message_display_name_fallback():
    """Test Message display name fallback logic."""
    # Message with display name
    message_with_display = Message(
        id="123",
        author_id="456",
        author_name="TestUser",
        author_display_name="Display Name",
        channel_id="789",
        channel_name="test-channel",
        content="Hello, world!",
        timestamp=datetime.now(),
    )

    # Message without display name (should fall back to author_name)
    message_without_display = Message(
        id="124",
        author_id="457",
        author_name="TestUser2",
        channel_id="789",
        channel_name="test-channel",
        content="Hello, world!",
        timestamp=datetime.now(),
    )

    # Test that display name logic works correctly
    assert message_with_display.display_name == "Display Name"
    assert message_without_display.display_name == "TestUser2"


def test_user_creation():
    """Test User model creation."""
    user = User(
        author_id="123",
        author_name="TestUser",
        author_display_name="Display Name",
        message_count=100,
        channels_active=5,
        avg_message_length=50.5,
    )

    assert user.author_id == "123"
    assert user.author_name == "TestUser"
    assert user.author_display_name == "Display Name"
    assert user.message_count == 100
    assert user.channels_active == 5
    assert user.avg_message_length == 50.5


def test_channel_creation():
    """Test Channel model creation."""
    channel = Channel(
        channel_id="123",
        channel_name="test-channel",
        message_count=1000,
        unique_users=50,
        avg_message_length=75.2,
    )

    assert channel.channel_id == "123"
    assert channel.channel_name == "test-channel"
    assert channel.message_count == 1000
    assert channel.unique_users == 50
    assert channel.avg_message_length == 75.2


def test_message_from_db_row():
    """Test Message creation from database row."""
    timestamp = datetime.now()
    db_row = (
        "123",
        "789",
        "test-channel",
        "456",
        "TestUser",
        "Display Name",
        "Hello, world!",
        timestamp.isoformat(),
        None,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        None,
    )

    message = Message.from_db_row(db_row)

    assert message.id == "123"
    assert message.author_id == "456"
    assert message.author_name == "TestUser"
    assert message.channel_id == "789"
    assert message.channel_name == "test-channel"
    assert message.content == "Hello, world!"
    assert message.has_reactions is True


def test_user_display_name_fallback():
    """Test User display_name property fallback."""
    user = User(
        author_id="123",
        author_name="Alice",
        author_display_name=None,  # No display name
        message_count=100,
    )

    assert user.display_name == "Alice"  # Should fall back to author_name


def test_user_activity_status():
    """Test User activity status."""
    # Active user
    active_user = User(author_id="123", author_name="Alice", message_count=100)
    assert active_user.is_active is True

    # Inactive user
    inactive_user = User(author_id="124", author_name="Bob", message_count=0)
    assert inactive_user.is_active is False


def test_channel_activity_levels():
    """Test Channel activity level property."""
    # High activity channel
    high_activity = Channel(
        channel_id="123", channel_name="general", message_count=1500
    )
    assert high_activity.activity_level == "high"

    # Medium activity channel
    medium_activity = Channel(
        channel_id="124", channel_name="random", message_count=500
    )
    assert medium_activity.activity_level == "medium"

    # Low activity channel
    low_activity = Channel(channel_id="125", channel_name="quiet", message_count=50)
    assert low_activity.activity_level == "low"


def test_channel_inactive_status():
    """Test Channel inactive status."""
    inactive_channel = Channel(
        channel_id="123", channel_name="inactive", message_count=0
    )

    assert inactive_channel.is_active is False
    assert inactive_channel.activity_level == "inactive"
