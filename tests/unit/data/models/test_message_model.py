"""Unit tests for Message model."""

from datetime import datetime

import pytest

from pepino.data.models import Message


class TestMessageModel:
    """Test cases for Message model."""

    @pytest.mark.parametrize(
        "test_case",
        [
            {
                "message_data": {
                    "id": "msg1",
                    "channel_id": "ch1",
                    "channel_name": "general",
                    "author_id": "user1",
                    "author_name": "Alice",
                    "content": "Hello world",
                    "timestamp": "2024-01-01T12:00:00Z",
                },
                "expected_fields": ["id", "channel_id", "author_id", "content"],
                "description": "basic message",
            },
            {
                "message_data": {
                    "id": "msg2",
                    "channel_id": "ch2",
                    "channel_name": "random",
                    "author_id": "user2",
                    "author_name": "Bob",
                    "author_display_name": "Bob Smith",
                    "content": "Test message with display name",
                    "timestamp": "2024-01-01T13:00:00Z",
                    "author_is_bot": True,
                },
                "expected_fields": ["author_display_name", "author_is_bot"],
                "description": "message with display name and bot flag",
            },
            {
                "message_data": {
                    "id": "msg3",
                    "channel_id": "ch3",
                    "channel_name": "test",
                    "author_id": "user3",
                    "author_name": "Carol",
                    "content": "",
                    "timestamp": "2024-01-01T14:00:00Z",
                    "has_attachments": True,
                    "has_embeds": True,
                },
                "expected_fields": ["has_attachments", "has_embeds"],
                "description": "message with attachments and embeds",
            },
        ],
    )
    def test_message_creation_variations(self, test_case):
        """Test message creation with various data using table-driven tests."""
        message_data = test_case["message_data"]
        expected_fields = test_case["expected_fields"]

        # Create message
        message = Message(**message_data)

        # Check expected fields
        for field in expected_fields:
            assert hasattr(message, field)
            assert getattr(message, field) == message_data[field]

    @pytest.mark.parametrize(
        "test_case",
        [
            {
                "timestamp_str": "2024-01-01T12:00:00Z",
                "expected_type": datetime,
                "description": "ISO format with Z",
            },
            {
                "timestamp_str": "2024-01-01T12:00:00+00:00",
                "expected_type": datetime,
                "description": "ISO format with timezone",
            },
            {
                "timestamp_str": "2024-01-01T12:00:00",
                "expected_type": datetime,
                "description": "ISO format without timezone",
            },
        ],
    )
    def test_timestamp_parsing_variations(self, test_case):
        """Test timestamp parsing with various formats."""
        timestamp_str = test_case["timestamp_str"]
        expected_type = test_case["expected_type"]

        message = Message(
            id="msg1",
            channel_id="ch1",
            channel_name="general",
            author_id="user1",
            author_name="Alice",
            content="Test message",
            timestamp=timestamp_str,
        )

        assert isinstance(message.timestamp, expected_type)

    def test_content_validation_empty_string(self):
        """Test content validation with empty string."""
        message = Message(
            id="msg1",
            channel_id="ch1",
            channel_name="general",
            author_id="user1",
            author_name="Alice",
            content="   ",  # Whitespace only
            timestamp="2024-01-01T12:00:00Z",
        )

        assert message.content == ""  # Should be normalized to empty string

    @pytest.mark.parametrize(
        "test_case",
        [
            {
                "author_is_bot": False,
                "is_system": False,
                "is_webhook": False,
                "expected_is_human": True,
                "description": "human message",
            },
            {
                "author_is_bot": True,
                "is_system": False,
                "is_webhook": False,
                "expected_is_human": False,
                "description": "bot message",
            },
            {
                "author_is_bot": False,
                "is_system": True,
                "is_webhook": False,
                "expected_is_human": False,
                "description": "system message",
            },
            {
                "author_is_bot": False,
                "is_system": False,
                "is_webhook": True,
                "expected_is_human": False,
                "description": "webhook message",
            },
        ],
    )
    def test_is_human_property(self, test_case):
        """Test is_human property logic."""
        author_is_bot = test_case["author_is_bot"]
        is_system = test_case["is_system"]
        is_webhook = test_case["is_webhook"]
        expected_is_human = test_case["expected_is_human"]

        message = Message(
            id="msg1",
            channel_id="ch1",
            channel_name="general",
            author_id="user1",
            author_name="Alice",
            content="Test message",
            timestamp="2024-01-01T12:00:00Z",
            author_is_bot=author_is_bot,
            is_system=is_system,
            is_webhook=is_webhook,
        )

        assert message.is_human == expected_is_human

    def test_display_name_property(self):
        """Test display_name property logic."""
        # Test with display name
        message1 = Message(
            id="msg1",
            channel_id="ch1",
            channel_name="general",
            author_id="user1",
            author_name="Alice",
            author_display_name="Alice Smith",
            content="Test message",
            timestamp="2024-01-01T12:00:00Z",
        )
        assert message1.display_name == "Alice Smith"

        # Test without display name
        message2 = Message(
            id="msg2",
            channel_id="ch1",
            channel_name="general",
            author_id="user2",
            author_name="Bob",
            content="Test message",
            timestamp="2024-01-01T12:00:00Z",
        )
        assert message2.display_name == "Bob"

    def test_content_length_property(self):
        """Test content_length property."""
        # Test with content
        message1 = Message(
            id="msg1",
            channel_id="ch1",
            channel_name="general",
            author_id="user1",
            author_name="Alice",
            content="Hello world",
            timestamp="2024-01-01T12:00:00Z",
        )
        assert message1.content_length == 11

        # Test with empty content
        message2 = Message(
            id="msg2",
            channel_id="ch1",
            channel_name="general",
            author_id="user2",
            author_name="Bob",
            content="",
            timestamp="2024-01-01T12:00:00Z",
        )
        assert message2.content_length == 0

        # Test with whitespace-only content (should be normalized to empty)
        message3 = Message(
            id="msg3",
            channel_id="ch1",
            channel_name="general",
            author_id="user3",
            author_name="Carol",
            content="   ",
            timestamp="2024-01-01T12:00:00Z",
        )
        assert message3.content_length == 0

    def test_message_required_fields(self):
        """Test that required fields are enforced."""
        # Should raise validation error for missing required fields
        with pytest.raises(ValueError):
            Message()  # Missing required fields

    def test_message_optional_fields_defaults(self):
        """Test that optional fields have correct defaults."""
        message = Message(
            id="msg1",
            channel_id="ch1",
            channel_name="general",
            author_id="user1",
            author_name="Alice",
            content="Test message",
            timestamp="2024-01-01T12:00:00Z",
        )

        assert message.author_display_name is None
        assert message.edited_timestamp is None
        assert message.author_is_bot is False
        assert message.is_system is False
        assert message.is_webhook is False
        assert message.has_attachments is False
        assert message.has_embeds is False
        assert message.has_stickers is False
        assert message.has_mentions is False
        assert message.has_reactions is False
        assert message.has_reference is False
        assert message.referenced_message_id is None

    @pytest.mark.parametrize(
        "test_case",
        [
            {
                "message_data": {
                    "id": "msg1",
                    "channel_id": "ch1",
                    "channel_name": "general",
                    "author_id": "user1",
                    "author_name": "Alice",
                    "content": "Hello world",
                    "timestamp": "2024-01-01T12:00:00Z",
                    "author_is_bot": True,
                    "has_attachments": True,
                },
                "expected_dict_keys": [
                    "id",
                    "channel_id",
                    "author_id",
                    "content",
                    "author_is_bot",
                    "has_attachments",
                ],
                "description": "message with boolean flags",
            }
        ],
    )
    def test_message_serialization(self, test_case):
        """Test message serialization to dict."""
        message_data = test_case["message_data"]
        expected_dict_keys = test_case["expected_dict_keys"]

        message = Message(**message_data)
        message_dict = message.model_dump()

        # Check that all expected keys are present
        for key in expected_dict_keys:
            assert key in message_dict
            assert message_dict[key] == message_data.get(key)

    def test_message_equality(self):
        """Test message equality comparison."""
        message1 = Message(
            id="msg1",
            channel_id="ch1",
            channel_name="general",
            author_id="user1",
            author_name="Alice",
            content="Hello world",
            timestamp="2024-01-01T12:00:00Z",
        )

        message2 = Message(
            id="msg1",
            channel_id="ch1",
            channel_name="general",
            author_id="user1",
            author_name="Alice",
            content="Hello world",
            timestamp="2024-01-01T12:00:00Z",
        )

        message3 = Message(
            id="msg2",
            channel_id="ch1",
            channel_name="general",
            author_id="user1",
            author_name="Alice",
            content="Hello world",
            timestamp="2024-01-01T12:00:00Z",
        )

        assert message1 == message2
        assert message1 != message3

    def test_message_string_representation(self):
        """Test message string representation."""
        message = Message(
            id="msg1",
            channel_id="ch1",
            channel_name="general",
            author_id="user1",
            author_name="Alice",
            content="Hello world",
            timestamp="2024-01-01T12:00:00Z",
        )

        message_str = str(message)
        assert "Hello world" in message_str
        assert "msg1" in message_str

    def test_from_db_row_method(self):
        """Test from_db_row class method."""
        # Test with dict input
        row_dict = {
            "id": "msg1",
            "channel_id": "ch1",
            "channel_name": "general",
            "author_id": "user1",
            "author_name": "Alice",
            "content": "Hello world",
            "timestamp": "2024-01-01T12:00:00Z",
        }

        message = Message.from_db_row(row_dict)
        assert message.id == "msg1"
        assert message.content == "Hello world"

    def test_message_with_edited_timestamp(self):
        """Test message with edited timestamp."""
        message = Message(
            id="msg1",
            channel_id="ch1",
            channel_name="general",
            author_id="user1",
            author_name="Alice",
            content="Hello world",
            timestamp="2024-01-01T12:00:00Z",
            edited_timestamp="2024-01-01T12:05:00Z",
        )

        assert isinstance(message.edited_timestamp, datetime)
        assert message.edited_timestamp > message.timestamp

    def test_message_with_referenced_message(self):
        """Test message with referenced message."""
        message = Message(
            id="msg1",
            channel_id="ch1",
            channel_name="general",
            author_id="user1",
            author_name="Alice",
            content="Hello world",
            timestamp="2024-01-01T12:00:00Z",
            referenced_message_id="msg0",
            has_reference=True,
        )

        assert message.referenced_message_id == "msg0"
        assert message.has_reference is True
