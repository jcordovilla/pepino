"""Unit tests for Channel model."""

from datetime import datetime

import pytest

from pepino.data.models import Channel


class TestChannelModel:
    """Test cases for Channel model."""

    @pytest.mark.parametrize(
        "test_case",
        [
            {
                "channel_data": {
                    "channel_id": "ch1",
                    "channel_name": "general",
                    "message_count": 100,
                    "unique_users": 25,
                    "avg_message_length": 50.5,
                },
                "expected_fields": ["channel_id", "channel_name", "message_count"],
                "description": "basic channel",
            },
            {
                "channel_data": {
                    "channel_id": "ch2",
                    "channel_name": "random",
                    "message_count": 200,
                    "unique_users": 50,
                    "avg_message_length": 75.2,
                    "first_message": "2024-01-01T12:00:00Z",
                    "last_message": "2024-12-01T12:00:00Z",
                },
                "expected_fields": ["first_message", "last_message"],
                "description": "channel with message dates",
            },
            {
                "channel_data": {
                    "channel_id": "ch3",
                    "channel_name": "test",
                    "message_count": 0,
                    "unique_users": 0,
                    "avg_message_length": 0.0,
                },
                "expected_fields": ["message_count", "unique_users"],
                "description": "inactive channel",
            },
        ],
    )
    def test_channel_creation_variations(self, test_case):
        """Test channel creation with various data using table-driven tests."""
        channel_data = test_case["channel_data"]
        expected_fields = test_case["expected_fields"]

        # Create channel
        channel = Channel(**channel_data)

        # Check expected fields
        for field in expected_fields:
            assert hasattr(channel, field)
            assert getattr(channel, field) == channel_data[field]

    @pytest.mark.parametrize(
        "test_case",
        [
            {
                "channel_data": {
                    "channel_id": "ch1",
                    "channel_name": "general",
                    "message_count": -5,  # Negative value
                    "unique_users": 25,
                    "avg_message_length": 50.5,
                },
                "expected_message_count": 0,  # Should be normalized to 0
                "description": "negative message count",
            },
            {
                "channel_data": {
                    "channel_id": "ch2",
                    "channel_name": "random",
                    "message_count": 100,
                    "unique_users": -10,  # Negative value
                    "avg_message_length": 50.5,
                },
                "expected_unique_users": 0,  # Should be normalized to 0
                "description": "negative unique users",
            },
            {
                "channel_data": {
                    "channel_id": "ch3",
                    "channel_name": "test",
                    "message_count": 100,
                    "unique_users": 25,
                    "avg_message_length": -5.5,  # Negative value
                },
                "expected_avg_length": 0.0,  # Should be normalized to 0.0
                "description": "negative average length",
            },
        ],
    )
    def test_channel_validation_negative_values(self, test_case):
        """Test channel validation with negative values."""
        channel_data = test_case["channel_data"]
        expected_message_count = test_case.get("expected_message_count")
        expected_unique_users = test_case.get("expected_unique_users")
        expected_avg_length = test_case.get("expected_avg_length")

        # Create channel
        channel = Channel(**channel_data)

        # Check that negative values are normalized
        if expected_message_count is not None:
            assert channel.message_count == expected_message_count
        if expected_unique_users is not None:
            assert channel.unique_users == expected_unique_users
        if expected_avg_length is not None:
            assert channel.avg_message_length == expected_avg_length

    @pytest.mark.parametrize(
        "test_case",
        [
            {
                "message_count": 0,
                "expected_is_active": False,
                "expected_activity_level": "inactive",
                "description": "inactive channel",
            },
            {
                "message_count": 50,
                "expected_is_active": True,
                "expected_activity_level": "low",
                "description": "low activity channel",
            },
            {
                "message_count": 500,
                "expected_is_active": True,
                "expected_activity_level": "medium",
                "description": "medium activity channel",
            },
            {
                "message_count": 1500,
                "expected_is_active": True,
                "expected_activity_level": "high",
                "description": "high activity channel",
            },
        ],
    )
    def test_channel_activity_properties(self, test_case):
        """Test channel activity properties."""
        message_count = test_case["message_count"]
        expected_is_active = test_case["expected_is_active"]
        expected_activity_level = test_case["expected_activity_level"]

        channel = Channel(
            channel_id="ch1",
            channel_name="test",
            message_count=message_count,
            unique_users=25,
            avg_message_length=50.0,
        )

        assert channel.is_active == expected_is_active
        assert channel.activity_level == expected_activity_level

    def test_channel_required_fields(self):
        """Test that required fields are enforced."""
        # Should raise validation error for missing required fields
        with pytest.raises(ValueError):
            Channel()  # Missing channel_id and channel_name

    def test_channel_optional_fields_defaults(self):
        """Test that optional fields have correct defaults."""
        channel = Channel(channel_id="ch1", channel_name="test")

        assert channel.message_count == 0
        assert channel.unique_users == 0
        assert channel.avg_message_length == 0.0
        assert channel.first_message is None
        assert channel.last_message is None

    @pytest.mark.parametrize(
        "test_case",
        [
            {
                "channel_data": {
                    "channel_id": "ch1",
                    "channel_name": "general",
                    "message_count": 100,
                    "unique_users": 25,
                    "avg_message_length": 50.5,
                    "first_message": "2024-01-01T12:00:00Z",
                    "last_message": "2024-12-01T12:00:00Z",
                },
                "expected_dict_keys": [
                    "channel_id",
                    "channel_name",
                    "message_count",
                    "unique_users",
                    "avg_message_length",
                    "first_message",
                    "last_message",
                ],
                "description": "complete channel data",
            },
            {
                "channel_data": {
                    "channel_id": "ch2",
                    "channel_name": "random",
                    "message_count": 200,
                    "unique_users": 50,
                    "avg_message_length": 75.2,
                },
                "expected_dict_keys": [
                    "channel_id",
                    "channel_name",
                    "message_count",
                    "unique_users",
                    "avg_message_length",
                ],
                "description": "channel without dates",
            },
        ],
    )
    def test_channel_serialization(self, test_case):
        """Test channel serialization to dict."""
        channel_data = test_case["channel_data"]
        expected_dict_keys = test_case["expected_dict_keys"]

        channel = Channel(**channel_data)
        channel_dict = channel.model_dump()

        # Check that all expected keys are present
        for key in expected_dict_keys:
            assert key in channel_dict
            assert channel_dict[key] == channel_data.get(key)

    def test_channel_equality(self):
        """Test channel equality comparison."""
        channel1 = Channel(
            channel_id="ch1",
            channel_name="general",
            message_count=100,
            unique_users=25,
            avg_message_length=50.5,
        )

        channel2 = Channel(
            channel_id="ch1",
            channel_name="general",
            message_count=100,
            unique_users=25,
            avg_message_length=50.5,
        )

        channel3 = Channel(
            channel_id="ch2",
            channel_name="random",
            message_count=100,
            unique_users=25,
            avg_message_length=50.5,
        )

        assert channel1 == channel2
        assert channel1 != channel3

    def test_channel_string_representation(self):
        """Test channel string representation."""
        channel = Channel(
            channel_id="ch1",
            channel_name="general",
            message_count=100,
            unique_users=25,
            avg_message_length=50.5,
        )

        channel_str = str(channel)
        assert "general" in channel_str
        assert "ch1" in channel_str

    @pytest.mark.parametrize(
        "test_case",
        [
            {
                "field_name": "message_count",
                "invalid_value": "not_a_number",
                "description": "invalid message count type",
            },
            {
                "field_name": "unique_users",
                "invalid_value": "not_a_number",
                "description": "invalid unique users type",
            },
            {
                "field_name": "avg_message_length",
                "invalid_value": "not_a_number",
                "description": "invalid average length type",
            },
        ],
    )
    def test_channel_invalid_field_types(self, test_case):
        """Test channel validation with invalid field types."""
        field_name = test_case["field_name"]
        invalid_value = test_case["invalid_value"]

        channel_data = {
            "channel_id": "ch1",
            "channel_name": "test",
            "message_count": 100,
            "unique_users": 25,
            "avg_message_length": 50.5,
        }
        channel_data[field_name] = invalid_value

        # Should raise validation error for invalid type
        with pytest.raises(ValueError):
            Channel(**channel_data)

    def test_channel_with_none_values(self):
        """Test channel creation with None values for optional fields."""
        channel = Channel(
            channel_id="ch1",
            channel_name="test",
            message_count=100,
            unique_users=25,
            avg_message_length=50.5,
            first_message=None,
            last_message=None,
        )

        assert channel.first_message is None
        assert channel.last_message is None
        assert channel.is_active is True
        assert channel.activity_level == "medium"

    def test_channel_edge_case_zero_activity(self):
        """Test channel with exactly zero activity."""
        channel = Channel(
            channel_id="ch1",
            channel_name="inactive",
            message_count=0,
            unique_users=0,
            avg_message_length=0.0,
        )

        assert channel.is_active is False
        assert channel.activity_level == "inactive"
        assert channel.message_count == 0
        assert channel.unique_users == 0
        assert channel.avg_message_length == 0.0

    def test_channel_edge_case_boundary_activity_levels(self):
        """Test channel activity level boundaries."""
        # Test low activity boundary
        low_channel = Channel(
            channel_id="ch1",
            channel_name="low",
            message_count=99,  # Just below medium threshold
            unique_users=25,
            avg_message_length=50.0,
        )
        assert low_channel.activity_level == "low"

        # Test medium activity boundary
        medium_channel = Channel(
            channel_id="ch2",
            channel_name="medium",
            message_count=999,  # Just below high threshold
            unique_users=25,
            avg_message_length=50.0,
        )
        assert medium_channel.activity_level == "medium"

        # Test high activity boundary
        high_channel = Channel(
            channel_id="ch3",
            channel_name="high",
            message_count=1000,  # At high threshold
            unique_users=25,
            avg_message_length=50.0,
        )
        assert high_channel.activity_level == "high"
