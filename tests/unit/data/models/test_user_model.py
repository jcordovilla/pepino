"""Unit tests for User model."""

from datetime import datetime

import pytest

from pepino.data.models import User


class TestUserModel:
    """Test cases for User model."""

    @pytest.mark.parametrize(
        "test_case",
        [
            {
                "user_data": {
                    "author_id": "user1",
                    "author_name": "Alice",
                    "message_count": 100,
                    "channels_active": 5,
                    "avg_message_length": 50.5,
                },
                "expected_fields": ["author_id", "author_name", "message_count"],
                "description": "basic user",
            },
            {
                "user_data": {
                    "author_id": "user2",
                    "author_name": "Bob",
                    "author_display_name": "Bob Smith",
                    "message_count": 200,
                    "channels_active": 10,
                    "avg_message_length": 75.2,
                    "first_message_date": "2024-01-01",
                    "last_message_date": "2024-12-01",
                },
                "expected_fields": [
                    "author_display_name",
                    "first_message_date",
                    "last_message_date",
                ],
                "description": "user with display name and dates",
            },
            {
                "user_data": {
                    "author_id": "user3",
                    "author_name": "Carol",
                    "message_count": 0,
                    "channels_active": 0,
                    "avg_message_length": 0.0,
                },
                "expected_fields": ["message_count", "channels_active"],
                "description": "inactive user",
            },
        ],
    )
    def test_user_creation_variations(self, test_case):
        """Test user creation with various data using table-driven tests."""
        user_data = test_case["user_data"]
        expected_fields = test_case["expected_fields"]

        # Create user
        user = User(**user_data)

        # Check expected fields
        for field in expected_fields:
            assert hasattr(user, field)
            assert getattr(user, field) == user_data[field]

    @pytest.mark.parametrize(
        "test_case",
        [
            {
                "user_data": {
                    "author_id": "user1",
                    "author_name": "Alice",
                    "message_count": -5,  # Negative value
                    "channels_active": 5,
                    "avg_message_length": 50.5,
                },
                "expected_message_count": 0,  # Should be normalized to 0
                "description": "negative message count",
            },
            {
                "user_data": {
                    "author_id": "user2",
                    "author_name": "Bob",
                    "message_count": 100,
                    "channels_active": -2,  # Negative value
                    "avg_message_length": 50.5,
                },
                "expected_channels_active": 0,  # Should be normalized to 0
                "description": "negative channels active",
            },
            {
                "user_data": {
                    "author_id": "user3",
                    "author_name": "Carol",
                    "message_count": 100,
                    "channels_active": 5,
                    "avg_message_length": -5.5,  # Negative value
                },
                "expected_avg_length": 0.0,  # Should be normalized to 0.0
                "description": "negative average length",
            },
        ],
    )
    def test_user_validation_negative_values(self, test_case):
        """Test user validation with negative values."""
        user_data = test_case["user_data"]
        expected_message_count = test_case.get("expected_message_count")
        expected_channels_active = test_case.get("expected_channels_active")
        expected_avg_length = test_case.get("expected_avg_length")

        # Create user
        user = User(**user_data)

        # Check that negative values are normalized
        if expected_message_count is not None:
            assert user.message_count == expected_message_count
        if expected_channels_active is not None:
            assert user.channels_active == expected_channels_active
        if expected_avg_length is not None:
            assert user.avg_message_length == expected_avg_length

    @pytest.mark.parametrize(
        "test_case",
        [
            {
                "message_count": 0,
                "expected_is_active": False,
                "description": "inactive user",
            },
            {
                "message_count": 1,
                "expected_is_active": True,
                "description": "active user with one message",
            },
            {
                "message_count": 100,
                "expected_is_active": True,
                "description": "very active user",
            },
        ],
    )
    def test_user_activity_property(self, test_case):
        """Test user activity property."""
        message_count = test_case["message_count"]
        expected_is_active = test_case["expected_is_active"]

        user = User(
            author_id="user1",
            author_name="Test User",
            message_count=message_count,
            channels_active=5,
            avg_message_length=50.0,
        )

        assert user.is_active == expected_is_active

    def test_user_required_fields(self):
        """Test that required fields are enforced."""
        # Should raise validation error for missing required fields
        with pytest.raises(ValueError):
            User()  # Missing author_id and author_name

    def test_user_optional_fields_defaults(self):
        """Test that optional fields have correct defaults."""
        user = User(author_id="user1", author_name="Test User")

        assert user.author_display_name is None
        assert user.message_count == 0
        assert user.channels_active == 0
        assert user.avg_message_length == 0.0
        assert user.first_message_date is None
        assert user.last_message_date is None

    @pytest.mark.parametrize(
        "test_case",
        [
            {
                "user_data": {
                    "author_id": "user1",
                    "author_name": "Alice",
                    "author_display_name": "Alice Smith",
                    "message_count": 100,
                    "channels_active": 5,
                    "avg_message_length": 50.5,
                    "first_message_date": "2024-01-01",
                    "last_message_date": "2024-12-01",
                },
                "expected_dict_keys": [
                    "author_id",
                    "author_name",
                    "author_display_name",
                    "message_count",
                    "channels_active",
                    "avg_message_length",
                    "first_message_date",
                    "last_message_date",
                ],
                "description": "complete user data",
            },
            {
                "user_data": {
                    "author_id": "user2",
                    "author_name": "Bob",
                    "message_count": 200,
                    "channels_active": 10,
                    "avg_message_length": 75.2,
                },
                "expected_dict_keys": [
                    "author_id",
                    "author_name",
                    "message_count",
                    "channels_active",
                    "avg_message_length",
                ],
                "description": "user without display name and dates",
            },
        ],
    )
    def test_user_serialization(self, test_case):
        """Test user serialization to dict."""
        user_data = test_case["user_data"]
        expected_dict_keys = test_case["expected_dict_keys"]

        user = User(**user_data)
        user_dict = user.model_dump()

        # Check that all expected keys are present
        for key in expected_dict_keys:
            assert key in user_dict
            assert user_dict[key] == user_data.get(key)

    def test_user_equality(self):
        """Test user equality comparison."""
        user1 = User(
            author_id="user1",
            author_name="Alice",
            message_count=100,
            channels_active=5,
            avg_message_length=50.5,
        )

        user2 = User(
            author_id="user1",
            author_name="Alice",
            message_count=100,
            channels_active=5,
            avg_message_length=50.5,
        )

        user3 = User(
            author_id="user2",
            author_name="Bob",
            message_count=100,
            channels_active=5,
            avg_message_length=50.5,
        )

        assert user1 == user2
        assert user1 != user3

    def test_user_string_representation(self):
        """Test user string representation."""
        user = User(
            author_id="user1",
            author_name="Alice",
            message_count=100,
            channels_active=5,
            avg_message_length=50.5,
        )

        user_str = str(user)
        assert "Alice" in user_str
        assert "user1" in user_str

    @pytest.mark.parametrize(
        "test_case",
        [
            {
                "field_name": "message_count",
                "invalid_value": "not_a_number",
                "description": "invalid message count type",
            },
            {
                "field_name": "channels_active",
                "invalid_value": "not_a_number",
                "description": "invalid channels active type",
            },
            {
                "field_name": "avg_message_length",
                "invalid_value": "not_a_number",
                "description": "invalid average length type",
            },
        ],
    )
    def test_user_invalid_field_types(self, test_case):
        """Test user validation with invalid field types."""
        field_name = test_case["field_name"]
        invalid_value = test_case["invalid_value"]

        user_data = {
            "author_id": "user1",
            "author_name": "Test User",
            "message_count": 100,
            "channels_active": 5,
            "avg_message_length": 50.5,
        }
        user_data[field_name] = invalid_value

        # Should raise validation error for invalid type
        with pytest.raises(ValueError):
            User(**user_data)

    def test_user_with_none_values(self):
        """Test user creation with None values for optional fields."""
        user = User(
            author_id="user1",
            author_name="Test User",
            message_count=100,
            channels_active=5,
            avg_message_length=50.5,
            author_display_name=None,
            first_message_date=None,
            last_message_date=None,
        )

        assert user.author_display_name is None
        assert user.first_message_date is None
        assert user.last_message_date is None
        assert user.is_active is True

    def test_user_edge_case_zero_activity(self):
        """Test user with exactly zero activity."""
        user = User(
            author_id="user1",
            author_name="Inactive User",
            message_count=0,
            channels_active=0,
            avg_message_length=0.0,
        )

        assert user.is_active is False
        assert user.message_count == 0
        assert user.channels_active == 0
        assert user.avg_message_length == 0.0

    def test_display_name_property(self):
        """Test display_name property logic."""
        # Test with display name
        user1 = User(
            author_id="user1",
            author_name="Alice",
            author_display_name="Alice Smith",
            message_count=100,
            channels_active=5,
            avg_message_length=50.5,
        )
        assert user1.display_name == "Alice Smith"

        # Test without display name
        user2 = User(
            author_id="user2",
            author_name="Bob",
            message_count=200,
            channels_active=10,
            avg_message_length=75.2,
        )
        assert user2.display_name == "Bob"

    def test_user_with_date_strings(self):
        """Test user with date string fields."""
        user = User(
            author_id="user1",
            author_name="Test User",
            message_count=100,
            channels_active=5,
            avg_message_length=50.5,
            first_message_date="2024-01-01",
            last_message_date="2024-12-01",
        )

        assert user.first_message_date == "2024-01-01"
        assert user.last_message_date == "2024-12-01"

    def test_user_with_none_display_name(self):
        """Test user with None display name."""
        user = User(
            author_id="user1",
            author_name="Test User",
            author_display_name=None,
            message_count=100,
            channels_active=5,
            avg_message_length=50.5,
        )

        assert user.author_display_name is None
        assert user.display_name == "Test User"  # Should fall back to author_name

    def test_user_activity_thresholds(self):
        """Test user activity at different thresholds."""
        # Test exactly at threshold
        user1 = User(
            author_id="user1",
            author_name="Threshold User",
            message_count=1,  # Exactly at active threshold
            channels_active=1,
            avg_message_length=10.0,
        )
        assert user1.is_active is True

        # Test just below threshold
        user2 = User(
            author_id="user2",
            author_name="Inactive User",
            message_count=0,  # Just below active threshold
            channels_active=0,
            avg_message_length=0.0,
        )
        assert user2.is_active is False
