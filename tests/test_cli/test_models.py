"""
Tests for CLI command parameter models.
"""

import pytest
from pydantic import ValidationError

from pepino.cli.models import (
    ChannelAnalysisParams,
    ListChannelsParams,
    ListUsersParams,
    OutputFormat,
    TemporalAnalysisParams,
    TopicsAnalysisParams,
    UserAnalysisParams,
)


class TestOutputFormat:
    """Test OutputFormat enum."""

    def test_output_format_values(self):
        """Test that all expected output formats exist."""
        assert OutputFormat.JSON == "json"
        assert OutputFormat.CSV == "csv"
        assert OutputFormat.TEXT == "text"


class TestUserAnalysisParams:
    """Test UserAnalysisParams model."""

    def test_valid_user_analysis_params(self):
        """Test creating valid UserAnalysisParams."""
        params = UserAnalysisParams(
            user="alice", limit=5, output="output.json", output_format=OutputFormat.JSON
        )
        assert params.user == "alice"
        assert params.limit == 5
        assert params.output == "output.json"
        assert params.output_format == OutputFormat.JSON

    def test_default_values(self):
        """Test default parameter values."""
        params = UserAnalysisParams()
        assert params.user is None
        assert params.limit == 10
        assert params.output is None
        assert params.output_format == OutputFormat.JSON

    def test_user_validation(self):
        """Test username validation and cleaning."""
        # Valid usernames
        params1 = UserAnalysisParams(user="alice")
        assert params1.user == "alice"

        params2 = UserAnalysisParams(user="@alice")
        assert params2.user == "alice"  # @ prefix removed

        params3 = UserAnalysisParams(user="  bob  ")
        assert params3.user == "bob"  # Whitespace trimmed

        # None user (valid for "top users" mode)
        params4 = UserAnalysisParams(user=None)
        assert params4.user is None

        # Empty string becomes None
        params5 = UserAnalysisParams(user="   ")
        assert params5.user is None


class TestChannelAnalysisParams:
    """Test ChannelAnalysisParams model."""

    def test_valid_channel_analysis_params(self):
        """Test creating valid ChannelAnalysisParams."""
        params = ChannelAnalysisParams(
            channel="general",
            limit=5,
            output="channels.csv",
            output_format=OutputFormat.CSV,
        )
        assert params.channel == "general"
        assert params.limit == 5
        assert params.output == "channels.csv"
        assert params.output_format == OutputFormat.CSV

    def test_channel_validation(self):
        """Test channel name validation and cleaning."""
        # Valid channel names
        params1 = ChannelAnalysisParams(channel="general")
        assert params1.channel == "general"

        params2 = ChannelAnalysisParams(channel="#general")
        assert params2.channel == "general"  # # prefix removed

        params3 = ChannelAnalysisParams(channel="  random  ")
        assert params3.channel == "random"  # Whitespace trimmed

        # None channel (valid for "top channels" mode)
        params4 = ChannelAnalysisParams(channel=None)
        assert params4.channel is None


class TestTopicsAnalysisParams:
    """Test TopicsAnalysisParams model."""

    def test_valid_topics_analysis_params(self):
        """Test creating valid TopicsAnalysisParams."""
        params = TopicsAnalysisParams(
            channel="general",
            n_topics=20,
            days_back=7,
            output="topics.json",
            output_format=OutputFormat.JSON,
        )
        assert params.channel == "general"
        assert params.n_topics == 20
        assert params.days_back == 7
        assert params.output == "topics.json"
        assert params.output_format == OutputFormat.JSON

    def test_default_values(self):
        """Test default parameter values."""
        params = TopicsAnalysisParams()
        assert params.channel is None
        assert params.n_topics == 10
        assert params.days_back == 30
        assert params.output is None
        assert params.output_format == OutputFormat.JSON

    def test_days_back_validation(self):
        """Test days_back validation."""
        # Valid values
        params1 = TopicsAnalysisParams(days_back=1)
        assert params1.days_back == 1

        params2 = TopicsAnalysisParams(days_back=365)
        assert params2.days_back == 365

        # Invalid values
        with pytest.raises(ValidationError):
            TopicsAnalysisParams(days_back=0)

        with pytest.raises(ValidationError):
            TopicsAnalysisParams(days_back=366)


class TestTemporalAnalysisParams:
    """Test TemporalAnalysisParams model."""

    def test_valid_temporal_analysis_params(self):
        """Test creating valid TemporalAnalysisParams."""
        params = TemporalAnalysisParams(
            channel="dev",
            days_back=14,
            granularity="week",
            output="temporal.csv",
            output_format=OutputFormat.CSV,
        )
        assert params.channel == "dev"
        assert params.days_back == 14
        assert params.granularity == "week"
        assert params.output == "temporal.csv"
        assert params.output_format == OutputFormat.CSV

    def test_granularity_validation(self):
        """Test granularity validation."""
        # Valid values
        params1 = TemporalAnalysisParams(granularity="day")
        assert params1.granularity == "day"

        params2 = TemporalAnalysisParams(granularity="week")
        assert params2.granularity == "week"

        # Invalid values
        with pytest.raises(ValidationError):
            TemporalAnalysisParams(granularity="hour")


class TestListParams:
    """Test list parameter models."""

    def test_list_users_params(self):
        """Test ListUsersParams validation."""
        # Valid limits
        params1 = ListUsersParams(limit=1)
        assert params1.limit == 1

        params2 = ListUsersParams(limit=999)
        assert params2.limit == 999

        # Clamped limits
        params3 = ListUsersParams(limit=1000)
        assert params3.limit == 999  # Clamped to max

        params4 = ListUsersParams(limit=0)
        assert params4.limit == 50  # Clamped to default

    def test_list_channels_params(self):
        """Test ListChannelsParams validation."""
        # Valid limits
        params1 = ListChannelsParams(limit=1)
        assert params1.limit == 1

        params2 = ListChannelsParams(limit=50)
        assert params2.limit == 50

        # Clamped limits
        params3 = ListChannelsParams(limit=100)
        assert params3.limit == 50  # Clamped to max

        params4 = ListChannelsParams(limit=0)
        assert params4.limit == 25  # Clamped to default


class TestFactoryFunctions:
    """Test factory functions for creating parameter models."""

    def test_create_user_analysis_params(self):
        """Test creating UserAnalysisParams via direct construction."""
        params = UserAnalysisParams(user="alice", limit=5, output="test.json")
        assert params.user == "alice"
        assert params.limit == 5
        assert params.output == "test.json"
        assert params.output_format == OutputFormat.JSON

    def test_create_channel_analysis_params(self):
        """Test creating ChannelAnalysisParams via direct construction."""
        params = ChannelAnalysisParams(channel="general", limit=3)
        assert params.channel == "general"
        assert params.limit == 3

    def test_create_topics_analysis_params(self):
        """Test creating TopicsAnalysisParams via direct construction."""
        params = TopicsAnalysisParams(channel="dev", n_topics=15, days_back=7)
        assert params.channel == "dev"
        assert params.n_topics == 15
        assert params.days_back == 7


class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_user_analysis_params_serialization(self):
        """Test UserAnalysisParams JSON serialization."""
        params = UserAnalysisParams(
            user="alice", limit=5, output="output.json", output_format=OutputFormat.JSON
        )
        data = params.model_dump()

        expected = {
            "user": "alice",
            "limit": 5,
            "output": "output.json",
            "output_format": "json",
        }
        assert data == expected

        # Test round-trip
        recreated = UserAnalysisParams.model_validate(data)
        assert recreated.user == params.user
        assert recreated.limit == params.limit
        assert recreated.output == params.output
        assert recreated.output_format == params.output_format
