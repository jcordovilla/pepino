"""
Tests for Discord command parameter models.
"""

import pytest
from pydantic import ValidationError

from pepino.discord.commands.models import (
    ActivityTrendsParams,
    AnalysisType,
    ChannelAnalysisParams,
    CommandStatus,
    SyncAndAnalyzeParams,
    TopicsAnalysisParams,
    UserAnalysisParams,
)


class TestAnalysisType:
    """Test AnalysisType enum."""

    def test_analysis_type_values(self):
        """Test that all expected analysis types exist."""
        assert AnalysisType.USER == "user"
        assert AnalysisType.CHANNEL == "channel"
        assert AnalysisType.TOPICS == "topics"
        assert AnalysisType.TRENDS == "trends"
        assert AnalysisType.TOP_USERS == "top_users"


class TestChannelAnalysisParams:
    """Test ChannelAnalysisParams model."""

    def test_valid_channel_analysis_params(self):
        """Test creating valid ChannelAnalysisParams."""
        params = ChannelAnalysisParams(channel="general", include_chart=False)
        assert params.channel == "general"
        assert params.include_chart is False

    def test_default_include_chart(self):
        """Test default include_chart value."""
        params = ChannelAnalysisParams(channel="general")
        assert params.include_chart is True

    def test_channel_cleaning(self):
        """Test channel name cleaning."""
        # Remove # prefix
        params1 = ChannelAnalysisParams(channel="#general")
        assert params1.channel == "general"

        # Trim whitespace
        params2 = ChannelAnalysisParams(channel="  random  ")
        assert params2.channel == "random"

        # Both operations
        params3 = ChannelAnalysisParams(channel="  #dev  ")
        assert params3.channel == "dev"

    def test_invalid_channel_names(self):
        """Test validation of invalid channel names."""
        # Empty string
        with pytest.raises(
            ValidationError, match="String should have at least 1 character"
        ):
            ChannelAnalysisParams(channel="")

        # Just whitespace
        with pytest.raises(ValidationError, match="Channel name cannot be empty"):
            ChannelAnalysisParams(channel="   ")

        # Just # symbol
        with pytest.raises(ValidationError, match="Channel name cannot be just '#'"):
            ChannelAnalysisParams(channel="#")

        # Just # with whitespace
        with pytest.raises(ValidationError, match="Channel name cannot be just '#'"):
            ChannelAnalysisParams(channel="  #  ")


class TestUserAnalysisParams:
    """Test UserAnalysisParams model."""

    def test_valid_user_analysis_params(self):
        """Test creating valid UserAnalysisParams."""
        params = UserAnalysisParams(user="alice", include_chart=False)
        assert params.user == "alice"
        assert params.include_chart is False

    def test_user_cleaning(self):
        """Test username cleaning."""
        # Remove @ prefix
        params1 = UserAnalysisParams(user="@alice")
        assert params1.user == "alice"

        # Trim whitespace
        params2 = UserAnalysisParams(user="  bob  ")
        assert params2.user == "bob"

        # Both operations
        params3 = UserAnalysisParams(user="  @charlie  ")
        assert params3.user == "charlie"

    def test_invalid_usernames(self):
        """Test validation of invalid usernames."""
        # Empty string
        with pytest.raises(
            ValidationError, match="String should have at least 1 character"
        ):
            UserAnalysisParams(user="")

        # Just whitespace
        with pytest.raises(ValidationError, match="Username cannot be empty"):
            UserAnalysisParams(user="   ")

        # Just @ symbol
        with pytest.raises(ValidationError, match="Username cannot be just '@'"):
            UserAnalysisParams(user="@")

        # Just @ with whitespace
        with pytest.raises(ValidationError, match="Username cannot be just '@'"):
            UserAnalysisParams(user="  @  ")


class TestTopicsAnalysisParams:
    """Test TopicsAnalysisParams model."""

    def test_valid_topics_analysis_params(self):
        """Test creating valid TopicsAnalysisParams."""
        params = TopicsAnalysisParams(channel="general", days_back=7, min_word_length=5)
        assert params.channel == "general"
        assert params.days_back == 7
        assert params.min_word_length == 5

    def test_default_values(self):
        """Test default parameter values."""
        params = TopicsAnalysisParams()
        assert params.channel is None
        assert params.days_back == 30
        assert params.min_word_length == 4

    def test_optional_channel_cleaning(self):
        """Test optional channel name cleaning."""
        # None remains None
        params1 = TopicsAnalysisParams(channel=None)
        assert params1.channel is None

        # Empty string becomes None
        params2 = TopicsAnalysisParams(channel="")
        assert params2.channel is None

        # Whitespace becomes None
        params3 = TopicsAnalysisParams(channel="   ")
        assert params3.channel is None

        # Valid channel gets cleaned
        params4 = TopicsAnalysisParams(channel="#general")
        assert params4.channel == "general"

    def test_days_back_validation(self):
        """Test days_back validation."""
        # Valid range
        params1 = TopicsAnalysisParams(days_back=1)
        assert params1.days_back == 1

        params2 = TopicsAnalysisParams(days_back=365)
        assert params2.days_back == 365

        # Invalid values
        with pytest.raises(ValidationError):
            TopicsAnalysisParams(days_back=0)

        with pytest.raises(ValidationError):
            TopicsAnalysisParams(days_back=366)

    def test_min_word_length_validation(self):
        """Test min_word_length validation."""
        # Valid range
        params1 = TopicsAnalysisParams(min_word_length=3)
        assert params1.min_word_length == 3

        params2 = TopicsAnalysisParams(min_word_length=20)
        assert params2.min_word_length == 20

        # Invalid values
        with pytest.raises(ValidationError):
            TopicsAnalysisParams(min_word_length=2)

        with pytest.raises(ValidationError):
            TopicsAnalysisParams(min_word_length=21)


class TestActivityTrendsParams:
    """Test ActivityTrendsParams model."""

    def test_valid_activity_trends_params(self):
        """Test creating valid ActivityTrendsParams."""
        params = ActivityTrendsParams(
            include_chart=False, days_back=14, granularity="week"
        )
        assert params.include_chart is False
        assert params.days_back == 14
        assert params.granularity == "week"

    def test_default_values(self):
        """Test default parameter values."""
        params = ActivityTrendsParams()
        assert params.include_chart is True
        assert params.days_back == 30
        assert params.granularity == "day"

    def test_granularity_validation(self):
        """Test granularity validation."""
        # Valid values
        params1 = ActivityTrendsParams(granularity="day")
        assert params1.granularity == "day"

        params2 = ActivityTrendsParams(granularity="week")
        assert params2.granularity == "week"

        # Invalid values
        with pytest.raises(ValidationError):
            ActivityTrendsParams(granularity="hour")

    def test_days_back_validation(self):
        """Test days_back validation."""
        # Valid range
        params1 = ActivityTrendsParams(days_back=7)
        assert params1.days_back == 7

        params2 = ActivityTrendsParams(days_back=365)
        assert params2.days_back == 365

        # Invalid values
        with pytest.raises(ValidationError):
            ActivityTrendsParams(days_back=6)

        with pytest.raises(ValidationError):
            ActivityTrendsParams(days_back=366)


class TestSyncAndAnalyzeParams:
    """Test SyncAndAnalyzeParams model."""

    def test_valid_sync_and_analyze_params(self):
        """Test creating valid SyncAndAnalyzeParams."""
        params = SyncAndAnalyzeParams(
            analysis_type=AnalysisType.USER, target="alice", force_sync=True
        )
        assert params.analysis_type == AnalysisType.USER
        assert params.target == "alice"
        assert params.force_sync is True

    def test_default_values(self):
        """Test default parameter values."""
        params = SyncAndAnalyzeParams(analysis_type=AnalysisType.TOPICS)
        assert params.analysis_type == AnalysisType.TOPICS
        assert params.target is None
        assert params.force_sync is False

    def test_target_cleaning(self):
        """Test target cleaning."""
        # Remove # prefix for channels
        params1 = SyncAndAnalyzeParams(
            analysis_type=AnalysisType.CHANNEL, target="#general"
        )
        assert params1.target == "general"

        # Remove @ prefix for users
        params2 = SyncAndAnalyzeParams(analysis_type=AnalysisType.USER, target="@alice")
        assert params2.target == "alice"

        # Empty string becomes None
        params3 = SyncAndAnalyzeParams(analysis_type=AnalysisType.TOPICS, target="   ")
        assert params3.target is None

    def test_analysis_requirements_validation(self):
        """Test validation of analysis-specific requirements."""
        # User analysis requires target
        params = SyncAndAnalyzeParams(analysis_type=AnalysisType.USER, target=None)
        with pytest.raises(
            ValueError, match="Target user name is required for user analysis"
        ):
            params.validate_analysis_requirements()

        # Channel analysis requires target
        params = SyncAndAnalyzeParams(analysis_type=AnalysisType.CHANNEL, target=None)
        with pytest.raises(
            ValueError, match="Target channel name is required for channel analysis"
        ):
            params.validate_analysis_requirements()

        # Topics analysis doesn't require target
        params = SyncAndAnalyzeParams(analysis_type=AnalysisType.TOPICS, target=None)
        params.validate_analysis_requirements()  # Should not raise

        # Trends analysis doesn't require target
        params = SyncAndAnalyzeParams(analysis_type=AnalysisType.TRENDS, target=None)
        params.validate_analysis_requirements()  # Should not raise


class TestCommandStatus:
    """Test CommandStatus model."""

    def test_valid_command_status(self):
        """Test creating valid CommandStatus."""
        status = CommandStatus(
            command="sync_and_analyze",
            user_id="123456789",
            guild_id="987654321",
            channel_id="555444333",
            started_at="2023-12-01T12:00:00Z",
            status="running",
            progress="Syncing messages...",
            error=None,
        )
        assert status.command == "sync_and_analyze"
        assert status.user_id == "123456789"
        assert status.guild_id == "987654321"
        assert status.channel_id == "555444333"
        assert status.started_at == "2023-12-01T12:00:00Z"
        assert status.status == "running"
        assert status.progress == "Syncing messages..."
        assert status.error is None

    def test_status_validation(self):
        """Test status field validation."""
        # Valid statuses
        for status_val in ["running", "completed", "failed"]:
            status = CommandStatus(
                command="test",
                user_id="123",
                started_at="2023-12-01T12:00:00Z",
                status=status_val,
            )
            assert status.status == status_val

        # Invalid status
        with pytest.raises(ValidationError):
            CommandStatus(
                command="test",
                user_id="123",
                started_at="2023-12-01T12:00:00Z",
                status="invalid",
            )


class TestFactoryFunctions:
    """Test direct model construction (replacing removed factory functions)."""

    def test_create_channel_analysis_params(self):
        """Test creating ChannelAnalysisParams via direct construction."""
        params = ChannelAnalysisParams(channel="general", include_chart=False)
        assert params.channel == "general"
        assert params.include_chart is False

    def test_create_user_analysis_params(self):
        """Test creating UserAnalysisParams via direct construction."""
        params = UserAnalysisParams(user="alice", include_chart=True)
        assert params.user == "alice"
        assert params.include_chart is True

    def test_create_topics_analysis_params(self):
        """Test creating TopicsAnalysisParams via direct construction."""
        params = TopicsAnalysisParams(channel="dev", days_back=14, min_word_length=5)
        assert params.channel == "dev"
        assert params.days_back == 14
        assert params.min_word_length == 5

    def test_create_activity_trends_params(self):
        """Test creating ActivityTrendsParams via direct construction."""
        params = ActivityTrendsParams(granularity="week", days_back=14)
        assert params.granularity == "week"
        assert params.days_back == 14

    def test_create_sync_and_analyze_params(self):
        """Test creating SyncAndAnalyzeParams via direct construction."""
        params = SyncAndAnalyzeParams(
            analysis_type=AnalysisType.CHANNEL, target="general", force_sync=False
        )
        assert params.target == "general"
        assert params.analysis_type == AnalysisType.CHANNEL
        assert params.force_sync is False


class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_channel_analysis_params_serialization(self):
        """Test ChannelAnalysisParams JSON serialization."""
        params = ChannelAnalysisParams(channel="general", include_chart=False)
        data = params.model_dump()

        expected = {"channel": "general", "include_chart": False}
        assert data == expected

        # Test round-trip
        recreated = ChannelAnalysisParams.model_validate(data)
        assert recreated.channel == params.channel
        assert recreated.include_chart == params.include_chart

    def test_sync_and_analyze_params_serialization(self):
        """Test SyncAndAnalyzeParams JSON serialization."""
        params = SyncAndAnalyzeParams(
            analysis_type=AnalysisType.USER, target="alice", force_sync=True
        )
        data = params.model_dump()

        expected = {"analysis_type": "user", "target": "alice", "force_sync": True}
        assert data == expected

        # Test round-trip
        recreated = SyncAndAnalyzeParams.model_validate(data)
        assert recreated.analysis_type == params.analysis_type
        assert recreated.target == params.target
        assert recreated.force_sync == params.force_sync
