"""Unit tests for temporal analysis."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pepino.analysis.temporal_analyzer import TemporalAnalyzer


@pytest.mark.asyncio
async def test_temporal_analyzer():
    """Test TemporalAnalyzer with mocked message repository"""
    mock_db_manager = MagicMock()
    analyzer = TemporalAnalyzer(mock_db_manager)

    # Mock temporal data (should match the structure expected by the analyzer)
    mock_temporal_data = [
        {"period": "2024-01-01", "message_count": 50, "unique_users": 15},
        {"period": "2024-01-02", "message_count": 75, "unique_users": 20},
        {"period": "2024-01-03", "message_count": 90, "unique_users": 25},
        {"period": "2024-01-04", "message_count": 65, "unique_users": 18},
        {"period": "2024-01-05", "message_count": 80, "unique_users": 22},
    ]

    with patch.object(
        analyzer.message_repo, "get_temporal_analysis_data", new_callable=AsyncMock
    ) as mock_get_data:
        mock_get_data.return_value = mock_temporal_data

        result = await analyzer.analyze()

        # Test basic structure
        assert result.success is True
        assert result.plugin == "TemporalAnalyzer"
        assert len(result.temporal_data) == 5

        # Test temporal data
        assert result.temporal_data[0].period == "2024-01-01"
        assert result.temporal_data[0].message_count == 50
        assert result.temporal_data[0].unique_users == 15

        # Test patterns
        assert result.patterns.total_messages == 360  # Sum of all message counts
        assert result.patterns.avg_messages_per_period == 72.0
        assert result.patterns.max_messages_in_period == 90
        assert result.patterns.min_messages_in_period == 50
        assert result.patterns.most_active_period == "2024-01-03"
        assert result.patterns.peak_user_count == 25
        assert result.patterns.total_periods == 5

        # Verify repository was called correctly
        mock_get_data.assert_called_once_with(
            channel_name=None, user_name=None, days_back=30, granularity="day"
        )


@pytest.mark.asyncio
async def test_temporal_analyzer_with_granularity():
    """Test TemporalAnalyzer with week granularity"""
    mock_db_manager = MagicMock()
    analyzer = TemporalAnalyzer(mock_db_manager)

    # Mock weekly data
    mock_temporal_data = [
        {"period": "2024-W01", "message_count": 300, "unique_users": 45},
        {"period": "2024-W02", "message_count": 450, "unique_users": 60},
        {"period": "2024-W03", "message_count": 380, "unique_users": 52},
        {"period": "2024-W04", "message_count": 420, "unique_users": 58},
    ]

    with patch.object(
        analyzer.message_repo, "get_temporal_analysis_data", new_callable=AsyncMock
    ) as mock_get_data:
        mock_get_data.return_value = mock_temporal_data

        result = await analyzer.analyze(granularity="week", days_back=28)

        assert result.success is True
        assert len(result.temporal_data) == 4
        assert result.temporal_data[0].period == "2024-W01"
        assert result.temporal_data[0].message_count == 300
        assert result.patterns.total_messages == 1550
        assert result.patterns.avg_messages_per_period == 387.5

        # Verify repository was called with correct parameters
        mock_get_data.assert_called_once_with(
            channel_name=None, user_name=None, days_back=28, granularity="week"
        )


@pytest.mark.asyncio
async def test_temporal_analyzer_with_channel():
    """Test TemporalAnalyzer with specific channel filter"""
    mock_db_manager = MagicMock()
    analyzer = TemporalAnalyzer(mock_db_manager)

    mock_temporal_data = [
        {"period": "2024-01-01", "message_count": 25, "unique_users": 8},
        {"period": "2024-01-02", "message_count": 35, "unique_users": 12},
        {"period": "2024-01-03", "message_count": 42, "unique_users": 15},
    ]

    with patch.object(
        analyzer.message_repo, "get_temporal_analysis_data", new_callable=AsyncMock
    ) as mock_get_data:
        mock_get_data.return_value = mock_temporal_data

        result = await analyzer.analyze(channel_name="general")

        assert result.success is True
        assert len(result.temporal_data) == 3
        assert result.patterns.total_messages == 102

        # Verify repository was called with channel filter
        mock_get_data.assert_called_once_with(
            channel_name="general", user_name=None, days_back=30, granularity="day"
        )


@pytest.mark.asyncio
async def test_temporal_analyzer_with_user():
    """Test TemporalAnalyzer with specific user filter"""
    mock_db_manager = MagicMock()
    analyzer = TemporalAnalyzer(mock_db_manager)

    mock_temporal_data = [
        {"period": "2024-01-01", "message_count": 5, "unique_users": 1},
        {"period": "2024-01-02", "message_count": 8, "unique_users": 1},
        {"period": "2024-01-03", "message_count": 12, "unique_users": 1},
    ]

    with patch.object(
        analyzer.message_repo, "get_temporal_analysis_data", new_callable=AsyncMock
    ) as mock_get_data:
        mock_get_data.return_value = mock_temporal_data

        result = await analyzer.analyze(user_name="alice")

        assert result.success is True
        assert len(result.temporal_data) == 3
        assert result.patterns.total_messages == 25
        assert result.patterns.peak_user_count == 1  # Single user

        # Verify repository was called with user filter
        mock_get_data.assert_called_once_with(
            channel_name=None, user_name="alice", days_back=30, granularity="day"
        )


@pytest.mark.asyncio
async def test_temporal_analyzer_no_data():
    """Test TemporalAnalyzer when no data is available"""
    mock_db_manager = MagicMock()
    analyzer = TemporalAnalyzer(mock_db_manager)

    with patch.object(
        analyzer.message_repo, "get_temporal_analysis_data", new_callable=AsyncMock
    ) as mock_get_data:
        mock_get_data.return_value = []  # Empty data

        result = await analyzer.analyze()

        assert result.success is False
        assert "No temporal data found" in result.error

        mock_get_data.assert_called_once()


@pytest.mark.asyncio
async def test_temporal_analyzer_trend_calculation():
    """Test TemporalAnalyzer trend calculation"""
    mock_db_manager = MagicMock()
    analyzer = TemporalAnalyzer(mock_db_manager)

    # Mock data showing increasing trend
    mock_temporal_data = [
        {"period": "2024-01-01", "message_count": 10, "unique_users": 5},
        {"period": "2024-01-02", "message_count": 15, "unique_users": 8},
        {"period": "2024-01-03", "message_count": 20, "unique_users": 10},
        {"period": "2024-01-04", "message_count": 25, "unique_users": 12},
    ]

    with patch.object(
        analyzer.message_repo, "get_temporal_analysis_data", new_callable=AsyncMock
    ) as mock_get_data:
        mock_get_data.return_value = mock_temporal_data

        result = await analyzer.analyze()

        assert result.success is True
        assert result.patterns.message_trend == "increasing"
        assert result.patterns.trend_percentage > 0  # Should show positive trend

        mock_get_data.assert_called_once()


@pytest.mark.asyncio
async def test_temporal_analyzer_repository_error():
    """Test TemporalAnalyzer when repository raises an exception"""
    mock_db_manager = MagicMock()
    analyzer = TemporalAnalyzer(mock_db_manager)

    with patch.object(
        analyzer.message_repo, "get_temporal_analysis_data", new_callable=AsyncMock
    ) as mock_get_data:
        mock_get_data.side_effect = Exception("Database connection error")

        result = await analyzer.analyze()

        assert result.success is False
        assert "Analysis failed: Database connection error" in result.error
