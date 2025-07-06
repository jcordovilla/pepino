"""Unit tests for topic analysis."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pepino.analysis.helpers.topic_analyzer import TopicAnalyzer
from pepino.analysis.helpers.data_facade import AnalysisDataFacade


@pytest.mark.asyncio
async def test_topic_analyzer():
    """Test TopicAnalyzer with mocked message repository"""
    mock_db_manager = MagicMock()
    facade = AnalysisDataFacade(mock_db_manager)
    analyzer = TopicAnalyzer(facade)

    # Mock messages with realistic content
    mock_messages = [
        "I love working with Python programming and data science frameworks like pandas",
        "Machine learning algorithms are fascinating to study and implement in real projects",
        "JavaScript and React development for web applications is really exciting",
        "Python is great for data analysis and machine learning tasks",
        "Web development with JavaScript frameworks like React and Vue is fun",
        "Data science projects often require pandas, numpy, and scikit-learn libraries",
    ]

    with patch.object(
        analyzer.message_repo, "get_messages_for_topic_analysis", new_callable=AsyncMock
    ) as mock_get_messages:
        mock_get_messages.return_value = mock_messages

        result = await analyzer.analyze()

        assert result.success is True
        assert result.plugin == "TopicAnalyzer"
        assert isinstance(result.topics, list)
        assert result.message_count == len(mock_messages)
        assert len(result.topics) > 0

        # Check that common words appear in topics
        topic_words = [topic.topic for topic in result.topics]
        topic_text = " ".join(topic_words).lower()
        assert any(
            word in topic_text
            for word in ["python", "data", "javascript", "development"]
        )

        # Verify repository was called correctly
        mock_get_messages.assert_called_once_with(
            channel_name=None,
            days_back=30,
            min_word_length=4,
            base_filter=analyzer.data_facade.base_filter,
        )


@pytest.mark.asyncio
async def test_topic_analyzer_with_channel():
    """Test TopicAnalyzer with specific channel"""
    mock_db_manager = MagicMock()
    facade = AnalysisDataFacade(mock_db_manager)
    analyzer = TopicAnalyzer(facade)

    mock_messages = [
        "Django web framework is excellent for Python development",
        "Flask microframework is great for small Python applications",
        "FastAPI is becoming popular for Python API development",
    ]

    with patch.object(
        analyzer.message_repo, "get_messages_for_topic_analysis", new_callable=AsyncMock
    ) as mock_get_messages:
        mock_get_messages.return_value = mock_messages

        result = await analyzer.analyze(channel_name="python-dev")

        assert result.success is True
        assert result.message_count == 3
        assert len(result.topics) > 0

        # Should contain Python-related topics
        topic_words = [topic.topic for topic in result.topics]
        topic_text = " ".join(topic_words).lower()
        assert "python" in topic_text or "django" in topic_text or "flask" in topic_text

        # Verify repository was called with channel name
        mock_get_messages.assert_called_once_with(
            channel_name="python-dev",
            days_back=30,
            min_word_length=4,
            base_filter=analyzer.data_facade.base_filter,
        )


@pytest.mark.asyncio
async def test_topic_analyzer_with_custom_parameters():
    """Test TopicAnalyzer with custom parameters"""
    mock_db_manager = MagicMock()
    facade = AnalysisDataFacade(mock_db_manager)
    analyzer = TopicAnalyzer(facade)

    mock_messages = [
        "Machine learning models require careful hyperparameter tuning",
        "Deep learning neural networks can solve complex classification problems",
        "Natural language processing tasks benefit from transformer architectures",
    ]

    with patch.object(
        analyzer.message_repo, "get_messages_for_topic_analysis", new_callable=AsyncMock
    ) as mock_get_messages:
        mock_get_messages.return_value = mock_messages

        result = await analyzer.analyze(
            channel_name="ai-research", days_back=7, min_word_length=5
        )

        assert result.success is True
        assert result.message_count == 3

        # With min_word_length=5, shorter words should be filtered out
        topic_words = [topic.topic for topic in result.topics]
        for word in topic_words:
            assert len(word) >= 5

        # Verify repository was called with custom parameters
        mock_get_messages.assert_called_once_with(
            channel_name="ai-research",
            days_back=7,
            min_word_length=5,
            base_filter=analyzer.data_facade.base_filter,
        )


@pytest.mark.asyncio
async def test_topic_analyzer_no_messages():
    """Test TopicAnalyzer when no messages are found"""
    mock_db_manager = MagicMock()
    facade = AnalysisDataFacade(mock_db_manager)
    analyzer = TopicAnalyzer(facade)

    with patch.object(
        analyzer.message_repo, "get_messages_for_topic_analysis", new_callable=AsyncMock
    ) as mock_get_messages:
        mock_get_messages.return_value = []  # No messages

        result = await analyzer.analyze()

        assert result.success is False
        assert "No messages found for topic analysis" in result.error

        mock_get_messages.assert_called_once()


@pytest.mark.asyncio
async def test_topic_analyzer_topic_structure():
    """Test that TopicAnalyzer returns properly structured topic data"""
    mock_db_manager = MagicMock()
    facade = AnalysisDataFacade(mock_db_manager)
    analyzer = TopicAnalyzer(facade)

    mock_messages = [
        "Python programming language is versatile and powerful",
        "Python development tools make coding efficient and enjoyable",
        "Programming with Python offers great flexibility",
    ]

    with patch.object(
        analyzer.message_repo, "get_messages_for_topic_analysis", new_callable=AsyncMock
    ) as mock_get_messages:
        mock_get_messages.return_value = mock_messages

        result = await analyzer.analyze()

        assert result.success is True
        assert len(result.topics) > 0

        # Check that each topic has the required structure
        for topic in result.topics:
            assert hasattr(topic, "topic")
            assert hasattr(topic, "frequency")
            assert hasattr(topic, "relevance_score")
            assert isinstance(topic.topic, str)
            assert isinstance(topic.frequency, int)
            assert isinstance(topic.relevance_score, float)
            assert topic.frequency > 0
            assert 0.0 <= topic.relevance_score <= 1.0

        # Topics should be sorted by relevance score (descending)
        scores = [topic.relevance_score for topic in result.topics]
        assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_topic_analyzer_repository_error():
    """Test TopicAnalyzer when repository raises an exception"""
    mock_db_manager = MagicMock()
    facade = AnalysisDataFacade(mock_db_manager)
    analyzer = TopicAnalyzer(facade)

    with patch.object(
        analyzer.message_repo, "get_messages_for_topic_analysis", new_callable=AsyncMock
    ) as mock_get_messages:
        mock_get_messages.side_effect = Exception("Database connection error")

        result = await analyzer.analyze()

        assert result.success is False
        assert "Analysis failed: Database connection error" in result.error
