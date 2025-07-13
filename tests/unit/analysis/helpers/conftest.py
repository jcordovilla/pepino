"""Shared fixtures and test data for analysis tests."""

import asyncio
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

from pepino.config import Settings
from pepino.data.database.manager import DatabaseManager


def load_test_data():
    """Load test data from YAML file."""
    test_data_path = Path(__file__).parent / "test_data.yaml"
    with open(test_data_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def test_data():
    """Load test data once per test session."""
    return load_test_data()


@pytest.fixture
def text_processing_data(test_data):
    """Get text processing test data."""
    return test_data.get("text_processing", {})


@pytest.fixture
def statistics_data(test_data):
    """Get statistics test data."""
    return test_data.get("statistics", {})


@pytest.fixture
def time_series_data(test_data):
    """Get time series test data."""
    return test_data.get("time_series", {})


@pytest.fixture(scope="module")
def test_settings():
    """Create test settings with in-memory database"""
    return Settings(
        database_sqlite_path=":memory:",
        discord_bot_token=None,  # Not needed for tests
        sync_batch_size=10,
        analysis_max_results=100,
        embedding_batch_size=16,
        logging_level="WARNING",
    )


@pytest.fixture(scope="module")
async def test_db(test_settings):
    """Create in-memory SQLite DB and populate with mock data"""
    db_manager = DatabaseManager(test_settings.database_sqlite_path)
    await db_manager.initialize()

    # Create messages table
    await db_manager.pool.execute(
        """
        CREATE TABLE messages (
            id TEXT PRIMARY KEY,
            author_id TEXT,
            author_display_name TEXT,
            author_name TEXT,
            channel_id TEXT,
            channel_name TEXT,
            content TEXT,
            timestamp TEXT,
            is_bot INTEGER DEFAULT 0,
            has_reactions INTEGER DEFAULT 0,
            has_reference INTEGER DEFAULT 0,
            thread_id TEXT,
            thread_name TEXT,
            referenced_message_id TEXT
        )
    """
    )

    # Create message_embeddings table
    await db_manager.pool.execute(
        """
        CREATE TABLE message_embeddings (
            message_id TEXT PRIMARY KEY,
            embedding BLOB,
            created_at TEXT,
            FOREIGN KEY (message_id) REFERENCES messages (id)
        )
    """
    )

    # Insert mock messages
    messages = [
        (
            "1",
            "1",
            "Alice",
            "alice",
            "ch1",
            "general",
            "Hello world! AI and ML are cool.",
            "2024-06-01T12:00:00Z",
            0,
            1,
            0,
            None,
            None,
            None,
        ),
        (
            "2",
            "1",
            "Alice",
            "alice",
            "ch2",
            "random",
            "Machine learning is fun.",
            "2024-06-02T13:00:00Z",
            0,
            0,
            1,
            None,
            None,
            "1",
        ),
        (
            "3",
            "2",
            "Bob",
            "bob",
            "ch1",
            "general",
            "I love data science.",
            "2024-06-01T14:00:00Z",
            0,
            1,
            0,
            None,
            None,
            None,
        ),
        (
            "4",
            "2",
            "Bob",
            "bob",
            "ch1",
            "general",
            "Deep learning and AI.",
            "2024-06-03T15:00:00Z",
            0,
            0,
            1,
            None,
            None,
            "3",
        ),
        (
            "5",
            "3",
            "Carol",
            "carol",
            "ch2",
            "random",
            "Natural language processing is a part of AI.",
            "2024-06-04T16:00:00Z",
            0,
            1,
            0,
            None,
            None,
            None,
        ),
        (
            "6",
            "4",
            "David",
            "david",
            "ch1",
            "general",
            "Python programming is essential for data science.",
            "2024-06-05T10:00:00Z",
            0,
            0,
            1,
            None,
            None,
            "4",
        ),
        (
            "7",
            "4",
            "David",
            "david",
            "ch1",
            "general",
            "Statistics and probability are important.",
            "2024-06-06T11:00:00Z",
            0,
            1,
            0,
            None,
            None,
            None,
        ),
        (
            "8",
            "5",
            "Eve",
            "eve",
            "ch2",
            "random",
            "Neural networks and deep learning.",
            "2024-06-07T12:00:00Z",
            0,
            0,
            1,
            None,
            None,
            "5",
        ),
    ]

    await db_manager.pool.executemany(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        messages,
    )
    await db_manager.pool.commit()

    try:
        yield db_manager
    finally:
        # Ensure cleanup even if test fails
        if db_manager.pool is not None:
            await db_manager.close()


@pytest.fixture
def mock_embedding_service():
    """Mock embedding service for tests."""
    with patch(
        "pepino.analysis.embedding_analyzer.SentenceTransformer"
    ) as mock_transformer:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]] * 10)
        mock_transformer.return_value = mock_model

        yield mock_transformer


@pytest.fixture
def mock_nlp_service():
    """Mock NLP service for tests."""
    with patch("pepino.analysis.nlp_analyzer.spacy") as mock_spacy:
        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_doc.noun_chunks = [MagicMock(text="test phrase")]
        mock_doc.ents = [MagicMock(text="Test Entity", label_="PERSON")]
        mock_doc.__iter__ = lambda self: iter(
            [MagicMock(text="test", pos_="NOUN", is_alpha=True)]
        )
        mock_doc.sents = [MagicMock()]
        mock_nlp.return_value = mock_doc
        mock_spacy.load.return_value = mock_nlp

        yield mock_spacy


@pytest.fixture
def mock_matplotlib():
    """Mock matplotlib for plotting tests."""
    with patch("matplotlib.pyplot") as mock_plt:
        mock_figure = MagicMock()
        mock_axes = MagicMock()
        mock_figure.add_subplot.return_value = mock_axes
        mock_plt.figure.return_value = mock_figure
        mock_plt.subplots.return_value = (mock_figure, mock_axes)

        yield mock_plt


@pytest.fixture
def mock_seaborn():
    """Mock seaborn for plotting tests."""
    with patch("seaborn.heatmap") as mock_heatmap:
        mock_heatmap.return_value = MagicMock()
        yield mock_heatmap


@pytest.fixture
def sentiment_test_cases():
    """Test cases for sentiment analysis."""
    return [
        {
            "text": "I love this project! It's amazing.",
            "expected_sentiment": "positive",
            "expected_score": 0.8,
        },
        {
            "text": "This is terrible and I hate it.",
            "expected_sentiment": "negative",
            "expected_score": -0.7,
        },
        {
            "text": "The weather is okay today.",
            "expected_sentiment": "neutral",
            "expected_score": 0.1,
        },
    ]


@pytest.fixture
def embedding_test_cases():
    """Test cases for embedding generation."""
    return [
        {
            "text": "Hello world",
            "expected_dimensions": 384,
            "description": "simple text",
        },
        {
            "text": "This is a longer text with multiple sentences. It should generate embeddings.",
            "expected_dimensions": 384,
            "description": "longer text",
        },
        {"text": "", "expected_dimensions": 384, "description": "empty text"},
    ]


@pytest.fixture
def conversation_test_cases():
    """Test cases for conversation analysis."""
    return [
        {
            "messages": [
                {
                    "author_id": "user1",
                    "content": "Hello",
                    "timestamp": "2024-01-01T12:00:00Z",
                },
                {
                    "author_id": "user2",
                    "content": "Hi there!",
                    "timestamp": "2024-01-01T12:01:00Z",
                },
                {
                    "author_id": "user1",
                    "content": "How are you?",
                    "timestamp": "2024-01-01T12:02:00Z",
                },
            ],
            "expected_threads": 1,
            "expected_participants": 2,
            "description": "simple conversation",
        },
        {
            "messages": [
                {
                    "author_id": "user1",
                    "content": "Topic A",
                    "timestamp": "2024-01-01T12:00:00Z",
                },
                {
                    "author_id": "user2",
                    "content": "Topic B",
                    "timestamp": "2024-01-01T13:00:00Z",
                },
                {
                    "author_id": "user1",
                    "content": "Back to A",
                    "timestamp": "2024-01-01T14:00:00Z",
                },
            ],
            "expected_threads": 2,
            "expected_participants": 2,
            "description": "multiple topics",
        },
    ]


@pytest.fixture
def cli_test_cases():
    """Test cases for CLI commands."""
    return [
        {
            "command": ["analyze", "users"],
            "expected_output": "user analysis",
            "description": "user analysis command",
        },
        {
            "command": ["analyze", "channels"],
            "expected_output": "channel analysis",
            "description": "channel analysis command",
        },
        {
            "command": ["export", "messages"],
            "expected_output": "exporting messages",
            "description": "export command",
        },
    ]
