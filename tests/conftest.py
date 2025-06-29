"""Shared test configuration and fixtures."""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pepino.config import Settings
from pepino.data.database.manager import DatabaseManager


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    os.unlink(db_path)


@pytest.fixture
def test_settings():
    """Create test settings."""
    return Settings(
        db_path=":memory:",
        discord_token=None,  # Not needed for tests
        sync_batch_size=10,
        analysis_max_results=100,
        embedding_batch_size=16,
        log_level="WARNING",
        debug=True,
    )


@pytest.fixture
async def test_db_manager(temp_db_path):
    """Create test database manager with temporary database."""
    db_manager = DatabaseManager(temp_db_path)
    await db_manager.initialize()

    # Create basic test tables
    await db_manager.execute_script(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            channel_id TEXT,
            channel_name TEXT,
            author_id TEXT,
            author_name TEXT,
            author_display_name TEXT,
            content TEXT,
            timestamp TEXT,
            is_bot INTEGER DEFAULT 0,
            has_reactions INTEGER DEFAULT 0,
            has_reference INTEGER DEFAULT 0
        );
        
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT,
            display_name TEXT,
            is_bot INTEGER DEFAULT 0,
            created_at TEXT,
            last_seen TEXT
        );
        
        CREATE TABLE IF NOT EXISTS channels (
            id TEXT PRIMARY KEY,
            name TEXT,
            type TEXT,
            created_at TEXT,
            last_message_at TEXT
        );
    """
    )

    try:
        yield db_manager
    finally:
        # Ensure cleanup even if test fails
        if db_manager.pool is not None:
            await db_manager.close()


@pytest.fixture
def mock_aiosqlite():
    """Mock aiosqlite for database operations."""
    with patch("aiosqlite.connect") as mock_connect:
        mock_connection = AsyncMock()
        mock_cursor = AsyncMock()

        # Set up mock cursor methods
        mock_cursor.fetchall.return_value = []
        mock_cursor.fetchone.return_value = None
        mock_cursor.execute = AsyncMock()
        mock_cursor.executemany = AsyncMock()

        # Set up mock connection methods
        mock_connection.cursor.return_value = mock_cursor
        mock_connection.execute = AsyncMock()
        mock_connection.executemany = AsyncMock()
        mock_connection.commit = AsyncMock()
        mock_connection.close = AsyncMock()

        mock_connect.return_value.__aenter__.return_value = mock_connection

        yield mock_connect
