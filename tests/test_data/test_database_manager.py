"""Unit tests for DatabaseManager."""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pepino.data.database.manager import DatabaseManager


class TestDatabaseManager:
    """Test cases for DatabaseManager."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)

    @pytest.fixture
    def db_manager(self, temp_db_path):
        """Create DatabaseManager instance."""
        return DatabaseManager(temp_db_path)

    @pytest.mark.asyncio
    async def test_initialize_success(self, db_manager):
        """Test successful database initialization."""
        await db_manager.initialize()
        assert db_manager.pool is not None

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, db_manager):
        """Test initialization when already initialized."""
        await db_manager.initialize()
        await db_manager.initialize()  # Should not raise error
        assert db_manager.pool is not None

    @pytest.mark.asyncio
    async def test_execute_success(self, db_manager):
        """Test successful execute operation."""
        await db_manager.initialize()
        await db_manager.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")

        # Verify table was created using execute_query
        result = await db_manager.execute_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='test'"
        )
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_execute_with_params(self, db_manager):
        """Test execute with parameters."""
        await db_manager.initialize()
        await db_manager.execute(
            "CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)"
        )
        await db_manager.execute(
            "INSERT INTO test (id, name) VALUES (?, ?)", (1, "test")
        )

        result = await db_manager.execute_query("SELECT * FROM test WHERE id = ?", (1,))
        assert len(result) == 1
        assert result[0]["name"] == "test"

    @pytest.mark.asyncio
    async def test_execute_many(self, db_manager):
        """Test execute_many operation."""
        await db_manager.initialize()
        await db_manager.execute(
            "CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)"
        )

        data = [(1, "test1"), (2, "test2"), (3, "test3")]
        await db_manager.execute_many("INSERT INTO test (id, name) VALUES (?, ?)", data)

        result = await db_manager.execute_query("SELECT * FROM test ORDER BY id")
        assert len(result) == 3
        assert result[0]["name"] == "test1"
        assert result[1]["name"] == "test2"
        assert result[2]["name"] == "test3"

    @pytest.mark.asyncio
    async def test_execute_script(self, db_manager):
        """Test execute_script operation."""
        await db_manager.initialize()
        script = """
        CREATE TABLE test1 (id INTEGER PRIMARY KEY);
        CREATE TABLE test2 (id INTEGER PRIMARY KEY);
        INSERT INTO test1 (id) VALUES (1);
        INSERT INTO test2 (id) VALUES (2);
        """
        await db_manager.execute_script(script)

        result1 = await db_manager.execute_query("SELECT * FROM test1")
        result2 = await db_manager.execute_query("SELECT * FROM test2")
        assert len(result1) == 1
        assert len(result2) == 1

    @pytest.mark.asyncio
    async def test_execute_not_initialized(self, db_manager):
        """Test execute when not initialized - should auto-initialize."""
        # Should not raise error, should auto-initialize
        await db_manager.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
        assert db_manager.pool is not None

    @pytest.mark.asyncio
    async def test_execute_many_not_initialized(self, db_manager):
        """Test execute_many when not initialized - should auto-initialize."""
        # Should not raise error, should auto-initialize
        await db_manager.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
        await db_manager.execute_many("INSERT INTO test (id) VALUES (?)", [(1,)])
        assert db_manager.pool is not None

    @pytest.mark.asyncio
    async def test_execute_script_not_initialized(self, db_manager):
        """Test execute_script when not initialized - should auto-initialize."""
        # Should not raise error, should auto-initialize
        await db_manager.execute_script("CREATE TABLE test (id INTEGER PRIMARY KEY)")
        assert db_manager.pool is not None

    @pytest.mark.asyncio
    async def test_close_success(self, db_manager):
        """Test successful close operation."""
        await db_manager.initialize()
        await db_manager.close()
        assert db_manager.pool is None

    @pytest.mark.asyncio
    async def test_close_not_initialized(self, db_manager):
        """Test close when not initialized."""
        await db_manager.close()  # Should not raise error
        assert db_manager.pool is None

    @pytest.mark.asyncio
    async def test_context_manager(self, temp_db_path):
        """Test using DatabaseManager as context manager."""
        async with DatabaseManager(temp_db_path) as db_manager:
            await db_manager.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
            result = await db_manager.execute_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='test'"
            )
            assert len(result) == 1

    @pytest.mark.asyncio
    async def test_database_path_property(self, db_manager):
        """Test database_path property."""
        assert db_manager.db_path is not None
        assert db_manager.db_path.endswith(".db")

    @pytest.mark.asyncio
    async def test_pool_property_before_init(self, db_manager):
        """Test pool property before initialization."""
        assert db_manager.pool is None

    @pytest.mark.asyncio
    async def test_pool_property_after_init(self, db_manager):
        """Test pool property after initialization."""
        await db_manager.initialize()
        assert db_manager.pool is not None

    @pytest.mark.asyncio
    async def test_pool_property_after_close(self, db_manager):
        """Test pool property after close."""
        await db_manager.initialize()
        await db_manager.close()
        assert db_manager.pool is None

    @pytest.mark.asyncio
    async def test_get_table_names(self, db_manager):
        """Test getting table names."""
        await db_manager.initialize()

        table_names = await db_manager.get_table_names()

        # Should have at least the messages table
        assert "messages" in table_names
        assert isinstance(table_names, list)

    @pytest.mark.asyncio
    async def test_get_table_info(self, db_manager):
        """Test getting table information."""
        await db_manager.initialize()

        table_info = await db_manager.get_table_info("messages")

        assert isinstance(table_info, list)
        assert len(table_info) > 0
        assert "name" in table_info[0]

    @pytest.mark.asyncio
    async def test_get_row_count(self, db_manager):
        """Test getting row count."""
        await db_manager.initialize()

        count = await db_manager.get_row_count("messages")

        assert isinstance(count, int)
        assert count >= 0

    @pytest.mark.asyncio
    async def test_execute_query_success(self, db_manager):
        """Test successful query execution."""
        await db_manager.initialize()

        result = await db_manager.execute_query(
            "SELECT COUNT(*) as count FROM messages"
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert "count" in result[0]

    @pytest.mark.asyncio
    async def test_execute_single_success(self, db_manager):
        """Test successful single query execution."""
        await db_manager.initialize()

        result = await db_manager.execute_single("SELECT COUNT(*) FROM messages")

        # execute_single returns sqlite3.Row objects, not tuples
        assert result is not None
        assert hasattr(result, "__getitem__")  # Should be a row-like object
        assert result[0] >= 0  # Count should be non-negative

    @pytest.mark.asyncio
    async def test_execute_script_success(self, db_manager):
        """Test successful script execution."""
        await db_manager.initialize()

        script = """
            CREATE TABLE IF NOT EXISTS test_script_table (
                id TEXT PRIMARY KEY,
                name TEXT
            );
            INSERT INTO test_script_table (id, name) VALUES ('test1', 'Test Name');
        """

        await db_manager.execute_script(script)

        # Verify script execution
        result = await db_manager.execute_single(
            "SELECT name FROM test_script_table WHERE id = ?", ("test1",)
        )
        assert result[0] == "Test Name"

    @pytest.mark.asyncio
    async def test_backup_database(self, db_manager, temp_db_path):
        """Test database backup functionality."""
        await db_manager.initialize()

        # Create backup
        backup_path = temp_db_path + ".backup"
        await db_manager.backup_database(backup_path)

        # Verify backup exists and has data
        assert os.path.exists(backup_path)
        assert os.path.getsize(backup_path) > 0

    @pytest.mark.asyncio
    async def test_vacuum_database(self, db_manager):
        """Test database vacuum operation."""
        await db_manager.initialize()

        # Vacuum should complete without error
        await db_manager.vacuum()

        # Database should still be accessible
        result = await db_manager.execute_single("SELECT COUNT(*) FROM messages")
        assert result is not None
        assert hasattr(result, "__getitem__")  # Should be a row-like object

    @pytest.mark.asyncio
    async def test_analyze_tables(self, db_manager):
        """Test database analyze operation."""
        await db_manager.initialize()

        # Analyze should complete without error
        await db_manager.analyze_tables()

        # Database should still be accessible
        result = await db_manager.execute_single("SELECT COUNT(*) FROM messages")
        assert result is not None
        assert hasattr(result, "__getitem__")  # Should be a row-like object

    @pytest.mark.parametrize(
        "test_case",
        [
            {
                "error_type": Exception,
                "error_message": "Database connection failed",
                "description": "connection error",
            },
            {
                "error_type": ValueError,
                "error_message": "Invalid database path",
                "description": "invalid path error",
            },
        ],
    )
    @pytest.mark.asyncio
    async def test_initialization_failure_variations(self, test_case):
        """Test initialization failure handling with various error types."""
        error_type = test_case["error_type"]
        error_message = test_case["error_message"]

        with patch("aiosqlite.connect") as mock_connect:
            mock_connect.side_effect = error_type(error_message)

            db_manager = DatabaseManager("invalid.db")

            with pytest.raises(error_type, match=error_message):
                await db_manager.initialize()
