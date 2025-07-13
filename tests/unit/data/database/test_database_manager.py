"""Unit tests for DatabaseManager."""

import os
import tempfile
from unittest.mock import MagicMock, patch

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

    def test_initialize_success(self, db_manager):
        """Test successful database initialization."""
        db_manager.initialize()
        assert hasattr(db_manager._local, 'connection')
        assert db_manager._local.connection is not None

    def test_initialize_already_initialized(self, db_manager):
        """Test initialization when already initialized."""
        db_manager.initialize()
        db_manager.initialize()  # Should not raise error
        assert hasattr(db_manager._local, 'connection')
        assert db_manager._local.connection is not None

    def test_execute_query_success(self, db_manager):
        """Test successful query execution."""
        db_manager.initialize()

        result = db_manager.execute_query(
            "SELECT COUNT(*) as count FROM messages"
        )

        assert isinstance(result, list)
        assert len(result) == 1
        assert "count" in result[0]

    def test_execute_query_with_params(self, db_manager):
        """Test execute_query with parameters."""
        db_manager.initialize()
        db_manager.execute_script("""
            CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT);
            INSERT INTO test (id, name) VALUES (1, 'test');
        """)

        result = db_manager.execute_query("SELECT * FROM test WHERE id = ?", (1,))
        assert len(result) == 1
        assert result[0]["name"] == "test"

    def test_execute_many(self, db_manager):
        """Test execute_many operation."""
        db_manager.initialize()
        db_manager.execute_script("CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)")

        data = [(1, "test1"), (2, "test2"), (3, "test3")]
        db_manager.execute_many("INSERT INTO test (id, name) VALUES (?, ?)", data)

        result = db_manager.execute_query("SELECT * FROM test ORDER BY id")
        assert len(result) == 3
        assert result[0]["name"] == "test1"
        assert result[1]["name"] == "test2"
        assert result[2]["name"] == "test3"

    def test_execute_script(self, db_manager):
        """Test execute_script operation."""
        db_manager.initialize()
        script = """
        CREATE TABLE test1 (id INTEGER PRIMARY KEY);
        CREATE TABLE test2 (id INTEGER PRIMARY KEY);
        INSERT INTO test1 (id) VALUES (1);
        INSERT INTO test2 (id) VALUES (2);
        """
        db_manager.execute_script(script)

        result1 = db_manager.execute_query("SELECT * FROM test1")
        result2 = db_manager.execute_query("SELECT * FROM test2")
        assert len(result1) == 1
        assert len(result2) == 1

    def test_execute_single_success(self, db_manager):
        """Test successful single query execution."""
        db_manager.initialize()

        result = db_manager.execute_single("SELECT COUNT(*) FROM messages")

        # execute_single returns sqlite3.Row objects, not tuples
        assert result is not None
        assert hasattr(result, "__getitem__")  # Should be a row-like object
        assert result[0] >= 0  # Count should be non-negative

    def test_execute_single_with_params(self, db_manager):
        """Test execute_single with parameters."""
        db_manager.initialize()
        db_manager.execute_script("""
            CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT);
            INSERT INTO test (id, name) VALUES (1, 'test');
        """)

        result = db_manager.execute_single("SELECT name FROM test WHERE id = ?", (1,))
        assert result[0] == "test"

    def test_execute_script_success(self, db_manager):
        """Test successful script execution."""
        db_manager.initialize()

        script = """
            CREATE TABLE IF NOT EXISTS test_script_table (
                id TEXT PRIMARY KEY,
                name TEXT
            );
            INSERT INTO test_script_table (id, name) VALUES ('test1', 'Test Name');
        """

        db_manager.execute_script(script)

        # Verify script execution
        result = db_manager.execute_single(
            "SELECT name FROM test_script_table WHERE id = ?", ("test1",)
        )
        assert result[0] == "Test Name"

    def test_backup_database(self, db_manager, temp_db_path):
        """Test database backup functionality."""
        db_manager.initialize()

        # Create backup
        backup_path = temp_db_path + ".backup"
        db_manager.backup_database(backup_path)

        # Verify backup exists and has data
        assert os.path.exists(backup_path)
        assert os.path.getsize(backup_path) > 0

    def test_vacuum_database(self, db_manager):
        """Test database vacuum operation."""
        db_manager.initialize()

        # Vacuum should complete without error
        db_manager.vacuum()

        # Database should still be accessible
        result = db_manager.execute_single("SELECT COUNT(*) FROM messages")
        assert result is not None
        assert hasattr(result, "__getitem__")  # Should be a row-like object

    def test_analyze_tables(self, db_manager):
        """Test database analyze operation."""
        db_manager.initialize()

        # Analyze should complete without error
        db_manager.analyze_tables()

        # Database should still be accessible
        result = db_manager.execute_single("SELECT COUNT(*) FROM messages")
        assert result is not None
        assert hasattr(result, "__getitem__")  # Should be a row-like object

    def test_get_table_names(self, db_manager):
        """Test getting table names."""
        db_manager.initialize()

        table_names = db_manager.get_table_names()

        # Should have at least the messages table
        assert "messages" in table_names
        assert isinstance(table_names, list)

    def test_get_table_info(self, db_manager):
        """Test getting table information."""
        db_manager.initialize()

        table_info = db_manager.get_table_info("messages")

        assert isinstance(table_info, list)
        assert len(table_info) > 0
        assert "name" in table_info[0]

    def test_get_row_count(self, db_manager):
        """Test getting row count."""
        db_manager.initialize()

        count = db_manager.get_row_count("messages")

        assert isinstance(count, int)
        assert count >= 0

    def test_database_path_property(self, db_manager):
        """Test database_path property."""
        assert db_manager.db_path is not None
        assert db_manager.db_path.endswith(".db")

    def test_connection_count_tracking(self, db_manager):
        """Test connection count tracking."""
        assert db_manager._connection_count == 0
        db_manager.initialize()
        assert db_manager._connection_count >= 1

    def test_context_manager(self, temp_db_path):
        """Test using DatabaseManager as context manager."""
        with DatabaseManager(temp_db_path) as db_manager:
            db_manager.execute_script("CREATE TABLE test (id INTEGER PRIMARY KEY)")
            result = db_manager.execute_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='test'"
            )
            assert len(result) == 1

    def test_close_connections(self, db_manager):
        """Test closing connections."""
        db_manager.initialize()
        assert hasattr(db_manager._local, 'connection')
        assert db_manager._local.connection is not None
        
        db_manager.close_connections()
        assert db_manager._local.connection is None

    def test_health_check(self, db_manager):
        """Test health check functionality."""
        db_manager.initialize()
        
        health = db_manager.health_check()
        assert health['status'] == 'healthy'
        assert 'total_messages' in health
        assert 'db_path' in health
        assert 'connection_count' in health

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
    def test_initialization_failure_variations(self, test_case):
        """Test initialization failure handling with various error types."""
        error_type = test_case["error_type"]
        error_message = test_case["error_message"]

        with patch("sqlite3.connect") as mock_connect:
            mock_connect.side_effect = error_type(error_message)

            db_manager = DatabaseManager("invalid.db")

            with pytest.raises(error_type, match=error_message):
                db_manager.initialize()
