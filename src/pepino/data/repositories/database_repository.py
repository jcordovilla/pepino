"""
Database repository for low-level database operations like exports and metadata.
"""

from typing import Any, Dict, List

from ..database.manager import DatabaseManager
from ...logging_config import get_logger

logger = get_logger(__name__)


class DatabaseRepository:
    """Repository for low-level database operations."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def get_table_names(self) -> List[str]:
        """Get all table names in the database."""
        try:
            query = "SELECT name FROM sqlite_master WHERE type='table'"
            results = self.db.execute_query(query)
            return [row['name'] for row in results if row['name'] != 'sqlite_sequence']

        except Exception as e:
            logger.error(f"Error getting table names: {e}")
            raise
    
    def get_table_row_count(self, table_name: str) -> int:
        """Get row count for a specific table."""
        try:
            query = f"SELECT COUNT(*) as count FROM {table_name}"
            result = self.db.execute_query(query, fetch_one=True)
            return result['count'] if result else 0
        except Exception as e:
            logger.error(f"Error getting row count for table {table_name}: {e}")
            return 0

    def export_table_data(self, table_name: str) -> Dict[str, Any]:
        """Export all data from a specific table."""
        try:
            query = f"SELECT * FROM {table_name}"
            results = self.db.execute_many(query)

            # Get column names
            column_query = f"PRAGMA table_info({table_name})"
            column_results = self.db.execute_many(column_query)
            columns = [row[1] for row in column_results]  # Column name is at index 1

            # Convert rows to dictionaries
            rows = [dict(zip(columns, row)) for row in results]

            return {"table": table_name, "columns": columns, "rows": rows}

        except Exception as e:
            logger.error(f"Error exporting table {table_name}: {e}")
            raise

    def export_all_tables(self) -> Dict[str, Any]:
        """Export data from all tables in the database."""
        try:
            tables = self.get_table_names()
            all_data = {}

            for table_name in tables:
                table_data = self.export_table_data(table_name)
                all_data[table_name] = {
                    "columns": table_data["columns"],
                    "rows": table_data["rows"],
                }

            return {"tables": all_data}

        except Exception as e:
            logger.error(f"Error exporting all tables: {e}")
            raise
