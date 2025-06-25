"""
Database repository for low-level database operations like exports and metadata.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from pepino.data.database.manager import DatabaseManager

logger = logging.getLogger(__name__)


class DatabaseRepository:
    """Repository for low-level database operations."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    async def get_table_names(self) -> List[str]:
        """Get all table names in the database."""
        try:
            query = "SELECT name FROM sqlite_master WHERE type='table'"
            results = await self.db.execute_many(query)
            return [row[0] for row in results]

        except Exception as e:
            logger.error(f"Error getting table names: {e}")
            raise

    async def export_table_data(self, table_name: str) -> Dict[str, Any]:
        """Export all data from a specific table."""
        try:
            query = f"SELECT * FROM {table_name}"
            results = await self.db.execute_many(query)

            # Get column names
            column_query = f"PRAGMA table_info({table_name})"
            column_results = await self.db.execute_many(column_query)
            columns = [row[1] for row in column_results]  # Column name is at index 1

            # Convert rows to dictionaries
            rows = [dict(zip(columns, row)) for row in results]

            return {"table": table_name, "columns": columns, "rows": rows}

        except Exception as e:
            logger.error(f"Error exporting table {table_name}: {e}")
            raise

    async def export_all_tables(self) -> Dict[str, Any]:
        """Export data from all tables in the database."""
        try:
            tables = await self.get_table_names()
            all_data = {}

            for table_name in tables:
                table_data = await self.export_table_data(table_name)
                all_data[table_name] = {
                    "columns": table_data["columns"],
                    "rows": table_data["rows"],
                }

            return {"tables": all_data}

        except Exception as e:
            logger.error(f"Error exporting all tables: {e}")
            raise
