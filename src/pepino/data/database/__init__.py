"""
Database package for database operations.

Provides database management, schema operations, and connection handling.
"""

from .manager import DatabaseManager
from .schema import get_schema_version, init_database, validate_schema

__all__ = [
    # Database management
    "DatabaseManager",
    # Schema operations
    "init_database",
    "validate_schema", 
    "get_schema_version",
]
