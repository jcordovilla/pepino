#!/usr/bin/env python3
"""Database setup and initialization script"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pepino.data.database.manager import DatabaseManager

def main():
    db_manager = DatabaseManager()
    # Initialize database by making a connection (triggers schema creation)
    with db_manager.get_connection() as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"Database initialized successfully! Found {len(tables)} tables:")
        for table in tables:
            print(f"  - {table[0]}")
    print("Database setup complete!")

if __name__ == "__main__":
    main() 