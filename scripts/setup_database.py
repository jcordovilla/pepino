#!/usr/bin/env python3
"""Database setup and initialization script"""

import asyncio
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pepino.data.database import DatabaseManager

async def main():
    db_manager = DatabaseManager()
    await db_manager.initialize()
    print("Database initialized successfully!")

if __name__ == "__main__":
    asyncio.run(main()) 