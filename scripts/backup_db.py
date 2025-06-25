#!/usr/bin/env python3
"""Database backup script"""

import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for logging imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pepino.logging_config import get_logger, setup_cli_logging

# Set up logging
setup_cli_logging(verbose=True)
logger = get_logger(__name__)

def backup_database():
    db_path = Path("data/discord_messages.db")
    backup_dir = Path("data/backups")
    backup_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"discord_messages_{timestamp}.db"
    
    try:
        shutil.copy2(db_path, backup_path)
        logger.info(f"✅ Database backed up to: {backup_path}")
    except Exception as e:
        logger.error(f"❌ Failed to backup database: {e}")
        raise

if __name__ == "__main__":
    backup_database() 