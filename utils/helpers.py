"""
Utility helper functions for Discord message analysis
"""
import os
import tempfile
from datetime import datetime
from typing import Optional


# Create temp directory for graphs
TEMP_DIR = tempfile.mkdtemp()


def format_timestamp(timestamp: str) -> str:
    """Format a timestamp string to a readable format"""
    try:
        # Handle different timestamp formats
        if 'T' in timestamp:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        return dt.strftime('%Y-%m-%d %H:%M')
    except:
        return timestamp


def cleanup_temp_files():
    """Clean up temporary graph files"""
    for file in os.listdir(TEMP_DIR):
        try:
            os.remove(os.path.join(TEMP_DIR, file))
        except:
            pass
    try:
        os.rmdir(TEMP_DIR)
    except:
        pass


def get_temp_dir() -> str:
    """Get the temporary directory path"""
    return TEMP_DIR


def get_temp_file_path(filename: str) -> str:
    """Get a path for a temporary file"""
    return os.path.join(TEMP_DIR, filename)


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be used as a filename"""
    # Remove emojis and special characters
    import re
    clean_name = re.sub(r'[^\w\s-]', '', name)
    return clean_name.replace(' ', '_').strip('_')


def ensure_temp_dir_exists():
    """Ensure the temp directory exists"""
    os.makedirs(TEMP_DIR, exist_ok=True)
