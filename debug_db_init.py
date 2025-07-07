#!/usr/bin/env python3

import os
import sqlite3
from datetime import datetime, timedelta
from src.pepino.data.database.schema import init_database

def debug_database_creation():
    """Debug the database creation process."""
    db_path = "debug_test.db"
    
    print(f"DEBUG: Creating database at {db_path}")
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Initialize database
    try:
        init_database(db_path)
        print("DEBUG: Database initialized successfully")
    except Exception as e:
        print(f"DEBUG: Error initializing database: {e}")
        return
    
    # Check if database file exists and has content
    if os.path.exists(db_path):
        size = os.path.getsize(db_path)
        print(f"DEBUG: Database file exists, size: {size} bytes")
    else:
        print("DEBUG: Database file does not exist!")
        return
    
    # Connect and check tables
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # List tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"DEBUG: Tables in database: {tables}")
        
        # Check messages table structure
        cursor.execute("PRAGMA table_info(messages);")
        columns = cursor.fetchall()
        print(f"DEBUG: Messages table has {len(columns)} columns")
        
        # Try to insert a test message
        base_time = datetime.utcnow() - timedelta(days=1)
        cursor.execute("""
            INSERT INTO messages (
                id, content, timestamp, author_id, author_name, channel_id, channel_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            "test_msg_1",
            "Test message",
            base_time.isoformat(),
            "user1",
            "test_user",
            "test_channel",
            "test_channel"
        ))
        
        conn.commit()
        print("DEBUG: Test message inserted successfully")
        
        # Check message count
        cursor.execute("SELECT COUNT(*) FROM messages")
        count = cursor.fetchone()[0]
        print(f"DEBUG: Messages in database: {count}")
        
        conn.close()
        
    except Exception as e:
        print(f"DEBUG: Error working with database: {e}")

if __name__ == "__main__":
    debug_database_creation() 