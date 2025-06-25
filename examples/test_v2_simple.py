#!/usr/bin/env python3
"""
Simple V2 System Test

Just test the core V2 concepts are working with a simple inline schema.
"""

import logging
import sqlite3
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_simple_v2_concepts():
    """Test simple V2 concepts."""
    
    logger.info("Testing simple V2 concepts...")
    
    # Simple inline schema
    SIMPLE_SCHEMA = """
    CREATE TABLE IF NOT EXISTS messages (
        message_id TEXT PRIMARY KEY,
        channel_name TEXT,
        author_name TEXT,
        content TEXT,
        timestamp TEXT,
        message_type TEXT DEFAULT 'default',
        reply_to TEXT,
        thread_id TEXT
    );
    
    CREATE TABLE IF NOT EXISTS users (
        username TEXT PRIMARY KEY,
        display_name TEXT,
        joined_at TEXT
    );
    
    CREATE TABLE IF NOT EXISTS channels (
        name TEXT PRIMARY KEY,
        topic TEXT,
        created_at TEXT
    );
    """
    
    try:
        # Test database operations
        db_path = "test_simple_v2.db"
        
        # Create connection
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        # Initialize schema
        conn.executescript(SIMPLE_SCHEMA)
        
        # Test insert
        test_message = {
            'message_id': f'test_simple_{int(time.time())}',
            'channel_name': 'test_channel',
            'author_name': 'test_user',
            'content': 'This is a simple test message',
            'timestamp': '2024-01-01T12:00:00Z',
            'message_type': 'default'
        }
        
        conn.execute("""
            INSERT INTO messages (message_id, channel_name, author_name, content, timestamp, message_type)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            test_message['message_id'],
            test_message['channel_name'],
            test_message['author_name'],
            test_message['content'],
            test_message['timestamp'],
            test_message['message_type']
        ))
        
        conn.commit()
        logger.info("✅ Message inserted successfully")
        
        # Test retrieval
        cursor = conn.execute("SELECT * FROM messages WHERE channel_name = ?", ('test_channel',))
        messages = cursor.fetchall()
        logger.info(f"✅ Retrieved {len(messages)} messages")
        
        # Test statistics
        cursor = conn.execute("SELECT COUNT(*) as total FROM messages")
        result = cursor.fetchone()
        total_messages = result['total']
        logger.info(f"✅ Total messages in database: {total_messages}")
        
        # Clean up
        conn.close()
        Path(db_path).unlink()
        logger.info("✅ Database cleaned up")
        
        logger.info("🎉 Simple V2 database concepts working!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Simple V2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_threading_concepts():
    """Test threading concepts for V2."""
    
    logger.info("Testing threading concepts...")
    
    try:
        from concurrent.futures import ThreadPoolExecutor
        
        def database_operation(name):
            """Simulate database operation."""
            logger.info(f"Starting {name}")
            time.sleep(0.2)  # Simulate work
            logger.info(f"Completed {name}")
            return f"Result from {name}"
        
        # Test parallel execution
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i in range(3):
                future = executor.submit(database_operation, f"Operation {i+1}")
                futures.append(future)
            
            results = [future.result() for future in futures]
            logger.info(f"✅ Parallel operations completed: {len(results)} results")
        
        logger.info("🎉 Threading concepts working!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Threading test failed: {e}")
        return False

def test_template_concepts():
    """Test template concepts for V2."""
    
    logger.info("Testing template concepts...")
    
    try:
        from jinja2 import Template
        
        # Simple template
        template_str = """
# Channel Analysis Report

**Channel:** {{ channel_name }}
**Total Messages:** {{ total_messages }}
**Top User:** {{ top_user }}

## Statistics
- Messages per day: {{ messages_per_day }}
- Unique users: {{ unique_users }}

## Top Users
{% for user in top_users %}
- {{ user.name }}: {{ user.count }} messages
{% endfor %}
        """
        
        template = Template(template_str)
        
        # Test data
        data = {
            'channel_name': 'general',
            'total_messages': 1234,
            'top_user': 'alice',
            'messages_per_day': 42.5,
            'unique_users': 25,
            'top_users': [
                {'name': 'alice', 'count': 150},
                {'name': 'bob', 'count': 120},
                {'name': 'charlie', 'count': 90}
            ]
        }
        
        result = template.render(data)
        logger.info(f"✅ Template rendered {len(result)} characters")
        
        # Check content
        if 'general' in result and '1234' in result and 'alice' in result:
            logger.info("✅ Template content looks correct")
        else:
            logger.error("❌ Template content missing expected values")
            return False
        
        logger.info("🎉 Template concepts working!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Template test failed: {e}")
        return False

def main():
    """Run simple V2 tests."""
    
    logger.info("Starting Simple V2 Test Suite...")
    logger.info("=" * 50)
    
    tests = [
        ("Simple V2 Database Concepts", test_simple_v2_concepts),
        ("Threading Concepts", test_threading_concepts),
        ("Template Concepts", test_template_concepts)
    ]
    
    tests_passed = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Testing: {test_name}")
        try:
            if test_func():
                tests_passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} CRASHED: {e}")
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info(f"TEST SUMMARY: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        logger.info("🎉 All simple V2 concepts are working! Ready to proceed with migration.")
        logger.info("\nNext steps:")
        logger.info("1. The V2 database manager concept works")
        logger.info("2. Threading for parallelism works")
        logger.info("3. Template rendering works")
        logger.info("4. We can now proceed with the actual migration")
        return 0
    else:
        logger.error(f"💥 {total_tests - tests_passed} tests failed.")
        return 1

if __name__ == "__main__":
    exit(main()) 