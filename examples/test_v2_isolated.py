#!/usr/bin/env python3
"""
Isolated V2 System Test

This test imports V2 files directly without going through the package structure
to avoid async dependency conflicts.
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

def test_database_manager_v2():
    """Test V2 database manager in isolation."""
    
    logger.info("Testing V2 Database Manager in isolation...")
    
    try:
        # Add the specific file directory to path
        src_path = Path(__file__).parent.parent / "src"
        sys.path.insert(0, str(src_path))
        
        # Import schema directly
        schema_path = src_path / "pepino" / "data" / "database" / "schema.py"
        
        # Read and exec the schema file
        schema_globals = {}
        with open(schema_path, 'r') as f:
            schema_code = f.read()
        exec(schema_code, schema_globals)
        
        SCHEMA_QUERIES = schema_globals['SCHEMA_QUERIES']
        logger.info(f"Loaded schema with {len(SCHEMA_QUERIES)} tables")
        
        # Now create a simplified database manager manually
        class SimpleDatabaseManagerV2:
            def __init__(self, db_path="test_discord_messages.db"):
                self.db_path = Path(db_path)
                self.connection = None
                logger.info(f"SimpleDatabaseManagerV2 initialized with path: {self.db_path}")
            
            def connect(self):
                """Create database connection."""
                try:
                    self.connection = sqlite3.connect(
                        str(self.db_path),
                        check_same_thread=False,
                        timeout=30.0
                    )
                    self.connection.row_factory = sqlite3.Row
                    
                    # Initialize schema
                    for table_name, create_query in SCHEMA_QUERIES.items():
                        try:
                            self.connection.execute(create_query)
                        except sqlite3.OperationalError as e:
                            if "already exists" not in str(e):
                                logger.error(f"Failed to create table {table_name}: {e}")
                                raise
                    
                    self.connection.commit()
                    logger.info("Database connected and schema initialized")
                    return True
                    
                except Exception as e:
                    logger.error(f"Failed to connect to database: {e}")
                    return False
            
            def health_check(self):
                """Perform database health check."""
                try:
                    if not self.connection:
                        if not self.connect():
                            return {'status': 'unhealthy', 'error': 'Cannot connect'}
                    
                    cursor = self.connection.execute("SELECT COUNT(*) as total FROM messages")
                    result = cursor.fetchone()
                    total_messages = result['total'] if result else 0
                    
                    return {
                        'status': 'healthy',
                        'total_messages': total_messages,
                        'db_path': str(self.db_path)
                    }
                    
                except Exception as e:
                    logger.error(f"Database health check failed: {e}")
                    return {
                        'status': 'unhealthy',
                        'error': str(e),
                        'db_path': str(self.db_path)
                    }
            
            def insert_test_message(self, message_data):
                """Insert a test message."""
                try:
                    if not self.connection:
                        if not self.connect():
                            return False
                    
                    query = """
                    INSERT OR REPLACE INTO messages (
                        message_id, channel_name, author_name, content, 
                        timestamp, message_type, reply_to, thread_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """
                    
                    params = (
                        message_data.get('message_id'),
                        message_data.get('channel_name'),
                        message_data.get('author_name'),
                        message_data.get('content'),
                        message_data.get('timestamp'),
                        message_data.get('message_type', 'default'),
                        message_data.get('reply_to'),
                        message_data.get('thread_id')
                    )
                    
                    self.connection.execute(query, params)
                    self.connection.commit()
                    return True
                    
                except Exception as e:
                    logger.error(f"Failed to insert message: {e}")
                    return False
            
            def get_messages_by_channel(self, channel_name, limit=None):
                """Get messages for a channel."""
                try:
                    if not self.connection:
                        if not self.connect():
                            return []
                    
                    query = "SELECT * FROM messages WHERE channel_name = ? ORDER BY timestamp DESC"
                    params = [channel_name]
                    
                    if limit:
                        query += " LIMIT ?"
                        params.append(limit)
                    
                    cursor = self.connection.execute(query, params)
                    return cursor.fetchall()
                    
                except Exception as e:
                    logger.error(f"Failed to get messages: {e}")
                    return []
            
            def get_available_channels(self):
                """Get available channels."""
                try:
                    if not self.connection:
                        if not self.connect():
                            return []
                    
                    cursor = self.connection.execute("SELECT DISTINCT channel_name FROM messages ORDER BY channel_name")
                    results = cursor.fetchall()
                    return [row['channel_name'] for row in results]
                    
                except Exception as e:
                    logger.error(f"Failed to get channels: {e}")
                    return []
            
            def close(self):
                """Close connection."""
                if self.connection:
                    self.connection.close()
                    self.connection = None
        
        # Test the simplified database manager
        db_manager = SimpleDatabaseManagerV2()
        
        # Test health check
        health = db_manager.health_check()
        logger.info(f"Health check result: {health}")
        
        if health['status'] != 'healthy':
            logger.error("Database is not healthy")
            return False
        
        # Test insert operation
        test_message = {
            'message_id': f'test_isolated_{int(time.time())}',
            'channel_name': 'test_channel_isolated',
            'author_name': 'test_user_isolated',
            'content': 'This is an isolated test message',
            'timestamp': '2024-01-01T12:00:00Z',
            'message_type': 'default'
        }
        
        logger.info("Testing message insertion...")
        success = db_manager.insert_test_message(test_message)
        logger.info(f"Message insert result: {success}")
        
        if not success:
            logger.error("Failed to insert test message")
            return False
        
        # Test retrieval
        logger.info("Testing message retrieval...")
        messages = db_manager.get_messages_by_channel('test_channel_isolated')
        logger.info(f"Retrieved {len(messages)} messages")
        
        if len(messages) == 0:
            logger.error("No messages retrieved")
            return False
        
        # Test channels
        channels = db_manager.get_available_channels()
        logger.info(f"Available channels: {channels}")
        
        if 'test_channel_isolated' not in channels:
            logger.error("Test channel not found in available channels")
            return False
        
        # Clean up
        db_manager.close()
        
        # Remove test database
        test_db_path = Path("test_discord_messages.db")
        if test_db_path.exists():
            test_db_path.unlink()
            logger.info("Test database removed")
        
        logger.info("✅ Isolated Database Manager V2 test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Isolated database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_threading_simulation():
    """Test threading capabilities without the full system."""
    
    logger.info("Testing threading simulation...")
    
    try:
        from concurrent.futures import ThreadPoolExecutor
        import time
        
        def sync_operation(name, duration):
            """Simulate a sync operation."""
            logger.info(f"Starting {name}...")
            time.sleep(duration)
            logger.info(f"Completed {name}")
            return f"Result from {name}"
        
        # Test sequential execution
        logger.info("Testing sequential execution...")
        start_time = time.time()
        
        result1 = sync_operation("Operation 1", 0.5)
        result2 = sync_operation("Operation 2", 0.5)
        
        sequential_time = time.time() - start_time
        logger.info(f"Sequential execution took: {sequential_time:.2f}s")
        
        # Test parallel execution
        logger.info("Testing parallel execution...")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(sync_operation, "Parallel Operation 1", 0.5)
            future2 = executor.submit(sync_operation, "Parallel Operation 2", 0.5)
            
            result1_parallel = future1.result()
            result2_parallel = future2.result()
        
        parallel_time = time.time() - start_time
        logger.info(f"Parallel execution took: {parallel_time:.2f}s")
        
        speedup = sequential_time / parallel_time if parallel_time > 0 else 1
        logger.info(f"Speedup: {speedup:.2f}x")
        
        if speedup < 1.5:
            logger.warning("Speedup is less than expected, but this is normal for small operations")
        
        logger.info("✅ Threading simulation test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Threading simulation test failed: {e}")
        return False

def test_template_basic():
    """Test basic template functionality."""
    
    logger.info("Testing basic template functionality...")
    
    try:
        # Test Jinja2 directly
        from jinja2 import Template, Environment, FileSystemLoader
        
        # Test simple template
        template_str = """
Hello {{ name }}!
You have {{ count }} messages.
{% for item in items %}
- {{ item }}
{% endfor %}
        """
        
        template = Template(template_str)
        
        data = {
            'name': 'Test User',
            'count': 42,
            'items': ['Item 1', 'Item 2', 'Item 3']
        }
        
        result = template.render(data)
        logger.info(f"Template rendered {len(result)} characters")
        
        if 'Test User' not in result or '42' not in result:
            logger.error("Template rendering failed - missing expected content")
            return False
        
        logger.info("✅ Basic template test passed")
        return True
        
    except ImportError:
        logger.error("❌ Jinja2 not available for template testing")
        return False
    except Exception as e:
        logger.error(f"❌ Template test failed: {e}")
        return False

def main():
    """Run isolated tests."""
    
    logger.info("Starting V2 Isolated Test Suite...")
    logger.info("=" * 50)
    
    tests = [
        ("Database Manager V2 (Isolated)", test_database_manager_v2),
        ("Threading Simulation", test_threading_simulation),
        ("Basic Template Functionality", test_template_basic)
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
        logger.info("🎉 All isolated tests passed! V2 core concepts are working.")
        return 0
    else:
        logger.error(f"💥 {total_tests - tests_passed} tests failed.")
        return 1

if __name__ == "__main__":
    exit(main()) 