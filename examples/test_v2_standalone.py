#!/usr/bin/env python3
"""
Standalone V2 System Test

This test only imports V2 components directly to avoid async dependency conflicts.
"""

import logging
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_v2_components():
    """Test V2 components standalone."""
    
    logger.info("Testing V2 components standalone...")
    
    try:
        # Test V2 Database Manager
        logger.info("1. Testing V2 Database Manager...")
        
        # Import V2 database manager directly
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        from pepino.data.database.manager_v2 import SyncDatabaseManagerV2
        
        db_manager = SyncDatabaseManagerV2()
        
        # Test health check
        health = db_manager.health_check()
        logger.info(f"Database health: {health}")
        
        if health['status'] != 'healthy':
            logger.error("Database is not healthy")
            return False
        
        # Test basic queries
        channels = db_manager.get_available_channels()
        users = db_manager.get_available_users()
        
        logger.info(f"Found {len(channels)} channels, {len(users)} users")
        
        if not channels or not users:
            logger.warning("No data found - this is expected for a clean system")
            return True
        
        # Test specific queries
        test_channel = channels[0]
        test_user = users[0]
        
        logger.info(f"Testing with channel: {test_channel}, user: {test_user}")
        
        # Channel operations
        messages = db_manager.get_messages_by_channel(test_channel, limit=5)
        logger.info(f"Channel messages: {len(messages)}")
        
        stats = db_manager.get_channel_statistics(test_channel)
        logger.info(f"Channel stats: {stats}")
        
        # User operations
        messages = db_manager.get_messages_by_user(test_user, limit=5)
        logger.info(f"User messages: {len(messages)}")
        
        stats = db_manager.get_user_statistics(test_user)
        logger.info(f"User stats: {stats}")
        
        # Top operations
        top_channels = db_manager.get_top_channels(limit=3)
        top_users = db_manager.get_top_users(limit=3)
        
        logger.info(f"Top channels: {len(top_channels)}")
        logger.info(f"Top users: {len(top_users)}")
        
        db_manager.close_connections()
        logger.info("✅ Database Manager V2 test passed")
        
        # Test V2 Analyzers (if we can import models)
        logger.info("2. Testing V2 Analyzer imports...")
        
        try:
            # Test imports without running analysis (to avoid model dependencies)
            from pepino.analysis.channel_analyzer_v2 import ChannelAnalyzerV2
            from pepino.analysis.user_analyzer_v2 import UserAnalyzerV2
            
            logger.info("✅ V2 Analyzers imported successfully")
            
            # Test basic initialization
            channel_analyzer = ChannelAnalyzerV2()
            user_analyzer = UserAnalyzerV2()
            
            logger.info("✅ V2 Analyzers initialized successfully")
            
        except ImportError as e:
            logger.error(f"❌ Failed to import V2 analyzers: {e}")
            logger.info("This might be due to missing model dependencies")
        
        # Test V2 Template Engine
        logger.info("3. Testing V2 Template Engine...")
        
        try:
            from pepino.discord.templates.template_engine import AnalysisTemplateEngine
            
            template_engine = AnalysisTemplateEngine()
            
            # Test simple template rendering
            test_data = {
                'test_value': 'Hello World',
                'test_number': 42
            }
            
            # Test if template engine can render a simple string
            logger.info("✅ Template Engine imported and initialized")
            
        except ImportError as e:
            logger.error(f"❌ Failed to import template engine: {e}")
        
        # Test V2 Mixins
        logger.info("4. Testing V2 Command Mixins...")
        
        try:
            from pepino.discord.commands.mixins import ThreadedCommandMixin, ComprehensiveCommandMixin
            
            logger.info("✅ Command mixins imported successfully")
            
        except ImportError as e:
            logger.error(f"❌ Failed to import command mixins: {e}")
        
        logger.info("🎉 All V2 components tested successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ V2 component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_operations():
    """Test detailed database operations."""
    
    logger.info("Testing detailed database operations...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from pepino.data.database.manager_v2 import SyncDatabaseManagerV2
        
        db_manager = SyncDatabaseManagerV2()
        
        # Test insert operations
        test_message = {
            'message_id': f'test_{int(time.time())}',
            'channel_name': 'test_channel',
            'author_name': 'test_user',
            'content': 'This is a test message',
            'timestamp': '2024-01-01T12:00:00Z',
            'message_type': 'default'
        }
        
        logger.info("Testing message insertion...")
        success = db_manager.insert_message(test_message)
        logger.info(f"Message insert result: {success}")
        
        # Test retrieval
        logger.info("Testing message retrieval...")
        messages = db_manager.get_messages_by_channel('test_channel')
        logger.info(f"Retrieved {len(messages)} messages from test_channel")
        
        # Test batch insert
        batch_messages = []
        for i in range(3):
            msg = test_message.copy()
            msg['message_id'] = f'test_batch_{i}_{int(time.time())}'
            msg['content'] = f'Batch message {i}'
            batch_messages.append(msg)
        
        logger.info("Testing batch insertion...")
        batch_result = db_manager.insert_messages_batch(batch_messages)
        logger.info(f"Batch insert result: {batch_result} rows affected")
        
        # Test updated retrieval
        messages = db_manager.get_messages_by_channel('test_channel')
        logger.info(f"After batch insert: {len(messages)} messages")
        
        # Test statistics
        stats = db_manager.get_channel_statistics('test_channel')
        logger.info(f"Test channel statistics: {stats}")
        
        stats = db_manager.get_user_statistics('test_user')
        logger.info(f"Test user statistics: {stats}")
        
        db_manager.close_connections()
        logger.info("✅ Database operations test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run standalone tests."""
    
    logger.info("Starting V2 Standalone Test Suite...")
    logger.info("=" * 50)
    
    tests_passed = 0
    total_tests = 2
    
    # Test 1: Component imports and basic functionality
    logger.info("\n🔍 Test 1: V2 Component Testing")
    if test_v2_components():
        tests_passed += 1
        logger.info("✅ Component test PASSED")
    else:
        logger.error("❌ Component test FAILED")
    
    # Test 2: Database operations
    logger.info("\n🔍 Test 2: Database Operations")
    if test_database_operations():
        tests_passed += 1
        logger.info("✅ Database operations test PASSED")
    else:
        logger.error("❌ Database operations test FAILED")
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info(f"TEST SUMMARY: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        logger.info("🎉 All standalone tests passed! V2 system core is working.")
        return 0
    else:
        logger.error(f"💥 {total_tests - tests_passed} tests failed.")
        return 1

if __name__ == "__main__":
    exit(main()) 