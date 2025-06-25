#!/usr/bin/env python3
"""
Test V2 System Components

This script tests the V2 system components to ensure they work correctly
before we proceed with the migration.

Tests:
1. V2 Database Manager
2. V2 Channel Analyzer
3. V2 User Analyzer
4. Template rendering
5. Threading system
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pepino.data.database.manager import SyncDatabaseManager
from pepino.analysis.channel_analyzer import ChannelAnalyzer
from pepino.analysis.user_analyzer import UserAnalyzer
from pepino.discord.templates.template_engine import AnalysisTemplateEngine
from pepino.discord.commands.mixins import ThreadedCommandMixin
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestV2System:
    """Test all V2 system components."""
    
    def __init__(self):
        self.db_manager = SyncDatabaseManager()
        self.channel_analyzer = ChannelAnalyzer()
        self.user_analyzer = UserAnalyzer()
        self.template_engine = AnalysisTemplateEngine()
        
        # Create thread pool for testing
        self.thread_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="test")
        
        logger.info("Test system initialized")
    
    def test_database_manager(self):
        """Test V2 database manager."""
        
        logger.info("Testing V2 Database Manager...")
        
        try:
            # Test health check
            health = self.db_manager.health_check()
            logger.info(f"Database health: {health}")
            assert health['status'] == 'healthy', "Database should be healthy"
            
            # Test basic queries
            channels = self.db_manager.get_available_channels()
            logger.info(f"Found {len(channels)} channels")
            assert len(channels) > 0, "Should have some channels"
            
            users = self.db_manager.get_available_users()
            logger.info(f"Found {len(users)} users")
            assert len(users) > 0, "Should have some users"
            
            # Test specific queries
            if channels:
                test_channel = channels[0]
                messages = self.db_manager.get_messages_by_channel(test_channel, limit=10)
                logger.info(f"Channel '{test_channel}' has {len(messages)} recent messages")
                
                stats = self.db_manager.get_channel_statistics(test_channel)
                logger.info(f"Channel stats: {stats}")
            
            if users:
                test_user = users[0]
                messages = self.db_manager.get_messages_by_user(test_user, limit=10)
                logger.info(f"User '{test_user}' has {len(messages)} recent messages")
                
                stats = self.db_manager.get_user_statistics(test_user)
                logger.info(f"User stats: {stats}")
            
            logger.info("✅ Database Manager tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database Manager test failed: {e}")
            return False
    
    def test_channel_analyzer(self):
        """Test V2 channel analyzer."""
        
        logger.info("Testing V2 Channel Analyzer...")
        
        try:
            # Get available channels
            channels = self.channel_analyzer.get_available_channels()
            logger.info(f"Analyzer found {len(channels)} channels")
            
            if not channels:
                logger.warning("No channels available for testing")
                return True
            
            # Test analysis
            test_channel = channels[0]
            logger.info(f"Testing analysis for channel: {test_channel}")
            
            start_time = time.time()
            analysis = self.channel_analyzer.analyze(test_channel)
            exec_time = time.time() - start_time
            
            if analysis:
                logger.info(f"Channel analysis completed in {exec_time:.2f}s")
                logger.info(f"Statistics: {analysis.statistics}")
                logger.info(f"Top users: {len(analysis.top_users)}")
                logger.info(f"Time patterns keys: {list(analysis.time_patterns.keys())}")
                logger.info(f"Summary keys: {list(analysis.summary.keys())}")
            else:
                logger.warning(f"No analysis result for channel: {test_channel}")
            
            # Test other methods
            top_channels = self.channel_analyzer.get_top_channels(limit=5)
            logger.info(f"Top 5 channels: {[ch['channel_name'] for ch in top_channels]}")
            
            health = self.channel_analyzer.get_channel_health(test_channel)
            logger.info(f"Channel health: {health}")
            
            logger.info("✅ Channel Analyzer tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Channel Analyzer test failed: {e}")
            return False
    
    def test_user_analyzer(self):
        """Test V2 user analyzer."""
        
        logger.info("Testing V2 User Analyzer...")
        
        try:
            # Get available users
            users = self.user_analyzer.get_available_users()
            logger.info(f"Analyzer found {len(users)} users")
            
            if not users:
                logger.warning("No users available for testing")
                return True
            
            # Test analysis
            test_user = users[0]
            logger.info(f"Testing analysis for user: {test_user}")
            
            start_time = time.time()
            analysis = self.user_analyzer.analyze(test_user)
            exec_time = time.time() - start_time
            
            if analysis:
                logger.info(f"User analysis completed in {exec_time:.2f}s")
                logger.info(f"Statistics: {analysis.statistics}")
                logger.info(f"Channel activity: {len(analysis.channel_activity)}")
                logger.info(f"Time patterns keys: {list(analysis.time_patterns.keys())}")
                logger.info(f"Summary keys: {list(analysis.summary.keys())}")
            else:
                logger.warning(f"No analysis result for user: {test_user}")
            
            # Test other methods
            top_users = self.user_analyzer.get_top_users(limit=5)
            logger.info(f"Top 5 users: {[user['username'] for user in top_users]}")
            
            health = self.user_analyzer.get_user_health(test_user)
            logger.info(f"User health: {health}")
            
            comparison = self.user_analyzer.get_user_channel_comparison(test_user)
            logger.info(f"Channel comparison: {comparison.get('channel_count', 0)} channels")
            
            logger.info("✅ User Analyzer tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ User Analyzer test failed: {e}")
            return False
    
    def test_template_rendering(self):
        """Test template rendering with V2 data."""
        
        logger.info("Testing Template Rendering...")
        
        try:
            # Get test data
            channels = self.channel_analyzer.get_available_channels()
            users = self.user_analyzer.get_available_users()
            
            if not channels or not users:
                logger.warning("Insufficient data for template testing")
                return True
            
            test_channel = channels[0]
            test_user = users[0]
            
            # Test channel analysis template
            logger.info(f"Testing channel template for: {test_channel}")
            channel_analysis = self.channel_analyzer.analyze(test_channel)
            
            if channel_analysis:
                template_data = {
                    'channel_name': test_channel,
                    'analysis': channel_analysis,
                    'statistics': channel_analysis.statistics,
                    'top_users': channel_analysis.top_users,
                    'time_patterns': channel_analysis.time_patterns,
                    'summary': channel_analysis.summary
                }
                
                rendered = self.template_engine.render_template(
                    'discord/channel_analysis.md.j2',
                    template_data
                )
                
                logger.info(f"Channel template rendered: {len(rendered)} characters")
                assert len(rendered) > 100, "Template should produce substantial output"
            
            # Test user analysis template
            logger.info(f"Testing user template for: {test_user}")
            user_analysis = self.user_analyzer.analyze(test_user)
            
            if user_analysis:
                template_data = {
                    'username': test_user,
                    'analysis': user_analysis,
                    'statistics': user_analysis.statistics,
                    'channel_activity': user_analysis.channel_activity,
                    'time_patterns': user_analysis.time_patterns,
                    'summary': user_analysis.summary
                }
                
                rendered = self.template_engine.render_template(
                    'discord/user_analysis.md.j2',
                    template_data
                )
                
                logger.info(f"User template rendered: {len(rendered)} characters")
                assert len(rendered) > 100, "Template should produce substantial output"
            
            # Test top users template
            top_users = self.user_analyzer.get_top_users(limit=5)
            if top_users:
                template_data = {
                    'limit': 5,
                    'days': 30,
                    'top_users': top_users,
                    'total_users': len(top_users)
                }
                
                rendered = self.template_engine.render_template(
                    'discord/top_users.md.j2',
                    template_data
                )
                
                logger.info(f"Top users template rendered: {len(rendered)} characters")
                assert len(rendered) > 50, "Template should produce some output"
            
            logger.info("✅ Template rendering tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Template rendering test failed: {e}")
            return False
    
    def test_threading_system(self):
        """Test threading capabilities."""
        
        logger.info("Testing Threading System...")
        
        try:
            # Test sync function execution in thread pool
            def sync_channel_analysis(channel_name):
                return self.channel_analyzer.analyze(channel_name)
            
            def sync_user_analysis(username):
                return self.user_analyzer.analyze(username)
            
            # Get test subjects
            channels = self.channel_analyzer.get_available_channels()
            users = self.user_analyzer.get_available_users()
            
            if not channels or not users:
                logger.warning("Insufficient data for threading tests")
                return True
            
            test_channel = channels[0]
            test_user = users[0]
            
            # Test sequential execution
            logger.info("Testing sequential execution...")
            start_time = time.time()
            
            channel_result = sync_channel_analysis(test_channel)
            user_result = sync_user_analysis(test_user)
            
            sequential_time = time.time() - start_time
            logger.info(f"Sequential execution took: {sequential_time:.2f}s")
            
            # Test parallel execution
            logger.info("Testing parallel execution...")
            start_time = time.time()
            
            future1 = self.thread_pool.submit(sync_channel_analysis, test_channel)
            future2 = self.thread_pool.submit(sync_user_analysis, test_user)
            
            channel_result_parallel = future1.result()
            user_result_parallel = future2.result()
            
            parallel_time = time.time() - start_time
            logger.info(f"Parallel execution took: {parallel_time:.2f}s")
            
            # Compare results
            speedup = sequential_time / parallel_time if parallel_time > 0 else 1
            logger.info(f"Speedup: {speedup:.2f}x")
            
            # Verify results are the same
            assert channel_result is not None, "Channel analysis should succeed"
            assert user_result is not None, "User analysis should succeed"
            assert channel_result_parallel is not None, "Parallel channel analysis should succeed"
            assert user_result_parallel is not None, "Parallel user analysis should succeed"
            
            logger.info("✅ Threading system tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Threading system test failed: {e}")
            return False
    
    def test_performance_metrics(self):
        """Test performance of V2 system."""
        
        logger.info("Testing Performance Metrics...")
        
        try:
            channels = self.channel_analyzer.get_available_channels()
            users = self.user_analyzer.get_available_users()
            
            if not channels or not users:
                logger.warning("Insufficient data for performance tests")
                return True
            
            # Test multiple operations
            test_operations = min(3, len(channels), len(users))
            
            logger.info(f"Running {test_operations} operations of each type...")
            
            # Channel analysis performance
            channel_times = []
            for i in range(test_operations):
                start_time = time.time()
                result = self.channel_analyzer.analyze(channels[i])
                exec_time = time.time() - start_time
                channel_times.append(exec_time)
                logger.info(f"Channel analysis {i+1}: {exec_time:.2f}s")
            
            # User analysis performance
            user_times = []
            for i in range(test_operations):
                start_time = time.time()
                result = self.user_analyzer.analyze(users[i])
                exec_time = time.time() - start_time
                user_times.append(exec_time)
                logger.info(f"User analysis {i+1}: {exec_time:.2f}s")
            
            # Calculate averages
            avg_channel_time = sum(channel_times) / len(channel_times)
            avg_user_time = sum(user_times) / len(user_times)
            
            logger.info(f"Average channel analysis time: {avg_channel_time:.2f}s")
            logger.info(f"Average user analysis time: {avg_user_time:.2f}s")
            
            # Performance assertions
            assert avg_channel_time < 5.0, "Channel analysis should be under 5 seconds"
            assert avg_user_time < 5.0, "User analysis should be under 5 seconds"
            
            logger.info("✅ Performance metrics tests passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Performance metrics test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests and report results."""
        
        logger.info("Starting V2 System Test Suite...")
        logger.info("=" * 50)
        
        tests = [
            ("Database Manager", self.test_database_manager),
            ("Channel Analyzer", self.test_channel_analyzer),
            ("User Analyzer", self.test_user_analyzer),
            ("Template Rendering", self.test_template_rendering),
            ("Threading System", self.test_threading_system),
            ("Performance Metrics", self.test_performance_metrics)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"\n🔍 Running {test_name} test...")
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    logger.info(f"✅ {test_name} test PASSED")
                else:
                    logger.error(f"❌ {test_name} test FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name} test CRASHED: {e}")
                results[test_name] = False
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("TEST SUMMARY:")
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"  {test_name}: {status}")
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All tests passed! V2 system is ready for migration.")
            return True
        else:
            logger.error(f"💥 {total - passed} tests failed. Please fix issues before migration.")
            return False
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.thread_pool.shutdown(wait=True)
            self.db_manager.close_connections()
            logger.info("Test cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


def main():
    """Run the test suite."""
    
    test_system = TestV2System()
    
    try:
        success = test_system.run_all_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Test suite crashed: {e}")
        return 1
    finally:
        test_system.cleanup()


if __name__ == "__main__":
    exit(main()) 