#!/usr/bin/env python3
"""
Demo: Sync Analyzers + Templates = MUCH Simpler!

This shows how sync analyzers eliminate all the async complexity
while making templates work perfectly.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pepino.data.database.sync_manager import SyncDatabaseManager
from src.pepino.analysis.sync_channel_analyzer import SyncChannelAnalyzer
from src.pepino.analysis.sync_user_analyzer import SyncUserAnalyzer
from src.pepino.analysis.sync_template_executor import SyncTemplateExecutor, SimpleTemplateExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_sync_analyzers():
    """Demo: Direct sync analyzer usage - super clean!"""
    
    print("🔧 Demo: Sync Analyzers - No Async Complexity!")
    print("=" * 60)
    
    # Initialize database - SYNC, no aiosqlite needed!
    db_manager = SyncDatabaseManager("discord_messages.db")
    
    # Create sync analyzers - simple!
    channel_analyzer = SyncChannelAnalyzer(db_manager)
    user_analyzer = SyncUserAnalyzer(db_manager)
    
    print("\n📋 Available channels:")
    channels = channel_analyzer.get_available_channels()
    for i, channel in enumerate(channels[:5], 1):
        print(f"  {i}. #{channel}")
    
    if channels:
        # Analyze first channel - SYNC, no await needed!
        first_channel = channels[0]
        print(f"\n🔍 Analyzing #{first_channel} (SYNC - so simple!)...")
        
        result = channel_analyzer.analyze(
            channel_name=first_channel,
            include_top_users=True,
            limit_users=3
        )
        
        if hasattr(result, 'statistics'):
            stats = result.statistics
            print(f"   💬 Messages: {stats.total_messages}")
            print(f"   👥 Users: {stats.unique_users}")
            print(f"   📅 Active days: {stats.active_days}")
            
            if result.top_users:
                print(f"   🏆 Top users:")
                for user in result.top_users[:3]:
                    print(f"     - {user.author_name}: {user.message_count} messages")
        else:
            print(f"   ❌ {result.error}")
    
    print("\n👥 Top users overall:")
    top_users = user_analyzer.get_top_users(limit=3)
    for i, user in enumerate(top_users, 1):
        print(f"  {i}. {user['display_name']}: {user['message_count']} messages")


def demo_sync_template_executor():
    """Demo: Sync Template Executor - Pre-computes everything!"""
    
    print("\n\n🎨 Demo: Sync Template Executor - Pre-Computes Everything!")
    print("=" * 60)
    
    # Initialize - SYNC, so simple!
    db_manager = SyncDatabaseManager("discord_messages.db")
    executor = SyncTemplateExecutor(db_manager)
    
    # Get available channels
    channel_analyzer = SyncChannelAnalyzer(db_manager)
    channels = channel_analyzer.get_available_channels()
    
    if channels:
        first_channel = channels[0]
        print(f"\n📝 Rendering template for #{first_channel}...")
        
        # Execute template - SYNC pre-computes everything!
        try:
            result = executor.execute_template(
                "outputs/discord/channel_analysis.md.j2",
                channel_name=first_channel,
                include_top_users=True,
                limit_users=5
            )
            
            print("\n✅ Template Result (first 500 chars):")
            print("-" * 40)
            print(result[:500])
            if len(result) > 500:
                print("... (truncated)")
            print("-" * 40)
            
        except Exception as e:
            print(f"❌ Template failed: {e}")
    else:
        print("No channels found!")


def demo_simple_template_executor():
    """Demo: Simple Template Executor - Direct data passing!"""
    
    print("\n\n🎯 Demo: Simple Template Executor - Direct Data!")
    print("=" * 60)
    
    # Get data manually - SYNC!
    db_manager = SyncDatabaseManager("discord_messages.db")
    channel_analyzer = SyncChannelAnalyzer(db_manager)
    
    channels = channel_analyzer.get_available_channels()
    
    if channels:
        first_channel = channels[0]
        print(f"\n1. Getting data for #{first_channel}...")
        
        # Get analysis data - SYNC!
        analysis_result = channel_analyzer.analyze(
            channel_name=first_channel,
            include_top_users=True
        )
        
        print("2. Rendering template with data...")
        
        # Render template with data - dead simple!
        executor = SimpleTemplateExecutor()
        
        try:
            result = executor.render(
                "outputs/discord/channel_analysis.md.j2",
                channel_analysis=analysis_result,
                channel_name=first_channel
            )
            
            print("\n✅ Simple Template Result (first 300 chars):")
            print("-" * 40)
            print(result[:300])
            if len(result) > 300:
                print("... (truncated)")
            print("-" * 40)
            
        except Exception as e:
            print(f"❌ Template failed: {e}")


def main():
    """Run all demos"""
    try:
        demo_sync_analyzers()
        demo_sync_template_executor()
        demo_simple_template_executor()
        
        print("\n🎉 All demos completed!")
        print("\n💡 Key Benefits of Sync Approach:")
        print("   ✅ No async/await complexity")
        print("   ✅ Templates work normally")
        print("   ✅ Easy to debug")
        print("   ✅ Clean separation of data & presentation")
        print("   ✅ Pre-computed data = predictable templates")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)


if __name__ == "__main__":
    main() 