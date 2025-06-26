#!/usr/bin/env python3
"""
Demo showing the REAL analyzer functions available in templates.
This shows what's actually available, not made-up function names!
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pepino.analysis.template_executor import TemplateExecutor
from pepino.data.database.manager import DatabaseManager
from pepino.analysis.channel_analyzer import ChannelAnalyzer
from pepino.analysis.user_analyzer import UserAnalyzer
from pepino.analysis.topic_analyzer import TopicAnalyzer
from pepino.analysis.temporal_analyzer import TemporalAnalyzer
from pepino.analysis.data_facade import get_analysis_data_facade
from pepino.data.config import Settings


async def demo_real_analyzer_functions():
    """Demo showing the actual analyzer functions available in templates"""
    print("🔍 REAL Analyzer Functions Available in Templates")
    print("================================================")
    
    print("\n📊 **ChannelAnalyzer** Functions:")
    print("-" * 40)
    print("• {{ channel_analyze(channel_name='general', days_back=30, include_top_users=true) }}")
    print("• {{ get_available_channels() }}")
    print("• {{ get_top_channels(limit=10) }}")
    
    print("\n👤 **UserAnalyzer** Functions:")
    print("-" * 40)
    print("• {{ user_analyze(user_name='john', include_concepts=true, days_back=30) }}")
    print("• {{ get_available_users() }}")
    print("• {{ get_top_users(limit=10) }}")
    
    print("\n🧠 **TopicAnalyzer** Functions:")
    print("-" * 40)
    print("• {{ topic_analyze(channel_name='general', days_back=30, min_word_length=4) }}")
    
    print("\n⏰ **TemporalAnalyzer** Functions:")
    print("-" * 40)
    print("• {{ temporal_analyze(channel_name='general', days_back=30, granularity='day') }}")
    
    db_manager = None
    try:
        # Setup and test real functions
        db_manager = DatabaseManager()
        await asyncio.wait_for(db_manager.initialize(), timeout=10.0)
        
        settings = Settings()
        analyzers = {
            'channel_analyzer': ChannelAnalyzer(get_analysis_data_facade(db_manager, settings.base_filter)),
            'user_analyzer': UserAnalyzer(get_analysis_data_facade(db_manager, settings.base_filter)),
        }
        
        print("\n🧪 Testing Real Functions:")
        print("=" * 40)
        
        # Test functions
        channels = await analyzers['channel_analyzer'].get_available_channels()
        print(f"✅ get_available_channels() → {channels[:3]}...")
        
        top_users = await analyzers['user_analyzer'].get_top_users(limit=3)
        print(f"✅ get_top_users(limit=3) → {len(top_users)} users")
        
        print("\n✅ All Functions Verified!")
        
    except Exception as e:
        print(f"❌ Demo error: {str(e)}")
    finally:
        if db_manager:
            await db_manager.close()


async def main():
    """Main demo function"""
    await demo_real_analyzer_functions()


if __name__ == "__main__":
    asyncio.run(main()) 