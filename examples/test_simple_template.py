#!/usr/bin/env python3
"""
Simple test for basic template functionality.
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
from pepino.analysis.data_facade import get_analysis_data_facade
from pepino.data.config import Settings


async def test_simple_template():
    """Test simple template with basic functions"""
    print("🧪 Testing Simple Template")
    print("===========================")
    
    db_manager = None
    try:
        # Setup
        db_manager = DatabaseManager()
        await asyncio.wait_for(db_manager.initialize(), timeout=10.0)
        
        settings = Settings()
        analyzers = {
            'channel_analyzer': ChannelAnalyzer(get_analysis_data_facade(db_manager, settings.base_filter)),
            'user_analyzer': UserAnalyzer(get_analysis_data_facade(db_manager, settings.base_filter)),
        }
        
        # Create template executor
        template_executor = TemplateExecutor(analyzers)
        
        print("\n" + "="*50)
        print("📝 SIMPLE TEMPLATE OUTPUT:")
        print("="*50)
        
        # Execute the simple test template
        result = await template_executor.execute_template(
            'discord/simple_test.md.j2',
            channel_name='test-channel',
            days_back=30
        )
        
        print(result)
        
        print("\n" + "="*50)
        print("✅ Simple Template Test Complete!")
        print("="*50)
        
    except Exception as e:
        print(f"❌ Test error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if db_manager:
            await db_manager.close()
            print("🔧 Database closed")


if __name__ == "__main__":
    asyncio.run(test_simple_template()) 