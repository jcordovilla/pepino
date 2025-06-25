#!/usr/bin/env python3
"""
Demo showing how to integrate templates into existing Discord commands.
This shows both the old way and the new template way side by side.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pepino.discord.commands.template_mixin import TemplateFormattingMixin
from pepino.discord.commands.message_formatter import DiscordMessageFormatter
from pepino.data.database.manager import DatabaseManager
from pepino.analysis.channel_analyzer import ChannelAnalyzer
from pepino.data.config import Settings


class ExampleAnalysisCommands(TemplateFormattingMixin):
    """
    Example showing how to integrate templates into your existing AnalysisCommands.
    
    This demonstrates the migration path:
    1. Inherit from TemplateFormattingMixin
    2. Keep existing code unchanged
    3. Gradually switch methods to use templates
    """
    
    def __init__(self):
        # Initialize like your existing AnalysisCommands
        self.message_formatter = DiscordMessageFormatter()
        
        # Call mixin init to add template functionality
        super().__init__()


async def demo_integration():
    """Demo showing template integration with existing commands"""
    print("🔧 Template Integration Demo")
    print("============================")
    
    db_manager = None
    try:
        # Setup like your existing commands
        db_manager = DatabaseManager()
        await asyncio.wait_for(db_manager.initialize(), timeout=10.0)
        
        settings = Settings()
        channel_analyzer = ChannelAnalyzer(db_manager, settings.base_filter)
        
        # Get sample channel
        channels = await asyncio.wait_for(
            channel_analyzer.get_available_channels(), 
            timeout=15.0
        )
        if not channels:
            print("❌ No channels found")
            return
            
        demo_channel = channels[0]
        print(f"📊 Testing with channel: #{demo_channel}")
        
        # Run analysis
        analysis_result = await asyncio.wait_for(
            channel_analyzer.analyze(
                channel_name=demo_channel,
                include_top_users=True,
                days_back=30,
                limit_users=5
            ),
            timeout=30.0
        )
        
        # Convert to dict for template compatibility
        if hasattr(analysis_result, 'model_dump'):
            analysis_dict = analysis_result.model_dump()
        else:
            analysis_dict = analysis_result.__dict__
        
        # Create command instance with template support
        commands = ExampleAnalysisCommands()
        
        print("\n" + "="*60)
        print("📝 TEMPLATE-BASED OUTPUT:")
        print("="*60)
        
        # Use template formatting (new way)
        commands.use_templates = True
        template_chunks = commands.format_channel_insights(
            analysis_dict, 
            include_chart=False
        )
        
        for chunk in template_chunks:
            print(chunk["content"])
        
        print("\n" + "="*60)
        print("🔍 HARDCODED OUTPUT (for comparison):")
        print("="*60)
        
        # Use original formatting (old way)
        commands.use_templates = False  
        hardcoded_chunks = commands.format_channel_insights(
            analysis_dict,
            include_chart=False
        )
        
        for chunk in hardcoded_chunks:
            print(chunk["content"])
        
        print("\n" + "="*60)
        print("✅ Integration Demo Complete!")
        print("="*60)
        
        print("\n💡 Integration Steps for Your Commands:")
        print("1. Add TemplateFormattingMixin to your AnalysisCommands class")
        print("2. Set self.use_templates = True to enable templates")
        print("3. Existing commands work unchanged!")
        print("4. Gradually migrate to template-based formatting")
        
    except Exception as e:
        print(f"❌ Demo error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if db_manager:
            try:
                await asyncio.wait_for(db_manager.close(), timeout=5.0)
                print("🔧 Database connections closed")
            except Exception as e:
                print(f"⚠️ Error closing database: {e}")


async def main():
    """Main demo function"""
    try:
        await demo_integration()
    except KeyboardInterrupt:
        print("\n🛑 Demo cancelled")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
    finally:
        print("\n🔚 Demo completed")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Exiting cleanly...")
    finally:
        print("🔚 All cleanup completed") 