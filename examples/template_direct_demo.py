#!/usr/bin/env python3
"""
Demo showing how templates call analyzer functions directly.
This is the new clean approach - no more formatters!
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pepino.analysis.template_executor import TemplateExecutor
from pepino.data.database.manager import DatabaseManager
from pepino.analysis.channel_analyzer import ChannelAnalyzer
from pepino.analysis.topic_analyzer import TopicAnalyzer
from pepino.data.config import Settings


async def demo_direct_template_calls():
    """Demo showing templates calling analyzer functions directly"""
    print("🎨 Direct Template Function Calls Demo")
    print("=====================================")
    
    db_manager = None
    try:
        # Setup analyzers
        db_manager = DatabaseManager()
        await asyncio.wait_for(db_manager.initialize(), timeout=10.0)
        
        from pepino.analysis.data_facade import get_analysis_data_facade
        
        with get_analysis_data_facade() as facade:
            analyzers = {
                'channel_analyzer': ChannelAnalyzer(facade),
                'topic_analyzer': TopicAnalyzer(facade)
            }
        
        # Create template executor
        template_executor = TemplateExecutor(analyzers)
        
        # Show template content
        print("\n📝 Template Content (channel_analysis_direct.md.j2):")
        print("=" * 60)
        print("""
{# Channel Analysis Template - Calls analyzer functions directly #}
{%- set channel_data = analyze_channel(channel_name, days_back=days_back|default(30)) -%}
{%- set topics_data = analyze_topics(channel_name=channel_name) -%}

## 📊 Channel Analysis: #{{ channel_name }}

**📊 Statistics:**
• Total Messages: {{ channel_data.statistics.total_messages | format_number }}
• Unique Users: {{ channel_data.statistics.unique_users | format_number }}

**🧠 Top Topics:**
{% for topic in topics_data.topics[:5] -%}
{{ loop.index }}. {{ topic }}
{% endfor -%}
        """)
        
        print("\n🔧 Template Function Calls:")
        print("=" * 60)
        print("• {{ analyze_channel(channel_name, days_back=30) }} → Calls ChannelAnalyzer.analyze()")
        print("• {{ analyze_topics(channel_name=channel_name) }} → Calls TopicAnalyzer.analyze()")
        print("• {{ analyze_user(username, include_patterns=true) }} → Calls UserAnalyzer.analyze()")
        print("• {{ get_top_users(limit=10) }} → Helper function")
        
        print("\n📊 Executing Template with Real Data:")
        print("=" * 60)
        
        # Execute template with direct function calls
        result = await template_executor.execute_template(
            'discord/channel_analysis_direct.md.j2',
            channel_name='🏘old-general-chat',
            days_back=30
        )
        
        print(result)
        
        print("\n" + "="*60)
        print("✅ Benefits of Direct Template Calls:")
        print("="*60)
        print("✅ No formatters needed - templates do everything")
        print("✅ Templates call analyzer functions directly")
        print("✅ Clean separation: Discord commands → Templates → Analyzers")
        print("✅ Easy to add new analysis combinations")
        print("✅ Templates can combine multiple analyzers")
        print("✅ Dynamic analysis based on template parameters")
        
    except Exception as e:
        print(f"❌ Demo error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if db_manager:
            try:
                await asyncio.wait_for(db_manager.close(), timeout=5.0)
                print("\n🔧 Database connections closed")
            except Exception as e:
                print(f"⚠️ Error closing database: {e}")


async def demo_template_composition():
    """Demo showing how templates can compose multiple analyzers"""
    print("\n\n🎯 Template Composition Demo")
    print("============================")
    
    print("📝 Server Health Template Example:")
    print("-" * 40)
    print("""
{# Server Health - Combines multiple analyzers #}
{%- set channels = get_channels() -%}
{%- set trends = analyze_activity(days_back=30) -%}
{%- set top_users = get_top_users(limit=5) -%}

## 🏥 Server Health Report

**📊 Overview:**
• Total Channels: {{ channels | length }}
• Total Messages (30d): {{ trends.total_messages | format_number }}
• Active Users: {{ top_users | length }}

**📈 Channel Activity:**
{% for channel in channels[:3] -%}
{%- set channel_data = analyze_channel(channel, days_back=7) -%}
• #{{ channel }}: {{ channel_data.statistics.total_messages }} messages
{% endfor -%}

**👥 Top Contributors:**
{% for user in top_users -%}
• {{ user.display_name }}: {{ user.message_count }} messages
{% endfor -%}
    """)
    
    print("\n🔧 Template Functions Available:")
    print("-" * 40)
    print("• analyze_channel(name, days_back=30, include_top_users=true)")
    print("• analyze_user(username, include_patterns=false)")
    print("• analyze_topics(channel_name=null, days_back=30)")
    print("• analyze_activity(days_back=30)")
    print("• get_top_users(channel_name=null, limit=10)")
    print("• get_channels()")
    print("• format_timespan(days)")


async def main():
    """Main demo function"""
    try:
        await demo_direct_template_calls()
        await demo_template_composition()
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