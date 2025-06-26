#!/usr/bin/env python3
"""
Demo script showing template-based channel analysis formatting.

This demonstrates how to use Jinja2 templates to produce the exact same
output as the current hardcoded formatting, but with much more flexibility.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path so we can import pepino
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pepino.analysis.template_engine import AnalysisTemplateEngine
from pepino.discord.commands.template_message_formatter import TemplateMessageFormatter
from pepino.data.database.manager import DatabaseManager
from pepino.analysis.channel_analyzer import ChannelAnalyzer
from pepino.data.config import Settings


async def demo_template_formatting():
    """
    Demo showing template-based vs hardcoded formatting.
    """
    db_manager = None
    try:
        print("🔧 Setting up template demo...")
        
        # Initialize components with timeout
        db_manager = DatabaseManager()
        await asyncio.wait_for(db_manager.initialize(), timeout=10.0)
        
        settings = Settings()
        from pepino.analysis.data_facade import get_analysis_data_facade
        data_facade = get_analysis_data_facade(db_manager, settings.base_filter)
        channel_analyzer = ChannelAnalyzer(data_facade)
        
        # Initialize template formatter
        template_formatter = TemplateMessageFormatter()
        
        print("✅ Template system initialized")
        
        # Get list of available channels with timeout
        channels = await asyncio.wait_for(
            channel_analyzer.get_available_channels(), 
            timeout=15.0
        )
        if not channels:
            print("❌ No channels found in database")
            return
        
        # Use first channel for demo
        demo_channel = channels[0]
        print(f"📊 Analyzing channel: #{demo_channel}")
        
        # Run channel analysis with timeout
        analysis_result = await asyncio.wait_for(
            channel_analyzer.analyze(
                channel_name=demo_channel,
                include_top_users=True,
                days_back=30,
                limit_users=10
            ),
            timeout=30.0
        )
        
        # Convert Pydantic model to dict for template rendering
        if hasattr(analysis_result, 'model_dump'):
            analysis_dict = analysis_result.model_dump()
        else:
            analysis_dict = analysis_result.__dict__
        
        print("✅ Analysis completed")
        
        # Format using template
        print("\n" + "="*60)
        print("📝 TEMPLATE-BASED OUTPUT:")
        print("="*60)
        
        template_output = template_formatter.format_channel_insights_template(
            analysis_dict, 
            include_charts=False
        )
        
        print(template_output)
        
        print("\n" + "="*60)
        print("🎯 Template system successfully reproduced channel analysis!")
        print("="*60)
        
        # Show template benefits
        print("\n💡 Template Benefits:")
        print("• ✨ Same output as hardcoded version")
        print("• 🔧 Easy to modify without code changes")
        print("• 🎨 Consistent formatting across all reports")
        print("• 🔄 Reusable across different contexts")
        print("• 🧪 Testable and maintainable")
        
    except asyncio.TimeoutError:
        print("❌ Operation timed out - this prevents hanging!")
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # CRITICAL: Always cleanup database connections
        if db_manager:
            try:
                await asyncio.wait_for(db_manager.close(), timeout=5.0)
                print("🔧 Database connections closed")
            except Exception as e:
                print(f"⚠️ Error closing database: {e}")
                # Force close if needed
                if hasattr(db_manager, 'pool') and db_manager.pool:
                    try:
                        await db_manager.pool.close()
                    except:
                        pass


def create_sample_data_demo():
    """
    Demo with sample data to show template rendering without database.
    """
    print("\n🧪 Running template demo with sample data...")
    
    # Create sample analysis result that matches your ChannelAnalysisResponse structure
    sample_analysis = {
        "success": True,
        "plugin": "ChannelAnalyzer",
        "channel_info": {
            "channel_name": "development"
        },
        "statistics": {
            "total_messages": 15847,
            "unique_users": 45,
            "avg_message_length": 127.3,
            "first_message": "2024-01-15T10:30:00",
            "last_message": "2024-06-17T15:45:00",
            "active_days": 89,
            "bot_messages": 1247,
            "human_messages": 14600,
            "unique_human_users": 42
        },
        "top_users": [
            {
                "author_id": "123456789",
                "author_name": "alice_dev",
                "display_name": "Alice (Dev)",
                "message_count": 1829,
                "avg_message_length": 156.2
            },
            {
                "author_id": "987654321", 
                "author_name": "bob_lead",
                "display_name": "Bob (Team Lead)",
                "message_count": 1456,
                "avg_message_length": 203.7
            },
            {
                "author_id": "456789123",
                "author_name": "charlie_backend",
                "display_name": "Charlie",
                "message_count": 987,
                "avg_message_length": 89.4
            }
        ],
        "engagement_metrics": {
            "total_replies": 3247,
            "original_posts": 11353,
            "posts_with_reactions": 2891,
            "replies_per_post": 0.29,
            "reaction_rate": 19.8
        },
        "peak_activity": {
            "peak_hours": [
                {"hour": "14", "messages": 1247},
                {"hour": "15", "messages": 1156},
                {"hour": "10", "messages": 987}
            ],
            "peak_days": [
                {"day": "Tuesday", "messages": 2847},
                {"day": "Wednesday", "messages": 2456},
                {"day": "Thursday", "messages": 2123}
            ]
        },
        "recent_activity": [
            {"date": "2024-06-17", "messages": 89},
            {"date": "2024-06-16", "messages": 127}, 
            {"date": "2024-06-15", "messages": 156},
            {"date": "2024-06-14", "messages": 201},
            {"date": "2024-06-13", "messages": 178}
        ],
        "health_metrics": {
            "weekly_active": 28,
            "inactive_users": 14,
            "total_channel_members": 67,
            "lurkers": 25,
            "participation_rate": 62.7
        },
        "top_topics": [
            "api design and implementation",
            "database optimization strategies", 
            "frontend react components",
            "testing and code coverage",
            "deployment and devops",
            "performance monitoring",
            "code review process",
            "security best practices"
        ]
    }
    
    # Initialize template engine
    template_engine = AnalysisTemplateEngine()
    
    print("\n" + "="*60)
    print("📝 TEMPLATE OUTPUT (Sample Data):")
    print("="*60)
    
    # Render using template
    output = template_engine.render_channel_analysis(sample_analysis)
    print(output)
    
    print("\n" + "="*60)
    print("✅ Template successfully rendered with sample data!")
    print("="*60)


async def main():
    """Main demo function with proper cleanup"""
    print("🚀 Channel Analysis Template Demo")
    print("==================================")
    
    try:
        # First, show sample data demo (no database required)
        create_sample_data_demo()
        
        # Then try real database demo if available
        await demo_template_formatting()
        
    except KeyboardInterrupt:
        print("\n🛑 Demo cancelled by user")
    except Exception as e:
        print(f"\n⚠️  Demo error: {str(e)}")
    finally:
        print("\n🔚 Demo completed - all resources cleaned up")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Exiting cleanly...")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
    finally:
        # Ensure any remaining tasks are cleaned up
        try:
            # Cancel any remaining tasks
            pending = asyncio.all_tasks()
            for task in pending:
                task.cancel()
        except:
            pass 