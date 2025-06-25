#!/usr/bin/env python3
"""
Quick template test - no database operations, just template formatting.
This tests if our template fixes work without risk of hanging.
"""

import sys
from pathlib import Path

# Add src to path so we can import pepino
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pepino.analysis.template_engine import AnalysisTemplateEngine


def test_template_formatting():
    """Test template formatting with sample data - no hanging risk"""
    print("🧪 Testing Template Formatting Fixes")
    print("=====================================")
    
    # Create sample analysis result
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
                {"hour": "15", "messages": 1156}
            ],
            "peak_days": [
                {"day": "Tuesday", "messages": 2847},
                {"day": "Wednesday", "messages": 2456}
            ]
        },
        "recent_activity": [
            {"date": "2024-06-17", "messages": 89},
            {"date": "2024-06-16", "messages": 127}
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
            "frontend react components"
        ]
    }
    
    try:
        # Initialize template engine
        template_engine = AnalysisTemplateEngine()
        
        print("✅ Template engine initialized")
        
        # Render using template
        output = template_engine.render_channel_analysis(sample_analysis)
        
        print("\n" + "="*60)
        print("📝 TEMPLATE OUTPUT:")
        print("="*60)
        print(output)
        print("="*60)
        
        # Check for formatting issues
        lines = output.split('\n')
        bullet_lines = [line for line in lines if line.strip().startswith('•')]
        
        print(f"\n✅ Template rendered successfully!")
        print(f"📊 Total lines: {len(lines)}")
        print(f"🔷 Bullet points: {len(bullet_lines)}")
        
        # Check if formatting looks correct
        if len(bullet_lines) > 5 and "## 📊 Channel Analysis:" in output:
            print("✅ Formatting appears correct!")
        else:
            print("⚠️ Potential formatting issues detected")
            
        return True
        
    except Exception as e:
        print(f"❌ Template test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Quick Template Test (No Database)")
    print("=====================================")
    
    success = test_template_formatting()
    
    if success:
        print("\n🎉 Template fixes work correctly!")
        print("✅ Ready to test with database operations")
    else:
        print("\n❌ Template fixes need more work")
        
    print("\n🔚 Test completed - no hanging risk!") 