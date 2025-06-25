#!/usr/bin/env python3
"""
Test Markdown Output Generator

This script generates actual markdown files you can open and view!
Perfect for testing the sync template system.
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional

# Simple data classes to match the analyzer response structure
@dataclass
class ChannelStatistics:
    total_messages: int
    unique_users: int
    avg_message_length: float
    active_days: int
    human_messages: int
    bot_messages: int
    first_message: str
    last_message: str
    unique_human_users: int

@dataclass
class TopUser:
    author_id: str
    author_name: str
    display_name: Optional[str]
    message_count: int
    avg_message_length: float

@dataclass
class EngagementMetrics:
    total_replies: int
    original_posts: int
    posts_with_reactions: int
    replies_per_post: float
    reaction_rate: float

@dataclass
class PeakHour:
    hour: str
    messages: int

@dataclass
class PeakActivity:
    peak_hours: List[PeakHour]

@dataclass
class RecentActivity:
    date: str
    messages: int

@dataclass
class ChannelAnalysisResponse:
    statistics: ChannelStatistics
    top_users: List[TopUser]
    engagement_metrics: EngagementMetrics
    peak_activity: PeakActivity
    recent_activity: List[RecentActivity]

def get_channel_data(channel_name: str) -> Optional[ChannelAnalysisResponse]:
    """Get channel data using simple sync queries"""
    
    db_path = "discord_messages.db"
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return None
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        # Basic statistics
        cursor = conn.execute("""
            SELECT 
                COUNT(*) as total_messages,
                COUNT(DISTINCT author_id) as unique_users,
                AVG(LENGTH(content)) as avg_message_length,
                MIN(timestamp) as first_message,
                MAX(timestamp) as last_message,
                COUNT(CASE WHEN author_is_bot = 1 THEN 1 END) as bot_messages,
                COUNT(CASE WHEN author_is_bot = 0 OR author_is_bot IS NULL THEN 1 END) as human_messages,
                COUNT(DISTINCT CASE WHEN author_is_bot = 0 OR author_is_bot IS NULL THEN author_id END) as unique_human_users,
                COUNT(DISTINCT DATE(timestamp)) as active_days
            FROM messages 
            WHERE channel_name = ?
            AND content IS NOT NULL
        """, (channel_name,))
        
        stats_row = cursor.fetchone()
        
        if not stats_row or stats_row['total_messages'] == 0:
            return None
        
        # Top users
        cursor = conn.execute("""
            SELECT 
                author_id,
                author_name,
                author_name as display_name,  -- Use author_name as display_name since display_name column doesn't exist
                COUNT(*) as message_count,
                AVG(LENGTH(content)) as avg_message_length
            FROM messages 
            WHERE channel_name = ?
            AND (author_is_bot = 0 OR author_is_bot IS NULL)
            GROUP BY author_id
            ORDER BY message_count DESC
            LIMIT 5
        """, (channel_name,))
        
        top_users_rows = cursor.fetchall()
        
        # Engagement metrics
        cursor = conn.execute("""
            SELECT 
                COUNT(CASE WHEN referenced_message_id IS NOT NULL THEN 1 END) as total_replies,
                COUNT(CASE WHEN referenced_message_id IS NULL THEN 1 END) as original_posts,
                COUNT(CASE WHEN has_reactions = 1 THEN 1 END) as posts_with_reactions
            FROM messages 
            WHERE channel_name = ?
            AND (author_is_bot = 0 OR author_is_bot IS NULL)
        """, (channel_name,))
        
        engagement_row = cursor.fetchone()
        
        # Peak hours
        cursor = conn.execute("""
            SELECT 
                strftime('%H', timestamp) as hour,
                COUNT(*) as messages
            FROM messages 
            WHERE channel_name = ?
            AND timestamp IS NOT NULL
            GROUP BY strftime('%H', timestamp)
            ORDER BY messages DESC
            LIMIT 3
        """, (channel_name,))
        
        peak_hours_rows = cursor.fetchall()
        
        # Recent activity
        cursor = conn.execute("""
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as messages
            FROM messages 
            WHERE channel_name = ?
            AND timestamp IS NOT NULL
            AND DATE(timestamp) >= DATE('now', '-7 days')
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            LIMIT 7
        """, (channel_name,))
        
        recent_activity_rows = cursor.fetchall()
        
        conn.close()
        
        # Build response object
        statistics = ChannelStatistics(
            total_messages=stats_row['total_messages'],
            unique_users=stats_row['unique_users'],
            avg_message_length=stats_row['avg_message_length'] or 0,
            active_days=stats_row['active_days'],
            human_messages=stats_row['human_messages'],
            bot_messages=stats_row['bot_messages'],
            first_message=stats_row['first_message'],
            last_message=stats_row['last_message'],
            unique_human_users=stats_row['unique_human_users']
        )
        
        top_users = [
            TopUser(
                author_id=row['author_id'],
                author_name=row['author_name'],
                display_name=row['display_name'],
                message_count=row['message_count'],
                avg_message_length=row['avg_message_length'] or 0
            )
            for row in top_users_rows
        ]
        
        total_replies = engagement_row['total_replies']
        original_posts = engagement_row['original_posts']
        posts_with_reactions = engagement_row['posts_with_reactions']
        
        engagement_metrics = EngagementMetrics(
            total_replies=total_replies,
            original_posts=original_posts,
            posts_with_reactions=posts_with_reactions,
            replies_per_post=total_replies / original_posts if original_posts > 0 else 0,
            reaction_rate=(posts_with_reactions / stats_row['human_messages'] * 100) if stats_row['human_messages'] > 0 else 0
        )
        
        peak_activity = PeakActivity(
            peak_hours=[
                PeakHour(hour=row['hour'], messages=row['messages'])
                for row in peak_hours_rows
            ]
        )
        
        recent_activity = [
            RecentActivity(date=row['date'], messages=row['messages'])
            for row in recent_activity_rows
        ]
        
        return ChannelAnalysisResponse(
            statistics=statistics,
            top_users=top_users,
            engagement_metrics=engagement_metrics,
            peak_activity=peak_activity,
            recent_activity=recent_activity
        )
        
    except Exception as e:
        print(f"❌ Error getting channel data: {e}")
        return None

def render_template_simple(template_content: str, **data) -> str:
    """Simple template rendering without Jinja2 dependency"""
    
    # This is a very basic template renderer for testing
    # In real usage, you'd use Jinja2
    
    result = template_content
    
    # Replace simple variables
    for key, value in data.items():
        result = result.replace(f"{{{{ {key} }}}}", str(value))
    
    # Handle channel_analysis.statistics fields
    if 'channel_analysis' in data and data['channel_analysis']:
        analysis = data['channel_analysis']
        if hasattr(analysis, 'statistics'):
            stats = analysis.statistics
            result = result.replace("{{ channel_analysis.statistics.total_messages }}", str(stats.total_messages))
            result = result.replace("{{ channel_analysis.statistics.unique_users }}", str(stats.unique_users))
            result = result.replace("{{ channel_analysis.statistics.avg_message_length|round(1) }}", f"{stats.avg_message_length:.1f}")
            result = result.replace("{{ channel_analysis.statistics.active_days }}", str(stats.active_days))
            result = result.replace("{{ channel_analysis.statistics.human_messages }}", str(stats.human_messages))
            result = result.replace("{{ channel_analysis.statistics.bot_messages }}", str(stats.bot_messages))
        
        # Handle top users (simplified)
        if hasattr(analysis, 'top_users') and analysis.top_users:
            users_section = "\n"
            for i, user in enumerate(analysis.top_users[:5], 1):
                users_section += f"{i}. **{user.display_name or user.author_name}**\n"
                users_section += f"   - Messages: {user.message_count}\n"
                users_section += f"   - Avg Length: {user.avg_message_length:.1f} chars\n"
            
            # Replace the Jinja2 loop with our generated content
            start_marker = "{% if channel_analysis and channel_analysis.top_users %}"
            end_marker = "{% else %}"
            
            start_idx = result.find(start_marker)
            end_idx = result.find(end_marker)
            
            if start_idx != -1 and end_idx != -1:
                # Find the actual loop content
                loop_start = result.find("{% for user in channel_analysis.top_users[:5] %}", start_idx)
                loop_end = result.find("{% endfor %}", loop_start) + len("{% endfor %}")
                
                if loop_start != -1 and loop_end != -1:
                    result = result[:loop_start] + users_section + result[loop_end:]
        
        # Handle engagement metrics
        if hasattr(analysis, 'engagement_metrics'):
            metrics = analysis.engagement_metrics
            result = result.replace("{{ channel_analysis.engagement_metrics.total_replies }}", str(metrics.total_replies))
            result = result.replace("{{ channel_analysis.engagement_metrics.original_posts }}", str(metrics.original_posts))
            result = result.replace("{{ channel_analysis.engagement_metrics.posts_with_reactions }}", str(metrics.posts_with_reactions))
            result = result.replace("{{ channel_analysis.engagement_metrics.replies_per_post|round(2) }}", f"{metrics.replies_per_post:.2f}")
            result = result.replace("{{ channel_analysis.engagement_metrics.reaction_rate|round(1) }}", f"{metrics.reaction_rate:.1f}")
        
        # Handle peak activity
        if hasattr(analysis, 'peak_activity') and analysis.peak_activity.peak_hours:
            hours_section = "\n"
            for hour in analysis.peak_activity.peak_hours:
                hours_section += f"- {hour.hour}:00 - {hour.messages} messages\n"
            
            # Replace peak hours section
            peak_start = result.find("{% for hour in channel_analysis.peak_activity.peak_hours %}")
            peak_end = result.find("{% endfor %}", peak_start) + len("{% endfor %}")
            
            if peak_start != -1 and peak_end != -1:
                result = result[:peak_start] + hours_section + result[peak_end:]
        
        # Handle recent activity
        if hasattr(analysis, 'recent_activity') and analysis.recent_activity:
            activity_section = "\n"
            for activity in analysis.recent_activity:
                activity_section += f"- **{activity.date}:** {activity.messages} messages\n"
            
            # Replace recent activity section
            activity_start = result.find("{% for activity in channel_analysis.recent_activity %}")
            activity_end = result.find("{% endfor %}", activity_start) + len("{% endfor %}")
            
            if activity_start != -1 and activity_end != -1:
                result = result[:activity_start] + activity_section + result[activity_end:]
    
    # Clean up remaining Jinja2 syntax
    import re
    result = re.sub(r'\{%.*?%\}', '', result, flags=re.DOTALL)
    result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)  # Clean up multiple newlines
    
    return result

def test_markdown_generation():
    """Test markdown generation and save to files"""
    
    print("🧪 Testing Markdown Output Generation")
    print("=" * 50)
    
    # Get available channels
    db_path = "discord_messages.db"
    if not Path(db_path).exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("""
        SELECT DISTINCT channel_name, COUNT(*) as messages
        FROM messages 
        WHERE timestamp >= datetime('now', '-30 days')
        GROUP BY channel_name
        ORDER BY messages DESC
        LIMIT 3
    """)
    
    channels = cursor.fetchall()
    conn.close()
    
    if not channels:
        print("❌ No channels found")
        return
    
    print(f"📋 Found {len(channels)} active channels, testing top 3...")
    
    # Read template
    template_path = Path("templates/outputs/discord/simple_sync_test.md.j2")
    if not template_path.exists():
        print(f"❌ Template not found: {template_path}")
        return
    
    template_content = template_path.read_text()
    
    # Create output directory
    output_dir = Path("temp/markdown_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate reports for each channel
    for i, (channel_name, message_count) in enumerate(channels, 1):
        print(f"\n{i}. Testing #{channel_name} ({message_count} messages)...")
        
        # Get channel data
        analysis_data = get_channel_data(channel_name)
        
        if not analysis_data:
            print(f"   ❌ No data for #{channel_name}")
            continue
        
        # Render template
        try:
            markdown_output = render_template_simple(
                template_content,
                channel_name=channel_name,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                channel_analysis=analysis_data
            )
            
            # Save to file
            output_file = output_dir / f"{channel_name.replace('#', '').replace('-', '_')}_report.md"
            output_file.write_text(markdown_output)
            
            print(f"   ✅ Generated: {output_file}")
            print(f"   📄 Size: {len(markdown_output)} characters")
            
        except Exception as e:
            print(f"   ❌ Failed to generate: {e}")
    
    print(f"\n🎉 Test completed! Check files in: {output_dir}")
    print(f"\n💡 To view the markdown:")
    print(f"   1. Open the .md files in VS Code or any markdown viewer")
    print(f"   2. Or use: cat {output_dir}/*.md")
    print(f"   3. Or preview in GitHub/GitLab")

if __name__ == "__main__":
    test_markdown_generation() 