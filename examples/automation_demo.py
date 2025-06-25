#!/usr/bin/env python3
"""
End-to-End Automation Demo Script

This script demonstrates how to use the new CLI listing commands to build
comprehensive automation workflows for Discord server analysis.
"""

import subprocess
import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Any

def run_pepino_command(command: List[str]) -> Dict[str, Any]:
    """Run a pepino CLI command and return parsed JSON output."""
    try:
        result = subprocess.run(
            ['poetry', 'run', 'python', '-m', 'pepino'] + command,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Try to parse JSON output
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            # If not JSON, return raw output
            return {'output': result.stdout, 'success': True}
            
    except subprocess.CalledProcessError as e:
        return {
            'success': False,
            'error': e.stderr,
            'output': e.stdout
        }

def get_channel_list() -> List[str]:
    """Get list of all available channels."""
    print("🔍 Fetching channel list...")
    
    result = run_pepino_command(['list', 'channels', '--format', 'json'])
    
    if result.get('success', True) and 'channels' in result:
        channels = [ch['name'] for ch in result['channels']]
        print(f"✅ Found {len(channels)} channels")
        return channels
    else:
        print(f"❌ Failed to get channels: {result.get('error', 'Unknown error')}")
        return []

def get_user_list() -> List[str]:
    """Get list of all available users."""
    print("🔍 Fetching user list...")
    
    result = run_pepino_command(['list', 'users', '--format', 'json'])
    
    if result.get('success', True) and 'users' in result:
        users = [user['name'] for user in result['users']]
        print(f"✅ Found {len(users)} users")
        return users
    else:
        print(f"❌ Failed to get users: {result.get('error', 'Unknown error')}")
        return []

def get_database_stats() -> Dict[str, Any]:
    """Get database statistics."""
    print("📊 Fetching database statistics...")
    
    result = run_pepino_command(['list', 'stats', '--format', 'json'])
    
    if result.get('success', True) and 'stats' in result:
        stats = result['stats']
        print(f"✅ Database: {stats['total_messages']:,} messages, {stats['total_channels']} channels, {stats['total_users']} users")
        return stats
    else:
        print(f"❌ Failed to get stats: {result.get('error', 'Unknown error')}")
        return {}

def analyze_channel(channel: str, output_dir: Path) -> Dict[str, Any]:
    """Analyze a specific channel and save results."""
    output_file = output_dir / f"channel_{channel.replace('#', '').replace(' ', '_')}.json"
    
    print(f"  📊 Analyzing channel: {channel}")
    
    result = run_pepino_command([
        'analyze', 'channels',
        '--channel', channel,
        '--format', 'json',
        '--output', str(output_file)
    ])
    
    if result.get('success', True):
        print(f"    ✅ Saved to {output_file}")
        return {'channel': channel, 'output_file': str(output_file), 'success': True}
    else:
        print(f"    ❌ Failed: {result.get('error', 'Unknown error')}")
        return {'channel': channel, 'success': False, 'error': result.get('error')}

def analyze_user(user: str, output_dir: Path) -> Dict[str, Any]:
    """Analyze a specific user and save results."""
    output_file = output_dir / f"user_{user.replace('@', '').replace(' ', '_')}.json"
    
    print(f"  👤 Analyzing user: {user}")
    
    result = run_pepino_command([
        'analyze', 'users',
        '--user', user,
        '--format', 'json',
        '--output', str(output_file)
    ])
    
    if result.get('success', True):
        print(f"    ✅ Saved to {output_file}")
        return {'user': user, 'output_file': str(output_file), 'success': True}
    else:
        print(f"    ❌ Failed: {result.get('error', 'Unknown error')}")
        return {'user': user, 'success': False, 'error': result.get('error')}

def comprehensive_server_analysis():
    """Perform comprehensive analysis of entire Discord server."""
    print("🚀 Starting Comprehensive Discord Server Analysis")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    # Get database overview
    stats = get_database_stats()
    if not stats:
        print("❌ Cannot proceed without database stats")
        return
    
    print(f"\n📊 Server Overview:")
    print(f"   Total Messages: {stats['total_messages']:,}")
    print(f"   Total Channels: {stats['total_channels']}")
    print(f"   Total Users: {stats['total_users']}")
    
    # Get all channels and users
    channels = get_channel_list()
    users = get_user_list()
    
    if not channels:
        print("❌ No channels found, skipping channel analysis")
    if not users:
        print("❌ No users found, skipping user analysis")
    
    # Analysis tracking
    channel_results = []
    user_results = []
    
    # Analyze all channels
    if channels:
        print(f"\n📺 Analyzing {len(channels)} channels...")
        for i, channel in enumerate(channels, 1):
            print(f"[{i}/{len(channels)}]", end=" ")
            result = analyze_channel(channel, output_dir)
            channel_results.append(result)
    
    # Analyze top users (limit to first 10 for demo)
    if users:
        top_users = users[:10]  # Limit for demo
        print(f"\n👥 Analyzing top {len(top_users)} users...")
        for i, user in enumerate(top_users, 1):
            print(f"[{i}/{len(top_users)}]", end=" ")
            result = analyze_user(user, output_dir)
            user_results.append(result)
    
    # Generate summary report
    summary = {
        'server_stats': stats,
        'analysis_summary': {
            'channels_analyzed': len([r for r in channel_results if r['success']]),
            'channels_failed': len([r for r in channel_results if not r['success']]),
            'users_analyzed': len([r for r in user_results if r['success']]),
            'users_failed': len([r for r in user_results if not r['success']]),
        },
        'channel_results': channel_results,
        'user_results': user_results,
    }
    
    # Save summary
    summary_file = output_dir / "analysis_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Analysis Complete!")
    print(f"   📁 Results saved to: {output_dir}")
    print(f"   📊 Summary: {summary_file}")
    print(f"   ✅ Channels: {summary['analysis_summary']['channels_analyzed']}")
    print(f"   ✅ Users: {summary['analysis_summary']['users_analyzed']}")
    
    if summary['analysis_summary']['channels_failed'] > 0:
        print(f"   ❌ Failed Channels: {summary['analysis_summary']['channels_failed']}")
    if summary['analysis_summary']['users_failed'] > 0:
        print(f"   ❌ Failed Users: {summary['analysis_summary']['users_failed']}")

def targeted_automation_example():
    """Example of targeted automation based on channel activity."""
    print("\n🎯 Targeted Automation Example")
    print("=" * 40)
    
    # Get channels with metadata
    print("🔍 Getting channels with statistics...")
    result = run_pepino_command(['list', 'channels', '--format', 'json'])
    
    if not (result.get('success', True) and 'channels' in result):
        print("❌ Cannot get channel data")
        return
    
    channels = result['channels']
    
    # Filter for high-activity channels (>100 messages)
    high_activity = [ch for ch in channels if ch['message_count'] > 100]
    
    print(f"📊 Found {len(high_activity)} high-activity channels (>100 messages)")
    
    # Analyze only high-activity channels
    output_dir = Path("targeted_analysis")
    output_dir.mkdir(exist_ok=True)
    
    for channel_data in high_activity[:5]:  # Limit to top 5
        channel_name = channel_data['name']
        message_count = channel_data['message_count']
        
        print(f"🔍 Analyzing high-activity channel: {channel_name} ({message_count:,} messages)")
        
        # Channel analysis
        analyze_channel(channel_name, output_dir)
        
        # Topic analysis for this channel
        print(f"  🏷️ Topic analysis for {channel_name}")
        run_pepino_command([
            'analyze', 'topics',
            '--channel', channel_name,
            '--format', 'json',
            '--output', str(output_dir / f"topics_{channel_name.replace('#', '').replace(' ', '_')}.json")
        ])

def automation_script_generator():
    """Generate bash scripts for common automation tasks."""
    print("\n🤖 Generating Automation Scripts")
    print("=" * 40)
    
    scripts_dir = Path("automation_scripts")
    scripts_dir.mkdir(exist_ok=True)
    
    # Script 1: Analyze all channels
    channel_script = scripts_dir / "analyze_all_channels.sh"
    with open(channel_script, 'w') as f:
        f.write("""#!/bin/bash
# Automated channel analysis script
# Generated by pepino automation demo

echo "🚀 Starting automated channel analysis..."

# Create output directory
mkdir -p channel_analysis_$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="channel_analysis_$(date +%Y%m%d_%H%M%S)"

# Get channel list and analyze each
poetry run python -m pepino list channels --format json --output channels_list.json

# Parse JSON and iterate (requires jq)
if command -v jq &> /dev/null; then
    jq -r '.channels[].name' channels_list.json | while read channel; do
        echo "📊 Analyzing channel: $channel"
        poetry run python -m pepino analyze channels -c "$channel" --format json --output "$OUTPUT_DIR/channel_${channel//[^a-zA-Z0-9]/_}.json"
    done
else
    echo "❌ jq not installed. Install with: brew install jq"
fi

echo "✅ Channel analysis complete! Results in: $OUTPUT_DIR"
""")
    
    # Script 2: Daily analysis cron job
    cron_script = scripts_dir / "daily_analysis.sh"
    with open(cron_script, 'w') as f:
        f.write("""#!/bin/bash
# Daily Discord analysis automation
# Add to cron: 0 6 * * * /path/to/daily_analysis.sh

cd /path/to/pepino  # Update this path

DATE=$(date +%Y%m%d)
OUTPUT_DIR="daily_analysis_$DATE"
mkdir -p "$OUTPUT_DIR"

echo "📊 Daily Discord Analysis - $DATE" | tee "$OUTPUT_DIR/analysis.log"

# Database stats
poetry run python -m pepino list stats --format json --output "$OUTPUT_DIR/stats_$DATE.json" 2>&1 | tee -a "$OUTPUT_DIR/analysis.log"

# Top channels analysis
poetry run python -m pepino analyze channels --limit 5 --format json --output "$OUTPUT_DIR/top_channels_$DATE.json" 2>&1 | tee -a "$OUTPUT_DIR/analysis.log"

# Server-wide topic analysis
poetry run python -m pepino analyze topics --format json --output "$OUTPUT_DIR/topics_$DATE.json" 2>&1 | tee -a "$OUTPUT_DIR/analysis.log"

# Temporal trends
poetry run python -m pepino analyze temporal --format json --output "$OUTPUT_DIR/temporal_$DATE.json" 2>&1 | tee -a "$OUTPUT_DIR/analysis.log"

echo "✅ Daily analysis complete!" | tee -a "$OUTPUT_DIR/analysis.log"
""")
    
    # Make scripts executable
    os.chmod(channel_script, 0o755)
    os.chmod(cron_script, 0o755)
    
    print(f"✅ Generated automation scripts:")
    print(f"   📄 {channel_script}")
    print(f"   📄 {cron_script}")
    print(f"\nUsage:")
    print(f"   ./automation_scripts/analyze_all_channels.sh")
    print(f"   # Add to cron: 0 6 * * * {cron_script.absolute()}")

if __name__ == "__main__":
    print("🎮 Pepino CLI Automation Demo")
    print("=" * 50)
    
    # Check if pepino is available
    try:
        result = subprocess.run(['poetry', 'run', 'python', '-m', 'pepino', '--help'], 
                              capture_output=True, check=True)
        print("✅ Pepino CLI available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Pepino CLI not available. Run 'poetry install' first.")
        exit(1)
    
    print("\nSelect automation demo:")
    print("1. Comprehensive Server Analysis")
    print("2. Targeted High-Activity Analysis") 
    print("3. Generate Automation Scripts")
    print("4. Run All Demos")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        comprehensive_server_analysis()
    elif choice == "2":
        targeted_automation_example()
    elif choice == "3":
        automation_script_generator()
    elif choice == "4":
        comprehensive_server_analysis()
        targeted_automation_example()
        automation_script_generator()
    else:
        print("❌ Invalid choice") 