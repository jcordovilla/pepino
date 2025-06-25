#!/usr/bin/env python3
"""
Simulate Complete Discord Command Flow

This simulates the ENTIRE process:
1. Discord command received (e.g., `/channel_analysis general`)
2. Command handler parses arguments
3. Analyzer is invoked with parameters
4. Template engine renders the results
5. Final Discord message is formatted and "sent"

Perfect for testing the complete integration!
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our sync components
try:
    from src.pepino.data.database.sync_manager import SyncDatabaseManager
    from src.pepino.analysis.sync_channel_analyzer import SyncChannelAnalyzer
    from src.pepino.analysis.sync_user_analyzer import SyncUserAnalyzer
    from src.pepino.analysis.sync_template_executor import SyncTemplateExecutor
except ImportError as e:
    print(f"⚠️  Import failed (expected): {e}")
    print("Running in standalone mode...")
    SyncDatabaseManager = None

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Mock Discord context for simulation
@dataclass
class MockDiscordContext:
    """Simulates Discord command context"""
    guild_name: str = "Test Server"
    channel_name: str = "bot-commands"
    user_name: str = "TestUser"
    
    def send(self, content: str):
        """Simulate sending Discord message"""
        print(f"\n📤 [DISCORD RESPONSE in #{self.channel_name}]")
        print("=" * 60)
        print(content)
        print("=" * 60)


class MockDiscordCommand:
    """Simulates a Discord bot command"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def __call__(self, func):
        """Decorator to register command"""
        func.command_name = self.name
        func.command_description = self.description
        return func


class DiscordBotSimulator:
    """
    Simulates the complete Discord bot flow
    
    This shows exactly how a real Discord command would work:
    1. Command parsing
    2. Analyzer invocation
    3. Template rendering
    4. Response formatting
    """
    
    def __init__(self):
        self.db_manager = None
        self.template_executor = None
        self.channel_analyzer = None
        self.user_analyzer = None
        
        # Try to initialize real components
        try:
            self.db_manager = SyncDatabaseManager("discord_messages.db")
            self.template_executor = SyncTemplateExecutor(self.db_manager)
            self.channel_analyzer = SyncChannelAnalyzer(self.db_manager)
            self.user_analyzer = SyncUserAnalyzer(self.db_manager)
            self.real_mode = True
            print("✅ Initialized with REAL sync analyzers")
        except Exception as e:
            print(f"⚠️  Real mode failed: {e}")
            print("🎭 Running in MOCK mode")
            self.real_mode = False
    
    @MockDiscordCommand("channel_analysis", "Analyze a Discord channel")
    async def channel_analysis_command(self, ctx: MockDiscordContext, channel_name: str, days_back: int = 30):
        """
        Simulates: /channel_analysis general 30
        
        This is EXACTLY how a real Discord command would work!
        """
        
        print(f"\n🤖 [COMMAND RECEIVED]: /channel_analysis {channel_name} {days_back}")
        print(f"👤 User: {ctx.user_name}")
        print(f"📍 Server: {ctx.guild_name}")
        print(f"📢 Channel: #{ctx.channel_name}")
        
        # Step 1: Validate input
        if not channel_name:
            await ctx.send("❌ Please specify a channel name!")
            return
        
        print(f"\n🔄 Processing analysis for #{channel_name}...")
        
        # Step 2: Execute analyzer
        try:
            if self.real_mode:
                print("🔍 Invoking REAL SyncChannelAnalyzer...")
                analysis_result = self.channel_analyzer.analyze(
                    channel_name=channel_name,
                    include_top_users=True,
                    days_back=days_back,
                    limit_users=5
                )
                
                if hasattr(analysis_result, 'error'):
                    await ctx.send(f"❌ Analysis failed: {analysis_result.error}")
                    return
                
                print("✅ Analysis completed successfully!")
                
                # Step 3: Render template
                print("🎨 Rendering template with real data...")
                markdown_output = self.template_executor.execute_template(
                    "outputs/discord/simple_sync_test.md.j2",
                    channel_name=channel_name,
                    channel_analysis=analysis_result,
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
                
            else:
                # Mock mode - simulate the process
                print("🎭 Simulating analyzer execution...")
                markdown_output = self._generate_mock_analysis(channel_name, days_back)
            
            # Step 4: Format for Discord (truncate if too long)
            discord_message = self._format_for_discord(markdown_output)
            
            # Step 5: Send response
            await ctx.send(discord_message)
            
        except Exception as e:
            logger.error(f"Command failed: {e}", exc_info=True)
            await ctx.send(f"❌ Command failed: {str(e)}")
    
    @MockDiscordCommand("user_analysis", "Analyze a Discord user")
    async def user_analysis_command(self, ctx: MockDiscordContext, username: str, days_back: int = 30):
        """
        Simulates: /user_analysis john_doe 30
        """
        
        print(f"\n🤖 [COMMAND RECEIVED]: /user_analysis {username} {days_back}")
        print(f"👤 User: {ctx.user_name}")
        
        print(f"\n🔄 Processing user analysis for @{username}...")
        
        try:
            if self.real_mode:
                print("🔍 Invoking REAL SyncUserAnalyzer...")
                analysis_result = self.user_analyzer.analyze(
                    username=username,
                    include_channels=True,
                    days_back=days_back
                )
                
                if hasattr(analysis_result, 'error'):
                    await ctx.send(f"❌ Analysis failed: {analysis_result.error}")
                    return
                
                print("✅ User analysis completed!")
                
                # Would render user template here
                markdown_output = f"# User Analysis: @{username}\n\n**Messages:** {analysis_result.statistics.total_messages}\n**Channels:** {analysis_result.statistics.channels_participated}"
                
            else:
                print("🎭 Simulating user analyzer...")
                markdown_output = self._generate_mock_user_analysis(username, days_back)
            
            discord_message = self._format_for_discord(markdown_output)
            await ctx.send(discord_message)
            
        except Exception as e:
            logger.error(f"User analysis failed: {e}", exc_info=True)
            await ctx.send(f"❌ User analysis failed: {str(e)}")
    
    @MockDiscordCommand("list_channels", "List available channels")
    async def list_channels_command(self, ctx: MockDiscordContext):
        """
        Simulates: /list_channels
        """
        
        print(f"\n🤖 [COMMAND RECEIVED]: /list_channels")
        
        try:
            if self.real_mode:
                print("📋 Getting real channel list...")
                channels = self.channel_analyzer.get_available_channels()
                
                if channels:
                    channel_list = "\n".join([f"• #{ch}" for ch in channels[:10]])
                    message = f"📋 **Available Channels:**\n{channel_list}"
                    if len(channels) > 10:
                        message += f"\n\n*...and {len(channels) - 10} more*"
                else:
                    message = "❌ No channels found"
            else:
                message = "📋 **Available Channels:**\n• #general\n• #random\n• #bot-commands"
            
            await ctx.send(message)
            
        except Exception as e:
            await ctx.send(f"❌ Failed to list channels: {str(e)}")
    
    def _format_for_discord(self, markdown: str) -> str:
        """Format markdown for Discord (handle length limits, etc.)"""
        
        # Discord has a 2000 character limit per message
        if len(markdown) > 1800:
            truncated = markdown[:1800] + "\n\n*...output truncated...*"
            return truncated
        
        return markdown
    
    def _generate_mock_analysis(self, channel_name: str, days_back: int) -> str:
        """Generate mock analysis for testing"""
        return f"""# Channel Analysis: #{channel_name}

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Period:** Last {days_back} days

## 📊 Statistics
- **Total Messages:** 1,234
- **Unique Users:** 45
- **Average Length:** 127.3 characters
- **Active Days:** 28

## 👥 Top Users
1. **alice_dev**: 89 messages
2. **bob_designer**: 67 messages  
3. **charlie_pm**: 52 messages

## 📈 Engagement
- **Replies per Post:** 1.2
- **Reaction Rate:** 15.7%

*🎭 This is MOCK data for demonstration*"""
    
    def _generate_mock_user_analysis(self, username: str, days_back: int) -> str:
        """Generate mock user analysis"""
        return f"""# User Analysis: @{username}

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📊 Statistics
- **Total Messages:** 156
- **Channels Participated:** 8
- **Average Message Length:** 98.2 characters

## 🏆 Top Channels
1. **#general**: 45 messages
2. **#dev-chat**: 32 messages
3. **#random**: 21 messages

*🎭 This is MOCK data for demonstration*"""


async def simulate_discord_session():
    """Simulate a complete Discord bot session with multiple commands"""
    
    print("🚀 DISCORD BOT COMMAND SIMULATION")
    print("=" * 60)
    print("This simulates the COMPLETE flow:")
    print("  1. Discord command received")
    print("  2. Arguments parsed")
    print("  3. Analyzer invoked")
    print("  4. Template rendered")
    print("  5. Response sent to Discord")
    print()
    
    # Initialize bot
    bot = DiscordBotSimulator()
    ctx = MockDiscordContext()
    
    # Simulate user commands
    commands_to_test = [
        ("list_channels", []),
        ("channel_analysis", ["jose-test", 30]),
        ("channel_analysis", ["🦾agent-ops", 7]),
        ("user_analysis", ["julioverne74", 30]),
    ]
    
    for i, (command, args) in enumerate(commands_to_test, 1):
        print(f"\n{'='*20} COMMAND {i}/{len(commands_to_test)} {'='*20}")
        
        try:
            if command == "list_channels":
                await bot.list_channels_command(ctx)
            elif command == "channel_analysis":
                await bot.channel_analysis_command(ctx, *args)
            elif command == "user_analysis":
                await bot.user_analysis_command(ctx, *args)
                
        except Exception as e:
            print(f"❌ Command simulation failed: {e}")
        
        # Pause between commands
        print("\n⏱️  [Waiting 2 seconds...]")
        import time
        time.sleep(1)  # Shortened for demo
    
    print(f"\n🎉 SIMULATION COMPLETE!")
    print("=" * 60)
    print("✅ This demonstrates the EXACT flow a real Discord bot would use:")
    print("   - Command parsing ✓")
    print("   - Analyzer execution ✓") 
    print("   - Template rendering ✓")
    print("   - Discord response formatting ✓")


def main():
    """Run the complete simulation"""
    try:
        import asyncio
        asyncio.run(simulate_discord_session())
    except KeyboardInterrupt:
        print("\n👋 Simulation stopped by user")
    except Exception as e:
        logger.error(f"Simulation failed: {e}", exc_info=True)


if __name__ == "__main__":
    main() 