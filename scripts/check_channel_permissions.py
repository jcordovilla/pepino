#!/usr/bin/env python3
"""
Script to check actual Discord channel permissions by connecting to the server.
Groups channels by accessible (bot can read) and not accessible (bot cannot read).
"""
import sys
import os
import asyncio
import discord
from discord.ext import commands
from typing import List, Dict, Tuple
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pepino.config import Settings

class ChannelPermissionChecker:
    def __init__(self, bot_token: str):
        self.bot_token = bot_token
        self.bot = commands.Bot(command_prefix='!', intents=discord.Intents.all())
        self.accessible_channels = []
        self.inaccessible_channels = []
        
    async def check_channel_permissions(self, channel: discord.abc.GuildChannel) -> bool:
        """Check if the bot can read messages in a specific channel."""
        try:
            # Check if bot has permission to view channel
            if not channel.permissions_for(channel.guild.me).view_channel:
                return False
            
            # Check if bot has permission to read message history
            if not channel.permissions_for(channel.guild.me).read_message_history:
                return False
            
            # For text channels, also check if we can read messages
            if isinstance(channel, discord.TextChannel):
                if not channel.permissions_for(channel.guild.me).read_messages:
                    return False
            
            # Try to fetch a message to verify actual access
            try:
                if isinstance(channel, discord.TextChannel):
                    # Try to get the last message
                    async for message in channel.history(limit=1):
                        break
                    return True
                elif isinstance(channel, discord.ForumChannel):
                    # For forum channels, check if we can view posts
                    async for thread in channel.archived_threads(limit=1):
                        break
                    return True
                elif isinstance(channel, discord.VoiceChannel):
                    # For voice channels, just check permissions
                    return True
                else:
                    # For other channel types, check basic permissions
                    return True
            except discord.Forbidden:
                return False
            except discord.HTTPException:
                # If we can't fetch messages but have permissions, assume accessible
                return True
                
        except Exception as e:
            print(f"Error checking permissions for {channel.name}: {e}")
            return False
    
    async def scan_guild_channels(self, guild: discord.Guild):
        """Scan all channels in a guild and check their accessibility."""
        print(f"🔍 Scanning guild: {guild.name} (ID: {guild.id})")
        print(f"   Total channels: {len(guild.channels)}")
        print()
        
        for channel in guild.channels:
            try:
                # Get channel info
                channel_info = {
                    'id': str(channel.id),
                    'name': channel.name,
                    'type': str(channel.type),
                    'position': channel.position,
                    'category': channel.category.name if channel.category else None,
                    'guild_id': str(guild.id),
                    'guild_name': guild.name
                }
                
                # Check if accessible
                is_accessible = await self.check_channel_permissions(channel)
                
                if is_accessible:
                    self.accessible_channels.append(channel_info)
                    print(f"✅ {channel.name} ({channel.type})")
                else:
                    self.inaccessible_channels.append(channel_info)
                    print(f"❌ {channel.name} ({channel.type})")
                    
            except Exception as e:
                print(f"⚠️  Error processing {getattr(channel, 'name', 'Unknown')}: {e}")
                # Add to inaccessible with error info
                self.inaccessible_channels.append({
                    'id': str(getattr(channel, 'id', 'unknown')),
                    'name': getattr(channel, 'name', 'Unknown'),
                    'type': str(getattr(channel, 'type', 'unknown')),
                    'error': str(e),
                    'guild_id': str(guild.id),
                    'guild_name': guild.name
                })
    
    async def run_check(self):
        """Run the permission check for all guilds the bot is in."""
        print("=== Discord Channel Permission Check ===\n")
        print(f"Bot: {self.bot.user.name}#{self.bot.user.discriminator}")
        print(f"Guilds: {len(self.bot.guilds)}")
        print()
        
        for guild in self.bot.guilds:
            await self.scan_guild_channels(guild)
            print()
        
        # Print summary
        self.print_summary()
        
        # Save results to file
        self.save_results()
    
    def print_summary(self):
        """Print a summary of the permission check results."""
        total_channels = len(self.accessible_channels) + len(self.inaccessible_channels)
        
        print("=" * 60)
        print("📊 PERMISSION CHECK SUMMARY")
        print("=" * 60)
        print(f"Total Channels: {total_channels}")
        print(f"✅ Accessible: {len(self.accessible_channels)} ({len(self.accessible_channels)/total_channels*100:.1f}%)")
        print(f"❌ Inaccessible: {len(self.inaccessible_channels)} ({len(self.inaccessible_channels)/total_channels*100:.1f}%)")
        print()
        
        if self.accessible_channels:
            print("✅ ACCESSIBLE CHANNELS:")
            print("-" * 40)
            for channel in self.accessible_channels:
                category_info = f" [{channel['category']}]" if channel['category'] else ""
                print(f"  • {channel['name']} ({channel['type']}){category_info}")
            print()
        
        if self.inaccessible_channels:
            print("❌ INACCESSIBLE CHANNELS:")
            print("-" * 40)
            for channel in self.inaccessible_channels:
                category_info = f" [{channel['category']}]" if channel['category'] else ""
                error_info = f" - {channel['error']}" if 'error' in channel else ""
                print(f"  • {channel['name']} ({channel['type']}){category_info}{error_info}")
            print()
    
    def save_results(self):
        """Save the results to a JSON file."""
        results = {
            'timestamp': discord.utils.utcnow().isoformat(),
            'bot_user': f"{self.bot.user.name}#{self.bot.user.discriminator}",
            'summary': {
                'total_channels': len(self.accessible_channels) + len(self.inaccessible_channels),
                'accessible_count': len(self.accessible_channels),
                'inaccessible_count': len(self.inaccessible_channels)
            },
            'accessible_channels': self.accessible_channels,
            'inaccessible_channels': self.inaccessible_channels
        }
        
        filename = f"channel_permissions_{discord.utils.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {filename}")

async def main():
    """Main function to run the permission check."""
    # Get bot token from environment or config
    bot_token = os.getenv('DISCORD_TOKEN')
    
    if not bot_token:
        try:
            settings = Settings()
            bot_token = settings.discord_token
        except Exception:
            pass

    if not bot_token:
        print("❌ No Discord bot token found!")
        print("Set DISCORD_TOKEN environment variable or configure in settings.")
        # Return a non-zero exit code instead of calling sys.exit() from inside
        # the coroutine. The caller will handle exiting the process.
        return 1
    
    # Create and run the checker
    checker = ChannelPermissionChecker(bot_token)
    
    @checker.bot.event
    async def on_ready():
        print(f"🤖 Bot connected as {checker.bot.user}")
        await checker.run_check()
        await checker.bot.close()
    
    exit_code = 0
    try:
        await checker.bot.start(bot_token)
    except discord.LoginFailure:
        print("❌ Invalid bot token!")
        exit_code = 1
    except Exception as e:
        print(f"❌ Error connecting to Discord: {e}")
        exit_code = 1
    finally:
        # Ensure proper cleanup of the bot and underlying aiohttp resources.
        try:
            if not checker.bot.is_closed():
                await checker.bot.close()
        except Exception:
            pass

        # Try to explicitly close any aiohttp ClientSession objects that
        # may be reachable from the bot internals (different discord.py
        # versions store the session in different places).
        try:
            import aiohttp

            sessions = set()

            # inspect common places
            candidates = [checker.bot]
            conn = getattr(checker.bot, '_connection', None)
            if conn:
                candidates.append(conn)
            http = getattr(checker.bot, 'http', None)
            if http:
                candidates.append(http)

            for obj in candidates:
                if not obj:
                    continue
                for name in dir(obj):
                    try:
                        val = getattr(obj, name)
                    except Exception:
                        continue
                    try:
                        if isinstance(val, aiohttp.ClientSession):
                            sessions.add(val)
                    except Exception:
                        # Some attributes may raise on isinstance checks; ignore
                        continue

            for s in sessions:
                try:
                    await s.close()
                except Exception:
                    pass
        except Exception:
            # If aiohttp isn't available or something fails, ignore and continue
            pass

        # Cancel any remaining pending tasks (except the current one) and wait
        try:
            current = asyncio.current_task()
            pending = [t for t in asyncio.all_tasks() if t is not current]
            if pending:
                for t in pending:
                    try:
                        t.cancel()
                    except Exception:
                        pass

                await asyncio.gather(*pending, return_exceptions=True)
        except Exception:
            pass

        # Give asyncio a moment to finalize closing transports/sessions
        try:
            await asyncio.sleep(0)
        except Exception:
            pass

    return exit_code

if __name__ == "__main__":
    # Run the async main and handle process exit outside the event loop.
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        exit_code = 1
    except Exception as e:
        print(f"❌ Unhandled error: {e}")
        exit_code = 1

    # Ensure we exit the process with the returned code
    sys.exit(exit_code or 0)
