import os
import discord
from discord.ext import commands
import logging
import base64
import io
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('discord')

# Bot configuration
intents = discord.Intents.default()
intents.message_content = True  # Enable message content intent
intents.members = True  # Enable member intent for user analysis

# Create bot instance
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    """Called when the bot is ready and connected to Discord"""
    logger.info(f'Logged in as {bot.user.name} ({bot.user.id})')
    logger.info('------')
    
    # Load the analysis commands
    try:
        await bot.load_extension('bot_commands')
        logger.info('Successfully loaded analysis commands')
        
        # Sync slash commands with Discord
        synced = await bot.tree.sync()
        logger.info(f'Successfully synced {len(synced)} slash commands')
        
        # List the synced commands
        for command in synced:
            logger.info(f'Synced command: {command.name}')
            
    except Exception as e:
        logger.error(f'Failed to load/sync commands: {e}')
        import traceback
        traceback.print_exc()

@bot.event
async def on_command_error(ctx, error):
    """Handle command errors"""
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("Command not found. Use /help_analysis to see available slash commands.")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"Missing required argument: {error.param.name}")
    else:
        await ctx.send(f"An error occurred: {str(error)}")

@bot.command(name='sync')
async def sync(ctx):
    """Manually sync slash commands"""
    try:
        synced = await bot.tree.sync()
        await ctx.send(f'Synced {len(synced)} command(s)')
        logger.info(f'Manually synced {len(synced)} command(s)')
    except Exception as e:
        await ctx.send(f'Failed to sync: {e}')
        logger.error(f'Failed to manually sync: {e}')

@bot.command(name='force_sync')
async def force_sync(ctx):
    """Force sync slash commands (admin only)"""
    try:
        # Clear existing commands first
        bot.tree.clear_commands()
        
        # Reload commands
        await bot.reload_extension('bot_commands')
        
        # Sync again
        synced = await bot.tree.sync()
        await ctx.send(f'Force synced {len(synced)} command(s)')
        
        logger.info(f'Force synced {len(synced)} command(s)')
        for command in synced:
            logger.info(f'Force synced: {command.name}')
            
    except Exception as e:
        await ctx.send(f'Failed to force sync: {e}')
        logger.error(f'Failed to force sync: {e}')
        import traceback
        traceback.print_exc()

@bot.command(name='test_autocomplete')
async def test_autocomplete(ctx):
    """Test if autocomplete data is available"""
    try:
        from bot_commands import AnalysisCommands
        
        # Create a temporary instance to test
        analysis_cog = AnalysisCommands(bot)
        await analysis_cog.analyzer.initialize()
        
        channels = await analysis_cog.analyzer.get_available_channels()
        users = await analysis_cog.analyzer.get_available_users()
        
        await ctx.send(f"Found {len(channels)} channels and {len(users)} users")
        
        # Show first few examples
        if channels:
            sample_channels = channels[:5]
            await ctx.send(f"Sample channels: {', '.join(sample_channels)}")
        
        if users:
            sample_users = [u for u in users[:5] if u]  # Filter out empty names
            await ctx.send(f"Sample users: {', '.join(sample_users)}")
            
    except Exception as e:
        await ctx.send(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function to run the bot"""
    # Get token from environment variable
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        raise ValueError("No Discord token found. Please set the DISCORD_TOKEN environment variable.")
    
    # Run the bot
    bot.run(token)

if __name__ == '__main__':
    main()