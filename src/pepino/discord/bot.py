"""
Main Discord bot instance and setup.
"""

import discord
from discord.ext import commands

from pepino.config import Settings
from pepino.data.database.manager import DatabaseManager
from pepino.logging_config import get_logger, setup_bot_logging

# Initialize settings
settings = Settings()

# Set up professional logging
setup_bot_logging()
logger = get_logger(__name__)

# Bot configuration from settings
intents = discord.Intents.default()
intents.message_content = settings.message_content_intent
intents.members = settings.members_intent

# Create bot instance
bot = commands.Bot(command_prefix=settings.command_prefix, intents=intents)

# Global database manager using settings
db_manager = DatabaseManager(settings.db_path)


@bot.event
async def on_ready():
    """Called when the bot is ready and connected to Discord"""
    logger.info(f"Logged in as {bot.user.name} ({bot.user.id})")

    # Load the analysis commands (only if not already loaded)
    try:
        if "pepino.discord.commands.analysis" not in bot.extensions:
            await bot.load_extension("pepino.discord.commands.analysis")
            logger.info("Successfully loaded analysis commands")

            # Sync slash commands with Discord
            synced = await bot.tree.sync()
            logger.info(f"Successfully synced {len(synced)} slash commands")

            # List the synced commands
            for command in synced:
                logger.info(f"Synced command: {command.name}")
        else:
            logger.info("Analysis commands already loaded - skipping")

    except Exception as e:
        logger.error(f"Failed to load/sync commands: {e}")
        import traceback

        traceback.print_exc()


@bot.event
async def on_command_error(ctx, error):
    """Handle command errors"""
    if isinstance(error, commands.CommandNotFound):
        await ctx.send(
            "Command not found. Use slash commands for analysis (e.g., /help_analysis)."
        )
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"Missing required argument: {error.param.name}")
    else:
        await ctx.send(f"An error occurred: {str(error)}")


@bot.command(name="sync")
async def sync(ctx):
    """Manually sync slash commands"""
    try:
        synced = await bot.tree.sync()
        await ctx.send(f"Synced {len(synced)} command(s)")
        logger.info(f"Manually synced {len(synced)} command(s)")
    except Exception as e:
        await ctx.send(f"Failed to sync: {e}")
        logger.error(f"Failed to manually sync: {e}")


@bot.command(name="force_sync")
async def force_sync(ctx):
    """Force sync slash commands (admin only)"""
    try:
        # Clear existing commands first
        bot.tree.clear_commands()

        # Reload commands
        await bot.reload_extension("pepino.discord.commands.analysis")

        # Sync again
        synced = await bot.tree.sync()
        await ctx.send(f"Force synced {len(synced)} command(s)")

        logger.info(f"Force synced {len(synced)} command(s)")
        for command in synced:
            logger.info(f"Force synced: {command.name}")

    except Exception as e:
        await ctx.send(f"Failed to force sync: {e}")
        logger.error(f"Failed to force sync: {e}")
        import traceback

        traceback.print_exc()


@bot.command(name="test_autocomplete")
async def test_autocomplete(ctx):
    """Test if autocomplete data is available"""
    try:
        from pepino.analysis.channel_analyzer import ChannelAnalyzer
        from pepino.analysis.user_analyzer import UserAnalyzer
        from pepino.analysis.data_facade import get_analysis_data_facade

        # Create a temporary instance to test using facade
        with get_analysis_data_facade() as facade:
            user_analyzer = UserAnalyzer(facade)
            channel_analyzer = ChannelAnalyzer(facade)

            channels = await channel_analyzer.get_available_channels()
            users = await user_analyzer.get_available_users()

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
        logger.error(f"Test autocomplete failed: {e}")
        import traceback

        traceback.print_exc()


def run_bot():
    """Run the Discord bot with settings validation."""
    try:
        # Validate configuration
        settings.validate_required()

        # Run the bot
        bot.run(settings.discord_token)

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        raise
