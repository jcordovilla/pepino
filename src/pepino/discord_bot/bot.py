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
intents.message_content = settings.discord_bot_message_content_intent
intents.members = settings.discord_bot_members_intent

# Create bot instance
bot = commands.Bot(command_prefix=settings.discord_bot_command_prefix, intents=intents)

# Global database manager using settings
db_manager = DatabaseManager(settings.database_sqlite_path)


@bot.event
async def on_ready():
    """Called when the bot is ready and connected to Discord"""
    logger.info(f"Logged in as {bot.user.name} ({bot.user.id})")

    # Load the analysis commands (only if not already loaded)
    try:
        if "pepino.discord_bot.commands.analysis" not in bot.extensions:
            await bot.load_extension("pepino.discord_bot.commands.analysis")
            logger.info("Successfully loaded analysis commands")

        if "pepino.discord_bot.commands.legacy" not in bot.extensions:
            await bot.load_extension("pepino.discord_bot.commands.legacy")
            logger.info("Successfully loaded legacy analysis commands")

        # Sync slash commands with Discord
        synced = await bot.tree.sync()
        logger.info(f"Successfully synced {len(synced)} slash commands: {[c.name for c in synced]}")

    except discord.Forbidden:
        logger.error("Bot lacks permissions to sync commands")
    except discord.HTTPException as e:
        logger.error(f"Discord API error during sync: {e}")
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
        

def run_bot():
    """Run the Discord bot with settings validation."""
    try:
        # Validate configuration
        settings.validate_required()

        # Run the bot
        bot.run(settings.discord_bot_token)

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        raise
