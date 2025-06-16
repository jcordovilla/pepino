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
        await bot.tree.sync()
        logger.info('Successfully synced slash commands')
    except Exception as e:
        logger.error(f'Failed to load analysis commands: {e}')

@bot.event
async def on_command_error(ctx, error):
    """Handle command errors"""
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("Command not found. Use !analyze list to see available commands.")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"Missing required argument: {error.param.name}")
    else:
        await ctx.send(f"An error occurred: {str(error)}")

@bot.command(name="analyze")
async def analyze(ctx, task: str = None, *args):
    """Run various analysis tasks on the message database"""
    try:
        if not task:
            await ctx.send("""
**Available Analysis Tasks:**
- `channel <channel_name>`: Get insights about a specific channel
- `user <user_id or name>`: Get insights about a specific user
- `wordfreq`: Show most common words
- `userstats`: Show user activity statistics
- `temporal`: Show temporal activity patterns
- `topics [channel_name]`: Analyze topics in messages (optionally specify a channel)
- `similar <message_id>`: Find similar messages
- `conversations`: Analyze conversation chains
- `runall`: Run all analyses at once

Example: `/analyze channel general` or `/analyze user Jose Cordovilla`
            """)
            return

        # Initialize analyzer
        analyzer = MessageAnalyzer('discord_messages.db')
        
        # Parse arguments
        args_dict = {}
        if args:
            if task == 'channel':
                args_dict['channel_name'] = ' '.join(args)
            elif task == 'user':
                args_dict['user_id'] = ' '.join(args)  # Join all args for user name
            elif task == 'similar':
                try:
                    args_dict['message_id'] = int(args[0])
                except ValueError:
                    await ctx.send("Error: message_id must be a number")
                    return
            elif task == 'topics' and args:
                args_dict['channel_name'] = ' '.join(args)
        
        # Run analysis
        result = await analyzer.run_analysis(task, args_dict)
        
        # Check if result contains a base64 image
        if result and result.startswith('data:image/png;base64,'):
            # Extract the base64 data
            base64_data = result.split(',')[1]
            image_data = base64.b64decode(base64_data)
            
            # Create a file-like object
            image_file = io.BytesIO(image_data)
            image_file.name = f'analysis_{task}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            
            # Send the image
            await ctx.send(file=discord.File(image_file, filename=image_file.name))
        else:
            # Send text result
            await ctx.send(result)
            
    except Exception as e:
        await ctx.send(f"Error running analysis: {str(e)}")
    finally:
        if 'analyzer' in locals():
            analyzer.close()

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