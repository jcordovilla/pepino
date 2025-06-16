import discord
from discord.ext import commands
from discord import app_commands
from analysis import DiscordBotAnalyzer
import asyncio
from typing import Optional, Union, List, Dict
from difflib import get_close_matches
import base64
import io
from datetime import datetime
import os

class AnalysisCommands(commands.Cog):
    """Commands for analyzing Discord messages"""
    
    def __init__(self, bot):
        self.bot = bot
        self.analyzer = DiscordBotAnalyzer()
        self.analysis_in_progress = False
    
    async def cog_load(self):
        """Initialize the analyzer when the cog is loaded"""
        try:
            await self.analyzer.initialize()
            print("Analysis commands initialized successfully")
        except Exception as e:
            print(f"Error initializing analysis commands: {str(e)}")
            raise

    @commands.Cog.listener()
    async def on_ready(self):
        """Called when the bot is ready"""
        print(f'Analysis commands loaded!')
    
    @app_commands.command(
        name="analyze",
        description="General analysis tasks (use dedicated commands for better autocomplete)"
    )
    @app_commands.describe(
        task="The analysis task to perform",
        channel="Channel name (better to use /channel_analysis or /topics_analysis)",
        user="Username (better to use /user_analysis)"
    )
    @app_commands.choices(task=[
        app_commands.Choice(name="wordfreq - Word frequency analysis", value="wordfreq"),
        app_commands.Choice(name="userstats - User statistics", value="userstats"),
        app_commands.Choice(name="channel - Channel analysis (use /channel_analysis)", value="channel"),
        app_commands.Choice(name="topics - Topic analysis (use /topics_analysis)", value="topics"),
        app_commands.Choice(name="temporal - Temporal analysis", value="temporal"),
        app_commands.Choice(name="user - User analysis (use /user_analysis)", value="user")
    ])
    async def analyze(
        self,
        interaction: discord.Interaction,
        task: str,
        channel: str = None,
        user: str = None
    ):
        """Analyze Discord messages with various tasks"""
        try:
            print(f"Analyze command called with task='{task}', channel='{channel}', user='{user}'")
            
            # Ensure analyzer is initialized
            try:
                await self.analyzer.initialize()
            except Exception as e:
                await interaction.response.send_message(f"Error initializing database: {str(e)}")
                return
            
            # Defer the response since analysis might take a moment
            await interaction.response.defer()
            
            # Handle different tasks
            if task == "wordfreq":
                result = await self.analyzer.update_word_frequencies()
            elif task == "userstats":
                result = await self.analyzer.update_user_statistics()
            elif task == "channel":
                print(f"Channel task: channel parameter = '{channel}' (type: {type(channel)})")
                if not channel:
                    await interaction.followup.send(f"❌ **Channel parameter missing!**\n\n**For better experience, use the dedicated command:**\n`/channel_analysis` - Select channel from autocomplete dropdown\n\n**Or provide channel parameter:**\nChannel parameter received: `{repr(channel)}`")
                    return
                print(f"Calling get_channel_insights with channel: '{channel}'")
                result = await self.analyzer.get_channel_insights(channel)
            elif task == "topics":
                args_dict = {}
                if channel:
                    print(f"Topics with channel filter: '{channel}'")
                    args_dict["channel_name"] = channel
                else:
                    print("Topics without channel filter")
                result = await self.analyzer.analyze_topics_spacy(args_dict)
            elif task == "temporal":
                result = await self.analyzer.update_temporal_stats()
            elif task == "user":
                print(f"User task: user parameter = '{user}' (type: {type(user)})")
                if not user:
                    await interaction.followup.send(f"❌ **User parameter missing!**\n\n**For better experience, use the dedicated command:**\n`/user_analysis` - Select user from autocomplete dropdown\n\n**Or provide user parameter:**\nUser parameter received: `{repr(user)}`")
                    return
                print(f"Calling get_user_insights with user: '{user}'")
                result = await self.analyzer.get_user_insights(user)
            else:
                await interaction.followup.send(f"Unknown task: {task}")
                return
            
            # Handle the result
            if isinstance(result, tuple):
                # If result is a tuple (text, file_path)
                text, file_path = result
                await interaction.followup.send(text, file=discord.File(file_path))
                # Clean up the temporary file
                try:
                    os.remove(file_path)
                except:
                    pass
            elif isinstance(result, str) and os.path.isfile(result):
                # If result is just a file path
                await interaction.followup.send(file=discord.File(result))
                # Clean up the temporary file
                try:
                    os.remove(result)
                except:
                    pass
            else:
                # If result is text, split it if too long
                if len(result) > 2000:
                    chunks = [result[i:i+1900] for i in range(0, len(result), 1900)]
                    for chunk in chunks:
                        await interaction.followup.send(chunk)
                else:
                    await interaction.followup.send(result)
                    
        except Exception as e:
            if not interaction.response.is_done():
                await interaction.response.send_message(f"Error during analysis: {str(e)}")
            else:
                await interaction.followup.send(f"Error during analysis: {str(e)}")

    @app_commands.command(
        name="list_users",
        description="List all available users for analysis"
    )
    async def list_users(self, interaction: discord.Interaction):
        """List all available users for analysis"""
        try:
            await interaction.response.defer()
            try:
                await self.analyzer.initialize()
            except Exception as e:
                await interaction.followup.send(f"Error initializing database: {str(e)}")
                return
                
            users = await self.analyzer.get_available_users()
            if users:
                user_list = "\n".join([f"• {u}" for u in users])
                await interaction.followup.send(f"**Available Users:**\n{user_list}")
            else:
                await interaction.followup.send("No users found in the database.")
        except Exception as e:
            await interaction.followup.send(f"Error listing users: {str(e)}")

    @app_commands.command(
        name="list_channels",
        description="List all available channels for analysis"
    )
    async def list_channels(self, interaction: discord.Interaction):
        """List all available channels for analysis"""
        try:
            await interaction.response.defer()
            try:
                await self.analyzer.initialize()
            except Exception as e:
                await interaction.followup.send(f"Error initializing database: {str(e)}")
                return
                
            channels = await self.analyzer.get_available_channels()
            if channels:
                channel_list = "\n".join([f"• {c}" for c in channels])
                await interaction.followup.send(f"**Available Channels:**\n{channel_list}")
            else:
                await interaction.followup.send("No channels found in the database.")
        except Exception as e:
            await interaction.followup.send(f"Error listing channels: {str(e)}")

    @analyze.autocomplete('channel')
    async def channel_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> List[app_commands.Choice[str]]:
        """Autocomplete for channel names"""
        try:
            print(f"Channel autocomplete called with current='{current}'")
            
            # Ensure analyzer is initialized
            await self.analyzer.initialize()
            
            # Get available channels
            channels = await self.analyzer.get_available_channels()
            print(f"Found {len(channels)} channels")
            
            # Filter channels based on current input
            if current:
                # Case-insensitive filtering
                filtered_channels = [
                    channel for channel in channels 
                    if current.lower() in channel.lower()
                ]
                print(f"Filtered to {len(filtered_channels)} channels matching '{current}'")
            else:
                filtered_channels = channels
                print(f"No filter, showing all {len(filtered_channels)} channels")
            
            # Limit to 25 choices (Discord limit)
            filtered_channels = filtered_channels[:25]
            
            # Return choices
            choices = [
                app_commands.Choice(name=channel, value=channel)
                for channel in filtered_channels
                if channel  # Skip empty channel names
            ]
            
            print(f"Returning {len(choices)} choices")
            return choices
            
        except Exception as e:
            print(f"Error in channel autocomplete: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    @analyze.autocomplete('user')
    async def user_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> List[app_commands.Choice[str]]:
        """Autocomplete for user names"""
        try:
            print(f"User autocomplete called with current='{current}'")
            
            # Ensure analyzer is initialized
            await self.analyzer.initialize()
            
            # Get available users
            users = await self.analyzer.get_available_users()
            print(f"Found {len(users)} users")
            
            # Filter users based on current input
            if current:
                # Case-insensitive filtering
                filtered_users = [
                    user for user in users 
                    if current.lower() in user.lower()
                ]
                print(f"Filtered to {len(filtered_users)} users matching '{current}'")
            else:
                filtered_users = users
                print(f"No filter, showing all {len(filtered_users)} users")
            
            # Limit to 25 choices (Discord limit)
            filtered_users = filtered_users[:25]
            
            # Return choices
            choices = [
                app_commands.Choice(name=user, value=user)
                for user in filtered_users
                if user  # Skip empty usernames
            ]
            
            print(f"Returning {len(choices)} user choices")
            return choices
            
        except Exception as e:
            print(f"Error in user autocomplete: {e}")
            import traceback
            traceback.print_exc()
            return []

    @app_commands.command(
        name="channel_analysis",
        description="Analyze a specific Discord channel"
    )
    @app_commands.describe(
        channel="Channel name to analyze"
    )
    async def channel_analysis(
        self,
        interaction: discord.Interaction,
        channel: str
    ):
        """Analyze a specific Discord channel"""
        try:
            print(f"Channel analysis command called with channel='{channel}'")
            
            await self.analyzer.initialize()
            await interaction.response.defer()
            
            result = await self.analyzer.get_channel_insights(channel)
            
            # Handle the result
            if isinstance(result, tuple):
                text_result, file_path = result
                if file_path and os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        file = discord.File(f, filename=os.path.basename(file_path))
                        await interaction.followup.send(text_result, file=file)
                else:
                    await interaction.followup.send(text_result)
            else:
                if len(result) > 2000:
                    chunks = [result[i:i+1900] for i in range(0, len(result), 1900)]
                    for chunk in chunks:
                        await interaction.followup.send(chunk)
                else:
                    await interaction.followup.send(result)
                    
        except Exception as e:
            if not interaction.response.is_done():
                await interaction.response.send_message(f"Error during channel analysis: {str(e)}")
            else:
                await interaction.followup.send(f"Error during channel analysis: {str(e)}")

    @channel_analysis.autocomplete('channel')
    async def channel_analysis_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> List[app_commands.Choice[str]]:
        """Autocomplete for channel analysis"""
        return await self.channel_autocomplete(interaction, current)

    @app_commands.command(
        name="user_analysis", 
        description="Analyze a specific Discord user"
    )
    @app_commands.describe(
        user="Username to analyze"
    )
    async def user_analysis(
        self,
        interaction: discord.Interaction,
        user: str
    ):
        """Analyze a specific Discord user"""
        try:
            print(f"User analysis command called with user='{user}'")
            
            await self.analyzer.initialize()
            await interaction.response.defer()
            
            result = await self.analyzer.get_user_insights(user)
            
            # Handle the result  
            if len(result) > 2000:
                chunks = [result[i:i+1900] for i in range(0, len(result), 1900)]
                for chunk in chunks:
                    await interaction.followup.send(chunk)
            else:
                await interaction.followup.send(result)
                    
        except Exception as e:
            if not interaction.response.is_done():
                await interaction.response.send_message(f"Error during user analysis: {str(e)}")
            else:
                await interaction.followup.send(f"Error during user analysis: {str(e)}")

    @user_analysis.autocomplete('user')
    async def user_analysis_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> List[app_commands.Choice[str]]:
        """Autocomplete for user analysis"""
        return await self.user_autocomplete(interaction, current)

    @app_commands.command(
        name="topics_analysis",
        description="Analyze topics in messages, optionally filtered by channel"
    )
    @app_commands.describe(
        channel="Optional: Channel name to filter analysis"
    )
    async def topics_analysis(
        self,
        interaction: discord.Interaction,
        channel: str = None
    ):
        """Analyze topics in messages with optional channel filter"""
        try:
            print(f"Topics analysis command called with channel='{channel}'")
            
            await self.analyzer.initialize()
            await interaction.response.defer()
            
            args_dict = {}
            if channel:
                args_dict["channel_name"] = channel
                
            result = await self.analyzer.analyze_topics_spacy(args_dict)
            
            # Handle the result
            if len(result) > 2000:
                chunks = [result[i:i+1900] for i in range(0, len(result), 1900)]
                for chunk in chunks:
                    await interaction.followup.send(chunk)
            else:
                await interaction.followup.send(result)
                    
        except Exception as e:
            if not interaction.response.is_done():
                await interaction.response.send_message(f"Error during topics analysis: {str(e)}")
            else:
                await interaction.followup.send(f"Error during topics analysis: {str(e)}")

    @topics_analysis.autocomplete('channel')
    async def topics_analysis_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> List[app_commands.Choice[str]]:
        """Autocomplete for topics analysis channel filter"""
        return await self.channel_autocomplete(interaction, current)

    @app_commands.command(
        name="help_analysis",
        description="Show available analysis commands and their usage"
    )
    async def help_analysis(self, interaction: discord.Interaction):
        """Show help for analysis commands"""
        help_text = """
**🧠 Discord Analysis Bot Commands**

**🎯 Recommended Commands (with autocomplete):**
• `/channel_analysis` - Analyze a specific channel (autocomplete available)
• `/user_analysis` - Analyze a specific user (autocomplete available)  
• `/topics_analysis` - Analyze discussion topics, optionally by channel (autocomplete available)

**📊 General Analysis Commands:**
• `/analyze wordfreq` - Update word frequency statistics
• `/analyze userstats` - Update user activity statistics
• `/analyze temporal` - Analyze temporal patterns

**📋 Utility Commands:**
• `/list_users` - Show all available users
• `/list_channels` - Show all available channels
• `/help_analysis` - Show this help message

**💡 Pro Tips:**
- Use the dedicated commands (`/channel_analysis`, `/user_analysis`, `/topics_analysis`) for the best experience
- These commands have autocomplete - just start typing and select from the dropdown!
- The general `/analyze` command is available but dedicated commands work better
        """
        await interaction.response.send_message(help_text)
        
    async def cog_unload(self):
        """Cleanup when the cog is unloaded"""
        if hasattr(self, 'analyzer'):
            try:
                await self.analyzer.close()
                print("Analysis commands cleaned up successfully")
            except Exception as e:
                print(f"Error cleaning up analysis commands: {str(e)}")

async def setup(bot):
    """Add the analysis commands to the bot"""
    try:
        await bot.add_cog(AnalysisCommands(bot))
        print('Analysis commands loaded successfully!')
    except Exception as e:
        print(f'Error loading analysis commands: {str(e)}')
        raise