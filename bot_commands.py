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
        description="Analyze Discord messages with various tasks"
    )
    @app_commands.describe(
        task="The analysis task to perform",
        args="Additional arguments for the task"
    )
    @app_commands.choices(task=[
        app_commands.Choice(name="wordfreq", value="wordfreq"),
        app_commands.Choice(name="userstats", value="userstats"),
        app_commands.Choice(name="channel", value="channel"),
        app_commands.Choice(name="topics", value="topics"),
        app_commands.Choice(name="temporal", value="temporal"),
        app_commands.Choice(name="user", value="user")
    ])
    async def analyze(
        self,
        interaction: discord.Interaction,
        task: str,
        args: str = None
    ):
        """Analyze Discord messages with various tasks"""
        try:
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
                if not args:
                    # Get list of channels for autocomplete
                    channels = await self.analyzer.get_available_channels()
                    channel_list = "\n".join([f"• {c}" for c in channels])
                    await interaction.followup.send(f"Please specify a channel name. Available channels:\n{channel_list}")
                    return
                result = await self.analyzer.get_channel_insights(args)
            elif task == "topics":
                result = await self.analyzer.analyze_topics()
            elif task == "temporal":
                result = await self.analyzer.update_temporal_stats()
            elif task == "user":
                if not args:
                    # Get list of users for autocomplete
                    users = await self.analyzer.get_available_users()
                    user_list = "\n".join([f"• {u}" for u in users])
                    await interaction.followup.send(f"Please specify a user name. Available users:\n{user_list}")
                    return
                result = await self.analyzer.get_user_insights(args)
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