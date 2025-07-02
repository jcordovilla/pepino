import discord
from discord.ext import commands
from discord import app_commands
from models import DiscordBotAnalyzer
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
        name="pepino_list_users",
        description="List all available users for analysis"
    )
    @app_commands.describe(
        limit="Number of users to show (default: 50, use 'all' or 999 for all users)"
    )
    async def list_users(self, interaction: discord.Interaction, limit: int = 50):
        """List all available users for analysis"""
        try:
            await interaction.response.defer()
            try:
                await self.analyzer.initialize()
            except Exception as e:
                await interaction.followup.send(f"Error initializing database: {str(e)}")
                return
            
            users = await self.analyzer.get_available_users()
            if not users:
                await interaction.followup.send("No users found in the database.")
                return
            
            total_users = len(users)
            
            # Handle 'all users' request
            if limit >= 999 or limit >= total_users:
                users_to_show = users
                show_all = True
            else:
                # Validate limit
                if limit < 1:
                    limit = 50
                users_to_show = users[:limit]
                show_all = False
            
            # Split users into chunks that fit Discord's message limit
            def chunk_users(user_list, max_length=1800):
                chunks = []
                current_chunk = []
                current_length = 0
                
                for user in user_list:
                    user_line = f"• {user}\n"
                    if current_length + len(user_line) > max_length and current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = [user]
                        current_length = len(user_line)
                    else:
                        current_chunk.append(user)
                        current_length += len(user_line)
                
                if current_chunk:
                    chunks.append(current_chunk)
                
                return chunks
            
            user_chunks = chunk_users(users_to_show)
            
            # Send the chunks
            for i, chunk in enumerate(user_chunks):
                if i == 0:
                    # First message with header
                    if show_all:
                        header = f"**All Available Users ({total_users} total):**\n"
                    else:
                        header = f"**Available Users ({len(users_to_show)} of {total_users} total):**\n"
                else:
                    # Subsequent messages
                    header = f"**Users (continued - part {i+1}):**\n"
                
                user_list = "\n".join([f"• {u}" for u in chunk])
                message = header + user_list
                
                # Add footer to last message
                if i == len(user_chunks) - 1 and not show_all:
                    message += f"\n\n*💡 Tip: Use `/list_users 999` to see all {total_users} users, or use autocomplete in analysis commands.*"
                elif i == len(user_chunks) - 1 and show_all:
                    message += f"\n\n*💡 Tip: Use autocomplete in analysis commands to quickly find users.*"
                
                await interaction.followup.send(message)
                
        except Exception as e:
            await interaction.followup.send(f"Error listing users: {str(e)}")

    @app_commands.command(
        name="pepino_list_channels",
        description="List all available channels for analysis"
    )
    @app_commands.describe(
        limit="Number of channels to show (default: 25, max: 50)"
    )
    async def list_channels(self, interaction: discord.Interaction, limit: int = 25):
        """List all available channels for analysis"""
        try:
            await interaction.response.defer()
            try:
                await self.analyzer.initialize()
            except Exception as e:
                await interaction.followup.send(f"Error initializing database: {str(e)}")
                return
            
            # Remove the hard limit of 50
            if limit < 1:
                limit = 25
            
            channels = await self.analyzer.get_available_channels()
            if channels:
                total_channels = len(channels)
                limited_channels = channels[:limit] if limit < total_channels else channels
                
                # Create the channel list
                channel_list = "\n".join([f"• {c}" for c in limited_channels])
                
                # Check message length
                header = f"**Available Channels ({len(limited_channels)} of {total_channels} total):**\n"
                message = header + channel_list
                
                if len(message) > 1900:  # Leave some buffer
                    # Truncate the list to fit in message
                    truncated_channels = []
                    current_length = len(header)
                    
                    for channel in limited_channels:
                        channel_line = f"• {channel}\n"
                        if current_length + len(channel_line) > 1800:
                            break
                        truncated_channels.append(channel)
                        current_length += len(channel_line)
                    
                    channel_list = "\n".join([f"• {c}" for c in truncated_channels])
                    message = header + channel_list + f"\n\n*Showing {len(truncated_channels)} channels. Use autocomplete in analysis commands to find specific channels.*"
                
                await interaction.followup.send(message)
            else:
                await interaction.followup.send("No channels found in the database.")
        except Exception as e:
            await interaction.followup.send(f"Error listing channels: {str(e)}")

    async def channel_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> List[app_commands.Choice[str]]:
        """Autocomplete for channel names and forum threads with current Discord channel mapping"""
        try:
            print(f"Channel autocomplete called with current='{current}'")
            await self.analyzer.initialize()
            # Get all selectable channels and threads
            entries = await self.analyzer.get_selectable_channels_and_threads()
            # entries: List[(label, channel_name, thread_id)]
            # Filter based on current input
            if current:
                filtered = [e for e in entries if current.lower() in e[0].lower()]
            else:
                filtered = entries
            filtered = filtered[:25]
            # Store as 'label|||channel_name|||thread_id' for value
            choices = [
                app_commands.Choice(
                    name=label,
                    value=f"{label}|||{ch}|||{tid if tid else ''}"
                ) for (label, ch, tid) in filtered if label
            ]
            print(f"Returning {len(choices)} choices")
            return choices
        except Exception as e:
            print(f"Error in channel autocomplete: {e}")
            import traceback
            traceback.print_exc()
            return []

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
        name="pepino_channel_analysis",
        description="Analyze a specific Discord channel or forum thread"
    )
    @app_commands.describe(
        channel="Channel or thread to analyze (autocomplete)"
    )
    async def channel_analysis(
        self,
        interaction: discord.Interaction,
        channel: str
    ):
        """Analyze a specific Discord channel or forum thread"""
        try:
            print(f"Channel analysis command called with channel='{channel}'")
            await self.analyzer.initialize()
            await interaction.response.defer()
            # Parse channel and thread info
            if '|||' in channel:
                label, channel_name, thread_id = channel.split('|||', 2)
                thread_id = thread_id if thread_id else None
            else:
                channel_name = channel
                thread_id = None
            result = await self.analyzer.get_channel_insights(channel_name, thread_id=thread_id)
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
        """Autocomplete for channel analysis (channels and threads)"""
        return await self.channel_autocomplete(interaction, current)

    @app_commands.command(
        name="pepino_user_analysis", 
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
            
            # Handle both text-only and (text, chart_path) tuple results
            if isinstance(result, tuple):
                text_result, chart_path = result
                
                # Send the text analysis
                if len(text_result) > 2000:
                    chunks = [text_result[i:i+1900] for i in range(0, len(text_result), 1900)]
                    for chunk in chunks:
                        await interaction.followup.send(chunk)
                else:
                    await interaction.followup.send(text_result)
                
                # Send the chart as an image
                if chart_path and os.path.exists(chart_path):
                    with open(chart_path, 'rb') as f:
                        chart_file = discord.File(f, filename='user_activity_chart.png')
                        await interaction.followup.send("📊 **Daily Activity Chart (Last 30 Days):**", file=chart_file)
            else:
                # Handle text-only result (fallback)
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
        name="pepino_topics_analysis",
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
        name="pepino_top_users", 
        description="Top 10 most active users with statistics and main topics"
    )
    async def top_users(self, interaction: discord.Interaction):
        """Top 10 most active users with statistics and main topics"""
        try:
            await self.analyzer.initialize()
            await interaction.response.defer()
            
            result = await self.analyzer.update_user_statistics()
            
            if len(result) > 2000:
                chunks = [result[i:i+1900] for i in range(0, len(result), 1900)]
                for chunk in chunks:
                    await interaction.followup.send(chunk)
            else:
                await interaction.followup.send(result)
                
        except Exception as e:
            if not interaction.response.is_done():
                await interaction.response.send_message(f"Error during user statistics analysis: {str(e)}")
            else:
                await interaction.followup.send(f"Error during user statistics analysis: {str(e)}")

    @app_commands.command(
        name="pepino_activity_trends",
        description="Server activity trends and patterns over the past 30 days"
    )
    async def activity_trends(self, interaction: discord.Interaction):
        """Server activity trends and patterns over the past 30 days"""
        try:
            await self.analyzer.initialize()
            await interaction.response.defer()
            
            result = await self.analyzer.update_temporal_stats()
            
            # Handle both text-only and (text, chart_path) tuple results
            if isinstance(result, tuple):
                text_result, chart_path = result
                
                # Send the text analysis in chunks if needed
                if len(text_result) > 2000:
                    chunks = [text_result[i:i+1900] for i in range(0, len(text_result), 1900)]
                    for chunk in chunks:
                        await interaction.followup.send(chunk)
                else:
                    await interaction.followup.send(text_result)
                
                # Send the chart as an image
                if chart_path and os.path.exists(chart_path):
                    with open(chart_path, 'rb') as f:
                        chart_file = discord.File(f, filename='activity_trends_chart.png')
                        await interaction.followup.send("📊 **Activity Trends Chart (Last 30 Days):**", file=chart_file)
            else:
                # Handle text-only result (fallback)
                if len(result) > 2000:
                    chunks = [result[i:i+1900] for i in range(0, len(result), 1900)]
                    for chunk in chunks:
                        await interaction.followup.send(chunk)
                else:
                    await interaction.followup.send(result)
                    
        except Exception as e:
            if not interaction.response.is_done():
                await interaction.response.send_message(f"Error during activity trends analysis: {str(e)}")
            else:
                await interaction.followup.send(f"Error during activity trends analysis: {str(e)}")

    @app_commands.command(
        name="pepino_help",
        description="Show available analysis commands and their usage"
    )
    async def help_analysis(self, interaction: discord.Interaction):
        """Show help for analysis commands"""
        help_text = """
**🧠 Discord Analysis Bot Commands**

**🎯 Analysis Commands (with autocomplete):**
• `/pepino_channel_analysis` - Detailed channel insights with key topics & concepts
• `/pepino_user_analysis` - User insights with contribution analysis & key topics
• `/pepino_topics_analysis` - Topic analysis with trends, optionally filtered by channel

**📊 Enhanced Statistical Analysis Commands:**
• `/pepino_top_users` - Top 10 most active users with statistics and main topics
• `/pepino_activity_trends` - **Enhanced** server activity trends with comprehensive analytics, semantic analysis, and activity charts

**📋 Utility Commands:**
• `/pepino_list_users` - Show all available users
• `/pepino_list_channels` - Show all available channels
• `/pepino_help` - Show this help message

**💡 Pro Tips:**
- Commands with autocomplete make it easy to find channels and users - just start typing!
- All commands are now dedicated and focused on specific analysis types
- Use `/list_users` or `/list_channels` if you need to see what's available
- The enhanced `/activity_trends` command now includes:
  - Overall server statistics and trends
  - Activity patterns by hour and day of week
  - Top channels and contributors
  - Server-wide semantic analysis
  - Activity visualization charts
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