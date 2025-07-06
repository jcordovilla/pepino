"""
Discord Analysis Commands:
"""

import logging
import traceback
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import Counter
import os

import discord
from discord import app_commands
from discord.ext import commands

from pepino.analysis.service import AnalysisService
from .mixins import ComprehensiveCommandMixin

logger = logging.getLogger(__name__)


class AnalysisCommands(ComprehensiveCommandMixin, commands.Cog):
    """
     Analysis Commands for Discord Bot
    
    Features:
    - Template-based report generation
    - Service layer integration
    - Performance monitoring
    - Comprehensive error handling
    - Parallel analysis operations when possible
    """

    def __init__(self, bot):
        super().__init__()
        self.bot = bot
        # Cache for autocomplete data to improve performance
        self._channel_cache = None
        self._user_cache = None
        self._cache_timestamp = 0
        self._cache_duration = 300  # 5 minutes
        logger.info("AnalysisCommands initialized with service layer integration")
    
    async def cog_unload(self):
        """Clean up resources when cog is unloaded."""
        await self.cleanup_thread_pool()
        logger.info("AnalysisCommands unloaded")
    
    def _get_cached_channels(self):
        """Get cached channel list or fetch fresh data."""
        import time
        current_time = time.time()
        
        if (self._channel_cache is None or 
            current_time - self._cache_timestamp > self._cache_duration):
            try:
                from pepino.data.database.manager import DatabaseManager
                from pepino.data.repositories.channel_repository import ChannelRepository
                
                db_manager = DatabaseManager()
                channel_repo = ChannelRepository(db_manager)
                self._channel_cache = channel_repo.get_available_channels(limit=50)
                self._cache_timestamp = current_time
                logger.debug("Refreshed channel cache")
            except Exception as e:
                logger.error(f"Failed to refresh channel cache: {e}")
                # Return empty list if cache refresh fails
                return []
        
        return self._channel_cache or []
    
    def _get_cached_users(self):
        """Get cached user list or fetch fresh data."""
        import time
        current_time = time.time()
        
        if (self._user_cache is None or 
            current_time - self._cache_timestamp > self._cache_duration):
            try:
                from pepino.data.database.manager import DatabaseManager
                from pepino.data.repositories.user_repository import UserRepository
                
                db_manager = DatabaseManager()
                user_repo = UserRepository(db_manager)
                users = user_repo.get_available_users(limit=50)
                self._user_cache = [user for user in users if user and user.strip()]
                self._cache_timestamp = current_time
                logger.debug("Refreshed user cache")
            except Exception as e:
                logger.error(f"Failed to refresh user cache: {e}")
                # Return empty list if cache refresh fails
                return []
        
        return self._user_cache or []
    
    # Autocomplete functions
    async def channel_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> List[app_commands.Choice[str]]:
        """Autocomplete function for channel names with improved filtering."""
        try:
            # Use cached data for speed
            channels = self._get_cached_channels()
            
            # Enhanced filtering logic
            if current:
                current_lower = current.lower()
                
                # Priority 1: Exact matches (case insensitive)
                exact_matches = [ch for ch in channels if ch.lower() == current_lower]
                
                # Priority 2: Starts with the input
                starts_with = [ch for ch in channels if ch.lower().startswith(current_lower) and ch.lower() != current_lower]
                
                # Priority 3: Contains the input anywhere
                contains = [ch for ch in channels if current_lower in ch.lower() and not ch.lower().startswith(current_lower)]
                
                # Combine with priority order and limit to Discord's 25 choice limit
                filtered_channels = (exact_matches + starts_with + contains)[:25]
            else:
                # No input - return first 25 channels (alphabetically sorted)
                filtered_channels = sorted(channels)[:25]
            
            return [
                app_commands.Choice(name=channel, value=channel)
                for channel in filtered_channels
            ]
            
        except Exception as e:
            logger.error(f"Channel autocomplete failed: {e}")
            return []  # Return empty list instead of error choice

    async def user_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> List[app_commands.Choice[str]]:
        """Autocomplete function for usernames with improved filtering."""
        try:
            # Use cached data for speed
            valid_users = self._get_cached_users()

            # Enhanced filtering logic
            if current:
                current_lower = current.lower()
                
                # Priority 1: Exact matches (case insensitive)
                exact_matches = [user for user in valid_users if user.lower() == current_lower]
                
                # Priority 2: Starts with the input
                starts_with = [user for user in valid_users if user.lower().startswith(current_lower) and user.lower() != current_lower]
                
                # Priority 3: Contains the input anywhere
                contains = [user for user in valid_users if current_lower in user.lower() and not user.lower().startswith(current_lower)]
                
                # Combine with priority order and limit to Discord's 25 choice limit
                filtered_users = (exact_matches + starts_with + contains)[:25]
            else:
                # No input - return first 25 users (alphabetically sorted)
                filtered_users = sorted(valid_users)[:25]

            return [
                app_commands.Choice(name=user, value=user)
                for user in filtered_users
            ]

        except Exception as e:
            logger.error(f"User autocomplete failed: {e}")
            return []  # Return empty list instead of error choice
    
    @app_commands.command(
        name="pulsecheck", 
        description="Show weekly channel analysis in Discord format"
    )
    @app_commands.describe(
        channel_name="Optional: Channel name to analyze (leave empty for all channels summary)",
        days_back="Optional: Number of days back from end date (default: 7)",
        end_date="Optional: End date for analysis (format: YYYY-MM-DD, default: today)"
    )
    @app_commands.autocomplete(channel_name=channel_autocomplete)
    async def analyze_channels(
        self,
        interaction: discord.Interaction,
        channel_name: Optional[str] = None,
        days_back: int = 7,
        end_date: str = None
    ):
        """Analyze channel activity with template integration."""
        try:
            await interaction.response.defer(thinking=True)
        except discord.errors.NotFound:
            # Interaction already timed out
            logger.warning("Interaction timed out before defer")
            return
        except Exception as e:
            logger.error(f"Failed to defer interaction: {e}")
            return
        
        try:
            # Run analysis in thread pool with timeout to avoid Discord interaction timeout
            result = await asyncio.wait_for(
                self.run_sync_in_thread(
                    self._sync_channel_analysis,
                    channel_name,
                    days_back,
                    end_date
                ),
                timeout=25.0  # Discord interactions timeout at 15 seconds, give some buffer
            )
            
            # Send result
            await self._send_long_message_slash(interaction, result)
            
        except asyncio.TimeoutError:
            error_msg = "❌ Analysis timed out. Please try with a smaller date range or fewer channels."
            logger.error("Channel analysis timed out")
            try:
                await interaction.followup.send(error_msg, ephemeral=True)
            except:
                await interaction.followup.send("❌ Analysis timed out", ephemeral=True)
        except Exception as e:
            error_msg = f"❌ Channel analysis failed: {str(e)}"
            logger.error(f"Channel analysis error: {e}")
            logger.error(traceback.format_exc())
            
            try:
                await interaction.followup.send(error_msg, ephemeral=True)
            except:
                await interaction.followup.send("❌ Analysis failed", ephemeral=True)
    
    def _sync_channel_analysis(self, channel_name: str, days_back: int, end_date: str) -> str:
        """Synchronous channel analysis using service factory."""
        try:
            from pepino.analysis.service import analysis_service
            
            with analysis_service() as context:
                # Use the unified pulsecheck method that handles both single channel and all channels
                return context.pulsecheck(
                    channel_name=channel_name, 
                    days_back=days_back, 
                    end_date=datetime.fromisoformat(end_date) if end_date else None,
                    output_format="md"
                )
                
        except Exception as e:
            logger.error(f"Error in channel analysis: {e}")
            return f"❌ Analysis failed: {str(e)}"
    
    @app_commands.command(name="top_contributors", description="Show top contributors analysis in Discord format")
    @app_commands.describe(
        limit="Number of top users to show (default: 10)",
        days="Number of days to look back (default: 30)"
    )
    async def top_users(
        self, 
        interaction: discord.Interaction, 
        limit: int = 10,
        days: int = 30
    ):
        """Show top users with template integration."""
        try:
            await interaction.response.defer(thinking=True)
        except discord.errors.NotFound:
            logger.warning("Interaction timed out before defer")
            return
        except Exception as e:
            logger.error(f"Failed to defer interaction: {e}")
            return
        
        try:
            # Run analysis in thread pool with timeout
            result = await asyncio.wait_for(
                self.run_sync_in_thread(
                    self._sync_top_users,
                    limit,
                    days
                ),
                timeout=25.0
            )
            
            # Send result
            await self._send_long_message_slash(interaction, result)
            
        except asyncio.TimeoutError:
            error_msg = "❌ Analysis timed out. Please try with a smaller date range."
            logger.error("Top users analysis timed out")
            try:
                await interaction.followup.send(error_msg, ephemeral=True)
            except:
                await interaction.followup.send("❌ Analysis timed out", ephemeral=True)
        except Exception as e:
            error_msg = f"❌ Top users analysis failed: {str(e)}"
            logger.error(f"Top users analysis error: {e}")
            logger.error(traceback.format_exc())
            
            try:
                await interaction.followup.send(error_msg, ephemeral=True)
            except:
                await interaction.followup.send("❌ Analysis failed", ephemeral=True)
    
    def _sync_top_users(self, limit: int, days: int) -> str:
        """Synchronous top users analysis using service factory."""
        try:
            from pepino.analysis.service import analysis_service
            from datetime import timedelta
            
            with analysis_service() as context:
                # Get contributors data
                contributors = context.data_facade.user_repository.get_top_users(limit=limit, days_back=days)
                messages = context.data_facade.message_repository.get_recent_messages(limit=10000, days_back=days)
                
                # Build a lookup: author_id -> most recent message (with display name)
                most_recent_message = {}
                for msg in messages:
                    author_id = msg.get('author_id') or msg.get('author_name')
                    if not author_id:
                        continue
                    if author_id not in most_recent_message or msg['timestamp'] > most_recent_message[author_id]['timestamp']:
                        most_recent_message[author_id] = msg
                
                enhanced_contributors = []
                for contributor in contributors:
                    author_id = contributor.get('author_id') or contributor.get('author_name', 'Unknown')
                    display_name = contributor.get('author_name', 'Unknown')
                    if author_id in most_recent_message:
                        display_name = most_recent_message[author_id].get('author_display_name') or most_recent_message[author_id].get('author_name', 'Unknown')
                    channel_activity = context.data_facade.user_repository.get_user_channel_activity(
                        contributor.get('author_name', 'Unknown'), days, limit=10
                    )
                    top_messages = [
                        {
                            'jump_url': f"https://discord.com/channels/unknown/{author_id}",
                            'reply_count': 0
                        }
                    ]
                    enhanced_contributor = {
                        'name': display_name,
                        'message_count': contributor.get('message_count', 0),
                        'channel_activity': channel_activity,
                        'top_messages': top_messages
                    }
                    enhanced_contributors.append(enhanced_contributor)
                
                period = {
                    'start_date': (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                    'end_date': datetime.now().strftime('%Y-%m-%d')
                }
                
                return context.template_engine.render_template('outputs/discord/top_contributors.md.j2',
                                             contributors=enhanced_contributors,
                                             period=period,
                                             format_number=lambda v: f"{v:,}",
                                             now=datetime.now)
                
        except Exception as e:
            logger.error(f"Top users analysis failed: {e}")
            return f"❌ Top users analysis failed: {str(e)}"
    
    @app_commands.command(name="top_channels", description="Show top channels summary report")
    @app_commands.describe(
        limit="Number of top channels to show (default: 5)",
        days_back="Number of days to look back (default: 7)"
    )
    async def top_channels(
        self, 
        interaction: discord.Interaction, 
        limit: int = 5,
        days_back: int = 7
    ):
        """Show top channels summary with template integration."""
        try:
            await interaction.response.defer(thinking=True)
        except discord.errors.NotFound:
            logger.warning("Interaction timed out before defer")
            return
        except Exception as e:
            logger.error(f"Failed to defer interaction: {e}")
            return
        
        try:
            # Run analysis in thread pool with timeout
            result = await asyncio.wait_for(
                self.run_sync_in_thread(
                    self._sync_top_channels,
                    limit,
                    days_back
                ),
                timeout=25.0
            )
            
            # Send result
            await self._send_long_message_slash(interaction, result)
            
        except asyncio.TimeoutError:
            error_msg = "❌ Analysis timed out. Please try with a smaller date range."
            logger.error("Top channels analysis timed out")
            try:
                await interaction.followup.send(error_msg, ephemeral=True)
            except:
                await interaction.followup.send("❌ Analysis timed out", ephemeral=True)
        except Exception as e:
            error_msg = f"❌ Top channels analysis failed: {str(e)}"
            logger.error(f"Top channels analysis error: {e}")
            logger.error(traceback.format_exc())
            
            try:
                await interaction.followup.send(error_msg, ephemeral=True)
            except:
                await interaction.followup.send("❌ Analysis failed", ephemeral=True)
    
    def _sync_top_channels(self, limit: int, days_back: int) -> str:
        """Synchronous top channels analysis using service factory."""
        try:
            from pepino.analysis.service import analysis_service
            from datetime import timedelta
            
            with analysis_service() as context:
                message_analyzer = context.analyzers.get('message')
                user_analyzer = context.analyzers.get('weekly_user')
                
                if not message_analyzer or not user_analyzer:
                    return "❌ Analysis failed: Required analyzers not available"
                
                # Get all channels
                channels = context.data_facade.message_repository.get_all_channels()
                
                # Analyze each channel and collect data
                channel_data = []
                total_messages = 0
                total_active_users = set()
                increasing_channels = []
                decreasing_channels = []
                
                for channel_name in channels:
                    try:
                        # Get channel analysis
                        message_analysis = message_analyzer.analyze_weekly_messages(channel_name, days_back)
                        user_analysis = user_analyzer.analyze_weekly_users(channel_name, days_back)
                        total_members = context.data_facade.channel_repository.get_channel_human_member_count(channel_name)
                        
                        # Skip channels with no activity
                        channel_messages = message_analysis.get('statistics', {}).get('total_messages', 0)
                        if channel_messages == 0:
                            continue
                        
                        # Get top contributors
                        top_contributors = context.data_facade.user_repository.get_top_users(
                            limit=3, days_back=days_back, channel_name=channel_name
                        )
                        
                        # Calculate participation rate
                        active_members = len(user_analysis.get('top_users', []))
                        participation_rate = round((active_members / total_members * 100) if total_members > 0 else 0, 1)
                        
                        # Determine trend
                        trend_percentage = message_analysis.get('trend', {}).get('percentage_change', 0)
                        trend_direction = "increasing" if trend_percentage > 0 else "decreasing" if trend_percentage < 0 else "unchanged"
                        
                        if trend_percentage > 0:
                            increasing_channels.append(channel_name)
                        elif trend_percentage < 0:
                            decreasing_channels.append(channel_name)
                        
                        # Add to totals
                        total_messages += channel_messages
                        total_active_users.update([user.get('author_name') for user in top_contributors])
                        
                        channel_data.append({
                            'name': channel_name,
                            'message_count': channel_messages,
                            'active_users': active_members,
                            'total_members': total_members,
                            'participation_rate': participation_rate,
                            'trend_direction': trend_direction,
                            'trend_percentage': trend_percentage,
                            'top_contributors': top_contributors[:3]
                        })
                        
                    except Exception as e:
                        logger.debug(f"Could not analyze channel {channel_name}: {e}")
                        continue
                
                # Sort by message count and limit
                channel_data.sort(key=lambda x: x['message_count'], reverse=True)
                channel_data = channel_data[:limit]
                
                # Prepare summary data
                summary = {
                    'total_channels': len(channels),
                    'active_channels': len(channel_data),
                    'total_messages': total_messages,
                    'total_active_users': len(total_active_users),
                    'increasing_channels': len(increasing_channels),
                    'decreasing_channels': len(decreasing_channels),
                    'period': {
                        'start_date': (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                        'end_date': datetime.now().strftime('%Y-%m-%d')
                    }
                }
                
                return context.template_engine.render_template('outputs/discord/top_channels.md.j2',
                                             channels=channel_data,
                                             summary=summary,
                                             format_number=lambda v: f"{v:,}",
                                             now=datetime.now)
                
        except Exception as e:
            logger.error(f"Top channels analysis failed: {e}")
            return f"❌ Top channels analysis failed: {str(e)}"
    
    @app_commands.command(name="list_channels", description="List all available channels")
    async def list_channels(self, interaction: discord.Interaction):
        """List channels with template integration."""
        try:
            await interaction.response.defer(thinking=True)
        except discord.errors.NotFound:
            logger.warning("Interaction timed out before defer")
            return
        except Exception as e:
            logger.error(f"Failed to defer interaction: {e}")
            return
        
        try:
            # Run analysis in thread pool with timeout
            result = await asyncio.wait_for(
                self.run_sync_in_thread(
                    self._sync_get_available_channels
                ),
                timeout=25.0
            )
            
            # Send result
            await self._send_long_message_slash(interaction, result)
            
        except asyncio.TimeoutError:
            error_msg = "❌ Channel list timed out. Please try again."
            logger.error("Channel list timed out")
            try:
                await interaction.followup.send(error_msg, ephemeral=True)
            except:
                await interaction.followup.send("❌ List timed out", ephemeral=True)
        except Exception as e:
            error_msg = f"❌ Channel list failed: {str(e)}"
            logger.error(f"Channel list error: {e}")
            logger.error(traceback.format_exc())
            
            try:
                await interaction.followup.send(error_msg, ephemeral=True)
            except:
                await interaction.followup.send("❌ List failed", ephemeral=True)
    
    def _sync_get_available_channels(self):
        """Synchronous channel list using service factory."""
        try:
            from pepino.analysis.service import analysis_service
            
            with analysis_service() as context:
                channels = context.data_facade.channel_repository.get_available_channels()
                
                # Get stats for each channel
                channel_data = []
                for channel_name in channels:
                    try:
                        stats = context.data_facade.channel_repository.get_channel_message_statistics(channel_name)
                        channel_data.append({
                            'channel_name': channel_name,
                            'name': channel_name,
                            'message_count': stats.get('total_messages', 0) if stats else 0,
                            'unique_users': stats.get('unique_users', 0) if stats else 0,
                            'avg_message_length': stats.get('avg_message_length', 0.0) if stats else 0.0,
                        })
                    except:
                        channel_data.append({
                            'channel_name': channel_name,
                            'name': channel_name,
                            'message_count': 0,
                            'unique_users': 0,
                            'avg_message_length': 0.0,
                        })
                
                return context.template_engine.render_template('outputs/discord/channel_list.md.j2',
                                             items=channel_data,
                                             total_count=len(channels),
                                             showing_count=len(channels),
                                             has_more=False,
                                             format_number=lambda v: f"{v:,}",
                                             now=datetime.now)
                
        except Exception as e:
            logger.error(f"Channel list failed: {e}")
            return f"❌ Channel list failed: {str(e)}"
    

    

    

    

    

    

    
    async def _send_long_message(self, ctx: commands.Context, content: str):
        """Send long message with pagination if needed."""
        if len(content) <= 2000:
            await ctx.send(content)
        else:
            # Split into chunks
            chunks = [content[i:i+1900] for i in range(0, len(content), 1900)]
            
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await ctx.send(chunk)
                else:
                    await ctx.send(f"*Continued...*\n{chunk}")
    
    async def _send_long_message_slash(self, interaction: discord.Interaction, content: str):
        """Send long message via slash command with pagination if needed."""
        try:
            if len(content) <= 2000:
                await interaction.followup.send(content)
            else:
                # Split into chunks
                chunks = [content[i:i+1900] for i in range(0, len(content), 1900)]
                
                for i, chunk in enumerate(chunks):
                    if i == 0:
                        await interaction.followup.send(chunk)
                    else:
                        await interaction.followup.send(f"*Continued...*\n{chunk}")
        except discord.errors.NotFound:
            logger.warning("Interaction not found when sending response")
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            try:
                await interaction.followup.send("❌ Failed to send analysis result", ephemeral=True)
            except:
                pass  # Give up if we can't even send the error message


async def setup(bot):
    """Setup the analysis commands cog."""
    await bot.add_cog(AnalysisCommands(bot))


async def teardown(bot):
    """Teardown the analysis commands cog."""
    await bot.remove_cog("AnalysisCommands") 