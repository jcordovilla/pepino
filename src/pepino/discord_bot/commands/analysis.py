"""
Discord Analysis Commands:
"""

import logging
import traceback
import asyncio
from typing import List, Optional, Literal
from datetime import datetime
from cachetools import TTLCache

import discord
from discord import app_commands
from discord.ext import commands

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
        
        # Load settings for configuration
        from pepino.config import Settings
        self.settings = Settings()
        
        # TTL caches with 1-hour expiration
        self._channel_cache = TTLCache(maxsize=1, ttl=3600)  # 1 hour
        self._user_cache = TTLCache(maxsize=1, ttl=3600)     # 1 hour
        
        logger.info("AnalysisCommands initialized with TTL cache and service layer integration")
    
    async def cog_unload(self):
        """Clean up resources when cog is unloaded."""
        await self.cleanup_thread_pool()
        logger.info("AnalysisCommands unloaded")
    
    def clear_caches(self):
        """Manually clear all caches."""
        self._channel_cache.clear()
        self._user_cache.clear()
        logger.info("Caches cleared manually")
    
    def get_cache_status(self):
        """Get cache status for monitoring."""
        return {
            'channels': {
                'cached': 'channels' in self._channel_cache,
                'count': len(self._channel_cache.get('channels', [])),
                'cache_size': len(self._channel_cache)
            },
            'users': {
                'cached': 'users' in self._user_cache,
                'count': len(self._user_cache.get('users', [])),
                'cache_size': len(self._user_cache)
            }
        }
    
    def _get_cached_data(self, data_type: Literal["channels", "users"]):
        """Generic cached data fetcher with TTL."""
        cache = self._channel_cache if data_type == "channels" else self._user_cache
        cache_key = data_type
        
        # Try to get from cache first
        if cache_key in cache:
            return cache[cache_key]
        
        # Cache miss or expired, fetch fresh data
        try:
            from pepino.data.database.manager import DatabaseManager
            
            db_manager = DatabaseManager(self.settings.database_sqlite_path)
            
            if data_type == "channels":
                from pepino.data.repositories.channel_repository import ChannelRepository
                repo = ChannelRepository(db_manager)
                data = repo.get_available_channels(limit=self.settings.discord_bot_cache_max_items)
            else:  # users
                from pepino.data.repositories.user_repository import UserRepository
                repo = UserRepository(db_manager)
                users = repo.get_available_users(limit=self.settings.discord_bot_cache_max_items)
                data = [user for user in users if user and user.strip()]
            
            # Store in cache (TTL handled automatically)
            cache[cache_key] = data
            logger.debug(f"Refreshed {data_type} cache")
            return data
        except Exception as e:
            logger.error(f"Failed to refresh {data_type} cache: {e}")
            return []
    
    def _get_cached_channels(self):
        """Get cached channel list or fetch fresh data with TTL."""
        return self._get_cached_data("channels")
    
    def _get_cached_users(self):
        """Get cached user list or fetch fresh data with TTL."""
        return self._get_cached_data("users")
    
    def _generic_autocomplete(self, data_type: Literal["channels", "users"], current: str) -> List[app_commands.Choice[str]]:
        """Generic autocomplete function with improved filtering for any data type."""
        try:
            # Use cached data for speed
            data = self._get_cached_data(data_type)
            
            # Enhanced filtering logic
            if current:
                current_lower = current.lower()
                
                # Priority 1: Exact matches (case insensitive)
                exact_matches = [item for item in data if item.lower() == current_lower]
                
                # Priority 2: Starts with the input
                starts_with = [item for item in data if item.lower().startswith(current_lower) and item.lower() != current_lower]
                
                # Priority 3: Contains the input anywhere
                contains = [item for item in data if current_lower in item.lower() and not item.lower().startswith(current_lower)]
                
                # Combine with priority order and limit to Discord's 25 choice limit
                filtered_data = (exact_matches + starts_with + contains)[:25]
            else:
                # No input - return first 25 items (alphabetically sorted)
                filtered_data = sorted(data)[:25]
            
            return [
                app_commands.Choice(name=item, value=item)
                for item in filtered_data
            ]
            
        except Exception as e:
            logger.error(f"{data_type.capitalize()} autocomplete failed: {e}")
            return []  # Return empty list instead of error choice

    async def channel_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> List[app_commands.Choice[str]]:
        """Autocomplete function for channel names with improved filtering."""
        return self._generic_autocomplete("channels", current)

    async def user_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> List[app_commands.Choice[str]]:
        """Autocomplete function for usernames with improved filtering."""
        return self._generic_autocomplete("users", current)
    
    @app_commands.command(
        name="channel_analysis", 
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
        """Analyze channel activity."""
        # Input validation
        if days_back < 1 or days_back > 365:
            await interaction.response.send_message("❌ Days must be between 1 and 365", ephemeral=True)
            return
        
        if end_date:
            try:
                datetime.fromisoformat(end_date)
            except ValueError:
                await interaction.response.send_message("❌ Invalid date format. Use YYYY-MM-DD", ephemeral=True)
                return
        
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
                timeout=self.settings.discord_bot_interaction_timeout_seconds
            )
            
            # Send result
            await self._send_long_message_slash(interaction, result)
            
        except asyncio.TimeoutError:
            await self._handle_timeout_error(interaction, "Channel analysis")
        except Exception as e:
            await self._handle_generic_error(interaction, "Channel analysis", e)
    
    def _sync_channel_analysis(self, channel_name: str, days_back: int, end_date: str) -> str:
        """Synchronous channel analysis using service factory."""
        try:
            from pepino.analysis.service import analysis_service
            
            with analysis_service() as service:
                # Use the service's public interface
                return service.pulsecheck(
                    channel_name=channel_name, 
                    days_back=days_back, 
                    end_date=datetime.fromisoformat(end_date) if end_date else None,
                    output_format="discord"
                )
                
        except Exception as e:
            logger.error(f"Error in channel analysis: {e}")
            return f"❌ Analysis failed: {str(e)}"
    
    @app_commands.command(name="top_contributors", description="Show top contributors analysis")
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
        """Show top users."""
        # Input validation
        if limit < 1 or limit > 50:
            await interaction.response.send_message("❌ Limit must be between 1 and 50", ephemeral=True)
            return
        
        if days < 1 or days > 365:
            await interaction.response.send_message("❌ Days must be between 1 and 365", ephemeral=True)
            return
        
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
                timeout=self.settings.discord_bot_interaction_timeout_seconds
            )
            
            # Send result
            await self._send_long_message_slash(interaction, result)
            
        except asyncio.TimeoutError:
            await self._handle_timeout_error(interaction, "Top users analysis")
        except Exception as e:
            await self._handle_generic_error(interaction, "Top users analysis", e)
    
    def _sync_top_users(self, limit: int, days: int) -> str:
        """Synchronous top users analysis using service factory."""
        try:
            from pepino.analysis.service import analysis_service
            
            with analysis_service() as service:
                # Use the service's public interface
                return service.top_contributors(
                    limit=limit, 
                    days_back=days, 
                    output_format="discord"
                )
                
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
        """Show top channels summary."""
        # Input validation
        if limit < 1 or limit > 20:
            await interaction.response.send_message("❌ Limit must be between 1 and 20", ephemeral=True)
            return
        
        if days_back < 1 or days_back > 365:
            await interaction.response.send_message("❌ Days must be between 1 and 365", ephemeral=True)
            return
        
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
                timeout=self.settings.discord_bot_interaction_timeout_seconds
            )
            
            # Send result
            await self._send_long_message_slash(interaction, result)
            
        except asyncio.TimeoutError:
            await self._handle_timeout_error(interaction, "Top channels analysis")
        except Exception as e:
            await self._handle_generic_error(interaction, "Top channels analysis", e)
    
    def _sync_top_channels(self, limit: int, days_back: int) -> str:
        """Synchronous top channels analysis using service factory."""
        try:
            from pepino.analysis.service import analysis_service
            
            with analysis_service() as service:
                # Use the service's public interface
                return service.top_channels(
                    limit=limit, 
                    days_back=days_back, 
                    output_format="discord"
                )
                
        except Exception as e:
            logger.error(f"Top channels analysis failed: {e}")
            return f"❌ Top channels analysis failed: {str(e)}"
    
    @app_commands.command(name="list_channels", description="List all available channels")
    async def list_channels(self, interaction: discord.Interaction):
        """List channels."""
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
                timeout=self.settings.discord_bot_interaction_timeout_seconds
            )
            
            # Send result
            await self._send_long_message_slash(interaction, result)
            
        except asyncio.TimeoutError:
            await self._handle_timeout_error(interaction, "Channel list")
        except Exception as e:
            await self._handle_generic_error(interaction, "Channel list", e)
    
    def _sync_get_available_channels(self):
        """Synchronous channel list using service factory."""
        try:
            from pepino.analysis.service import analysis_service
            
            with analysis_service() as service:
                # Use the service's public interface
                return service.list_channels(output_format="discord")
                
        except Exception as e:
            logger.error(f"Channel list failed: {e}")
            return f"❌ Channel list failed: {str(e)}"

    @app_commands.command(name="database_stats", description="Show database statistics and health report")
    async def database_stats(self, interaction: discord.Interaction):
        """Show database statistics and health report."""
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
                    self._sync_database_stats
                ),
                timeout=self.settings.discord_bot_interaction_timeout_seconds
            )
            
            # Send result
            await self._send_long_message_slash(interaction, result)
            
        except asyncio.TimeoutError:
            await self._handle_timeout_error(interaction, "Database statistics")
        except Exception as e:
            await self._handle_generic_error(interaction, "Database statistics", e)
    
    def _sync_database_stats(self) -> str:
        """Synchronous database statistics using service factory."""
        try:
            from pepino.analysis.service import analysis_service
            
            with analysis_service() as service:
                # Use the service's public interface
                return service.database_stats(output_format="discord")
                
        except Exception as e:
            logger.error(f"Database statistics failed: {e}")
            return f"❌ Database statistics failed: {str(e)}"
    
    @app_commands.command(
        name="detailed_user_analysis",
        description="Show detailed user analysis (new system, unique template)"
    )
    @app_commands.describe(
        username="Username to analyze",
        days_back="Number of days to look back (default: 30)"
    )
    async def detailed_user_analysis(
        self,
        interaction: discord.Interaction,
        username: str,
        days_back: int = 30
    ):
        """Show detailed user analysis (new system, unique template)."""
        if days_back < 1 or days_back > 365:
            await interaction.response.send_message("❌ Days must be between 1 and 365", ephemeral=True)
            return
        try:
            await interaction.response.defer(thinking=True)
        except discord.errors.NotFound:
            logger.warning("Interaction timed out before defer")
            return
        except Exception as e:
            logger.error(f"Failed to defer interaction: {e}")
            return
        try:
            from pepino.analysis.service import analysis_service
            import asyncio
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: analysis_service().detailed_user_analysis(username, days_back, output_format="discord")
            )
            await self._send_long_message_slash(interaction, result)
        except asyncio.TimeoutError:
            await self._handle_timeout_error(interaction, "Detailed user analysis")
        except Exception as e:
            await self._handle_generic_error(interaction, "Detailed user analysis", e)
    
    @app_commands.command(
        name="detailed_topic_analysis",
        description="Show detailed topic analysis (new system, unique template)"
    )
    @app_commands.describe(
        channel_name="Optional: Channel name to analyze (leave empty for all channels)",
        n_topics="Number of topics to extract (default: 10)",
        days_back="Number of days to look back (optional)"
    )
    async def detailed_topic_analysis(
        self,
        interaction: discord.Interaction,
        channel_name: Optional[str] = None,
        n_topics: int = 10,
        days_back: Optional[int] = None
    ):
        """Show detailed topic analysis (new system, unique template)."""
        if n_topics < 1 or n_topics > 50:
            await interaction.response.send_message("❌ Number of topics must be between 1 and 50", ephemeral=True)
            return
        if days_back is not None and (days_back < 1 or days_back > 365):
            await interaction.response.send_message("❌ Days must be between 1 and 365", ephemeral=True)
            return
        try:
            await interaction.response.defer(thinking=True)
        except discord.errors.NotFound:
            logger.warning("Interaction timed out before defer")
            return
        except Exception as e:
            logger.error(f"Failed to defer interaction: {e}")
            return
        try:
            from pepino.analysis.service import analysis_service
            import asyncio
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: analysis_service().detailed_topic_analysis(channel_name, n_topics, days_back, output_format="discord")
            )
            await self._send_long_message_slash(interaction, result)
        except asyncio.TimeoutError:
            await self._handle_timeout_error(interaction, "Detailed topic analysis")
        except Exception as e:
            await self._handle_generic_error(interaction, "Detailed topic analysis", e)

    @app_commands.command(
        name="detailed_temporal_analysis",
        description="Show detailed temporal analysis (new system, unique template)"
    )
    @app_commands.describe(
        channel_name="Optional: Channel name to analyze (leave empty for all channels)",
        days_back="Number of days to look back (optional)",
        granularity="Time granularity (hourly, daily, weekly, default: daily)"
    )
    @app_commands.autocomplete(channel_name=channel_autocomplete)
    async def detailed_temporal_analysis(
        self,
        interaction: discord.Interaction,
        channel_name: Optional[str] = None,
        days_back: Optional[int] = None,
        granularity: str = "daily"
    ):
        """Show detailed temporal analysis (new system, unique template)."""
        if granularity not in ["hourly", "daily", "weekly"]:
            await interaction.response.send_message("❌ Granularity must be hourly, daily, or weekly", ephemeral=True)
            return
        if days_back is not None and (days_back < 1 or days_back > 365):
            await interaction.response.send_message("❌ Days must be between 1 and 365", ephemeral=True)
            return
        try:
            await interaction.response.defer(thinking=True)
        except discord.errors.NotFound:
            logger.warning("Interaction timed out before defer")
            return
        except Exception as e:
            logger.error(f"Failed to defer interaction: {e}")
            return
        try:
            from pepino.analysis.service import analysis_service
            import asyncio
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: analysis_service().detailed_temporal_analysis(channel_name, days_back, granularity, output_format="discord")
            )
            await self._send_long_message_slash(interaction, result)
        except asyncio.TimeoutError:
            await self._handle_timeout_error(interaction, "Detailed temporal analysis")
        except Exception as e:
            await self._handle_generic_error(interaction, "Detailed temporal analysis", e)

    @app_commands.command(
        name="activity_trends",
        description="Show activity trends analysis with chart generation"
    )
    @app_commands.describe(
        channel_name="Optional: Channel name to analyze (leave empty for all channels)",
        days_back="Number of days to look back (optional)"
    )
    @app_commands.autocomplete(channel_name=channel_autocomplete)
    async def activity_trends(
        self,
        interaction: discord.Interaction,
        channel_name: Optional[str] = None,
        days_back: Optional[int] = None
    ):
        """Show activity trends analysis with chart generation."""
        if days_back is not None and (days_back < 1 or days_back > 365):
            await interaction.response.send_message("❌ Days must be between 1 and 365", ephemeral=True)
            return
        try:
            await interaction.response.defer(thinking=True)
        except discord.errors.NotFound:
            logger.warning("Interaction timed out before defer")
            return
        except Exception as e:
            logger.error(f"Failed to defer interaction: {e}")
            return
        try:
            from pepino.analysis.service import analysis_service
            import asyncio
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: analysis_service().activity_trends_analysis(channel_name, days_back, output_format="discord")
            )
            await self._send_long_message_slash(interaction, result)
        except asyncio.TimeoutError:
            await self._handle_timeout_error(interaction, "Activity trends analysis")
        except Exception as e:
            await self._handle_generic_error(interaction, "Activity trends analysis", e)

    @app_commands.command(
        name="server_overview",
        description="Show comprehensive server overview analysis"
    )
    @app_commands.describe(
        days_back="Number of days to look back (optional, default: all time)"
    )
    async def server_overview(
        self,
        interaction: discord.Interaction,
        days_back: Optional[int] = None
    ):
        """Show comprehensive server overview analysis."""
        if days_back is not None and (days_back < 1 or days_back > 365):
            await interaction.response.send_message("❌ Days must be between 1 and 365", ephemeral=True)
            return
        try:
            await interaction.response.defer(thinking=True)
        except discord.errors.NotFound:
            logger.warning("Interaction timed out before defer")
            return
        except Exception as e:
            logger.error(f"Failed to defer interaction: {e}")
            return
        try:
            from pepino.analysis.service import analysis_service
            import asyncio
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: analysis_service().server_overview_analysis(days_back, output_format="discord")
            )
            await self._send_long_message_slash(interaction, result)
        except asyncio.TimeoutError:
            await self._handle_timeout_error(interaction, "Server overview analysis")
        except Exception as e:
            await self._handle_generic_error(interaction, "Server overview analysis", e)
    
    async def _handle_timeout_error(self, interaction: discord.Interaction, operation_name: str):
        """Handle timeout errors consistently."""
        error_msg = f"❌ {operation_name} timed out. Please try with a smaller date range."
        logger.error(f"{operation_name} timed out")
        try:
            await interaction.followup.send(error_msg, ephemeral=True)
        except:
            await interaction.followup.send("❌ Analysis timed out", ephemeral=True)
    
    async def _handle_validation_error(self, interaction: discord.Interaction, message: str):
        """Handle validation errors."""
        try:
            await interaction.followup.send(f"❌ {message}", ephemeral=True)
        except:
            await interaction.followup.send("❌ Invalid input", ephemeral=True)
    
    async def _handle_generic_error(self, interaction: discord.Interaction, operation_name: str, error: Exception):
        """Handle generic errors with proper logging."""
        error_msg = f"❌ {operation_name} failed: {str(error)}"
        logger.error(f"{operation_name} error: {error}")
        logger.error(traceback.format_exc())
        
        try:
            await interaction.followup.send(error_msg, ephemeral=True)
        except:
            await interaction.followup.send("❌ Analysis failed", ephemeral=True)
    
    async def _send_long_message_slash(self, interaction: discord.Interaction, content: str):
        """Send long message via slash command with pagination if needed."""
        try:
            if len(content) <= self.settings.discord_bot_message_character_limit:
                await interaction.followup.send(content)
            else:
                # Split into chunks
                chunks = [content[i:i+self.settings.discord_bot_message_chunk_size] for i in range(0, len(content), self.settings.discord_bot_message_chunk_size)]
                
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