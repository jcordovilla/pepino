"""
Discord Analysis Commands:
"""

import logging
import traceback
from typing import Dict, List, Optional, Any

import discord
from discord import app_commands
from discord.ext import commands

from pepino.templates.template_engine import TemplateEngine
from ...analysis.channel_analyzer import ChannelAnalyzer
from ...analysis.user_analyzer import UserAnalyzer
from ...analysis.topic_analyzer import TopicAnalyzer
from ...analysis.temporal_analyzer import TemporalAnalyzer
from .mixins import ComprehensiveCommandMixin

logger = logging.getLogger(__name__)


class AnalysisCommands(ComprehensiveCommandMixin, commands.Cog):
    """
     Analysis Commands for Discord Bot
    
    Features:
    - Template-based report generation
    - Sync analyzers with thread pool execution
    - Performance monitoring
    - Comprehensive error handling
    - Parallel analysis operations when possible
    """

    def __init__(self, bot):
        super().__init__()
        self.bot = bot
        
        # Database manager and settings will be initialized per command
        self.db_manager = None
        self.settings = None
        
        logger.info("AnalysisCommands initialized with template system and sync analyzers")
    
    def _get_database_manager(self):
        """Get database manager instance."""
        from pepino.data.database.manager import DatabaseManager
        return DatabaseManager("discord_messages.db")
    
    def _get_settings(self):
        """Get settings instance.""" 
        from pepino.data.config import Settings
        return Settings()
    
    def _create_template_engine(self, data_facade):
        """Create template engine with analyzer helpers and NLP capabilities."""
        from ...templates.template_engine import TemplateEngine
        from ...analysis.channel_analyzer import ChannelAnalyzer
        from ...analysis.user_analyzer import UserAnalyzer
        from ...analysis.topic_analyzer import TopicAnalyzer
        from ...analysis.temporal_analyzer import TemporalAnalyzer
        
        # Try to import NLP service, handle gracefully if unavailable
        try:
            from ...analysis.nlp_analyzer import NLPService
            nlp_service = NLPService()
        except ImportError:
            logger.warning("NLP service not available, templates will have limited NLP capabilities")
            nlp_service = None
        
        # Create analyzers
        analyzers = {
            'channel': ChannelAnalyzer(data_facade),
            'user': UserAnalyzer(data_facade),
            'topic': TopicAnalyzer(data_facade),
            'temporal': TemporalAnalyzer(data_facade)
        }
        
        return TemplateEngine(
            analyzers=analyzers,
            data_facade=data_facade,
            nlp_service=nlp_service
        )
    
    async def cog_unload(self):
        """Clean up resources when cog is unloaded."""
        await self.cleanup_thread_pool()
        logger.info("AnalysisCommands unloaded")
    
    # Autocomplete functions
    async def channel_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> List[app_commands.Choice[str]]:
        """Autocomplete function for channel names with improved filtering."""
        try:
            # Get available channels directly without thread pool for speed
            from ...analysis.data_facade import get_analysis_data_facade
            from ...data.config import Settings
            
            settings = Settings()
            with get_analysis_data_facade(base_filter=settings.base_filter) as facade:
                # Get more channels for better autocomplete experience
                channels = facade.channel_repository.get_available_channels(limit=200)
            
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
            # Get available users directly without thread pool for speed
            from ...analysis.data_facade import get_analysis_data_facade
            from ...data.config import Settings

            settings = Settings()
            with get_analysis_data_facade(base_filter=settings.base_filter) as facade:
                # Get more users for better autocomplete experience
                users = facade.user_repository.get_available_users(limit=200)

            # Filter out empty/None usernames
            valid_users = [user for user in users if user and user.strip()]

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
    
    @app_commands.command(name="channel_analysis", description="Analyze channel activity patterns and statistics")
    @app_commands.describe(channel_name="Channel name to analyze (leave empty for current channel)")
    @app_commands.autocomplete(channel_name=channel_autocomplete)
    async def channel_analysis(
        self, 
        interaction: discord.Interaction, 
        channel_name: Optional[str] = None
    ):
        """
        Analyze channel activity using templates.
        
        Usage: /channel_analysis [channel_name]
        
        If no channel_name provided, analyzes current channel.
        """
        
        try:
            # Defer the response immediately to avoid timeout
            await interaction.response.defer()

            # Determine target channel
            if not channel_name:
                if isinstance(interaction.channel, discord.DMChannel):
                    await interaction.followup.send("❌ Please specify a channel name when using this command in DMs.")
                    return
                channel_name = interaction.channel.name
            
            # Execute sync operation in thread pool with performance tracking
            result, exec_time = await self.execute_tracked_sync_operation(
                "channel_analysis",
                self._sync_channel_analysis,
                channel_name
            )
            
            # Send single comprehensive result
            if result:
                final_message = f"{result}\n\n✅ Analysis completed in {exec_time:.2f}s"
                await self._send_long_message_slash(interaction, final_message)
            else:
                await interaction.followup.send(f"❌ No data found for channel **{channel_name}**")

        except Exception as e:
            logger.error(f"Channel analysis failed: {e}")
            logger.error(traceback.format_exc())
            if not interaction.response.is_done():
                await interaction.response.send_message(f"❌ Analysis failed: {str(e)}")
            else:
                await interaction.followup.send(f"❌ Analysis failed: {str(e)}")
    
    def _sync_channel_analysis(self, channel_name: str) -> str:
        """Sync channel analysis using enhanced template system with analyzer helpers."""
        
        from ...analysis.data_facade import get_analysis_data_facade
        from ...analysis.channel_analyzer import ChannelAnalyzer
        from ...data.config import Settings
        
        settings = Settings()
        with get_analysis_data_facade(base_filter=settings.base_filter) as facade:
            # Create template engine with analyzer helpers
            template_engine = self._create_template_engine(facade)
            
            channel_analyzer = ChannelAnalyzer(facade)
            
            # Get channel analysis data
            analysis_result = channel_analyzer.analyze(channel_name=channel_name)
            
            if not analysis_result or not analysis_result.statistics:
                return None
            
            # Get recent messages for template context
            recent_messages = []
            try:
                # Use synchronous method that returns data directly
                messages_data = facade.message_repository.get_channel_messages(channel_name, days_back=7, limit=100)
                if messages_data:
                    recent_messages = [
                        {
                            'id': msg.get('id'),
                            'content': msg.get('content', ''),
                            'author': msg.get('username') or msg.get('author', ''),
                            'timestamp': msg.get('timestamp', ''),
                            'channel': msg.get('channel_name', channel_name)
                        }
                        for msg in messages_data
                    ]
            except Exception as e:
                logger.warning(f"Could not fetch messages for template context: {e}")
            
            # Prepare enhanced template data using correct ChannelAnalysisResponse attributes
            template_data = {
                'data': {
                    'channel_name': channel_name,
                    'date_range': {
                        'start': analysis_result.statistics.first_message or 'Unknown',
                        'end': analysis_result.statistics.last_message or 'Unknown'
                    },
                    'statistics': analysis_result.statistics,
                    'user_stats': analysis_result.top_users or [],
                    'peak_activity': analysis_result.peak_activity,
                    'engagement_metrics': analysis_result.engagement_metrics,
                    'health_metrics': analysis_result.health_metrics,
                    'content_analysis': {
                        'common_words': []  # This would be populated by topic analysis if needed
                    }
                },
                'analysis': analysis_result,
                'channel_name': channel_name,
                'top_users': analysis_result.top_users or [],
                'peak_activity': analysis_result.peak_activity,
                'engagement_metrics': analysis_result.engagement_metrics,
                'health_metrics': analysis_result.health_metrics,
                'recent_activity': analysis_result.recent_activity or []
            }
            
            # Render template with message context
            return template_engine.render_template(
                'outputs/discord/channel_analysis.md.j2',
                messages=recent_messages,
                **template_data
            )
    
    @app_commands.command(name="user_analysis", description="Analyze user activity patterns and statistics")
    @app_commands.describe(username="Username to analyze (leave empty for yourself)")
    @app_commands.autocomplete(username=user_autocomplete)
    async def user_analysis(
        self, 
        interaction: discord.Interaction, 
        username: Optional[str] = None
    ):
        """
        Analyze user activity using templates.
        
        Usage: /user_analysis [username]
        
        If no username provided, analyzes the command author.
        """
        
        try:
            # Defer the response immediately to avoid timeout
            await interaction.response.defer()

            # Determine target user
            if not username:
                username = interaction.user.display_name
            
            # Execute sync operation in thread pool with performance tracking
            result, exec_time = await self.execute_tracked_sync_operation(
                "user_analysis",
                self._sync_user_analysis,
                username
            )
            
            # Send single comprehensive result
            if result:
                final_message = f"{result}\n\n✅ Analysis completed in {exec_time:.2f}s"
                await self._send_long_message_slash(interaction, final_message)
            else:
                await interaction.followup.send(f"❌ No data found for user **{username}**")

        except Exception as e:
            logger.error(f"User analysis failed: {e}")
            logger.error(traceback.format_exc())
            if not interaction.response.is_done():
                await interaction.response.send_message(f"❌ Analysis failed: {str(e)}")
            else:
                await interaction.followup.send(f"❌ Analysis failed: {str(e)}")
    
    def _sync_user_analysis(self, username: str) -> str:
        """Sync user analysis using enhanced template system with analyzer helpers."""
        
        from ...analysis.data_facade import get_analysis_data_facade
        from ...analysis.user_analyzer import UserAnalyzer
        from ...data.config import Settings
        
        settings = Settings()
        with get_analysis_data_facade(base_filter=settings.base_filter) as facade:
            # Create template engine with analyzer helpers
            template_engine = self._create_template_engine(facade)
            
            user_analyzer = UserAnalyzer(facade)
            
            # Get user analysis data
            analysis_result = user_analyzer.analyze(username=username)
            
            if not analysis_result or not analysis_result.statistics:
                return None
            
            # Get recent messages from this user for template context
            recent_messages = []
            try:
                # Use synchronous method that returns data directly
                messages_data = facade.message_repository.get_user_messages(username, days_back=7, limit=100)
                if messages_data:
                    recent_messages = [
                        {
                            'id': msg.get('id'),
                            'content': msg.get('content', ''),
                            'author': username,
                            'timestamp': msg.get('timestamp', ''),
                            'channel': msg.get('channel_name', '')
                        }
                        for msg in messages_data
                    ]
            except Exception as e:
                logger.warning(f"Could not fetch user messages for template context: {e}")
            
            # Prepare enhanced template data using correct UserAnalysisResponse attributes
            template_data = {
                'username': username,
                'analysis': analysis_result,
                'statistics': analysis_result.statistics,
                'user_info': analysis_result.user_info,
                'concepts': analysis_result.concepts or []
            }
            
            # Render template with message context
            return template_engine.render_template(
                'outputs/discord/user_analysis.md.j2',
                messages=recent_messages,
                **template_data
            )
    
    @app_commands.command(name="topics_analysis", description="Analyze popular topics and keywords in a channel")
    @app_commands.describe(
        channel_name="Channel name to analyze (leave empty for current channel)",
        days="Number of days to look back (default: 7)"
    )
    @app_commands.autocomplete(channel_name=channel_autocomplete)
    async def topics_analysis(
        self, 
        interaction: discord.Interaction, 
        channel_name: Optional[str] = None,
        days: int = 7
    ):
        """
        Analyze topics and keywords in a channel using templates.
        
        Usage: /topics_analysis [channel_name] [days]
        
        If no channel_name provided, analyzes current channel.
        """
        
        try:
            # Determine target channel first
            if not channel_name:
                if isinstance(interaction.channel, discord.DMChannel):
                    await interaction.response.send_message("❌ Please specify a channel name when using this command in DMs.")
                    return
                channel_name = interaction.channel.name
            
            # Send initial response and defer for long operation
            await interaction.response.send_message(f"🔍 Analyzing topics in **{channel_name}** (last {days} days)... This may take a moment.")
            
            # Execute sync operation in thread pool with performance tracking
            result, exec_time = await self.execute_tracked_sync_operation(
                "topics_analysis",
                self._sync_topics_analysis,
                channel_name, days
            )
            
            # Send result
            if result:
                await self._send_long_message_slash(interaction, result)
                await interaction.followup.send(f"✅ Topics analysis completed in {exec_time:.2f}s")
            else:
                await interaction.followup.send(f"❌ No topic data found for channel **{channel_name}**")

        except Exception as e:
            logger.error(f"Topics analysis failed: {e}")
            logger.error(traceback.format_exc())
            if not interaction.response.is_done():
                await interaction.response.send_message(f"❌ Analysis failed: {str(e)}")
            else:
                await interaction.followup.send(f"❌ Analysis failed: {str(e)}")
    
    def _sync_topics_analysis(self, channel_name: str, days: int) -> str:
        """Sync topics analysis using enhanced template system with analyzer helpers."""
        
        from ...analysis.data_facade import get_analysis_data_facade
        from ...analysis.topic_analyzer import TopicAnalyzer
        from ...data.config import Settings
        
        settings = Settings()
        with get_analysis_data_facade(base_filter=settings.base_filter) as facade:
            # Create template engine with analyzer helpers
            template_engine = self._create_template_engine(facade)
            
            topic_analyzer = TopicAnalyzer(facade)
            
            # Get topics analysis data  
            analysis_result = topic_analyzer.analyze(
                channel_name=channel_name,
                days_back=days
            )
            
            # Check if analysis was successful
            if not analysis_result or not hasattr(analysis_result, 'success') or not analysis_result.success:
                return None
                
            if not hasattr(analysis_result, 'topics') or not analysis_result.topics:
                return None
            
            # Get recent messages for template context and NLP analysis
            recent_messages = []
            try:
                # Use synchronous method that returns data directly
                messages_data = facade.message_repository.get_channel_messages(channel_name, days_back=days, limit=200)
                if messages_data:
                    recent_messages = [
                        {
                            'id': msg.get('id'),
                            'content': msg.get('content', ''),
                            'author': msg.get('username') or msg.get('author', ''),
                            'timestamp': msg.get('timestamp', ''),
                            'channel': msg.get('channel_name', channel_name)
                        }
                        for msg in messages_data
                    ]
            except Exception as e:
                logger.warning(f"Could not fetch messages for template context: {e}")
            
            # Prepare enhanced template data
            template_data = {
                'channel_name': channel_name,
                'days': days,
                'analysis': analysis_result,
                'topics': analysis_result.topics,
                'message_count': analysis_result.message_count,
                'capabilities_used': analysis_result.capabilities_used if hasattr(analysis_result, 'capabilities_used') else ['topic_analysis']
            }
            
            # Add domain analysis data if available from hybrid approach
            if hasattr(analysis_result, '_domain_analysis'):
                template_data['_domain_analysis'] = analysis_result._domain_analysis
            
            # Render template with message context for enhanced NLP analysis
            return template_engine.render_template(
                'outputs/discord/topic_analysis.md.j2',
                messages=recent_messages,
                **template_data
            )
    
    @app_commands.command(name="activity_trends", description="Analyze temporal activity trends and patterns")
    @app_commands.describe(
        channel_name="Channel name to analyze (leave empty for current channel)",
        days="Number of days to look back (default: 30)"
    )
    @app_commands.autocomplete(channel_name=channel_autocomplete)
    async def activity_trends(
        self, 
        interaction: discord.Interaction, 
        channel_name: Optional[str] = None,
        days: int = 30
    ):
        """
        Analyze temporal activity trends using templates.
        
        Usage: /activity_trends [channel_name] [days]
        
        If no channel_name provided, analyzes current channel.
        """
        
        try:
            # Determine target channel first
            if not channel_name:
                if isinstance(interaction.channel, discord.DMChannel):
                    await interaction.response.send_message("❌ Please specify a channel name when using this command in DMs.")
                    return
                channel_name = interaction.channel.name
            
            # Send initial response and defer for long operation
            await interaction.response.send_message(f"🔍 Analyzing activity trends in **{channel_name}** (last {days} days)... This may take a moment.")
            
            # Execute sync operation in thread pool with performance tracking
            result, exec_time = await self.execute_tracked_sync_operation(
                "activity_trends",
                self._sync_activity_trends,
                channel_name, days
            )
            
            # Send result
            if result:
                await self._send_long_message_slash(interaction, result)
                await interaction.followup.send(f"✅ Activity trends analysis completed in {exec_time:.2f}s")
            else:
                await interaction.followup.send(f"❌ No activity data found for channel **{channel_name}**")

        except Exception as e:
            logger.error(f"Activity trends analysis failed: {e}")
            logger.error(traceback.format_exc())
            if not interaction.response.is_done():
                await interaction.response.send_message(f"❌ Analysis failed: {str(e)}")
            else:
                await interaction.followup.send(f"❌ Analysis failed: {str(e)}")
    
    def _sync_activity_trends(self, channel_name: str, days: int) -> str:
        """Sync activity trends analysis using enhanced template system with analyzer helpers."""
        
        from ...analysis.data_facade import get_analysis_data_facade
        from ...analysis.temporal_analyzer import TemporalAnalyzer
        from ...data.config import Settings
        
        settings = Settings()
        with get_analysis_data_facade(base_filter=settings.base_filter) as facade:
            # Create template engine with analyzer helpers
            template_engine = self._create_template_engine(facade)
            
            temporal_analyzer = TemporalAnalyzer(facade)
            
            # Get temporal analysis data
            analysis_result = temporal_analyzer.analyze(
                channel_name=channel_name,
                days_back=days,
                granularity="day"
            )
            
            if not analysis_result or not analysis_result.temporal_data:
                return None
            
            # Get recent messages for template context
            recent_messages = []
            try:
                # Use synchronous method that returns data directly
                messages_data = facade.message_repository.get_channel_messages(channel_name, days_back=days, limit=150)
                if messages_data:
                    recent_messages = [
                        {
                            'id': msg.get('id'),
                            'content': msg.get('content', ''),
                            'author': msg.get('username') or msg.get('author', ''),
                            'timestamp': msg.get('timestamp', ''),
                            'channel': msg.get('channel_name', channel_name)
                        }
                        for msg in messages_data
                    ]
            except Exception as e:
                logger.warning(f"Could not fetch messages for template context: {e}")
            
            # Prepare enhanced template data
            template_data = {
                'channel_name': channel_name,
                'days': days,
                'analysis': analysis_result,
                'temporal_data': analysis_result.temporal_data,
                'patterns': analysis_result.patterns
            }
            
            # Render template with message context
            return template_engine.render_template(
                'outputs/discord/activity_trends.md.j2',
                messages=recent_messages,
                **template_data
            )
    
    @app_commands.command(name="top_users", description="Show most active users across all channels")
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
        """
        Show top active users using templates.
        
        Usage: /top_users [limit] [days]
        
        Shows the most active users by message count.
        """
        
        try:
            # Defer the response immediately to avoid timeout
            await interaction.response.defer()

            # Execute sync operation in thread pool with performance tracking
            result, exec_time = await self.execute_tracked_sync_operation(
                "top_users",
                self._sync_top_users,
                limit, days
            )
            
            # Send single comprehensive result
            if result:
                final_message = f"{result}\n\n✅ Analysis completed in {exec_time:.2f}s"
                await self._send_long_message_slash(interaction, final_message)
            else:
                await interaction.followup.send(f"❌ No user data found")

        except Exception as e:
            logger.error(f"Top users analysis failed: {e}")
            logger.error(traceback.format_exc())
            if not interaction.response.is_done():
                await interaction.response.send_message(f"❌ Analysis failed: {str(e)}")
            else:
                await interaction.followup.send(f"❌ Analysis failed: {str(e)}")
    
    def _sync_top_users(self, limit: int, days: int) -> str:
        """Sync top users analysis using enhanced template system with analyzer helpers."""
        
        from ...analysis.data_facade import get_analysis_data_facade
        from ...analysis.user_analyzer import UserAnalyzer
        from ...data.config import Settings
        
        settings = Settings()
        with get_analysis_data_facade(base_filter=settings.base_filter) as facade:
            # Create template engine with analyzer helpers
            template_engine = self._create_template_engine(facade)
            
            user_analyzer = UserAnalyzer(facade)
            
            # Get top users data
            top_users = user_analyzer.get_top_users(limit=limit, days=days)
            
            if not top_users:
                return None
            
            # Prepare enhanced template data
            template_data = {
                'limit': limit,
                'days': days,
                'top_users': top_users,
                'total_users': len(top_users)
            }
            
            # Render template (no specific messages needed for top users overview)
            return template_engine.render_template(
                'outputs/discord/top_users.md.j2',
                **template_data
            )
    
    @app_commands.command(name="list_channels", description="List all available channels for analysis")
    async def list_channels(self, interaction: discord.Interaction):
        """List available channels for analysis."""
        
        try:
            # Defer the response immediately to avoid timeout
            await interaction.response.defer()

            # Execute sync operation in thread pool
            channels, exec_time = await self.execute_tracked_sync_operation(
                "list_channels",
                self._sync_get_available_channels
            )
            
            if channels:
                channel_list = "\n".join([f"• **{channel}**" for channel in channels[:30]])
                if len(channels) > 30:
                    channel_list += f"\n... and {len(channels) - 30} more channels"
                
                embed = discord.Embed(
                    title="📺 Available Channels for Analysis",
                    description=channel_list,
                    color=discord.Color.blue()
                )
                embed.set_footer(text=f"Total: {len(channels)} channels • Retrieved in {exec_time:.2f}s")
                await interaction.followup.send(embed=embed)
            else:
                await interaction.followup.send("❌ No channels found in database")

        except Exception as e:
            logger.error(f"List channels failed: {e}")
            if not interaction.response.is_done():
                await interaction.response.send_message(f"❌ Failed to fetch channels: {str(e)}")
            else:
                await interaction.followup.send(f"❌ Failed to fetch channels: {str(e)}")
    
    def _sync_get_available_channels(self):
        """Get available channels using data facade."""
        from ...analysis.data_facade import get_analysis_data_facade
        
        with get_analysis_data_facade() as facade:
            return facade.channel_repository.get_available_channels()
    
    @app_commands.command(name="list_users", description="List all available users for analysis")
    async def list_users(self, interaction: discord.Interaction):
        """List available users for analysis."""
        
        try:
            # Defer the response immediately to avoid timeout
            await interaction.response.defer()

            # Execute sync operation in thread pool
            users, exec_time = await self.execute_tracked_sync_operation(
                "list_users",
                self._sync_get_available_users
            )
            
            if users:
                user_list = "\n".join([f"• **{user}**" for user in users[:20]])
                if len(users) > 20:
                    user_list += f"\n... and {len(users) - 20} more users"
                
                embed = discord.Embed(
                    title="👥 Available Users for Analysis",
                    description=user_list,
                    color=discord.Color.green()
                )
                embed.set_footer(text=f"Total: {len(users)} users • Retrieved in {exec_time:.2f}s")
                await interaction.followup.send(embed=embed)
            else:
                await interaction.followup.send("❌ No users found in database")
                
        except Exception as e:
            logger.error(f"List users failed: {e}")
            if not interaction.response.is_done():
                await interaction.response.send_message(f"❌ Failed to fetch users: {str(e)}")
            else:
                await interaction.followup.send(f"❌ Failed to fetch users: {str(e)}")
    
    def _sync_get_available_users(self):
        """Get available users using data facade."""
        from ...analysis.data_facade import get_analysis_data_facade
        
        with get_analysis_data_facade() as facade:
            return facade.user_repository.get_available_users()
    
    @app_commands.command(name="performance_metrics", description="Show performance metrics for analysis commands")
    async def performance_metrics(self, interaction: discord.Interaction):
        """Show performance metrics for analysis commands."""
        
        try:
            metrics = self.get_command_metrics()
            
            if not metrics:
                await interaction.response.send_message("📊 No performance metrics available yet")
                return
            
            embed = discord.Embed(
                title="⚡ Analysis Commands Performance Metrics",
                color=discord.Color.gold()
            )
            
            for command, stats in metrics.items():
                embed.add_field(
                    name=command,
                    value=f"Executions: {stats['executions']}\n"
                          f"Avg Time: {stats['avg_time']:.2f}s\n"
                          f"Recent: {stats['recent_time']:.2f}s",
                    inline=True
                )
            
            # Add thread pool status
            status = self.get_thread_pool_status()
            embed.add_field(
                name="Thread Pool Status",
                value=f"Workers: {status['max_workers']}\n"
                      f"Active Ops: {status['active_operations']}\n"
                      f"Pool Active: {status['thread_pool_active']}",
                inline=True
            )
            
            await interaction.response.send_message(embed=embed)

        except Exception as e:
            logger.error(f"Performance metrics failed: {e}")
            await interaction.response.send_message(f"❌ Failed to get metrics: {str(e)}")
    
    @app_commands.command(name="sync_commands", description="Manually sync slash commands with Discord")
    async def sync_commands(self, interaction: discord.Interaction):
        """Manually sync slash commands"""
        try:
            await interaction.response.send_message("🔄 Syncing slash commands...")
            
            synced = await self.bot.tree.sync()
            await interaction.followup.send(f"✅ Synced {len(synced)} command(s)")
            
            logger.info(f"Manually synced {len(synced)} command(s) via slash command")
            
        except Exception as e:
            logger.error(f"Failed to manually sync: {e}")
            if not interaction.response.is_done():
                await interaction.response.send_message(f"❌ Failed to sync: {e}")
            else:
                await interaction.followup.send(f"❌ Failed to sync: {e}")

    @app_commands.command(name="force_sync", description="Force sync slash commands (clears and reloads)")
    async def force_sync_commands(self, interaction: discord.Interaction):
        """Force sync slash commands (admin only)"""
        try:
            await interaction.response.send_message("🔄 Force syncing slash commands...")
            
            # Clear existing commands first
            self.bot.tree.clear_commands()

            # Reload commands
            await self.bot.reload_extension("pepino.discord.commands.analysis")

            # Sync again
            synced = await self.bot.tree.sync()
            await interaction.followup.send(f"✅ Force synced {len(synced)} command(s)")

            logger.info(f"Force synced {len(synced)} command(s) via slash command")
            for command in synced:
                logger.info(f"Force synced: {command.name}")

        except Exception as e:
            logger.error(f"Failed to force sync: {e}")
            if not interaction.response.is_done():
                await interaction.response.send_message(f"❌ Failed to force sync: {e}")
            else:
                await interaction.followup.send(f"❌ Failed to force sync: {e}")

    @app_commands.command(name="test_data", description="Test if database data is available for analysis")
    async def test_data_availability(self, interaction: discord.Interaction):
        """Test if autocomplete data is available"""
        try:
            await interaction.response.send_message("🔍 Testing data availability...")
            
            from ...analysis.data_facade import get_analysis_data_facade
            
            with get_analysis_data_facade() as facade:
                channels = facade.channel_repository.get_available_channels()
                users = facade.user_repository.get_available_users()

                embed = discord.Embed(
                    title="📊 Database Data Availability Test",
                    color=discord.Color.blue()
                )
                
                embed.add_field(
                    name="Channels",
                    value=f"Found {len(channels)} channels",
                    inline=True
                )
                
                embed.add_field(
                    name="Users",
                    value=f"Found {len(users)} users",
                    inline=True
                )
                
                # Show first few examples
                if channels:
                    sample_channels = channels[:5]
                    embed.add_field(
                        name="Sample Channels",
                        value="\n".join([f"• {ch}" for ch in sample_channels]),
                        inline=False
                    )

                if users:
                    sample_users = [u for u in users[:5] if u]  # Filter out empty names
                    embed.add_field(
                        name="Sample Users",
                        value="\n".join([f"• {u}" for u in sample_users]),
                        inline=False
                    )

                await interaction.followup.send(embed=embed)

        except Exception as e:
            logger.error(f"Test data availability failed: {e}")
            if not interaction.response.is_done():
                await interaction.response.send_message(f"❌ Test failed: {e}")
            else:
                await interaction.followup.send(f"❌ Test failed: {e}")
    
    async def _send_long_message(self, ctx: commands.Context, content: str):
        """Send long message, splitting if necessary."""
        
        max_length = 2000
        
        if len(content) <= max_length:
            await ctx.send(content)
            return
        
        # Split into chunks
        chunks = []
        current_chunk = ""
        
        for line in content.split('\n'):
            if len(current_chunk) + len(line) + 1 > max_length:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = line
            else:
                if current_chunk:
                    current_chunk += '\n' + line
                else:
                    current_chunk = line
        
        if current_chunk:
            chunks.append(current_chunk)
        
        # Send chunks
        for i, chunk in enumerate(chunks):
            if i == 0:
                await ctx.send(chunk)
            else:
                await ctx.send(f"**Continued ({i+1}/{len(chunks)}):**\n{chunk}")

    async def _send_long_message_slash(self, interaction: discord.Interaction, content: str):
        """Send long message using slash command followup, splitting if necessary."""
        
        max_length = 2000
        
        if len(content) <= max_length:
            await interaction.followup.send(content)
            return
        
        # Split into chunks
        chunks = []
        current_chunk = ""
        
        for line in content.split('\n'):
            if len(current_chunk) + len(line) + 1 > max_length:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = line
            else:
                if current_chunk:
                    current_chunk += '\n' + line
                else:
                    current_chunk = line
        
        if current_chunk:
            chunks.append(current_chunk)
        
        # Send chunks
        for i, chunk in enumerate(chunks):
            if i == 0:
                await interaction.followup.send(chunk)
            else:
                await interaction.followup.send(f"**Continued ({i+1}/{len(chunks)}):**\n{chunk}")


async def setup(bot):
    """Setup function for the cog."""
    await bot.add_cog(AnalysisCommands(bot))
    logger.info("AnalysisCommands cog loaded")


async def teardown(bot):
    """Teardown function for the cog."""
    cog = bot.get_cog("AnalysisCommands")
    if cog:
        await cog.cog_unload()
    logger.info("AnalysisCommands cog unloaded") 