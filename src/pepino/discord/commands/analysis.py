"""
Discord Analysis Commands:
"""

import logging
import traceback
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import Counter
import os

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
        from pepino.config import Settings
        settings = Settings()
        return DatabaseManager(settings.db_path)
    
    def _get_settings(self):
        """Get settings instance.""" 
        from pepino.config import Settings
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
            from ...config import Settings
            
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
        """Autocomplete function for usernames with display names for better UX."""
        try:
            # Get available users with display names directly without thread pool for speed
            from ...analysis.data_facade import get_analysis_data_facade
            from ...config import Settings

            settings = Settings()
            with get_analysis_data_facade(base_filter=settings.base_filter) as facade:
                # Get users with display names for better autocomplete experience
                users_data = facade.user_repository.get_users_for_autocomplete(limit=200)

            # Filter out empty/None usernames
            valid_users = [user_data for user_data in users_data if user_data['author_name'] and user_data['author_name'].strip()]

            # Enhanced filtering logic with display names
            if current:
                current_lower = current.lower()
                
                filtered_users = []
                
                for user_data in valid_users:
                    author_name = user_data['author_name']
                    display_name = user_data['display_name']
                    
                    # Check matches in both author_name and display_name
                    author_match = current_lower in author_name.lower()
                    display_match = current_lower in display_name.lower()
                    
                    if author_match or display_match:
                        # Prioritize exact matches, then starts with, then contains
                        priority = 0
                        if author_name.lower() == current_lower or display_name.lower() == current_lower:
                            priority = 1  # Exact match
                        elif author_name.lower().startswith(current_lower) or display_name.lower().startswith(current_lower):
                            priority = 2  # Starts with
                        else:
                            priority = 3  # Contains
                        
                        filtered_users.append((priority, user_data))
                
                # Sort by priority and limit to Discord's 25 choice limit
                filtered_users.sort(key=lambda x: (x[0], x[1]['display_name'].lower()))
                filtered_users = [user_data for _, user_data in filtered_users[:25]]
            else:
                # No input - return first 25 users (sorted by message count, already ordered by the query)
                filtered_users = valid_users[:25]

            return [
                app_commands.Choice(
                    name=f"{user_data['display_name']} ({user_data['message_count']} msgs)",
                    value=user_data['author_name']
                )
                for user_data in filtered_users
            ]

        except Exception as e:
            logger.error(f"User autocomplete failed: {e}")
            return []  # Return empty list instead of error choice
    
    @app_commands.command(
        name="analyze_channels", 
        description="Analyze channels with optional temporal and channel filters"
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
        channel_name: str = None,
        days_back: int = 7,
        end_date: str = None
    ):
        """Analyze channels with optional temporal and channel filters"""
        try:
            logger.info(f"Analyze channels command called with channel_name='{channel_name}', days_back={days_back}, end_date='{end_date}'")
            
            # Defer the response immediately to avoid timeout
            await interaction.response.defer()
            
            # Parse and validate end_date
            end_datetime = None
            if end_date:
                try:
                    end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
                except ValueError:
                    await interaction.followup.send("❌ Invalid date format. Please use YYYY-MM-DD format. Example: 2024-12-25")
                    return
            else:
                end_datetime = datetime.now()
            
            # Calculate start_date from days_back and end_date
            start_datetime = end_datetime - timedelta(days=days_back)
            
            # Validate days_back
            if days_back <= 0:
                await interaction.followup.send(f"❌ Days back must be a positive number. You provided: {days_back}")
                return
            
            # Prepare analysis parameters
            analysis_params = {
                'start_date': start_datetime.strftime('%Y-%m-%d'),
                'end_date': end_datetime.strftime('%Y-%m-%d'),
                'days_back': days_back
            }
            
            # Execute analysis based on channel_name
            if channel_name:
                # Single channel analysis
                logger.info(f"Performing single channel analysis for: {channel_name}")
                result, exec_time = await self.execute_tracked_sync_operation(
                    "analyze_single_channel",
                    self._sync_analyze_single_channel,
                    channel_name, **analysis_params
                )
                logger.debug(f"All channels summary analysis with chart complete in {exec_time:.2f}s")
            else:
                # All channels summary analysis with chart
                logger.info("Performing all channels summary analysis with chart")
                analysis_params['include_chart'] = True
                analysis_params['chart_type'] = 'daily_messages'
                result, exec_time = await self.execute_tracked_sync_operation(
                    "analyze_all_channels_summary",
                    self._sync_analyze_all_channels_summary,
                    **analysis_params
                )
                logger.debug(f"All channels summary analysis with chart complete in {exec_time:.2f}s")
            
            # Handle the result
            if isinstance(result, tuple):
                # Both single channel and cross-channel analysis can return (text, chart_path)
                text_result, chart_path = result
                
                if chart_path and os.path.exists(chart_path):
                    # Send text and chart together as one message
                    await interaction.followup.send(
                        text_result,
                        file=discord.File(chart_path, filename="daily_activity_chart.png")
                    )
                    # Clean up the temporary file
                    try:
                        os.remove(chart_path)
                    except Exception as e:
                        logger.warning(f"Could not remove temporary chart file {chart_path}: {e}")
                else:
                    # Send just the text if no chart
                    await self._send_long_message_slash(interaction, text_result)
            else:
                # Text-only result (no chart generated)
                await self._send_long_message_slash(interaction, result)
            
        except Exception as e:
            logger.error(f"Analyze channels command failed: {e}")
            logger.error(traceback.format_exc())
            if not interaction.response.is_done():
                await interaction.response.send_message(f"❌ Analysis failed: {str(e)}")
            else:
                await interaction.followup.send(f"❌ Analysis failed: {str(e)}")
    
    @app_commands.command(name="user_analysis", description="Analyze a specific user's activity")
    @app_commands.describe(
        username="Username to analyze",
        days="Number of days to look back (default: all time)",
        include_semantic="Include semantic analysis of user content"
    )
    @app_commands.autocomplete(username=user_autocomplete)
    async def user_analysis(
        self, 
        interaction: discord.Interaction, 
        username: str, 
        days: Optional[int] = None,
        include_semantic: bool = True
    ):
        """Analyze a specific user's activity."""
        await interaction.response.defer()
        
        try:
            # Execute sync operation in thread pool with performance tracking
            result, exec_time = await self.execute_tracked_sync_operation(
                "user_analysis",
                self._sync_user_analysis,
                username, days, include_semantic
            )
            
            # Send result
            if result:
                await self._send_long_message_slash(interaction, result)
            else:
                await interaction.followup.send(f"❌ No data found for user '{username}'")
            
        except Exception as e:
            logger.error(f"User analysis failed: {e}")
            await interaction.followup.send(f"❌ Error analyzing user: {e}")
    
    def _sync_user_analysis(self, username: str, days: Optional[int], include_semantic: bool) -> str:
        """Sync user analysis using enhanced template system with analyzer helpers."""
        
        from ...analysis.data_facade import get_analysis_data_facade
        from ...analysis.user_analyzer import UserAnalyzer
        from ...config import Settings
        
        settings = Settings()
        with get_analysis_data_facade(base_filter=settings.base_filter) as facade:
            # Create template engine with analyzer helpers
            template_engine = self._create_template_engine(facade)
            
            # Initialize analyzer with facade
            analyzer = UserAnalyzer(facade)
            
            # Run enhanced analysis
            analysis = analyzer.analyze_enhanced(username, days, include_semantic)
            
            if not analysis:
                return None
            
            # Prepare template data
            template_data = {
                'user_info': {
                    'display_name': analysis.user_info.display_name,
                    'author_name': analysis.user_info.author_id
                },
                'statistics': {
                    'message_count': analysis.statistics.message_count,
                    'channels_active': analysis.statistics.channels_active,
                    'active_days': analysis.statistics.active_days,
                    'avg_message_length': analysis.statistics.avg_message_length,
                    'first_message_date': analysis.statistics.first_message_date,
                    'last_message_date': analysis.statistics.last_message_date
                },
                'channel_activity': [
                    {
                        'channel_name': activity.channel_name,
                        'message_count': activity.message_count,
                        'avg_message_length': activity.avg_message_length
                    }
                    for activity in analysis.channel_activity
                ],
                'time_patterns': [
                    {
                        'period': pattern.period,
                        'message_count': pattern.message_count
                    }
                    for pattern in analysis.time_patterns
                ],
                'semantic_analysis': None
            }
            
            # Add semantic analysis if available
            if analysis.semantic_analysis:
                template_data['semantic_analysis'] = {
                    'key_entities': analysis.semantic_analysis.key_entities,
                    'technology_terms': analysis.semantic_analysis.technology_terms,
                    'key_concepts': analysis.semantic_analysis.key_concepts
                }
            
            # Render template (no specific messages needed for user analysis overview)
            return template_engine.render_template(
                "outputs/discord/user_analysis.md.j2",
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
        from ...config import Settings
        
        settings = Settings()
        with get_analysis_data_facade(base_filter=settings.base_filter) as facade:
            # Create template engine with analyzer helpers
            template_engine = self._create_template_engine(facade)
            
            # First check if we have enough messages for topic analysis
            try:
                messages_data = facade.message_repository.get_channel_messages(channel_name, days_back=days, limit=200)
                message_count = len(messages_data) if messages_data else 0
                
                # Filter to get human messages only
                human_messages = [msg for msg in messages_data if not msg.get('author_is_bot', False)] if messages_data else []
                human_message_count = len(human_messages)
                
                # Check if we have enough messages for meaningful topic analysis
                if human_message_count < 10:
                    # Not enough messages for topic analysis
                    return f"""**🧠 Topic Analysis: #{channel_name}**

**📊 Analysis Summary:**
• **Messages Found:** {human_message_count} human messages in last {days} days
• **Status:** ⚠️ Insufficient data for topic analysis

**💡 Recommendation:**
Topic analysis requires at least 10-15 messages to identify meaningful patterns. This channel has only {human_message_count} human messages in the last {days} days.

**🔍 Suggestions:**
• Try increasing the time period (e.g., 30 or 60 days)
• Check if this is an active discussion channel
• Consider analyzing a more active channel first

---
*Analysis generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')}*"""
                
            except Exception as e:
                logger.warning(f"Error checking message count for topic analysis: {e}")
                return f"❌ Error accessing channel data: {str(e)}"
            
            topic_analyzer = TopicAnalyzer(facade)
            
            # Get topics analysis data  
            analysis_result = topic_analyzer.analyze(
                channel_name=channel_name,
                days_back=days
            )
            
            # Check if analysis was successful
            if not analysis_result or not hasattr(analysis_result, 'success') or not analysis_result.success:
                return f"""**🧠 Topic Analysis: #{channel_name}**

**📊 Analysis Summary:**
• **Status:** ❌ Topic analysis failed
• **Error:** Unable to process messages for topic extraction

**💡 Recommendation:**
There may be an issue with the topic analysis engine. Please try again or contact support.

---
*Analysis generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')}*"""
                
            # Check if we have topics
            if not hasattr(analysis_result, 'topics') or not analysis_result.topics:
                # We have messages but no topics were found - this is the clustering issue
                return f"""**🧠 Topic Analysis: #{channel_name}**

**📊 Analysis Summary:**
• **Messages Analyzed:** {analysis_result.message_count} messages in last {days} days
• **Status:** ⚠️ No distinct topics identified

**🔍 Analysis Details:**
The topic analysis engine processed {analysis_result.message_count} messages but could not identify distinct discussion topics. This typically happens when:

• **Messages are too diverse** - No clear thematic clusters
• **Limited conversation depth** - Messages are brief or fragmented  
• **Mixed discussion topics** - Too many different subjects discussed
• **Insufficient clustering data** - Need more messages for pattern recognition

**💡 Recommendations:**
• Try analyzing a longer time period (30-60 days)
• Look for channels with more focused discussions
• Consider channels with longer, more detailed conversations

**🎯 Alternative Analysis:**
You might want to try:
• `/activity_trends {channel_name}` - See when people are most active
• `/channel_analysis {channel_name}` - Get overall channel health metrics
• `/top_users` - Find the most active contributors

---
*Analysis generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')}*"""
            
            # Get recent messages for template context and NLP analysis
            recent_messages = []
            try:
                # Use synchronous method that returns data directly
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
        from ...config import Settings
        
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
        from ...config import Settings
        
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
    
    def _sync_analyze_single_channel(self, channel_name: str, **analysis_params) -> str:
        """Sync single channel analysis with temporal filtering."""
        
        from ...analysis.data_facade import get_analysis_data_facade
        from ...analysis.channel_analyzer import ChannelAnalyzer
        from ...analysis.temporal_analyzer import TemporalAnalyzer
        from ...config import Settings
        
        settings = Settings()
        with get_analysis_data_facade(base_filter=settings.base_filter) as facade:
            # Create template engine with analyzer helpers
            template_engine = self._create_template_engine(facade)
            
            channel_analyzer = ChannelAnalyzer(facade)
            
            # Apply temporal filtering if provided
            analyzer_params = {}
            if 'days_back' in analysis_params:
                analyzer_params['days'] = analysis_params['days_back']
            # Note: ChannelAnalyzer.analyze() doesn't support start_date/end_date directly
            # It only supports 'days' parameter for temporal filtering
            
            # Get channel analysis data with temporal filtering
            analysis_result = channel_analyzer.analyze(channel_name=channel_name, **analyzer_params)
            
            # Debug: Raw analysis data
            logger.debug(f"Raw analysis_result.statistics: {getattr(analysis_result, 'statistics', None)}")
            logger.debug(f"Raw analysis_result.top_users: {getattr(analysis_result, 'top_users', None)}")
            logger.debug(f"Raw analysis_result.engagement_metrics: {getattr(analysis_result, 'engagement_metrics', None)}")
            
            if not analysis_result or not analysis_result.statistics:
                return None
            
            # Get total human members for percentage calculations
            total_human_members = 0
            try:
                total_human_members = facade.channel_repository.get_channel_human_member_count(channel_name)
                logger.debug(f"Raw total_human_members: {total_human_members}")
            except Exception as e:
                logger.warning(f"Could not get total human members for channel {channel_name}: {e}")
            
            # Debug: Raw user activity
            try:
                all_users = facade.channel_repository.get_channel_user_activity(channel_name, days=None, limit=100)
                logger.debug(f"Raw all_users (all time): {all_users}")
                recent_users = facade.channel_repository.get_channel_user_activity(channel_name, days=30, limit=100)
                logger.debug(f"Raw recent_users (30d): {recent_users}")
            except Exception as e:
                logger.warning(f"Could not fetch user activity for debug: {e}")
            
            # Debug: Raw messages
            try:
                all_messages = facade.message_repository.get_channel_messages(channel_name, days_back=None, limit=100)
                logger.debug(f"Raw all_messages (all time, up to 100): {all_messages}")
                recent_messages_debug = facade.message_repository.get_channel_messages(channel_name, days_back=7, limit=100)
                logger.debug(f"Raw recent_messages (7d, up to 100): {recent_messages_debug}")
            except Exception as e:
                logger.warning(f"Could not fetch messages for debug: {e}")
            
            # Calculate participation summary
            participation_summary = None
            if analysis_result.top_users:
                num_contributors = min(len(analysis_result.top_users), 5)
                top_messages = sum(user.message_count for user in analysis_result.top_users[:num_contributors])
                total_messages = analysis_result.statistics.total_messages
                logger.debug(f"Participation calculation: {num_contributors} contributors, {top_messages} messages, {total_messages} total")
                
                if total_messages > 0:
                    concentration = (top_messages / total_messages) * 100
                    if concentration > 70:
                        participation_summary = f"{num_contributors} top contributors posted {concentration:.0f}% of all messages (highly concentrated)"
                    elif concentration > 50:
                        participation_summary = f"{num_contributors} top contributors posted {concentration:.0f}% of all messages"
                    else:
                        participation_summary = f"{num_contributors} top contributors posted {concentration:.0f}% of all messages (well distributed)"
                    logger.debug(f"Participation summary set: {participation_summary}")
                else:
                    logger.debug("No total messages for participation calculation")
            else:
                logger.debug("No top users for participation calculation")
            
            # Calculate lost interest summary
            lost_interest_summary = None
            lost_interest_users = []
            try:
                # Get users who posted before but not in last 30 days
                all_users = facade.channel_repository.get_channel_user_activity(channel_name, days=None, limit=100)
                recent_users = facade.channel_repository.get_channel_user_activity(channel_name, days=30, limit=100)
                
                recent_usernames = {user['author_name'] for user in recent_users}
                inactive_users = [user for user in all_users if user['author_name'] not in recent_usernames]
                
                if inactive_users:
                    # Get days since last message for inactive users
                    from datetime import datetime, timezone
                    now = datetime.now(timezone.utc)
                    
                    for user in inactive_users[:5]:  # Limit to top 5 inactive users
                        if user['last_message']:
                            try:
                                last_msg_date = datetime.fromisoformat(user['last_message'].replace('Z', '+00:00'))
                                days_inactive = (now - last_msg_date).days
                                lost_interest_users.append({
                                    'display_name': user.get('author_display_name'),
                                    'author_name': user['author_name'],
                                    'days_inactive': days_inactive,
                                    'message_count': user['message_count']
                                })
                            except:
                                continue
                    
                    if lost_interest_users:
                        # Sort by days inactive and message count
                        lost_interest_users.sort(key=lambda x: (x['days_inactive'], x['message_count']), reverse=True)
                        inactive_count = len(lost_interest_users)
                        if inactive_count > 0:
                            lost_interest_summary = f"{inactive_count} former contributors inactive for 30+ days"
            except Exception as e:
                logger.warning(f"Could not calculate lost interest for channel {channel_name}: {e}")
            
            # Calculate engagement summary
            engagement_summary = None
            if analysis_result.engagement_metrics:
                reaction_rate = analysis_result.engagement_metrics.reaction_rate * 100
                if reaction_rate > 80:
                    engagement_summary = f"High ({reaction_rate:.0f}% reaction rate)"
                elif reaction_rate > 50:
                    engagement_summary = f"Moderate ({reaction_rate:.0f}% reaction rate)"
                else:
                    engagement_summary = f"Low ({reaction_rate:.0f}% reaction rate)"
            
            # Calculate trend summary (placeholder for now)
            trend_summary = None
            try:
                # Compare current period vs previous period
                current_messages = facade.message_repository.get_channel_messages(channel_name, days_back=7)
                previous_messages = facade.message_repository.get_channel_messages(channel_name, days_back=14, limit=len(current_messages) * 2)
                
                if current_messages and previous_messages:
                    current_count = len([m for m in current_messages if not m.get('author_is_bot', False)])
                    previous_count = len([m for m in previous_messages if not m.get('author_is_bot', False)])
                    
                    if previous_count > 0:
                        change_percent = ((current_count - previous_count) / previous_count) * 100
                        if change_percent > 20:
                            trend_summary = f"Activity increasing (+{change_percent:.0f}% vs. last week)"
                        elif change_percent < -20:
                            trend_summary = f"Activity decreasing ({change_percent:.0f}% vs. last week)"
                        else:
                            trend_summary = f"Activity stable ({change_percent:+.0f}% vs. last week)"
            except Exception as e:
                logger.warning(f"Could not calculate trend for channel {channel_name}: {e}")
            
            # Calculate bot activity summary
            bot_activity_summary = None
            if analysis_result.statistics.bot_messages > 0:
                bot_percentage = (analysis_result.statistics.bot_messages / analysis_result.statistics.total_messages) * 100
                logger.debug(f"Bot activity calculation: {analysis_result.statistics.bot_messages} bot messages, {analysis_result.statistics.human_messages} human messages, {bot_percentage:.1f}%")
                
                if analysis_result.statistics.bot_messages > analysis_result.statistics.human_messages:
                    bot_activity_summary = f"Bots posted {bot_percentage:.0f}% of messages (more than humans)"
                    logger.debug(f"Bot activity summary set: {bot_activity_summary}")
                elif bot_percentage > 10:  # Only show if bots > 10%
                    bot_activity_summary = f"Bots posted {bot_percentage:.0f}% of messages (less than humans)"
                    logger.debug(f"Bot activity summary set: {bot_activity_summary}")
                else:
                    logger.debug("Bot activity not significant enough to show")
            else:
                logger.debug("No bot messages found")
            
            # Calculate response time (placeholder for now)
            response_time = None
            try:
                # This would require analyzing reply chains and timestamps
                # For now, we'll skip this calculation
                pass
            except Exception as e:
                logger.warning(f"Could not calculate response time for channel {channel_name}: {e}")
            
            # Calculate recent activity summary
            recent_activity_summary = None
            try:
                recent_messages = facade.message_repository.get_channel_messages(channel_name, days_back=7)
                if recent_messages:
                    human_recent = len([m for m in recent_messages if not m.get('author_is_bot', False)])
                    previous_messages = facade.message_repository.get_channel_messages(channel_name, days_back=14, limit=len(recent_messages) * 2)
                    if previous_messages:
                        human_previous = len([m for m in previous_messages if not m.get('author_is_bot', False)])
                        if human_previous > 0:
                            change_percent = ((human_recent - human_previous) / human_previous) * 100
                            if change_percent > 0:
                                recent_activity_summary = f"{human_recent} messages in last 7 days (up {change_percent:.0f}% from previous week)"
                            else:
                                recent_activity_summary = f"{human_recent} messages in last 7 days (down {abs(change_percent):.0f}% from previous week)"
                        else:
                            recent_activity_summary = f"{human_recent} messages in last 7 days"
            except Exception as e:
                logger.warning(f"Could not calculate recent activity for channel {channel_name}: {e}")
            
            # Get topic analysis for frequent terms
            topic_analysis = None
            frequent_terms = []
            try:
                # Use NLPService for concept extraction instead of TopicAnalyzer
                from ...analysis.nlp_analyzer import NLPService
                nlp_service = NLPService()
                nlp_service.initialize()
                
                # Get recent messages for NLP analysis
                recent_messages_data = facade.message_repository.get_channel_messages(
                    channel_name, 
                    days_back=analyzer_params.get('days', 7), 
                    limit=200
                )
                
                if recent_messages_data:
                    # Extract concepts from all recent messages
                    all_concepts = []
                    for msg in recent_messages_data:
                        content = msg.get('content', '')
                        if content and len(content.strip()) > 10:  # Only analyze substantial messages
                            concepts = nlp_service.extract_concepts(content)
                            all_concepts.extend(concepts)
                    
                    # Count concept frequencies
                    concept_counts = Counter(all_concepts)
                    
                    # Get top 5 most frequent concepts
                    for concept, count in concept_counts.most_common(5):
                        if count >= 2:  # Only include concepts mentioned at least twice
                            frequent_terms.append({
                                'term': concept,
                                'frequency': count,
                                'relevance': min(count / len(recent_messages_data), 1.0)  # Normalize relevance
                            })
                    
                    logger.debug(f"Found {len(frequent_terms)} frequent terms for channel {channel_name}")
                else:
                    logger.debug(f"No recent messages for NLP analysis in channel {channel_name}")
            except Exception as e:
                logger.warning(f"Could not perform NLP analysis for channel {channel_name}: {e}")
            
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
            
            # Also get temporal data for this channel for charting
            temporal_analyzer = TemporalAnalyzer(facade)
            temporal_result = temporal_analyzer.analyze(channel_name=channel_name, days_back=analyzer_params.get('days', 7), granularity='day')
            temporal_data = getattr(temporal_result, 'temporal_data', None)
            
            # Prepare template data
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
                        'common_words': frequent_terms  # Now populated by topic analysis
                    },
                    'total_human_members': total_human_members
                },
                'analysis': analysis_result,
                'channel_name': channel_name,
                'top_users': analysis_result.top_users or [],
                'peak_activity': analysis_result.peak_activity,
                'engagement_metrics': analysis_result.engagement_metrics,
                'health_metrics': analysis_result.health_metrics,
                'recent_activity': analysis_result.recent_activity or [],
                'total_human_members': total_human_members,
                'participation_summary': participation_summary,
                'lost_interest_summary': lost_interest_summary,
                'lost_interest_users': lost_interest_users,
                'engagement_summary': engagement_summary,
                'trend_summary': trend_summary,
                'bot_activity_summary': bot_activity_summary,
                'response_time': response_time,
                'recent_activity_summary': recent_activity_summary,
                'channel_health': True,  # Flag to show health metrics
                'analysis_params': analysis_params,
                'recent_messages': recent_messages,
                'trends': {
                    'message_trend': 'stable',  # Default value
                    'user_trend': 'stable',     # Default value
                    'activity_trend': 'stable'  # Default value
                },
                'cross_channel_stats': {
                    'total_channels': 1,
                    'total_messages': analysis_result.statistics.total_messages,
                    'total_users': analysis_result.statistics.unique_users
                },
                # Add topic analysis data
                'topic_analysis': topic_analysis,
                'frequent_terms': frequent_terms,
                'temporal_data': temporal_data
            }
            
            # Debug logging for template data
            logger.debug(f"Template data summary fields:")
            logger.debug(f"  participation_summary: {participation_summary}")
            logger.debug(f"  lost_interest_summary: {lost_interest_summary}")
            logger.debug(f"  engagement_summary: {engagement_summary}")
            logger.debug(f"  trend_summary: {trend_summary}")
            logger.debug(f"  bot_activity_summary: {bot_activity_summary}")
            logger.debug(f"  recent_activity_summary: {recent_activity_summary}")
            logger.debug(f"  total_human_members: {total_human_members}")
            
            # Generate daily messages chart if temporal data is available
            chart_path = None
            if temporal_data:
                try:
                    chart_path = self._generate_daily_messages_chart(temporal_data, analysis_params)
                    if not chart_path:
                        logger.warning("Chart generation returned None")
                except Exception as e:
                    logger.warning(f"Could not generate chart: {e}")
            
            # Render template with message context
            text_result = template_engine.render_template(
                'outputs/discord/channel_analysis.md.j2',
                messages=recent_messages,
                **template_data
            )
            
            # Return tuple if chart was generated, otherwise just the text
            if chart_path:
                return (text_result, chart_path)
            else:
                return text_result
    
    def _sync_analyze_all_channels_summary(self, **analysis_params):
        """Sync all channels summary analysis with chart generation."""
        
        from ...analysis.data_facade import get_analysis_data_facade
        from ...analysis.temporal_analyzer import TemporalAnalyzer
        from ...config import Settings
        
        settings = Settings()
        with get_analysis_data_facade(base_filter=settings.base_filter) as facade:
            # Create template engine with analyzer helpers
            template_engine = self._create_template_engine(facade)
            
            temporal_analyzer = TemporalAnalyzer(facade)
            
            # Get temporal analysis data for all channels
            temporal_params = {}
            if 'days_back' in analysis_params:
                temporal_params['days_back'] = analysis_params['days_back']
            # Note: TemporalAnalyzer.analyze() doesn't support start_date/end_date directly
            # It only supports 'days_back' parameter for temporal filtering
            
            # Analyze all channels with temporal filtering
            analysis_result = temporal_analyzer.analyze(
                channel_name=None,  # None means all channels
                granularity="day",
                **temporal_params
            )
            
            if not analysis_result or not analysis_result.temporal_data:
                return None
            
            # Get cross-channel statistics using data facade
            channels_summary = facade.get_cross_channel_summary(
                days_back=analysis_params.get('days_back', 7),
                limit=10
            )
            
            # Calculate overall statistics
            total_channels = len(channels_summary)
            total_messages = sum(ch['message_count'] for ch in channels_summary)
            avg_messages_per_channel = total_messages / total_channels if total_channels > 0 else 0
            
            # Prepare template data for all channels summary
            template_data = {
                'analysis_type': 'all_channels_summary',
                'date_range': {
                    'start': analysis_params.get('start_date', ''),
                    'end': analysis_params.get('end_date', ''),
                    'days': analysis_params.get('days_back', 7)
                },
                'temporal_analysis': analysis_result,
                'temporal_data': analysis_result.temporal_data,
                'patterns': analysis_result.patterns,
                'channels_summary': channels_summary,
                'temporal_filter': analysis_params,
                # Add required template variables
                'trends': {
                    'total_messages': total_messages,
                    'avg_messages_per_period': sum([d.message_count for d in analysis_result.temporal_data]) / len(analysis_result.temporal_data) if analysis_result.temporal_data else 0,
                    'max_messages_in_period': max([d.message_count for d in analysis_result.temporal_data]) if analysis_result.temporal_data else 0,
                    'message_trend': 'stable',  # Default trend
                    'trend_percentage': 0.0  # Default percentage
                },
                'channel_name': 'All Channels',
                'days': analysis_params.get('days_back', 7),
                'peak_analysis': {
                    'peak_hour': None,
                    'peak_day': None
                },
                # Add cross-channel summary data
                'cross_channel_stats': {
                    'total_channels_analyzed': total_channels,
                    'total_messages': total_messages,
                    'avg_messages_per_channel': avg_messages_per_channel,
                    'most_active_channel': channels_summary[0]['name'] if channels_summary else 'None',
                    'most_active_count': channels_summary[0]['message_count'] if channels_summary else 0
                }
            }
            
            # Generate chart if requested
            chart_path = None
            if analysis_result.temporal_data:
                try:
                    chart_path = self._generate_daily_messages_chart(analysis_result.temporal_data, analysis_params)
                    if not chart_path:
                        logger.warning("Chart generation returned None")
                except Exception as e:
                    logger.warning(f"Could not generate chart: {e}")
            
            # Fallback: Generate chart from channels_summary if temporal chart failed
            if not chart_path and channels_summary:
                try:
                    chart_path = self._generate_channels_summary_chart(channels_summary, analysis_params)
                except Exception as e:
                    logger.warning(f"Could not generate fallback chart: {e}")
            
            # Render template 
            if channels_summary:
                # Create custom summary for all channels
                text_result = f"""## 📊 All Channels Summary Analysis

**📈 Time Period:** {analysis_params.get('start_date', '')} to {analysis_params.get('end_date', '')} ({analysis_params.get('days_back', 7)} days)

**📋 Overall Statistics:**
• Total Messages: {total_messages:,}
• Active Channels: {total_channels}
• Average Messages per Channel: {avg_messages_per_channel:.1f}

**🏆 Top 10 Most Active Channels:**
"""
                for i, channel in enumerate(channels_summary, 1):
                    text_result += f"{i}. **#{channel['name']}**: {channel['message_count']:,} messages\n"
                
                text_result += f"""
**📈 Daily Activity Pattern:**
• Average Messages per Day: {template_data['trends']['avg_messages_per_period']:.1f}
• Peak Day: {template_data['trends']['max_messages_in_period']} messages

**💡 Insights:**
• Most active channel: **#{template_data['cross_channel_stats']['most_active_channel']}** with {template_data['cross_channel_stats']['most_active_count']:,} messages
• Channel diversity: {total_channels} channels with significant activity
• Activity concentration: {template_data['cross_channel_stats']['most_active_count']/total_messages*100:.1f}% of messages in the top channel
"""
            else:
                # Fallback to activity trends template if no channel data
                text_result = template_engine.render_template(
                    'outputs/discord/activity_trends.md.j2',
                    **template_data
                )
            
            # Return tuple if chart was generated, otherwise just the text
            if chart_path:
                return (text_result, chart_path)
            else:
                return text_result
    
    def _generate_daily_messages_chart(self, temporal_data, analysis_params):
        """Generate daily messages bar chart with average line."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from datetime import datetime
            import numpy as np
            
            if not temporal_data:
                logger.warning("No temporal data for chart generation")
                return None
            
            # Extract dates and message counts from temporal data
            dates = []
            message_counts = []
            
            for data_point in temporal_data:
                # Handle different possible data structures
                if hasattr(data_point, 'period') and hasattr(data_point, 'message_count'):
                    # Handle TemporalDataPoint objects (from TemporalAnalyzer)
                    try:
                        date = datetime.strptime(data_point.period, '%Y-%m-%d')
                        dates.append(date)
                        message_counts.append(data_point.message_count)
                    except (ValueError, AttributeError):
                        continue
                elif hasattr(data_point, 'date') and hasattr(data_point, 'message_count'):
                    try:
                        date = datetime.strptime(data_point.date, '%Y-%m-%d')
                        dates.append(date)
                        message_counts.append(data_point.message_count)
                    except (ValueError, AttributeError):
                        continue
                elif hasattr(data_point, 'timestamp') and hasattr(data_point, 'count'):
                    try:
                        # Handle timestamp-based data
                        if isinstance(data_point.timestamp, str):
                            date = datetime.strptime(data_point.timestamp, '%Y-%m-%d')
                        else:
                            date = data_point.timestamp
                        dates.append(date)
                        message_counts.append(data_point.count)
                    except (ValueError, AttributeError):
                        continue
                elif isinstance(data_point, dict):
                    # Handle dictionary format
                    try:
                        date_str = data_point.get('period') or data_point.get('date') or data_point.get('timestamp')
                        count = data_point.get('message_count') or data_point.get('count')
                        if date_str and count is not None:
                            date = datetime.strptime(date_str, '%Y-%m-%d')
                            dates.append(date)
                            message_counts.append(count)
                    except (ValueError, AttributeError):
                        continue
            
            if not dates or not message_counts:
                logger.warning("No valid temporal data for chart generation")
                return None
            
            # Sort by date
            sorted_data = sorted(zip(dates, message_counts))
            dates, message_counts = zip(*sorted_data)
            
            # Create the chart
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Bar chart for daily messages
            bars = ax.bar(dates, message_counts, alpha=0.7, color='#5865F2', label='Daily Messages')
            
            # Average line
            avg_messages = np.mean(message_counts)
            ax.axhline(y=avg_messages, color='#ED4245', linestyle='--', linewidth=2, 
                      label=f'Average ({avg_messages:.1f} messages)')
            
            # Formatting
            ax.set_xlabel('Date')
            ax.set_ylabel('Number of Messages')
            ax.set_title(f"Daily Message Activity ({analysis_params.get('start_date', '')} to {analysis_params.get('end_date', '')})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates) // 10)))
            plt.xticks(rotation=45)
            
            # Tight layout to prevent label cutoff
            plt.tight_layout()
            
            # Save chart
            chart_path = f"temp/daily_messages_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            os.makedirs(os.path.dirname(chart_path), exist_ok=True)
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except ImportError:
            logger.warning("Matplotlib not available for chart generation")
            return None
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            return None
    
    def _generate_channels_summary_chart(self, channels_summary, analysis_params):
        """Generate channels summary bar chart as fallback when temporal data is not available."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from datetime import datetime
            
            if not channels_summary:
                logger.warning("No channels summary data for chart generation")
                return None
            
            # Extract channel names and message counts
            channel_names = [ch['name'] for ch in channels_summary]
            message_counts = [ch['message_count'] for ch in channels_summary]
            
            # Create the chart
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Bar chart for channel activity
            bars = ax.bar(range(len(channel_names)), message_counts, alpha=0.7, color='#5865F2', label='Messages per Channel')
            
            # Average line
            avg_messages = np.mean(message_counts)
            ax.axhline(y=avg_messages, color='#ED4245', linestyle='--', linewidth=2, 
                      label=f'Average ({avg_messages:.1f} messages)')
            
            # Formatting
            ax.set_xlabel('Channels')
            ax.set_ylabel('Number of Messages')
            ax.set_title(f"Channel Activity Summary ({analysis_params.get('start_date', '')} to {analysis_params.get('end_date', '')})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Set x-axis labels (channel names)
            ax.set_xticks(range(len(channel_names)))
            ax.set_xticklabels(channel_names, rotation=45, ha='right')
            
            # Tight layout to prevent label cutoff
            plt.tight_layout()
            
            # Save chart
            chart_path = f"temp/channels_summary_chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            os.makedirs(os.path.dirname(chart_path), exist_ok=True)
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except ImportError:
            logger.warning("Matplotlib not available for chart generation")
            return None
        except Exception as e:
            logger.error(f"Channels summary chart generation failed: {e}")
            return None
    
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
            
            # Clear existing commands first - specify guild or global
            if interaction.guild:
                # Clear guild-specific commands
                self.bot.tree.clear_commands(guild=interaction.guild)
                logger.info(f"Cleared guild commands for {interaction.guild.name}")
            else:
                # Clear global commands  
                self.bot.tree.clear_commands(guild=None)
                logger.info("Cleared global commands")

            # Reload commands
            try:
                await self.bot.reload_extension("pepino.discord.commands.analysis")
                logger.info("Successfully reloaded pepino.discord.commands.analysis")
            except Exception as reload_error:
                logger.error(f"Failed to reload extension: {reload_error}")
                await interaction.followup.send(f"❌ Failed to reload extension: {reload_error}")
                return

            # Debug: Check what commands are registered before sync
            if interaction.guild:
                commands_before = self.bot.tree.get_commands(guild=interaction.guild)
                logger.info(f"Commands before sync (guild): {[cmd.name for cmd in commands_before]}")
            else:
                commands_before = self.bot.tree.get_commands()
                logger.info(f"Commands before sync (global): {[cmd.name for cmd in commands_before]}")

            # Sync again
            if interaction.guild:
                synced = await self.bot.tree.sync(guild=interaction.guild)
                sync_scope = f"guild {interaction.guild.name}"
            else:
                synced = await self.bot.tree.sync()
                sync_scope = "globally"
                
            await interaction.followup.send(f"✅ Force synced {len(synced)} command(s) {sync_scope}")

            logger.info(f"Force synced {len(synced)} command(s) {sync_scope} via slash command")
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