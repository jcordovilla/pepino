"""
DiscordBotAnalyzer class for async Discord message analysis
"""
import aiosqlite
from typing import List, Dict, Optional, Any, Union, Tuple
from models.base_analyzer import MessageAnalyzer
from database import get_available_channels, get_available_users, get_channel_name_mapping, get_available_channels_with_mapping, filter_boilerplate_phrases, get_selectable_channels_and_threads
from analysis.statistics import update_user_statistics, update_temporal_stats_async, get_enhanced_activity_trends
from analysis.topics import analyze_topics_spacy, extract_concepts_from_content
from analysis.insights import resolve_channel_name, get_user_insights, get_channel_insights


class DiscordBotAnalyzer(MessageAnalyzer):
    """Enhanced analyzer for Discord bot with async database operations"""
    
    def __init__(self, db_path: str = 'discord_messages.db'):
        # Initialize without calling parent __init__ since we use async connections
        self.db_path = db_path
        self.pool = None
        self.initialized = False
        
        # Base filter to exclude sesh bot and test channels
        self.base_filter = """
            author_id != 'sesh' 
            AND author_id != '1362434210895364327'
            AND author_name != 'sesh'
            AND LOWER(author_name) != 'pepe'
            AND LOWER(author_name) != 'pepino'
            AND channel_name NOT LIKE '%test%' 
            AND channel_name NOT LIKE '%playground%' 
            AND channel_name NOT LIKE '%pg%'
        """

    async def initialize(self):
        """Initialize async database connection"""
        if not self.initialized:
            try:
                import aiosqlite
                self.pool = await aiosqlite.connect(self.db_path)
                self.pool.row_factory = aiosqlite.Row
                self.initialized = True
            except ImportError:
                raise ImportError("aiosqlite is required for DiscordBotAnalyzer. Install with: pip install aiosqlite")

    async def close(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()
            self.initialized = False

    def __del__(self):
        """Cleanup method to handle connection cleanup safely"""
        try:
            # For DiscordBotAnalyzer, we use async connections (self.pool)
            # The pool should be closed via the async close() method
            # We don't have a self.conn attribute, so we override the base class __del__
            pass
        except:
            pass

    async def get_available_channels(self) -> List[str]:
        """Get list of available channels in the database (ordered by activity)"""
        await self.initialize()
        return await get_available_channels(self.pool, self.base_filter)

    async def get_available_users(self) -> List[str]:
        """Get list of available users in the database (display names preferred)"""
        await self.initialize()
        return await get_available_users(self.pool, self.base_filter)

    async def get_channel_name_mapping(self, bot_guilds=None) -> Dict[str, str]:
        """Create a mapping of old database channel names to current Discord channel names"""
        await self.initialize()
        return await get_channel_name_mapping(self.pool, bot_guilds)

    async def get_available_channels_with_mapping(self, bot_guilds=None) -> List[str]:
        """Get available channels with current Discord names when possible"""
        await self.initialize()
        return await get_available_channels_with_mapping(self.pool, bot_guilds)

    def filter_boilerplate_phrases(self, topics):
        """Remove common template/boilerplate phrases from topic list."""
        return filter_boilerplate_phrases(topics)

    async def analyze_topics_spacy(self, args: dict = None) -> str:
        """Simplified topic analysis with clean, actionable insights"""
        await self.initialize()
        return await analyze_topics_spacy(self.pool, self.base_filter, args)

    async def extract_concepts_from_content(self, messages) -> List[str]:
        """Extract most relevant and frequent topics from user messages"""
        return await extract_concepts_from_content(messages)

    async def update_user_statistics(self, args: dict = None) -> str:
        """Enhanced user activity statistics with concept analysis - overrides base class method"""
        await self.initialize()
        return await update_user_statistics(self.pool, self.base_filter, args)

    async def update_temporal_stats(self) -> str:
        """Update temporal statistics for activity trends - overrides base class method"""
        await self.initialize()
        return await get_enhanced_activity_trends(self.pool, self.base_filter)

    async def resolve_channel_name(self, user_input: str, bot_guilds=None) -> str:
        """Resolve user input to the actual database channel name"""
        await self.initialize()
        return await resolve_channel_name(self.pool, user_input, self.base_filter, bot_guilds)

    async def get_user_insights(self, user_name: str) -> Union[str, Tuple[str, str]]:
        """Get comprehensive insights for a specific user matching the original format"""
        await self.initialize()
        return await get_user_insights(self.pool, self.base_filter, user_name)

    async def get_channel_insights(self, channel_name: str, thread_id: str = None):
        """Get comprehensive channel statistics and insights"""
        await self.initialize()
        return await get_channel_insights(self.pool, self.base_filter, channel_name, thread_id=thread_id)

    async def get_selectable_channels_and_threads(self):
        await self.initialize()
        return await get_selectable_channels_and_threads(self.pool)
