"""
Analysis Service

Provides a clean, encapsulated interface for analysis operations.
Hides internal complexity while exposing dedicated methods for each analysis command.
"""

import logging
from typing import Dict, Any, Optional, Literal, List
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Generator

from pepino.config import Settings
from pepino.analysis.helpers.data_facade import get_analysis_data_facade
from pepino.analysis.templates.template_engine import TemplateEngine

logger = logging.getLogger(__name__)

OutputFormat = Literal["cli", "md"]


class AnalysisService:
    """
    Public interface for analysis operations.
    
    Provides dedicated methods for each analysis command, hiding internal complexity
    like template engines and specific template names.
    """
    
    def __init__(self, db_path: Optional[str] = None, base_filter: Optional[str] = None):
        """
        Initialize analysis service with optional database path and base filter.
        
        Args:
            db_path: Optional database path (uses settings default if None)
            base_filter: Optional base filter for data queries
        """
        self.settings = Settings()
        self.db_path = db_path or self.settings.db_path
        self.base_filter = base_filter or self.settings.base_filter
        
        # Lazy initialization - only create when needed
        self._data_facade = None
        self._analyzers = None
        self._template_engine = None
        self._nlp_service = None
        
        logger.debug(f"AnalysisService initialized for {self.db_path}")
    
    @property
    def data_facade(self):
        """Get data facade instance (lazy initialization)."""
        if self._data_facade is None:
            # Create database manager with the correct database path
            from pepino.data.database.manager import DatabaseManager
            db_manager = DatabaseManager(self.db_path)
            self._data_facade = get_analysis_data_facade(db_manager=db_manager, base_filter=self.base_filter)
        return self._data_facade
    
    @property
    def analyzers(self):
        """Get analyzer instances (lazy initialization)."""
        if self._analyzers is None:
            self._analyzers = self._create_analyzers()
        return self._analyzers
    
    @property
    def template_engine(self):
        """Get template engine instance (lazy initialization)."""
        if self._template_engine is None:
            self._template_engine = self._create_template_engine()
        return self._template_engine
    
    @property
    def nlp_service(self):
        """Get NLP service instance (lazy initialization)."""
        if self._nlp_service is None:
            self._nlp_service = self._create_nlp_service()
        return self._nlp_service
    
    def _create_analyzers(self) -> Dict[str, Any]:
        """Create analyzer instances."""
        try:
            from pepino.analysis.helpers.user_analyzer import UserAnalyzer
            from pepino.analysis.helpers.message_analyzer import MessageAnalyzer
            from pepino.analysis.helpers.weekly_user_analyzer import WeeklyUserAnalyzer
            from pepino.analysis.helpers.channel_analyzer import ChannelAnalyzer
            from pepino.analysis.helpers.topic_analyzer import TopicAnalyzer
            from pepino.analysis.helpers.temporal_analyzer import TemporalAnalyzer
            
            return {
                'user': UserAnalyzer(self.data_facade),
                'message': MessageAnalyzer(self.data_facade),
                'weekly_user': WeeklyUserAnalyzer(self.data_facade),
                'channel': ChannelAnalyzer(self.data_facade),
                'topic': TopicAnalyzer(self.data_facade),
                'temporal': TemporalAnalyzer(self.data_facade),
            }
        except ImportError as e:
            logger.warning(f"Could not load all analyzer classes: {e}")
            return {}
    
    def _create_template_engine(self) -> TemplateEngine:
        """Create template engine with all dependencies."""
        return TemplateEngine(
            templates_dir="src/pepino/analysis/templates",
            analyzers=self.analyzers,
            data_facade=self.data_facade,
            nlp_service=self.nlp_service
        )
    
    def _create_nlp_service(self):
        """Create NLP service if available."""
        try:
            from pepino.analysis.nlp_analyzer import NLPService
            return NLPService()
        except ImportError:
            logger.debug("NLP service not available")
            return None
    
    def pulsecheck(self, channel_name: Optional[str] = None, days_back: int = 7, 
                   end_date: Optional[datetime] = None, output_format: OutputFormat = "cli") -> str:
        """
        Generate weekly channel analysis (pulsecheck).
        
        Args:
            channel_name: Optional channel name (None for all channels)
            days_back: Number of days to look back (default: 7)
            end_date: End date for analysis (default: now)
            output_format: Output format ("cli" or "md")
            
        Returns:
            Formatted analysis string
        """
        end_date = end_date or datetime.now()
        
        message_analyzer = self.analyzers.get('message')
        user_analyzer = self.analyzers.get('weekly_user')
        
        if not message_analyzer or not user_analyzer:
            return "❌ Analysis failed: Required analyzers not available"
        
        # Get analysis data based on whether channel is specified
        if channel_name:
            # Single channel analysis
            message_analysis = message_analyzer.analyze_messages(channel_name, days_back, end_date)
            user_analysis = user_analyzer.analyze_weekly_users(channel_name, days_back, end_date)
            total_members = self.data_facade.channel_repository.get_channel_human_member_count(channel_name)
        else:
            # All channels analysis
            message_analysis = message_analyzer.analyze_messages(None, days_back, end_date)
            user_analysis = user_analyzer.analyze_weekly_users_all_channels(days_back, end_date)
            # For all channels, get total members across all channels
            total_members = self.data_facade.channel_repository.get_total_human_member_count()
        
        # Combine all data into a single 'data' dictionary as expected by the template
        data = {
            **message_analysis,
            **user_analysis,
            'total_members': total_members,
            'period': {
                'start_date': (end_date - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            }
        }
        
        template_name = f"outputs/{output_format}/channel_analysis.{'txt' if output_format == 'cli' else 'md'}.j2"
        
        return self.template_engine.render_template(
            template_name,
            channel_name=channel_name,
            data=data,
            total_members=total_members,
            format_number=lambda v: f"{v:,}",
            now=datetime.now
        )
    
    def top_contributors(self, channel_name: Optional[str] = None, limit: int = 10, 
                        days_back: int = 30, end_date: Optional[datetime] = None, 
                        output_format: OutputFormat = "cli") -> str:
        """
        Generate top contributors analysis.
        
        Args:
            channel_name: Optional channel name (None for all channels)
            limit: Number of top users to show (default: 10)
            days_back: Number of days to look back (default: 30)
            end_date: End date for analysis (default: now)
            output_format: Output format ("cli" or "md")
            
        Returns:
            Formatted analysis string
        """
        end_date = end_date or datetime.now()
        
        user_analyzer = self.analyzers.get('user')
        message_analyzer = self.analyzers.get('message')
        
        if not user_analyzer or not message_analyzer:
            return "❌ Analysis failed: Required analyzers not available"
        
        # Get contributors data using analyzer methods or data facade directly
        if channel_name:
            # Use data facade directly for channel-specific queries
            contributors = self.data_facade.user_repository.get_top_users(
                limit=limit, days_back=days_back, channel_name=channel_name
            )
            messages = self.data_facade.message_repository.get_channel_messages(
                channel_name, days_back=days_back, limit=1000
            )
        else:
            # Use analyzer method for general top users
            contributors = user_analyzer.get_top_users(limit=limit, days=days_back)
            messages = self.data_facade.message_repository.get_recent_messages(
                limit=10000, days_back=days_back
            )
        
        # Build enhanced contributors data
        enhanced_contributors = self._build_enhanced_contributors(contributors, messages, user_analyzer, days_back)
        
        period = {
            'start_date': (end_date - timedelta(days=days_back)).strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        }
        
        template_name = f"outputs/{output_format}/top_contributors.{'txt' if output_format == 'cli' else 'md'}.j2"
        
        return self.template_engine.render_template(
            template_name,
            channel_name=channel_name,
            contributors=enhanced_contributors,
            period=period,
            format_number=lambda v: f"{v:,}",
            now=datetime.now
        )
    
    def top_channels(self, limit: int = 5, days_back: int = 7, 
                    end_date: Optional[datetime] = None, output_format: OutputFormat = "cli") -> str:
        """
        Generate top channels summary report.
        
        Args:
            limit: Number of top channels to show (default: 5)
            days_back: Number of days to look back (default: 7)
            end_date: End date for analysis (default: now)
            output_format: Output format ("cli" or "md")
            
        Returns:
            Formatted analysis string
        """
        end_date = end_date or datetime.now()
        
        message_analyzer = self.analyzers.get('message')
        user_analyzer = self.analyzers.get('weekly_user')
        
        if not message_analyzer or not user_analyzer:
            return "❌ Analysis failed: Required analyzers not available"
        
        # Get all channels
        channels = self.data_facade.message_repository.get_all_channels()
        
        # Analyze each channel and collect data
        channel_data = []
        total_messages = 0
        total_active_users = set()
        increasing_channels = []
        decreasing_channels = []
        
        for channel_name in channels:
            try:
                # Get channel analysis
                message_analysis = message_analyzer.analyze_messages(channel_name, days_back, end_date)
                user_analysis = user_analyzer.analyze_weekly_users(channel_name, days_back, end_date)
                total_members = self.data_facade.channel_repository.get_channel_human_member_count(channel_name)
                
                # Skip channels with no activity
                channel_messages = message_analysis.get('statistics', {}).get('total_messages', 0)
                if channel_messages == 0:
                    continue
                
                # Get top contributors using data facade directly
                top_contributors = self.data_facade.user_repository.get_top_users(
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
                'start_date': (end_date - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            }
        }
        
        template_name = f"outputs/{output_format}/top_channels.{'txt' if output_format == 'cli' else 'md'}.j2"
        
        # Prepare data structure that matches template expectations
        data = {
            'period_start': summary['period']['start_date'],
            'period_end': summary['period']['end_date'],
            'top_channels': channel_data,
            'total_active_channels': summary['active_channels'],
            'total_channels': summary['total_channels'],
            'total_messages': summary['total_messages'],
            'total_active_users': summary['total_active_users'],
            'increasing_channels': increasing_channels,
            'decreasing_channels': decreasing_channels,
            'most_engaged_team': channel_data[0] if channel_data else {'name': 'None', 'participation_rate': 0},
            'peak_activity_times': 'Morning (9-12 AM)',  # Placeholder - could be calculated
            'top_contributors': [],  # Placeholder - could be calculated
            'avg_participation_rate': round(sum(ch.get('participation_rate', 0) for ch in channel_data) / len(channel_data) if channel_data else 0, 1)
        }
        
        return self.template_engine.render_template(
            template_name,
            data=data,
            format_number=lambda v: f"{v:,}",
            now=datetime.now
        )
    
    def list_channels(self, output_format: OutputFormat = "cli") -> str:
        """
        List all available channels.
        
        Args:
            output_format: Output format ("cli" or "md")
            
        Returns:
            Formatted channel list string
        """
        channels = self.data_facade.channel_repository.get_available_channels()
        
        template_name = f"outputs/{output_format}/channel_list.{'txt' if output_format == 'cli' else 'md'}.j2"
        
        return self.template_engine.render_template(
            template_name,
            channels=channels
        )
    
    def _build_enhanced_contributors(self, contributors, messages, user_analyzer, days_back):
        """Build enhanced contributors data with display names and activity."""
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
            
            channel_activity = self.data_facade.user_repository.get_user_channel_activity(
                contributor.get('author_name', 'Unknown'), days_back, limit=10
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
        
        return enhanced_contributors
    
    def close(self):
        """Close the analysis service and clean up resources."""
        if self._data_facade:
            self._data_facade.close()
        logger.debug("AnalysisService closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience function for common operations
@contextmanager
def analysis_service(db_path: Optional[str] = None, base_filter: Optional[str] = None) -> Generator[AnalysisService, None, None]:
    """
    Context manager for analysis operations.
    
    Args:
        db_path: Optional database path (uses settings default if None)
        base_filter: Optional base filter for data queries (uses settings default if None)
        
    Yields:
        AnalysisService instance
    """
    # Use settings for defaults if not provided
    settings = Settings()
    actual_db_path = db_path or settings.db_path
    actual_base_filter = base_filter or settings.base_filter
    service = AnalysisService(db_path=actual_db_path, base_filter=actual_base_filter)
    try:
        yield service
    finally:
        service.close() 