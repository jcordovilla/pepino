"""
Channel Analysis Service

Handles all channel-related analysis operations including pulsecheck, top channels,
and channel listing. Focuses specifically on channel analysis domain.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import traceback

from .base_service import BaseAnalysisService, OutputFormat

logger = logging.getLogger(__name__)


class ChannelAnalysisService(BaseAnalysisService):
    """
    Specialized service for channel analysis operations.
    
    Handles:
    - Pulsecheck (weekly channel analysis)
    - Top channels analysis
    - Channel listing
    - Channel activity patterns
    """
    
    def _create_analyzers(self) -> Dict[str, Any]:
        """Create channel-specific analyzers."""
        try:
            from pepino.analysis.helpers.message_analyzer import MessageAnalyzer
            from pepino.analysis.helpers.weekly_user_analyzer import WeeklyUserAnalyzer
            from pepino.analysis.helpers.channel_analyzer import ChannelAnalyzer
            
            return {
                'message': MessageAnalyzer(self.data_facade),
                'weekly_user': WeeklyUserAnalyzer(self.data_facade),
                'channel': ChannelAnalyzer(self.data_facade),
            }
        except ImportError as e:
            logger.warning(f"Could not load channel analyzer classes: {e}")
            return {}
    
    def pulsecheck(self, channel_name: Optional[str] = None, days_back: int = 7, 
                   end_date: Optional[datetime] = None, output_format: OutputFormat = "cli") -> str:
        """
        Generate weekly channel analysis (pulsecheck).
        
        Args:
            channel_name: Optional channel name (None for all channels)
            days_back: Number of days to look back (default: 7)
            end_date: End date for analysis (default: now)
            output_format: Output format ("cli" or "discord")
            
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
            total_members = self.data_facade.channel_repository.get_total_human_member_count()
        
        # Combine all data into a single 'data' dictionary
        data = {
            **message_analysis,
            **user_analysis,
            'total_members': total_members,
            'period': {
                'start_date': (end_date - timedelta(days=days_back)).strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            }
        }
        
        # Add channel activity data when no specific channel is selected
        if not channel_name:
            data.update(self._get_cross_channel_activity_data(days_back))
        
        return self.render_template(
            "channel_analysis",
            output_format=output_format,
            channel_name=channel_name,
            data=data,
            total_members=total_members
        )
    
    def top_channels(self, limit: int = 5, days_back: int = 7, 
                    end_date: Optional[datetime] = None, output_format: OutputFormat = "cli") -> str:
        """
        Generate top channels summary report.
        
        Args:
            limit: Number of top channels to show (default: 5)
            days_back: Number of days to look back (default: 7)
            end_date: End date for analysis (default: now)
            output_format: Output format ("cli" or "discord")
            
        Returns:
            Formatted analysis string
        """
        end_date = end_date or datetime.now()
        
        message_analyzer = self.analyzers.get('message')
        user_analyzer = self.analyzers.get('weekly_user')
        
        if not message_analyzer or not user_analyzer:
            return "❌ Analysis failed: Required analyzers not available"
        
        # Get all channels and analyze each one
        channels = self.data_facade.message_repository.get_all_channels()
        channel_data = self._analyze_all_channels(channels, message_analyzer, user_analyzer, days_back, end_date)
        # Sort by message count and limit
        channel_data.sort(key=lambda x: x['total_messages'], reverse=True)
        channel_data = channel_data[:limit]
        # Prepare summary data
        summary = self._prepare_top_channels_summary(channel_data, channels, days_back, end_date)
        return self.render_template(
            "top_channels",
            output_format=output_format,
            data=summary
        )
    
    def list_channels(self, output_format: OutputFormat = "cli") -> str:
        """
        List all available channels.
        
        Args:
            output_format: Output format ("cli" or "discord")
            
        Returns:
            Formatted channel list string
        """
        channels = self.data_facade.channel_repository.get_available_channels()
        
        return self.render_template(
            "channel_list",
            output_format=output_format,
            channels=channels
        )
    
    def _get_cross_channel_activity_data(self, days_back: int) -> Dict[str, Any]:
        """Get cross-channel activity data for all-channels analysis."""
        # Get top 5 most active channels (excluding bot traffic)
        most_active_channels = self.data_facade.channel_repository.get_top_channels_by_message_count(
            limit=5, days=days_back
        )
        
        # Get all channels with activity in the period (including 0 messages)
        all_channels_with_activity = self.data_facade.channel_repository.get_all_channels_with_activity(
            days=days_back
        )
        
        # Get least active channels (excluding bot traffic)
        active_channels = [ch for ch in all_channels_with_activity if ch['message_count'] > 0]
        least_active_channels = active_channels[-5:] if len(active_channels) >= 5 else active_channels
        
        # Calculate active/inactive distribution
        total_channels = len(all_channels_with_activity)
        if total_channels > 0:
            avg_messages = sum(ch['message_count'] for ch in all_channels_with_activity) / total_channels
            min_active_threshold = max(5, int(avg_messages * 0.05))
            
            n_active = sum(1 for ch in all_channels_with_activity if ch['message_count'] >= min_active_threshold)
            n_inactive = total_channels - n_active
            pct_active = (n_active / total_channels) * 100
            
            if pct_active >= 70:
                qualifier = 'good'
            elif pct_active >= 50:
                qualifier = 'moderate'
            else:
                qualifier = 'poor'
        else:
            n_active = n_inactive = pct_active = 0
            qualifier = 'poor'
            min_active_threshold = 0
        
        return {
            'active_channels_count': n_active,
            'inactive_channels_count': n_inactive,
            'active_channels_percentage': pct_active,
            'active_channels_qualifier': qualifier,
            'active_channels_min_threshold': min_active_threshold,
            'total_channels_analyzed': total_channels,
            'most_active_channels': most_active_channels,
            'least_active_channels': least_active_channels
        }
    
    def _analyze_all_channels(self, channels: list, message_analyzer, user_analyzer, 
                            days_back: int, end_date: datetime) -> list:
        """Analyze all channels and collect data."""
        channel_data = []
        for channel_name in channels:
            try:
                # Get channel analysis
                message_analysis = message_analyzer.analyze_messages(channel_name, days_back, end_date)
                user_analysis = user_analyzer.analyze_weekly_users(channel_name, days_back, end_date)
                total_members = self.data_facade.channel_repository.get_channel_human_member_count(channel_name)
                
                # Skip channels with no activity
                channel_messages = message_analysis.get('total_messages', 0)
                if channel_messages == 0:
                    continue
                # Get top contributors
                top_contributors = self.data_facade.user_repository.get_top_users(
                    limit=3, days_back=days_back, channel_name=channel_name
                )
                # Calculate participation rate
                active_members = len(user_analysis.get('top_users', []))
                participation_rate = round((active_members / total_members * 100) if total_members > 0 else 0, 1)
                # Determine trend
                trend_percentage = message_analysis.get('trend', {}).get('percentage_change', 0)
                trend_direction = "increasing" if trend_percentage > 0 else "decreasing" if trend_percentage < 0 else "unchanged"
                channel_data.append({
                    'name': channel_name,
                    'total_messages': channel_messages,  # renamed from message_count
                    'active_members': active_members,    # renamed from active_users
                    'total_members': total_members,
                    'participation_rate': participation_rate,
                    'trend_direction': trend_direction,
                    'trend_percentage': trend_percentage,
                    'top_contributors': [
                        {
                            'display_name': getattr(user, 'display_name', None) or getattr(user, 'author_name', None),
                            'author_name': getattr(user, 'author_name', None),
                            'message_count': getattr(user, 'message_count', 0)
                        } for user in top_contributors[:3]
                    ]
                })
            except Exception as e:
                print(f'EXCEPTION in channel {channel_name}: {e}')
                traceback.print_exc()
                continue
        return channel_data
    
    def _prepare_top_channels_summary(self, channel_data: list, all_channels: list, 
                                    days_back: int, end_date: datetime) -> Dict[str, Any]:
        """Prepare summary data for top channels analysis."""
        total_messages = sum(ch['total_messages'] for ch in channel_data)
        total_active_users = set()
        increasing_channels = []
        decreasing_channels = []
        
        for ch in channel_data:
            total_active_users.update([user.get('author_name') for user in ch.get('top_contributors', [])])
            if ch['trend_percentage'] > 0:
                increasing_channels.append(ch['name'])
            elif ch['trend_percentage'] < 0:
                decreasing_channels.append(ch['name'])
        
        return {
            'period_start': (end_date - timedelta(days=days_back)).strftime('%Y-%m-%d'),
            'period_end': end_date.strftime('%Y-%m-%d'),
            'top_channels': channel_data,
            'total_active_channels': len(channel_data),
            'total_channels': len(all_channels),
            'total_messages': total_messages,
            'total_active_users': len(total_active_users),
            'increasing_channels': increasing_channels,
            'decreasing_channels': decreasing_channels,
            'most_engaged_team': channel_data[0] if channel_data else {'name': 'None', 'participation_rate': 0},
            'peak_activity_times': 'Morning (9-12 AM)',  # TODO: Replace with real calculation if available
            'top_contributors': [
                contributor for ch in channel_data for contributor in ch.get('top_contributors', [])
            ][:5],
            'avg_participation_rate': round(sum(ch.get('participation_rate', 0) for ch in channel_data) / len(channel_data) if channel_data else 0, 1)
        } 