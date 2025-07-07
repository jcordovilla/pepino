"""
Activity Trends Analyzer

Provides activity trend analysis with chart generation capabilities.
This analyzer works with the legacy activity_trends template to generate
visualizations of message activity patterns.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from pepino.analysis.helpers.data_facade import AnalysisDataFacade, get_analysis_data_facade
from pepino.analysis.models import TemporalAnalysisResponse, TemporalDataPoint, TemporalPatterns

logger = logging.getLogger(__name__)


class ActivityTrendsAnalyzer:
    """
    Activity Trends Analyzer for generating activity visualizations.
    
    Features:
    - Activity pattern analysis
    - Chart generation for Discord embedding
    - Cross-channel activity summary
    - Trend analysis with visualizations
    """
    
    def __init__(self, data_facade: Optional[AnalysisDataFacade] = None):
        """Initialize activity trends analyzer with data facade."""
        if data_facade is None:
            self.data_facade = get_analysis_data_facade()
            self._owns_facade = True
        else:
            self.data_facade = data_facade
            self._owns_facade = False
            
        logger.info("ActivityTrendsAnalyzer initialized")
    
    def analyze(
        self,
        channel_name: Optional[str] = None,
        days_back: Optional[int] = None,
        time_period: Optional[str] = None,
        output_format: str = "cli"
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze activity trends and generate chart data.
        
        Args:
            channel_name: Optional channel name to filter by
            days_back: Number of days to look back
            time_period: Optional time period description
            output_format: "cli" or "discord" (chart only generated for discord)
            
        Returns:
            Dictionary with analysis results and chart data
        """
        try:
            logger.info(f"Starting activity trends analysis for channel: {channel_name}")
            
            # Get messages using data facade
            if channel_name:
                messages = self.data_facade.message_repository.get_channel_messages(
                    channel_name, days_back=days_back, limit=10000
                )
            else:
                messages = self.data_facade.message_repository.get_recent_messages(
                    limit=10000, days_back=days_back
                )
            
            if not messages:
                logger.warning("No messages found for activity trends analysis")
                return None
            
            # Analyze temporal patterns
            temporal_data = self._analyze_temporal_data(messages)
            patterns = self._analyze_patterns(temporal_data)
            
            # Generate chart only for Discord format
            chart_base64 = ""
            if output_format == "discord":
                chart_base64 = self._generate_activity_chart(temporal_data, channel_name)
            
            # Get cross-channel stats if no specific channel
            cross_channel_stats = None
            if not channel_name:
                cross_channel_stats = self._get_cross_channel_stats(days_back)
            
            return {
                'temporal_data': temporal_data,
                'patterns': patterns,
                'chart_base64': chart_base64,
                'cross_channel_stats': cross_channel_stats,
                'time_period': time_period,
                'channel_name': channel_name,
                'days_back': days_back
            }
            
        except Exception as e:
            logger.error(f"Activity trends analysis failed: {e}")
            return None
    
    def _analyze_temporal_data(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze temporal data for daily activity patterns."""
        try:
            from collections import Counter
            period_counts = Counter()
            
            for msg in messages:
                timestamp = msg.timestamp
                if not timestamp:
                    continue
                
                try:
                    dt = timestamp if isinstance(timestamp, datetime) else datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))
                    period = dt.strftime('%Y-%m-%d')
                    period_counts[period] += 1
                except Exception as e:
                    logger.debug(f"Failed to parse timestamp {timestamp}: {e}")
                    continue
            
            # Convert to list format and sort by date
            temporal_data = []
            for period in sorted(period_counts.keys()):
                temporal_data.append({
                    'period': period,
                    'message_count': period_counts[period]
                })
            
            return temporal_data
            
        except Exception as e:
            logger.error(f"Failed to analyze temporal data: {e}")
            return []
    
    def _analyze_patterns(self, temporal_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in temporal data."""
        try:
            if not temporal_data:
                return {}
            
            total_messages = sum(item['message_count'] for item in temporal_data)
            total_periods = len(temporal_data)
            
            if total_periods == 0:
                return {}
            
            avg_messages_per_period = total_messages / total_periods
            
            # Find peak periods
            max_messages = max(item['message_count'] for item in temporal_data)
            min_messages = min(item['message_count'] for item in temporal_data)
            
            most_active_period = None
            for item in temporal_data:
                if item['message_count'] == max_messages:
                    most_active_period = item['period']
                    break
            
            # Calculate trend
            if len(temporal_data) >= 2:
                first_half = temporal_data[:len(temporal_data)//2]
                second_half = temporal_data[len(temporal_data)//2:]
                
                first_avg = sum(item['message_count'] for item in first_half) / len(first_half) if first_half else 0
                second_avg = sum(item['message_count'] for item in second_half) / len(second_half) if second_half else 0
                
                if second_avg > first_avg * 1.1:
                    message_trend = "increasing"
                    trend_percentage = ((second_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0
                elif second_avg < first_avg * 0.9:
                    message_trend = "decreasing"
                    trend_percentage = ((first_avg - second_avg) / first_avg * 100) if first_avg > 0 else 0
                else:
                    message_trend = "stable"
                    trend_percentage = 0
            else:
                message_trend = "stable"
                trend_percentage = 0
            
            return {
                'total_messages': total_messages,
                'avg_messages_per_period': round(avg_messages_per_period, 1),
                'most_active_period': most_active_period,
                'max_messages_in_period': max_messages,
                'min_messages_in_period': min_messages,
                'message_trend': message_trend,
                'trend_percentage': round(trend_percentage, 1),
                'trend_timeframe': f"over {total_periods} days"
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze patterns: {e}")
            return {}
    
    def _generate_activity_chart(self, temporal_data: List[Dict[str, Any]], channel_name: Optional[str] = None) -> str:
        """Generate activity chart as base64 string."""
        try:
            from pepino.analysis.visualization.charts import create_daily_activity_chart
            
            title = "Daily Message Activity"
            if channel_name:
                title = f"Daily Message Activity - {channel_name}"
            
            return create_daily_activity_chart(temporal_data, title)
            
        except Exception as e:
            logger.error(f"Failed to generate activity chart: {e}")
            return ""
    
    def _get_cross_channel_stats(self, days_back: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get cross-channel statistics."""
        try:
            # Get top channels
            top_channels = self.data_facade.channel_repository.get_top_channels_by_message_count(
                limit=1, days=days_back
            )
            
            # Get total stats using correct method names
            total_channels = self.data_facade.channel_repository.get_total_channel_count()
            # Note: get_total_message_count doesn't accept days_back parameter, so we get all-time count
            total_messages = self.data_facade.message_repository.get_total_message_count()
            total_users = self.data_facade.user_repository.get_total_user_count()
            
            most_active_channel = None
            most_active_count = 0
            
            if top_channels:
                most_active_channel = top_channels[0].get('channel_name', 'Unknown')
                most_active_count = top_channels[0].get('message_count', 0)
            
            return {
                'total_channels': total_channels,
                'total_messages': total_messages,
                'total_users': total_users,
                'most_active_channel': most_active_channel,
                'most_active_count': most_active_count
            }
            
        except Exception as e:
            logger.error(f"Failed to get cross-channel stats: {e}")
            return None
    
    def __del__(self):
        """Cleanup if we own the facade."""
        if hasattr(self, '_owns_facade') and self._owns_facade and hasattr(self, 'data_facade'):
            try:
                self.data_facade.close()
            except:
                pass 