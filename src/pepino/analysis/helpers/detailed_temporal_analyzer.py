"""
Detailed Temporal Analyzer

Provides advanced temporal analysis of message activity patterns using the new data facade pattern.
This analyzer reuses the legacy temporal analysis logic while maintaining the new architecture.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import Counter

from pepino.analysis.helpers.data_facade import AnalysisDataFacade, get_analysis_data_facade
from pepino.analysis.models import TemporalAnalysisResponse, TemporalDataPoint, TemporalPatterns

logger = logging.getLogger(__name__)


class DetailedTemporalAnalyzer:
    """
    Detailed Temporal Analyzer for advanced activity pattern analysis.
    
    Features:
    - Activity timeline analysis with multiple granularities
    - Peak activity detection and trend analysis
    - Advanced pattern recognition
    - Time-based user activity analysis
    """
    
    def __init__(self, data_facade: Optional[AnalysisDataFacade] = None):
        """Initialize detailed temporal analyzer with data facade."""
        if data_facade is None:
            self.data_facade = get_analysis_data_facade()
            self._owns_facade = True
        else:
            self.data_facade = data_facade
            self._owns_facade = False
            
        logger.info("DetailedTemporalAnalyzer initialized")
    
    def analyze(
        self,
        channel_name: Optional[str] = None,
        days_back: Optional[int] = None,
        granularity: str = "daily"
    ) -> Optional[TemporalAnalysisResponse]:
        """
        Analyze temporal patterns in messages using advanced techniques.
        
        Args:
            channel_name: Optional channel name to filter by
            days_back: Number of days to look back
            granularity: Time granularity ("hourly", "daily", "weekly")
            
        Returns:
            TemporalAnalysisResponse with analysis results
        """
        try:
            logger.info(f"Starting detailed temporal analysis for channel: {channel_name}")
            
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
                logger.warning("No messages found for detailed temporal analysis")
                return None
            
            # Analyze temporal patterns using legacy logic
            temporal_data = self._analyze_temporal_data(messages, granularity)
            patterns = self._analyze_patterns(temporal_data, granularity)
            
            # Convert to proper model objects
            temporal_data_points = [
                TemporalDataPoint(
                    period=item['period'],
                    message_count=item['message_count'],
                    unique_users=item['unique_users']
                )
                for item in temporal_data
            ]
            
            temporal_patterns = TemporalPatterns(
                total_messages=patterns.get('total_messages', 0),
                avg_messages_per_period=patterns.get('avg_messages_per_period', 0.0),
                max_messages_in_period=patterns.get('max_messages_in_period', 0),
                min_messages_in_period=patterns.get('min_messages_in_period', 0),
                most_active_period=patterns.get('most_active_period'),
                message_trend=patterns.get('message_trend', 'stable'),
                trend_percentage=patterns.get('trend_percentage', 0.0),
                trend_timeframe=patterns.get('trend_timeframe', 'over analysis period'),
                peak_user_count=patterns.get('peak_user_count', 0),
                total_periods=patterns.get('total_periods', 0),
                # Enhanced pattern analysis fields
                peak_hours=patterns.get('peak_hours', []),
                peak_days=patterns.get('peak_days', []),
                quiet_periods=patterns.get('quiet_periods', []),
                trends=patterns.get('trends', [])
            )
            
            return TemporalAnalysisResponse(
                temporal_data=temporal_data_points,
                patterns=temporal_patterns,
                capabilities_used=["detailed_temporal_analysis"]
            )
            
        except Exception as e:
            logger.error(f"Detailed temporal analysis failed: {e}")
            return None
    
    def _analyze_temporal_data(self, messages: List[Dict[str, Any]], granularity: str) -> List[Dict[str, Any]]:
        """Analyze temporal data based on granularity using legacy logic."""
        try:
            period_counts = Counter()
            user_counts = Counter()
            
            for msg in messages:
                timestamp = msg.timestamp
                if not timestamp:
                    continue
                
                try:
                    dt = timestamp if isinstance(timestamp, datetime) else datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))
                    
                    if granularity == "hourly":
                        period = dt.strftime('%Y-%m-%d %H:00')
                    elif granularity == "daily":
                        period = dt.strftime('%Y-%m-%d')
                    elif granularity == "weekly":
                        period = dt.strftime('%Y-W%W')
                    else:
                        period = dt.strftime('%Y-%m-%d')
                    
                    period_counts[period] += 1
                    
                    # Count unique users per period
                    author = msg.author_name or 'unknown'
                    # Get all messages in this period and count unique users
                    period_users = set()
                    for m in messages:
                        if not m.timestamp:
                            continue
                        try:
                            m_dt = m.timestamp if isinstance(m.timestamp, datetime) else datetime.fromisoformat(str(m.timestamp).replace('Z', '+00:00'))
                            m_period = None
                            if granularity == "hourly":
                                m_period = m_dt.strftime('%Y-%m-%d %H:00')
                            elif granularity == "daily":
                                m_period = m_dt.strftime('%Y-%m-%d')
                            elif granularity == "weekly":
                                m_period = m_dt.strftime('%Y-W%W')
                            else:
                                m_period = m_dt.strftime('%Y-%m-%d')
                            
                            if m_period == period:
                                period_users.add(m.author_name or 'unknown')
                        except Exception:
                            continue
                    
                    user_counts[period] = len(period_users)
                    
                except Exception as e:
                    logger.debug(f"Failed to parse timestamp {timestamp}: {e}")
                    continue
            
            # Convert to list format
            temporal_data = []
            for period in sorted(period_counts.keys()):
                temporal_data.append({
                    'period': period,
                    'message_count': period_counts[period],
                    'unique_users': user_counts[period]
                })
            
            return temporal_data
            
        except Exception as e:
            logger.error(f"Failed to analyze temporal data: {e}")
            return []
    
    def _analyze_patterns(self, temporal_data: List[Dict[str, Any]], granularity: str) -> Dict[str, Any]:
        """Analyze patterns in temporal data using legacy logic."""
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
            
            # Peak user count
            peak_user_count = max(item['unique_users'] for item in temporal_data) if temporal_data else 0
            
            # Enhanced pattern analysis for template compatibility
            peak_hours = []
            peak_days = []
            quiet_periods = []
            trends = []
            
            # Analyze peak activity periods
            if granularity == "hourly":
                # For hourly data, find peak hours
                for item in temporal_data:
                    if item['message_count'] >= max_messages * 0.8:  # 80% of peak
                        try:
                            hour = item['period'].split(' ')[1].split(':')[0]
                            peak_hours.append(hour)
                        except:
                            pass
            elif granularity == "daily":
                # For daily data, find peak days
                for item in temporal_data:
                    if item['message_count'] >= max_messages * 0.8:  # 80% of peak
                        try:
                            from datetime import datetime
                            dt = datetime.fromisoformat(item['period'])
                            day_name = dt.strftime('%A')
                            peak_days.append(day_name)
                        except:
                            pass
            elif granularity == "weekly":
                # For weekly data, find peak weeks
                for item in temporal_data:
                    if item['message_count'] >= max_messages * 0.8:  # 80% of peak
                        peak_days.append(f"Week {item['period']}")
            
            # Find quiet periods (lowest activity)
            quiet_threshold = min_messages + (max_messages - min_messages) * 0.2  # Bottom 20%
            for item in temporal_data:
                if item['message_count'] <= quiet_threshold:
                    quiet_periods.append(item['period'])
            
            # Generate trend insights
            if message_trend == "increasing":
                trends.append(f"Activity is trending upward with {trend_percentage:.1f}% increase")
            elif message_trend == "decreasing":
                trends.append(f"Activity is trending downward with {trend_percentage:.1f}% decrease")
            else:
                trends.append("Activity remains relatively stable")
            
            # Add more insights based on data
            if max_messages > avg_messages_per_period * 2:
                trends.append(f"Peak activity ({max_messages} messages) is significantly higher than average ({avg_messages_per_period:.1f})")
            
            if len(set(item['unique_users'] for item in temporal_data)) > 1:
                trends.append(f"Activity involves {peak_user_count} unique users at peak")
            
            return {
                'total_messages': total_messages,
                'avg_messages_per_period': round(avg_messages_per_period, 1),
                'most_active_period': most_active_period,
                'max_messages_in_period': max_messages,
                'min_messages_in_period': min_messages,
                'message_trend': message_trend,
                'trend_percentage': round(trend_percentage, 1),
                'trend_timeframe': f"over {granularity} periods",
                'peak_user_count': peak_user_count,
                'total_periods': total_periods,
                # Enhanced pattern data for template compatibility
                'peak_hours': peak_hours,
                'peak_days': peak_days,
                'quiet_periods': quiet_periods,
                'trends': trends
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze patterns: {e}")
            return {}
    
    def __del__(self):
        """Cleanup if we own the facade."""
        if hasattr(self, '_owns_facade') and self._owns_facade and hasattr(self, 'data_facade'):
            try:
                self.data_facade.close()
            except:
                pass 