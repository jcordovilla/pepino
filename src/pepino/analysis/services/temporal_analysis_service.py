"""
Temporal Analysis Service

Handles all temporal-related analysis operations including detailed temporal analysis,
activity trends, and time-based patterns. Focuses specifically on temporal analysis domain.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .base_service import BaseAnalysisService, OutputFormat

logger = logging.getLogger(__name__)


class TemporalAnalysisService(BaseAnalysisService):
    """
    Specialized service for temporal analysis operations.
    
    Handles:
    - Detailed temporal analysis
    - Activity trends analysis
    - Time-based patterns
    - Temporal data visualization
    """
    
    def _create_analyzers(self) -> Dict[str, Any]:
        """Create temporal-specific analyzers."""
        try:
            from pepino.analysis.helpers.temporal_analyzer import TemporalAnalyzer
            from pepino.analysis.helpers.detailed_temporal_analyzer import DetailedTemporalAnalyzer
            from pepino.analysis.helpers.activity_trends_analyzer import ActivityTrendsAnalyzer
            
            return {
                'temporal': TemporalAnalyzer(self.data_facade),
                'detailed_temporal': DetailedTemporalAnalyzer(self.data_facade),
                'activity_trends': ActivityTrendsAnalyzer(self.data_facade),
            }
        except ImportError as e:
            logger.warning(f"Could not load temporal analyzer classes: {e}")
            return {}
    
    def detailed_temporal_analysis(self, channel_name: Optional[str] = None, 
                                  days_back: Optional[int] = None, granularity: str = "daily", 
                                  output_format: OutputFormat = "cli") -> str:
        """
        Perform a detailed temporal analysis.
        
        Args:
            channel_name: The channel to analyze (or None for all channels)
            days_back: Number of days to look back
            granularity: Time granularity ("hourly", "daily", "weekly")
            output_format: "cli" or "discord"
            
        Returns:
            Rendered analysis string
        """
        detailed_temporal_analyzer = self.analyzers.get('detailed_temporal')
        
        if not detailed_temporal_analyzer:
            return "❌ Analysis failed: Detailed temporal analyzer not available"
        
        analysis = detailed_temporal_analyzer.analyze(channel_name=channel_name, days_back=days_back, granularity=granularity)
        if not analysis:
            return f"❌ No data found for temporal analysis"

        return self.render_template(
            "detailed_temporal_analysis",
            output_format=output_format,
            temporal_data=analysis.temporal_data,
            patterns=analysis.patterns,
            capabilities_used=analysis.capabilities_used,
            granularity=granularity,
            channel_name=channel_name,
            days_back=days_back
        )
    
    def activity_trends_analysis(self, channel_name: Optional[str] = None, 
                                days_back: Optional[int] = None, output_format: OutputFormat = "cli") -> str:
        """
        Perform activity trends analysis with chart generation.
        
        Args:
            channel_name: The channel to analyze (or None for all channels)
            days_back: Number of days to look back
            output_format: "cli" or "discord" (chart only works in discord)
            
        Returns:
            Rendered analysis string with embedded chart
        """
        activity_trends_analyzer = self.analyzers.get('activity_trends')
        
        if not activity_trends_analyzer:
            return "❌ Analysis failed: Activity trends analyzer not available"
        
        analysis = activity_trends_analyzer.analyze(channel_name=channel_name, days_back=days_back, output_format=output_format)
        if not analysis:
            return f"❌ No data found for activity trends analysis"

        # For CLI format, don't include chart data
        if output_format == "cli":
            analysis['chart_base64'] = None

        return self.render_template(
            "activity_trends",
            output_format=output_format,
            patterns=analysis['patterns'],
            chart_base64=analysis['chart_base64'],
            cross_channel_stats=analysis['cross_channel_stats'],
            time_period=analysis['time_period']
        ) 