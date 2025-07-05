"""
Temporal analysis plugins.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from ..logging_config import get_logger
from .data_facade import AnalysisDataFacade, get_analysis_data_facade
from .models import (
    AnalysisErrorResponse,
    TemporalAnalysisResponse,
    TemporalDataPoint,
    TemporalPatterns,
)

logger = get_logger(__name__)


class TemporalAnalyzer:
    """
    Temporal analysis plugin.
    
    Analyzes temporal patterns in message data using the data facade pattern
    for centralized repository management and proper separation of concerns.
    
    All database operations are abstracted through the data facade for proper
    dependency injection support and testability.
    """

    def __init__(self, data_facade: Optional[AnalysisDataFacade] = None):
        """Initialize temporal analyzer with data facade."""
        if data_facade is None:
            self.data_facade = get_analysis_data_facade()
            self._owns_facade = True
        else:
            self.data_facade = data_facade
            self._owns_facade = False
            
        logger.info("TemporalAnalyzer initialized with data facade pattern")

    def analyze(
        self, **kwargs
    ) -> Union[TemporalAnalysisResponse, AnalysisErrorResponse]:
        """
        Analyze temporal patterns based on provided parameters.

        Expected kwargs:
            - channel_name: str (optional)
            - user_name: str (optional)
            - days_back: int (optional, default 30)
            - granularity: str (optional, 'hour'|'day'|'week', default 'day')
        """
        try:
            logger.info(f"Starting temporal analysis with kwargs: {kwargs}")
            
            # Get temporal data using data facade (now synchronous)
            temporal_data = self.data_facade.message_repository.get_temporal_analysis_data(
                channel_name=kwargs.get("channel_name"),
                user_name=kwargs.get("user_name"),
                days_back=kwargs.get("days_back"),
                granularity=kwargs.get("granularity", "day"),
            )

            logger.info(f"Retrieved temporal data: {len(temporal_data) if temporal_data else 0} records")

            if not temporal_data:
                return AnalysisErrorResponse(
                    error="No temporal data found for analysis",
                    plugin="TemporalAnalyzer",
                )

            # Process temporal patterns
            patterns = self._analyze_patterns(temporal_data)

            # Create and return response
            return TemporalAnalysisResponse(
                temporal_data=[TemporalDataPoint(**point) for point in temporal_data],
                patterns=TemporalPatterns(**patterns),
            )

        except Exception as e:
            logger.error(f"Error in temporal analysis: {type(e).__name__}: {e}", exc_info=True)
            return AnalysisErrorResponse(
                error=f"Analysis failed: {str(e)}", plugin="TemporalAnalyzer"
            )

    def _analyze_patterns(
        self, temporal_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze temporal patterns in the provided data."""
        if not temporal_data:
            return {
                "total_messages": 0,
                "avg_messages_per_period": 0.0,
                "max_messages_in_period": 0,
                "min_messages_in_period": 0,
                "most_active_period": None,
                "message_trend": "stable",
                "trend_percentage": 0.0,
                "peak_user_count": 0,
                "total_periods": 0,
            }

        message_counts = [item["message_count"] for item in temporal_data]
        user_counts = [item["unique_users"] for item in temporal_data]

        # Calculate basic statistics
        total_messages = sum(message_counts)
        avg_messages_per_period = (
            total_messages / len(message_counts) if message_counts else 0
        )
        max_messages = max(message_counts) if message_counts else 0
        min_messages = min(message_counts) if message_counts else 0

        # Find most active period
        most_active_period = (
            temporal_data[message_counts.index(max_messages)]["period"]
            if message_counts
            else None
        )

        # Calculate trend (comparing first half vs second half)
        mid_point = len(message_counts) // 2
        if mid_point > 0:
            first_half_avg = sum(message_counts[:mid_point]) / mid_point
            second_half_avg = sum(message_counts[mid_point:]) / (
                len(message_counts) - mid_point
            )

            if first_half_avg > 0:
                trend_percentage = (
                    (second_half_avg - first_half_avg) / first_half_avg
                ) * 100
                message_trend = (
                    "increasing" if second_half_avg > first_half_avg else "decreasing"
                )
            else:
                trend_percentage = 0
                message_trend = "stable"
        else:
            message_trend = "stable"
            trend_percentage = 0

        return {
            "total_messages": total_messages,
            "avg_messages_per_period": round(avg_messages_per_period, 1),
            "max_messages_in_period": max_messages,
            "min_messages_in_period": min_messages,
            "most_active_period": most_active_period,
            "message_trend": message_trend,
            "trend_percentage": round(trend_percentage, 1),
            "peak_user_count": max(user_counts) if user_counts else 0,
            "total_periods": len(temporal_data),
        }
