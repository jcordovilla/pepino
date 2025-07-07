"""
Server Analysis Service

Handles server overview analysis operations including comprehensive server statistics,
health metrics, and engagement analysis. Focuses specifically on server-level analysis domain.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .base_service import BaseAnalysisService, OutputFormat

logger = logging.getLogger(__name__)


class ServerAnalysisService(BaseAnalysisService):
    """
    Specialized service for server analysis operations.
    
    Handles:
    - Server overview analysis
    - Comprehensive server statistics
    - Server health metrics
    - Cross-channel engagement analysis
    """
    
    def _create_analyzers(self) -> Dict[str, Any]:
        """Create server-specific analyzers."""
        try:
            from pepino.analysis.helpers.server_overview_analyzer import ServerOverviewAnalyzer
            
            return {
                'server_overview': ServerOverviewAnalyzer(
                    self.data_facade.message_repository,
                    self.data_facade.user_repository,
                    self.data_facade.channel_repository
                ),
            }
        except ImportError as e:
            logger.warning(f"Could not load server analyzer classes: {e}")
            return {}
    
    def server_overview_analysis(self, days_back: Optional[int] = None, 
                                output_format: OutputFormat = "cli") -> str:
        """
        Perform comprehensive server overview analysis.
        
        Args:
            days_back: Number of days to look back (None for all time)
            output_format: "cli" or "discord"
            
        Returns:
            Rendered analysis string
        """
        server_overview_analyzer = self.analyzers.get('server_overview')
        
        if not server_overview_analyzer:
            return "❌ Analysis failed: Server overview analyzer not available"
        
        analysis = server_overview_analyzer.analyze(days=days_back)
        if not analysis:
            return f"❌ No data found for server overview analysis"

        return self.render_template(
            "server_overview",
            output_format=output_format,
            total_messages=analysis.total_messages,
            total_users=analysis.total_users,
            total_channels=analysis.total_channels,
            active_users=analysis.active_users,
            messages_per_day=analysis.messages_per_day,
            messages_per_user=analysis.messages_per_user,
            top_channels=analysis.top_channels,
            top_contributors=analysis.top_contributors,
            activity_trends=analysis.activity_trends,
            analysis_period_days=analysis.analysis_period_days
        ) 