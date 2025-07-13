"""
Topic Analysis Service

Handles all topic-related analysis operations including detailed topic analysis
and topic extraction. Focuses specifically on topic analysis domain.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .base_service import BaseAnalysisService, OutputFormat

logger = logging.getLogger(__name__)


class TopicAnalysisService(BaseAnalysisService):
    """
    Specialized service for topic analysis operations.
    
    Handles:
    - Detailed topic analysis
    - Topic extraction and clustering
    - Content analysis
    - Topic trends
    """
    
    def _create_analyzers(self) -> Dict[str, Any]:
        """Create topic-specific analyzers."""
        try:
            from pepino.analysis.helpers.topic_analyzer import TopicAnalyzer
            from pepino.analysis.helpers.detailed_topic_analyzer import DetailedTopicAnalyzer
            
            return {
                'topic': TopicAnalyzer(self.data_facade),
                'detailed_topic': DetailedTopicAnalyzer(self.data_facade),
            }
        except ImportError as e:
            logger.warning(f"Could not load topic analyzer classes: {e}")
            return {}
    
    def detailed_topic_analysis(self, channel_name: Optional[str] = None, n_topics: int = 10, 
                               days_back: Optional[int] = None, output_format: OutputFormat = "cli") -> str:
        """
        Perform a detailed topic analysis.
        
        Args:
            channel_name: The channel to analyze (or None for all channels)
            n_topics: Number of topics to extract
            days_back: Number of days to look back
            output_format: "cli" or "discord"
            
        Returns:
            Rendered analysis string
        """
        detailed_topic_analyzer = self.analyzers.get('detailed_topic')
        
        if not detailed_topic_analyzer:
            return "❌ Analysis failed: Detailed topic analyzer not available"
        
        analysis = detailed_topic_analyzer.analyze(channel_name=channel_name, n_topics=n_topics, days_back=days_back)
        if not analysis:
            return f"❌ No data found for topic analysis"

        return self.render_template(
            "detailed_topic_analysis",
            output_format=output_format,
            topics=analysis.topics,
            message_count=analysis.message_count,
            capabilities_used=analysis.capabilities_used,
            n_topics=analysis.n_topics,
            days_back=analysis.days_back,
            channel_name=analysis.channel_name,
            summary=analysis.summary
        ) 