"""
Database Analysis Service

Handles database statistics and health reporting operations. Focuses specifically 
on database analysis domain.
"""

import logging
from typing import Dict, Any, Optional

from .base_service import BaseAnalysisService, OutputFormat

logger = logging.getLogger(__name__)


class DatabaseAnalysisService(BaseAnalysisService):
    """
    Specialized service for database analysis operations.
    
    Handles:
    - Database statistics
    - Database health reporting
    - Table statistics
    - Data summary
    """
    
    def _create_analyzers(self) -> Dict[str, Any]:
        """Create database-specific analyzers."""
        try:
            from pepino.analysis.helpers.database_analyzer import DatabaseAnalyzer
            
            return {
                'database': DatabaseAnalyzer(self.data_facade),
            }
        except ImportError as e:
            logger.warning(f"Could not load database analyzer classes: {e}")
            return {}
    
    def database_stats(self, output_format: OutputFormat = "cli") -> str:
        """
        Generate database statistics and health report.
        
        Args:
            output_format: Output format ("cli" or "discord")
            
        Returns:
            Formatted database statistics string
        """
        database_analyzer = self.analyzers.get('database')
        
        if not database_analyzer:
            return "❌ Analysis failed: Database analyzer not available"
        
        # Get database analysis
        analysis_result = database_analyzer.analyze()
        
        if not analysis_result or not analysis_result.success:
            return "❌ Database analysis failed"
        
        return self.render_template(
            "database_stats",
            output_format=output_format,
            database_info=analysis_result.database_info,
            table_stats=analysis_result.table_stats,
            summary=analysis_result.summary
        ) 