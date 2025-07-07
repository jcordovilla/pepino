"""
Analysis Service (Legacy - Use UnifiedAnalysisService for new code)

This module is maintained for backward compatibility. For new code, use the
modularized services in pepino.analysis.services package.

The new modularized approach provides:
- Better separation of concerns
- Easier testing and maintenance
- Cleaner code organization
- Domain-specific services

See pepino.analysis.services.unified_analysis_service for the recommended approach.
"""

import logging
from typing import Dict, Any, Optional, Literal, List
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Generator

from pepino.config import Settings

# Import the new unified service for backward compatibility
from .services.unified_analysis_service import UnifiedAnalysisService, analysis_service

logger = logging.getLogger(__name__)

OutputFormat = Literal["cli", "discord"]


class AnalysisService(UnifiedAnalysisService):
    """
    Legacy AnalysisService - now inherits from UnifiedAnalysisService.
    
    This class is maintained for backward compatibility. For new code,
    use UnifiedAnalysisService directly or the specialized services.
    
    The new modularized approach provides better separation of concerns
    and easier maintenance.
    """
    
    def __init__(self, db_path: Optional[str] = None, base_filter: Optional[str] = None):
        """
        Initialize analysis service with optional database path and base filter.
        
        Args:
            db_path: Optional database path (uses settings default if None)
            base_filter: Optional base filter for data queries
        """
        super().__init__(db_path, base_filter)
        logger.warning("AnalysisService is deprecated. Use UnifiedAnalysisService or specialized services instead.")


# Re-export the analysis_service context manager for backward compatibility
__all__ = ['AnalysisService', 'analysis_service'] 