"""
Base Analysis Service

Provides common functionality and contract for all specialized analysis services.
"""

import logging
from typing import Dict, Any, Optional, Literal
from contextlib import contextmanager
from datetime import datetime

from pepino.config import Settings
from pepino.analysis.helpers.data_facade import get_analysis_data_facade
from pepino.analysis.templates.template_engine import TemplateEngine

logger = logging.getLogger(__name__)

OutputFormat = Literal["cli", "discord"]


class BaseAnalysisService:
    """
    Base class for all analysis services.
    
    Provides common functionality:
    - Data facade management
    - Template engine setup
    - Analyzer initialization
    - Common utility methods
    """
    
    def __init__(self, db_path: Optional[str] = None, base_filter: Optional[str] = None):
        """
        Initialize base analysis service.
        
        Args:
            db_path: Optional database path (uses settings default if None)
            base_filter: Optional base filter for data queries
        """
        self.settings = Settings()
        self.db_path = db_path or self.settings.database_sqlite_path
        self.base_filter = base_filter or self.settings.analysis_base_filter_sql
        
        # Lazy initialization - only create when needed
        self._data_facade = None
        self._analyzers = None
        self._template_engine = None
        self._nlp_service = None
        
        logger.debug(f"{self.__class__.__name__} initialized for {self.db_path}")
    
    @property
    def data_facade(self):
        """Get data facade instance (lazy initialization)."""
        if self._data_facade is None:
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
        """Create analyzer instances. Override in subclasses for specific analyzers."""
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
    
    def render_template(self, template_name: str, output_format: OutputFormat = "cli", **kwargs) -> str:
        """
        Render a template with the specified format.
        
        Args:
            template_name: Base template name (without extension)
            output_format: Output format ("cli" or "discord")
            **kwargs: Data to pass to the template
            
        Returns:
            Rendered template string
        """
        # Determine file extension based on format
        extension = "txt" if output_format == "cli" else "md"
        full_template_name = f"outputs/{output_format}/{template_name}.{extension}.j2"
        
        return self.template_engine.render_template(
            full_template_name,
            format_number=lambda v: f"{v:,}",
            now=datetime.now,
            **kwargs
        )
    
    def close(self):
        """Close the analysis service and clean up resources."""
        if self._data_facade:
            self._data_facade.close()
        logger.debug(f"{self.__class__.__name__} closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close() 