"""
Discord Template Engine

Jinja2-based template rendering engine for Discord bot analysis responses.
Provides template loading, custom filters, and chart generation integration for Discord message formatting.
"""
from jinja2 import Environment, FileSystemLoader, Template
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
import yaml

from pepino.logging_config import get_logger
from pepino.analysis.visualization.charts import (
    create_activity_graph,
    create_channel_activity_pie, 
    create_user_activity_bar,
    create_word_cloud
)

logger = get_logger(__name__)


class TemplateEngine:
    """
    Configurable Jinja2 template engine for multi-domain template rendering.
    
    Provides template loading, customizable filters, and chart generation integration
    for creating rich formatted outputs across different domains (Discord, reports, etc.).
    
    Features:
    - Template loading from configurable directory
    - Pluggable filter system for domain-specific formatting
    - Chart generation functions available in templates
    - Custom number and date formatting
    - Error handling and logging
    - Support for multiple output formats
    
    Usage:
        # General usage
        engine = TemplateEngine()
        result = engine.render_template("reports/monthly_summary.md.j2", data=report_data)
        
        # Discord-specific usage (with Discord filters loaded by default)
        engine = TemplateEngine()
        result = engine.render_template("discord/channel_analysis.md.j2", data=analysis_data)
    """

    def __init__(self, templates_dir: str = "templates", analyzers: Optional[Dict[str, Any]] = None):
        """
        Initialize the template engine.
        
        Args:
            templates_dir: Base directory for templates (default: "templates")
            analyzers: Optional dict of analyzer instances for template use
        """
        
        templates_path = Path(templates_dir)
        if not templates_path.exists():
            # Try relative to project root
            templates_path = Path.cwd() / templates_dir
        
        if not templates_path.exists():
            logger.warning(f"Templates directory not found: {templates_dir}")
            # Create basic environment anyway
            self.env = Environment()
        else:
            logger.info(f"Loading templates from: {templates_path}")
            self.env = Environment(
                loader=FileSystemLoader(str(templates_path)),
                trim_blocks=True,
                lstrip_blocks=True
            )
        
        # Store analyzer references for template use
        self.analyzers = analyzers or {}
        
        # Add custom filters that match current formatting
        self.env.filters.update({
            'format_number': self._format_number,
            'format_timestamp': self._format_timestamp,
            'format_percentage': self._format_percentage,
            'discord_code': self._discord_code_block,
            'discord_bold': self._discord_bold,
            'truncate_smart': self._truncate_smart,
            'sort_by': self._sort_by,
            'top_n': self._top_n,
            'format_trend': self._format_trend,
            # CLI-specific filters
            'cli_safe': self._cli_safe_text,
            'terminal_width': self._terminal_width_wrap
        })
        
        # Add custom functions
        self.env.globals.update({
            'now': datetime.now,
            'range': range,
            'enumerate': enumerate,
            'len': len,
            'max': max,
            'min': min,
            'sum': sum,
            # Chart generation functions
            'create_activity_chart': self._create_activity_chart,
            'create_pie_chart': self._create_pie_chart,
            'create_bar_chart': self._create_bar_chart,
            'create_wordcloud': self._create_wordcloud
        })
        
        # Make analyzer functions available in templates
        self._register_analyzer_functions()
    
    def render_template(self, template_name: str, **kwargs) -> str:
        """
        Render a template with the provided data.
        
        Args:
            template_name: Template file path relative to templates directory
            **kwargs: Data to pass to the template
            
        Returns:
            Rendered template as string
            
        Raises:
            TemplateNotFound: If template file doesn't exist
            TemplateSyntaxError: If template has syntax errors
        """
        try:
            template = self.env.get_template(template_name)
            return template.render(**kwargs)
        except Exception as e:
            logger.error(f"Error rendering template {template_name}: {str(e)}", exc_info=True)
            return f"❌ Error rendering template: {str(e)}"

    def render_string(self, template_string: str, **kwargs) -> str:
        """
        Render a template from a string.
        
        Args:
            template_string: Template content as string
            **kwargs: Data to pass to the template
            
        Returns:
            Rendered template as string
        """
        try:
            template = Template(template_string)
            return template.render(**kwargs)
        except Exception as e:
            logger.error(f"Error rendering template string: {str(e)}", exc_info=True)
            return f"❌ Error rendering template: {str(e)}"

    def _register_analyzer_functions(self):
        """Register analyzer functions for use in templates."""
        if self.analyzers:
            for name, analyzer in self.analyzers.items():
                self.env.globals[f"{name}_analyze"] = analyzer.analyze
                # Add any other analyzer methods as needed

    # Custom filters that match current formatting behavior
    def _format_number(self, value: Union[int, float]) -> str:
        """Format number with commas for Discord display."""
        if value is None:
            return "0"
        if isinstance(value, (int, float)):
            return f"{value:,.0f}" if isinstance(value, int) or value.is_integer() else f"{value:,.1f}"
        return str(value)
    
    def _format_timestamp(self, timestamp: str) -> str:
        """Format timestamp for Discord display."""
        if not timestamp:
            return "Never"
        try:
            # Handle different timestamp formats
            if 'T' in timestamp:
                # ISO format
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            else:
                # Assume it's already formatted or a different format
                dt = datetime.fromisoformat(timestamp)
            return dt.strftime("%Y-%m-%d at %H:%M")
        except Exception:
            # Fallback - just return the timestamp as-is
            return timestamp
    
    def _format_percentage(self, value: float, decimals: int = 1) -> str:
        """Format value as percentage."""
        if value is None:
            return "0.0%"
        return f"{value:.{decimals}f}%"
    
    def _discord_code_block(self, text: str, language: str = "") -> str:
        """Wrap text in Discord code block."""
        return f"```{language}\n{text}\n```"
    
    def _discord_bold(self, text: str) -> str:
        """Make text bold for Discord."""
        return f"**{text}**"
    
    def _truncate_smart(self, text: str, length: int = 100) -> str:
        """Smart truncation at word boundaries."""
        if not text or len(text) <= length:
            return text
        return text[:length].rsplit(' ', 1)[0] + "..."
    
    def _sort_by(self, items: List[Dict], key: str, reverse: bool = True) -> List[Dict]:
        """Sort list of dictionaries by specified key."""
        if not items:
            return []
        return sorted(items, key=lambda x: x.get(key, 0), reverse=reverse)
    
    def _top_n(self, items: List, n: int = 10) -> List:
        """Get top N items from a list."""
        if not items:
            return []
        return items[:n]
    
    def _format_trend(self, trend: str, percentage: float) -> str:
        """Format trend with emoji for Discord."""
        emoji = "📈" if trend == "increasing" else "📉" if trend == "decreasing" else "➡️"
        return f"{emoji} {trend.title()} ({percentage:+.1f}%)"
    
    def _create_activity_chart(self, dates: List[str], counts: List[int], 
                               title: str = "Activity Over Time") -> str:
        """
        Create activity chart and return file path.
        
        Args:
            dates: List of date strings
            counts: List of message counts
            title: Chart title
            
        Returns:
            File path to generated chart or empty string if failed
        """
        try:
            return create_activity_graph(dates, counts, title)
        except Exception as e:
            logger.error(f"Error creating activity chart: {e}")
            return ""
    
    def _create_pie_chart(self, names: List[str], counts: List[int], 
                          title: str = "Distribution") -> str:
        """
        Create pie chart and return file path.
        
        Args:
            names: List of category names
            counts: List of values
            title: Chart title
            
        Returns:
            File path to generated chart or empty string if failed
        """
        try:
            return create_channel_activity_pie(names, counts, title)
        except Exception as e:
            logger.error(f"Error creating pie chart: {e}")
            return ""
    
    def _create_bar_chart(self, names: List[str], counts: List[int], 
                          title: str = "Activity Ranking") -> str:
        """
        Create bar chart and return file path.
        
        Args:
            names: List of category names
            counts: List of values
            title: Chart title
            
        Returns:
            File path to generated chart or empty string if failed
        """
        try:
            return create_user_activity_bar(names, counts, title)
        except Exception as e:
            logger.error(f"Error creating bar chart: {e}")
            return ""
    
    def _create_wordcloud(self, text_data: Union[str, List[str]], 
                          title: str = "Word Cloud") -> str:
        """
        Create word cloud and return file path.
        
        Args:
            text_data: Text content for word cloud
            title: Chart title
            
        Returns:
            File path to generated chart or empty string if failed
        """
        try:
            if isinstance(text_data, list):
                text_data = " ".join(text_data)
            return create_word_cloud(text_data, title)
        except Exception as e:
            logger.error(f"Error creating wordcloud: {e}")
            return ""

    def _cli_safe_text(self, text: str) -> str:
        """
        Make text CLI-safe by handling terminal compatibility.
        
        Args:
            text: Input text that may contain emojis or special characters
            
        Returns:
            CLI-safe text
        """
        if not text:
            return ""
        
        # Option 1: Keep emojis (most terminals support them now)
        return text
        
        # Option 2: Replace with ASCII equivalents (uncomment if needed)
        # emoji_replacements = {
        #     '📊': '[STATS]',
        #     '👥': '[USERS]',
        #     '📈': '[METRICS]',
        #     '🧠': '[TOPICS]',
        #     '⏰': '[TIME]',
        #     '👤': '[USER]',
        #     '✅': '[OK]',
        #     '❌': '[ERROR]'
        # }
        # 
        # for emoji, replacement in emoji_replacements.items():
        #     text = text.replace(emoji, replacement)
        # 
        # return text

    def _terminal_width_wrap(self, text: str, width: int = 80) -> str:
        """
        Wrap text to terminal width for better readability.
        
        Args:
            text: Text to wrap
            width: Maximum line width (default: 80)
            
        Returns:
            Wrapped text
        """
        if not text:
            return ""
        
        try:
            import textwrap
            return textwrap.fill(text, width=width, 
                               break_long_words=False, 
                               break_on_hyphens=False)
        except ImportError:
            # Fallback if textwrap not available
            return text 