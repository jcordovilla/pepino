"""
Discord Template Engine

Jinja2-based template rendering engine for Discord bot analysis responses.
Provides template loading, custom filters, chart generation integration, analyzer helpers,
and NLP capabilities for Discord message formatting.
"""
from jinja2 import Environment, FileSystemLoader, Template
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from pepino.analysis.helpers.data_facade import AnalysisDataFacade
    from pepino.analysis.nlp_analyzer import NLPService
    from pepino.analysis.helpers.channel_analyzer import ChannelAnalyzer
    from pepino.analysis.helpers.user_analyzer import UserAnalyzer
    from pepino.analysis.helpers.topic_analyzer import TopicAnalyzer
    from pepino.analysis.helpers.temporal_analyzer import TemporalAnalyzer

# Type alias for supported analyzer types
AnalyzerType = Union['ChannelAnalyzer', 'UserAnalyzer', 'TopicAnalyzer', 'TemporalAnalyzer']
from datetime import datetime
import json
import yaml

from pepino.logging_config import get_logger
# Optional visualization imports
try:
    from pepino.analysis.visualization.charts import (
        create_activity_graph,
        create_channel_activity_pie, 
        create_user_activity_bar,
        create_word_cloud
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    # Visualization not available - create dummy functions
    def create_activity_graph(*args, **kwargs): return "Chart not available"
    def create_channel_activity_pie(*args, **kwargs): return "Chart not available"
    def create_user_activity_bar(*args, **kwargs): return "Chart not available"
    def create_word_cloud(*args, **kwargs): return "Chart not available"
    VISUALIZATION_AVAILABLE = False

logger = get_logger(__name__)


class TemplateEngine:
    """
    Configurable Jinja2 template engine for multi-domain template rendering with analyzer helpers.
    
    Provides template loading, customizable filters, chart generation integration,
    analyzer helper functions, and NLP capabilities for creating rich formatted outputs
    across different domains (Discord, reports, etc.).
    
    Features:
    - Template loading from configurable directory
    - Pluggable filter system for domain-specific formatting
    - Chart generation functions available in templates
    - Analyzer helper functions for dynamic analysis
    - NLP helper functions for text analysis
    - Message data access for template-driven analysis
    - Custom number and date formatting
    - Error handling and logging
    - Support for multiple output formats
    
    Usage:
        # With analyzer helpers
        engine = TemplateEngine(analyzers=analyzer_dict, data_facade=facade)
        result = engine.render_template("discord/channel_analysis.md.j2", 
                                       data=analysis_data, 
                                       messages=message_list)
        
        # Templates can now use:
        # {{ analyze_sentiment(messages | join(' ')) }}
        # {{ extract_concepts(data.content) }}
        # {{ channel_analyzer.get_top_users(limit=5) }}
    """

    def __init__(self, templates_dir: str = "templates", analyzers: Optional[Dict[str, AnalyzerType]] = None, 
                 data_facade: Optional['AnalysisDataFacade'] = None, nlp_service: Optional['NLPService'] = None):
        """
        Initialize the template engine with analyzer helpers and NLP capabilities.
        
        Args:
            templates_dir: Base directory for templates (default: "templates")
            analyzers: Optional dict of analyzer instances for template use
            data_facade: Optional data facade for database access
            nlp_service: Optional NLP service for text analysis
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
        
        # Template cache for performance
        self._template_cache = {}
        self._cache_enabled = True
        
        # Store analyzer and service references for template use
        self.analyzers: Dict[str, AnalyzerType] = analyzers or {}
        self.data_facade: Optional['AnalysisDataFacade'] = data_facade
        self.nlp_service: Optional['NLPService'] = nlp_service
        
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
            'terminal_width': self._terminal_width_wrap,
            'split_name': self._split_name,
            # New NLP filters
            'extract_concepts': self._filter_extract_concepts,
            'analyze_sentiment': self._filter_analyze_sentiment,
            'get_entities': self._filter_get_entities,
            'extract_phrases': self._filter_extract_phrases
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
        
        # Make analyzer functions and NLP helpers available in templates
        self._register_analyzer_functions()
        self._register_nlp_functions()
        self._register_data_helpers()
    
    def render_template(self, template_name: str, messages: Optional[List[Dict]] = None, **kwargs) -> str:
        """
        Render a template with the provided data, including optional message context.
        
        Args:
            template_name: Template file path relative to templates directory
            messages: Optional list of message dicts for NLP analysis
            **kwargs: Data to pass to the template
            
        Returns:
            Rendered template as string
            
        Raises:
            TemplateNotFound: If template file doesn't exist
            TemplateSyntaxError: If template has syntax errors
        """
        try:
            # Check cache first
            cached_template = self.get_cached_template(template_name)
            if cached_template:
                logger.debug(f"Using cached template: {template_name}")
                template = cached_template
            else:
                # Validate template first
                validation = self.validate_template(template_name)
                if not validation['valid']:
                    error_msg = f"Template validation failed for '{template_name}': {', '.join(validation['errors'])}"
                    logger.error(error_msg)
                    return f"❌ {error_msg}"
                
                template = self.env.get_template(template_name)
                # Cache the template for future use
                self.cache_template(template_name, template)
            
            # Add message data to template context if provided
            if messages:
                kwargs['messages'] = messages
                kwargs['message_count'] = len(messages)
                # Pre-compute some common message aggregations
                kwargs['all_message_text'] = ' '.join(self._safe_get_message_content(msg) for msg in messages if self._safe_get_message_content(msg))
                kwargs['message_authors'] = list(set(self._safe_get_message_author(msg) for msg in messages if self._safe_get_message_author(msg)))
            
            # Check for missing required variables (only if not cached)
            if not cached_template:
                missing_vars = []
                for var in validation.get('required_variables', []):
                    if var not in kwargs:
                        missing_vars.append(var)
                
                if missing_vars:
                    error_msg = f"Missing required variables for template '{template_name}': {', '.join(missing_vars)}"
                    logger.error(error_msg)
                    return f"❌ {error_msg}"
            
            result = template.render(**kwargs)
            logger.debug(f"Successfully rendered template '{template_name}'")
            return result
            
        except Exception as e:
            error_msg = f"Error rendering template '{template_name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            return f"❌ {error_msg}"

    def render_string(self, template_string: str, messages: Optional[List[Dict]] = None, **kwargs) -> str:
        """
        Render a template from a string with optional message context.
        
        Args:
            template_string: Template content as string
            messages: Optional list of message dicts for NLP analysis
            **kwargs: Data to pass to the template
            
        Returns:
            Rendered template as string
        """
        try:
            # Use the environment to create template so it has access to all registered functions
            template = self.env.from_string(template_string)
            
            # Add message data to template context if provided
            if messages:
                kwargs['messages'] = messages
                kwargs['message_count'] = len(messages)
                kwargs['all_message_text'] = ' '.join(self._safe_get_message_content(msg) for msg in messages if self._safe_get_message_content(msg))
                kwargs['message_authors'] = list(set(self._safe_get_message_author(msg) for msg in messages if self._safe_get_message_author(msg)))
            
            return template.render(**kwargs)
        except Exception as e:
            logger.error(f"Error rendering template string: {str(e)}", exc_info=True)
            return f"❌ Error rendering template: {str(e)}"

    def _safe_get_message_content(self, msg) -> str:
        """Safely get message content from either dict or Pydantic model."""
        if hasattr(msg, 'content'):
            return msg.content or ''
        elif isinstance(msg, dict):
            return msg.get('content', '')
        return ''

    def _safe_get_message_author(self, msg) -> str:
        """Safely get message author from either dict or Pydantic model."""
        if hasattr(msg, 'author_name'):
            return msg.author_name or ''
        elif hasattr(msg, 'author'):
            return msg.author or ''
        elif isinstance(msg, dict):
            return msg.get('author_name', msg.get('author', ''))
        return ''

    def _safe_get_message_id(self, msg) -> str:
        """Safely get message ID from either dict or Pydantic model."""
        if hasattr(msg, 'id'):
            return msg.id or ''
        elif isinstance(msg, dict):
            return msg.get('id', '')
        return ''

    def _register_analyzer_functions(self):
        """Register analyzer functions for use in templates."""
        if self.analyzers:
            for name, analyzer in self.analyzers.items():
                # Register the analyzer instance itself
                self.env.globals[f"{name}_analyzer"] = analyzer
                
                # Register common analyzer methods directly
                if hasattr(analyzer, 'analyze'):
                    self.env.globals[f"{name}_analyze"] = analyzer.analyze
                if hasattr(analyzer, 'get_top_users'):
                    self.env.globals[f"get_top_users"] = analyzer.get_top_users
                if hasattr(analyzer, 'get_top_channels'):
                    self.env.globals[f"get_top_channels"] = analyzer.get_top_channels
                if hasattr(analyzer, 'get_available_users'):
                    self.env.globals[f"get_available_users"] = analyzer.get_available_users
                if hasattr(analyzer, 'get_available_channels'):
                    self.env.globals[f"get_available_channels"] = analyzer.get_available_channels

    def _register_nlp_functions(self):
        """Register NLP helper functions for templates."""
        if self.nlp_service:
            # Direct NLP functions
            self.env.globals['extract_concepts'] = self._safe_extract_concepts
            self.env.globals['analyze_sentiment'] = self._safe_analyze_sentiment
            self.env.globals['get_named_entities'] = self._safe_get_entities
            self.env.globals['extract_key_phrases'] = self._safe_extract_phrases
            self.env.globals['analyze_complexity'] = self._safe_analyze_complexity
            
            # Batch processing helpers
            self.env.globals['analyze_messages_sentiment'] = self._analyze_messages_sentiment
            self.env.globals['extract_message_concepts'] = self._extract_message_concepts
            self.env.globals['get_message_entities'] = self._get_message_entities
        else:
            # Provide no-op functions if NLP service not available
            self.env.globals['extract_concepts'] = lambda text: []
            self.env.globals['analyze_sentiment'] = lambda text: {'sentiment': 'neutral', 'score': 0.0}
            self.env.globals['get_named_entities'] = lambda text: []
            self.env.globals['extract_key_phrases'] = lambda text: []
            self.env.globals['analyze_complexity'] = lambda text: {'complexity': 'medium', 'score': 0.5}
            self.env.globals['analyze_messages_sentiment'] = lambda messages: []
            self.env.globals['extract_message_concepts'] = lambda messages: []
            self.env.globals['get_message_entities'] = lambda messages: []

    def _register_data_helpers(self):
        """Register data access helper functions for templates."""
        if self.data_facade:
            # Repository access helpers
            self.env.globals['get_messages_by_channel'] = self._get_messages_by_channel
            self.env.globals['get_messages_by_user'] = self._get_messages_by_user
            self.env.globals['get_recent_messages'] = self._get_recent_messages
            self.env.globals['search_messages'] = self._search_messages

    # NLP Filter Functions (for use with | syntax)
    def _filter_extract_concepts(self, text: str) -> List[str]:
        """Filter to extract concepts from text."""
        return self._safe_extract_concepts(text)

    def _filter_analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Filter to analyze sentiment of text."""
        return self._safe_analyze_sentiment(text)

    def _filter_get_entities(self, text: str) -> List[Dict]:
        """Filter to get named entities from text."""
        return self._safe_get_entities(text)

    def _filter_extract_phrases(self, text: str) -> List[str]:
        """Filter to extract key phrases from text."""
        return self._safe_extract_phrases(text)

    # Safe NLP Function Wrappers
    def _safe_extract_concepts(self, text: str) -> List[str]:
        """Safely extract concepts with error handling."""
        if not self.nlp_service or not text:
            return []
        try:
            return self.nlp_service.extract_concepts(text)
        except Exception as e:
            logger.error(f"Error extracting concepts: {e}")
            return []

    def _safe_analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Safely analyze sentiment with error handling."""
        if not self.nlp_service or not text:
            return {'sentiment': 'neutral', 'score': 0.0}
        try:
            return self.nlp_service.analyze_sentiment(text)
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {'sentiment': 'neutral', 'score': 0.0}

    def _safe_get_entities(self, text: str) -> List[Dict]:
        """Safely get named entities with error handling."""
        if not self.nlp_service or not text:
            return []
        try:
            return self.nlp_service.get_named_entities(text)
        except Exception as e:
            logger.error(f"Error getting entities: {e}")
            return []

    def _safe_extract_phrases(self, text: str) -> List[str]:
        """Safely extract key phrases with error handling."""
        if not self.nlp_service or not text:
            return []
        try:
            return self.nlp_service.extract_key_phrases(text)
        except Exception as e:
            logger.error(f"Error extracting phrases: {e}")
            return []

    def _safe_analyze_complexity(self, text: str) -> Dict[str, Any]:
        """Safely analyze text complexity with error handling."""
        if not self.nlp_service or not text:
            return {'complexity': 'medium', 'score': 0.5}
        try:
            return self.nlp_service.analyze_text_complexity(text)
        except Exception as e:
            logger.error(f"Error analyzing complexity: {e}")
            return {'complexity': 'medium', 'score': 0.5}

    # Batch Message Processing Functions
    def _analyze_messages_sentiment(self, messages: List[Dict]) -> List[Dict]:
        """Analyze sentiment for a list of messages."""
        if not messages or not self.nlp_service:
            return []
        
        results = []
        for msg in messages:
            content = self._safe_get_message_content(msg)
            if content:
                sentiment = self._safe_analyze_sentiment(content)
                results.append({
                    'message_id': self._safe_get_message_id(msg),
                    'author': self._safe_get_message_author(msg),
                    'sentiment': sentiment['sentiment'],
                    'score': sentiment['score'],
                    'content_preview': content[:100] + '...' if len(content) > 100 else content
                })
        
        return results

    def _extract_message_concepts(self, messages: List[Dict]) -> List[Dict]:
        """Extract concepts from a list of messages."""
        if not messages or not self.nlp_service:
            return []
        
        all_text = ' '.join(self._safe_get_message_content(msg) for msg in messages if self._safe_get_message_content(msg))
        if not all_text:
            return []
        
        concepts = self._safe_extract_concepts(all_text)
        return [{'concept': concept, 'frequency': 1} for concept in concepts[:10]]

    def _get_message_entities(self, messages: List[Dict]) -> List[Dict]:
        """Get named entities from a list of messages."""
        if not messages or not self.nlp_service:
            return []
        
        all_text = ' '.join(self._safe_get_message_content(msg) for msg in messages if self._safe_get_message_content(msg))
        if not all_text:
            return []
        
        entities = self._safe_get_entities(all_text)
        return [{'entity': entity, 'type': 'unknown'} for entity in entities[:10]]

    # Data Access Helper Functions
    def _get_messages_by_channel(self, channel_name: str, limit: int = 100) -> List[Dict]:
        """Get recent messages from a specific channel."""
        if not self.data_facade:
            return []
        try:
            return self.data_facade.message_repository.get_messages_by_channel(channel_name, limit)
        except Exception as e:
            logger.error(f"Error getting messages by channel: {e}")
            return []

    def _get_messages_by_user(self, username: str, limit: int = 100) -> List[Dict]:
        """Get recent messages from a specific user."""
        if not self.data_facade:
            return []
        try:
            return self.data_facade.message_repository.get_messages_by_user(username, limit)
        except Exception as e:
            logger.error(f"Error getting messages by user: {e}")
            return []

    def _get_recent_messages(self, limit: int = 100, channel_name: Optional[str] = None) -> List[Dict]:
        """Get recent messages, optionally filtered by channel."""
        if not self.data_facade:
            return []
        try:
            if channel_name:
                return self.data_facade.message_repository.get_messages_by_channel(channel_name, limit)
            else:
                return self.data_facade.message_repository.get_recent_messages(limit)
        except Exception as e:
            logger.error(f"Error getting recent messages: {e}")
            return []

    def _search_messages(self, query: str, limit: int = 50) -> List[Dict]:
        """Search messages by content."""
        if not self.data_facade:
            return []
        try:
            return self.data_facade.message_repository.search_messages(query, limit)
        except Exception as e:
            logger.error(f"Error searching messages: {e}")
            return []

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
    
    def _format_trend(self, trend: str, percentage: float, timeframe: str = "") -> str:
        """Format trend string for display."""
        if trend == "increasing":
            return f"📈 Increasing (+{abs(percentage):.1f}% {timeframe})"
        elif trend == "decreasing":
            return f"📉 Decreasing (−{abs(percentage):.1f}% {timeframe})"
        else:
            return f"Stable (no significant change {timeframe})"
    
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

    def _split_name(self, name: str) -> str:
        """
        Split a name into first and last name format.
        
        Args:
            name: Full name string
            
        Returns:
            Formatted name as "first-name last-name"
        """
        if not name:
            return "Unknown"
        
        # Split by common separators and take first two parts
        parts = name.replace('_', ' ').replace('.', ' ').split()
        if len(parts) >= 2:
            return f"{parts[0]} {parts[1]}"
        elif len(parts) == 1:
            return parts[0]
        else:
            return name

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

    def validate_template(self, template_name: str) -> Dict[str, Any]:
        """
        Validate a template for syntax errors and required variables.
        
        Args:
            template_name: Template file path relative to templates directory
            
        Returns:
            Dictionary with validation results
        """
        try:
            template = self.env.get_template(template_name)
            
            # Check if template exists and can be loaded
            validation_result = {
                'valid': True,
                'template_name': template_name,
                'exists': True,
                'syntax_valid': True,
                'required_variables': [],
                'optional_variables': [],
                'errors': []
            }
            
            # Try to render with empty context to check for required variables
            try:
                template.render()
                validation_result['required_variables'] = []
            except Exception as e:
                # Extract variable names from error message
                error_msg = str(e)
                if 'UndefinedError' in error_msg:
                    # Extract variable name from error
                    import re
                    var_match = re.search(r"'([^']+)' is undefined", error_msg)
                    if var_match:
                        validation_result['required_variables'].append(var_match.group(1))
                    validation_result['errors'].append(f"Missing required variable: {error_msg}")
            
            return validation_result
            
        except Exception as e:
            return {
                'valid': False,
                'template_name': template_name,
                'exists': False,
                'syntax_valid': False,
                'required_variables': [],
                'optional_variables': [],
                'errors': [f"Template validation failed: {str(e)}"]
            }

    def list_available_templates(self) -> List[str]:
        """
        List all available templates in the templates directory.
        
        Returns:
            List of template file paths
        """
        try:
            templates = []
            if hasattr(self.env.loader, 'list_templates'):
                templates = self.env.loader.list_templates()
            else:
                # Fallback: scan templates directory
                templates_path = Path(self.env.loader.searchpath[0]) if self.env.loader.searchpath else Path("templates")
                if templates_path.exists():
                    for template_file in templates_path.rglob("*.j2"):
                        templates.append(str(template_file.relative_to(templates_path)))
            
            return sorted(templates)
        except Exception as e:
            logger.error(f"Error listing templates: {e}")
            return []

    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a template.
        
        Args:
            template_name: Template file path relative to templates directory
            
        Returns:
            Dictionary with template information
        """
        try:
            template = self.env.get_template(template_name)
            template_path = Path(self.env.loader.searchpath[0]) / template_name if self.env.loader.searchpath else Path("templates") / template_name
            
            info = {
                'name': template_name,
                'exists': template_path.exists(),
                'size': template_path.stat().st_size if template_path.exists() else 0,
                'modified': datetime.fromtimestamp(template_path.stat().st_mtime) if template_path.exists() else None,
                'validation': self.validate_template(template_name)
            }
            
            return info
            
        except Exception as e:
            return {
                'name': template_name,
                'exists': False,
                'size': 0,
                'modified': None,
                'validation': {'valid': False, 'errors': [str(e)]}
            }

    def clear_cache(self) -> None:
        """
        Clear the template cache.
        """
        self._template_cache.clear()
        logger.debug("Template cache cleared")

    def get_cached_template(self, template_name: str) -> Optional['Template']:
        """
        Get a cached template if available.
        
        Args:
            template_name: Template file path
            
        Returns:
            Cached template or None if not cached
        """
        return self._template_cache.get(template_name)

    def cache_template(self, template_name: str, template: 'Template') -> None:
        """
        Cache a compiled template.
        
        Args:
            template_name: Template file path
            template: Compiled template object
        """
        if self._cache_enabled:
            self._template_cache[template_name] = template
            logger.debug(f"Cached template: {template_name}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get template cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'cache_enabled': self._cache_enabled,
            'cached_templates': len(self._template_cache),
            'cache_keys': list(self._template_cache.keys())
        } 