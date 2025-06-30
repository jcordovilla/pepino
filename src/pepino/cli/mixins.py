"""
CLI Command Mixins

Provides template rendering and output formatting utilities for CLI commands.
Similar to Discord command mixins but optimized for terminal output.
"""

import logging
from typing import Any, Dict, Optional
from pathlib import Path

import click

from pepino.templates.template_engine import TemplateEngine
from pepino.cli.validation import ValidationError, ErrorHandler, InputValidator

logger = logging.getLogger(__name__)


class CLITemplateMixin:
    """
    Mixin that adds template rendering capabilities to CLI commands.
    
    Provides consistent template-based output formatting across all CLI commands
    with support for multiple output formats (text, JSON, CSV).
    
    Usage:
        class MyAnalysisCommand(CLITemplateMixin):
            def analyze_something(self, data, output_format="text"):
                if output_format == "text":
                    return self.render_cli_template("analysis.txt.j2", data)
                else:
                    return self.format_structured_output(data, output_format)
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Template engine instances will be created per-operation with specific context
        logger.debug("CLITemplateMixin initialized")
    
    def create_template_engine(self, data_facade=None):
        """Create template engine with analyzer helpers and NLP capabilities for CLI."""
        # Try to import analyzers and NLP service
        try:
            from pepino.analysis.channel_analyzer import ChannelAnalyzer
            from pepino.analysis.user_analyzer import UserAnalyzer
            from pepino.analysis.topic_analyzer import TopicAnalyzer
            from pepino.analysis.temporal_analyzer import TemporalAnalyzer
            
            # Try to import NLP service, handle gracefully if unavailable
            try:
                from pepino.analysis.nlp_analyzer import NLPService
                nlp_service = NLPService()
            except ImportError:
                logger.warning("NLP service not available in CLI, templates will have limited NLP capabilities")
                nlp_service = None
            
            # Create analyzers if data facade is provided
            analyzers = {}
            if data_facade:
                analyzers = {
                    'channel': ChannelAnalyzer(data_facade),
                    'user': UserAnalyzer(data_facade),
                    'topic': TopicAnalyzer(data_facade),
                    'temporal': TemporalAnalyzer(data_facade)
                }
            
            return TemplateEngine(
                analyzers=analyzers,
                data_facade=data_facade,
                nlp_service=nlp_service
            )
            
        except ImportError as e:
            logger.warning(f"Could not load analyzer classes for templates: {e}")
            # Fallback to basic template engine
            return TemplateEngine()
    
    def render_cli_template(self, template_name: str, data: Dict[str, Any], 
                           data_facade=None, messages=None) -> str:
        """
        Render a CLI template with the given data and optional analyzer helpers.
        
        Args:
            template_name: Template filename (e.g., 'channel_analysis.txt.j2')
            data: Data to pass to the template
            data_facade: Optional data facade for analyzer helpers
            messages: Optional list of message dicts for NLP analysis
            
        Returns:
            Rendered template string
        """
        try:
            # Create template engine with analyzer helpers if data facade provided
            template_engine = self.create_template_engine(data_facade)
            
            return template_engine.render_template(
                f"outputs/cli/{template_name}", 
                messages=messages,
                **data
            )
        except Exception as e:
            logger.error(f"Error rendering CLI template {template_name}: {e}")
            return f"❌ Error rendering output: {e}"
    
    def handle_output(
        self, 
        data: Dict[str, Any], 
        output_file: Optional[str] = None, 
        output_format: str = "text",
        template_name: Optional[str] = None,
        data_facade=None,
        messages=None
    ):
        """
        Handle output formatting and writing for CLI commands.
        
        Args:
            data: Data to output
            output_file: Optional file to write to
            output_format: Format to use (text, json, csv)
            template_name: Template to use for text format (e.g., 'analysis.txt.j2')
            data_facade: Optional data facade for analyzer helpers
            messages: Optional list of message dicts for NLP analysis
        """
        
        if output_format == "text" and not output_file and template_name:
            # Render template and display directly
            formatted_output = self.render_cli_template(template_name, data, data_facade, messages)
            click.echo(formatted_output)
        
        elif output_format == "text" and output_file and template_name:
            # Render template and write to file
            formatted_output = self.render_cli_template(template_name, data, data_facade, messages)
            with open(output_file, 'w') as f:
                f.write(formatted_output)
            click.echo(f"✅ Template output written to {output_file}")
        
        else:
            # Use structured output (JSON/CSV)
            self._write_structured_output(data, output_file, output_format)
    
    def _write_structured_output(
        self, 
        data: Dict[str, Any], 
        output_file: Optional[str], 
        output_format: str
    ):
        """Write structured output (JSON/CSV) to file or stdout."""
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if output_format == "json":
                import json
                with open(output_path, "w") as f:
                    json.dump(data, f, indent=2, default=str)
                click.echo(f"✅ JSON output written to {output_file}")
            
            elif output_format == "csv":
                self._write_csv_output(data, output_path)
                click.echo(f"✅ CSV output written to {output_file}")
        
        else:
            # Output to stdout
            if output_format == "json":
                import json
                click.echo(json.dumps(data, indent=2, default=str))
            
            elif output_format == "csv":
                self._write_csv_output(data, None)
    
    def _write_csv_output(self, data: Dict[str, Any], output_path: Optional[Path]):
        """Write data as CSV to file or stdout."""
        import csv
        import sys
        
        # Flatten nested data for CSV
        flattened = self._flatten_dict(data)
        
        if output_path:
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(flattened.keys())
                writer.writerow(flattened.values())
        else:
            writer = csv.writer(sys.stdout)
            writer.writerow(flattened.keys())
            writer.writerow(flattened.values())
    
    def _flatten_dict(self, data: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
        """Flatten nested dictionary for CSV output."""
        items = []
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(self._flatten_dict(item, f"{new_key}_{i}", sep=sep).items())
                    else:
                        items.append((f"{new_key}_{i}", item))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def show_template_success(self, operation: str, template_name: str = None, exec_time: float = None):
        """Show success message for template-based operations."""
        ErrorHandler.show_success_message(operation, f"template: {template_name}" if template_name else None)
        if exec_time:
            click.echo(f"⏱️  Execution time: {exec_time:.2f}s")
    
    def show_template_error(self, operation: str, error: str):
        """Show error message for template operations."""
        ErrorHandler.handle_generic_error(Exception(error), operation)
    
    def validate_and_sanitize_inputs(self, **kwargs) -> Dict[str, Any]:
        """
        Validate and sanitize CLI inputs using the validation utilities.
        
        Args:
            **kwargs: Input parameters to validate
            
        Returns:
            Dictionary of validated inputs
            
        Raises:
            ValidationError: If any input is invalid
        """
        try:
            from pepino.cli.validation import validate_cli_inputs
            return validate_cli_inputs(**kwargs)
        except ValidationError as e:
            ErrorHandler.handle_validation_error(e)
            raise
    
    def safe_render_template(self, template_name: str, data: Dict[str, Any], 
                           data_facade=None, messages=None, ctx=None) -> str:
        """
        Safely render a template with comprehensive error handling.
        
        Args:
            template_name: Template filename
            data: Data to pass to the template
            data_facade: Optional data facade
            messages: Optional message list
            ctx: Click context for error handling
            
        Returns:
            Rendered template string or error message
        """
        try:
            return self.render_cli_template(template_name, data, data_facade, messages)
        except Exception as e:
            ErrorHandler.handle_template_error(e, template_name, ctx)
            return f"❌ Template rendering failed: {str(e)}"
    
    def safe_handle_output(self, data: Dict[str, Any], output_file: Optional[str] = None, 
                          output_format: str = "text", template_name: Optional[str] = None,
                          data_facade=None, messages=None, ctx=None):
        """
        Safely handle output with comprehensive error handling.
        
        Args:
            data: Data to output
            output_file: Optional output file
            output_format: Output format
            template_name: Template name for text format
            data_facade: Optional data facade
            messages: Optional message list
            ctx: Click context for error handling
        """
        try:
            self.handle_output(data, output_file, output_format, template_name, data_facade, messages)
        except Exception as e:
            ErrorHandler.handle_generic_error(e, "output handling", ctx)


class CLIAnalysisMixin(CLITemplateMixin):
    """
    Specialized mixin for analysis commands with template integration.
    
    Provides common patterns for analysis commands like handling
    missing data, formatting results, and consistent error handling.
    """
    
    def handle_analysis_result(
        self,
        result: Dict[str, Any],
        operation_name: str,
        template_name: str,
        output_file: Optional[str] = None,
        output_format: str = "text",
        data_facade=None,
        messages=None,
        ctx=None
    ):
        """
        Handle analysis result with consistent error checking and output formatting.
        
        Args:
            result: Analysis result data
            operation_name: Name of the operation (for error messages)
            template_name: Template to use for text output
            output_file: Optional output file
            output_format: Output format
            data_facade: Optional data facade for analyzer helpers
            messages: Optional list of message dicts for NLP analysis
            ctx: Click context for error handling
        """
        
        # Check for errors in result
        if not result:
            ErrorHandler.handle_analysis_error(Exception("No data returned"), operation_name, ctx)
            return
        
        if isinstance(result, dict) and result.get('error'):
            ErrorHandler.handle_analysis_error(Exception(result['error']), operation_name, ctx)
            return
        
        if isinstance(result, dict) and not result.get('success', True):
            error_msg = result.get('error', 'Analysis failed')
            ErrorHandler.handle_analysis_error(Exception(error_msg), operation_name, ctx)
            return
        
        # Handle successful result with safe output handling
        try:
            self.safe_handle_output(result, output_file, output_format, template_name, data_facade, messages, ctx)
            ErrorHandler.show_success_message(operation_name)
        except Exception as e:
            ErrorHandler.handle_analysis_error(e, operation_name, ctx)
    
    def handle_list_result(
        self,
        items: list,
        operation_name: str,
        template_name: str,
        output_file: Optional[str] = None,
        output_format: str = "text",
        limit: Optional[int] = None,
        ctx=None
    ):
        """
        Handle list result (users, channels, etc.) with template formatting.
        
        Args:
            items: List of items to display
            operation_name: Name of the operation
            template_name: Template to use
            output_file: Optional output file
            output_format: Output format
            limit: Optional limit for items shown
            ctx: Click context for error handling
        """
        
        if not items:
            ErrorHandler.show_warning_message(f"No items found for {operation_name}")
            return
        
        # Prepare data for template
        data = {
            'items': items[:limit] if limit else items,
            'total_count': len(items),
            'showing_count': min(len(items), limit) if limit else len(items),
            'has_more': limit and len(items) > limit
        }
        
        try:
            self.safe_handle_output(data, output_file, output_format, template_name, ctx=ctx)
            ErrorHandler.show_success_message(operation_name, f"found {len(items)} items")
        except Exception as e:
            ErrorHandler.handle_generic_error(e, operation_name, ctx) 