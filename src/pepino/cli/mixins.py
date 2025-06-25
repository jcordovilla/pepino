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
        
        # Initialize template engine (lazy loaded)
        self._template_engine = None
        
        logger.debug("CLITemplateMixin initialized")
    
    @property
    def template_engine(self) -> TemplateEngine:
        """Get or create the template engine instance (lazy loading)."""
        if self._template_engine is None:
            self._template_engine = TemplateEngine()
            logger.debug("Template engine initialized for CLI")
        return self._template_engine
    
    def render_cli_template(self, template_name: str, data: Dict[str, Any]) -> str:
        """
        Render a CLI template with the given data.
        
        Args:
            template_name: Template filename (e.g., 'channel_analysis.txt.j2')
            data: Data to pass to the template
            
        Returns:
            Rendered template string
        """
        try:
            return self.template_engine.render_template(f"outputs/cli/{template_name}", **data)
        except Exception as e:
            logger.error(f"Error rendering CLI template {template_name}: {e}")
            return f"❌ Error rendering output: {e}"
    
    def handle_output(
        self, 
        data: Dict[str, Any], 
        output_file: Optional[str] = None, 
        output_format: str = "text",
        template_name: Optional[str] = None
    ):
        """
        Handle output formatting and writing for CLI commands.
        
        Args:
            data: Data to output
            output_file: Optional file to write to
            output_format: Format to use (text, json, csv)
            template_name: Template to use for text format (e.g., 'analysis.txt.j2')
        """
        
        if output_format == "text" and not output_file and template_name:
            # Render template and display directly
            formatted_output = self.render_cli_template(template_name, data)
            click.echo(formatted_output)
        
        elif output_format == "text" and output_file and template_name:
            # Render template and write to file
            formatted_output = self.render_cli_template(template_name, data)
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
        message = f"✅ {operation} completed"
        if exec_time:
            message += f" in {exec_time:.2f}s"
        click.echo(message)
    
    def show_template_error(self, operation: str, error: str):
        """Show error message for template operations."""
        click.echo(f"❌ {operation} failed: {error}", err=True)


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
        output_format: str = "text"
    ):
        """
        Handle analysis result with consistent error checking and output formatting.
        
        Args:
            result: Analysis result data
            operation_name: Name of the operation (for error messages)
            template_name: Template to use for text output
            output_file: Optional output file
            output_format: Output format
        """
        
        # Check for errors in result
        if not result:
            self.show_template_error(operation_name, "No data returned from analysis")
            return
        
        if isinstance(result, dict) and result.get('error'):
            self.show_template_error(operation_name, result['error'])
            return
        
        if isinstance(result, dict) and not result.get('success', True):
            error_msg = result.get('error', 'Analysis failed')
            self.show_template_error(operation_name, error_msg)
            return
        
        # Handle successful result
        self.handle_output(result, output_file, output_format, template_name)
        
        # Success message removed for cleaner output
    
    def handle_list_result(
        self,
        items: list,
        operation_name: str,
        template_name: str,
        output_file: Optional[str] = None,
        output_format: str = "text",
        limit: Optional[int] = None
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
        """
        
        if not items:
            self.show_template_error(operation_name, "No items found")
            return
        
        # Prepare data for template
        data = {
            'items': items[:limit] if limit else items,
            'total_count': len(items),
            'showing_count': min(len(items), limit) if limit else len(items),
            'has_more': limit and len(items) > limit
        }
        
        self.handle_output(data, output_file, output_format, template_name)
        
        # Success message removed for cleaner output 