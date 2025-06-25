"""
Template System

Unified template engine for multi-domain template rendering across Discord, CLI, and other outputs.
Provides Jinja2-based template rendering with chart integration and domain-specific filters.
"""

from .template_engine import TemplateEngine

__all__ = ['TemplateEngine'] 