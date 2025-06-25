"""
Discord Template System

Template engine for Discord bot responses.
Re-exports the main TemplateEngine from the shared templates package.
"""

from pepino.templates.template_engine import TemplateEngine

__all__ = ['TemplateEngine'] 