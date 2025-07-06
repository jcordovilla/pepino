"""
Discord bot package for command and event handling.

Provides Discord bot functionality with analysis service integration.
"""

from .bot import bot, run_bot

# Service integration for Discord commands
from ..analysis.service import AnalysisService, analysis_service

__all__ = [
    "bot",
    "run_bot",
    "AnalysisService",
    "analysis_service",
]
