"""
Pepino - Discord Message Analysis Tool

A comprehensive tool for analyzing Discord server messages with advanced
statistical analysis, topic modeling, and visualization capabilities.

Primary Interface:
    AnalysisService: Main service for all analysis operations
    analysis_service: Context manager for convenient usage
"""

# Core packages
from . import analysis, data, data_operations

# Optional discord_bot import
try:
    from . import discord_bot
    DISCORD_BOT_AVAILABLE = True
except ImportError:
    DISCORD_BOT_AVAILABLE = False

# Primary service interface
from .analysis.service import AnalysisService, analysis_service

# Core data models and utilities
from .data import DatabaseManager, Message, MessageRepository

# Optional discord_bot imports
if DISCORD_BOT_AVAILABLE:
    from .discord_bot import bot, run_bot
else:
    bot = None
    run_bot = None

# Type definitions for clean interfaces
from .analysis.service import OutputFormat

__all__ = [
    # Packages
    "data",
    "analysis",
    "data_operations",
    # Primary service interface
    "AnalysisService",
    "analysis_service",
    # Core classes
    "DatabaseManager",
    "Message", 
    "MessageRepository",
    # Type definitions
    "OutputFormat",
]

# Add discord_bot to __all__ if available
if DISCORD_BOT_AVAILABLE:
    __all__.extend(["discord_bot", "bot", "run_bot"])
