"""
Pepino - Discord Message Analysis Tool

A comprehensive tool for analyzing Discord server messages with advanced
statistical analysis, topic modeling, and visualization capabilities.

Primary Interface:
    AnalysisService: Main service for all analysis operations
    analysis_service: Context manager for convenient usage
"""

# Core packages
from . import analysis, data, discord_bot, data_operations

# Primary service interface
from .analysis.service import AnalysisService, analysis_service

# Core data models and utilities
from .data import DatabaseManager, Message, MessageRepository
from .discord_bot import bot, run_bot

# Type definitions for clean interfaces
from .analysis.service import OutputFormat

__all__ = [
    # Packages
    "data",
    "discord_bot", 
    "analysis",
    "data_operations",
    # Primary service interface
    "AnalysisService",
    "analysis_service",
    # Core classes
    "DatabaseManager",
    "Message", 
    "MessageRepository",
    "bot",
    "run_bot",
    # Type definitions
    "OutputFormat",
]
