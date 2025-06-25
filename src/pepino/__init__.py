"""
Pepino - Discord Message Analysis Tool

A comprehensive tool for analyzing Discord server messages with advanced
statistical analysis, topic modeling, and visualization capabilities.
"""

# Core packages
from . import analysis, data, discord

# Convenience imports
from .data import DatabaseManager, Message, MessageRepository
from .discord import DiscordClient

__all__ = [
    # Packages
    "data",
    "discord",
    "analysis",
    # Core classes
    "DatabaseManager",
    "Message",
    "MessageRepository",
    "DiscordClient",
]
