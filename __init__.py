"""
Discord Message Analysis Package

A comprehensive package for analyzing Discord server messages with advanced NLP,
text processing, visualization, and database operations.

Modules:
- bot_commands.py: Discord bot command implementations
- bot.py: Discord bot setup and configuration
- fetch_messages.py: Message fetching and database population

Packages:
- core/: Core text processing and NLP functions
- visualization/: Chart and graph generation utilities  
- analysis/: Embedding operations, topics, statistics, and insights
- database/: Database operations and schema management
- models/: Analyzer class definitions (MessageAnalyzer, DiscordBotAnalyzer)
- utils/: Helper functions and utilities
"""

__version__ = "1.0.0"
__author__ = "Pepino Team"

# Import main classes for easy access
from .models import MessageAnalyzer, DiscordBotAnalyzer

# For backward compatibility, also make available at package level
__all__ = [
    "MessageAnalyzer",
    "DiscordBotAnalyzer",
]

__all__ = [
    "MessageAnalyzer",
    "DiscordBotAnalyzer",
    "LegacyMessageAnalyzer", 
    "LegacyDiscordBotAnalyzer"
]
