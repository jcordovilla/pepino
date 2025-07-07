"""
Data Operations Package

Handles Discord data synchronization and export operations.
Separate from analysis operations to maintain separation of concerns.

Primary Interface:
    DataOperationsService: Main service for data operations
    DataExporter: Export functionality
"""

from .service import DataOperationsService
from .exporters import DataExporter

# Discord sync components - imported lazily to avoid discord.py dependency
# from .discord_sync import SyncManager, DiscordClient

__all__ = [
    # Primary services
    "DataOperationsService",
    "DataExporter",
    # Discord sync - available but not imported at module level
    # "SyncManager", 
    # "DiscordClient",
] 