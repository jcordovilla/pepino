"""
Database operations for Discord message analysis.

This module provides database schema management, query utilities, and 
data access functions for Discord message analysis operations.
"""

from .queries import (
    init_database_schema,
    get_available_channels,
    get_available_users,
    get_channel_name_mapping,
    get_available_channels_with_mapping,
    filter_boilerplate_phrases,
    get_all_channel_names,
    find_similar_channel_names,
    get_selectable_channels_and_threads
)

__all__ = [
    "init_database_schema",
    "get_available_channels",
    "get_available_users",
    "get_channel_name_mapping",
    "get_available_channels_with_mapping", 
    "filter_boilerplate_phrases",
    "get_all_channel_names",
    "find_similar_channel_names",
    "get_selectable_channels_and_threads"
]
