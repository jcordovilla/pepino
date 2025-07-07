"""
Central data layer for the Pepino application.

Provides database access, models, and repositories for all data operations.
"""

# Database management
try:
    from .database.manager import DatabaseManager
    DATABASE_AVAILABLE = True
except ImportError as e:
    DATABASE_AVAILABLE = False
    DatabaseManager = None

# Repository layer
try:
    from .repositories import (
        ChannelRepository, 
        MessageRepository, 
        UserRepository,
        DatabaseRepository,
        SyncRepository
    )
    REPOSITORIES_AVAILABLE = True
except ImportError as e:
    REPOSITORIES_AVAILABLE = False
    MessageRepository = None
    ChannelRepository = None
    UserRepository = None
    DatabaseRepository = None
    SyncRepository = None

# Core data models
from .models.channel import Channel
from .models.message import Message
from .models.user import User

# Configuration
from pepino.config import Settings, settings

__all__ = [
    # Database management
    "DatabaseManager",
    "DATABASE_AVAILABLE",
    # Repository layer
    "MessageRepository",
    "ChannelRepository", 
    "UserRepository",
    "DatabaseRepository",
    "SyncRepository",
    "REPOSITORIES_AVAILABLE",
    # Core models
    "Message",
    "Channel",
    "User",
    # Configuration
    "Settings",
    "settings",
]
