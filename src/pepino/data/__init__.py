"""
Central data layer for the Pepino application.

Provides database access, models, and repositories for all data operations.
"""

try:
    from .database.manager import DatabaseManager

    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    DatabaseManager = None

try:
    from .repositories import ChannelRepository, MessageRepository, UserRepository

    REPOSITORIES_AVAILABLE = True
except ImportError:
    REPOSITORIES_AVAILABLE = False
    MessageRepository = None
    ChannelRepository = None
    UserRepository = None

from ..config import Settings, settings
from .models.channel import Channel
from .models.message import Message
from .models.user import User

__all__ = [
    "DatabaseManager",
    "DATABASE_AVAILABLE",
    "MessageRepository",
    "ChannelRepository",
    "UserRepository",
    "REPOSITORIES_AVAILABLE",
    "Message",
    "Channel",
    "User",
    "Settings",
    "settings",
]
