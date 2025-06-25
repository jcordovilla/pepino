"""
Interaction data extraction utilities for Discord messages.
"""

from typing import Any, Dict, Optional


def extract_interaction_data(message) -> Optional[Dict[str, Any]]:
    """
    Extract interaction data if the message is a response to an interaction

    Args:
        message: Discord message object

    Returns:
        Interaction data dictionary or None
    """
    if hasattr(message, "interaction_metadata"):  # Updated to use interaction_metadata
        metadata = message.interaction_metadata
        return {
            "id": str(metadata.id) if metadata and hasattr(metadata, "id") else None,
            "type": str(metadata.type)
            if metadata and hasattr(metadata, "type")
            else None,
            "name": metadata.name if metadata and hasattr(metadata, "name") else None,
            "user_id": str(metadata.user_id)
            if metadata and hasattr(metadata, "user_id")
            else None,
            "guild_id": str(metadata.guild_id)
            if metadata and hasattr(metadata, "guild_id")
            else None,
            "channel_id": str(metadata.channel_id)
            if metadata and hasattr(metadata, "channel_id")
            else None,
            "data": metadata.data if metadata and hasattr(metadata, "data") else None,
        }
    return None
