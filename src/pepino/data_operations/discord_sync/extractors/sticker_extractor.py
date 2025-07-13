"""
Sticker data extraction utilities for Discord messages.
"""

from typing import Any, Dict, List


def extract_sticker_data(message) -> List[Dict[str, Any]]:
    """
    Extract sticker data from a message, robust to missing fields

    Args:
        message: Discord message object

    Returns:
        List of sticker dictionaries
    """
    stickers = []
    if hasattr(message, "stickers"):
        for sticker in message.stickers:
            stickers.append(
                {
                    "id": str(getattr(sticker, "id", "")),
                    "name": getattr(sticker, "name", None),
                    "format_type": str(getattr(sticker, "format_type", None)),
                    "url": getattr(sticker, "url", None),
                }
            )
    return stickers
