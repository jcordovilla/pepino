"""
Emoji extraction utilities for Discord messages.
"""

import re
from typing import Dict, List


def extract_emojis(text: str) -> Dict[str, List[str]]:
    """
    Extract emojis from text.

    Args:
        text: The text to extract emojis from

    Returns:
        Dictionary with 'unicode_emojis' and 'custom_emojis' lists
    """
    # Fixed regex for Unicode emojis to capture common emoji ranges
    unicode_emojis = re.findall(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000026FF\U00002700-\U000027BF]",
        text,
    )
    custom_emojis = re.findall(r"<a?:\w+:\d+>", text)
    return {"unicode_emojis": unicode_emojis, "custom_emojis": custom_emojis}
