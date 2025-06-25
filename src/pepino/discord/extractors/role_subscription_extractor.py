"""
Role subscription data extraction utilities for Discord messages.
"""

from typing import Any, Dict, Optional


def extract_role_subscription_data(message) -> Optional[Dict[str, Any]]:
    """
    Extract role subscription data if present

    Args:
        message: Discord message object

    Returns:
        Role subscription data dictionary or None
    """
    if hasattr(message, "role_subscription_data"):
        return {
            "role_subscription_listing_id": str(
                message.role_subscription_data.role_subscription_listing_id
            ),
            "tier_name": message.role_subscription_data.tier_name,
            "total_months_subscribed": message.role_subscription_data.total_months_subscribed,
            "is_renewal": message.role_subscription_data.is_renewal,
        }
    return None
