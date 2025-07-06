"""
Component extraction utilities for Discord messages.
"""

from typing import Any, Dict, List

import discord


def extract_components(message) -> List[Dict[str, Any]]:
    """
    Extract all components from a message (buttons, select menus, etc.)

    Args:
        message: Discord message object

    Returns:
        List of component dictionaries
    """
    components = []
    if hasattr(message, "components"):
        for component in message.components:
            if isinstance(component, discord.ui.Button):
                components.append(
                    {
                        "type": "button",
                        "label": component.label,
                        "custom_id": component.custom_id,
                        "style": str(component.style),
                        "disabled": component.disabled,
                        "url": component.url,
                    }
                )
            elif isinstance(component, discord.ui.Select):
                components.append(
                    {
                        "type": "select",
                        "custom_id": component.custom_id,
                        "placeholder": component.placeholder,
                        "min_values": component.min_values,
                        "max_values": component.max_values,
                        "options": [
                            {
                                "label": option.label,
                                "value": option.value,
                                "description": option.description,
                                "default": option.default,
                            }
                            for option in component.options
                        ],
                    }
                )
    return components
