"""
Message extraction utilities for Discord messages.
"""

import logging
from typing import Any, Dict, Optional

import discord

from .component_extractor import extract_components
from .emoji_extractor import extract_emojis
from .interaction_extractor import extract_interaction_data
from .role_subscription_extractor import extract_role_subscription_data
from .sticker_extractor import extract_sticker_data

logger = logging.getLogger(__name__)


class MessageExtractor:
    """Handles extraction of data from Discord messages"""

    @staticmethod
    def extract_message_data(message: discord.Message) -> Dict[str, Any]:
        """
        Extract all data from a Discord message

        Args:
            message: Discord message object

        Returns:
            Dictionary containing all extracted message data
        """

        # Helper function to safely get role type
        def get_role_type(role):
            return str(getattr(role, "type", "role"))

        # Helper function to safely get overwrite data
        def get_overwrite_data(overwrite):
            try:
                return {
                    "id": str(getattr(overwrite, "id", "")),
                    "type": get_role_type(overwrite),
                    "allow": getattr(getattr(overwrite, "allow", None), "value", None),
                    "deny": getattr(getattr(overwrite, "deny", None), "value", None),
                }
            except Exception as e:
                logger.warning(f"Error processing overwrite: {e}")
                return None

        # Helper for safe attachment dict
        def safe_attachment(att):
            return {
                "url": getattr(att, "url", None),
                "filename": getattr(att, "filename", None),
                "size": getattr(att, "size", None),
                "content_type": getattr(att, "content_type", None),
                "height": getattr(att, "height", None),
                "width": getattr(att, "width", None),
                "ephemeral": getattr(att, "ephemeral", None),
                "description": getattr(att, "description", None),
                # Only include if present
                **(
                    {"duration_secs": getattr(att, "duration_secs", None)}
                    if hasattr(att, "duration_secs")
                    else {}
                ),
                **(
                    {"waveform": getattr(att, "waveform", None)}
                    if hasattr(att, "waveform")
                    else {}
                ),
            }

        # Helper for safe referenced message
        def safe_referenced_message(ref):
            if not ref or not hasattr(ref, "resolved") or not ref.resolved:
                return None
            resolved = ref.resolved
            return {
                "id": str(getattr(resolved, "id", "")),
                "content": getattr(resolved, "content", None),
                "author_id": str(getattr(getattr(resolved, "author", None), "id", ""))
                if getattr(resolved, "author", None)
                else None,
                "channel_id": str(getattr(getattr(resolved, "channel", None), "id", ""))
                if getattr(resolved, "channel", None)
                else None,
                "guild_id": str(getattr(getattr(resolved, "guild", None), "id", ""))
                if getattr(resolved, "guild", None)
                else None,
            }

        # Helper for safe message reference
        def safe_message_reference(ref):
            if not ref:
                return None
            return {
                "message_id": str(getattr(ref, "message_id", "")),
                "channel_id": str(getattr(ref, "channel_id", "")),
                "guild_id": str(getattr(ref, "guild_id", ""))
                if getattr(ref, "guild_id", None)
                else None,
                "fail_if_not_exists": getattr(ref, "fail_if_not_exists", None),
            }

        return {
            "id": str(message.id),
            "content": getattr(message, "content", "")
            .replace("\u2028", " ")
            .replace("\u2029", " ")
            .strip(),
            "timestamp": message.created_at.isoformat(),
            "edited_timestamp": message.edited_at.isoformat()
            if getattr(message, "edited_at", None)
            else None,
            "jump_url": getattr(message, "jump_url", None),
            "author": MessageExtractor._extract_author_data(message),
            "user_presence": MessageExtractor._extract_user_presence_data(message),
            "guild": MessageExtractor._extract_guild_data(message),
            "channel": MessageExtractor._extract_channel_data(
                message, get_overwrite_data
            ),
            "thread": MessageExtractor._extract_thread_data(message),
            "mentions": [
                str(getattr(user, "id", ""))
                for user in getattr(message, "mentions", [])
            ],
            "mention_everyone": getattr(message, "mention_everyone", None),
            "mention_roles": [
                str(getattr(role, "id", ""))
                for role in getattr(message, "role_mentions", [])
            ],
            "mention_channels": [
                {
                    "id": str(getattr(channel, "id", "")),
                    "name": getattr(channel, "name", None),
                    "guild_id": str(getattr(getattr(channel, "guild", None), "id", ""))
                    if getattr(channel, "guild", None)
                    else None,
                }
                for channel in getattr(message, "channel_mentions", [])
            ],
            "referenced_message_id": str(
                getattr(getattr(message, "reference", None), "message_id", "")
            )
            if getattr(message, "reference", None)
            else None,
            "referenced_message": safe_referenced_message(
                getattr(message, "reference", None)
            ),
            "attachments": [
                safe_attachment(att) for att in getattr(message, "attachments", [])
            ],
            "embeds": [embed.to_dict() for embed in getattr(message, "embeds", [])],
            "reactions": MessageExtractor._extract_reactions_data(message),
            "emoji_stats": extract_emojis(getattr(message, "content", "")),
            "pinned": getattr(message, "pinned", None),
            "flags": getattr(getattr(message, "flags", None), "value", None)
            if getattr(message, "flags", None)
            else None,
            "nonce": getattr(message, "nonce", None),
            "type": str(getattr(message, "type", "")),
            "is_system": message.is_system() if hasattr(message, "is_system") else None,
            "mentions_everyone": getattr(message, "mention_everyone", None),
            "has_reactions": bool(getattr(message, "reactions", [])),
            "components": extract_components(message),
            "interaction": extract_interaction_data(message),
            "stickers": extract_sticker_data(message),
            "role_subscription_data": extract_role_subscription_data(message),
            "application_id": str(getattr(message, "application_id", ""))
            if getattr(message, "application_id", None)
            else None,
            "application": MessageExtractor._extract_application_data(message),
            "activity": MessageExtractor._extract_activity_data(message),
            "position": getattr(message, "position", None),
            "role_subscription_listing_id": str(
                getattr(message, "role_subscription_listing_id", "")
            )
            if hasattr(message, "role_subscription_listing_id")
            else None,
            "webhook_id": str(getattr(message, "webhook_id", ""))
            if getattr(message, "webhook_id", None)
            else None,
            "tts": getattr(message, "tts", None),
            # Only include if present
            **(
                {"suppress_embeds": getattr(message, "suppress_embeds", None)}
                if hasattr(message, "suppress_embeds")
                else {}
            ),
            "allowed_mentions": getattr(
                getattr(message, "allowed_mentions", None), "to_dict", lambda: None
            )()
            if getattr(message, "allowed_mentions", None)
            else None,
            "message_reference": safe_message_reference(
                getattr(message, "reference", None)
            ),
        }

    @staticmethod
    def _extract_author_data(message: discord.Message) -> Dict[str, Any]:
        """Extract author data from message"""
        return {
            "id": str(getattr(message.author, "id", "")),
            "name": getattr(message.author, "name", None),
            "discriminator": getattr(message.author, "discriminator", None),
            "display_name": getattr(
                message.author,
                "display_name",
                getattr(message.author, "name", None),
            ),
            "is_bot": getattr(message.author, "bot", None),
            "avatar_url": str(
                getattr(getattr(message.author, "avatar", None), "url", "")
            )
            if getattr(message.author, "avatar", None)
            else None,
            "accent_color": getattr(
                getattr(message.author, "accent_color", None), "value", None
            )
            if getattr(message.author, "accent_color", None)
            else None,
            "banner_url": str(
                getattr(getattr(message.author, "banner", None), "url", "")
            )
            if getattr(message.author, "banner", None)
            else None,
            "color": getattr(getattr(message.author, "color", None), "value", None)
            if getattr(message.author, "color", None)
            else None,
            "created_at": message.author.created_at.isoformat()
            if getattr(message.author, "created_at", None)
            else None,
            "default_avatar_url": str(
                getattr(getattr(message.author, "default_avatar", None), "url", "")
            )
            if getattr(message.author, "default_avatar", None)
            else None,
            "public_flags": getattr(
                getattr(message.author, "public_flags", None), "value", None
            )
            if getattr(message.author, "public_flags", None)
            else None,
            "system": getattr(message.author, "system", None),
            "verified": getattr(message.author, "verified", None),
        }

    @staticmethod
    def _extract_user_presence_data(
        message: discord.Message,
    ) -> Optional[Dict[str, Any]]:
        """Extract user presence data from message"""
        if not isinstance(message.author, discord.Member):
            return None

        return {
            "status": str(getattr(message.author, "status", "")),
            "activity": str(
                getattr(getattr(message.author, "activity", None), "name", "")
            )
            if getattr(message.author, "activity", None)
            else None,
            "desktop_status": str(getattr(message.author, "desktop_status", ""))
            if hasattr(message.author, "desktop_status")
            else None,
            "mobile_status": str(getattr(message.author, "mobile_status", ""))
            if hasattr(message.author, "mobile_status")
            else None,
            "web_status": str(getattr(message.author, "web_status", ""))
            if hasattr(message.author, "web_status")
            else None,
        }

    @staticmethod
    def _extract_guild_data(message: discord.Message) -> Optional[Dict[str, Any]]:
        """Extract guild data from message"""
        if not getattr(message, "guild", None):
            return None

        return {
            "id": str(getattr(message.guild, "id", "")),
            "name": getattr(message.guild, "name", None),
            "member_count": getattr(message.guild, "member_count", None),
            "description": getattr(message.guild, "description", None),
            "icon_url": str(getattr(getattr(message.guild, "icon", None), "url", ""))
            if getattr(message.guild, "icon", None)
            else None,
            "banner_url": str(
                getattr(getattr(message.guild, "banner", None), "url", "")
            )
            if getattr(message.guild, "banner", None)
            else None,
            "splash_url": str(
                getattr(getattr(message.guild, "splash", None), "url", "")
            )
            if getattr(message.guild, "splash", None)
            else None,
            "discovery_splash_url": str(
                getattr(getattr(message.guild, "discovery_splash", None), "url", "")
            )
            if getattr(message.guild, "discovery_splash", None)
            else None,
            "features": getattr(message.guild, "features", None),
            "verification_level": str(getattr(message.guild, "verification_level", "")),
            "explicit_content_filter": str(
                getattr(message.guild, "explicit_content_filter", "")
            ),
            "mfa_level": str(getattr(message.guild, "mfa_level", "")),
            "premium_tier": str(getattr(message.guild, "premium_tier", "")),
            "premium_subscription_count": getattr(
                message.guild, "premium_subscription_count", None
            ),
        }

    @staticmethod
    def _extract_channel_data(
        message: discord.Message, get_overwrite_data
    ) -> Dict[str, Any]:
        """Extract channel data from message"""
        return {
            "id": str(getattr(message.channel, "id", "")),
            "name": getattr(message.channel, "name", None),
            "type": str(getattr(message.channel, "type", "")),
            "topic": getattr(message.channel, "topic", None),
            "nsfw": getattr(message.channel, "nsfw", None),
            "position": getattr(message.channel, "position", None),
            "slowmode_delay": getattr(message.channel, "slowmode_delay", None),
            "category_id": str(getattr(message.channel, "category_id", ""))
            if getattr(message.channel, "category_id", None)
            else None,
            "overwrites": [
                overwrite_data
                for overwrite in getattr(message.channel, "overwrites", [])
                if (overwrite_data := get_overwrite_data(overwrite)) is not None
            ],
        }

    @staticmethod
    def _extract_thread_data(message: discord.Message) -> Optional[Dict[str, Any]]:
        """Extract thread data from message"""
        if not hasattr(message, "thread") or not getattr(message, "thread", None):
            return None

        return {
            "id": str(getattr(message.thread, "id", "")),
            "name": getattr(message.thread, "name", None),
            "archived": getattr(message.thread, "archived", None),
            "auto_archive_duration": getattr(
                message.thread, "auto_archive_duration", None
            ),
            "locked": getattr(message.thread, "locked", None),
            "member_count": getattr(message.thread, "member_count", None),
            "message_count": getattr(message.thread, "message_count", None),
            "owner_id": str(getattr(message.thread, "owner_id", "")),
            "parent_id": str(getattr(message.thread, "parent_id", "")),
            "slowmode_delay": getattr(message.thread, "slowmode_delay", None),
        }

    @staticmethod
    def _extract_reactions_data(message: discord.Message) -> list:
        """Extract reactions data from message"""
        return [
            {
                "emoji": str(getattr(reaction, "emoji", "")),
                "count": getattr(reaction, "count", None),
                "me": getattr(reaction, "me", None),
                "emoji_id": str(getattr(getattr(reaction, "emoji", None), "id", ""))
                if getattr(reaction, "emoji", None) and hasattr(reaction.emoji, "id")
                else None,
                "emoji_name": getattr(getattr(reaction, "emoji", None), "name", None)
                if getattr(reaction, "emoji", None) and hasattr(reaction.emoji, "name")
                else None,
                "emoji_animated": getattr(
                    getattr(reaction, "emoji", None), "animated", None
                )
                if getattr(reaction, "emoji", None)
                and hasattr(reaction.emoji, "animated")
                else None,
            }
            for reaction in getattr(message, "reactions", [])
        ]

    @staticmethod
    def _extract_application_data(message: discord.Message) -> Optional[Dict[str, Any]]:
        """Extract application data from message"""
        if not getattr(message, "application", None):
            return None

        return {
            "id": str(getattr(getattr(message, "application", None), "id", "")),
            "name": getattr(getattr(message, "application", None), "name", None),
            "description": getattr(
                getattr(message, "application", None), "description", None
            ),
            "icon_url": str(
                getattr(
                    getattr(getattr(message, "application", None), "icon", None),
                    "url",
                    "",
                )
            )
            if getattr(getattr(message, "application", None), "icon", None)
            else None,
            "cover_image_url": str(
                getattr(
                    getattr(getattr(message, "application", None), "cover_image", None),
                    "url",
                    "",
                )
            )
            if getattr(getattr(message, "application", None), "cover_image", None)
            else None,
            "bot_public": getattr(
                getattr(message, "application", None), "bot_public", None
            ),
            "bot_require_code_grant": getattr(
                getattr(message, "application", None),
                "bot_require_code_grant",
                None,
            ),
            "terms_of_service_url": getattr(
                getattr(message, "application", None), "terms_of_service_url", None
            ),
            "privacy_policy_url": getattr(
                getattr(message, "application", None), "privacy_policy_url", None
            ),
        }

    @staticmethod
    def _extract_activity_data(message: discord.Message) -> Optional[Dict[str, Any]]:
        """Extract activity data from message"""
        if not getattr(message, "activity", None):
            return None

        return {
            "type": str(getattr(getattr(message, "activity", None), "type", "")),
            "party_id": getattr(getattr(message, "activity", None), "party_id", None),
            "application_id": str(
                getattr(getattr(message, "activity", None), "application_id", "")
            ),
            "name": getattr(getattr(message, "activity", None), "name", None),
            "state": getattr(getattr(message, "activity", None), "state", None),
            "details": getattr(getattr(message, "activity", None), "details", None),
            "timestamps": getattr(
                getattr(message, "activity", None), "timestamps", None
            ),
            "assets": getattr(getattr(message, "activity", None), "assets", None),
            "sync_id": getattr(getattr(message, "activity", None), "sync_id", None),
            "session_id": getattr(
                getattr(message, "activity", None), "session_id", None
            ),
            "flags": getattr(getattr(message, "activity", None), "flags", None),
        }
