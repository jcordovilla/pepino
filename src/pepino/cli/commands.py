"""
CLI commands for Discord analytics.
"""

import asyncio
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from pepino.analysis.channel_analyzer import ChannelAnalyzer
from pepino.analysis.conversation_analyzer import ConversationService
from pepino.analysis.embedding_analyzer import EmbeddingService
from pepino.analysis.nlp_analyzer import NLPService
from pepino.analysis.similarity_analyzer import SimilarityService
from pepino.analysis.temporal_analyzer import TemporalAnalyzer
from pepino.analysis.topic_analyzer import TopicAnalyzer
from pepino.analysis.user_analyzer import UserAnalyzer
from pepino.config import Settings
from pepino.data.database.manager import DatabaseManager
from pepino.data.repositories import ChannelRepository, MessageRepository
from pepino.logging_config import get_logger, setup_cli_logging

from . import persistence
from .mixins import CLIAnalysisMixin

logger = get_logger(__name__)


class CLIAnalysisCommands(CLIAnalysisMixin):
    """
    CLI Analysis Commands with template integration.

    Uses mixin pattern for consistent template rendering and output formatting.
    """

    def __init__(self):
        super().__init__()
        logger.info("CLI Analysis Commands initialized with template support")

    def analyze_users(
        self,
        ctx_obj: Dict[str, Any],
        user: Optional[str],
        limit: int,
        output: Optional[str],
        output_format: str,
        analyze_all: bool = False,
    ):
        """Analyze user activity with template integration."""
        try:
            from .persistence import analyze_user, get_database_manager

            if user and analyze_all:
                click.echo(
                    "❌ Please specify either --user or --all, not both.", err=True
                )
                return

            if analyze_all:
                from pepino.analysis.data_facade import get_analysis_data_facade

                settings = Settings()
                db_path = ctx_obj.get("db_path") or settings.db_path

                aggregated_results: List[Dict[str, Any]] = []
                errors: List[Dict[str, Any]] = []

                try:
                    with get_database_manager(db_path) as db_manager:
                        data_facade = get_analysis_data_facade(
                            db_manager, settings.base_filter
                        )
                        user_analyzer = UserAnalyzer(data_facade)

                        metadata_list = user_analyzer.get_available_users_metadata()
                        if not metadata_list:
                            self.show_template_error(
                                "User analysis", "No users available for analysis"
                            )
                            return

                        for metadata in metadata_list:
                            username_value = metadata.get("author_name")
                            if not username_value:
                                errors.append(
                                    {
                                        "author_name": username_value,
                                        "error": "User record missing author_name; skipped",
                                    }
                                )
                                continue

                            try:
                                analysis_model = user_analyzer.analyze_enhanced(
                                    username_value, include_semantic=False
                                )
                                if not analysis_model:
                                    errors.append(
                                        {
                                            "author_name": username_value,
                                            "error": "No analysis data returned",
                                        }
                                    )
                                    continue

                                if hasattr(analysis_model, "model_dump"):
                                    result_dict = analysis_model.model_dump()
                                elif hasattr(analysis_model, "dict"):
                                    result_dict = analysis_model.dict()
                                else:
                                    result_dict = analysis_model

                                author_id = metadata.get("author_id") or username_value
                                display_name = (
                                    metadata.get("display_name") or username_value
                                )

                                stats_dict = result_dict.get("statistics") or {}
                                curated_stats = {
                                    "message_count": stats_dict.get("message_count"),
                                    "channels_active": stats_dict.get(
                                        "channels_active"
                                    ),
                                    "active_days": stats_dict.get("active_days"),
                                    "avg_message_length": stats_dict.get(
                                        "avg_message_length"
                                    ),
                                }
                                curated_stats = {
                                    key: value
                                    for key, value in curated_stats.items()
                                    if value is not None
                                }

                                first_message_at = metadata.get(
                                    "first_message_at"
                                ) or stats_dict.get("first_message_date")
                                last_message_at = metadata.get(
                                    "last_message_at"
                                ) or stats_dict.get("last_message_date")

                                user_payload: Dict[str, Any] = {
                                    "user_id": author_id,
                                    "username": username_value,
                                    "display_name": display_name,
                                    "is_bot": metadata.get("is_bot", False),
                                    "guilds": metadata.get("guilds", []),
                                    "account_created_at": metadata.get(
                                        "account_created_at"
                                    ),
                                    "first_message_at": first_message_at,
                                    "last_message_at": last_message_at,
                                    "channel_activity": result_dict.get(
                                        "channel_activity"
                                    ),
                                    "time_patterns": result_dict.get("time_patterns"),
                                    "semantic_analysis": result_dict.get(
                                        "semantic_analysis"
                                    ),
                                    "top_topics": result_dict.get("top_topics"),
                                    "full_analysis": result_dict,
                                }

                                if curated_stats:
                                    user_payload["statistics"] = curated_stats

                                aggregated_results.append(user_payload)
                            except Exception as exc:
                                errors.append(
                                    {
                                        "author_name": username_value,
                                        "error": f"Unexpected error: {exc}",
                                    }
                                )

                except Exception as exc:
                    self.show_template_error(
                        "User analysis", f"Failed to analyze users: {exc}"
                    )
                    return

                aggregated_payload: Dict[str, Any] = {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "total_users": len(metadata_list),
                    "analyzed_users": len(aggregated_results),
                    "users": aggregated_results,
                }

                if errors:
                    aggregated_payload["errors"] = errors

                final_format = output_format
                if final_format != "json":
                    final_format = "json"
                    if ctx_obj.get("verbose"):
                        click.echo(
                            "ℹ️  --format text/csv is not supported with --all. Forcing JSON output.",
                            err=True,
                        )

                target_output = output
                if not target_output:
                    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                    target_output = f"stats/user_analysis_all_{timestamp}.json"

                self.handle_output(aggregated_payload, target_output, final_format)
                return

            data = analyze_user(user, limit, ctx_obj.get("db_path"))

            # Handle response based on analysis type
            if user and "user_analysis" in data:
                result = data["user_analysis"]
                self.handle_analysis_result(
                    result,
                    "User analysis",
                    "user_analysis.txt.j2",
                    output,
                    output_format,
                )
            else:
                # Handle list of top users with template
                result = data.get("top_users", [])
                enhanced_data = {
                    "top_users": result,
                    "limit": limit,
                    "total_users": len(result),
                }
                self.handle_analysis_result(
                    enhanced_data,
                    "Top users analysis",
                    "top_users.txt.j2",
                    output,
                    output_format,
                )

        except RuntimeError as e:
            self.show_template_error("User analysis", str(e))
        except Exception as e:
            self.show_template_error("User analysis", f"Unexpected error: {e}")
            if ctx_obj.get("verbose"):
                raise

    def analyze_channels(
        self,
        ctx_obj: Dict[str, Any],
        channel: Optional[str],
        limit: int,
        output: Optional[str],
        output_format: str,
        analyze_all: bool = False,
    ):
        """Analyze channel activity with template integration."""
        try:
            from pepino.analysis.data_facade import get_analysis_data_facade

            from .persistence import analyze_channel, get_database_manager

            if analyze_all:
                settings = Settings()
                db_path = ctx_obj.get("db_path") or settings.db_path

                aggregated_results: List[Dict[str, Any]] = []
                errors: List[Dict[str, Any]] = []
                channel_metadata_list: List[Dict[str, Any]] = []

                try:
                    with get_database_manager(db_path) as db_manager:
                        data_facade = get_analysis_data_facade(
                            db_manager, settings.base_filter
                        )
                        channel_analyzer = ChannelAnalyzer(data_facade)
                        try:
                            topic_analyzer = TopicAnalyzer(data_facade)
                        except Exception as e:
                            topic_analyzer = None
                            logger.warning(
                                f"Topic analyzer unavailable for channel aggregation: {e}"
                            )

                        channel_metadata_list = (
                            channel_analyzer.get_available_channels_metadata()
                        )
                        parent_name_lookup = {
                            meta.get("channel_id"): meta.get("channel_name")
                            for meta in channel_metadata_list
                            if meta.get("channel_id")
                        }

                        if not channel_metadata_list:
                            self.show_template_error(
                                "Channel analysis", "No channels available for analysis"
                            )
                            return

                        for channel_meta in channel_metadata_list:
                            channel_name = channel_meta.get("channel_name")
                            channel_id = channel_meta.get("channel_id")
                            if not channel_name:
                                errors.append(
                                    {
                                        "channel_name": channel_name,
                                        "channel_id": channel_id,
                                        "error": "Channel name is empty",
                                    }
                                )
                                continue

                            try:
                                analysis_model = channel_analyzer.analyze(
                                    channel_name,
                                    include_patterns=True,
                                    channel_id=channel_id,
                                )
                                if not analysis_model:
                                    errors.append(
                                        {
                                            "channel_name": channel_name,
                                            "channel_id": channel_id,
                                            "error": "No analysis data returned",
                                        }
                                    )
                                    continue

                                if hasattr(analysis_model, "model_dump"):
                                    result_dict = analysis_model.model_dump()
                                elif hasattr(analysis_model, "dict"):
                                    result_dict = analysis_model.dict()
                                elif isinstance(analysis_model, dict):
                                    result_dict = analysis_model
                                else:
                                    raise RuntimeError(
                                        f"Channel analysis result for {channel_name} is not serializable: {type(analysis_model)}"
                                    )

                                total_human_members = 0
                                try:
                                    total_human_members = data_facade.channel_repository.get_channel_human_member_count(
                                        channel_name, channel_id=channel_id
                                    )
                                except Exception as e:
                                    logger.warning(
                                        f"Could not get total human members for channel {channel_name}: {e}"
                                    )

                                top_users = result_dict.get("top_users") or []
                                participation_summary = None
                                if top_users:
                                    top_5_messages = sum(
                                        user.get("message_count", 0)
                                        for user in top_users[:5]
                                    )
                                    total_messages = result_dict.get(
                                        "statistics", {}
                                    ).get("total_messages", 0)
                                    if total_messages > 0:
                                        concentration = (
                                            top_5_messages / total_messages
                                        ) * 100
                                        if concentration > 70:
                                            participation_summary = f"5 top contributors posted {concentration:.0f}% of all messages (highly concentrated)"
                                        elif concentration > 50:
                                            participation_summary = f"5 top contributors posted {concentration:.0f}% of all messages"
                                        else:
                                            participation_summary = f"5 top contributors posted {concentration:.0f}% of all messages (well distributed)"

                                lost_interest_summary = None
                                lost_interest_users: List[Dict[str, Any]] = []
                                try:
                                    all_users = data_facade.channel_repository.get_channel_user_activity(
                                        channel_name,
                                        days=None,
                                        limit=100,
                                        channel_id=channel_id,
                                    )
                                    recent_users = data_facade.channel_repository.get_channel_user_activity(
                                        channel_name,
                                        days=30,
                                        limit=100,
                                        channel_id=channel_id,
                                    )

                                    recent_usernames = {
                                        user["author_name"] for user in recent_users
                                    }
                                    inactive_users = [
                                        user
                                        for user in all_users
                                        if user["author_name"] not in recent_usernames
                                    ]

                                    if inactive_users:
                                        now = datetime.now(timezone.utc)
                                        for user in inactive_users[:5]:
                                            last_message_ts = user.get("last_message")
                                            if not last_message_ts:
                                                continue
                                            try:
                                                last_msg_date = datetime.fromisoformat(
                                                    last_message_ts.replace(
                                                        "Z", "+00:00"
                                                    )
                                                )
                                            except Exception:
                                                continue
                                            days_inactive = (now - last_msg_date).days
                                            lost_interest_users.append(
                                                {
                                                    "display_name": user.get(
                                                        "author_display_name"
                                                    ),
                                                    "author_name": user["author_name"],
                                                    "days_inactive": days_inactive,
                                                    "message_count": user.get(
                                                        "message_count", 0
                                                    ),
                                                }
                                            )

                                        if lost_interest_users:
                                            lost_interest_users.sort(
                                                key=lambda x: (
                                                    x["days_inactive"],
                                                    x["message_count"],
                                                ),
                                                reverse=True,
                                            )
                                            inactive_count = len(lost_interest_users)
                                            if inactive_count > 0:
                                                lost_interest_summary = f"{inactive_count} former contributors inactive for at least 30 days"
                                except Exception as e:
                                    logger.warning(
                                        f"Could not calculate lost interest for channel {channel_name}: {e}"
                                    )

                                engagement_summary = None
                                engagement_metrics = result_dict.get(
                                    "engagement_metrics"
                                )
                                if engagement_metrics:
                                    reaction_rate = engagement_metrics.get(
                                        "reaction_rate"
                                    )
                                    if reaction_rate is not None:
                                        if reaction_rate > 80:
                                            engagement_summary = f"High engagement ({reaction_rate:.0f}% reaction rate)"
                                        elif reaction_rate > 50:
                                            engagement_summary = f"Moderate engagement ({reaction_rate:.0f}% reaction rate)"
                                        else:
                                            engagement_summary = f"Low engagement ({reaction_rate:.0f}% reaction rate)"

                                trend_summary = None
                                try:
                                    current_messages = data_facade.message_repository.get_channel_messages(
                                        channel_name, days_back=7, channel_id=channel_id
                                    )
                                    previous_messages = data_facade.message_repository.get_channel_messages(
                                        channel_name,
                                        days_back=14,
                                        limit=len(current_messages) * 2
                                        if current_messages
                                        else None,
                                        channel_id=channel_id,
                                    )

                                    if current_messages and previous_messages:
                                        current_count = len(
                                            [
                                                m
                                                for m in current_messages
                                                if not m.get("author_is_bot", False)
                                            ]
                                        )
                                        previous_count = len(
                                            [
                                                m
                                                for m in previous_messages
                                                if not m.get("author_is_bot", False)
                                            ]
                                        )
                                        if previous_count > 0:
                                            change_percent = (
                                                (current_count - previous_count)
                                                / previous_count
                                            ) * 100
                                            if change_percent > 20:
                                                trend_summary = f"Activity increasing (+{change_percent:.0f}% compared to previous 7 days)"
                                            elif change_percent < -20:
                                                trend_summary = f"Activity decreasing ({change_percent:.0f}% compared to previous 7 days)"
                                            else:
                                                trend_summary = f"Activity stable ({change_percent:+.0f}% compared to previous 7 days)"
                                except Exception as e:
                                    logger.warning(
                                        f"Could not calculate trend for channel {channel_name}: {e}"
                                    )

                                bot_activity_summary = None
                                statistics = result_dict.get("statistics") or {}
                                bot_messages = statistics.get("bot_messages", 0)
                                total_messages = statistics.get("total_messages", 0)
                                human_messages = statistics.get("human_messages", 0)
                                if total_messages > 0 and bot_messages:
                                    bot_percentage = (
                                        bot_messages / total_messages
                                    ) * 100
                                    if bot_messages > human_messages:
                                        bot_activity_summary = f"Bots posted {bot_percentage:.0f}% of messages (more than humans)"
                                    elif bot_percentage > 10:
                                        bot_activity_summary = f"Bots posted {bot_percentage:.0f}% of messages (less than humans)"

                                recent_activity_summary = None
                                try:
                                    recent_messages = data_facade.message_repository.get_channel_messages(
                                        channel_name, days_back=7, channel_id=channel_id
                                    )
                                    if recent_messages:
                                        human_recent = len(
                                            [
                                                m
                                                for m in recent_messages
                                                if not m.get("author_is_bot", False)
                                            ]
                                        )
                                        previous_messages = data_facade.message_repository.get_channel_messages(
                                            channel_name,
                                            days_back=14,
                                            limit=len(recent_messages) * 2,
                                            channel_id=channel_id,
                                        )
                                        if previous_messages:
                                            human_previous = len(
                                                [
                                                    m
                                                    for m in previous_messages
                                                    if not m.get("author_is_bot", False)
                                                ]
                                            )
                                            if human_previous > 0:
                                                change_percent = (
                                                    (human_recent - human_previous)
                                                    / human_previous
                                                ) * 100
                                                if change_percent >= 0:
                                                    recent_activity_summary = f"{human_recent} messages in last 7 days (up {change_percent:.0f}% vs previous 7 days)"
                                                else:
                                                    recent_activity_summary = f"{human_recent} messages in last 7 days (down {abs(change_percent):.0f}% vs previous 7 days)"
                                        if recent_activity_summary is None:
                                            recent_activity_summary = f"{human_recent} messages in last 7 days"
                                except Exception as e:
                                    logger.warning(
                                        f"Could not calculate recent activity for channel {channel_name}: {e}"
                                    )

                                if topic_analyzer and not result_dict.get("top_topics"):
                                    try:
                                        topic_result = topic_analyzer.analyze(
                                            channel_name=channel_name,
                                            top_n=5,
                                            days_back=None,
                                        )
                                        if topic_result:
                                            if hasattr(topic_result, "model_dump"):
                                                topic_payload = (
                                                    topic_result.model_dump()
                                                )
                                            elif hasattr(topic_result, "dict"):
                                                topic_payload = topic_result.dict()
                                            elif isinstance(topic_result, dict):
                                                topic_payload = topic_result
                                            else:
                                                topic_payload = {}
                                            result_dict[
                                                "top_topics"
                                            ] = topic_payload.get("topics", [])
                                    except Exception as e:
                                        logger.warning(
                                            f"Topic analysis failed for channel {channel_name}: {e}"
                                        )

                                result_dict["total_human_members"] = total_human_members
                                result_dict[
                                    "participation_summary"
                                ] = participation_summary
                                result_dict[
                                    "lost_interest_summary"
                                ] = lost_interest_summary
                                result_dict["lost_interest_users"] = lost_interest_users
                                result_dict["engagement_summary"] = engagement_summary
                                result_dict["trend_summary"] = trend_summary
                                result_dict[
                                    "bot_activity_summary"
                                ] = bot_activity_summary
                                result_dict[
                                    "recent_activity_summary"
                                ] = recent_activity_summary
                                result_dict["channel_health"] = True

                                parent_id = channel_meta.get("parent_id")
                                parent_name = (
                                    channel_meta.get("parent_name")
                                    if channel_meta
                                    else None
                                )
                                if not parent_name and parent_id:
                                    parent_name = parent_name_lookup.get(parent_id)

                                metadata_payload = {
                                    "channel_id": channel_id,
                                    "channel_name": channel_name,
                                    "parent_id": parent_id,
                                }
                                if parent_name:
                                    metadata_payload["parent_name"] = parent_name

                                channel_info = result_dict.get("channel_info") or {}
                                channel_info["channel_id"] = channel_id
                                channel_info["parent_id"] = parent_id
                                if parent_name:
                                    channel_info["parent_name"] = parent_name
                                result_dict["channel_info"] = channel_info
                                result_dict["metadata"] = metadata_payload.copy()

                                aggregated_results.append(
                                    {
                                        "channel_id": channel_id,
                                        "channel_name": channel_name,
                                        "metadata": metadata_payload,
                                        "message_statistics": result_dict.get(
                                            "statistics"
                                        ),
                                        "user_activity": {
                                            "top_users": result_dict.get("top_users"),
                                            "lost_interest_users": lost_interest_users,
                                        },
                                        "time_patterns": {
                                            "peak_activity": result_dict.get(
                                                "peak_activity"
                                            ),
                                            "recent_activity": result_dict.get(
                                                "recent_activity"
                                            ),
                                            "daily_activity": result_dict.get(
                                                "daily_activity_data"
                                            ),
                                        },
                                        "content_analysis": {
                                            "top_topics": result_dict.get("top_topics"),
                                            "content_clusters": result_dict.get(
                                                "content_clusters"
                                            ),
                                            "keywords": result_dict.get("keywords"),
                                        },
                                        "engagement_metrics": result_dict.get(
                                            "engagement_metrics"
                                        ),
                                        "health_metrics": result_dict.get(
                                            "health_metrics"
                                        ),
                                        "summaries": {
                                            "participation": participation_summary,
                                            "trend": trend_summary,
                                            "recent_activity": recent_activity_summary,
                                            "bot_activity": bot_activity_summary,
                                            "lost_interest": lost_interest_summary,
                                            "engagement": engagement_summary,
                                        },
                                        "full_analysis": result_dict,
                                    }
                                )
                            except Exception as e:
                                errors.append(
                                    {
                                        "channel_name": channel_name,
                                        "channel_id": channel_id,
                                        "error": f"Unexpected error: {e}",
                                    }
                                )
                except Exception as e:
                    self.show_template_error(
                        "Channel analysis", f"Failed to analyze channels: {e}"
                    )
                    return

                aggregated_payload = {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "total_channels": len(channel_metadata_list),
                    "analyzed_channels": sum(
                        1
                        for result in aggregated_results
                        if result.get("full_analysis")
                    ),
                    "channels": aggregated_results,
                }

                if errors:
                    aggregated_payload["errors"] = errors

                final_format = output_format
                if final_format != "json":
                    final_format = "json"
                    if ctx_obj.get("verbose"):
                        click.echo(
                            "ℹ️  --format text/csv is not supported with --all. Forcing JSON output.",
                            err=True,
                        )

                target_output = output
                if not target_output:
                    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                    target_output = f"stats/channel_analysis_all_{timestamp}.json"

                self.handle_output(
                    aggregated_payload,
                    target_output,
                    final_format,
                )
                return

            data = analyze_channel(channel, limit, ctx_obj.get("db_path"))

            # Handle response based on analysis type
            if channel and "channel_analysis" in data:
                result = data["channel_analysis"]
                self.handle_analysis_result(
                    result,
                    "Channel analysis",
                    "channel_analysis.txt.j2",
                    output,
                    output_format,
                )
            else:
                # Handle list of top channels with template
                result = data.get("top_channels", [])
                enhanced_data = {
                    "top_channels": result,
                    "limit": limit,
                    "total_channels": len(result),
                }
                self.handle_analysis_result(
                    enhanced_data,
                    "Top channels analysis",
                    "top_channels.txt.j2",
                    output,
                    output_format,
                )

        except RuntimeError as e:
            self.show_template_error("Channel analysis", str(e))
        except Exception as e:
            self.show_template_error("Channel analysis", f"Unexpected error: {e}")
            if ctx_obj.get("verbose"):
                raise

    def analyze_topics(
        self,
        ctx_obj: Dict[str, Any],
        channel: Optional[str],
        n_topics: int,
        days_back: Optional[int],
        output: Optional[str],
        output_format: str,
    ):
        """Analyze topics with template integration and NLP capabilities."""
        try:
            from ..analysis.data_facade import get_analysis_data_facade
            from ..analysis.topic_analyzer import TopicAnalyzer
            from ..config import Settings
            from .persistence import get_database_manager

            # Enhanced topic analysis with NLP capabilities
            settings = Settings()
            db_path = ctx_obj.get("db_path") or settings.db_path

            with get_database_manager(db_path) as db_manager:
                # Create data facade with proper base filter
                data_facade = get_analysis_data_facade(db_manager, settings.base_filter)

                # Create topic analyzer
                topic_analyzer = TopicAnalyzer(data_facade)

                # Perform analysis
                analysis_result = topic_analyzer.analyze(
                    channel_name=channel, top_n=n_topics, days_back=days_back
                )

                if not analysis_result:
                    self.show_template_error(
                        "Topic analysis", "No analysis result returned"
                    )
                    return

                # Check if we got an error response
                if hasattr(analysis_result, "error") and analysis_result.error:
                    self.show_template_error("Topic analysis", analysis_result.error)
                    return

                # Check if we have topics
                if not hasattr(analysis_result, "topics") or not analysis_result.topics:
                    self.show_template_error(
                        "Topic analysis", "No topics found for the specified criteria"
                    )
                    return

                # Get recent messages for NLP analysis context (increased limit for better concept extraction)
                recent_messages = []
                try:
                    if channel:
                        messages_data = (
                            data_facade.message_repository.get_messages_by_channel(
                                channel, limit=500
                            )
                        )
                    else:
                        messages_data = (
                            data_facade.message_repository.get_recent_messages(
                                limit=500, days_back=days_back
                            )
                        )

                    if messages_data:
                        recent_messages = [
                            {
                                "id": msg.get("id"),
                                "content": msg.get("content", ""),
                                "author": msg.get("username") or msg.get("author", ""),
                                "timestamp": msg.get("timestamp", ""),
                                "channel": msg.get("channel_name", channel or "all"),
                            }
                            for msg in messages_data
                        ]
                except Exception as e:
                    if ctx_obj.get("verbose"):
                        self.show_template_error(
                            "Message context loading",
                            f"Could not fetch messages for NLP analysis: {e}",
                        )

                # Prepare enhanced template data with NLP capabilities
                template_data = {
                    "channel_name": channel,
                    "days_back": days_back,
                    "n_topics": n_topics,
                    "analysis": analysis_result,
                    "topics": analysis_result.topics,
                    "message_count": analysis_result.message_count,
                    "capabilities_used": analysis_result.capabilities_used
                    if hasattr(analysis_result, "capabilities_used")
                    else ["topic_analysis"],
                }

                # Add domain analysis data if available from hybrid approach
                if hasattr(analysis_result, "_domain_analysis"):
                    template_data["_domain_analysis"] = analysis_result._domain_analysis

                # Convert Pydantic model to dict for compatibility
                result_dict = (
                    analysis_result.model_dump()
                    if hasattr(analysis_result, "model_dump")
                    else analysis_result
                )
                final_data = {**template_data, **result_dict}

                # Handle template rendering with NLP context
                self.handle_analysis_result(
                    final_data,
                    "Topic analysis",
                    "topic_analysis.txt.j2",
                    output,
                    output_format,
                    data_facade=data_facade,
                    messages=recent_messages,
                )

        except RuntimeError as e:
            self.show_template_error("Topic analysis", str(e))
        except Exception as e:
            self.show_template_error("Topic analysis", f"Unexpected error: {e}")
            if ctx_obj.get("verbose"):
                raise

    def analyze_temporal(
        self,
        ctx_obj: Dict[str, Any],
        channel: Optional[str],
        days_back: Optional[int],
        granularity: str,
        output: Optional[str],
        output_format: str,
    ):
        """Analyze temporal patterns with template integration."""
        try:
            from .persistence import analyze_temporal

            data = analyze_temporal(
                channel, days_back, granularity, ctx_obj.get("db_path")
            )

            # Handle Pydantic response models
            if "temporal_analysis" in data:
                result = data["temporal_analysis"]
                self.handle_analysis_result(
                    result,
                    "Temporal analysis",
                    "temporal_analysis.txt.j2",
                    output,
                    output_format,
                )
            else:
                self.handle_output(data, output, output_format)

        except RuntimeError as e:
            self.show_template_error("Temporal analysis", str(e))
        except Exception as e:
            self.show_template_error("Temporal analysis", f"Unexpected error: {e}")
            if ctx_obj.get("verbose"):
                raise


# Global CLI analysis commands instance
_cli_analysis = CLIAnalysisCommands()


def validate_db_path(ctx, param, value):
    """Click callback to validate database path."""
    if not value:
        return value

    try:
        # Convert to Path object for better handling
        path = Path(value)

        # Resolve to absolute path
        abs_path = path.resolve()

        # Check if parent directory exists or can be created
        parent_dir = abs_path.parent
        if not parent_dir.exists():
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
                click.echo(f"Created database directory: {parent_dir}", err=True)
            except PermissionError:
                raise click.BadParameter(
                    f"Cannot create database directory: {parent_dir} (permission denied)"
                )
            except OSError as e:
                raise click.BadParameter(
                    f"Cannot create database directory: {parent_dir} ({e})"
                )

        # Check write permissions on parent directory
        if not parent_dir.is_dir():
            raise click.BadParameter(
                f"Database path parent is not a directory: {parent_dir}"
            )

        # Check write permissions using os.access
        import os

        if not os.access(parent_dir, os.W_OK):
            raise click.BadParameter(
                f"No write permission for database directory: {parent_dir}"
            )

        # If database file exists, check if it's readable
        if abs_path.exists():
            if not abs_path.is_file():
                raise click.BadParameter(
                    f"Database path exists but is not a file: {abs_path}"
                )
            if not os.access(abs_path, os.R_OK):
                raise click.BadParameter(
                    f"Cannot read existing database file: {abs_path}"
                )

        return str(abs_path)

    except click.BadParameter:
        raise
    except Exception as e:
        raise click.BadParameter(f"Invalid database path '{value}': {e}")


@click.group()
@click.option(
    "--db-path",
    default="data/discord_messages.db",
    callback=validate_db_path,
    help="Database path",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def cli(ctx, db_path: str, verbose: bool):
    """Discord Analytics CLI - Analyze Discord server data."""
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db_path
    ctx.obj["verbose"] = verbose

    # Set up professional logging
    setup_cli_logging(verbose=verbose)


@cli.command(name="help")
@click.pass_context
def help_command(ctx):
    """Show comprehensive help for all available commands."""
    help_text = """
🤖 Pepino Analytics CLI - Available Commands

═══════════════════════════════════════════════════════════════════════════════

📊 ANALYSIS COMMANDS (pepino analyze <command>)
───────────────────────────────────────────────────────────────────────────────
  users           Analyze user activity and behavior patterns
                  Options: --user, --limit, --output, --format
                  
  channels        Analyze channel activity and statistics  
                  Options: --channel, --limit, --output, --format
                  
  topics          Analyze topics and themes in messages
                  Options: --channel, --topics, --days, --output, --format
                  
  temporal        Analyze temporal patterns and activity trends
                  Options: --channel, --days, --granularity, --output, --format
                  
  conversations   Analyze conversation patterns and threads
                  Options: --channel, --output, --format
                  
  similar         Find messages similar to a query text
                  Options: --query, --limit, --threshold, --output, --format
                  
  embeddings      Generate embeddings for semantic analysis
                  Options: --batch-size, --output, --format
                  
  sentiment       Analyze sentiment of messages
                  Options: --channel, --limit, --output, --format
                  
  duplicates      Detect duplicate or near-duplicate messages
                  Options: --channel, --threshold, --output, --format

🔄 SYNC COMMANDS (pepino sync <command>) [ADMIN ONLY]
───────────────────────────────────────────────────────────────────────────────
  run             Sync Discord data (smart: only updates if stale)
                  Options: --force, --full, --clear, --timeout
                  
  status          Check sync status and data freshness
                  Options: None

📋 LIST COMMANDS (pepino list <command>)
───────────────────────────────────────────────────────────────────────────────
  channels        List available channels for analysis
                  Options: --limit (default: 0 for all), --output, --format
                  
  users           List available users for analysis
                  Options: --limit (default: 0 for all), --output, --format
                  
  stats           Show database statistics and overview
                  Options: --output, --format

⚡ PERFORMANCE COMMANDS (pepino performance <command>) [ADMIN ONLY]
───────────────────────────────────────────────────────────────────────────────
  metrics         Show performance metrics for analysis operations
                  Options: --output, --format
                  
  benchmark       Benchmark analysis operations for optimization
                  Options: --operations, --iterations, --output, --format
                  
  profile         Profile specific analysis operations
                  Options: --operation (required), --args, --output

🧪 TEST COMMANDS (pepino test <command>) [ADMIN ONLY]
───────────────────────────────────────────────────────────────────────────────
  data            Test database data integrity and availability
                  Options: --output, --format
                  
  analysis        Test analysis operations functionality
                  Options: --operation, --sample-size, --output, --format
                  
  templates       Test template rendering system
                  Options: --template, --output, --format
                  
  dependencies    Test system dependencies and requirements
                  Options: --output, --format

🚀 TOP-LEVEL COMMANDS
───────────────────────────────────────────────────────────────────────────────
  start           Start the Discord bot
                  Options: --token, --prefix, --debug
                  
  export-data     Export data from database to files
                  Options: --table (messages/users/channels), --output, --format
                  
  help            Show this help message

═══════════════════════════════════════════════════════════════════════════════

💡 USAGE EXAMPLES:
  pepino analyze users --limit 20 --format json
  pepino analyze users --user alice_dev
  pepino analyze topics --channel general --days 30
  pepino analyze channels --limit 10 --output channels.csv
  pepino sync run --force
  pepino list channels --limit 10
  pepino list users                  # Shows all users (no limit)
  pepino performance metrics
  pepino test data
  pepino start --debug
  pepino export-data --table messages --format json

📝 NOTES:
  • Most commands support --output and --format options for custom output
  • Available formats: text, json, csv (varies by command)
  • Use --help with any command for detailed options and defaults
  • Admin commands (sync, performance, test) require appropriate permissions
  • Time filtering: use --days for recent data, omit for all-time analysis
  • List commands: use --limit 0 or omit --limit to show all results

═══════════════════════════════════════════════════════════════════════════════
"""
    click.echo(help_text)


@cli.group(name="analyze")
@click.pass_context
def analyze(ctx):
    """Analyze Discord data and generate insights."""
    pass


@analyze.command(name="users")
@click.option("--user", "-u", help="Specific user to analyze")
@click.option(
    "--limit", "-l", default=10, help="Number of top users to show (default: 10)"
)
@click.option("--output", "-o", help="Output file (JSON or CSV)")
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json", "csv"]),
    help="Output format",
)
@click.option(
    "--all",
    "analyze_all",
    is_flag=True,
    help="Analyze every user and aggregate the results",
)
@click.pass_context
def analyze_users(
    ctx,
    user: Optional[str],
    limit: int,
    output: Optional[str],
    output_format: str,
    analyze_all: bool,
):
    """Analyze user activity and statistics."""
    _cli_analysis.analyze_users(
        ctx.obj, user, limit, output, output_format, analyze_all
    )


@analyze.command(name="channels")
@click.option("--channel", "-c", help="Specific channel to analyze")
@click.option(
    "--limit", "-l", default=10, help="Number of top channels to show (default: 10)"
)
@click.option("--output", "-o", help="Output file (JSON or CSV)")
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json", "csv"]),
    help="Output format",
)
@click.option(
    "--all",
    "analyze_all",
    is_flag=True,
    help="Analyze every channel and aggregate the results",
)
@click.pass_context
def analyze_channels(
    ctx,
    channel: Optional[str],
    limit: int,
    output: Optional[str],
    output_format: str,
    analyze_all: bool,
):
    """Analyze channel activity and statistics."""
    if channel and analyze_all:
        click.echo("❌ Please specify either --channel or --all, not both.", err=True)
        ctx.exit(1)

    _cli_analysis.analyze_channels(
        ctx.obj, channel, limit, output, output_format, analyze_all
    )


@analyze.command(name="topics")
@click.option("--channel", "-c", help="Specific channel to analyze (optional)")
@click.option(
    "--topics",
    "-t",
    "n_topics",
    default=20,
    help="Number of top topics to show (default: 20)",
)
@click.option(
    "--days",
    "-d",
    "days_back",
    type=int,
    default=None,
    help="Days to look back (optional, default: all time)",
)
@click.option("--output", "-o", help="Output file (JSON or CSV)")
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json", "csv"]),
    help="Output format",
)
@click.pass_context
def analyze_topics(
    ctx,
    channel: Optional[str],
    n_topics: int,
    days_back: Optional[int],
    output: Optional[str],
    output_format: str,
):
    """Analyze trending topics and keywords."""
    _cli_analysis.analyze_topics(
        ctx.obj, channel, n_topics, days_back, output, output_format
    )


@analyze.command(name="temporal")
@click.option("--channel", "-c", help="Specific channel to analyze (optional)")
@click.option(
    "--days",
    "-d",
    "days_back",
    type=int,
    default=None,
    help="Days to look back (optional, default: all time)",
)
@click.option(
    "--granularity",
    "-g",
    default="day",
    type=click.Choice(["hour", "day", "week"]),
    help="Time granularity",
)
@click.option("--output", "-o", help="Output file (JSON or CSV)")
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json", "csv"]),
    help="Output format",
)
@click.pass_context
def analyze_temporal(
    ctx,
    channel: Optional[str],
    days_back: Optional[int],
    granularity: str,
    output: Optional[str],
    output_format: str,
):
    """Analyze temporal activity patterns."""
    _cli_analysis.analyze_temporal(
        ctx.obj, channel, days_back, granularity, output, output_format
    )


@analyze.command(name="conversations")
@click.option("--channel", help="Channel to analyze conversations for")
@click.option("--output", type=click.Path(), help="Output file path")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "csv", "text"]),
    default="text",
    help="Output format",
)
@click.pass_context
def analyze_conversations(
    ctx, channel: Optional[str], output: Optional[str], output_format: str
):
    """Analyze conversation patterns and engagement."""
    _analyze_conversations(ctx.obj, channel, output, output_format)


@analyze.command(name="similar")
@click.option("--query", required=True, help="Text to find similar messages for")
@click.option("--limit", default=10, help="Number of similar messages to find")
@click.option("--threshold", default=0.5, help="Similarity threshold")
@click.option("--output", type=click.Path(), help="Output file path")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "csv", "text"]),
    default="text",
    help="Output format",
)
@click.pass_context
def find_similar(
    ctx,
    query: str,
    limit: int,
    threshold: float,
    output: Optional[str],
    output_format: str,
):
    """Find messages similar to the given query."""
    _find_similar(ctx.obj, query, limit, threshold, output, output_format)


@analyze.command(name="embeddings")
@click.option("--batch-size", default=100, help="Batch size for processing")
@click.option("--output", type=click.Path(), help="Output file path")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "csv", "text"]),
    default="text",
    help="Output format",
)
@click.pass_context
def generate_embeddings(
    ctx, batch_size: int, output: Optional[str], output_format: str
):
    """Generate embeddings for all messages."""
    _generate_embeddings(ctx.obj, batch_size, output, output_format)


@analyze.command(name="sentiment")
@click.option("--channel", help="Specific channel to analyze (optional)")
@click.option("--limit", default=100, help="Number of recent messages to analyze")
@click.option("--output", type=click.Path(), help="Output file path")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "csv", "text"]),
    default="text",
    help="Output format",
)
@click.pass_context
def analyze_sentiment(
    ctx, channel: Optional[str], limit: int, output: Optional[str], output_format: str
):
    """Analyze sentiment and extract entities from messages using NLP."""
    try:
        from .persistence import analyze_sentiment_async

        data = analyze_sentiment_async(
            channel_name=channel, days_back=30, db_path=ctx.obj.get("db_path")
        )

        _write_output(data, output, output_format)

        if data.get("error"):
            click.echo(f"❌ {data['error']}", err=True)
        else:
            click.echo(
                f"✅ Sentiment analysis completed! Analyzed {data['total_messages']} messages."
            )

    except Exception as e:
        click.echo(f"❌ Unexpected error analyzing sentiment: {e}", err=True)
        if ctx.obj.get("verbose"):
            raise


@analyze.command(name="duplicates")
@click.option("--channel", help="Specific channel to analyze (optional)")
@click.option(
    "--threshold",
    default=0.9,
    help="Similarity threshold for duplicate detection (0.0-1.0)",
)
@click.option("--output", type=click.Path(), help="Output file path")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "csv", "text"]),
    default="text",
    help="Output format",
)
@click.pass_context
def detect_duplicates(
    ctx,
    channel: Optional[str],
    threshold: float,
    output: Optional[str],
    output_format: str,
):
    """Detect duplicate messages using similarity analysis."""
    try:
        from .persistence import detect_duplicates_async

        data = detect_duplicates_async(
            channel_name=channel,
            similarity_threshold=threshold,
            db_path=ctx.obj.get("db_path"),
        )

        _write_output(data, output, output_format)

        if data.get("error"):
            click.echo(f"❌ {data['error']}", err=True)
        else:
            click.echo(
                f"✅ Duplicate detection completed! Found {data['total_duplicate_pairs']} duplicate pairs."
            )

    except Exception as e:
        click.echo(f"❌ Error detecting duplicates: {e}", err=True)
        if ctx.obj.get("verbose"):
            raise


@cli.command()
@click.option("--table", help="Specific table to export (messages, users, channels)")
@click.option("--output", type=click.Path(), help="Output file path")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "csv"]),
    default="csv",
    help="Output format",
)
@click.pass_context
def export_data(ctx, table: Optional[str], output: Optional[str], output_format: str):
    """Export data from the database."""
    _export_data(ctx.obj, table, output, output_format)


@cli.command(name="start")
@click.option("--token", help="Discord bot token (overrides environment variable)")
@click.option("--prefix", default="!", help="Bot command prefix (default: !)")
@click.option("--debug", is_flag=True, help="Run in debug mode")
@click.pass_context
def start(ctx, token: Optional[str], prefix: str, debug: bool):
    """Start the Discord bot for interactive analysis."""
    try:
        from pepino.discord.bot import run_bot as start_bot

        # Update settings if command line options provided
        settings = Settings()

        if token:
            settings.discord_token = token
        if prefix != "!":
            settings.command_prefix = prefix
        if debug:
            settings.debug = True
            settings.log_level = "DEBUG"

        # Validate required settings
        if not settings.discord_token:
            click.echo(
                "❌ Error: DISCORD_TOKEN is required. Set it via environment variable or --token option.",
                err=True,
            )
            click.echo(
                "   Example: export DISCORD_TOKEN='your_bot_token_here'", err=True
            )
            ctx.exit(1)

        click.echo("🤖 Starting Discord bot...")
        click.echo(f"   Command prefix: {settings.command_prefix}")
        click.echo(f"   Database path: {settings.db_path}")
        click.echo(f"   Debug mode: {settings.debug}")
        click.echo(f"   Log level: {settings.log_level}")
        click.echo("   Press Ctrl+C to stop the bot")
        click.echo("")

        # Start the bot
        start_bot()

    except KeyboardInterrupt:
        click.echo("\n🛑 Bot stopped by user")
    except ValueError as e:
        click.echo(f"❌ Configuration error: {e}", err=True)
        ctx.exit(1)
    except Exception as e:
        click.echo(f"❌ Failed to start bot: {e}", err=True)
        if debug:
            import traceback

            traceback.print_exc()
        ctx.exit(1)


@cli.group(name="sync")
@click.pass_context
def sync(ctx):
    """Sync Discord data and manage sync operations."""
    pass


@sync.command(name="run")
@click.option("--force", is_flag=True, help="Force sync even if data is fresh")
@click.option("--full", is_flag=True, help="Complete re-sync (re-downloads everything)")
@click.option(
    "--clear", is_flag=True, help="Clear existing data before sync (use with --full)"
)
@click.option("--timeout", default=300, help="Sync timeout in seconds (default: 300)")
@click.pass_context
def sync_run(ctx, force: bool, full: bool, clear: bool, timeout: int):
    """Sync Discord data (smart: only updates if data is stale)."""
    if full:
        # Full sync mode
        timeout = (
            timeout if timeout != 300 else 600
        )  # Increase default timeout for full sync
        asyncio.run(
            _sync_data(
                ctx.obj,
                force=True,
                timeout=timeout,
                incremental=False,
                clear_existing=clear,
            )
        )
    else:
        # Incremental sync mode
        if clear:
            click.echo("⚠️  --clear option only works with --full flag", err=True)
            ctx.exit(1)
        asyncio.run(_sync_data(ctx.obj, force, timeout, incremental=True))


@sync.command(name="status")
@click.pass_context
def sync_status(ctx):
    """Check sync status and data freshness."""
    asyncio.run(_sync_status(ctx.obj))


@cli.group(name="list")
@click.pass_context
def list_cmd(ctx):
    """List available data for analysis and automation."""
    pass


@list_cmd.command(name="channels")
@click.option("--limit", "-l", default=0, help="Number of channels to show (0 for all)")
@click.option("--output", "-o", help="Output file (JSON or CSV)")
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json", "csv"]),
    help="Output format",
)
@click.pass_context
def list_channels(ctx, limit: int, output: Optional[str], output_format: str):
    """List all available channels for analysis."""
    _list_channels(ctx.obj, limit, output, output_format)


@list_cmd.command(name="users")
@click.option("--limit", "-l", default=0, help="Number of users to show (0 for all)")
@click.option("--output", "-o", help="Output file (JSON or CSV)")
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json", "csv"]),
    help="Output format",
)
@click.pass_context
def list_users(ctx, limit: int, output: Optional[str], output_format: str):
    """List all available users for analysis."""
    _list_users(ctx.obj, limit, output, output_format)


@list_cmd.command(name="stats")
@click.option("--output", "-o", help="Output file (JSON or CSV)")
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json", "csv"]),
    help="Output format",
)
@click.pass_context
def list_stats(ctx, output: Optional[str], output_format: str):
    """Show database statistics for automation planning."""
    _list_stats(ctx.obj, output, output_format)


def _analyze_conversations(
    ctx_obj: Dict[str, Any],
    channel: Optional[str],
    output: Optional[str],
    output_format: str,
):
    """Analyze conversation patterns."""
    try:
        conversation_service = ConversationService()

        # Analyze conversations
        conversations = conversation_service.analyze_conversations(
            ctx_obj["db_path"], channel
        )

        # Calculate engagement
        engagement = conversation_service.calculate_engagement(
            ctx_obj["db_path"], channel
        )

        data = {"conversations": conversations, "engagement": engagement}

        _write_output(data, output, output_format)
        click.echo("✅ Conversation analysis completed successfully!")

    except Exception as e:
        click.echo(f"❌ Error analyzing conversations: {e}")
        if ctx_obj["verbose"]:
            raise


def _find_similar(
    ctx_obj: Dict[str, Any],
    query: str,
    limit: int,
    threshold: float,
    output: Optional[str],
    output_format: str,
):
    """Find similar messages."""
    try:
        embedding_service = EmbeddingService()
        embedding_service.initialize()

        similar_messages = embedding_service.find_similar_messages(
            ctx_obj["db_path"], query, limit, threshold
        )

        data = {
            "query": query,
            "similar_messages": similar_messages,
            "total_found": len(similar_messages),
        }

        _write_output(data, output, output_format)
        click.echo(f"✅ Found {len(similar_messages)} similar messages!")

    except Exception as e:
        click.echo(f"❌ Error finding similar messages: {e}")
        if ctx_obj["verbose"]:
            raise


def _generate_embeddings(
    ctx_obj: Dict[str, Any], batch_size: int, output: Optional[str], output_format: str
):
    """Generate embeddings for messages."""
    try:
        embedding_service = EmbeddingService()
        embedding_service.initialize()

        with click.progressbar(length=100, label="Generating embeddings") as bar:
            processed_count = embedding_service.batch_process_messages(
                ctx_obj["db_path"], batch_size=batch_size
            )
            bar.update(100)

        # Get statistics
        stats = embedding_service.get_embedding_statistics(ctx_obj["db_path"])

        data = {
            "embedding_stats": stats,
            "processed": processed_count,
            "total_messages": stats.get("total_messages", 0),
            "embedded_messages": stats.get("embedded_messages", 0),
            "coverage_percentage": stats.get("coverage_percentage", 0),
        }

        _write_output(data, output, output_format)
        click.echo(f"✅ Generated embeddings for {processed_count} messages!")

    except Exception as e:
        click.echo(f"❌ Error generating embeddings: {e}")
        if ctx_obj["verbose"]:
            raise


def _export_data(
    ctx_obj: Dict[str, Any],
    table: Optional[str],
    output: Optional[str],
    output_format: str,
):
    """Export data from database."""
    try:
        if table:
            # Export specific table
            data = persistence.export_table_data_async(
                table, output, output_format, ctx_obj["db_path"]
            )
        else:
            # Export all tables
            data = persistence.export_all_tables_async(ctx_obj["db_path"])

        _write_output(data, output, output_format)
        click.echo("✅ Data export completed successfully!")

    except Exception as e:
        click.echo(f"❌ Error exporting data: {e}")
        if ctx_obj.get("verbose"):
            raise


def _format_channel_analysis_for_cli(result: Dict[str, Any]) -> str:
    """Format comprehensive channel analysis for CLI display (Discord-style)."""
    channel_info = result["channel_info"]
    stats = result["statistics"]
    top_users = result.get("top_users", [])
    engagement = result.get("engagement_metrics")
    peak_activity = result.get("peak_activity")
    recent_activity = result.get("recent_activity", [])
    health_metrics = result.get("health_metrics")
    top_topics = result.get("top_topics", [])
    total_human_members = result.get("total_human_members", 0)

    def format_timestamp(timestamp: str) -> str:
        try:
            if not timestamp:
                return "Unknown"
            from datetime import datetime

            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M")
        except:
            return str(timestamp)[:16] if timestamp else "Unknown"

    output = f"📊 Channel Analysis: #{channel_info['channel_name']}\n\n"

    # Basic Statistics with Human/Bot breakdown
    output += "Basic Statistics:\n"
    output += f"• Total Messages: {stats['total_messages']:,}\n"
    if stats.get("human_messages", 0) > 0 and stats.get("bot_messages", 0) > 0:
        human_pct = stats["human_messages"] / stats["total_messages"] * 100
        bot_pct = stats["bot_messages"] / stats["total_messages"] * 100
        output += f"• Human Messages: {stats['human_messages']:,} ({human_pct:.1f}%)\n"
        output += f"• Bot Messages: {stats['bot_messages']:,} ({bot_pct:.1f}%)\n"
    output += f"• Total Unique Users: {stats['unique_users']:,}\n"
    if stats.get("unique_human_users", 0) > 0:
        percentage_str = ""
        if total_human_members and total_human_members > 0:
            percentage = stats["unique_human_users"] / total_human_members * 100
            percentage_str = f" ({percentage:.2f}%)"
        output += (
            f"• Unique Human Users: {stats['unique_human_users']:,}{percentage_str}\n"
        )
    output += (
        f"• Average Message Length: {stats['avg_message_length']:.1f} characters\n"
    )
    output += f"• First Message: {format_timestamp(stats['first_message'])}\n"
    output += f"• Last Message: {format_timestamp(stats['last_message'])}\n\n"

    # Human Engagement Metrics
    if engagement:
        output += "📈 Human Engagement Metrics:\n"
        output += f"• Average Replies per Original Post: {engagement['replies_per_post']:.2f}\n"
        output += f"• Posts with Reactions: {engagement['reaction_rate']:.1f}% ({engagement['posts_with_reactions']}/{stats.get('human_messages', stats['total_messages'])})\n"
        output += f"• Total Replies: {engagement['total_replies']:,} | Original Posts: {engagement['original_posts']:,}\n"
        output += f"• Note: Bot messages excluded from engagement calculations\n\n"

    # Top Contributors
    if top_users:
        output += "👥 Top Human Contributors:\n"
        for user in top_users[:5]:
            display_name = user.get("display_name") or user.get(
                "author_name", "Unknown"
            )
            output += f"• {display_name} ({user['message_count']:,} messages)\n"
        output += "\n"

    # Peak Activity Times
    if peak_activity:
        if peak_activity.get("peak_hours"):
            output += "Peak Activity Hours:\n"
            for hour_data in peak_activity["peak_hours"][:3]:
                output += f"• {hour_data['hour']}: {hour_data['messages']:,} messages\n"
            output += "\n"

        if peak_activity.get("peak_days"):
            output += "Peak Activity Days:\n"
            for day_data in peak_activity["peak_days"][:3]:
                output += f"• {day_data['day']}: {day_data['messages']:,} messages\n"
            output += "\n"

    # Recent Activity (Last 7 Days)
    if recent_activity:
        output += "Recent Activity (Last 7 Days):\n"
        for activity in recent_activity[:7]:
            output += f"• {activity['date']}: {activity['messages']:,} messages\n"
        output += "\n"

    # Channel Health Metrics
    if health_metrics:
        output += "📈 Channel Health Metrics:\n"
        output += f"• Weekly Active Users: {health_metrics['weekly_active']:,}\n"
        output += f"• Inactive Users: {health_metrics['inactive_users']:,}\n"
        if health_metrics.get("total_channel_members", 0) > 0:
            output += f"• Total Channel Members: {health_metrics['total_channel_members']:,}\n"
            output += f"• Participation Rate: {(health_metrics['participation_rate'] * 100):.1f}%\n"
        output += "\n"

    # Top Topics Discussed
    if top_topics:
        output += "🔍 Top Topics:\n"
        for i, topic in enumerate(top_topics[:5], 1):
            if topic and topic.strip():  # Only show non-empty topics
                output += f"• {topic}\n"

    return output


def _write_output(data: Dict[str, Any], output: Optional[str], output_format: str):
    """Write output to file or stdout."""
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_format == "json":
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        elif output_format == "csv":
            _write_csv(data, output_path)
        else:  # text
            with open(output_path, "w") as f:
                _write_text(data, f)
    else:
        if output_format == "json":
            click.echo(json.dumps(data, indent=2, default=str))
        elif output_format == "csv":
            _write_csv(data, None)
        else:  # text
            _write_text(data, None)


def _write_csv(data: Dict[str, Any], output_path: Optional[Path]):
    """Write data as CSV."""
    if "tables" in data:
        # Multiple tables
        for table_name, table_data in data["tables"].items():
            if output_path:
                csv_path = output_path.parent / f"{table_name}.csv"
            else:
                csv_path = None

            if csv_path:
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=table_data["columns"])
                    writer.writeheader()
                    writer.writerows(table_data["rows"])
            else:
                # Write to stdout
                import sys

                writer = csv.DictWriter(sys.stdout, fieldnames=table_data["columns"])
                writer.writeheader()
                writer.writerows(table_data["rows"])
    else:
        # Single table or analysis result
        if output_path:
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                _flatten_dict_to_csv(data, writer)
        else:
            import sys

            writer = csv.writer(sys.stdout)
            _flatten_dict_to_csv(data, writer)


def _flatten_dict_to_csv(data: Dict[str, Any], writer):
    """Flatten nested dictionary to CSV format."""

    def flatten_dict(d, parent_key="", sep="_"):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(
                            flatten_dict(item, f"{new_key}_{i}", sep=sep).items()
                        )
                    else:
                        items.append((f"{new_key}_{i}", item))
            else:
                items.append((new_key, v))
        return dict(items)

    flattened = flatten_dict(data)
    writer.writerow(flattened.keys())
    writer.writerow(flattened.values())


def _write_text(data: Dict[str, Any], output_file):
    """Write data as formatted text."""

    def write_text_recursive(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                output_file.write(f"{'  ' * indent}{key}:\n")
                write_text_recursive(value, indent + 1)
            elif isinstance(value, list):
                output_file.write(f"{'  ' * indent}{key}:\n")
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        output_file.write(f"{'  ' * (indent + 1)}[{i}]:\n")
                        write_text_recursive(item, indent + 2)
                    else:
                        output_file.write(f"{'  ' * (indent + 1)}[{i}]: {item}\n")
            else:
                output_file.write(f"{'  ' * indent}{key}: {value}\n")

    if output_file:
        write_text_recursive(data)
    else:
        # Write to stdout
        import sys

        for key, value in data.items():
            if isinstance(value, dict):
                click.echo(f"{key}:")
                for k, v in value.items():
                    click.echo(f"  {k}: {v}")
            elif isinstance(value, list):
                click.echo(f"{key}:")
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        click.echo(f"  [{i}]:")
                        for k, v in item.items():
                            click.echo(f"    {k}: {v}")
                    else:
                        click.echo(f"  [{i}]: {item}")
            else:
                click.echo(f"{key}: {value}")


async def _sync_data(
    ctx_obj: Dict[str, Any],
    force: bool,
    timeout: int,
    incremental: bool = True,
    clear_existing: bool = False,
):
    """Perform data sync operation."""
    try:
        import asyncio

        from pepino.discord.sync.sync_manager import SyncManager

        # Create sync manager
        sync_manager = SyncManager(ctx_obj["db_path"])

        click.echo("🔄 Starting Discord data sync...")
        if incremental and not force:
            click.echo("   Mode: Incremental (only syncs if data is stale)")
        elif incremental:
            click.echo("   Mode: Incremental (forced)")
        else:
            click.echo("   Mode: Full sync")
            if clear_existing:
                click.echo("   ⚠️  Clearing existing data first")

        # Run sync with timeout
        try:
            if incremental:
                result = await asyncio.wait_for(
                    sync_manager.run_incremental_sync(force=force), timeout=timeout
                )
            else:
                result = await asyncio.wait_for(
                    sync_manager.run_full_sync(clear_existing=clear_existing),
                    timeout=timeout,
                )

            # Display results
            if result.sync_performed:
                click.echo(f"✅ Sync completed successfully!")
                click.echo(f"   New messages: {getattr(result, 'new_messages', 0)}")
                if hasattr(result, "updated_channels"):
                    click.echo(f"   Updated channels: {result.updated_channels}")
                click.echo(f"   Duration: {result.duration:.1f}s")

                if hasattr(result, "last_sync") and result.last_sync:
                    click.echo(f"   Last sync: {result.last_sync}")
            else:
                reason = getattr(
                    result, "reason", getattr(result, "error", "Unknown reason")
                )
                click.echo(f"ℹ️  Sync not performed: {reason}")

        except asyncio.TimeoutError:
            click.echo(f"⏰ Sync operation timed out after {timeout} seconds")
            ctx_obj["exit_code"] = 1

    except Exception as e:
        click.echo(f"❌ Error during sync: {e}")
        if ctx_obj.get("verbose"):
            import traceback

            traceback.print_exc()
        ctx_obj["exit_code"] = 1


async def _sync_status(ctx_obj: Dict[str, Any]):
    """Check sync status."""
    try:
        from datetime import datetime, timezone

        from pepino.discord.sync.sync_manager import SyncManager

        # Create sync manager
        sync_manager = SyncManager(ctx_obj["db_path"])

        # Get last sync info
        last_sync = await sync_manager.get_last_sync_time()
        is_stale = await sync_manager.is_data_stale()

        click.echo("📊 Sync Status")
        click.echo("=" * 40)

        if last_sync:
            click.echo(f"Last sync: {last_sync}")

            # Calculate time since last sync
            now = datetime.now(timezone.utc)
            time_diff = now - last_sync
            hours = int(time_diff.total_seconds() / 3600)
            minutes = int((time_diff.total_seconds() % 3600) / 60)

            click.echo(f"Time since last sync: {hours}h {minutes}m ago")

            if is_stale:
                click.echo("🟡 Data is stale (sync recommended)")
            else:
                click.echo("🟢 Data is fresh")
        else:
            click.echo("❌ No sync records found")
            click.echo("🔄 Initial sync required")

        # Get database info
        try:
            # Get database statistics using persistence layer
            stats = await persistence.get_database_statistics_async(ctx_obj["db_path"])

            click.echo(f"\nDatabase Statistics:")
            click.echo(f"Messages: {stats['message_count']:,}")
            click.echo(f"Channels: {stats['channel_count']}")
            click.echo(f"Users: {stats['user_count']}")

        except Exception as e:
            click.echo(f"⚠️  Could not retrieve database statistics: {e}")

    except Exception as e:
        click.echo(f"❌ Error checking sync status: {e}")
        if ctx_obj.get("verbose"):
            import traceback

            traceback.print_exc()


def _list_channels(
    ctx_obj: Dict[str, Any], limit: int, output: Optional[str], output_format: str
):
    """List all available channels for analysis."""
    try:
        from .persistence import get_database_manager

        with get_database_manager(ctx_obj.get("db_path")) as db_manager:
            settings = Settings()
            from pepino.analysis.data_facade import get_analysis_data_facade

            data_facade = get_analysis_data_facade(db_manager, settings.base_filter)
            channel_analyzer = ChannelAnalyzer(data_facade)

            # Get all channels
            channels = channel_analyzer.get_available_channels()

            if not channels:
                click.echo("❌ No channels found in database")
                return

            # Apply limit if specified
            if limit > 0:
                channels = channels[:limit]

            # Prepare data for output
            if output_format in ["json", "csv"]:
                # Rich format with metadata - use data facade instead of direct queries
                channel_data = []
                for channel in channels:
                    try:
                        # Get basic channel statistics using data facade repository
                        stats_data = data_facade.channel_repository.get_channel_message_statistics(
                            channel
                        )

                        if stats_data:
                            channel_data.append(
                                {
                                    "name": channel,
                                    "message_count": stats_data.get(
                                        "total_messages", 0
                                    ),
                                    "unique_users": stats_data.get("unique_users", 0),
                                    "avg_message_length": round(
                                        stats_data.get("avg_message_length", 0.0), 2
                                    ),
                                }
                            )
                        else:
                            channel_data.append(
                                {
                                    "name": channel,
                                    "message_count": 0,
                                    "unique_users": 0,
                                    "avg_message_length": 0.0,
                                }
                            )
                    except:
                        # Fallback for channels without stats
                        channel_data.append(
                            {
                                "name": channel,
                                "message_count": 0,
                                "unique_users": 0,
                                "avg_message_length": 0.0,
                            }
                        )

                data = {"channels": channel_data, "total_count": len(channels)}
            else:
                # Simple text format - just names
                data = {"channels": channels, "total_count": len(channels)}

            # Handle output based on format
            if output_format == "text" and not output:
                # Use template for console text output - need rich data like JSON format
                channel_data = []
                for channel in channels:
                    try:
                        # Get basic channel statistics using data facade repository
                        stats_data = data_facade.channel_repository.get_channel_message_statistics(
                            channel
                        )

                        if stats_data:
                            channel_data.append(
                                {
                                    "channel_name": channel,
                                    "name": channel,
                                    "message_count": stats_data.get(
                                        "total_messages", 0
                                    ),
                                    "unique_users": stats_data.get("unique_users", 0),
                                    "avg_message_length": stats_data.get(
                                        "avg_message_length", 0.0
                                    ),
                                }
                            )
                        else:
                            channel_data.append(
                                {
                                    "channel_name": channel,
                                    "name": channel,
                                    "message_count": 0,
                                    "unique_users": 0,
                                    "avg_message_length": 0.0,
                                }
                            )
                    except:
                        # Fallback for channels without stats
                        channel_data.append(
                            {
                                "channel_name": channel,
                                "name": channel,
                                "message_count": 0,
                                "unique_users": 0,
                                "avg_message_length": 0.0,
                            }
                        )

                template_data = {
                    "items": channel_data,
                    "total_count": len(
                        channel_analyzer.get_available_channels()
                    ),  # Total in DB
                    "showing_count": len(channels),
                    "has_more": limit > 0 and len(channels) >= limit,
                }

                try:
                    from pepino.templates.template_engine import TemplateEngine

                    template_engine = TemplateEngine()
                    result = template_engine.render_template(
                        "outputs/cli/channel_list.txt.j2", **template_data
                    )
                    click.echo(result)
                except Exception as e:
                    # Fallback to simple output and show error if verbose
                    if ctx_obj.get("verbose"):
                        click.echo(f"Template error: {e}", err=True)
                    click.echo(f"📺 Found {len(channels)} channels:")
                    for i, channel in enumerate(channels, 1):
                        click.echo(f"{i:3d}. {channel}")
                    if limit > 0 and len(channels) == limit:
                        click.echo(
                            f"\n*Showing first {limit} channels. Use --limit 0 to see all.*"
                        )
            else:
                # File output or non-text format
                _write_output(data, output, output_format)
                if output_format != "text":
                    click.echo(f"✅ Listed {len(channels)} channels")

    except Exception as e:
        click.echo(f"❌ Error listing channels: {e}", err=True)
        if ctx_obj.get("verbose"):
            raise


def _list_users(
    ctx_obj: Dict[str, Any], limit: int, output: Optional[str], output_format: str
):
    """List all available users for analysis."""
    try:
        from .persistence import get_database_manager

        with get_database_manager(ctx_obj.get("db_path")) as db_manager:
            settings = Settings()
            from pepino.analysis.data_facade import get_analysis_data_facade

            data_facade = get_analysis_data_facade(db_manager, settings.base_filter)
            user_analyzer = UserAnalyzer(data_facade)

            # Get all users
            users = user_analyzer.get_available_users()

            if not users:
                click.echo("❌ No users found in database")
                return

            # Apply limit if specified
            if limit > 0:
                users = users[:limit]

            # Prepare data for output
            if output_format in ["json", "csv"]:
                # Rich format with metadata
                user_data = []
                for user in users:
                    try:
                        # Get basic user statistics using data facade repository
                        stats_data = (
                            data_facade.user_repository.get_user_message_statistics(
                                user
                            )
                        )

                        if stats_data:
                            user_data.append(
                                {
                                    "name": user,
                                    "display_name": user,
                                    "author_id": "",  # We don't have easy access to author_id in this context
                                    "message_count": stats_data.get(
                                        "total_messages", 0
                                    ),
                                    "channels_active": stats_data.get(
                                        "channels_active", 0
                                    ),
                                }
                            )
                        else:
                            user_data.append(
                                {
                                    "name": user,
                                    "display_name": user,
                                    "author_id": "",
                                    "message_count": 0,
                                    "channels_active": 0,
                                }
                            )
                    except:
                        # Fallback for users without stats
                        user_data.append(
                            {
                                "name": user,
                                "display_name": user,
                                "author_id": "",
                                "message_count": 0,
                                "channels_active": 0,
                            }
                        )

                data = {"users": user_data, "total_count": len(users)}
            else:
                # Simple text format - just names
                data = {"users": users, "total_count": len(users)}

            _write_output(data, output, output_format)

            # Console output
            if output_format == "text":
                click.echo(f"👥 Found {len(users)} users:")
                for i, user in enumerate(users, 1):
                    click.echo(f"{i:3d}. {user}")
                if limit > 0 and len(users) == limit:
                    click.echo(
                        f"\n*Showing first {limit} users. Use --limit 0 to see all.*"
                    )
            else:
                click.echo(f"✅ Listed {len(users)} users")

    except Exception as e:
        click.echo(f"❌ Error listing users: {e}", err=True)
        if ctx_obj.get("verbose"):
            raise


def _list_stats(ctx_obj: Dict[str, Any], output: Optional[str], output_format: str):
    """Show database statistics for automation planning."""
    try:
        from .persistence import get_database_manager

        with get_database_manager(ctx_obj.get("db_path")) as db_manager:
            settings = Settings()
            from pepino.analysis.data_facade import get_analysis_data_facade

            data_facade = get_analysis_data_facade(db_manager, settings.base_filter)

            # Get basic counts using data facade repository
            channels = data_facade.channel_repository.get_available_channels()
            user_count = data_facade.message_repository.get_distinct_user_count()
            message_count = data_facade.message_repository.get_total_message_count()

            # Prepare stats data
            stats_data = {
                "database_path": ctx_obj.get("db_path", "data/discord_messages.db"),
                "total_channels": len(channels),
                "total_users": user_count,
                "total_messages": message_count,
                "base_filter": settings.base_filter,
                "channels": channels[:10]
                if output_format == "text"
                else channels,  # Sample for text, all for structured
            }

            _write_output({"stats": stats_data}, output, output_format)

            # Console output
            if output_format == "text":
                click.echo("📊 Database Statistics:")
                click.echo(f"   Database: {stats_data['database_path']}")
                click.echo(f"   Channels: {stats_data['total_channels']:,}")
                click.echo(f"   Users: {stats_data['total_users']:,}")
                click.echo(f"   Messages: {stats_data['total_messages']:,}")
                click.echo(f"   Filter: {stats_data['base_filter'][:100]}...")
                click.echo(
                    f"\n📺 Sample Channels ({min(10, len(channels))} of {len(channels)}):"
                )
                for i, channel in enumerate(channels[:10], 1):
                    click.echo(f"   {i:2d}. {channel}")
            else:
                click.echo(
                    f"✅ Database stats: {message_count:,} messages, {len(channels)} channels, {user_count} users"
                )

    except Exception as e:
        click.echo(f"❌ Error getting database stats: {e}", err=True)
        if ctx_obj.get("verbose"):
            raise


@cli.group(name="performance")
@click.pass_context
def performance(ctx):
    """Performance monitoring and optimization commands."""
    pass


@performance.command(name="metrics")
@click.option("--output", "-o", help="Output file (JSON or CSV)")
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json", "csv"]),
    help="Output format",
)
@click.pass_context
def performance_metrics(ctx, output: Optional[str], output_format: str):
    """Show performance metrics for analysis operations."""
    _performance_metrics(ctx.obj, output, output_format)


@performance.command(name="benchmark")
@click.option(
    "--operations", "-o", multiple=True, help="Specific operations to benchmark"
)
@click.option("--iterations", "-i", default=3, help="Number of iterations (default: 3)")
@click.option("--output", help="Output file (JSON or CSV)")
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json", "csv"]),
    help="Output format",
)
@click.pass_context
def performance_benchmark(
    ctx, operations: tuple, iterations: int, output: Optional[str], output_format: str
):
    """Benchmark analysis operations for performance optimization."""
    _performance_benchmark(ctx.obj, operations, iterations, output, output_format)


@performance.command(name="profile")
@click.option("--operation", "-o", required=True, help="Operation to profile")
@click.option("--args", "-a", help="Arguments for the operation (JSON format)")
@click.option("--output", help="Output file for profiling results")
@click.pass_context
def performance_profile(
    ctx, operation: str, args: Optional[str], output: Optional[str]
):
    """Profile a specific analysis operation for detailed performance analysis."""
    _performance_profile(ctx.obj, operation, args, output)


@cli.group(name="test")
@click.pass_context
def test(ctx):
    """Testing and validation commands."""
    pass


@test.command(name="data")
@click.option("--output", "-o", help="Output file (JSON or CSV)")
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json", "csv"]),
    help="Output format",
)
@click.pass_context
def test_data(ctx, output: Optional[str], output_format: str):
    """Test data integrity and availability."""
    _test_data_integrity(ctx.obj, output, output_format)


@test.command(name="analysis")
@click.option("--operation", "-o", help="Specific analysis operation to test")
@click.option(
    "--sample-size", "-s", default=10, help="Sample size for testing (default: 10)"
)
@click.option("--output", help="Output file (JSON or CSV)")
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json", "csv"]),
    help="Output format",
)
@click.pass_context
def test_analysis(
    ctx,
    operation: Optional[str],
    sample_size: int,
    output: Optional[str],
    output_format: str,
):
    """Test analysis operations with sample data."""
    _test_analysis_operations(ctx.obj, operation, sample_size, output, output_format)


@test.command(name="templates")
@click.option("--template", "-t", help="Specific template to test")
@click.option("--output", help="Output file (JSON or CSV)")
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json", "csv"]),
    help="Output format",
)
@click.pass_context
def test_templates(
    ctx, template: Optional[str], output: Optional[str], output_format: str
):
    """Test template rendering with sample data."""
    _test_template_rendering(ctx.obj, template, output, output_format)


@test.command(name="dependencies")
@click.option("--output", "-o", help="Output file (JSON or CSV)")
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json", "csv"]),
    help="Output format",
)
@click.pass_context
def test_dependencies(ctx, output: Optional[str], output_format: str):
    """Test system dependencies and configuration."""
    _test_system_dependencies(ctx.obj, output, output_format)


def _performance_metrics(
    ctx_obj: Dict[str, Any], output: Optional[str], output_format: str
):
    """Show performance metrics for analysis operations."""
    try:
        from .persistence import get_database_manager

        with get_database_manager(ctx_obj.get("db_path")) as db_manager:
            # Get performance metrics from the database or in-memory tracking
            metrics_data = {
                "database_path": ctx_obj.get("db_path", "data/discord_messages.db"),
                "query_performance": {
                    "avg_query_time": "N/A",
                    "slow_queries": [],
                    "total_queries": 0,
                },
                "memory_usage": {"current_mb": "N/A", "peak_mb": "N/A"},
                "analysis_performance": {
                    "user_analysis_avg_time": "N/A",
                    "channel_analysis_avg_time": "N/A",
                    "topic_analysis_avg_time": "N/A",
                },
                "note": "Performance metrics tracking not yet implemented. This is a placeholder for future functionality.",
            }

            _write_output({"performance_metrics": metrics_data}, output, output_format)

    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        click.echo(f"❌ Error getting performance metrics: {e}", err=True)
        if ctx_obj.get("verbose"):
            raise


def _performance_benchmark(
    ctx_obj: Dict[str, Any],
    operations: tuple,
    iterations: int,
    output: Optional[str],
    output_format: str,
):
    """Benchmark analysis operations for performance optimization."""
    try:
        import time

        from .persistence import get_database_manager

        with get_database_manager(ctx_obj.get("db_path")) as db_manager:
            settings = Settings()
            from pepino.analysis.data_facade import get_analysis_data_facade

            data_facade = get_analysis_data_facade(db_manager, settings.base_filter)

            # Default operations to benchmark
            default_operations = [
                "user_analysis",
                "channel_analysis",
                "topic_analysis",
                "temporal_analysis",
            ]
            ops_to_test = list(operations) if operations else default_operations

            benchmark_results = {
                "benchmark_config": {
                    "iterations": iterations,
                    "operations": ops_to_test,
                    "database_path": ctx_obj.get("db_path", "data/discord_messages.db"),
                },
                "results": {},
            }

            click.echo(
                f"🔍 Benchmarking {len(ops_to_test)} operations with {iterations} iterations each..."
            )

            for operation in ops_to_test:
                click.echo(f"  Testing {operation}...")
                times = []

                for i in range(iterations):
                    start_time = time.time()

                    try:
                        # Simplified benchmark tests
                        if operation == "user_analysis":
                            users = data_facade.user_repository.get_available_users()
                            if users:
                                # Just get basic user stats, don't run full analysis
                                user_count = len(users)
                        elif operation == "channel_analysis":
                            channels = (
                                data_facade.channel_repository.get_available_channels()
                            )
                            if channels:
                                channel_count = len(channels)
                        elif operation == "topic_analysis":
                            # Simple message count for topic analysis benchmark
                            message_count = (
                                data_facade.message_repository.get_total_message_count()
                            )
                        elif operation == "temporal_analysis":
                            # Simple temporal query benchmark
                            message_count = (
                                data_facade.message_repository.get_total_message_count()
                            )

                        end_time = time.time()
                        times.append(end_time - start_time)

                    except Exception as e:
                        click.echo(f"    ❌ Error in iteration {i+1}: {e}")
                        times.append(float("inf"))

                # Calculate statistics
                valid_times = [t for t in times if t != float("inf")]
                if valid_times:
                    avg_time = sum(valid_times) / len(valid_times)
                    min_time = min(valid_times)
                    max_time = max(valid_times)

                    benchmark_results["results"][operation] = {
                        "avg_time_seconds": round(avg_time, 4),
                        "min_time_seconds": round(min_time, 4),
                        "max_time_seconds": round(max_time, 4),
                        "successful_iterations": len(valid_times),
                        "failed_iterations": len(times) - len(valid_times),
                    }

                    click.echo(
                        f"    ✅ {operation}: {avg_time:.4f}s avg ({min_time:.4f}s - {max_time:.4f}s)"
                    )
                else:
                    benchmark_results["results"][operation] = {
                        "error": "All iterations failed",
                        "successful_iterations": 0,
                        "failed_iterations": len(times),
                    }
                    click.echo(f"    ❌ {operation}: All iterations failed")

            _write_output({"benchmark": benchmark_results}, output, output_format)

    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        click.echo(f"❌ Error running benchmark: {e}", err=True)
        if ctx_obj.get("verbose"):
            raise


def _performance_profile(
    ctx_obj: Dict[str, Any], operation: str, args: Optional[str], output: Optional[str]
):
    """Profile a specific analysis operation for detailed performance analysis."""
    try:
        import cProfile
        import io
        import json
        import pstats

        click.echo(f"🔍 Profiling operation: {operation}")

        # Parse arguments if provided
        operation_args = {}
        if args:
            try:
                operation_args = json.loads(args)
            except json.JSONDecodeError:
                click.echo(f"❌ Invalid JSON arguments: {args}", err=True)
                return

        # Create a profiler
        profiler = cProfile.Profile()

        # Profile the operation
        profiler.enable()

        # Placeholder for actual operation profiling
        # This would need to be implemented based on specific operations
        click.echo(f"⚠️  Operation profiling not yet implemented for: {operation}")
        click.echo(f"   Arguments: {operation_args}")

        profiler.disable()

        # Get profiling results
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.sort_stats("cumulative")
        stats.print_stats(20)  # Top 20 functions

        profile_results = {
            "operation": operation,
            "arguments": operation_args,
            "profile_output": stats_stream.getvalue(),
            "note": "Detailed operation profiling not yet implemented",
        }

        if output:
            with open(output, "w") as f:
                json.dump(profile_results, f, indent=2)
            click.echo(f"✅ Profile results written to {output}")
        else:
            click.echo("\n📊 Profile Results:")
            click.echo(profile_results["profile_output"])

    except Exception as e:
        logger.error(f"Error profiling operation: {e}")
        click.echo(f"❌ Error profiling operation: {e}", err=True)
        if ctx_obj.get("verbose"):
            raise


def _test_data_integrity(
    ctx_obj: Dict[str, Any], output: Optional[str], output_format: str
):
    """Test data integrity and availability."""
    try:
        from .persistence import get_database_manager

        with get_database_manager(ctx_obj.get("db_path")) as db_manager:
            settings = Settings()
            from pepino.analysis.data_facade import get_analysis_data_facade

            data_facade = get_analysis_data_facade(db_manager, settings.base_filter)

            test_results = {
                "database_path": ctx_obj.get("db_path", "data/discord_messages.db"),
                "tests": {},
            }

            click.echo("🔍 Testing data integrity...")

            # Test 1: Database connectivity
            try:
                message_count = data_facade.message_repository.get_total_message_count()
                test_results["tests"]["database_connectivity"] = {
                    "status": "PASS",
                    "message": f"Successfully connected to database with {message_count} messages",
                }
                click.echo("  ✅ Database connectivity")
            except Exception as e:
                test_results["tests"]["database_connectivity"] = {
                    "status": "FAIL",
                    "message": f"Database connection failed: {e}",
                }
                click.echo("  ❌ Database connectivity")

            # Test 2: Data availability
            try:
                channels = data_facade.channel_repository.get_available_channels()
                users = data_facade.user_repository.get_available_users()

                if channels and users:
                    test_results["tests"]["data_availability"] = {
                        "status": "PASS",
                        "message": f"Found {len(channels)} channels and {len(users)} users",
                    }
                    click.echo("  ✅ Data availability")
                else:
                    test_results["tests"]["data_availability"] = {
                        "status": "WARN",
                        "message": f"Limited data: {len(channels)} channels, {len(users)} users",
                    }
                    click.echo("  ⚠️  Data availability (limited)")
            except Exception as e:
                test_results["tests"]["data_availability"] = {
                    "status": "FAIL",
                    "message": f"Data availability check failed: {e}",
                }
                click.echo("  ❌ Data availability")

            # Test 3: Data consistency
            try:
                # Basic consistency checks
                total_messages = (
                    data_facade.message_repository.get_total_message_count()
                )
                distinct_users = (
                    data_facade.message_repository.get_distinct_user_count()
                )

                if total_messages > 0 and distinct_users > 0:
                    test_results["tests"]["data_consistency"] = {
                        "status": "PASS",
                        "message": f"Data consistency checks passed: {total_messages} messages from {distinct_users} users",
                    }
                    click.echo("  ✅ Data consistency")
                else:
                    test_results["tests"]["data_consistency"] = {
                        "status": "FAIL",
                        "message": "No valid data found",
                    }
                    click.echo("  ❌ Data consistency")
            except Exception as e:
                test_results["tests"]["data_consistency"] = {
                    "status": "FAIL",
                    "message": f"Data consistency check failed: {e}",
                }
                click.echo("  ❌ Data consistency")

            # Summary
            passed = sum(
                1 for test in test_results["tests"].values() if test["status"] == "PASS"
            )
            total = len(test_results["tests"])

            test_results["summary"] = {
                "total_tests": total,
                "passed": passed,
                "failed": total - passed,
                "overall_status": "PASS" if passed == total else "FAIL",
            }

            click.echo(f"\n📊 Test Summary: {passed}/{total} tests passed")

            _write_output({"data_integrity": test_results}, output, output_format)

    except Exception as e:
        logger.error(f"Error testing data integrity: {e}")
        click.echo(f"❌ Error testing data integrity: {e}", err=True)
        if ctx_obj.get("verbose"):
            raise


def _test_analysis_operations(
    ctx_obj: Dict[str, Any],
    operation: Optional[str],
    sample_size: int,
    output: Optional[str],
    output_format: str,
):
    """Test analysis operations with sample data."""
    try:
        from .persistence import get_database_manager

        with get_database_manager(ctx_obj.get("db_path")) as db_manager:
            settings = Settings()
            from pepino.analysis.data_facade import get_analysis_data_facade

            data_facade = get_analysis_data_facade(db_manager, settings.base_filter)

            test_results = {
                "test_config": {
                    "operation": operation or "all",
                    "sample_size": sample_size,
                    "database_path": ctx_obj.get("db_path", "data/discord_messages.db"),
                },
                "tests": {},
            }

            # Operations to test
            operations_to_test = (
                [operation]
                if operation
                else [
                    "user_analysis",
                    "channel_analysis",
                    "topic_analysis",
                    "temporal_analysis",
                ]
            )

            click.echo(f"🔍 Testing {len(operations_to_test)} analysis operations...")

            for op in operations_to_test:
                click.echo(f"  Testing {op}...")

                try:
                    if op == "user_analysis":
                        users = data_facade.user_repository.get_available_users()
                        if users:
                            # Test with first few users
                            test_users = users[: min(sample_size, len(users))]
                            success_count = 0

                            for user in test_users:
                                try:
                                    # Basic user stats test
                                    user_messages = data_facade.message_repository.get_user_messages(
                                        user, limit=10
                                    )
                                    if user_messages:
                                        success_count += 1
                                except:
                                    pass

                            test_results["tests"][op] = {
                                "status": "PASS" if success_count > 0 else "FAIL",
                                "message": f"Successfully analyzed {success_count}/{len(test_users)} users",
                                "sample_size": len(test_users),
                            }
                        else:
                            test_results["tests"][op] = {
                                "status": "FAIL",
                                "message": "No users available for testing",
                            }

                    elif op == "channel_analysis":
                        channels = (
                            data_facade.channel_repository.get_available_channels()
                        )
                        if channels:
                            test_channels = channels[: min(sample_size, len(channels))]
                            success_count = 0

                            for channel in test_channels:
                                try:
                                    # Basic channel stats test
                                    channel_messages = data_facade.message_repository.get_channel_messages(
                                        channel, limit=10
                                    )
                                    if channel_messages:
                                        success_count += 1
                                except:
                                    pass

                            test_results["tests"][op] = {
                                "status": "PASS" if success_count > 0 else "FAIL",
                                "message": f"Successfully analyzed {success_count}/{len(test_channels)} channels",
                                "sample_size": len(test_channels),
                            }
                        else:
                            test_results["tests"][op] = {
                                "status": "FAIL",
                                "message": "No channels available for testing",
                            }

                    else:
                        # Placeholder for other operations
                        test_results["tests"][op] = {
                            "status": "SKIP",
                            "message": f"Testing for {op} not yet implemented",
                        }

                    if test_results["tests"][op]["status"] == "PASS":
                        click.echo(f"    ✅ {op}")
                    elif test_results["tests"][op]["status"] == "FAIL":
                        click.echo(f"    ❌ {op}")
                    else:
                        click.echo(f"    ⚠️  {op} (skipped)")

                except Exception as e:
                    test_results["tests"][op] = {
                        "status": "FAIL",
                        "message": f"Test failed with error: {e}",
                    }
                    click.echo(f"    ❌ {op}")

            # Summary
            passed = sum(
                1 for test in test_results["tests"].values() if test["status"] == "PASS"
            )
            total = len(
                [
                    test
                    for test in test_results["tests"].values()
                    if test["status"] != "SKIP"
                ]
            )

            test_results["summary"] = {
                "total_tests": total,
                "passed": passed,
                "failed": total - passed,
                "overall_status": "PASS" if passed == total else "FAIL",
            }

            click.echo(f"\n📊 Test Summary: {passed}/{total} analysis operations passed")

            _write_output({"analysis_tests": test_results}, output, output_format)

    except Exception as e:
        logger.error(f"Error testing analysis operations: {e}")
        click.echo(f"❌ Error testing analysis operations: {e}", err=True)
        if ctx_obj.get("verbose"):
            raise


def _test_template_rendering(
    ctx_obj: Dict[str, Any],
    template: Optional[str],
    output: Optional[str],
    output_format: str,
):
    """Test template rendering with sample data."""
    try:
        from pepino.templates.template_engine import TemplateEngine

        from .persistence import get_database_manager

        with get_database_manager(ctx_obj.get("db_path")) as db_manager:
            settings = Settings()
            from pepino.analysis.data_facade import get_analysis_data_facade

            data_facade = get_analysis_data_facade(db_manager, settings.base_filter)

            template_engine = TemplateEngine(data_facade)

            test_results = {
                "test_config": {
                    "template": template or "all",
                    "database_path": ctx_obj.get("db_path", "data/discord_messages.db"),
                },
                "tests": {},
            }

            # Common templates to test
            templates_to_test = (
                [template]
                if template
                else [
                    "outputs/cli/user_analysis.txt.j2",
                    "outputs/cli/channel_analysis.txt.j2",
                    "outputs/cli/topic_analysis.txt.j2",
                    "outputs/discord/user_analysis.md.j2",
                ]
            )

            click.echo(f"🔍 Testing {len(templates_to_test)} templates...")

            for template_name in templates_to_test:
                click.echo(f"  Testing {template_name}...")

                try:
                    # Create sample data for template testing
                    sample_data = {
                        "user_info": {
                            "display_name": "Test User",
                            "author_name": "testuser",
                        },
                        "statistics": {"message_count": 100, "channels_active": 5},
                        "channel_name": "test-channel",
                        "days": 30,
                        "analysis": {"success": True},
                        "topics": [{"topic": "test", "frequency": 10}],
                    }

                    # Test template rendering
                    rendered = template_engine.render_template(
                        template_name, **sample_data
                    )

                    if rendered and len(rendered.strip()) > 0:
                        test_results["tests"][template_name] = {
                            "status": "PASS",
                            "message": f"Template rendered successfully ({len(rendered)} characters)",
                            "sample_output": rendered[:200] + "..."
                            if len(rendered) > 200
                            else rendered,
                        }
                        click.echo(f"    ✅ {template_name}")
                    else:
                        test_results["tests"][template_name] = {
                            "status": "FAIL",
                            "message": "Template rendered but output was empty",
                        }
                        click.echo(f"    ❌ {template_name}")

                except Exception as e:
                    test_results["tests"][template_name] = {
                        "status": "FAIL",
                        "message": f"Template rendering failed: {e}",
                    }
                    click.echo(f"    ❌ {template_name}")

            # Summary
            passed = sum(
                1 for test in test_results["tests"].values() if test["status"] == "PASS"
            )
            total = len(test_results["tests"])

            test_results["summary"] = {
                "total_tests": total,
                "passed": passed,
                "failed": total - passed,
                "overall_status": "PASS" if passed == total else "FAIL",
            }

            click.echo(f"\n📊 Test Summary: {passed}/{total} templates passed")

            _write_output({"template_tests": test_results}, output, output_format)

    except Exception as e:
        logger.error(f"Error testing templates: {e}")
        click.echo(f"❌ Error testing templates: {e}", err=True)
        if ctx_obj.get("verbose"):
            raise


def _test_system_dependencies(
    ctx_obj: Dict[str, Any], output: Optional[str], output_format: str
):
    """Test system dependencies and configuration."""
    try:
        import sqlite3
        import sys

        test_results = {
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "database_path": ctx_obj.get("db_path", "data/discord_messages.db"),
            },
            "tests": {},
        }

        click.echo("🔍 Testing system dependencies...")

        # Test 1: Python version
        try:
            if sys.version_info >= (3, 8):
                test_results["tests"]["python_version"] = {
                    "status": "PASS",
                    "message": f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                }
                click.echo("  ✅ Python version")
            else:
                test_results["tests"]["python_version"] = {
                    "status": "FAIL",
                    "message": f"Python {sys.version_info.major}.{sys.version_info.minor} (requires 3.8+)",
                }
                click.echo("  ❌ Python version")
        except Exception as e:
            test_results["tests"]["python_version"] = {
                "status": "FAIL",
                "message": f"Python version check failed: {e}",
            }
            click.echo("  ❌ Python version")

        # Test 2: SQLite
        try:
            sqlite_version = sqlite3.sqlite_version
            test_results["tests"]["sqlite"] = {
                "status": "PASS",
                "message": f"SQLite {sqlite_version}",
            }
            click.echo("  ✅ SQLite")
        except Exception as e:
            test_results["tests"]["sqlite"] = {
                "status": "FAIL",
                "message": f"SQLite check failed: {e}",
            }
            click.echo("  ❌ SQLite")

        # Test 3: Required packages
        required_packages = ["click", "discord", "jinja2", "pydantic"]

        for package in required_packages:
            try:
                __import__(package)
                test_results["tests"][f"package_{package}"] = {
                    "status": "PASS",
                    "message": f"{package} imported successfully",
                }
                click.echo(f"  ✅ {package}")
            except ImportError as e:
                test_results["tests"][f"package_{package}"] = {
                    "status": "FAIL",
                    "message": f"{package} import failed: {e}",
                }
                click.echo(f"  ❌ {package}")

        # Test 4: Optional packages
        optional_packages = ["spacy", "bertopic", "sentence_transformers"]

        for package in optional_packages:
            try:
                __import__(package)
                test_results["tests"][f"optional_{package}"] = {
                    "status": "PASS",
                    "message": f"{package} available",
                }
                click.echo(f"  ✅ {package} (optional)")
            except ImportError:
                test_results["tests"][f"optional_{package}"] = {
                    "status": "WARN",
                    "message": f"{package} not available (optional)",
                }
                click.echo(f"  ⚠️  {package} (optional, not installed)")

        # Summary
        passed = sum(
            1 for test in test_results["tests"].values() if test["status"] == "PASS"
        )
        warned = sum(
            1 for test in test_results["tests"].values() if test["status"] == "WARN"
        )
        total = len(test_results["tests"])

        test_results["summary"] = {
            "total_tests": total,
            "passed": passed,
            "warned": warned,
            "failed": total - passed - warned,
            "overall_status": "PASS" if passed + warned == total else "FAIL",
        }

        click.echo(
            f"\n📊 Test Summary: {passed}/{total} dependencies passed, {warned} warnings"
        )

        _write_output({"dependency_tests": test_results}, output, output_format)

    except Exception as e:
        logger.error(f"Error testing dependencies: {e}")
        click.echo(f"❌ Error testing dependencies: {e}", err=True)
        if ctx_obj.get("verbose"):
            raise
