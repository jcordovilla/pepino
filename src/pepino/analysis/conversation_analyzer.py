"""
Conversation analysis service for analyzing message threads and engagement patterns.
"""

import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from pepino.data.config import Settings
from pepino.data.database.manager import DatabaseManager
from pepino.data.repositories.message_repository import MessageRepository

logger = logging.getLogger(__name__)


class ConversationService:
    """Service for analyzing conversation patterns and engagement."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()

    def analyze_conversations(
        self, db_path: str, channel_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze conversation patterns in a channel or across all channels."""
        try:
            # Initialize database and repository using context manager
            with DatabaseManager(db_path) as db_manager:
                message_repo = MessageRepository(db_manager)

                # Get conversation threads using repository
                threads_data = message_repo.get_conversation_threads(
                    channel_id, limit=50
                )

                # Get reply chains using repository
                reply_chains_data = message_repo.get_reply_chains_data(
                    channel_id, limit=50
                )

                # Calculate engagement metrics using repository
                engagement_metrics = message_repo.get_engagement_metrics(channel_id)

            return {
                "threads": [
                    {
                        "thread_id": thread["thread_id"],
                        "thread_name": thread["thread_name"],
                        "message_count": thread["message_count"],
                        "unique_participants": thread["unique_participants"],
                        "start_time": thread["start_time"],
                        "end_time": thread["end_time"],
                        "avg_message_length": round(thread["avg_message_length"], 2),
                        "duration_hours": self._calculate_duration(
                            thread["start_time"], thread["end_time"]
                        ),
                    }
                    for thread in threads_data
                ],
                "reply_chains": reply_chains_data,
                "engagement_metrics": engagement_metrics,
            }

        except Exception as e:
            logger.error(f"Failed to analyze conversations: {e}")
            raise

    def get_reply_chains(
        self, db_path: str, message_id: str, max_depth: int = 5
    ) -> List[Dict[str, Any]]:
        """Get the complete reply chain for a specific message."""
        try:
            # Initialize database and repository using context manager
            with DatabaseManager(db_path) as db_manager:
                message_repo = MessageRepository(db_manager)

                chain = []
                current_message_id = message_id
                depth = 0

                while current_message_id and depth < max_depth:
                    # Get the current message using repository
                    message = message_repo.get_message_by_id(current_message_id)

                    if not message:
                        break

                    chain.append(
                        {
                            "message_id": message["message_id"],
                            "content": message["content"],
                            "author": message["author"],
                            "timestamp": message["timestamp"],
                            "depth": depth,
                            "referenced_message_id": message["referenced_message_id"],
                        }
                    )

                    # Get replies to this message using repository
                    replies = message_repo.get_replies_to_message(
                        current_message_id, limit=1
                    )

                    if replies:
                        current_message_id = replies[0]["message_id"]
                        depth += 1
                    else:
                        break
            return chain

        except Exception as e:
            logger.error(f"Failed to get reply chain: {e}")
            raise

    def calculate_engagement(
        self, db_path: str, channel_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calculate engagement metrics for messages."""
        try:
            # Initialize database and repository using context manager
            with DatabaseManager(db_path) as db_manager:
                message_repo = MessageRepository(db_manager)

                # Get messages with engagement data using repository
                engagement_data = message_repo.get_engagement_data(
                    channel_id, limit=100
                )

                # Calculate engagement scores
                processed_data = []
                for msg in engagement_data:
                    # Simple engagement score based on replies and reactions
                    engagement_score = msg["reply_count"] * 2 + (
                        1 if msg["has_reactions"] else 0
                    )

                    processed_data.append(
                        {
                            "message_id": msg["message_id"],
                            "content": msg["content"][:100] + "..."
                            if len(msg["content"]) > 100
                            else msg["content"],
                            "author": msg["author"],
                            "timestamp": msg["timestamp"],
                            "reply_count": msg["reply_count"],
                            "has_reactions": msg["has_reactions"],
                            "engagement_score": engagement_score,
                        }
                    )

                # Sort by engagement score
                processed_data.sort(key=lambda x: x["engagement_score"], reverse=True)

            return {
                "top_engaged_messages": processed_data[:20],
                "total_analyzed": len(processed_data),
                "avg_engagement_score": sum(
                    x["engagement_score"] for x in processed_data
                )
                / len(processed_data)
                if processed_data
                else 0,
            }

        except Exception as e:
            logger.error(f"Failed to calculate engagement: {e}")
            raise

    def analyze_conversation_flow(
        self, db_path: str, channel_id: str, time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Analyze conversation flow patterns in a channel."""
        try:
            # Initialize database and repository using context manager
            with DatabaseManager(db_path) as db_manager:
                message_repo = MessageRepository(db_manager)

                # Get conversation flow data using repository
                flow_data = message_repo.get_conversation_flow_data(
                    channel_id, time_window_hours
                )

            if not flow_data:
                return {"error": "No messages found in time window"}

            # Convert to format expected by analysis methods
            messages = [
                (
                    msg["author"],
                    msg["timestamp"],
                    msg["message_length"],
                    msg["has_reactions"],
                    msg["has_reference"],
                )
                for msg in flow_data
            ]

            # Analyze conversation flow
            flow_analysis = {
                "total_messages": len(messages),
                "unique_participants": len(set(msg[0] for msg in messages)),
                "avg_message_length": sum(msg[2] for msg in messages) / len(messages),
                "conversation_turns": self._analyze_conversation_turns(messages),
                "response_patterns": self._analyze_response_patterns(messages),
                "engagement_patterns": self._analyze_engagement_patterns(messages),
            }

            return flow_analysis

        except Exception as e:
            logger.error(f"Failed to analyze conversation flow: {e}")
            raise

    def _calculate_duration(self, start_time: str, end_time: str) -> float:
        """Calculate duration between two timestamps in hours."""
        try:
            start = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            end = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            duration = end - start
            return round(duration.total_seconds() / 3600, 2)
        except:
            return 0.0

    def _analyze_conversation_turns(self, messages: List[Tuple]) -> Dict[str, Any]:
        """Analyze conversation turn-taking patterns."""
        if len(messages) < 2:
            return {"avg_turn_duration": 0, "turn_distribution": {}}

        turn_durations = []
        current_speaker = None
        turn_start = None

        for msg in messages:
            author, timestamp = msg[0], msg[1]

            if current_speaker != author:
                if current_speaker and turn_start:
                    # Calculate turn duration
                    try:
                        start_time = datetime.fromisoformat(
                            turn_start.replace("Z", "+00:00")
                        )
                        end_time = datetime.fromisoformat(
                            timestamp.replace("Z", "+00:00")
                        )
                        duration = (end_time - start_time).total_seconds()
                        turn_durations.append(duration)
                    except:
                        pass

                current_speaker = author
                turn_start = timestamp

        # Calculate turn statistics
        avg_turn_duration = (
            sum(turn_durations) / len(turn_durations) if turn_durations else 0
        )

        # Analyze turn distribution
        speaker_counts = Counter(msg[0] for msg in messages)
        turn_distribution = dict(speaker_counts.most_common(10))

        return {
            "avg_turn_duration_seconds": round(avg_turn_duration, 2),
            "turn_distribution": turn_distribution,
            "total_turns": len(turn_durations),
        }

    def _analyze_response_patterns(self, messages: List[Tuple]) -> Dict[str, Any]:
        """Analyze response patterns and timing."""
        response_times = []
        response_counts = defaultdict(int)

        for i, msg in enumerate(messages):
            if msg[4]:  # has_reference
                response_counts[msg[0]] += 1

                # Find the referenced message
                for j in range(
                    i - 1, max(0, i - 10), -1
                ):  # Look back up to 10 messages
                    if messages[j][0] != msg[0]:  # Different speaker
                        try:
                            ref_time = datetime.fromisoformat(
                                messages[j][1].replace("Z", "+00:00")
                            )
                            resp_time = datetime.fromisoformat(
                                msg[1].replace("Z", "+00:00")
                            )
                            response_time = (resp_time - ref_time).total_seconds()
                            if 0 < response_time < 3600:  # Between 0 and 1 hour
                                response_times.append(response_time)
                        except:
                            pass
                        break

        avg_response_time = (
            sum(response_times) / len(response_times) if response_times else 0
        )

        return {
            "avg_response_time_seconds": round(avg_response_time, 2),
            "total_responses": len(response_times),
            "response_counts_by_user": dict(response_counts),
            "most_responsive_users": sorted(
                response_counts.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }

    def _analyze_engagement_patterns(self, messages: List[Tuple]) -> Dict[str, Any]:
        """Analyze engagement patterns in conversations."""
        reactions_count = sum(1 for msg in messages if msg[3])  # has_reactions
        references_count = sum(1 for msg in messages if msg[4])  # has_reference

        # Engagement by user
        user_engagement = defaultdict(
            lambda: {"messages": 0, "reactions": 0, "references": 0}
        )

        for msg in messages:
            author, has_reactions, has_reference = msg[0], msg[3], msg[4]
            user_engagement[author]["messages"] += 1
            if has_reactions:
                user_engagement[author]["reactions"] += 1
            if has_reference:
                user_engagement[author]["references"] += 1

        # Calculate engagement scores
        for user, data in user_engagement.items():
            data["engagement_score"] = (
                data["reactions"] * 2 + data["references"] * 3
            ) / data["messages"]

        return {
            "total_reactions": reactions_count,
            "total_references": references_count,
            "reaction_rate": reactions_count / len(messages) if messages else 0,
            "reference_rate": references_count / len(messages) if messages else 0,
            "user_engagement": dict(user_engagement),
        }
