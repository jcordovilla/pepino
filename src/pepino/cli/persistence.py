"""
CLI persistence operations - orchestrates repositories for CLI commands.
Follows the same pattern as discord.data.persistence for consistency.
"""

import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional

from pepino.data.config import Settings
from pepino.data.database.manager import DatabaseManager
from pepino.data.repositories import (
    ChannelRepository,
    DatabaseRepository,
    MessageRepository,
)

logger = logging.getLogger(__name__)


@contextmanager
def get_database_manager(
    db_path: Optional[str] = None,
):
    """Context manager for database connections in CLI operations."""
    settings = Settings()

    # Use provided path or fall back to settings default
    # db_path is already validated by Click callback if provided
    final_db_path = db_path or settings.db_path

    db_manager = DatabaseManager(final_db_path)

    try:
        # DatabaseManager initializes automatically when first connection is made
        yield db_manager
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise RuntimeError(f"❌ Failed to initialize database: {e}")
    finally:
        db_manager.close_connections()


def analyze_user(
    user: Optional[str], limit: int, db_path: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze user activity by orchestrating analyzer calls."""
    from ..analysis.user_analyzer import UserAnalyzer

    settings = Settings()

    try:
        with get_database_manager(db_path) as db_manager:
            from ..analysis.data_facade import get_analysis_data_facade
            data_facade = get_analysis_data_facade(db_manager, settings.base_filter)
            user_analyzer = UserAnalyzer(data_facade)

            if user:
                # Returns a validated Pydantic model
                result = user_analyzer.analyze(
                    username=user, include_patterns=True
                )
                if result:
                    return {"user_analysis": result.model_dump()}
                else:
                    return {"user_analysis": None}
            else:
                # Legacy behavior for multiple users
                users = user_analyzer.get_top_users(limit)
                return {"top_users": users}
    except RuntimeError:
        # Re-raise database configuration errors as-is (they have user-friendly messages)
        raise
    except Exception as e:
        logger.error(f"User analysis failed: {e}")
        raise RuntimeError(f"❌ Analysis failed: {e}")

# Keep async version for backwards compatibility
async def analyze_user_async(
    user: Optional[str], limit: int, db_path: Optional[str] = None
) -> Dict[str, Any]:
    """Async wrapper for analyze_user."""
    return analyze_user(user, limit, db_path)


def analyze_channel(
    channel: Optional[str], limit: int, db_path: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze channel activity by orchestrating analyzer calls."""
    from ..analysis.channel_analyzer import ChannelAnalyzer

    settings = Settings()

    try:
        with get_database_manager(db_path) as db_manager:
            from ..analysis.data_facade import get_analysis_data_facade
            data_facade = get_analysis_data_facade(db_manager, settings.base_filter)
            channel_analyzer = ChannelAnalyzer(data_facade)

            if channel:
                # Returns a validated Pydantic model
                result = channel_analyzer.analyze(
                    channel_name=channel, include_patterns=True
                )
                if result:
                    # Get total human members for percentage calculations
                    total_human_members = 0
                    try:
                        total_human_members = data_facade.channel_repository.get_channel_human_member_count(channel)
                    except Exception as e:
                        logger.warning(f"Could not get total human members for channel {channel}: {e}")
                    
                    # Calculate participation summary
                    participation_summary = None
                    if result.top_users and len(result.top_users) >= 5:
                        top_5_messages = sum(user.message_count for user in result.top_users[:5])
                        total_messages = result.statistics.total_messages
                        if total_messages > 0:
                            concentration = (top_5_messages / total_messages) * 100
                            if concentration > 70:
                                participation_summary = f"5 top contributors posted {concentration:.0f}% of all messages (highly concentrated)"
                            elif concentration > 50:
                                participation_summary = f"5 top contributors posted {concentration:.0f}% of all messages"
                            else:
                                participation_summary = f"5 top contributors posted {concentration:.0f}% of all messages (well distributed)"
                    
                    # Calculate lost interest summary
                    lost_interest_summary = None
                    lost_interest_users = []
                    try:
                        # Get users who posted before but not in last 30 days
                        all_users = data_facade.channel_repository.get_channel_user_activity(channel, days=None, limit=100)
                        recent_users = data_facade.channel_repository.get_channel_user_activity(channel, days=30, limit=100)
                        
                        recent_usernames = {user['author_name'] for user in recent_users}
                        inactive_users = [user for user in all_users if user['author_name'] not in recent_usernames]
                        
                        if inactive_users:
                            # Get days since last message for inactive users
                            from datetime import datetime, timezone
                            now = datetime.now(timezone.utc)
                            
                            for user in inactive_users[:5]:  # Limit to top 5 inactive users
                                if user['last_message']:
                                    try:
                                        last_msg_date = datetime.fromisoformat(user['last_message'].replace('Z', '+00:00'))
                                        days_inactive = (now - last_msg_date).days
                                        lost_interest_users.append({
                                            'display_name': user.get('author_display_name'),
                                            'author_name': user['author_name'],
                                            'days_inactive': days_inactive,
                                            'message_count': user['message_count']
                                        })
                                    except:
                                        continue
                            
                            if lost_interest_users:
                                # Sort by days inactive and message count
                                lost_interest_users.sort(key=lambda x: (x['days_inactive'], x['message_count']), reverse=True)
                                inactive_count = len(lost_interest_users)
                                if inactive_count > 0:
                                    lost_interest_summary = f"{inactive_count} former contributors inactive for 30+ days"
                    except Exception as e:
                        logger.warning(f"Could not calculate lost interest for channel {channel}: {e}")
                    
                    # Calculate engagement summary
                    engagement_summary = None
                    if result.engagement_metrics:
                        reaction_rate = result.engagement_metrics.reaction_rate * 100
                        if reaction_rate > 80:
                            engagement_summary = f"High ({reaction_rate:.0f}% reaction rate)"
                        elif reaction_rate > 50:
                            engagement_summary = f"Moderate ({reaction_rate:.0f}% reaction rate)"
                        else:
                            engagement_summary = f"Low ({reaction_rate:.0f}% reaction rate)"
                    
                    # Calculate trend summary
                    trend_summary = None
                    try:
                        # Compare current period vs previous period
                        current_messages = data_facade.message_repository.get_channel_messages(channel, days_back=7)
                        previous_messages = data_facade.message_repository.get_channel_messages(channel, days_back=14, limit=len(current_messages) * 2)
                        
                        if current_messages and previous_messages:
                            current_count = len([m for m in current_messages if not m.get('author_is_bot', False)])
                            previous_count = len([m for m in previous_messages if not m.get('author_is_bot', False)])
                            
                            if previous_count > 0:
                                change_percent = ((current_count - previous_count) / previous_count) * 100
                                if change_percent > 20:
                                    trend_summary = f"Activity increasing (+{change_percent:.0f}% vs. last week)"
                                elif change_percent < -20:
                                    trend_summary = f"Activity decreasing ({change_percent:.0f}% vs. last week)"
                                else:
                                    trend_summary = f"Activity stable ({change_percent:+.0f}% vs. last week)"
                    except Exception as e:
                        logger.warning(f"Could not calculate trend for channel {channel}: {e}")
                    
                    # Calculate bot activity summary
                    bot_activity_summary = None
                    if result.statistics.bot_messages > 0:
                        bot_percentage = (result.statistics.bot_messages / result.statistics.total_messages) * 100
                        if result.statistics.bot_messages > result.statistics.human_messages:
                            bot_activity_summary = f"Bots posted {bot_percentage:.0f}% of messages (more than humans)"
                        elif bot_percentage > 10:  # Only show if bots > 10%
                            bot_activity_summary = f"Bots posted {bot_percentage:.0f}% of messages (less than humans)"
                    
                    # Calculate response time (placeholder for now)
                    response_time = None
                    
                    # Calculate recent activity summary
                    recent_activity_summary = None
                    try:
                        recent_messages = data_facade.message_repository.get_channel_messages(channel, days_back=7)
                        if recent_messages:
                            human_recent = len([m for m in recent_messages if not m.get('author_is_bot', False)])
                            previous_messages = data_facade.message_repository.get_channel_messages(channel, days_back=14, limit=len(recent_messages) * 2)
                            if previous_messages:
                                human_previous = len([m for m in previous_messages if not m.get('author_is_bot', False)])
                                if human_previous > 0:
                                    change_percent = ((human_recent - human_previous) / human_previous) * 100
                                    if change_percent > 0:
                                        recent_activity_summary = f"{human_recent} messages in last 7 days (up {change_percent:.0f}% from previous week)"
                                    else:
                                        recent_activity_summary = f"{human_recent} messages in last 7 days (down {abs(change_percent):.0f}% from previous week)"
                                else:
                                    recent_activity_summary = f"{human_recent} messages in last 7 days"
                    except Exception as e:
                        logger.warning(f"Could not calculate recent activity for channel {channel}: {e}")
                    
                    # Convert to dict and add all summary fields
                    result_dict = result.model_dump()
                    result_dict['total_human_members'] = total_human_members
                    result_dict['participation_summary'] = participation_summary
                    result_dict['lost_interest_summary'] = lost_interest_summary
                    result_dict['lost_interest_users'] = lost_interest_users
                    result_dict['engagement_summary'] = engagement_summary
                    result_dict['trend_summary'] = trend_summary
                    result_dict['bot_activity_summary'] = bot_activity_summary
                    result_dict['response_time'] = response_time
                    result_dict['recent_activity_summary'] = recent_activity_summary
                    result_dict['channel_health'] = True
                    
                    return {"channel_analysis": result_dict}
                else:
                    return {"channel_analysis": None}
            else:
                # Legacy behavior for multiple channels
                channels = channel_analyzer.get_top_channels(limit)
                return {"top_channels": channels}
    except RuntimeError:
        # Re-raise database configuration errors as-is
        raise
    except Exception as e:
        logger.error(f"Channel analysis failed: {e}")
        raise RuntimeError(f"❌ Analysis failed: {e}")

# Keep async version for backwards compatibility
async def analyze_channel_async(
    channel: Optional[str], limit: int, db_path: Optional[str] = None
) -> Dict[str, Any]:
    """Async wrapper for analyze_channel."""
    return analyze_channel(channel, limit, db_path)


def analyze_topics(
    channel: Optional[str], n_topics: int, days_back: int, db_path: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze topics by orchestrating analyzer calls."""
    from ..analysis.topic_analyzer import TopicAnalyzer
    from ..analysis.data_facade import get_analysis_data_facade

    try:
        with get_analysis_data_facade(db_path) as facade:
            topic_analyzer = TopicAnalyzer(facade)

            result = topic_analyzer.analyze(
                channel_name=channel, top_n=n_topics, days_back=days_back
            )

            # Convert Pydantic model to dict for output
            result_dict = (
                result.model_dump() if hasattr(result, "model_dump") else result
            )
            return {"topic_analysis": result_dict}
    except RuntimeError:
        # Re-raise database configuration errors as-is
        raise
    except Exception as e:
        logger.error(f"Topic analysis failed: {e}")
        raise RuntimeError(f"❌ Analysis failed: {e}")

# Keep async version for backwards compatibility
async def analyze_topics_async(
    channel: Optional[str], n_topics: int, days_back: int, db_path: Optional[str] = None
) -> Dict[str, Any]:
    """Async wrapper for analyze_topics."""
    return analyze_topics(channel, n_topics, days_back, db_path)


def analyze_temporal(
    channel: Optional[str],
    days_back: int,
    granularity: str,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze temporal patterns by orchestrating analyzer calls."""
    from ..analysis.temporal_analyzer import TemporalAnalyzer
    from ..analysis.data_facade import get_analysis_data_facade

    try:
        with get_analysis_data_facade(db_path) as facade:
            temporal_analyzer = TemporalAnalyzer(facade)

            result = temporal_analyzer.analyze(
                channel_name=channel, days_back=days_back, granularity=granularity
            )

            # Convert Pydantic model to dict for output
            result_dict = (
                result.model_dump() if hasattr(result, "model_dump") else result
            )
            return {"temporal_analysis": result_dict}
    except RuntimeError:
        # Re-raise database configuration errors as-is
        raise
    except Exception as e:
        logger.error(f"Temporal analysis failed: {e}")
        raise RuntimeError(f"❌ Analysis failed: {e}")

# Keep async version for backwards compatibility
async def analyze_temporal_async(
    channel: Optional[str],
    days_back: int,
    granularity: str,
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Async wrapper for analyze_temporal."""
    return analyze_temporal(channel, days_back, granularity, db_path)


def analyze_sentiment_async(
    user_name: Optional[str] = None,
    channel_name: Optional[str] = None,
    db_path: Optional[str] = None,
    days_back: int = 30,
) -> Dict[str, Any]:
    """Analyze sentiment patterns."""
    try:
        db_path = db_path or get_database_path()
        
        # Initialize NLP service
        nlp_service = NLPService()
        nlp_service.initialize()
        
        # Get database manager and create data facade
        with get_database_manager(db_path) as db_manager:
            settings = Settings()
            from ..analysis.data_facade import get_analysis_data_facade
            data_facade = get_analysis_data_facade(db_manager, settings.base_filter)
            
            # Get messages for analysis
            if user_name:
                # Get messages from specific user
                messages = data_facade.message_repository.get_user_messages(user_name, days_back=days_back)
            elif channel_name:
                # Get messages from specific channel
                messages = data_facade.message_repository.get_channel_messages(channel_name, days_back=days_back)
            else:
                # Get recent messages
                messages = data_facade.message_repository.get_recent_messages(limit=1000, days_back=days_back)
            
            if not messages:
                return {"error": "No messages found for sentiment analysis"}
            
            # Analyze sentiment for each message
            sentiment_results = []
            for msg in messages:
                if msg.get("content"):
                    sentiment = nlp_service.analyze_sentiment(msg["content"])
                    sentiment_results.append({
                        "message_id": msg["message_id"],
                        "content": msg["content"][:100],
                        "author": msg["author_name"],
                        "channel": msg["channel_name"],
                        "timestamp": msg["timestamp"],
                        **sentiment
                    })
            
            # Calculate overall statistics
            positive_count = sum(1 for s in sentiment_results if s["sentiment"] == "positive")
            negative_count = sum(1 for s in sentiment_results if s["sentiment"] == "negative")
            neutral_count = sum(1 for s in sentiment_results if s["sentiment"] == "neutral")
            
            avg_score = sum(s["score"] for s in sentiment_results) / len(sentiment_results) if sentiment_results else 0
            
            return {
                "total_messages": len(sentiment_results),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "neutral_count": neutral_count,
                "positive_percentage": (positive_count / len(sentiment_results)) * 100 if sentiment_results else 0,
                "negative_percentage": (negative_count / len(sentiment_results)) * 100 if sentiment_results else 0,
                "neutral_percentage": (neutral_count / len(sentiment_results)) * 100 if sentiment_results else 0,
                "average_score": round(avg_score, 3),
                "sample_messages": sentiment_results[:10],
                "user_filter": user_name,
                "channel_filter": channel_name,
                "days_back": days_back,
            }
            
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        return {"error": str(e)}


def detect_duplicates_async(
    channel_name: Optional[str] = None,
    db_path: Optional[str] = None,
    similarity_threshold: float = 0.9,
) -> Dict[str, Any]:
    """Detect duplicate messages using similarity analysis."""
    try:
        db_path = db_path or get_database_path()
        
        # Initialize similarity service
        similarity_service = SimilarityService()
        
        # Get channel ID if channel name provided using data facade
        channel_id = None
        if channel_name:
            with get_database_manager(db_path) as db_manager:
                settings = Settings()
                from ..analysis.data_facade import get_analysis_data_facade
                data_facade = get_analysis_data_facade(db_manager, settings.base_filter)
                
                channel = data_facade.channel_repository.get_channel_by_name(channel_name)
                if channel:
                    channel_id = channel.channel_id
                else:
                    return {"error": f"Channel '{channel_name}' not found"}
        
        # Detect duplicates
        duplicates = similarity_service.detect_duplicates(
            db_path, channel_id, similarity_threshold
        )
        
        return {
            "total_duplicate_pairs": len(duplicates),
            "similarity_threshold": similarity_threshold,
            "channel_filter": channel_name,
            "duplicate_pairs": duplicates[:20],  # Limit output
        }
        
    except Exception as e:
        logger.error(f"Error detecting duplicates: {e}")
        return {"error": str(e)}


def get_database_statistics(
    db_path: Optional[str] = None,
) -> Dict[str, int]:
    """Get aggregated database statistics by orchestrating repository calls."""
    try:
        with get_database_manager(db_path) as db_manager:
            settings = Settings()
            from ..analysis.data_facade import get_analysis_data_facade
            data_facade = get_analysis_data_facade(db_manager, settings.base_filter)

            # Orchestrate multiple repository calls to aggregate statistics
            message_count = data_facade.message_repository.get_total_message_count()
            channel_count = data_facade.channel_repository.get_distinct_channel_count()
            user_count = data_facade.message_repository.get_distinct_user_count()

            return {
                "message_count": message_count,
                "channel_count": channel_count,
                "user_count": user_count,
            }
    except RuntimeError:
        # Re-raise database configuration errors as-is
        raise
    except Exception as e:
        logger.error(f"Database statistics failed: {e}")
        raise RuntimeError(f"❌ Failed to get database statistics: {e}")

# Keep async version for backwards compatibility  
async def get_database_statistics_async(
    db_path: Optional[str] = None,
) -> Dict[str, int]:
    """Async wrapper for get_database_statistics."""
    return get_database_statistics(db_path)


def export_table_data_async(
    table_name: str,
    output_file: str,
    format: str = "json",
    db_path: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Export table data to file."""
    try:
        with get_database_manager(db_path) as db_manager:
            # Simple export - get all data from specified table
            if table_name == "messages":
                query = "SELECT * FROM messages"
                if limit:
                    query += f" LIMIT {limit}"
                rows = db_manager.execute_query(query, fetch_all=True)
            elif table_name == "channels":
                query = "SELECT DISTINCT channel_name FROM messages"
                rows = db_manager.execute_query(query, fetch_all=True)
            elif table_name == "users":
                query = "SELECT DISTINCT author_name FROM messages"
                rows = db_manager.execute_query(query, fetch_all=True)
            else:
                return {"error": f"Unknown table: {table_name}"}
            
            return {
                "table": table_name,
                "format": format,
                "rows": rows or [],
                "total_rows": len(rows) if rows else 0
            }
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return {"error": str(e)}


def export_all_tables_async(db_path: Optional[str] = None) -> Dict[str, Any]:
    """Export all tables to files."""
    try:
        tables = {}
        
        # Export messages table
        messages_data = export_table_data_async("messages", "", "json", db_path, 100)
        tables["messages"] = messages_data
        
        # Export channels
        channels_data = export_table_data_async("channels", "", "json", db_path)
        tables["channels"] = channels_data
        
        # Export users
        users_data = export_table_data_async("users", "", "json", db_path)
        tables["users"] = users_data
        
        return {
            "tables": tables,
            "exported_tables": ["messages", "channels", "users"]
        }
    except Exception as e:
        logger.error(f"Export all tables failed: {e}")
        return {"error": str(e)}
