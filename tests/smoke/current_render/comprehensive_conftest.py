"""
Comprehensive Test Fixtures

This file contains comprehensive test fixtures based on actual template variable analysis.
Generated on: 2025-07-13T13:36:55.536887

These fixtures are designed to provide all the data that templates expect,
ensuring that template rendering tests have complete coverage.
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List

from pepino.analysis.service import analysis_service


@pytest.fixture
def sample_pulsecheck_data() -> Dict[str, Any]:
    """Sample Channel Analysis (Pulsecheck) data for testing."""
    return {
        "channel_name": "general",
        "total_members": 150,
        "data": {
            "period": {
                "start_date": "2024-06-12",
                "end_date": "2024-07-12"
            },
            "user_stats": {
                "current_human_users": 45,
                "trend_direction": "increasing",
                "human_user_change_pct": 12.5
            },
            "participation_distribution": {
                "distribution": "normal",
                "top_contributors_percentage": 75.5
            },
            "message_stats": {
                "total_messages": 1250,
                "human_messages": 1100,
                "bot_messages": 150,
                "human_percentage": 88.0,
                "bot_percentage": 12.0,
                "activity_rate": 178.6,
                "avg_message_length": 85.5
            },
            "activity_patterns": {
                "peak_hours": [
                    {"hour_range": "14-16", "day": "Monday", "count": 25},
                    {"hour_range": "15-17", "day": "Tuesday", "count": 22},
                    {"hour_range": "13-15", "day": "Wednesday", "count": 20}
                ]
            },
            "top_terms": [
                {"term": "Python"},
                {"term": "Discord"},
                {"term": "API"},
                {"term": "Development"},
                {"term": "Community"}
            ],
            "top_contributors": [
                {"name": "Alice Smith", "message_count": 89},
                {"name": "Bob Johnson", "message_count": 67},
                {"name": "Charlie Brown", "message_count": 45},
                {"name": "Diana Prince", "message_count": 34},
                {"name": "Eve Wilson", "message_count": 28}
            ],
            "top_commented_messages": [
                {"jump_url": "https://discord.com/channels/123/456/789", "author": "alice", "reply_count": 5},
                {"jump_url": "https://discord.com/channels/123/456/790", "author": "bob", "reply_count": 3},
                {"jump_url": "https://discord.com/channels/123/456/791", "author": "charlie", "reply_count": 2}
            ],
            "lost_interest_users": [
                {"name": "inactive_user1"},
                {"name": "inactive_user2"}
            ],
            "active_channels_count": 8,
            "inactive_channels_count": 3,
            "active_channels_percentage": 72.7,
            "active_channels_qualifier": "good",
            "active_channels_min_threshold": 5,
            "total_channels_analyzed": 11,
            "most_active_channels": [
                {"channel_name": "general", "message_count": 450, "unique_users": 45},
                {"channel_name": "random", "message_count": 320, "unique_users": 38},
                {"channel_name": "help", "message_count": 280, "unique_users": 32}
            ],
            "least_active_channels": [
                {"channel_name": "archive", "message_count": 2, "unique_users": 1},
                {"channel_name": "announcements", "message_count": 1, "unique_users": 1}
            ]
        }
    }


@pytest.fixture
def sample_top_contributors_data() -> Dict[str, Any]:
    """Sample Top Contributors data for testing."""
    return {
        "channel_name": "general",
        "contributors": [
            {
                "name": "Alice Smith",
                "message_count": 89,
                "channel_activity": [
                    {"channel_name": "general", "message_count": 45},
                    {"channel_name": "random", "message_count": 23},
                    {"channel_name": "help", "message_count": 21}
                ],
                "top_messages": [
                    {"jump_url": "https://discord.com/channels/123/456/789", "reply_count": 5},
                    {"jump_url": "https://discord.com/channels/123/456/790", "reply_count": 3}
                ]
            },
            {
                "name": "Bob Johnson",
                "message_count": 67,
                "channel_activity": [
                    {"channel_name": "general", "message_count": 35},
                    {"channel_name": "help", "message_count": 20},
                    {"channel_name": "random", "message_count": 12}
                ],
                "top_messages": [
                    {"jump_url": "https://discord.com/channels/123/456/791", "reply_count": 3},
                    {"jump_url": "https://discord.com/channels/123/456/792", "reply_count": 2}
                ]
            },
            {
                "name": "Charlie Brown",
                "message_count": 45,
                "channel_activity": [
                    {"channel_name": "random", "message_count": 30},
                    {"channel_name": "general", "message_count": 15}
                ],
                "top_messages": [
                    {"jump_url": "https://discord.com/channels/123/456/793", "reply_count": 2}
                ]
            }
        ],
        "period": {
            "start_date": "2024-06-12",
            "end_date": "2024-07-12"
        }
    }


@pytest.fixture
def sample_database_stats_data() -> Dict[str, Any]:
    """Sample Database Statistics data for testing."""
    return {
        "database_info": {
            "file_path": "/path/to/database.db",
            "size_mb": 25.5,
            "last_modified": "2024-07-12T15:45:00Z"
        },
        "table_stats": [
            {
                "table_name": "messages",
                "row_count": 15420,
                "size_mb": 15.4,
                "last_insert": "2024-07-12T15:45:00Z"
            },
            {
                "table_name": "users",
                "row_count": 89,
                "size_mb": 0.1,
                "last_insert": "2024-07-12T14:30:00Z"
            },
            {
                "table_name": "channels",
                "row_count": 15,
                "size_mb": 0.05,
                "last_insert": "2024-07-12T13:15:00Z"
            }
        ],
        "summary": {
            "total_messages": 15420,
            "total_users": 89,
            "total_channels": 15,
            "date_range": {
                "start": "2023-06-15T08:00:00Z",
                "end": "2024-07-12T23:59:59Z"
            },
            "most_active_channel": {
                "name": "general",
                "message_count": 4500
            },
            "most_active_user": {
                "name": "alice",
                "message_count": 1200
            },
            "avg_messages_per_day": 42.3,
            "avg_messages_per_user": 173.3,
            "avg_messages_per_channel": 1028.0
        }
    }


@pytest.fixture
def sample_detailed_temporal_analysis_data() -> Dict[str, Any]:
    """Sample Detailed Temporal Analysis data for testing."""
    return {
        "temporal_data": [
            {"period": "2024-06-12", "message_count": 45, "unique_users": 12},
            {"period": "2024-06-13", "message_count": 52, "unique_users": 15},
            {"period": "2024-06-14", "message_count": 38, "unique_users": 10},
            {"period": "2024-06-15", "message_count": 61, "unique_users": 18},
            {"period": "2024-06-16", "message_count": 29, "unique_users": 8}
        ],
        "patterns": {
            "total_messages": 225,
            "avg_messages_per_period": 45.0,
            "max_messages_in_period": 61,
            "min_messages_in_period": 29,
            "most_active_period": "2024-06-15",
            "message_trend": "increasing",
            "trend_percentage": 12.5,
            "trend_timeframe": "over analysis period",
            "peak_user_count": 18,
            "total_periods": 5
        },
        "capabilities_used": ["temporal_analysis", "pattern_detection", "trend_analysis"],
        "granularity": "daily",
        "channel_name": "general",
        "days_back": 30
    }


@pytest.fixture
def sample_detailed_topic_analysis_data() -> Dict[str, Any]:
    """Sample Detailed Topic Analysis data for testing."""
    return {
        "topics": [
            {"topic": "Python Development", "frequency": 450, "relevance_score": 0.45},
            {"topic": "Discord Integration", "frequency": 320, "relevance_score": 0.32},
            {"topic": "Community Discussion", "frequency": 280, "relevance_score": 0.28},
            {"topic": "API Documentation", "frequency": 180, "relevance_score": 0.18},
            {"topic": "Bug Reports", "frequency": 120, "relevance_score": 0.12}
        ],
        "message_count": 1000,
        "capabilities_used": ["bertopic_modeling", "sentence_transformers", "semantic_clustering"],
        "n_topics": 10,
        "days_back": 30,
        "channel_name": "general",
        "summary": {
            "total_topics": 5,
            "avg_relevance_score": 0.27,
            "most_frequent_topic": "Python Development"
        }
    }


@pytest.fixture
def sample_detailed_user_analysis_data() -> Dict[str, Any]:
    """Sample Detailed User Analysis data for testing."""
    return {
        "user_info": {
            "author_id": "123456789",
            "display_name": "Alice Smith"
        },
        "statistics": {
            "author_id": "123456789",
            "author_name": "alice",
            "message_count": 89,
            "channels_active": 5,
            "avg_message_length": 120.5,
            "first_message_date": "2024-01-15T10:30:00Z",
            "last_message_date": "2024-07-12T15:45:00Z",
            "total_messages": 89,
            "active_days": 45,
            "unique_channels": 5,
            "avg_messages_per_day": 2.0
        },
        "channel_activity": [
            {
                "channel_name": "general",
                "message_count": 45
            },
            {
                "channel_name": "random",
                "message_count": 23
            }
        ],
        "time_patterns": {
            "hourly_activity": [
                {"hour": "09", "message_count": 5},
                {"hour": "10", "message_count": 8},
                {"hour": "11", "message_count": 12},
                {"hour": "12", "message_count": 15},
                {"hour": "13", "message_count": 18},
                {"hour": "14", "message_count": 22},
                {"hour": "15", "message_count": 20},
                {"hour": "16", "message_count": 16}
            ],
            "daily_activity": [
                {"day": "Monday", "message_count": 15},
                {"day": "Tuesday", "message_count": 18},
                {"day": "Wednesday", "message_count": 12},
                {"day": "Thursday", "message_count": 20},
                {"day": "Friday", "message_count": 14},
                {"day": "Saturday", "message_count": 8},
                {"day": "Sunday", "message_count": 6}
            ],
            "peak_activity_hour": "14"
        },
        "semantic_analysis": {
            "key_entities": ["Python", "Discord", "API"],
            "technology_terms": ["bot", "webhook", "database"],
            "key_concepts": ["development", "automation", "community"]
        },
        "top_topics": [
            {"topic": "Python Development", "frequency": 15, "relevance_score": 0.17},
            {"topic": "Discord Integration", "frequency": 12, "relevance_score": 0.13},
            {"topic": "Community Discussion", "frequency": 8, "relevance_score": 0.09}
        ],
        "concepts": ["Python", "Discord", "API", "Development"],
        "activity_patterns": ["Most active in afternoon", "Prefers technical discussions"],
        "recommendations": ["Consider mentoring others", "Share more code examples"],
        "summary": "Alice is a highly active user with 89 messages across 5 channels, averaging 120.5 characters per message. Most active in the afternoon hours and prefers technical discussions about Python and Discord development.",
        "messages": [
            {
                "content": "Hello world! This is a test message about Python development.",
                "author": "alice",
                "timestamp": "2024-07-12T15:45:00Z"
            },
            {
                "content": "I'm working on a Discord bot integration.",
                "author": "alice",
                "timestamp": "2024-07-12T14:30:00Z"
            }
        ]
    }


@pytest.fixture
def sample_server_overview_analysis_data() -> Dict[str, Any]:
    """Sample Server Overview Analysis data for testing."""
    return {
        "server_stats": 15420,
        "engagement_metrics": {
            "active_users_percentage": 85.2,
            "messages_per_user": 173.3,
            "channels_per_user": 2.1
        },
        "date_range": {
            "start": "2023-06-15T08:00:00Z",
            "end": "2024-07-12T23:59:59Z"
        },
        "most_active_channel": {
            "name": "general",
            "message_count": 4500
        },
        "most_active_user": {
            "name": "alice",
            "message_count": 1200
        },
        "top_channels": [
            {"name": "general", "message_count": 4500},
            {"name": "random", "message_count": 3200},
            {"name": "help", "message_count": 2800}
        ],
        "top_users": [
            {"username": "alice", "message_count": 1200},
            {"username": "bob", "message_count": 890},
            {"username": "charlie", "message_count": 670}
        ],
        "temporal_data": [
            {"date": "2024-06-12", "messages": 45},
            {"date": "2024-06-13", "messages": 52},
            {"date": "2024-06-14", "messages": 38}
        ],
        "daily_activity_data": [
            {"date": "2024-06-12", "messages": 45},
            {"date": "2024-06-13", "messages": 52},
            {"date": "2024-06-14", "messages": 38}
        ],
        "topic_analysis": {
            "top_topics": ["Python Development", "Discord Integration", "Community Discussion"]
        },
        "time_period": 30,
        "analysis_period_days": 30,
        "messages_per_day": 42.3,
        "messages_per_user": 173.3,
        "messages_per_channel": 1028.0,
        "server_stats": 15420,
        "total_messages": 15420,
        "total_users": 89,
        "total_channels": 15,
        "active_users": 76,
        "top_contributors": [
            {"username": "alice", "message_count": 1200},
            {"username": "bob", "message_count": 890},
            {"username": "charlie", "message_count": 670}
        ],
        "activity_trends": [
            {"date": "2024-06-12", "message_count": 45},
            {"date": "2024-06-13", "message_count": 52},
            {"date": "2024-06-14", "message_count": 38}
        ]
    }


@pytest.fixture
def sample_activity_trends_analysis_data() -> Dict[str, Any]:
    """Sample Activity Trends Analysis data for testing."""
    return {
        "patterns": {
            "total_messages": 225,
            "avg_messages_per_period": 45.0,
            "max_messages_in_period": 61,
            "min_messages_in_period": 29,
            "most_active_period": "2024-06-15",
            "message_trend": "increasing",
            "trend_percentage": 12.5,
            "trend_timeframe": "over analysis period",
            "peak_user_count": 18,
            "total_periods": 5
        },
        "chart_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
        "cross_channel_stats": {
            "total_channels": 15,
            "active_channels": 12,
            "most_active_channel": "general"
        },
        "time_period": 30
    }


@pytest.fixture
def sample_top_channels_data() -> Dict[str, Any]:
    """Sample Top Channels data for testing."""
    result = {
        "data": {
            "channels": [
                {
                    "name": "general",
                    "message_count": 450,
                    "active_members": 45,
                    "total_members": 150,
                    "participation_rate": 30.0,
                    "trend_direction": "increasing",
                    "trend_percentage": 15.2,
                    "top_contributors": [
                        {"name": "Alice Smith", "message_count": 89},
                        {"name": "Bob Johnson", "message_count": 67}
                    ]
                },
                {
                    "name": "random",
                    "message_count": 320,
                    "active_members": 38,
                    "total_members": 120,
                    "participation_rate": 31.7,
                    "trend_direction": "stable",
                    "trend_percentage": 0.0,
                    "top_contributors": [
                        {"name": "Charlie Brown", "message_count": 45},
                        {"name": "Diana Prince", "message_count": 34}
                    ]
                },
                {
                    "name": "help",
                    "message_count": 280,
                    "active_members": 32,
                    "total_members": 100,
                    "participation_rate": 32.0,
                    "trend_direction": "decreasing",
                    "trend_percentage": -5.1,
                    "top_contributors": [
                        {"name": "Eve Wilson", "message_count": 28},
                        {"name": "Frank Miller", "message_count": 22}
                    ]
                }
            ],
            "summary": {
                "total_channels": 15,
                "active_channels": 12,
                "avg_messages_per_channel": 1050,
                "avg_participation_rate": 31.2
            },
            "period_start": "2024-06-12",
            "period_end": "2024-07-12",
            "total_active_channels": 12,
            "total_channels": 15,
            "most_engaged_team": {
                "name": "general",
                "participation_rate": 30.0
            },
            "peak_activity_times": "2-4 PM weekdays",
            "top_contributors": [
                {"display_name": "Alice Smith", "message_count": 89},
                {"display_name": "Bob Johnson", "message_count": 67}
            ],
            "total_messages": 1050,
            "total_active_users": 45,
            "avg_participation_rate": 31.2,
            "increasing_channels": ["general", "random"],
            "decreasing_channels": ["help"],
            "top_channels": [
                {"name": "general", "message_count": 450},
                {"name": "random", "message_count": 320},
                {"name": "help", "message_count": 280}
            ]
        }
    }
    # Add top-level variables for template access
    result["period_start"] = "2024-06-12"
    result["period_end"] = "2024-07-12"
    return result


@pytest.fixture
def sample_list_channels_data() -> Dict[str, Any]:
    """Sample Channel List data for testing."""
    return {
        "channels": [
            {
                "channel_name": "general",
                "message_count": 450,
                "unique_users": 45
            },
            {
                "channel_name": "random",
                "message_count": 320,
                "unique_users": 38
            },
            {
                "channel_name": "help",
                "message_count": 280,
                "unique_users": 32
            },
            {
                "channel_name": "announcements",
                "message_count": 150,
                "unique_users": 25
            },
            {
                "channel_name": "archive",
                "message_count": 50,
                "unique_users": 8
            }
        ]
    }


@pytest.fixture
def comprehensive_analysis_service():
    """Get comprehensive analysis service for testing."""
    return analysis_service()


def normalize_output(text: str) -> str:
    """Normalize output text for comparison."""
    if not text:
        return ""
    # Remove timestamps and other variable content
    import re
    text = re.sub(r'\d{4}-\d{2}-\d{2}', 'YYYY-MM-DD', text)
    text = re.sub(r'\d{2}:\d{2}:\d{2}', 'HH:MM:SS', text)
    text = re.sub(r'\d{4}-\d{2}-\d{2} at \d{2}:\d{2}', 'YYYY-MM-DD at HH:MM', text)
    return text.strip()


def verify_template_data_coverage():
    """
    Verify that all template variables are covered by the fixtures.
    This function can be used to ensure comprehensive coverage.
    """
    # This would be implemented to check all templates against fixtures
    # For now, it's a placeholder for future verification
    pass 