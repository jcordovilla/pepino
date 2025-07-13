"""
Updated Test Fixtures

This file contains updated test fixtures based on template variable analysis.
Generated on: 2025-07-13T13:36:55.536887
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List

from pepino.analysis.service import analysis_service



@pytest.fixture
def sample_pulsecheck_data() -> Dict[str, Any]:
    """Sample Channel Analysis (Pulsecheck) data for testing."""
    return {
    "channel.channel_name": "general",
    "channel.message_count": 450,
    "channel.unique_users": 45,
    "channel_name": "general",
    "contributor.message_count": 89,
    "contributor.name": "Alice Smith",
    "data.active_channels_count": 8,
    "data.participation_distribution.distribution": "normal",
    "data.period.end_date": "2024-07-12",
    "data.period.start_date": "2024-06-12",
    "data.top_commented_messages": [
        {
            "jump_url": "https://discord.com/channels/123/456/789",
            "author": "alice",
            "reply_count": 5
        }
    ],
    "data.total_channels_analyzed": 11,
    "data.user_stats.current_human_users": 45,
    "message.author": "alice",
    "message.jump_url": "https://discord.com/channels/123/456/789",
    "message.reply_count": 5,
    "pattern.count": 25,
    "pattern.day": "Monday",
    "pattern.hour_range": "14-16",
    "term.term": "Python"
}


@pytest.fixture
def sample_top_contributors_data() -> Dict[str, Any]:
    """Sample Top Contributors data for testing."""
    return {
    "channel_name": "general",
    "contributor.name": "Alice Smith",
    "contributors": [
        {
            "name": "Alice Smith",
            "message_count": 89,
            "channel_activity": [
                {
                    "channel_name": "general",
                    "message_count": 45
                }
            ],
            "top_messages": [
                {
                    "jump_url": "https://discord.com/channels/123/456/789",
                    "reply_count": 5
                }
            ]
        }
    ],
    "message.jump_url": "https://discord.com/channels/123/456/789"
}


@pytest.fixture
def sample_database_stats_data() -> Dict[str, Any]:
    """Sample Database Statistics data for testing."""
    return {
    "table_stats": [
        {
            "table_name": "messages",
            "row_count": 15420,
            "size_mb": 15.4,
            "last_insert": "2024-07-12T15:45:00Z"
        }
    ]
}


@pytest.fixture
def sample_detailed_temporal_analysis_data() -> Dict[str, Any]:
    """Sample Detailed Temporal Analysis data for testing."""
    return {
    "channel_name": "general",
    "day": null,
    "days_back": null,
    "hour": null,
    "patterns": {
        "total_messages": 225,
        "avg_messages_per_period": 45.0,
        "message_trend": "increasing"
    },
    "period": {
        "start_date": "2024-06-12",
        "end_date": "2024-07-12"
    },
    "temporal_data": [
        {
            "period": "2024-06-12",
            "message_count": 45,
            "unique_users": 12
        }
    ],
    "trend": null
}


@pytest.fixture
def sample_detailed_topic_analysis_data() -> Dict[str, Any]:
    """Sample Detailed Topic Analysis data for testing."""
    return {
    "capabilities_used": null,
    "capability": null,
    "channel_name": "general",
    "days_back": null,
    "n_topics": [],
    "topics": [
        {
            "topic": "Python Development",
            "frequency": 450,
            "relevance_score": 0.45
        }
    ]
}


@pytest.fixture
def sample_detailed_user_analysis_data() -> Dict[str, Any]:
    """Sample Detailed User Analysis data for testing."""
    return {
    "channel.channel_name": "general",
    "channel_activity": [
        {
            "channel_name": "general",
            "message_count": 45,
            "avg_message_length": 125.0
        }
    ],
    "concepts": null,
    "messages": [
        {
            "content": "Hello world!",
            "author": "alice",
            "timestamp": "2024-07-12T15:45:00Z"
        }
    ],
    "summary": {
        "total_messages": 15420,
        "total_users": 89,
        "total_channels": 15,
        "avg_messages_per_day": 42.3
    },
    "time_patterns": [
        {
            "period": "Morning (06-11)",
            "message_count": 25
        }
    ]
}


@pytest.fixture
def sample_server_overview_analysis_data() -> Dict[str, Any]:
    """Sample Server Overview Analysis data for testing."""
    return {
    "analysis_period_days": null
}


@pytest.fixture
def sample_activity_trends_analysis_data() -> Dict[str, Any]:
    """Sample Activity Trends Analysis data for testing."""
    return {
    "chart_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
    "cross_channel_stats": {
        "total_channels": 15,
        "active_channels": 12
    },
    "time_period": 30
}


@pytest.fixture
def sample_top_channels_data() -> Dict[str, Any]:
    """Sample Top Channels data for testing."""
    return {
    "contributor.message_count": 89
}


@pytest.fixture
def sample_list_channels_data() -> Dict[str, Any]:
    """Sample Channel List data for testing."""
    return {
    "channel": null,
    "channels": [
        {
            "channel_name": "general",
            "message_count": 450,
            "unique_users": 45
        }
    ]
}


@pytest.fixture
def updated_analysis_service():
    """Get updated analysis service for testing."""
    return analysis_service()

def normalize_output(text: str) -> str:
    """Normalize output text for comparison."""
    if not text:
        return ""
    # Remove timestamps and other variable content
    import re
    text = re.sub(r'\d{{4}}-\d{{2}}-\d{{2}}', 'YYYY-MM-DD', text)
    text = re.sub(r'\d{{2}}:\d{{2}}:\d{{2}}', 'HH:MM:SS', text)
    text = re.sub(r'\d{{4}}-\d{{2}}-\d{{2}} at \d{{2}}:\d{{2}}', 'YYYY-MM-DD at HH:MM', text)
    return text.strip()
