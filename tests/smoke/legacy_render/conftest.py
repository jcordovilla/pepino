"""
Fixtures for legacy template rendering smoke tests.
"""

import pytest
import json
import os
from pathlib import Path
from typing import Dict, Any, List

from src.pepino.analysis.legacy.service import legacy_analysis_service
from src.pepino.data.database.manager import DatabaseManager
from src.pepino.data.repositories.user_repository import UserRepository
from src.pepino.data.repositories.channel_repository import ChannelRepository
from src.pepino.data.repositories.message_repository import MessageRepository


@pytest.fixture(scope="session")
def test_data_dir():
    """Get the test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def golden_outputs_dir():
    """Get the golden outputs directory."""
    return Path(__file__).parent / "golden_outputs"


@pytest.fixture(scope="session")
def legacy_service():
    """Get a legacy analysis service context manager."""
    from src.pepino.analysis.legacy.service import legacy_analysis_service
    return legacy_analysis_service


@pytest.fixture(scope="session")
def sample_user_data():
    """Sample user data for testing."""
    return {
        "username": "test_user",
        "days_back": 30,
        "user_stats": {
            "total_messages": 150,
            "avg_message_length": 45.2,
            "active_days": 12,
            "unique_channels": 5
        },
        "channel_activity": [
            {"channel_name": "general", "message_count": 80, "percentage": 53.3},
            {"channel_name": "random", "message_count": 40, "percentage": 26.7},
            {"channel_name": "help", "message_count": 30, "percentage": 20.0}
        ],
        "hourly_patterns": [
            {"hour": 9, "message_count": 15},
            {"hour": 10, "message_count": 25},
            {"hour": 14, "message_count": 30},
            {"hour": 15, "message_count": 35},
            {"hour": 16, "message_count": 25},
            {"hour": 17, "message_count": 20}
        ],
        "daily_patterns": [
            {"day": "Monday", "message_count": 20},
            {"day": "Tuesday", "message_count": 25},
            {"day": "Wednesday", "message_count": 30},
            {"day": "Thursday", "message_count": 35},
            {"day": "Friday", "message_count": 40}
        ],
        "content_sample": [
            "Hello everyone! How's it going?",
            "I have a question about the project.",
            "Thanks for the help with that issue.",
            "Looking forward to the meeting tomorrow.",
            "Great work on the latest update!"
        ],
        "sentiment_analysis": {
            "positive": 60,
            "neutral": 30,
            "negative": 10
        },
        "key_concepts": [
            "project", "meeting", "help", "update", "work"
        ]
    }


@pytest.fixture(scope="session")
def sample_database_data():
    """Sample database statistics data for testing."""
    return {
        "database_info": {
            "file_path": "/test/path/discord_messages.db",
            "size": "15.2 MB",
            "last_modified": "2025-07-12"
        },
        "table_stats": [
            {
                "name": "messages",
                "rows": 5000,
                "size": "12.1 MB",
                "last_insert": "2025-07-12"
            },
            {
                "name": "users",
                "rows": 150,
                "size": "2.1 MB",
                "last_insert": "2025-07-12"
            },
            {
                "name": "channels",
                "rows": 25,
                "size": "0.5 MB",
                "last_insert": "2025-07-12"
            }
        ],
        "summary": {
            "total_messages": 5000,
            "total_users": 150,
            "total_channels": 25,
            "date_range": "2025-01-01 to 2025-07-12",
            "most_active_channel": {"name": "general", "message_count": 1200},
            "most_active_user": {"name": "test_user", "message_count": 500}
        },
        "health_status": {
            "database_exists": True,
            "messages_available": True,
            "user_data_available": True,
            "channel_data_available": True
        }
    }


@pytest.fixture(scope="session")
def sample_topic_data():
    """Sample topic analysis data for testing."""
    return {
        "analysis_overview": {
            "scope": "All Channels",
            "total_messages": 1000,
            "topics_extracted": 8,
            "capabilities_used": ["bertopic_modeling", "spacy_nlp"]
        },
        "topics": [
            {
                "topic": "Technology & Development",
                "frequency": 250,
                "relevance_score": 0.25
            },
            {
                "topic": "Community & Discussion",
                "frequency": 200,
                "relevance_score": 0.20
            },
            {
                "topic": "Business & Work",
                "frequency": 150,
                "relevance_score": 0.15
            },
            {
                "topic": "Learning & Education",
                "frequency": 100,
                "relevance_score": 0.10
            }
        ],
        "topic_distribution": {
            "Technology & Development": 25.0,
            "Community & Discussion": 20.0,
            "Business & Work": 15.0,
            "Learning & Education": 10.0
        },
        "most_discussed_topic": "Technology & Development",
        "least_discussed_topic": "Learning & Education",
        "technical_details": {
            "analysis_method": "BERTopic with spaCy preprocessing",
            "advanced_features": ["bertopic_modeling", "spacy_nlp"]
        }
    }


@pytest.fixture(scope="session")
def sample_temporal_data():
    """Sample temporal analysis data for testing."""
    return {
        "analysis_overview": {
            "scope": "All Channels",
            "granularity": "Daily",
            "capabilities_used": ["temporal_analysis"]
        },
        "daily_activity": [
            {"date": "2025-07-01", "messages": 45, "users": 12},
            {"date": "2025-07-02", "messages": 52, "users": 15},
            {"date": "2025-07-03", "messages": 38, "users": 10},
            {"date": "2025-07-04", "messages": 61, "users": 18},
            {"date": "2025-07-05", "messages": 48, "users": 14},
            {"date": "2025-07-06", "messages": 55, "users": 16},
            {"date": "2025-07-07", "messages": 42, "users": 13}
        ],
        "hourly_patterns": [
            {"hour": 9, "message_count": 25},
            {"hour": 10, "message_count": 35},
            {"hour": 11, "message_count": 40},
            {"hour": 12, "message_count": 30},
            {"hour": 13, "message_count": 45},
            {"hour": 14, "message_count": 50},
            {"hour": 15, "message_count": 55},
            {"hour": 16, "message_count": 45},
            {"hour": 17, "message_count": 35},
            {"hour": 18, "message_count": 25}
        ],
        "weekly_patterns": [
            {"day": "Monday", "message_count": 180, "users": 45},
            {"day": "Tuesday", "message_count": 200, "users": 50},
            {"day": "Wednesday", "message_count": 220, "users": 55},
            {"day": "Thursday", "message_count": 190, "users": 48},
            {"day": "Friday", "message_count": 160, "users": 40},
            {"day": "Saturday", "message_count": 120, "users": 30},
            {"day": "Sunday", "message_count": 100, "users": 25}
        ],
        "statistical_summary": {
            "total_messages": 1170,
            "average_per_period": 167.1,
            "peak_activity": 61,
            "lowest_activity": 38
        }
    }


def save_golden_output(output_dir: Path, filename: str, content: str):
    """Save content as a golden output file."""
    output_file = output_dir / filename
    output_file.write_text(content, encoding='utf-8')


def load_golden_output(output_dir: Path, filename: str) -> str:
    """Load content from a golden output file."""
    output_file = output_dir / filename
    if not output_file.exists():
        return ""
    return output_file.read_text(encoding='utf-8')


def normalize_output(content: str) -> str:
    """Normalize output content for comparison (remove timestamps, etc.)."""
    import re
    from datetime import datetime
    
    # Remove timestamps like "Generated: 2025-07-12 22:30:37"
    content = re.sub(r'Generated: \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', 'Generated: TIMESTAMP', content)
    
    # Remove file paths that might change
    content = re.sub(r'File Path: .*', 'File Path: /test/path/discord_messages.db', content)
    
    # Remove any other time-sensitive information
    content = re.sub(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', 'TIMESTAMP', content)
    
    return content.strip() 