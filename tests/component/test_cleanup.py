import os
import pytest

def test_artifact_cleanup():
    # These files should not exist after cleanup
    for fname in ["test_database.db", "test_data.json", "analysis_validation_results.json", "discord_messages.db"]:
        assert not os.path.exists(os.path.join(os.path.dirname(__file__), "..", fname)), f"{fname} should be cleaned up"
    # Logs directory should be empty or not exist
    logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    if os.path.exists(logs_dir):
        assert not os.listdir(logs_dir), "logs directory should be empty after cleanup"
    # __pycache__ should not exist
    pycache_dir = os.path.join(os.path.dirname(__file__), "..", "__pycache__")
    assert not os.path.exists(pycache_dir), "__pycache__ should be cleaned up" 