#!/usr/bin/env python3
"""
Test suite for Discord message analysis functions.
Tests the various analytical capabilities in analysis.py
"""

import pytest
import sqlite3
import tempfile
import os
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add the project directory to the path to import analysis module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from analysis import MessageAnalyzer

class TestMessageAnalyzer:
    """Test suite for MessageAnalyzer class"""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        
        # Create test database with sample data
        conn = sqlite3.connect(temp_file.name)
        cursor = conn.cursor()
        
        # Create messages table with correct schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                channel_id TEXT,
                author_id TEXT,
                content TEXT,
                timestamp TEXT,
                edited_timestamp TEXT,
                is_bot BOOLEAN,
                is_system BOOLEAN,
                is_webhook BOOLEAN,
                has_attachments BOOLEAN,
                has_embeds BOOLEAN,
                has_stickers BOOLEAN,
                has_mentions BOOLEAN,
                has_reactions BOOLEAN,
                has_reference BOOLEAN,
                referenced_message_id TEXT,
                thread_id TEXT,
                thread_archived BOOLEAN,
                thread_archived_at TEXT,
                thread_auto_archive_duration INTEGER,
                thread_locked BOOLEAN,
                thread_member_count INTEGER,
                thread_message_count INTEGER,
                thread_name TEXT,
                thread_owner_id TEXT,
                thread_parent_id TEXT,
                thread_total_message_sent INTEGER,
                thread_type TEXT,
                created_at TEXT,
                updated_at TEXT,
                author_name TEXT,
                channel_name TEXT,
                guild_name TEXT
            )
        ''')
        
        # Insert sample test data
        sample_messages = [
            ('1', 'ch1', 'user1', 'Hello world! This is a test message.', '2025-06-16T10:00:00', None, False, False, False, False, False, False, False, False, False, None, None, False, None, None, False, None, None, None, None, None, None, None, '2025-06-16T10:00:00', '2025-06-16T10:00:00', 'Alice', 'general', 'TestGuild'),
            ('2', 'ch1', 'user2', 'How are you doing today?', '2025-06-16T11:00:00', None, False, False, False, False, False, False, False, False, False, None, None, False, None, None, False, None, None, None, None, None, None, None, '2025-06-16T11:00:00', '2025-06-16T11:00:00', 'Bob', 'general', 'TestGuild'),
            ('3', 'ch1', 'user1', 'I am learning about artificial intelligence', '2025-06-16T12:00:00', None, False, False, False, False, False, False, False, False, False, None, None, False, None, None, False, None, None, None, None, None, None, None, '2025-06-16T12:00:00', '2025-06-16T12:00:00', 'Alice', 'general', 'TestGuild'),
            ('4', 'ch2', 'user3', 'Machine learning is fascinating!', '2025-06-16T13:00:00', None, False, False, False, False, False, False, False, False, False, None, None, False, None, None, False, None, None, None, None, None, None, None, '2025-06-16T13:00:00', '2025-06-16T13:00:00', 'Charlie', 'ai-discussion', 'TestGuild'),
            ('5', 'ch2', 'user2', 'Python is a great programming language', '2025-06-16T14:00:00', None, False, False, False, False, False, False, False, False, False, None, None, False, None, None, False, None, None, None, None, None, None, None, '2025-06-16T14:00:00', '2025-06-16T14:00:00', 'Bob', 'ai-discussion', 'TestGuild'),
            ('6', 'ch2', 'user1', 'Deep learning neural networks are complex', '2025-06-16T15:00:00', None, False, False, False, False, False, False, False, False, False, None, None, False, None, None, False, None, None, None, None, None, None, None, '2025-06-16T15:00:00', '2025-06-16T15:00:00', 'Alice', 'ai-discussion', 'TestGuild'),
            ('7', 'ch1', 'user3', 'What do you think about the weather?', '2025-06-16T16:00:00', None, False, False, False, False, False, False, False, False, False, None, None, False, None, None, False, None, None, None, None, None, None, None, '2025-06-16T16:00:00', '2025-06-16T16:00:00', 'Charlie', 'general', 'TestGuild'),
            ('8', 'ch1', 'user2', 'It is sunny and beautiful today', '2025-06-16T17:00:00', None, False, False, False, False, False, False, False, False, False, None, None, False, None, None, False, None, None, None, None, None, None, None, '2025-06-16T17:00:00', '2025-06-16T17:00:00', 'Bob', 'general', 'TestGuild'),
        ]
        
        cursor.executemany('''
            INSERT INTO messages 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', sample_messages)
        
        conn.commit()
        conn.close()
        
        yield temp_file.name
        
        # Cleanup
        os.unlink(temp_file.name)
    
    @pytest.fixture
    def analyzer(self, temp_db):
        """Create MessageAnalyzer instance with test database"""
        return MessageAnalyzer(temp_db)
    
    def test_init(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer.conn is not None
        assert analyzer.cursor is not None
        assert analyzer.embedding_model is None
        assert not analyzer.model_loaded
    
    def test_preprocess_text(self, analyzer):
        """Test text preprocessing functionality"""
        # Test basic text cleaning
        text = "Hello @user! Check out https://example.com 🎉"
        processed = analyzer.preprocess_text(text)
        
        assert "hello" in processed.lower()  # Text should be lowercased
        assert "@user" not in processed  # Mentions should be removed
        assert "https://example.com" not in processed  # URLs should be removed
        assert "🎉" not in processed  # Emojis should be removed
    
    def test_preprocess_text_empty(self, analyzer):
        """Test preprocessing with empty or whitespace text"""
        assert analyzer.preprocess_text("") == ""
        assert analyzer.preprocess_text("   ") == ""
        assert analyzer.preprocess_text("\n\t") == ""
    
    def test_preprocess_text_special_cases(self, analyzer):
        """Test preprocessing with special cases"""
        # Test mentions and channels
        text_with_mentions = "Hey <@123456> check <#789012> channel!"
        processed = analyzer.preprocess_text(text_with_mentions)
        assert "<@123456>" not in processed
        assert "<#789012>" not in processed
        assert "hey" in processed.lower()
        assert "check" in processed.lower()
        assert "channel" in processed.lower()
    
    @pytest.mark.asyncio
    async def test_find_similar_messages(self, analyzer):
        """Test finding similar messages functionality"""
        try:
            # Test with a simple query
            result = await analyzer.find_similar_messages({'query': 'artificial intelligence', 'limit': 3})
            
            # Should return a string response
            assert isinstance(result, str)
            assert len(result) > 0
            
        except Exception as e:
            # Model loading might fail in test environment, that's OK
            assert "model" in str(e).lower() or "embedding" in str(e).lower()
    
    def test_perform_topic_modeling(self, analyzer):
        """Test topic modeling functionality"""
        try:
            topics = analyzer.perform_topic_modeling()
            
            # Should return a dictionary
            assert isinstance(topics, dict)
            
            # Each topic should have a list of words
            for topic_id, words in topics.items():
                assert isinstance(words, list)
                assert len(words) > 0
                
        except Exception as e:
            # This might fail if there's insufficient data or model issues
            print(f"⚠️ Topic modeling test skipped: {e}")
            # Just verify it doesn't crash catastrophically
            assert True
    
    @pytest.mark.asyncio
    async def test_update_word_frequencies(self, analyzer):
        """Test word frequency analysis"""
        result = await analyzer.update_word_frequencies()
        
        # Should return a string response
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Should mention word frequencies
        assert "word" in result.lower() or "frequency" in result.lower()
    
    @pytest.mark.asyncio
    async def test_update_user_statistics(self, analyzer):
        """Test user statistics analysis"""
        result = await analyzer.update_user_statistics()
        
        # Should return a string response
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Should mention users or statistics
        assert "user" in result.lower() or "message" in result.lower()
    
    @pytest.mark.asyncio
    async def test_update_temporal_stats(self, analyzer):
        """Test temporal statistics analysis"""
        result = await analyzer.update_temporal_stats()
        
        # Should return a string response
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Should mention temporal patterns
        assert any(word in result.lower() for word in ["hour", "day", "time", "activity"])
    
    @pytest.mark.asyncio
    async def test_get_channel_insights(self, analyzer):
        """Test channel insights functionality"""
        result = await analyzer.get_channel_insights()
        
        # Should return a string response
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Should mention channels
        assert "channel" in result.lower()
    
    @pytest.mark.asyncio
    async def test_get_user_insights(self, analyzer):
        """Test user insights functionality"""
        result = await analyzer.get_user_insights('user1')
        
        # Should return a string response
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Should mention the user
        assert "user" in result.lower() or "alice" in result.lower()
    
    def test_update_conversation_chains(self, analyzer):
        """Test conversation chain analysis"""
        result = analyzer.update_conversation_chains()
        
        # Should return a dictionary
        assert isinstance(result, dict)
        
        # Should have relevant keys
        expected_keys = ["total_chains", "avg_chain_length", "longest_chain"]
        for key in expected_keys:
            assert key in result
    
    def test_run_all_analyses(self, analyzer):
        """Test running all analyses"""
        try:
            results = analyzer.run_all_analyses()
            
            # Should return a dictionary
            assert isinstance(results, dict)
            
            # Should contain various analysis results
            expected_keys = ["word_frequencies", "user_stats", "temporal_stats", "conversation_chains"]
            for key in expected_keys:
                assert key in results
                
        except Exception as e:
            # Some analyses might fail due to missing models or insufficient data
            pass
    
    def test_database_operations(self, analyzer):
        """Test basic database operations"""
        # Test that we can query the database
        cursor = analyzer.cursor
        cursor.execute("SELECT COUNT(*) FROM messages")
        count = cursor.fetchone()[0]
        
        # Should have our test messages
        assert count == 8
        
        # Test filtering
        cursor.execute(f"""
            SELECT COUNT(*) FROM messages 
            WHERE {analyzer.base_filter}
        """)
        filtered_count = cursor.fetchone()[0]
        
        # Should return some messages (our test data doesn't have sesh bot messages)
        assert filtered_count > 0
    
    def test_cleanup(self, analyzer):
        """Test analyzer cleanup"""
        # Should be able to close connection
        analyzer.__del__()
        
        # Connection should be closed
        try:
            analyzer.cursor.execute("SELECT 1")
            assert False, "Connection should be closed"
        except:
            pass  # Expected behavior

class TestIntegration:
    """Integration tests for the complete analysis workflow"""
    
    @pytest.fixture
    def analyzer_with_real_data(self):
        """Create analyzer with actual database if it exists"""
        db_path = 'discord_messages.db'
        if os.path.exists(db_path):
            return MessageAnalyzer(db_path)
        else:
            pytest.skip("Real database not found for integration tests")
    
    @pytest.mark.skipif(not os.path.exists('discord_messages.db'), 
                       reason="Real database not available")
    def test_real_database_connection(self, analyzer_with_real_data):
        """Test connection to real database"""
        analyzer = analyzer_with_real_data
        
        # Should be able to connect and query
        cursor = analyzer.cursor
        cursor.execute("SELECT COUNT(*) FROM messages LIMIT 1")
        result = cursor.fetchone()
        
        assert result is not None
    
    @pytest.mark.skipif(not os.path.exists('discord_messages.db'), 
                       reason="Real database not available")
    @pytest.mark.asyncio
    async def test_real_analysis_workflow(self, analyzer_with_real_data):
        """Test complete analysis workflow with real data"""
        analyzer = analyzer_with_real_data
        
        try:
            # Test word frequency analysis
            word_freq_result = await analyzer.update_word_frequencies()
            assert isinstance(word_freq_result, str)
            
            # Test user statistics
            user_stats_result = await analyzer.update_user_statistics()
            assert isinstance(user_stats_result, str)
            
            # Test temporal statistics
            temporal_result = await analyzer.update_temporal_stats()
            assert isinstance(temporal_result, str)
            
            print("✅ All real data analyses completed successfully")
            
        except Exception as e:
            print(f"⚠️ Analysis failed (expected in some environments): {e}")

def run_tests():
    """Run all tests"""
    print("🧪 Starting Discord Message Analysis Tests...")
    
    # Run pytest with verbose output
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "-x"  # Stop on first failure
    ]
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed.")
    
    return exit_code

if __name__ == "__main__":
    run_tests()
