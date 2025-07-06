"""
Tests for the weekly analysis system.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# from pepino.analysis.weekly_analysis_service import WeeklyAnalysisService
from pepino.analysis.helpers.message_analyzer import MessageAnalyzer
from pepino.analysis.helpers.weekly_user_analyzer import WeeklyUserAnalyzer


class TestMessageAnalyzer:
    """Test the MessageAnalyzer class."""
    
    def test_analyze_weekly_messages_empty(self):
        """Test analysis with no messages."""
        mock_facade = Mock()
        mock_facade.message_repository.get_channel_messages.return_value = []
        
        analyzer = MessageAnalyzer(mock_facade)
        result = analyzer.analyze_weekly_messages("test-channel", 7)
        
        assert result['total_messages'] == 0
        assert result['message_stats']['total_messages'] == 0
        assert len(result['top_terms']) == 0
    
    def test_analyze_weekly_messages_with_data(self):
        """Test analysis with sample message data."""
        mock_facade = Mock()
        
        # Sample message data
        sample_messages = [
            {
                'id': '1',
                'author_id': 'user1',
                'author_name': 'User1',
                'content': 'Hello world',
                'timestamp': datetime.now(),
                'author_is_bot': False,
                'has_reactions': True,
                'referenced_message_id': None
            },
            {
                'id': '2',
                'author_id': 'user2',
                'author_name': 'User2',
                'content': 'Test message',
                'timestamp': datetime.now(),
                'author_is_bot': False,
                'has_reactions': False,
                'referenced_message_id': '1'
            }
        ]
        
        mock_facade.message_repository.get_channel_messages.return_value = sample_messages
        
        analyzer = MessageAnalyzer(mock_facade)
        result = analyzer.analyze_weekly_messages("test-channel", 7)
        
        assert result['total_messages'] == 2
        assert result['message_stats']['human_messages'] == 2
        assert result['message_stats']['bot_messages'] == 0


class TestWeeklyUserAnalyzer:
    """Test the WeeklyUserAnalyzer class."""
    
    def test_analyze_weekly_users_empty(self):
        """Test analysis with no users."""
        mock_facade = Mock()
        mock_facade.message_repository.get_channel_messages.return_value = []
        
        analyzer = WeeklyUserAnalyzer(mock_facade)
        result = analyzer.analyze_weekly_users("test-channel", 7)
        
        assert result['total_participants'] == 0
        assert len(result['top_contributors']) == 0
    
    def test_analyze_weekly_users_with_data(self):
        """Test analysis with sample user data."""
        mock_facade = Mock()
        
        # Sample message data
        sample_messages = [
            {
                'id': '1',
                'author_id': 'user1',
                'author_name': 'User1',
                'content': 'Hello world',
                'timestamp': datetime.now(),
                'author_is_bot': False
            },
            {
                'id': '2',
                'author_id': 'user1',
                'author_name': 'User1',
                'content': 'Another message',
                'timestamp': datetime.now(),
                'author_is_bot': False
            }
        ]
        
        mock_facade.message_repository.get_channel_messages.return_value = sample_messages
        
        analyzer = WeeklyUserAnalyzer(mock_facade)
        result = analyzer.analyze_weekly_users("test-channel", 7)
        
        assert result['total_participants'] == 1
        assert len(result['top_contributors']) == 1
        assert result['top_contributors'][0]['message_count'] == 2


class TestWeeklyAnalysisService:
    """Test the WeeklyAnalysisService class."""
    
    @patch('pepino.analysis.weekly_analysis_service.DatabaseManager')
    @patch('pepino.analysis.weekly_analysis_service.get_analysis_data_facade')
    def test_analyze_channel_weekly(self, mock_get_facade, mock_db_manager):
        """Test the main analysis method."""
        # Setup mocks
        mock_facade = Mock()
        mock_get_facade.return_value = mock_facade
        
        # Mock the analyzers
        mock_message_analyzer = Mock()
        mock_user_analyzer = Mock()
        
        # Sample analysis results
        mock_message_analyzer.analyze_weekly_messages.return_value = {
            'period': {'start_date': '2024-01-01', 'end_date': '2024-01-07'},
            'message_stats': {'total_messages': 10, 'human_messages': 8, 'bot_messages': 2},
            'activity_patterns': {'peak_hours': [], 'peak_days': []},
            'top_terms': [],
            'top_reaction_messages': [],
            'top_commented_messages': [],
            'total_messages': 10
        }
        
        mock_user_analyzer.analyze_weekly_users.return_value = {
            'user_stats': {'current_human_users': 5, 'trend_direction': 'stable'},
            'top_contributors': [],
            'participation_distribution': {'distribution': 'Well'},
            'lost_interest_users': [],
            'total_participants': 5
        }
        
        mock_user_analyzer.get_channel_member_count.return_value = 20
        
        # Create service with mocked analyzers
        service = WeeklyAnalysisService()
        service.message_analyzer = mock_message_analyzer
        service.user_analyzer = mock_user_analyzer
        
        # Test the analysis
        result = service.analyze_channel_weekly("test-channel", 7)
        
        # Verify the result contains expected content
        assert "Weekly Channel Analysis" in result
        assert "test-channel" in result
        assert "2024-01-01" in result
        assert "2024-01-07" in result
        
        # Verify analyzers were called
        mock_message_analyzer.analyze_weekly_messages.assert_called_once_with("test-channel", 7)
        mock_user_analyzer.analyze_weekly_users.assert_called_once_with("test-channel", 7)
        mock_user_analyzer.get_channel_member_count.assert_called_once_with("test-channel")


if __name__ == "__main__":
    pytest.main([__file__]) 