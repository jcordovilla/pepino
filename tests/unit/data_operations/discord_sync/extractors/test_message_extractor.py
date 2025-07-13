from unittest.mock import MagicMock

from pepino.data_operations.discord_sync.extractors.message_extractor import MessageExtractor


def test_extract_message_data():
    mock_message = MagicMock()
    mock_message.id = 123
    mock_message.content = "hello world"
    mock_message.author.id = 456
    mock_message.author.name = "user"
    mock_message.created_at.isoformat.return_value = "2024-01-01T00:00:00Z"
    data = MessageExtractor.extract_message_data(mock_message)
    assert data["id"] == "123"
    assert data["content"] == "hello world"
    assert data["author"]["id"] == "456"
