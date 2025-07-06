from pepino.data_operations.discord_sync.extractors.emoji_extractor import extract_emojis


def test_extract_emoji_data():
    """Test extracting emoji data from text."""
    text = "Hello 😊 world! <:custom:123456789>"
    result = extract_emojis(text)
    assert "unicode_emojis" in result
    assert "custom_emojis" in result
    assert "😊" in result["unicode_emojis"]
    assert "<:custom:123456789>" in result["custom_emojis"]
