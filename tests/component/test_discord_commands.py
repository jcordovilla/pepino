import pytest
from unittest.mock import MagicMock, AsyncMock
from pepino.discord_bot.commands.analysis import AnalysisCommands
import discord

@pytest.mark.asyncio
async def test_discord_bot_pulsecheck_command(analysis_commands, mock_discord_interaction):
    await analysis_commands.analyze_channels(
        mock_discord_interaction,
        channel_name="🏘old-general-chat",
        days_back=7
    )
    mock_discord_interaction.response.defer.assert_called_once()
    mock_discord_interaction.followup.send.assert_called()
    sent_content = mock_discord_interaction.followup.send.call_args[0][0]
    assert "🏘old-general-chat" in sent_content
    assert "messages" in sent_content.lower()

@pytest.mark.asyncio
async def test_discord_bot_top_contributors_command(analysis_commands, mock_discord_interaction):
    await analysis_commands.top_users(
        mock_discord_interaction,
        limit=5,
        days=30
    )
    mock_discord_interaction.response.defer.assert_called_once()
    mock_discord_interaction.followup.send.assert_called()
    sent_content = mock_discord_interaction.followup.send.call_args[0][0]
    assert "contributors" in sent_content.lower()
    assert "oscarsan.chez" in sent_content

@pytest.mark.asyncio
async def test_discord_bot_top_channels_command(analysis_commands, mock_discord_interaction):
    await analysis_commands.top_channels(
        mock_discord_interaction,
        limit=5,
        days_back=7
    )
    mock_discord_interaction.response.defer.assert_called_once()
    mock_discord_interaction.followup.send.assert_called()
    sent_content = mock_discord_interaction.followup.send.call_args[0][0]
    assert "channels" in sent_content.lower()
    assert "🏘old-general-chat" in sent_content

@pytest.mark.asyncio
async def test_discord_bot_list_channels_command(analysis_commands, mock_discord_interaction):
    await analysis_commands.list_channels(mock_discord_interaction)
    mock_discord_interaction.response.defer.assert_called_once()
    mock_discord_interaction.followup.send.assert_called()
    sent_content = mock_discord_interaction.followup.send.call_args[0][0]
    assert "channels" in sent_content.lower()
    expected_channels = ['🏘old-general-chat', '🦾agent-ops', 'jose-test', '🏛netarch-general', '🛝discord-pg']
    for channel in expected_channels:
        assert channel in sent_content 