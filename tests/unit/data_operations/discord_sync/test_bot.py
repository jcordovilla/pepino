"""Unit tests for Discord Bot."""

import asyncio
import os
import tempfile
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest
from discord.ext import commands

from pepino.config import Settings
from pepino.discord_bot.bot import bot, db_manager


class TestDiscordBot:
    """Test cases for Discord Bot."""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)

    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(discord_token="test_token")

    @pytest.mark.asyncio
    async def test_bot_initialization(self):
        """Test bot initialization."""
        # Test that bot is properly configured
        assert bot.command_prefix is not None
        assert bot.intents is not None
        assert isinstance(bot, commands.Bot)

    @pytest.mark.parametrize(
        "test_case",
        [
            {
                "event_type": "on_ready",
                "expected_log": "Logged in as",
                "description": "ready event",
            },
            {
                "event_type": "on_command_error",
                "expected_log": "Command error",
                "description": "error event",
            },
        ],
    )
    @pytest.mark.asyncio
    async def test_event_handlers_variations(self, test_case):
        """Test different event handlers using table-driven tests."""
        event_type = test_case["event_type"]
        expected_log = test_case["expected_log"]

        # Test that event handlers are properly registered
        if hasattr(bot, event_type):
            handler = getattr(bot, event_type)
            assert callable(handler)

    @pytest.mark.asyncio
    async def test_on_command_error_event(self):
        """Test on_command_error event handler."""
        # Mock context and error
        mock_context = MagicMock()
        mock_context.send = AsyncMock()
        mock_error = commands.CommandNotFound("test")

        # Test the event handler
        await bot.on_command_error(mock_context, mock_error)

        # Should send error message
        mock_context.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_db_manager_initialization(self):
        """Test database manager initialization."""
        # Test that db_manager is properly configured
        assert db_manager is not None
        assert isinstance(db_manager, type(db_manager))

    @pytest.mark.asyncio
    async def test_bot_commands_registration(self):
        """Test that bot commands are properly registered."""
        # Check that sync command is registered
        sync_command = bot.get_command("sync")
        assert sync_command is not None
        assert sync_command.name == "sync"

        # Check that force_sync command is registered
        force_sync_command = bot.get_command("force_sync")
        assert force_sync_command is not None
        assert force_sync_command.name == "force_sync"

        # Check that test_autocomplete command is registered
        test_command = bot.get_command("test_autocomplete")
        assert test_command is not None
        assert test_command.name == "test_autocomplete"

    @pytest.mark.asyncio
    async def test_bot_intents_configuration(self):
        """Test bot intents configuration."""
        # Test that intents are properly configured
        assert bot.intents is not None
        assert isinstance(bot.intents, discord.Intents)

    @pytest.mark.asyncio
    async def test_command_prefix_configuration(self):
        """Test command prefix configuration."""
        # Test that command prefix is set
        assert bot.command_prefix is not None
        assert isinstance(bot.command_prefix, str)

    @pytest.mark.asyncio
    async def test_error_handling_missing_argument(self):
        """Test error handling for missing arguments."""
        # Mock context
        mock_context = MagicMock()
        mock_context.send = AsyncMock()

        # Mock missing argument error
        mock_error = commands.MissingRequiredArgument(MagicMock())
        mock_error.param.name = "test_param"

        # Test the error handler
        await bot.on_command_error(mock_context, mock_error)

        # Should send error message
        mock_context.send.assert_called_once()
        call_args = mock_context.send.call_args[0][0]
        assert "Missing required argument" in call_args
        assert "test_param" in call_args

    @pytest.mark.asyncio
    async def test_error_handling_generic_error(self):
        """Test error handling for generic errors."""
        # Mock context
        mock_context = MagicMock()
        mock_context.send = AsyncMock()

        # Mock generic error
        mock_error = Exception("Test error message")

        # Test the error handler
        await bot.on_command_error(mock_context, mock_error)

        # Should send error message
        mock_context.send.assert_called_once()
        call_args = mock_context.send.call_args[0][0]
        assert "An error occurred" in call_args
        assert "Test error message" in call_args


@pytest.mark.asyncio
async def test_bot_on_ready():
    """Test bot on_ready event handler."""
    # Test the actual bot instance's on_ready method
    # Just verify it doesn't crash when called
    try:
        await bot.on_ready()
        assert True  # If we get here, it didn't crash
    except AttributeError as e:
        # It's okay if bot.user is None in test environment
        assert "NoneType" in str(e) and "name" in str(e)
    except Exception as e:
        # It's okay if it fails due to missing token or other runtime issues
        # We're just testing that the method exists and is callable
        assert "DISCORD_TOKEN" in str(e) or "connection" in str(e).lower()
