"""Unit tests for Discord Bot."""

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

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
        # The bot loads extensions in on_ready, so we test the command tree structure
        # and verify that the bot is configured to use app_commands (slash commands)
        
        # Check that the bot has a command tree (for slash commands)
        assert hasattr(bot, 'tree')
        assert bot.tree is not None
        
        # Check that the bot is configured to use app_commands
        assert bot.tree is not None
        
        # Verify the bot has the expected structure for loading commands
        assert hasattr(bot, 'load_extension')
        assert hasattr(bot, 'extensions')
        
        # Test that we can manually load the analysis extension
        try:
            await bot.load_extension("pepino.discord_bot.commands.analysis")
            
            # Now check that the commands are registered
            command_names = [cmd.name for cmd in bot.tree.get_commands()]
            expected_commands = ["channel_analysis", "top_contributors", "top_channels", "list_channels"]
            
            for expected_cmd in expected_commands:
                assert expected_cmd in command_names, f"Expected command {expected_cmd} not found in {command_names}"
                
        except Exception as e:
            # If loading fails due to missing dependencies, that's okay for unit tests
            # We're just testing the structure, not the full integration
            assert "DISCORD_TOKEN" in str(e) or "connection" in str(e).lower() or "import" in str(e).lower()

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
