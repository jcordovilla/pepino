"""Unit tests for DiscordClient."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest

from pepino.data_operations.discord_sync.discord_client import DiscordClient


class TestDiscordClient:
    """Test cases for DiscordClient."""

    @pytest.fixture
    def discord_client(self, data_store):
        """Create DiscordClient instance."""
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.members = True
        return DiscordClient(data_store=data_store, intents=intents)

    @pytest.mark.asyncio
    async def test_initialize_success(self, discord_client):
        """Test successful client initialization."""
        # DiscordClient doesn't have an initialize method - it's ready after construction
        assert discord_client.data_store is not None
        assert discord_client.rate_limit_delay == 0.1
        assert discord_client.max_retries == 3

    @pytest.mark.asyncio
    async def test_client_attributes(self, discord_client):
        """Test client attributes are set correctly."""
        assert hasattr(discord_client, "data_store")
        assert hasattr(discord_client, "new_data")
        assert hasattr(discord_client, "rate_limit_delay")
        assert hasattr(discord_client, "max_retries")
        assert hasattr(discord_client, "sync_logger")

    @pytest.mark.asyncio
    async def test_sync_with_retry_success(self, discord_client):
        """Test successful sync with retry."""
        mock_channel = MagicMock()
        mock_message = MagicMock()
        mock_message.id = "msg1"

        # Create a proper async iterator for history
        async def async_iter():
            yield mock_message

        mock_channel.history = MagicMock(return_value=async_iter())

        with patch(
            "pepino.discord_bot.extractors.MessageExtractor.extract_message_data"
        ) as mock_extract:
            mock_extract.return_value = {"id": "msg1", "content": "test"}

            result = await discord_client.sync_with_retry(mock_channel)

            assert len(result) == 1
            assert result[0]["id"] == "msg1"
            mock_extract.assert_called_once_with(mock_message)

    @pytest.mark.asyncio
    async def test_sync_with_retry_forbidden_error(self, discord_client):
        """Test sync with retry handles forbidden error."""
        mock_channel = MagicMock()
        mock_channel.history = MagicMock(
            side_effect=discord.Forbidden(MagicMock(), "Forbidden")
        )

        with pytest.raises(discord.Forbidden, match="Forbidden"):
            await discord_client.sync_with_retry(mock_channel)

    @pytest.mark.asyncio
    async def test_sync_with_retry_http_error_retry(self, discord_client):
        """Test sync with retry handles HTTP errors with retries."""
        mock_channel = MagicMock()
        mock_message = MagicMock()
        mock_message.id = "msg1"

        # Create async iterator for successful case
        async def async_iter():
            yield mock_message

        mock_channel.history = MagicMock(
            side_effect=[
                discord.HTTPException(
                    MagicMock(status=429), "HTTP 429"
                ),  # Rate limit error
                async_iter(),  # Success on retry
            ]
        )

        with patch("asyncio.sleep") as mock_sleep:
            with patch(
                "pepino.discord_bot.extractors.MessageExtractor.extract_message_data"
            ) as mock_extract:
                mock_extract.return_value = {"id": "msg1", "content": "test"}

                result = await discord_client.sync_with_retry(mock_channel)

                assert mock_sleep.called  # Should have slept between retries
                assert len(result) == 1

    @pytest.mark.asyncio
    async def test_sync_channel_members_success(self, discord_client):
        """Test successful channel members sync."""
        mock_channel = MagicMock()
        mock_guild = MagicMock()
        mock_member = MagicMock()

        # Set up mock member
        mock_member.id = "user1"
        mock_member.name = "TestUser"
        mock_member.display_name = "TestUser"
        mock_member.roles = []
        mock_member.joined_at = None
        mock_member.bot = False

        mock_guild.members = [mock_member]
        mock_channel.guild = mock_guild

        # Mock permissions - return a MagicMock with boolean attributes
        mock_perms = MagicMock()
        mock_perms.read_messages = True
        mock_perms.send_messages = True
        mock_perms.manage_messages = False
        mock_perms.read_message_history = True
        mock_perms.add_reactions = True
        mock_perms.attach_files = False
        mock_perms.embed_links = True
        mock_perms.mention_everyone = False
        mock_channel.permissions_for = MagicMock(return_value=mock_perms)

        result = await discord_client._sync_channel_members(mock_channel)

        assert len(result) == 1
        assert result[0]["user_id"] == "user1"
        assert result[0]["user_name"] == "TestUser"

    @pytest.mark.asyncio
    async def test_sync_channel_members_no_access(self, discord_client):
        """Test channel members sync with no access."""
        mock_channel = MagicMock()
        mock_guild = MagicMock()
        mock_member = MagicMock()

        mock_guild.members = [mock_member]
        mock_channel.guild = mock_guild

        # Mock permissions - no read access
        mock_perms = MagicMock()
        mock_perms.read_messages = False
        mock_channel.permissions_for = MagicMock(return_value=mock_perms)

        result = await discord_client._sync_channel_members(mock_channel)

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_sync_channel_members_with_roles(self, discord_client):
        """Test channel members sync with roles."""
        mock_channel = MagicMock()
        mock_guild = MagicMock()
        mock_member = MagicMock()
        mock_role = MagicMock()

        # Set up mock role
        mock_role.id = "role1"
        mock_role.name = "TestRole"
        mock_role.color.value = 0xFF0000
        mock_role.position = 1
        mock_role.permissions.value = 1024

        mock_member.id = "user1"
        mock_member.name = "TestUser"
        mock_member.display_name = "TestUser"
        mock_member.roles = [mock_role]
        mock_member.joined_at = None
        mock_member.bot = False

        mock_guild.members = [mock_member]
        mock_channel.guild = mock_guild

        # Mock permissions
        mock_perms = MagicMock()
        mock_perms.read_messages = True
        mock_perms.send_messages = True
        mock_perms.manage_messages = False
        mock_perms.read_message_history = True
        mock_perms.add_reactions = True
        mock_perms.attach_files = False
        mock_perms.embed_links = True
        mock_perms.mention_everyone = False
        mock_channel.permissions_for = MagicMock(return_value=mock_perms)

        result = await discord_client._sync_channel_members(mock_channel)

        assert len(result) == 1
        member = result[0]
        assert member["user_id"] == "user1"
        assert "TestRole" in member["user_roles"]
        assert member["member_permissions"] is not None

    @pytest.mark.asyncio
    async def test_get_sync_log(self, discord_client):
        """Test getting sync log."""
        # Add some sync data
        discord_client.sync_logger.add_guild_sync("Test Guild", "123456")
        discord_client.sync_logger.add_messages_synced(10)
        discord_client.sync_logger.add_channel_skip(
            "Test Guild", "test-channel", "789", "No access"
        )

        sync_log = discord_client.get_sync_log()

        # Check for actual attributes in sync log (now Pydantic model)
        assert hasattr(sync_log, "guilds_synced")
        assert hasattr(sync_log, "total_messages_synced")
        assert hasattr(sync_log, "errors")
        assert sync_log.total_messages_synced == 10

    @pytest.mark.asyncio
    async def test_client_with_custom_data_store(self):
        """Test client with custom data store."""
        custom_data_store = {
            "messages": {"existing": "data"},
            "users": {"user1": {"name": "Test"}},
            "channels": {"ch1": {"name": "general"}},
            "guilds": {"guild1": {"name": "Test Guild"}},
        }

        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.members = True
        client = DiscordClient(data_store=custom_data_store, intents=intents)

        assert client.data_store == custom_data_store
        assert client.new_data == {}

    @pytest.mark.asyncio
    async def test_client_rate_limit_configuration(self):
        """Test client rate limit configuration."""
        data_store = {}
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.members = True

        # Note: DiscordClient doesn't accept rate_limit_delay as a parameter
        # The rate_limit_delay is hardcoded in __init__
        client = DiscordClient(data_store=data_store, intents=intents)

        # Test that we can modify the rate limit delay after creation
        client.rate_limit_delay = 0.5
        assert client.rate_limit_delay == 0.5
        assert client.max_retries == 3

    @pytest.mark.asyncio
    async def test_client_retry_configuration(self):
        """Test client retry configuration."""
        data_store = {}
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.members = True

        # Note: DiscordClient doesn't accept max_retries as a parameter
        # The max_retries is hardcoded in __init__
        client = DiscordClient(data_store=data_store, intents=intents)

        # Test that we can modify the max retries after creation
        client.max_retries = 5
        assert client.rate_limit_delay == 0.1
        assert client.max_retries == 5

    @pytest.mark.asyncio
    async def test_sync_with_retry_max_retries_exceeded(self, discord_client):
        """Test sync with retry when max retries are exceeded."""
        mock_channel = MagicMock()
        mock_channel.history = MagicMock(
            side_effect=discord.HTTPException(MagicMock(status=500), "HTTP 500")
        )

        with patch("asyncio.sleep") as mock_sleep:
            with pytest.raises(discord.HTTPException, match="HTTP 500"):
                await discord_client.sync_with_retry(mock_channel)

            # Should have retried max_retries times
            assert mock_sleep.call_count == discord_client.max_retries - 1

    @pytest.mark.asyncio
    async def test_sync_with_retry_with_last_message_id(self, discord_client):
        """Test sync with retry using last message ID."""
        mock_channel = MagicMock()
        mock_message = MagicMock()
        mock_message.id = "msg2"

        # Create async iterator for history
        async def async_iter():
            yield mock_message

        mock_channel.history = MagicMock(return_value=async_iter())

        with patch(
            "pepino.discord_bot.extractors.MessageExtractor.extract_message_data"
        ) as mock_extract:
            mock_extract.return_value = {"id": "msg2", "content": "test"}

            result = await discord_client.sync_with_retry(
                mock_channel, last_message_id=123
            )

            assert len(result) == 1
            assert result[0]["id"] == "msg2"
            # Verify history was called with after parameter
            mock_channel.history.assert_called_once()
            call_args = mock_channel.history.call_args
            assert call_args[1]["after"] is not None
