from pepino.discord.data.sync_logger import SyncLogger


def test_sync_logger_add_and_get():
    logger = SyncLogger()
    logger.add_guild_sync("TestGuild", "123")
    logger.add_messages_synced(5)
    entry = logger.get_log_entry()
    assert hasattr(entry, "guilds_synced")
    assert entry.total_messages_synced == 5
