# Support Guide

Need help with Pepino? You're in the right place! This guide will help you get the support you need.

## 🚀 Quick Help

### Common Issues & Solutions

| Issue | Quick Fix |
|-------|-----------|
| **"No Discord token found"** | Check `.env` file has `DISCORD_TOKEN=your_token` |
| **"Database not found"** | Run `pepino sync run` to initialize |
| **"Import errors"** | Run `poetry install` to install dependencies |
| **Commands not working** | Try `pepino start --debug` for detailed logs |
| **Bot not responding** | Check bot permissions and token validity |

### Quick Diagnostic
```bash
# Check if everything is set up correctly
poetry run python -c "
from pepino.config import Settings
from pepino.data.database.manager import DatabaseManager
import asyncio

async def check_setup():
    try:
        settings = Settings()
        print('✅ Configuration loaded')
        print(f'Discord token: {bool(settings.discord_token)}')
        
        db_manager = DatabaseManager()
        await db_manager.initialize()
        print('✅ Database connected')
        await db_manager.close()
        
    except Exception as e:
        print(f'❌ Setup issue: {e}')

asyncio.run(check_setup())
"
```

## 📞 Getting Help

### 1. **Check Documentation First**
- [README.md](../README.md) - Quick start and basic usage
- [Operations Guide](operations.md) - Detailed setup and configuration
- [Development Guide](development.md) - Architecture and technical details

### 2. **Search Existing Issues**
- [GitHub Issues](https://github.com/your-username/pepino/issues) - Check if your problem is already reported
- [GitHub Discussions](https://github.com/your-username/pepino/discussions) - Community Q&A

### 3. **Create a New Issue**
If you can't find an answer, create a new issue with:

**For Bugs:**
- **Environment**: OS, Python version, Pepino version
- **Steps to reproduce**: Exact commands and actions
- **Expected vs actual behavior**: What should happen vs what happens
- **Error messages**: Full error output and stack traces
- **Logs**: Relevant log output (use `--debug` flag)

**For Feature Requests:**
- **Problem description**: What you're trying to accomplish
- **Proposed solution**: How you think it should work
- **Use case**: Why this feature would be valuable

### 4. **Community Support**
- **Discord Server**: Join our community Discord for real-time help
- **GitHub Discussions**: Ask questions and share solutions
- **Stack Overflow**: Tag questions with `pepino` and `discord-analytics`

## 🔧 Troubleshooting

### Discord Bot Issues

**Bot Not Responding:**
```bash
# Check if bot is running
ps aux | grep pepino

# Check logs
pepino start --debug

# Verify token
echo $DISCORD_TOKEN
```

**Permission Errors:**
- Bot needs "Read Message History" permission
- Check channel-specific permissions
- Verify bot role hierarchy

**Commands Not Found:**
- Ensure bot has "Use Slash Commands" permission
- Try re-inviting bot with updated permissions
- Check if commands are registered (restart bot)

### Data Sync Issues

**"No Data Found" Errors:**
```bash
# Check sync status
pepino sync status

# Run initial sync
pepino sync run

# Check database
ls -la discord_messages.db
```

**Sync Failures:**
- Verify Discord token has required permissions
- Check network connectivity
- Ensure Discord servers are accessible
- Try force sync: `/sync_and_analyze force_sync:true`

**Stale Data Warnings:**
```bash
# Check last sync
pepino sync status

# Force fresh sync
pepino sync run --full --clear

# Incremental update
pepino sync run --force
```

### Performance Issues

**Slow Analysis:**
- Database too large: Consider data cleanup
- Network latency: Increase sync timeout
- Memory usage: Monitor during peak times

**High Memory Usage:**
- Reduce `MAX_MESSAGES_PER_ANALYSIS` in config
- Use smaller time ranges for analysis
- Consider database optimization

### Configuration Issues

**Environment Variables:**
```bash
# Check current settings
poetry run python -c "
from pepino.config import Settings
settings = Settings()
print(f'Discord token: {bool(settings.discord_token)}')
print(f'Database: {settings.database_url}')
print(f'Base filter: {settings.analysis_base_filter_sql}')
"
```

**Database Connection:**
```bash
# Test database connectivity
sqlite3 discord_messages.db "SELECT COUNT(*) FROM messages;"
```

## 📊 Monitoring & Health Checks

### System Health
```bash
# Check overall system status
pepino sync status
pepino analyze database-stats

# Monitor logs
tail -f logs/pepino.log
tail -f logs/errors.log
```

### Performance Metrics
- **Database size**: `du -h discord_messages.db`
- **Message count**: `sqlite3 discord_messages.db "SELECT COUNT(*) FROM messages;"`
- **Sync frequency**: Check last sync timestamp
- **Response times**: Monitor command execution times

## 🛠️ Advanced Troubleshooting

### Debug Mode
```bash
# Run with verbose logging
pepino start --debug

# Check specific components
poetry run python -c "
import asyncio
from pepino.data.database.manager import DatabaseManager
from pepino.data.repositories.message_repository import MessageRepository

async def test_repos():
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()
        message_repo = MessageRepository(db_manager)
        print('✅ Repository layer working')
        await db_manager.close()
    except Exception as e:
        print(f'❌ Repository error: {e}')

asyncio.run(test_repos())
"
```

### Database Maintenance
```bash
# Database optimization
sqlite3 discord_messages.db "VACUUM;"
sqlite3 discord_messages.db "ANALYZE;"

# Backup before maintenance
cp discord_messages.db discord_messages.db.backup
```

### Dependency Issues
```bash
# Reinstall dependencies
poetry install --sync

# Check for conflicts
poetry show --tree

# Update dependencies
poetry update
```

## 📚 Learning Resources

### Documentation
- [Architecture Guide](development.md) - System design and patterns
- [Testing Guide](testing.md) - Testing strategy and debugging
- [Contributing Guide](../CONTRIBUTORS.md) - Development setup and process

### External Resources
- [Discord.py Documentation](https://discordpy.readthedocs.io/)
- [Discord Developer Portal](https://discord.com/developers/docs)
- [Python Poetry Documentation](https://python-poetry.org/docs/)

## 🆘 Emergency Support

For critical issues affecting production use:

1. **Check logs immediately**: `tail -f logs/errors.log`
2. **Create urgent issue**: Tag with `urgent` or `critical`
3. **Provide context**: Include environment and impact details
4. **Follow up**: Monitor issue for responses

## 🤝 Contributing to Support

Help improve support for everyone:

- **Answer questions** in GitHub Discussions
- **Improve documentation** when you find gaps
- **Report issues** with clear, reproducible steps
- **Share solutions** that worked for you

**We're here to help you succeed with Pepino!** 🚀 