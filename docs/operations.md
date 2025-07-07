# Operations Guide

## Quick Start

### 1. Complete Setup
```bash
# One command setup (installs everything)
make dev-setup

# Or manual setup
poetry install
poetry run python scripts/install_spacy.py
```

### 2. Configure Environment
```bash
# Copy template and edit
cp .env.example .env
# Add: DISCORD_TOKEN=your_bot_token_here
```

### 3. Run Application
```bash
# Discord bot
pepino start

# CLI analysis
pepino analyze users --limit 10
```

## Configuration

### Required Settings
```bash
# .env file
DISCORD_TOKEN=your_discord_bot_token_here
```

### Configuration Reference

| Setting | Default | Description |
|---------|---------|-------------|
| **Discord** | | |
| `DISCORD_TOKEN` | *(required)* | Discord bot token |
| `COMMAND_PREFIX` | `!` | Bot command prefix |
| `MESSAGE_CONTENT_INTENT` | `true` | Enable message content access |
| `MEMBERS_INTENT` | `true` | Enable server members access |
| **Database** | | |
| `DATABASE_URL` | `sqlite:///data/discord_messages.db` | Database connection |
| `DB_PATH` | `discord_messages.db` | SQLite database file path |
| **Analysis** | | |
| `MAX_MESSAGES` | `10000` | Maximum messages to analyze |
| `MIN_MESSAGE_LENGTH` | `50` | Minimum message length for analysis |
| `MAX_MESSAGES_PER_ANALYSIS` | `800` | Limit per single analysis |
| `TOPIC_MODEL_N_COMPONENTS` | `5` | Topic modeling components |
| **Visualization** | | |
| `CHART_DPI` | `300` | Chart resolution quality |
| `CHART_FORMAT` | `png` | Chart output format |
| `TEMP_DIRECTORY` | `temp` | Temporary files directory (auto-created) |
| **Logging** | | |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `DEBUG` | `false` | Enable debug mode |

### Complete .env Template
[.env.example](../.env.example)

## Running the Application

```bash
# Discord bot
pepino start

# CLI analysis  
pepino analyze users
pepino analyze channels
pepino analyze topics

# Data sync
pepino sync run
pepino sync status

# Help
pepino --help
```

### Discord Bot Mode
```bash
# Start bot
pepino start

# With options
pepino start --token "your_token" --debug

# Using Make
make run
```

### CLI Analysis Mode
```bash
# Core analysis commands
pepino analyze users --limit 10
pepino analyze channels --channel general
pepino analyze topics --channel dev-team
pepino analyze temporal --days 30

# Data management
pepino sync run                # Smart sync
pepino sync run --force        # Force sync
pepino export-data --format csv

# Automation-friendly
pepino list channels --format json
pepino list users --format csv
```

### CLI Options
```bash
# Common options for all commands
--verbose                      # Detailed output
--format json|csv|text         # Output format
--output FILE                  # Save to file
--limit N                      # Limit results

# Analysis-specific options
--channel NAME                 # Filter by channel  
--days N                       # Time range
--topics N                     # Number of topics
```

### E2E Automation & Scripting
```bash
# 🆕 Listing Commands for Automation
pepino list channels                    # List all available channels
pepino list users                      # List all available users  
pepino list stats                      # Database statistics

# Automation-friendly formats
pepino list channels --format json     # Machine-readable channel list
pepino list users --format csv         # CSV export for spreadsheets
pepino list stats --output stats.json  # Save stats for monitoring

# Example automation workflow
channels=$(pepino list channels --format json)
for channel in $(echo $channels | jq -r '.channels[].name'); do
    pepino analyze channels -c "$channel" --output "analysis_${channel}.json"
done
```

## Discord Bot Setup

### 1. Create Bot Application
1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Create new application
3. Go to "Bot" section
4. Create bot and copy token to `.env`

### 2. Required Permissions
- **View Channels**
- **Read Message History**
- **Send Messages** (for bot responses)
- **Use Slash Commands**
- **Embed Links** (for charts)
- **Attach Files** (for exports)
- **Add Reactions** (for interactive features)

### 3. Required Intents
```bash
# In .env
MESSAGE_CONTENT_INTENT=true
MEMBERS_INTENT=true
```

### 4. Invite Bot
Use OAuth2 URL with permissions:
```
https://discord.com/api/oauth2/authorize?client_id=YOUR_BOT_ID&permissions=68608&scope=bot%20applications.commands
```

### 5. Available Slash Commands
Once deployed, the bot provides these slash commands:

| Command | Parameters | Description |
|---------|------------|-------------|
| `/user_analysis` | `user`, `include_chart` | Deep user activity and engagement analysis |
| `/channel_analysis` | `channel`, `include_chart` | Channel insights with contributor charts |
| `/topics_analysis` | `channel` (optional) | AI-powered topic extraction and trends |
| `/top_users` | - | Most active users with statistics |
| `/activity_trends` | `include_chart` | Server activity patterns over time |
| `/list_users` | `limit` | Browse available users (supports autocomplete) |
| `/list_channels` | `limit` | Browse available channels |
| `/sync_status` | - | Check data freshness and sync recommendations |
| `/sync_and_analyze` | `analysis_type`, `target`, `force_sync` | Sync fresh data + run analysis |
| `/help_analysis` | - | Show detailed command help |

*All commands support autocomplete and include optional charts*

**📖 For detailed bot operations and troubleshooting, see the Discord bot setup section above**
**🚀 For a simple getting started guide, see the [Discord Bot Quick Start Guide](../README.md#-discord-bot-quick-start-guide) in the README**

## Database Operations

```bash
# Initialize (auto-initialized on first run)
pepino sync run

# Backup database
make backup-db

# Check status
sqlite3 discord_messages.db "SELECT COUNT(*) FROM messages;"
```

## Development Operations

### Code Quality & Testing
```bash
# Complete development setup
make dev-setup

# Code quality (replaces 4 separate commands)
make lint                      # Check formatting, imports, types, style
make lint-fix                  # Auto-fix formatting issues

# Testing
make test                      # Run all tests (fast with mocking)
make test-cov                  # Tests with coverage report
make test-watch                # Watch mode for development

# Development cycle
make dev                       # Format + fast tests
```

### Logging & Monitoring
```bash
# View logs (professional logging system)
make logs                      # Recent application logs
make logs-follow               # Follow logs in real-time
make logs-errors               # Error logs only
make logs-discord              # Discord bot logs

# Debug mode
make debug                     # Run with verbose logging
pepino start --debug           # Debug Discord bot

# Log management
make logs-clean                # Clean old log files
```

### Log Files
```
logs/
├── pepino.log                # Main application (all levels)
├── errors.log                # Errors and critical messages  
└── discord.log               # Discord bot specific logs
```

### Test Architecture
- **Fast Execution**: Tests complete in under 2 seconds (no database setup)
- **Repository Mocking**: Uses `unittest.mock.AsyncMock` for isolation
- **Good Coverage**: All analysis modules with comprehensive edge cases

## Monitoring & Troubleshooting

### Quick Fixes

| Issue | Solution |
|-------|----------|
| "No Discord token found" | Check `.env` file has `DISCORD_TOKEN=your_token` |
| "Database not found" | Run `pepino sync run` to initialize |
| "Import errors" | Run `make dev-setup` or `poetry install` |
| Commands not working | Try `pepino start --debug` for detailed logs |

### Monitoring
```bash
# Check status
pepino sync status
make logs                      # View recent logs
make logs-follow               # Monitor in real-time
make logs-errors               # Check for issues

# Performance
du -h discord_messages.db      # Database size
sqlite3 discord_messages.db "SELECT COUNT(*) FROM messages;"
```

## Production Deployment

### Environment Variables
```bash
# Set in production environment (not .env file)
export DISCORD_TOKEN=your_token
export DATABASE_URL=postgresql://user:pass@host:5432/db
export LOG_LEVEL=WARNING
```

### Database Migration
```bash
# For PostgreSQL production
export DATABASE_URL=postgresql://user:pass@host:5432/pepino
pepino sync run
```

### Process Management
```bash
# Using systemd or similar
poetry run python -m pepino
```

### Health Checks
```bash
# Simple health check
poetry run python -c "from pepino.data.database.manager import DatabaseManager; import asyncio; asyncio.run(DatabaseManager().initialize())"
```

## Configuration Validation

### Test Configuration
```bash
# Validate settings
poetry run python -c "
from pepino.data.config import Settings
try:
    settings = Settings()
    print('✅ Configuration valid')
    print(f'Discord token: {bool(settings.discord_bot_token)}')
    print(f'Database: {settings.database_url}')
    print(f'Base filter: {settings.analysis_base_filter_sql}')
except Exception as e:
    print(f'❌ Configuration error: {e}')
"
```

### Test Repository Layer
```bash
# Test database connectivity and repositories
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

### Environment Check
```bash
# Check required dependencies
poetry run python -c "
import discord
import aiosqlite
import spacy
print('✅ All dependencies available')
"
```

### Development Workflow

#### Pre-commit Checklist
```bash
# Run before committing changes
poetry run black src/ tests/           # Format code
poetry run isort src/ tests/           # Sort imports
poetry run mypy src/                   # Type checking
poetry run flake8 src/ tests/          # Linting
poetry run pytest                     # Run tests
```

#### Testing New Features
1. **Write Tests First**: Create tests using repository mocking
2. **Implement Feature**: Add analyzer or repository methods
3. **Verify Coverage**: Ensure comprehensive test coverage
4. **Integration Test**: Test with actual CLI/Discord commands

#### Repository Pattern Development
```python
# Example: Adding new repository method
class MessageRepository:
    async def get_new_analysis_data(self, **kwargs) -> List[Dict]:
        # Implement database query
        pass

# Example: Testing with mocks
@pytest.mark.asyncio
async def test_new_analyzer():
    with patch.object(analyzer.message_repo, 'get_new_analysis_data', new_callable=AsyncMock) as mock_method:
        mock_method.return_value = test_data
        result = await analyzer.analyze()
        assert result["success"] is True
```

This guide covers the essential operations needed to configure, run, and maintain the Pepino Discord analytics system with its modern repository-based architecture and fast testing approach. 