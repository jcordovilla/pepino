# Pepino - Discord Analytics Bot

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency-poetry-blue.svg)](https://python-poetry.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A powerful Discord analytics system that extracts deep insights from server activity with both interactive Discord bot and CLI interfaces.

## ✨ Features

### 🎯 **Dual Interface**
- **Discord Bot**: Interactive slash commands with real-time charts
- **CLI Tool**: Powerful automation and scripting capabilities

### 📊 **Smart Analytics**
- **User Insights**: Activity patterns, peak hours, conversation topics
- **Channel Intelligence**: Engagement metrics, contributor analysis, topic trends
- **Conversation Analysis**: Thread tracking, reply patterns, sentiment analysis

### 🚀 **Production Ready**
- **Repository Pattern**: Clean architecture with 95%+ test coverage
- **Async Performance**: Built for scale with asyncio throughout
- **Professional Logging**: Structured logs with rotation and monitoring

## 🚀 Quick Start

```bash
# Complete setup in one command
make dev-setup

# Copy and edit environment
cp env.example .env
# Add your DISCORD_TOKEN

# Start the bot
pepino start
```

### Discord Bot Setup
1. Create bot at [Discord Developer Portal](https://discord.com/developers/applications)
2. Enable **Message Content Intent** and **Server Members Intent**  
3. Copy token to `.env` file
4. Invite with `View Channels`, `Read Message History`, `Use Slash Commands` permissions

## 🎮 Discord Commands

| Command | Description |
|---------|-------------|
| `/user_analysis <user>` | Deep user activity and engagement analysis |
| `/channel_analysis <channel>` | Channel insights with contributor charts |
| `/topics_analysis [channel]` | AI-powered topic extraction and trends |
| `/top_users` | Most active users with statistics |
| `/activity_trends` | Server activity patterns over time |
| `/sync_and_analyze` | Sync fresh data + run analysis |
| `/list_users` / `/list_channels` | Browse available data |

*All commands support autocomplete and include optional charts*

### Example Output
```
📊 User Analysis: alice_dev

📈 Activity Overview:
• 1,247 messages across 23 channels
• Most Active: #development (312 messages)
• Peak Hours: Tuesday 2-4 PM
• Reply Rate: 67.8%

🏷️ Top Topics:
• deployment, ci/cd, testing (127 messages)
• react, frontend, ui (89 messages)
```

## 🤖 Discord Bot Operations

**Starting the Bot**
```bash
# Simplest form
pepino start

# With options
pepino start --token "your_token" --debug

# Using Make
make run
```

**Configuration**: Uses unified settings from `.env` file and command options.

**📖 Advanced bot operations: [docs/bot_operations.md](docs/bot_operations.md)**

## 💻 CLI Interface

**Purpose**: Automation, scripting, batch analysis, and CI/CD integration.

**Common Workflows**
```bash
# Data management
pepino sync status              # Check data freshness
pepino sync run                 # Smart sync (only if needed)

# Analysis (automation-friendly)
pepino analyze users --limit 20 --format json
pepino analyze channels --channel general --output report.csv
pepino analyze topics --channel dev-team --format json

# Scripting and automation
pepino list channels --format json | jq '.channels[].name'
```

**📖 Complete CLI reference: [docs/operations.md](docs/operations.md)**

## ⚙️ Configuration

**Required**
```bash
DISCORD_TOKEN=your_bot_token_here
```

**Key Settings** (optional)
```bash
LOG_LEVEL=INFO                  # DEBUG, INFO, WARNING
DATABASE_URL=sqlite:///data/discord_messages.db
MAX_MESSAGES=10000             # Analysis limit
CHART_DPI=300                  # Chart quality
```

**📖 Complete configuration: [docs/operations.md](docs/operations.md)**

## 🧪 Development

```bash
# Setup
make dev-setup

# Quality checks
make lint                      # Check code quality
make test                      # Run fast tests
make test-cov                  # Tests with coverage

# Development cycle
make dev                       # Format + fast tests
```

## 📚 Documentation

- **[Operations Guide](docs/operations.md)** - Complete setup, configuration, commands
- **[Bot Operations](docs/bot_operations.md)** - Discord bot setup and troubleshooting  
- **[Architecture](docs/architecture.md)** - Technical details and design decisions

## What's New

### Latest Updates
- **Smart Sync**: Only syncs when data is stale, with progress feedback
- **Enhanced Charts**: User activity timelines and channel contributor visualizations
- **Repository Pattern**: Clean architecture with comprehensive test coverage
- **Professional Logging**: Structured logs with rotation and monitoring targets
- **Unified CLI**: Single `pepino` command with intuitive subcommands

### Architecture Highlights
- **Modern Python**: Full async/await, type hints, Pydantic models
- **Repository Pattern**: Clean data access with easy testing
- **Service Layer**: Business logic separation for maintainability
- **Quality Automation**: Black, isort, mypy, flake8 with Make targets
