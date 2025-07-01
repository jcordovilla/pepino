# Discord Analytics Bot

A comprehensive Discord analytics system with **modular architecture** that provides deep insights into server activity, user engagement, and discussion topics.

## 🏗️ Architecture

**Professional modular design** with clean separation of concerns:

- **`models/`** - Main analyzer classes (`MessageAnalyzer`, `DiscordBotAnalyzer`)
- **`analysis/`** - ML/NLP analysis (embeddings, topics, statistics, insights)
- **`visualization/`** - Chart generation and data visualization
- **`database/`** - Database operations and schema management
- **`core/`** - Text processing and NLP utilities
- **`utils/`** - Helper functions and utilities

## 🚀 Features

- **📊 Channel Analysis** - Deep insights into channel activity and topics
- **👥 User Statistics** - Top contributors with activity patterns (human-only metrics)
- **📈 Activity Trends** - Server-wide activity visualization with charts
- **🧠 Topic Analysis** - AI-powered topic extraction and discussion themes
- **🤖 vs 👤 Bot/Human Separation** - All metrics exclude bots for accurate insights
- **� Interactive Commands** - Discord slash commands for real-time analytics

## 💬 Available Commands

| Command | Description |
|---------|-------------|
| `/pepino_channel_analysis` | Channel insights with topics & activity charts |
| `/pepino_user_analysis` | Individual user analysis with contribution patterns |
| `/pepino_top_users` | Top 10 most active users with main topics |
| `/pepino_topics_analysis` | Discussion themes and topic patterns |
| `/pepino_activity_trends` | Server activity trends with charts |
| `/pepino_list_users` | List all available users for analysis |
| `/pepino_list_channels` | List all available channels for analysis |
| `/pepino_help` | Show all available commands |

## 🛠️ Setup

### Prerequisites
- Python 3.7+
- Discord bot token

### Quick Start
```bash
# Clone and setup
git clone <repo-url>
cd pepino
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure environment
echo "DISCORD_TOKEN=your_bot_token_here" > .env

# Fetch messages and start bot
python fetch_messages.py
python bot.py
```

### Discord Bot Setup
1. Create bot at [Discord Developer Portal](https://discord.com/developers/applications)
2. Enable **Message Content Intent** and **Server Members Intent**
3. Invite with permissions: `View Channels`, `Read Message History`

## 📊 Sample Output

### Channel Analysis
```
📊 Basic Statistics:
• Total Messages: 1,234 (89.2% human, 10.8% bot)
• Unique Human Users: 45
• Human Engagement Metrics:
  - Average Replies per Post: 1.23
  - Posts with Reactions: 15.6%
```

### User Statistics
```
📊 Top 10 Human User Activity

1. Oscar Sanchez
   • Messages: 663 • Channels: 58
   • Most Active: #general (187 messages)
   • Main Topics: session, linkedin, onboarding
```

## �️ File Structure

```
pepino/
├── bot.py                     # Discord bot with slash commands
├── bot_commands.py            # Command implementations  
├── fetch_messages.py          # Message fetching
├── models/                    # Main analyzer classes
├── analysis/                  # ML/NLP analysis modules
├── visualization/             # Chart generation
├── database/                  # Database operations
├── core/                      # Text processing & NLP
└── utils/                     # Helper functions
```

## 🔧 Development

### Using Individual Modules
```python
# Main classes (for Discord bot)
from models import DiscordBotAnalyzer

# Individual analysis functions
from analysis.insights import get_user_insights
from analysis.topics import analyze_topics_spacy
from visualization import create_user_activity_chart
```

### Benefits
- **Modular**: Each component is independent
- **Testable**: Easy to test individual modules
- **Maintainable**: Clear separation of concerns
- **Extensible**: Easy to add new features

## 🛡️ Notes

- Store bot token securely in `.env` file (not tracked in git)
- Requires Discord permissions: `View Channels`, `Read Message History`
- Analytics focus on patterns and engagement metrics
- All statistics exclude bot messages for accurate human insights
