# Discord Analytics Bot

> **📌 Choose Your Branch:**
> - **`main`** - Full-featured Discord analytics bot with advanced insights, charts, and AI analysis
> - **`simple-fetcher`** - Basic Discord message fetcher for building your own analytics

A comprehensive Discord analytics system that fetches, analyzes, and visualizes Discord server activity with advanced insights and bot/human differentiation.

## 🚀 Features

### Core Analytics
- **📊 Channel Analysis** - Deep insights into channel activity, engagement, and topics
- **👥 User Statistics** - Top contributors with activity patterns (human-only metrics)
- **📈 Activity Trends** - Server-wide activity visualization with 30-day charts
- **🧠 Topic Analysis** - AI-powered topic extraction and discussion themes
- **📱 Bot Commands** - Interactive Discord slash commands for real-time analytics

### Key Differentiators
- **🤖 vs 👤 Bot/Human Separation** - All metrics exclude bots for accurate human engagement
- **📊 Visual Charts** - Matplotlib-generated activity trends and patterns
- **🎯 Smart Topic Extraction** - Relevant topics using NLP and frequency analysis
- **� Engagement Metrics** - Reply rates, reaction rates, participation analysis
- **🕐 Temporal Patterns** - Peak hours, daily/weekly activity breakdowns

## � Available Commands

| Command | Description |
|---------|-------------|
| `/channel_analysis` | Detailed channel insights with key topics & activity charts |
| `/user_analysis` | Individual user analysis with contribution patterns |
| `/top_users` | Top 10 most active human users with main topics |
| `/topics_analysis` | Channel topic themes and discussion patterns |
| `/activity_trends` | Server activity trends with 30-day evolution charts |

## 🛠️ Setup

### Prerequisites
- Python 3.7+
- Discord bot token with proper permissions

### Quick Start
1. **Clone and setup**
   ```bash
   git clone <repo-url>
   cd pepino
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure environment**
   ```bash
   # Create .env file
   echo "DISCORD_TOKEN=your_bot_token_here" > .env
   ```

3. **Discord Bot Setup**
   - Create bot at [Discord Developer Portal](https://discord.com/developers/applications)
   - Enable **Message Content Intent** and **Server Members Intent**
   - Invite with permissions: `View Channels`, `Read Message History`

4. **Fetch data and run bot**
   ```bash
   # Fetch messages (one-time setup)
   python fetch_messages.py
   
   # Start analytics bot
   python bot.py
   ```

## 📊 Sample Analytics Output

### Channel Analysis
```
📊 Basic Statistics:
• Total Messages: 1,234 (89.2% human, 10.8% bot)
• Unique Human Users: 45
• Average Message Length: 156.3 characters

📈 Human Engagement Metrics:
• Average Replies per Original Post: 1.23
• Posts with Reactions: 15.6% (192/1,234)
• Note: Bot messages excluded from calculations

📈 Channel Health Metrics (Human Activity):
• Weekly Active Human Members: 12 (26.7% of total)
• Human Participation Rate: 78.3%
• Activity Ratio: 12 active / 8 inactive / 25 lurkers
```

### Activity Trends
- **30-day server evolution chart** with daily message trends
- **Weekly pattern analysis** showing peak activity days
- **Hourly breakdown** with morning/afternoon/evening percentages
- **Visual matplotlib charts** saved and sent to Discord

### User Statistics
```
📊 Top 10 Human User Activity Statistics

1. Oscar Sanchez
• Messages: 663 • Channels: 58 • Avg Length: 276 chars
• Most Active: #netarch-general (187 messages)
• Main Topics: session, linkedin, onboarding

2. Jose Cordovilla  
• Messages: 358 • Channels: 38 • Avg Length: 176 chars
• Most Active: #netarch-general (96 messages)
• Main Topics: workshop, session, deployment
```

## 🗃️ Database Schema

**Enhanced Tables:**
- `messages` - All Discord messages with bot detection (`author_is_bot` field)
- `channel_members` - Complete channel membership data
- `sync_logs` - Synchronization history and statistics

**Auto-generated Analysis:**
- Message embeddings for similarity analysis
- Topic modeling results with spaCy NLP
- Temporal activity patterns and trends
- User engagement metrics (human-only)

## 🎯 Key Analytics Insights

### What Makes This Special
1. **Human-Centric Metrics** - Excludes bots from all engagement calculations
2. **Visual Analytics** - Charts and graphs for trend analysis
3. **Real-time Discord Integration** - Use commands directly in your server
4. **Intelligent Topic Extraction** - Finds meaningful discussion themes
5. **Comprehensive Health Metrics** - Activity ratios, participation rates

### Bot vs Human Differentiation
- **Message counts** show percentage breakdown
- **Engagement metrics** calculated on human activity only
- **User statistics** completely exclude bot accounts
- **Activity trends** focus on human participation patterns

## 🔧 File Structure

```
pepino/
├── bot.py                 # Discord bot with slash commands
├── bot_commands.py        # Command implementations
├── analysis.py           # Core analytics engine
├── fetch_messages.py     # Message fetching and storage
├── requirements.txt      # Dependencies
├── discord_messages.db   # SQLite database
└── temp/                # Generated charts and visualizations
```

## 📈 Example Use Cases

- **Community Management** - Track engagement and identify active contributors
- **Content Strategy** - Understand what topics drive the most discussion
- **Growth Analysis** - Monitor server health and member participation
- **Moderation Insights** - Identify channels needing attention
- **Trend Analysis** - Spot emerging topics and discussion patterns

## 🛡️ Security & Privacy

- Bot token stored securely in `.env` (not tracked in git)
- Database contains conversation data - handle appropriately
- Analytics focus on patterns, not individual message content
- Respects Discord's Terms of Service and API limits
