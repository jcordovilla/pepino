# Pepino - Advanced Discord Analytics Platform

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency-poetry-blue.svg)](https://python-poetry.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Pepino** is a comprehensive Discord analytics platform that transforms your server conversations into actionable insights. Whether you're managing a community, running a business team, or analyzing social interactions, Pepino provides deep intelligence about user behavior, conversation patterns, and community dynamics.

> **🚀 New to Pepino?** Jump straight to the [Discord Bot Quick Start Guide](#-discord-bot-quick-start-guide) for immediate hands-on testing!

## 🎯 What Pepino Does

### **Core Purpose**
Pepino analyzes Discord server activity to help you:
- **Understand Your Community**: Who are your most active members? When is your server busiest?
- **Optimize Engagement**: Which channels drive the most conversation? What topics resonate?
- **Track Growth**: How is your community evolving over time?
- **Make Data-Driven Decisions**: Use real insights instead of gut feelings

## 🤝 Collaboration Guidelines

**Welcome contributors!** This project is open for collaboration. Please follow these guidelines to ensure smooth development:

### **🔒 Security & Privacy First**
- **Never commit sensitive data**: Database files (*.db), message exports (*.json), or real Discord tokens
- **Use example data**: When creating tests or documentation, use synthetic/example data only
- **Review .gitignore**: Before committing, ensure no sensitive files are tracked
- **Environment variables**: Always use `.env` files for secrets (never commit them)

### **💻 Development Standards**
- **Branch naming**: Use descriptive names like `feature/user-analytics` or `fix/chart-rendering`
- **Commit messages**: Use conventional commits format: `feat:`, `fix:`, `docs:`, `refactor:`
- **Code quality**: Run `make quality` before submitting PRs (formatting, linting, type checks)
- **Testing**: Include tests for new features and ensure existing tests pass

### **📋 Contribution Process**
1. **Fork & branch**: Create a feature branch from the latest `main`
2. **Develop & test**: Implement changes with appropriate tests
3. **Document**: Update README/docs if adding new features
4. **Submit PR**: Include clear description of changes and any breaking changes
5. **Code review**: Address feedback promptly and professionally

### **🛡️ Data Handling Rules**
- **Real Discord data**: Only use for local development, never share or commit
- **Test data**: Create synthetic examples for documentation and tests
- **User privacy**: Anonymize any user identifiers in examples or screenshots
- **Compliance**: Ensure all contributions respect Discord's Terms of Service

### **❓ Getting Help**
- **Issues**: Use GitHub Issues for bugs, feature requests, and questions
- **Discussions**: Use GitHub Discussions for general questions and ideas
- **Documentation**: Check existing docs before asking questions

### **📜 Open Source & Attribution**
- **License**: This project is open source under MIT License - see [LICENSE](LICENSE) file
- **Attribution**: Contributors retain credit for their contributions in git history
- **Commercial use**: Permitted under MIT License terms
- **Derivative works**: Encouraged! Please maintain attribution to original project
- **Third-party libraries**: Ensure any added dependencies have compatible licenses
- **Community driven**: This project belongs to the community - treat it with respect
- **Name origin**: Pepino is named after a previous bot called "Pepe" - thanks to [@reisenhardt](https://github.com/reisenhardt) for the name!

---

### **Key Capabilities**

#### 🔍 **User Analytics**
- **Activity Patterns**: When users are most active (hourly, daily, weekly trends)
- **Engagement Metrics**: Message frequency, reply rates, conversation starters
- **Topic Interests**: What subjects each user discusses most
- **Channel Preferences**: Where users spend their time and contribute most

#### 📊 **Channel Intelligence**
- **Health Metrics**: Message volume, user participation, response times
- **Content Analysis**: Topic extraction, conversation themes, sentiment patterns
- **Contributor Analysis**: Who drives discussions, lurker vs. active ratios
- **Growth Tracking**: How channel activity changes over time

#### 🧠 **Conversation Analysis**
- **Topic Modeling**: AI-powered extraction of discussion themes
- **Thread Tracking**: How conversations develop and branch
- **Sentiment Analysis**: Emotional tone of discussions
- **Interaction Patterns**: Reply chains, mention networks, collaboration flows

#### 🏢 **Server Overview**
- **Community Health**: Overall activity levels, growth trends, engagement quality
- **Peak Times**: When your server is most active for optimal event scheduling
- **Channel Performance**: Which channels succeed and which need attention
- **Member Insights**: New vs. veteran member behavior patterns

## ✨ Features & Interfaces

### 🎮 **Discord Bot Interface**
Perfect for **real-time insights** and **interactive exploration**.

**Why Use the Bot?**
- Instant analysis without leaving Discord
- Visual charts and graphs embedded in responses
- Autocomplete for easy channel/user selection
- Perfect for moderators and community managers

**Key Commands:**
```
/pepino_channel_analytics overview    # Complete channel health report
/pepino_channel_analytics topics     # AI topic analysis
/pepino_user_analytics alice_dev     # Deep user insights
/pepino_server_analytics overview    # Server-wide statistics
/pepino_lists users                  # Browse community members
```

### 💻 **CLI Interface**
Perfect for **automation**, **scripting**, and **detailed analysis**.

**Why Use the CLI?**
- Batch processing and automation
- Export data in multiple formats (JSON, CSV, TXT)
- Integration with scripts and CI/CD pipelines
- Detailed control over analysis parameters

**Key Commands:**
```bash
pepino analyze users --limit 20 --format json
pepino analyze channels --channel general --days 30
pepino sync run --progress                    # Update data
pepino list channels --format csv             # Export channel list
```

## 🚀 Quick Start Guide

### **Step 1: Installation**
```bash
# Clone and setup (recommended for development)
git clone https://github.com/your-repo/pepino.git
cd pepino
make dev-setup

# OR install via pip (when available)
pip install pepino-discord
```

### **Step 2: Discord Bot Setup**
1. **Create Discord Application**
   - Visit [Discord Developer Portal](https://discord.com/developers/applications)
   - Click "New Application" and give it a name
   - Go to "Bot" section and click "Add Bot"

2. **Configure Permissions**
   - Enable **Message Content Intent** (required for reading messages)
   - Enable **Server Members Intent** (for user analysis)
   - Generate and copy your bot token

3. **Invite Bot to Server**
   - Go to "OAuth2" > "URL Generator"
   - Select "bot" and "applications.commands" scopes
   - Select permissions: `View Channels`, `Read Message History`, `Use Slash Commands`
   - Use generated URL to invite bot

### **Step 3: Configuration**
```bash
# Copy environment template
cp env.example .env

# Edit .env file
DISCORD_TOKEN=your_bot_token_here
LOG_LEVEL=INFO
DATABASE_URL=sqlite:///data/discord_messages.db
```

### **Step 4: First Run**
```bash
# Start the bot
pepino start

# In Discord, try:
/pepino_help                    # See all available commands
/pepino_server_analytics overview  # Get server overview
```

## 📚 Learning Guide: Understanding Analytics

### **What is User Analytics?**

User analytics helps you understand individual member behavior:

**Example: Analyzing User "alice_dev"**
```
📊 User Analysis: alice_dev

📈 Activity Overview:
• 1,247 messages across 23 channels (High engagement)
• Most Active: #development (312 messages, 25% of activity)
• Peak Hours: Tuesday 2-4 PM (Perfect for 1:1s)
• Reply Rate: 67.8% (Very collaborative)

🏷️ Discussion Topics:
• deployment, ci/cd, testing (127 messages) - DevOps focus
• react, frontend, ui (89 messages) - Frontend expertise
• team, planning, meetings (56 messages) - Leadership involvement

💡 Insights:
- alice_dev is a core contributor with deep technical knowledge
- Best time to reach her: Tuesday afternoons
- She bridges technical and planning discussions
```

**Use Cases:**
- **Team Management**: Identify subject matter experts
- **Scheduling**: Find optimal meeting times
- **Recognition**: Highlight valuable contributors
- **Onboarding**: Match new members with active mentors

### **What is Channel Analytics?**

Channel analytics reveals how spaces are used and performing:

**Example: Analyzing #development Channel**
```
📊 Channel Analysis: #development

📈 Health Metrics:
• 2,847 messages from 45 users (Very active)
• Average: 23 messages/day (Healthy pace)
• Response Time: 1.2 hours average (Good responsiveness)
• Engagement Rate: 78% (High participation)

👥 Top Contributors:
• alice_dev: 312 messages (11% of channel)
• bob_backend: 287 messages (10% of channel)
• carol_frontend: 201 messages (7% of channel)

🏷️ Discussion Topics:
• Bug Reports & Fixes (487 messages, 17%)
• Code Reviews (334 messages, 12%)
• Architecture Discussions (298 messages, 10%)

💡 Insights:
- Healthy technical discussion channel
- Good balance of contributors (no single person dominates)
- Focus on practical development issues
```

**Use Cases:**
- **Channel Optimization**: Identify underused or overwhelming channels
- **Content Strategy**: Understand what topics drive engagement
- **Moderation**: Spot channels that need more attention
- **Team Balance**: Ensure diverse participation

### **What is Topic Analysis?**

Topic analysis uses AI to understand conversation themes:

**How It Works:**
1. **Message Collection**: Gathers messages from specified timeframe
2. **Content Cleaning**: Removes noise (mentions, emojis, links)
3. **Quality Assessment**: Determines if content suits advanced AI analysis
4. **Topic Extraction**: 
   - **Advanced Mode**: Uses BERTopic neural networks for complex discussions
   - **Discord Mode**: Uses pattern recognition for casual conversations
5. **Result Presentation**: Shows meaningful themes with relevance scores

**Example: AI Topic Analysis**
```
🎯 Enhanced Topic Analysis in #general-chat

📊 Analysis Summary:
• Messages Analyzed: 1,455
• Analysis Method: Discord-Optimized Analysis
• Time Period: Last 30 days

💬 Discussion Themes Identified:
1. Meeting & Scheduling (892 mentions, 61.3% of messages)
2. Project Updates (234 mentions, 16.1% of messages)
3. Tool Discussions (156 mentions, 10.7% of messages)
4. Help & Questions (98 mentions, 6.7% of messages)
```

**Use Cases:**
- **Content Planning**: Understand what your community cares about
- **FAQ Creation**: Identify common questions and topics
- **Event Planning**: See what activities generate interest
- **Community Direction**: Track evolving interests over time

## 🎮 Discord Commands Reference

### **Channel Analytics Commands**

#### `/pepino_channel_analytics overview`
**Purpose**: Complete health check of a channel
**Best For**: Moderators, community managers
**Example Output**: Activity metrics, top contributors, engagement rates, health scores

```
Options:
- channel_name: Specific channel (default: current channel)
- days: Time period (default: all time)
- end_date: Analysis end point (default: today)
```

#### `/pepino_channel_analytics topics`
**Purpose**: AI-powered topic extraction
**Best For**: Understanding conversation themes
**Example Output**: Discussion topics with relevance percentages

```
Options:
- channel_name: Target channel (required)
- days: Analysis period (default: all time)
```

### **User Analytics Commands**

#### `/pepino_user_analytics`
**Purpose**: Deep dive into user behavior
**Best For**: Team leads, HR, community insights
**Example Output**: Activity patterns, peak times, topic interests, channel preferences

```
Options:
- username: Target user (with autocomplete)
- days: Analysis period (default: 30 days)
- include_semantic: Advanced topic analysis (default: true)
```

### **Server Analytics Commands**

#### `/pepino_server_analytics overview`
**Purpose**: Server-wide health and activity metrics
**Best For**: Server owners, community health monitoring
**Example Output**: Total activity, growth trends, top channels/users, engagement metrics

#### `/pepino_server_analytics top_users`
**Purpose**: Most active community members
**Best For**: Recognition, moderation insights
**Example Output**: Ranked list with activity metrics

```
Options:
- limit: Number of users to show (default: 10)
- days: Time period (default: 30 days)
```

### **Utility Commands**

#### `/pepino_lists users` / `/pepino_lists channels`
**Purpose**: Browse available data
**Best For**: Exploring your data, finding usernames/channels
**Features**: Pagination for large lists, search-friendly output

#### `/pepino_help`
**Purpose**: Command reference and help
**Best For**: Learning available features

## 💻 CLI Commands Reference

### **Analysis Commands**

#### User Analysis
```bash
# Basic user analysis
pepino analyze users alice_dev

# Multiple users with options
pepino analyze users alice_dev bob_backend --days 30 --format json

# Export to file
pepino analyze users --limit 20 --output top_users.csv
```

#### Channel Analysis
```bash
# Single channel deep dive
pepino analyze channels --channel general --days 7

# All channels overview
pepino analyze channels --format json --output channels_report.json

# Specific metrics
pepino analyze channels --channel dev --include-charts
```

#### Topic Analysis
```bash
# Channel topics
pepino analyze topics --channel general --days 30

# Server-wide topics
pepino analyze topics --days 7 --format json
```

### **Data Management Commands**

#### Sync Operations
```bash
# Check data status
pepino sync status

# Smart sync (only if needed)
pepino sync run

# Force full sync
pepino sync run --force

# Sync with progress
pepino sync run --progress
```

#### Data Exploration
```bash
# List all channels
pepino list channels

# List users with activity
pepino list users --format json

# Export data
pepino list channels --format csv --output channels.csv
```

### **Utility Commands**

#### Performance & Testing
```bash
# Test database connection
pepino test database

# Performance benchmarks
pepino test performance

# Validate configuration
pepino test config
```

## ⚙️ Configuration Guide

### **Environment Variables**

#### **Required Settings**
```bash
# Discord bot token (required)
DISCORD_TOKEN=your_bot_token_here
```

#### **Database Configuration**
```bash
# Database location (default: data/discord_messages.db)
DATABASE_URL=sqlite:///data/discord_messages.db

# For PostgreSQL (advanced)
DATABASE_URL=postgresql://user:pass@localhost/pepino
```

#### **Logging & Debugging**
```bash
# Log level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO

# Log file location
LOG_FILE=logs/pepino.log

# Enable debug mode
DEBUG=true
```

#### **Analysis Settings**
```bash
# Maximum messages to analyze (performance tuning)
MAX_MESSAGES=10000

# Chart quality (DPI)
CHART_DPI=300

# Analysis timeout (seconds)
ANALYSIS_TIMEOUT=300
```

#### **Bot Behavior**
```bash
# Command prefix for text commands (if enabled)
COMMAND_PREFIX=!

# Enable/disable specific features
ENABLE_CHARTS=true
ENABLE_TOPICS=true
ENABLE_SENTIMENT=false
```

### **Advanced Configuration**

#### **Performance Tuning**
```bash
# Database connection pool
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10

# Async workers
WORKER_THREADS=4

# Memory limits
MAX_MEMORY_MB=1024
```

#### **Security & Privacy**
```bash
# Data retention (days)
DATA_RETENTION_DAYS=90

# Anonymize user data
ANONYMIZE_USERS=false

# Rate limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600
```

## 🏗️ Architecture & Design

### **System Architecture**

Pepino follows modern Python best practices with a clean, modular architecture:

```
pepino/
├── analysis/           # Core analytics engine
│   ├── analyzers/     # User, channel, topic analyzers
│   ├── models/        # Data models and responses
│   └── utils/         # Helper functions
├── data/              # Data access layer
│   ├── repositories/  # Database operations
│   ├── models/        # Database models
│   └── database/      # Connection management
├── discord/           # Discord bot interface
│   ├── commands/      # Slash commands
│   ├── sync/          # Data synchronization
│   └── extractors/    # Message extraction
├── cli/               # Command-line interface
└── templates/         # Output formatting
```

### **Key Design Principles**

#### **Repository Pattern**
- Clean separation between business logic and data access
- Easy testing with mock repositories
- Consistent data operations across interfaces

#### **Async-First Design**
- Non-blocking operations for better performance
- Concurrent analysis when possible
- Responsive Discord bot interactions

#### **Type Safety**
- Full type hints throughout codebase
- Pydantic models for data validation
- mypy compliance for catching errors early

#### **Modular Analytics**
- Pluggable analyzer system
- Easy to add new analysis types
- Consistent interfaces across analyzers

### **Data Flow**

1. **Data Collection**: Discord messages → Database
2. **Analysis Request**: User command → Analysis engine
3. **Processing**: Raw data → Insights (using appropriate analyzer)
4. **Formatting**: Results → Human-readable output (via templates)
5. **Delivery**: Formatted output → Discord/CLI

## 🧪 Development & Testing

### **Development Setup**
```bash
# Complete development environment
make dev-setup

# Install dependencies only
poetry install

# Activate virtual environment
poetry shell
```

### **Code Quality**
```bash
# Format code
make format

# Lint code
make lint

# Type checking
make type-check

# All quality checks
make quality
```

### **Testing**
```bash
# Run all tests
make test

# Tests with coverage
make test-cov

# Fast tests only
make test-fast

# Integration tests
make test-integration
```

### **Development Workflow**
```bash
# Complete development cycle
make dev        # Format + lint + fast tests

# Before committing
make quality    # All quality checks
make test-cov   # Full test suite with coverage
```

## 📖 Use Cases & Examples

### **Community Management**

**Scenario**: You run a Discord server for a programming community

**Questions Pepino Answers:**
- Who are your most helpful members? (User analytics)
- Which channels need more moderation attention? (Channel health)
- What topics are trending in your community? (Topic analysis)
- When should you schedule events for maximum attendance? (Activity patterns)

**Workflow:**
```bash
# Weekly community health check
/pepino_server_analytics overview

# Identify helpful members for recognition
/pepino_server_analytics top_users --days 7

# Check if #help channel is working well
/pepino_channel_analytics overview --channel help

# See what people are discussing lately
/pepino_channel_analytics topics --channel general --days 7
```

### **Team Management**

**Scenario**: You manage a development team using Discord

**Questions Pepino Answers:**
- Who are the subject matter experts in different areas? (User topic analysis)
- Are team members collaborating effectively? (Reply rates, interaction patterns)
- Which channels are most/least active? (Channel comparison)
- What's the team's communication rhythm? (Activity patterns)

**Workflow:**
```bash
# Monthly team review
pepino analyze users --format json --output team_analysis.json

# Check project channel health
/pepino_channel_analytics overview --channel project-alpha

# Find expertise areas
pepino analyze topics --channel development --days 30
```

### **Research & Analytics**

**Scenario**: You're studying online community behavior

**Questions Pepino Answers:**
- How do conversation topics evolve over time? (Temporal topic analysis)
- What makes some channels more engaging than others? (Engagement metrics)
- How do user participation patterns differ? (User behavior analysis)
- What are the characteristics of healthy online communities? (Health metrics)

**Workflow:**
```bash
# Export data for external analysis
pepino list users --format csv --output users.csv
pepino list channels --format json --output channels.json

# Analyze specific time periods
pepino analyze channels --days 7 --format json
pepino analyze topics --days 30 --format json
```

## 🚀 Advanced Features

### **Smart Topic Analysis**

Pepino automatically chooses the best analysis method for your content:

- **BERTopic Neural Analysis**: For in-depth technical discussions, long-form content
- **Discord-Optimized Analysis**: For casual conversations, short messages, social chat

**Example: Technical Channel**
```
🧠 Neural Topics Discovered:
1. Kubernetes Deployment Strategies (45 messages, 23.4% relevance)
2. React Performance Optimization (32 messages, 16.7% relevance)
3. Database Migration Patterns (28 messages, 14.6% relevance)
```

**Example: Casual Channel**
```
💬 Discussion Themes Identified:
1. Meeting & Scheduling (403 mentions, 61.5% of messages)
2. Tool Discussions (156 mentions, 10.7% of messages)
3. Help & Questions (98 mentions, 6.7% of messages)
```

### **Interactive Charts**

Visual analytics embedded directly in Discord:

- **User Activity Timelines**: See when users are most active
- **Channel Growth Charts**: Track channel health over time
- **Topic Trend Visualizations**: Watch how interests evolve
- **Engagement Heatmaps**: Find peak activity periods

### **Export & Integration**

Multiple output formats for different use cases:

```bash
# JSON for programmatic use
pepino analyze users --format json

# CSV for spreadsheet analysis
pepino list channels --format csv

# Human-readable reports
pepino analyze channels --format txt
```

## 🔧 Troubleshooting

### **Common Issues**

#### **Bot Not Responding**
1. Check bot permissions in Discord server
2. Verify `DISCORD_TOKEN` in `.env` file
3. Ensure bot has "Message Content Intent" enabled
4. Check logs: `tail -f logs/pepino.log`

#### **No Data Available**
1. Run initial sync: `pepino sync run`
2. Check database location: `ls -la data/`
3. Verify bot can read message history
4. Check date ranges in commands

#### **Analysis Takes Too Long**
1. Reduce analysis period: `--days 7` instead of all time
2. Limit message count: `MAX_MESSAGES=5000` in `.env`
3. Use faster analysis: disable charts temporarily
4. Check system resources: `top` or `htop`

#### **Charts Not Generating**
1. Install chart dependencies: `poetry install --extras charts`
2. Check `ENABLE_CHARTS=true` in `.env`
3. Verify write permissions in temporary directory
4. Check matplotlib backend: `export MPLBACKEND=Agg`

### **Performance Optimization**

#### **For Large Servers**
```bash
# Optimize database
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

# Limit analysis scope
MAX_MESSAGES=10000
ANALYSIS_TIMEOUT=300

# Use faster sync
pepino sync run --incremental
```

#### **For Memory Constraints**
```bash
# Reduce memory usage
MAX_MEMORY_MB=512
WORKER_THREADS=2
DB_POOL_SIZE=3
```

## 📚 Additional Resources

### **Documentation**
- **[Operations Guide](docs/operations.md)** - Complete setup and configuration
- **[Bot Operations](docs/bot_operations.md)** - Discord bot management
- **[Architecture](docs/architecture.md)** - Technical implementation details
- **[Testing Guide](docs/testing.md)** - Development and testing procedures

### **Community & Support**
- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Questions and community help
- **Wiki**: Extended documentation and examples

### **Contributing**
We welcome contributions! See our contributing guidelines for:
- Code style and standards
- Testing requirements
- Documentation standards
- Pull request process

---

## 🎉 What's New

### **Latest Features**
- **🧠 Intelligent Topic Analysis**: AI automatically chooses best analysis method
- **📊 Enhanced Visualizations**: Interactive charts with better design
- **⚡ Smart Sync**: Only syncs when data is stale, with progress tracking
- **🎯 Discord-Optimized Analytics**: Better insights for casual conversations
- **🏗️ Repository Architecture**: Clean, testable, maintainable codebase

### **Recent Improvements**
- **Performance**: 3x faster analysis with async processing
- **Accuracy**: Better topic extraction for Discord conversations
- **Usability**: Autocomplete, pagination, and user-friendly error messages
- **Reliability**: Comprehensive test suite with 95%+ coverage
- **Documentation**: Complete guides for all user types

---

## 🚀 Discord Bot Quick Start Guide

**For colleagues testing the bot for the first time** 🤖

This is a simple guide to try out the Discord bot commands. No setup needed - just type and see what happens!

### 🚀 Getting Started

1. **Find the bot** in your Discord server (look for "Pepino" in the member list)
2. **Type `/pepino`** in any channel to see available commands
3. **Try the commands below** - they all start with `/pepino_`

---

### 📋 Essential Commands to Try

#### 🏠 **Server Overview**
```
/pepino_server_analytics overview
```
**What it shows**: Overall server stats, most active channels and users, activity trends
**Good for**: Getting a bird's eye view of your Discord server

**Example output**: Total messages, active users, busiest channels, growth trends

---

#### 👤 **User Analysis**
```
/pepino_user_analytics username:alice
```
**What it shows**: Deep dive into a specific user's activity
**Good for**: Understanding individual member behavior

**Example output**: 
- How many messages they've sent
- Which channels they use most
- When they're most active
- What topics they discuss

**💡 Tip**: Use autocomplete - start typing a username and Discord will suggest options

---

#### 📊 **Channel Analysis**
```
/pepino_channel_analytics overview channel_name:general
```
**What it shows**: Complete health check of a specific channel
**Good for**: Understanding how well a channel is performing

**Example output**:
- Message volume and activity level
- Top contributors
- Response times
- Engagement metrics
- Visual charts (when available)

---

#### 🎯 **Topic Analysis**
```
/pepino_channel_analytics topics channel_name:general
```
**What it shows**: AI analysis of what people are talking about
**Good for**: Understanding conversation themes and interests

**Example output**:
- Main discussion topics with percentages
- Technology/tool mentions
- Meeting and activity patterns
- Question/help themes

---

#### 📈 **Top Users**
```
/pepino_server_analytics top_users
```
**What it shows**: Most active members in your server
**Good for**: Recognizing valuable community contributors

**Example output**: Ranked list of users with activity stats

---

#### 📝 **Browse Data**
```
/pepino_lists users
/pepino_lists channels
```
**What it shows**: Available users and channels you can analyze
**Good for**: Finding usernames/channels to use in other commands

---

### 🎨 What to Expect

#### **📊 Charts & Visuals**
Some commands include charts showing:
- Daily activity patterns
- User activity timelines
- Channel growth trends

#### **🤖 Smart Analysis**
The bot automatically chooses the best analysis method:
- **Advanced AI** for technical discussions
- **Pattern recognition** for casual conversations

#### **⏱️ Response Times**
- Simple commands: Instant
- Analysis commands: 5-30 seconds
- Complex analysis: Up to 1 minute

---

### 💡 Pro Tips

#### **🔍 Use Autocomplete**
- Start typing channel/user names and Discord will suggest options
- Much easier than typing full names

#### **📅 Try Different Time Periods**
```
/pepino_user_analytics username:alice days:7
/pepino_channel_analytics overview channel_name:general days:30
```

#### **🎯 Start Simple**
1. Try `/pepino_server_analytics overview` first
2. Then pick a channel: `/pepino_channel_analytics overview`
3. Then try a user: `/pepino_user_analytics`

#### **📱 Works Everywhere**
- Any channel (the bot will analyze the right data)
- Desktop and mobile Discord
- DMs with the bot

---

### 🤔 Common Questions

**Q: "No data found" - what's wrong?**
A: The bot needs to sync data first. Ask the admin to run a sync, or try a different channel/user.

**Q: Analysis is taking forever?**
A: Large servers take longer. Try adding `days:7` to limit the analysis period.

**Q: Can I break anything?**
A: Nope! These are read-only commands. You're just viewing data, not changing anything.

**Q: Commands not showing up?**
A: Make sure you type `/pepino_` (with underscore) and the bot has proper permissions.

---

### 🎯 Quick Command Cheat Sheet

| What you want to know | Command to use |
|----------------------|----------------|
| "How's our server doing overall?" | `/pepino_server_analytics overview` |
| "Who are our most active members?" | `/pepino_server_analytics top_users` |
| "How's the #general channel?" | `/pepino_channel_analytics overview channel_name:general` |
| "What do people talk about in #dev?" | `/pepino_channel_analytics topics channel_name:dev` |
| "How active is Alice?" | `/pepino_user_analytics username:alice` |
| "What channels/users can I analyze?" | `/pepino_lists channels` or `/pepino_lists users` |

---

### 🚀 Have Fun Exploring!

The bot is designed to be intuitive - just try commands and see what insights you discover about your Discord community!

**Questions?** Ask in the channel or DM the person who set up the bot.

---

*This bot analyzes your Discord conversations to provide insights about community activity, user behavior, and discussion topics. All analysis is based on message history the bot can access.*

---

Created by **Jose Cordovilla** with vibecoding tools like Github Copilot and Cursor and a lot of dedication. Jose is Volunteer Network Architect at MIT Professional Education's GenAI Global Community
