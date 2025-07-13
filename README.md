# Pepino - Advanced Discord Analytics Platform

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/dependency-poetry-blue.svg)](https://python-poetry.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Pepino** transforms your Discord conversations into actionable insights. Whether you're managing a community, running a team, or analyzing social interactions, Pepino reveals who's active, what they're talking about, and when engagement peaks.

> **🚀 Want to see it in action?** Jump to the [Quick Start](#-discord-bot-quick-start-guide) and try the commands right now!

## 🎯 What Pepino Reveals About Your Community

### **Instant Insights**
- **Who are your most active members?** Top contributors with engagement metrics
- **Which channels drive conversation?** Channel health and participation rates  
- **What topics resonate?** AI-powered topic extraction from discussions
- **When is your server busiest?** Activity patterns and peak times
- **How healthy is your community?** Server-wide metrics and growth trends

### **Real Example Output**
```
📊 Top Contributors (Last 30 Days)
1. Alice Dev: 247 messages, 67% reply rate
   Most Active: #development (89 messages)
   Peak Time: Tuesday 2-4 PM

2. Bob Backend: 189 messages, 45% reply rate  
   Most Active: #architecture (67 messages)
   Top Topics: deployment, testing, ci/cd

📈 Channel Health: #general
• 1,247 messages from 45 users (Very active)
• 78% engagement rate (High participation)
• Response time: 1.2 hours average
• Trending topics: meetings, planning, tools
```

## 🎮 Multiple Ways to Get Insights

### **Discord Bot (Instant Analysis)**
Perfect for **real-time insights** without leaving Discord.

**Try these commands right now:**
```
/channel_analysis channel_name:general    # Complete channel health check
/top_contributors limit:10                # Most active members
/detailed_user_analysis username:alice    # Deep user insights
/server_overview                          # Server-wide statistics
/database_stats                           # Data health report
```

**Why use the bot?**
- Instant analysis without leaving Discord
- Visual charts and graphs embedded in responses
- Autocomplete for easy channel/user selection
- Perfect for moderators and community managers

### **Command Line (Automation & Export)**
Perfect for **automation**, **scripting**, and **detailed analysis**.

**Key commands:**
```bash
# Quick insights
pepino analyze top-contributors --limit 10
pepino analyze pulsecheck --channel general
pepino analyze database-stats

# Export data for external analysis
pepino data export messages --format csv
pepino list channels --format json

# Sync fresh data
pepino data sync --force
```

**Why use the CLI?**
- Batch processing and automation
- Export data in multiple formats (JSON, CSV, Excel)
- Integration with scripts and CI/CD pipelines
- Detailed control over analysis parameters

## 🚀 Discord Bot Quick Start Guide

**For immediate testing - no setup needed!** 🤖

### 🚀 Getting Started
1. **Find the bot** in your Discord server (look for "Pepino" in member list)
2. **Type `/`** in any channel to see available commands
3. **Try these commands** - they all start with `/`

### 📋 Essential Commands to Try

#### 🏠 **Server Overview**
```
/server_overview
```
**What it shows**: Overall server stats, most active channels and users, activity trends
**Value**: Get a bird's eye view of your Discord community health

#### 👤 **User Analysis**
```
/detailed_user_analysis username:alice
```
**What it shows**: Deep dive into a specific user's activity patterns
**Value**: Understand individual member behavior and engagement

#### 📊 **Channel Analysis**
```
/channel_analysis channel_name:general
```
**What it shows**: Complete health check of a specific channel
**Value**: Understand how well a channel is performing and what drives engagement

#### 🎯 **Topic Analysis**
```
/detailed_topic_analysis channel_name:general
```
**What it shows**: AI analysis of what people are talking about
**Value**: Understand conversation themes and community interests

#### 📈 **Top Contributors**
```
/top_contributors limit:10
```
**What it shows**: Most active members in your server
**Value**: Recognize valuable community contributors and identify leaders

### 🎨 What to Expect

#### **📊 Real Data**
- Actual message counts and engagement metrics
- Real user activity patterns and channel health
- Live data from your Discord server

#### **🤖 Smart Analysis**
- AI automatically chooses the best analysis method
- Advanced AI for technical discussions
- Pattern recognition for casual conversations

#### **⏱️ Analysis Results**
- Simple commands: Quick response
- Analysis commands: Varies by data size
- Complex analysis: Depends on analysis complexity

### 💡 Pro Tips

#### **🔍 Use Autocomplete**
- Start typing channel/user names and Discord will suggest options
- Much easier than typing full names

#### **📅 Try Different Time Periods**
```
/detailed_user_analysis username:alice days_back:7
/channel_analysis channel_name:general days_back:30
```

#### **🎯 Start Simple**
1. Try `/server_overview` first
2. Then pick a channel: `/channel_analysis`
3. Then try a user: `/detailed_user_analysis`

---

## 💻 CLI Quick Start

### **Installation**
```bash
# Clone and setup
git clone https://github.com/your-repo/pepino.git
cd pepino
make dev-setup

# OR install via pip (when available)
pip install pepino-discord
```

### **Configuration**
```bash
# Copy environment template
cp env.example .env

# Edit .env file
DISCORD_TOKEN=your_bot_token_here
```

### **First Run**
```bash
# Start the bot
pepino start

# Or run analysis directly
pepino analyze top-contributors --limit 5
pepino analyze database-stats
```

## 🏗️ Architecture Highlights

### **Modular Design**
- **Specialized Services**: Each analysis type has its own service
- **Template Engine**: Consistent output formatting across interfaces
- **Repository Pattern**: Clean data access and easy testing
- **Async-First**: Non-blocking operations for better performance

### **Data Flow**
1. **Discord Messages** → Database (via sync)
2. **Analysis Request** → Specialized Service
3. **Raw Data** → Insights (using appropriate analyzer)
4. **Results** → Human-readable output (via templates)
5. **Delivery** → Discord/CLI

## 🎯 Use Cases & Value

### **Community Management**
**Questions Pepino Answers:**
- Who are your most helpful members? (User analytics)
- Which channels need more attention? (Channel health)
- What topics are trending? (Topic analysis)
- When should you schedule events? (Activity patterns)

**Workflow:**
```bash
# Weekly community health check
/server_overview

# Identify helpful members for recognition
/top_contributors limit:10

# Check if #help channel is working well
/channel_analysis channel_name:help
```

### **Team Management**
**Questions Pepino Answers:**
- Who are the subject matter experts? (User topic analysis)
- Are team members collaborating effectively? (Reply rates, interactions)
- Which channels are most/least active? (Channel comparison)
- What's the team's communication rhythm? (Activity patterns)

**Workflow:**
```bash
# Monthly team review
pepino analyze top-contributors --limit 20

# Check project channel health
/channel_analysis channel_name:project-alpha

# Find expertise areas
/detailed_topic_analysis channel_name:development
```

### **Research & Analytics**
**Questions Pepino Answers:**
- How do conversation topics evolve? (Temporal topic analysis)
- What makes channels more engaging? (Engagement metrics)
- How do user participation patterns differ? (User behavior analysis)

**Workflow:**
```bash
# Export data for external analysis
pepino data export messages --format csv
pepino list channels --format json

# Analyze specific time periods
pepino analyze pulsecheck --channel general --days 7
```

## 🚀 Advanced Features

### **Smart Topic Analysis**
Pepino automatically chooses the best analysis method:
- **Advanced AI**: For technical discussions, long-form content
- **Pattern Recognition**: For casual conversations, social chat

### **Interactive Charts**
Visual analytics embedded directly in Discord:
- User activity timelines
- Channel growth charts
- Topic trend visualizations
- Engagement heatmaps

### **Export & Integration**
Multiple output formats for different use cases:
```bash
# JSON for programmatic use
pepino analyze top-contributors --format json

# CSV for spreadsheet analysis
pepino list channels --format csv

# Human-readable reports
pepino analyze pulsecheck --format txt
```

## ⚙️ Quick Configuration

### **Required Settings**
```bash
# Discord bot token (required)
DISCORD_TOKEN=your_bot_token_here

# Database location (default: data/discord_messages.db)
DATABASE_URL=sqlite:///data/discord_messages.db

# Log level
LOG_LEVEL=INFO
```

### **Discord Bot Setup**
1. **Create Discord Application** at [Discord Developer Portal](https://discord.com/developers/applications)
2. **Add Bot** and enable Message Content Intent
3. **Generate Token** and add to `.env` file
4. **Invite Bot** to your server with proper permissions

## 📚 Additional Resources

### **Documentation**
- **[Operations Guide](docs/operations.md)** - Complete setup and configuration
- **[Development Guide](docs/development.md)** - Architecture and technical details
- **[Testing Guide](docs/testing.md)** - Testing strategy and guidelines

### **Community & Support**
- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Questions and community help

---

## 🎉 What's New

### **Latest Features**
- **🧠 Intelligent Topic Analysis**: AI automatically chooses best analysis method
- **📊 Enhanced Visualizations**: Interactive charts with better design
- **⚡ Smart Sync**: Only syncs when data is stale, with progress tracking
- **🎯 Discord-Optimized Analytics**: Better insights for casual conversations
- **🏗️ Modular Architecture**: Clean, testable, maintainable codebase

### **Recent Improvements**
- **Accuracy**: Better topic extraction for Discord conversations
- **Usability**: Autocomplete, pagination, and user-friendly error messages
- **Reliability**: Comprehensive test suite with modular architecture

---

**Ready to unlock insights from your Discord community?** Start with the [Discord Bot Quick Start](#-discord-bot-quick-start-guide) and see what Pepino can reveal about your server! 🚀

*This bot analyzes your Discord conversations to provide insights about community activity, user behavior, and discussion topics. All analysis is based on message history the bot can access.*
<<<<<<< HEAD

---

Created by **Jose Cordovilla** with vibecoding tools like Github Copilot and Cursor and a lot of dedication. Jose is Volunteer Network Architect at MIT Professional Education's GenAI Global Community
=======
>>>>>>> 1ca9acd (feat: implement clean architecture with comprehensive modularization)
