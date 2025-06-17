# Discord Analysis Bot 🤖

A powerful Discord bot that analyzes your server's message data and provides comprehensive statistics through easy-to-use slash commands.

## ✨ Features

- 📊 **Channel Analysis** - Detailed insights about specific channels
- � **User Analysis** - User activity patterns and statistics  
- 🧠 **Topic Analysis** - Discussion topics using advanced NLP
- � **Statistical Reports** - Word frequency, user stats, temporal patterns
- � **Smart Autocomplete** - Find channels and users easily
- 📱 **Modern Interface** - All commands use Discord's slash command system

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone and navigate to directory
git clone <your-repo-url>
cd pepino

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or venv\Scripts\activate  # Windows

# Install dependencies  
pip install -r requirements.txt
```

### 2. Configure Discord Bot
1. Create a Discord bot at [Discord Developer Portal](https://discord.com/developers/applications)
2. Copy the bot token to `.env`:
   ```env
   DISCORD_TOKEN=your_discord_bot_token_here
   ```
3. Invite bot to your server with permissions: `Read Messages`, `Read Message History`, `View Channels`

### 3. Fetch Messages
```bash
python fetch_messages.py
```

### 4. Start Analysis Bot
```bash
python bot.py
```

## 💬 Bot Commands

All commands use Discord's modern slash command system with autocomplete:

### 🎯 **Main Analysis Commands**
- `/channel_analysis` - Analyze a specific channel (with autocomplete)
- `/user_analysis` - Analyze a specific user (with autocomplete)  
- `/topics_analysis` - Analyze discussion topics, optionally by channel

### 📊 **Statistical Commands**
- `/wordfreq_analysis` - Most common words across all messages
- `/userstats_analysis` - User activity statistics 
- `/temporal_analysis` - Activity patterns by time

### 📋 **Utility Commands**
- `/list_users` - Show all available users
- `/list_channels` - Show all available channels
- `/help_analysis` - Show all commands

## 🛠️ File Structure

```
pepino/
├── bot.py              # Main Discord bot
├── bot_commands.py     # Slash command implementations  
├── analysis.py         # Statistical analysis engine
├── fetch_messages.py   # Message fetching utility
├── requirements.txt    # Dependencies
├── .env               # Bot token (create this)
├── discord_messages.db # SQLite database (auto-created)
└── archive/           # Test files and backups
```

## 🔧 Configuration

**Environment Variables (.env)**
```env
DISCORD_TOKEN=your_discord_bot_token_here
```

**Database**: SQLite database (`discord_messages.db`) stores all message data with full metadata.

## 🎨 Example Outputs

**Channel Analysis:**
```
📊 Channel Analysis: #general-chat

Basic Statistics:
• Total Messages: 1,234
• Unique Users: 87
• Average Message Length: 64.2 characters
• Most Active User: John Doe (123 messages)

Top Contributors:
• John Doe: 123 messages
• Jane Smith: 98 messages
• Bob Wilson: 76 messages
```

**User Analysis:**
```
👤 User Analysis: John Doe

General Statistics:
• Total Messages: 456
• Active Channels: 8
• Average Message Length: 72.1 characters  
• Most Active Channel: #general-chat (123 messages)

Most Used Words:
• awesome: 45 times
• project: 32 times
• thanks: 28 times
```

## 🚨 Troubleshooting

**Bot doesn't respond:**
- Check bot permissions in Discord server
- Verify bot token in `.env` file
- Ensure bot is online and invited to server

**No data found:**
- Run `python fetch_messages.py` first
- Check database file exists: `discord_messages.db`
- Verify bot can access channels

**Autocomplete not working:**
- Make sure database has data
- Check that users/channels exist in database
- Try `/list_users` or `/list_channels` commands

## 🔒 Security

- Keep your bot token secure in `.env` file
- Never commit `.env` or database files to git
- Database contains message data - handle appropriately

---

**Need help?** Use `/help_analysis` in Discord for command reference!

---

**Author:**  
Jose Cordovilla  
GenAI Global Network Architect