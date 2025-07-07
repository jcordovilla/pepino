# Pepino - Discord Message Fetcher

A lightweight Python application that connects to Discord servers and fetches messages, storing them in a SQLite database for analysis and archival purposes.

## 🚀 Features

- 🤖 **Discord Integration** - Fetches messages from Discord servers using bot token
- 💾 **SQLite Storage** - Stores messages in local database with full metadata
- 📊 **Rich Data Capture** - Includes emojis, reactions, attachments, and user presence
- 🔄 **Incremental Sync** - Only fetches new messages on subsequent runs
- 📝 **Comprehensive Logging** - Detailed sync progress and error handling
- 🛡️ **Error Resilient** - Gracefully handles permission errors and API limits

## 📋 Prerequisites

- **Python 3.8+** (recommended 3.9 or higher)
- **Discord Bot Token** with appropriate permissions
- **Server Access** to Discord server(s) you want to fetch messages from

## 🛠️ Quick Setup

1. **Clone and Setup**
   ```bash
   git clone https://github.com/jcordovilla/pepino.git
   cd pepino
   ```

2. **Create Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # macOS/Linux
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment**
   ```bash
   # Create .env file
   echo "DISCORD_TOKEN=your_bot_token_here" > .env
   ```

## 🤖 Discord Bot Setup

### 1. Create Discord Application
- Visit [Discord Developer Portal](https://discord.com/developers/applications)
- Click **"New Application"** and give it a name
- Navigate to **"Bot"** section → **"Add Bot"**
- Copy the bot token to your `.env` file

### 2. Configure Bot Permissions
Enable these intents in the Bot section:
- ✅ **Message Content Intent** (required for reading messages)
- ✅ **Server Members Intent** (for user data)

### 3. Invite Bot to Server
- Go to **"OAuth2"** → **"URL Generator"**
- Select scopes: `bot`
- Select permissions:
  - `View Channels`
  - `Read Message History` 
  - `Read Messages`
- Use generated URL to invite bot to your server

## 🚀 Usage

**Start the message fetcher:**
```bash
python fetch_messages.py
```

### What happens during sync:
1. 🔗 **Connect** to Discord using your bot token
2. 🗄️ **Initialize** SQLite database (`data/discord_messages.db`)
3. 📥 **Fetch** messages from all accessible channels
4. 💾 **Store** messages with complete metadata
5. 📊 **Log** sync progress and statistics

### Sample output:
```
🚀 Starting Discord message sync...
📊 Found 3 guilds, 15 channels
📥 Syncing guild: My Server (123 members)
  ✅ #general: 1,234 messages synced
  ✅ #announcements: 56 messages synced
✨ Sync complete! Total: 1,290 new messages
```

## 🗃️ Database Schema

### `messages` table
**Complete message data including:**
- Message content, timestamps, and IDs
- Author details (ID, name, display name, avatar)
- Guild and channel information
- Attachments, embeds, and reactions
- Emoji usage and user mentions
- Message references (replies, threads)

### `sync_logs` table  
**Synchronization tracking:**
- Sync timestamps and duration
- Guilds and channels processed
- Message counts and statistics
- Error logs and retry attempts

## 📁 Project Structure

```
pepino/
├── fetch_messages.py      # 🎯 Main application
├── requirements.txt       # 📦 Dependencies (discord.py, python-dotenv)
├── .env                  # 🔐 Environment variables (not tracked)
├── .gitignore           # 🚫 Git ignore rules
├── data/                # 🗄️ Data directory
│   └── discord_messages.db  # 💾 SQLite database (not tracked)
└── README.md            # 📖 This documentation
```

## ⚙️ Configuration

### Environment Variables
```bash
# Required
DISCORD_TOKEN=your_discord_bot_token_here

# Optional  
GUILD_ID=specific_guild_id_to_sync_only  # Sync single server
LOG_LEVEL=INFO                           # DEBUG, INFO, WARNING, ERROR
```

### Customization Options
- **Database location**: Modify `db_path` in `fetch_messages.py`
- **Message limits**: Adjust `limit` parameter in channel sync
- **Emoji patterns**: Customize regex in `extract_emojis()` function
- **Sync intervals**: Configure incremental sync behavior

## 🔧 Troubleshooting

### Common Issues

**🚫 Permission Errors**
- Verify bot has required permissions in Discord server
- Check that bot can access target channels
- Ensure bot is still in the server (not kicked)

**🔑 Token Issues**  
- Confirm Discord bot token is correct in `.env`
- Check if token has expired or been regenerated
- Verify `.env` file is in the project root directory

**🗄️ Database Issues**
- Delete `data/discord_messages.db` to start fresh
- Check file/directory permissions
- Ensure sufficient disk space available

**📡 Network/API Issues**
- Discord API rate limits (automatically handled)
- Check internet connection stability
- Verify Discord service status

### Getting Help
- Check the logs for detailed error messages
- Enable DEBUG logging: `LOG_LEVEL=DEBUG` in `.env`
- Review Discord bot permissions in server settings

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🔒 Security & Privacy

### Important Notes
- 🚨 **Never commit** your `.env` file or Discord tokens to version control
- 🔐 **Keep bot tokens secure** and regenerate if compromised  
- 💾 **Database contains sensitive data** - handle Discord message data appropriately
- 🛡️ **Follow Discord ToS** - ensure compliance with Discord's Terms of Service
- 🔒 **Local storage only** - messages are stored locally, not transmitted elsewhere

### Data Handling
- Messages are stored locally in SQLite database
- No data is sent to external services
- Database should be treated as sensitive/private data
- Consider encryption for additional security if needed
