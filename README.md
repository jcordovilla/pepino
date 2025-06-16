# Discord Message Fetcher

A Python application that connects to Discord servers and fetches messages, storing them in a SQLite database for analysis and archival purposes.

## Features

- 🤖 Fetches messages from Discord servers using a bot token
- 💾 Stores messages in SQLite database with full metadata
- 📊 Captures emojis, reactions, attachments, and user presence
- 🔄 Incremental sync (only fetches new messages)
- 📝 Comprehensive logging and error handling
- 🛡️ Handles permission errors gracefully

## Prerequisites

- Python 3.7 or higher
- Discord bot token
- Access to Discord server(s) you want to fetch messages from

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd pepino
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```bash
   cp .env.example .env  # If example exists, or create manually
   ```
   
   Add your Discord bot token to `.env`:
   ```
   DISCORD_TOKEN=your_discord_bot_token_here
   GUILD_ID=your_guild_id_here              # Optional
   ```

## Discord Bot Setup

1. **Create a Discord Application**
   - Go to [Discord Developer Portal](https://discord.com/developers/applications)
   - Click "New Application" and give it a name
   - Navigate to the "Bot" section
   - Click "Add Bot"
   - Copy the bot token to your `.env` file

2. **Invite the Bot to Your Server**
   - In the Discord Developer Portal, go to "OAuth2" → "URL Generator"
   - Select scopes: `bot`
   - Select bot permissions: `Read Messages`, `Read Message History`, `View Channels`
   - Use the generated URL to invite the bot to your server

3. **Required Bot Permissions**
   - View Channels
   - Read Message History
   - Read Messages/View Message History

## Usage

**Run the message fetcher:**
```bash
python fetch_messages.py
```

The application will:
1. Connect to Discord using your bot token
2. Initialize the SQLite database (`discord_messages.db`)
3. Fetch messages from all accessible channels
4. Store messages with full metadata
5. Log sync progress and any errors

## Database Schema

The SQLite database contains two main tables:

### `messages` table
Stores all Discord messages with fields including:
- Message content, timestamps, and metadata
- Author information (ID, name, avatar, etc.)
- Guild and channel details
- Attachments, embeds, and reactions
- Emoji statistics and mentions

### `sync_logs` table
Tracks synchronization history:
- Sync timestamps
- Guilds and channels processed
- Error logs and statistics

## File Structure

```
pepino/
├── fetch_messages.py     # Main application
├── requirements.txt      # Python dependencies
├── .env                 # Environment variables (not tracked)
├── .gitignore          # Git ignore rules
├── discord_messages.db # SQLite database (not tracked)
└── README.md           # This file
```

## Configuration

- **Database location**: Modify `db_path` parameter in functions (default: `discord_messages.db`)
- **Message limits**: Adjust `limit` parameter in `channel.history()` call
- **Emoji detection**: Customize regex patterns in `extract_emojis()` function

## Troubleshooting

**Permission Errors**
- Ensure your bot has proper permissions in the Discord server
- Check that the bot can access the channels you want to fetch from

**Token Issues**
- Verify your Discord bot token is correct in `.env`
- Make sure the token hasn't expired or been regenerated

**Database Issues**
- Delete `discord_messages.db` to start fresh if needed
- Check file permissions in the project directory

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Security Notes

- Never commit your `.env` file or Discord tokens to version control
- Keep your bot token secure and regenerate if compromised
- The SQLite database may contain sensitive conversation data - handle appropriately

## Bot Analysis Commands

The application includes a Discord bot with comprehensive message analysis capabilities. Use these commands to test all analytical functionalities:

### Basic Analysis Commands

**Word Frequency Analysis**
```
/analyze wordfreq
```

**User Statistics**
```
/analyze userstats
```

**Temporal Activity Patterns**
```
/analyze temporal
```

**Topic Analysis**
```
/analyze topics
```

### Channel Analysis

**Analyze Specific Channels**
```
/analyze channel general-chat
/analyze channel 🦾agent-ops
/analyze channel 🗂agent-ops-resources
/analyze channel 👾discord-managers
```

**List Available Channels**
```
/list_channels
```

### User Analysis

**Search by Display Name**
```
/analyze user Arturo Cuevas
/analyze user Jose Cordovilla
```

**Search by Username**
```
/analyze user arthurcaves
/analyze user julioverne74
```

**Test Fuzzy Matching**
```
/analyze user Arturo
/analyze user Jose
/analyze user arthur
```

**List Available Users**
```
/list_users
```

### Advanced Analysis Commands

**Topic Analysis for Specific Channel**
```
/analyze topics general-chat
/analyze topics 🦾agent-ops
```

**Similar Message Finding (if implemented)**
```
/analyze similar 1234567890
```

**Conversation Analysis (if implemented)**
```
/analyze conversations
```

**Run All Analyses**
```
/analyze runall
```

### Test Cases for Edge Cases

**Non-existent Users**
```
/analyze user NonExistentUser
/analyze user XYZ123
```

**Non-existent Channels**
```
/analyze channel nonexistent-channel
/analyze channel test123
```

**Special Characters in Names**
```
/analyze user José
/analyze channel #general
```

**Case Sensitivity Tests**
```
/analyze user ARTURO CUEVAS
/analyze user arturo cuevas
/analyze channel GENERAL-CHAT
```

**Partial Name Matching**
```
/analyze user Art
/analyze user Cuevas
/analyze channel agent
```

### Bot Command Tests

**Help and Information**
```
/analyze
/analyze help
```

**Error Handling**
```
/analyze invalidcommand
/analyze user
/analyze channel
```

### Performance Tests

**Large Data Analysis**
```
/analyze userstats
/analyze wordfreq
/analyze temporal
```

> **Note**: The user analysis feature supports both display names (e.g., "Jose Cordovilla") and usernames (e.g., "julioverne74") with intelligent fuzzy matching for approximate matches.
