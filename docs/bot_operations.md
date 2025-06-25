# Discord Bot Advanced Operations

This guide covers advanced Discord bot operations including sync workflows, troubleshooting, and production deployment.

**📋 For basic bot setup, see [operations.md](operations.md)**

## 🔄 Sync-Enabled Analysis Workflow

### How Round-Trip Sync Works

The bot supports **sync-then-analyze workflows** using a single Discord token:

1. 🔍 **Data Freshness Check**
   - Checks last sync timestamp
   - Determines if sync is needed (configurable threshold)

2. 🔄 **Intelligent Sync** (if needed)
   - "Syncing Discord data..." status message
   - Incremental sync (only new data)
   - 5-minute timeout protection
   - Success/failure feedback

3. 📊 **Analysis Execution**
   - Database reconnection
   - Requested analysis with fresh data
   - Chart generation (if enabled)
   - Formatted results delivery

### Key Benefits

✅ **Always Fresh Data**: Analysis uses the most recent Discord data
✅ **User Control**: Choose when to sync vs. use existing data  
✅ **Intelligent Defaults**: Only syncs when data is actually stale
✅ **Transparent Process**: Clear feedback at each step
✅ **Error Resilience**: Continues with existing data if sync fails
✅ **Single Token**: No need for separate bot and sync tokens

### Best Practices

**For Regular Use:**
- Use `/sync_and_analyze` for important analysis
- Use standard commands for quick checks on existing data
- Check `/sync_status` to understand data freshness

**For Real-Time Analysis:**
- Set `force_sync:true` for guaranteed fresh data
- Use during active periods for real-time insights
- Consider server load during busy periods

### Sync Command Examples
```
/sync_and_analyze analysis_type:User Analysis target:john_doe
/sync_and_analyze analysis_type:Channel Analysis target:general
/sync_and_analyze analysis_type:Topics Analysis target:dev-chat
/sync_and_analyze analysis_type:Activity Trends
/sync_and_analyze analysis_type:Top Users force_sync:true
```

## 🎨 Chart Features

### Chart Types Generated
- **User Activity**: Daily message timeline with trend analysis
- **Channel Activity**: Top contributor pie charts with percentages
- **Activity Trends**: Server-wide patterns over time
- **Topic Visualization**: Topic distribution and trends

### Chart Customization
- **Toggle Charts**: Use `include_chart:false` to disable chart generation
- **Automatic Optimization**: Charts automatically sized for Discord
- **Multiple Formats**: PNG output optimized for Discord display
- **Smart Caching**: Charts cached temporarily to improve performance

## 🛠️ Troubleshooting

### Common Issues

**Bot Not Responding:**
```bash
# Check if bot is running
ps aux | grep pepino

# Check logs
pepino start --debug

# Verify token
echo $DISCORD_TOKEN
```

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

**Performance Issues:**
- Database too large: Consider data cleanup
- Network latency: Increase sync timeout
- Memory usage: Monitor during peak times

### Permission Errors

**Missing Read History:**
- Bot needs "Read Message History" permission
- Check channel-specific permissions
- Verify bot role hierarchy

**Command Not Found:**
- Ensure bot has "Use Slash Commands" permission
- Try re-inviting bot with updated permissions
- Check if commands are registered (restart bot)

### Data Sync Issues

**Stale Data Warnings:**
```bash
# Check last sync
pepino sync status

# Force fresh sync
pepino sync run --full --clear

# Incremental update
pepino sync run --force
```

**Partial Sync Results:**
- Check for network interruptions
- Verify all required channels are accessible
- Review sync logs for errors

## 🔧 Advanced Configuration

### Sync Settings Customization

```bash
# Custom sync thresholds
export AUTO_SYNC_THRESHOLD_HOURS=2  # Sync if data older than 2 hours
export SYNC_TIMEOUT_SECONDS=600     # 10-minute timeout

# Disable auto-sync (manual only)
export ALLOW_FORCE_SYNC=false

# Disable progress feedback
export SYNC_FEEDBACK_ENABLED=false
```

### Database Optimization

```bash
# Database maintenance
pepino analyze temporal --output backup.json
sqlite3 discord_messages.db "VACUUM;"
sqlite3 discord_messages.db "ANALYZE;"
```

### Production Settings

```bash
# Production environment
export DEBUG=false
export LOG_LEVEL=WARNING
export DB_PATH=/opt/pepino/production.db

# Systemd service
pepino start --prefix !prod
```

## 📈 Monitoring and Analytics

### Health Checks
- Monitor sync status regularly
- Track database growth
- Monitor command response times
- Check error rates in logs

### Performance Metrics
- Sync duration trends
- Database query performance  
- Command usage patterns
- Chart generation times

## 🔒 Security Considerations

### Token Security
- Never commit tokens to version control
- Use environment variables or secure vaults
- Rotate tokens periodically
- Monitor for unauthorized usage

### Data Privacy
- Review what data is collected
- Implement data retention policies
- Consider GDPR compliance for EU users
- Secure database access

### Access Control
- Limit bot permissions to minimum required
- Use role-based channel permissions
- Monitor command usage logs
- Implement rate limiting if needed

## 🚀 Deployment Options

### Local Development
```bash
# Development setup
cp .env.example .env
# Edit .env with your settings
pepino start --debug
```

### Production Deployment

**Systemd Service** (`/etc/systemd/system/pepino-bot.service`):
```ini
[Unit]
Description=Pepino Discord Analysis Bot
After=network.target

[Service]
Type=simple
User=pepino
WorkingDirectory=/opt/pepino
Environment=DISCORD_TOKEN=your_token_here
Environment=DB_PATH=/opt/pepino/production.db
ExecStart=/opt/pepino/venv/bin/python -m pepino start
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Docker Deployment:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install poetry && poetry install --no-dev
CMD ["poetry", "run", "python", "-m", "pepino", "start"]
```

**Process Manager (PM2):**
```bash
pm2 start "poetry run python -m pepino start" --name pepino-bot
pm2 save
pm2 startup
```

## 📋 Command Quick Reference

| Command | Purpose | Sync Support |
|---------|---------|--------------|
| `/user_analysis` | Individual user stats | ❌ (use existing data) |
| `/channel_analysis` | Channel statistics | ❌ (use existing data) |
| `/topics_analysis` | Topic extraction | ❌ (use existing data) |
| `/activity_trends` | Server trends | ❌ (use existing data) |
| `/top_users` | Leaderboards | ❌ (use existing data) |
| `/sync_status` | Check data freshness | ✅ (sync info only) |
| `/sync_and_analyze` | **Fresh data analysis** | ✅ (full round-trip) |
| `/list_users` | Available users | ❌ (use existing data) |
| `/list_channels` | Available channels | ❌ (use existing data) |
| `/help_analysis` | Command help | ❌ (static help) |

**💡 Pro Tip**: Use `/sync_and_analyze` for important analysis when you need the freshest data, and regular commands for quick checks on existing data.

---

This documentation provides comprehensive guidance for operating the Discord bot with the new sync-enabled analysis capabilities. The round-trip workflow ensures users always have access to the freshest data while maintaining the simplicity of single-token operation. 