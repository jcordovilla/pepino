# Template Integration Guide

This guide shows how to integrate the new Jinja2 template system into your existing Discord analysis commands.

## 🎯 What We Built

### Templates for All Commands:
- `templates/outputs/discord/channel_analysis.md.j2` → `/channel_analysis` command
- `templates/outputs/discord/user_analysis.md.j2` → `/user_analysis` command  
- `templates/outputs/discord/topic_analysis.md.j2` → `/topics_analysis` command
- `templates/outputs/discord/activity_trends.md.j2` → `/activity_trends` command
- `templates/outputs/discord/top_users.md.j2` → `/top_users` command

### Template Engine:
- `src/pepino/analysis/template_engine.py` - Core template rendering
- `src/pepino/discord/commands/template_message_formatter.py` - Discord-specific formatting
- `src/pepino/discord/commands/template_mixin.py` - Integration layer

## 🔧 Integration Steps

### Step 1: Update Your AnalysisCommands Class

```python
# In src/pepino/discord/commands/analysis.py

from pepino.discord.commands.template_mixin import TemplateFormattingMixin

class AnalysisCommands(commands.Cog, TemplateFormattingMixin):
    """Commands for analyzing Discord messages"""

    def __init__(self, bot):
        self.bot = bot
        self.db_manager = DatabaseManager()
        self.message_formatter = DiscordMessageFormatter()
        
        # Initialize template support
        super().__init__()  # This calls TemplateFormattingMixin.__init__()
        
        # Your existing initialization code...
```

### Step 2: Enable Templates (Gradual Migration)

```python
# In your AnalysisCommands.__init__ method
async def cog_load(self):
    """Initialize the analyzer when the cog is loaded"""
    # Your existing initialization...
    
    # Enable templates (set to False to keep using original formatting)
    self.use_templates = True
    
    logger.info("Analysis commands initialized with template support")
```

### Step 3: Update Individual Commands (Example)

```python
# Before (original hardcoded formatting):
@app_commands.command(name="channel_analysis")
async def channel_analysis(self, interaction, channel: str, include_chart: bool = True):
    # ... analysis logic ...
    
    # Old way:
    formatted_response = self.message_formatter.format_channel_insights(
        channel_insights, chart_data, channel, include_chart
    )

# After (with template support):
@app_commands.command(name="channel_analysis") 
async def channel_analysis(self, interaction, channel: str, include_chart: bool = True):
    # ... same analysis logic ...
    
    # New way (automatically uses templates if self.use_templates = True):
    formatted_response = self.format_channel_insights(
        channel_insights, chart_data, channel, include_chart
    )
    
    # Everything else stays the same!
```

## 🚀 Benefits After Integration

### 1. **Backward Compatibility**
- Existing commands work unchanged
- Switch templates on/off with `self.use_templates = True/False`
- Gradual migration - no big bang changes

### 2. **Same Output Quality**
- Templates produce identical formatting to your current hardcoded version
- All sections, emojis, spacing preserved
- Charts still work the same way

### 3. **Easy Maintenance**
- Change formatting by editing `.j2` files, no code changes
- Add new report types by adding templates
- Consistent styling across all commands

### 4. **Template-Specific Commands** (Optional)
Add new template-only commands:

```python
@app_commands.command(name="server_health_report")
async def server_health_report(self, interaction, days_back: int = 30):
    """Multi-faceted server health report using templates"""
    
    # This would combine multiple analyzers using template configs
    template_executor = TemplateExecutor(self.template_engine, self.analyzers)
    config = await template_executor.execute_config_template(
        'server_health',
        days_back=days_back
    )
    report = await template_executor.execute_composite_report(config)
    
    await interaction.followup.send(report)
```

## 📁 File Structure After Integration

```
src/pepino/
├── analysis/
│   ├── template_engine.py          # NEW: Core template engine
│   └── (existing analyzers...)
├── discord/commands/
│   ├── analysis.py                 # MODIFIED: Add TemplateFormattingMixin
│   ├── template_message_formatter.py  # NEW: Template-based formatting
│   ├── template_mixin.py           # NEW: Integration layer
│   └── message_formatter.py       # EXISTING: Keep for fallback
└── ...

templates/
├── outputs/discord/
│   ├── channel_analysis.md.j2     # NEW: All command templates
│   ├── user_analysis.md.j2
│   ├── topic_analysis.md.j2
│   ├── activity_trends.md.j2
│   └── top_users.md.j2
└── configs/                       # NEW: For composite reports
    └── server_health.yaml.j2
```

## 🧪 Testing Your Integration

Run the integration demo to verify everything works:

```bash
poetry run python examples/template_integration_demo.py
```

This will show:
- ✅ Template output (new way)
- ✅ Hardcoded output (old way) 
- ✅ Side-by-side comparison
- ✅ No hanging issues
- ✅ Proper database cleanup

## 🎯 Next Steps

1. **Start with one command**: Begin with `/channel_analysis` 
2. **Test thoroughly**: Compare template vs hardcoded output
3. **Migrate gradually**: Move one command at a time
4. **Add new features**: Use templates for composite reports
5. **Clean up**: Remove old hardcoded formatting once satisfied

## 🔧 Troubleshooting

**Templates not loading?**
- Check templates directory exists: `templates/outputs/discord/`
- Verify Jinja2 is installed: `poetry add jinja2`

**Output looks different?**
- Set `self.use_templates = False` to compare
- Check template syntax in `.j2` files
- Verify data structure matches expected format

**Still hanging?**
- All async operations have timeout protection
- Database connections properly closed
- Error handling prevents crashes

## ✅ Ready to Use!

Your analysis commands are now template-powered while maintaining full backward compatibility. You can switch between template and hardcoded formatting at any time, making this a zero-risk migration! 