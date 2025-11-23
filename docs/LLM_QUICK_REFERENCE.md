# Pepino LLM Integration - Quick Reference Guide

## What is Pepino?
Discord analytics platform that combines statistical analysis with optional LLM-generated narrative summaries.

## LLM Usage Summary

### Where LLM is Used
| Component | File | Purpose |
|-----------|------|---------|
| User Summaries | `scripts/summarize_users.py` | Generate narrative prose about user activity |
| Channel Summaries | `scripts/summarize_channels.py` | Generate narrative prose about channel purpose/engagement |

### Where LLM is NOT Used
- Core analysis engines (UserAnalyzer, ChannelAnalyzer, TopicAnalyzer)
- CLI commands (pepino analyze, pepino list, pepino sync)
- Discord bot commands
- Database operations
- All use statistical and traditional NLP methods instead

## The Three Layers

```
┌─────────────────────────────────────────────────────┐
│ LAYER 3: NARRATIVE GENERATION (Optional)           │
│ - Local Ollama LLM (deepseek-r1:8b)                │
│ - Generates human-readable summaries                │
│ - Runs AFTER core analysis                          │
└─────────────────────────────────────────────────────┘
          ↑
          │ consumes JSON output from
          │
┌─────────────────────────────────────────────────────┐
│ LAYER 2: CONTEXT BUILDING                          │
│ - Database message queries                          │
│ - Lightweight NLP preprocessing                     │
│ - Token frequency analysis                          │
│ - No LLM, just data preparation                     │
└─────────────────────────────────────────────────────┘
          ↑
          │ consumes output from
          │
┌─────────────────────────────────────────────────────┐
│ LAYER 1: CORE ANALYSIS (Statistical)               │
│ - UserAnalyzer: Message stats, activity patterns    │
│ - ChannelAnalyzer: Engagement, health metrics       │
│ - TopicAnalyzer: BERTopic clustering               │
│ - NLPService: spaCy-based text analysis             │
│ - TemporalAnalyzer: Time-based patterns             │
│ - NO LLM, just statistics and embeddings            │
└─────────────────────────────────────────────────────┘
```

## Quick Setup

### Prerequisites
- Local Ollama installation: https://ollama.ai
- Model: `deepseek-r1:8b` (default) or any other Ollama model
- Database: `data/discord_messages.db` (SQLite)

### Step 1: Prepare Analyzed Data
```bash
# Generate user analysis JSON
poetry run pepino analyze users --all --output stats/users.json

# OR generate channel analysis JSON
poetry run pepino analyze channels --all --output stats/channels.json
```

### Step 2: Generate Summaries with LLM
```bash
# User summaries (top 100 users)
poetry run python scripts/summarize_users.py \
  --input stats/users.json \
  --output stats/user_summaries.json

# Channel summaries (all channels)
poetry run python scripts/summarize_channels.py \
  --input stats/channels.json \
  --output stats/channel_summaries.json
```

## Key Files & Line Numbers

| File | Key Function | Lines | Purpose |
|------|--------------|-------|---------|
| `scripts/summarize_users.py` | `_call_llm()` | 388-419 | HTTP call to Ollama API |
| `scripts/summarize_users.py` | `_prepare_user_context()` | 319-385 | Build prompt context |
| `scripts/summarize_users.py` | `_analyze_messages()` | 283-316 | NLP preprocessing |
| `scripts/summarize_channels.py` | `_call_llm()` | 242-272 | HTTP call to Ollama API |
| `scripts/summarize_channels.py` | `_prepare_channel_context()` | 158-239 | Build prompt context |

## LLM API Details

### Endpoint
```
POST http://localhost:11434/api/generate
```

### Request Payload
```json
{
  "model": "deepseek-r1:8b",
  "prompt": "system prompt + user prompt + context",
  "stream": false,
  "options": {"temperature": 0.6}
}
```

### Environment Variables
```bash
OLLAMA_HOST=http://localhost:11434          # Primary
OLLAMA_BASE_URL=http://localhost:11434      # Secondary
OLLAMA_URL=http://localhost:11434           # Tertiary
# Default: http://localhost:11434
```

### Configuration
- **Timeout**: 300 seconds (5 minutes)
- **Temperature**: 0.6 (fixed)
- **Model**: Configurable via `--model` flag
- **Max Words**: Configurable via `--max-words` flag (default: 120 for users, 100 for channels)

## Prompts Used

### User Summary System Prompt
```
You are an expert community analyst. Write compact prose summaries (under {max_words} words) 
that describe a Discord member's focus areas, tone, and engagement. Use concrete evidence 
(message volumes, activity spans, topics, style, recommendations, key channels) when available. 
Avoid bullet points. Do not speculate or fabricate. Be critical and objective.
```

### Channel Summary System Prompt
```
You are an expert community analyst. Craft concise (under 100 words) narrative summaries 
describing the purpose, recurring content, and engagement level of Discord channels. 
Use an informative but friendly tone. Avoid bullet points. Mention concrete signals 
(message counts, top contributors, recent activity, engagement stats) when available. 
Do not fabricate data.
```

## Data Sent to LLM

### User Context Includes
- User ID, display name, username
- Message count, active channels, active days
- Average message length, date range
- Top 5 channels by activity
- BERTopic-derived topics with frequencies
- Token frequency analysis (top words, bigrams)
- Up to 3 long-form message examples (320 chars max each)

### Channel Context Includes
- Channel ID, name, parent category
- Total/human/bot message counts
- Unique users, average message length
- Engagement metrics (replies per post, reaction rate)
- Health metrics (weekly active, lurkers, participation rate)
- Top topics with frequencies
- Content clusters and keywords (if available)

## Output Formats

### User Summaries JSON
```json
{
  "summaries": [
    {
      "user_id": "123456789",
      "username": "alice",
      "display_name": "Alice",
      "summary": "LLM-generated narrative...",
      "generated_at": "2024-11-23T14:32:00Z",
      "model": "deepseek-r1:8b",
      "statistics": {
        "message_count": 847,
        "channels_active": 12
      },
      "messages_analyzed": 500,
      "long_message_samples": [...]
    }
  ]
}
```

### Channel Summaries JSON
```json
{
  "summaries": [
    {
      "channel_key": "987654321",
      "channel_id": "987654321",
      "channel_name": "dev-announcements",
      "parent_id": "567890",
      "parent_name": "Dev",
      "summary": "LLM-generated narrative...",
      "generated_at": "2024-11-23T14:32:00Z",
      "model": "deepseek-r1:8b"
    }
  ]
}
```

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Connection Error | Ollama not running | Start Ollama: `ollama serve` |
| 404 Not Found | Wrong endpoint | Check OLLAMA_HOST, default is http://localhost:11434 |
| Model not found | Model not downloaded | Run `ollama pull deepseek-r1:8b` |
| Timeout (300s) | Model too slow | Use smaller model like `mistral:7b` or increase timeout |
| JSON decode error | Invalid response | Check Ollama logs for errors |

## Common Commands

```bash
# Download a model
ollama pull deepseek-r1:8b
ollama pull mistral:7b
ollama pull llama2:7b

# Start Ollama service
ollama serve

# List available models
ollama list

# Test LLM endpoint
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "deepseek-r1:8b", "prompt": "Say hello", "stream": false}'
```

## Performance Notes

- **Ollama API calls**: Typically 10-30 seconds per summary (deepseek-r1:8b)
- **Database queries**: 100-500 messages per user, ~2-5 seconds
- **NLP preprocessing**: <1 second per user/channel
- **Batch size**: Default 100 users, but limited by LLM speed
- **Total time for 100 users**: ~30-60 minutes depending on model

## Customization

### Change Model
```bash
poetry run python scripts/summarize_users.py \
  --input stats/users.json \
  --model llama2:7b  # or mistral:7b, etc.
```

### Change Word Limit
```bash
poetry run python scripts/summarize_users.py \
  --input stats/users.json \
  --max-words 200  # default is 120
```

### Change Ollama Host
```bash
OLLAMA_HOST=http://remote-server:11434 \
poetry run python scripts/summarize_users.py \
  --input stats/users.json
```

## Non-LLM Alternative

If you don't want to use LLM summaries, the core Pepino functionality still works perfectly:
- All statistical analysis (UserAnalyzer, ChannelAnalyzer)
- Topic modeling (BERTopic)
- Time patterns
- Engagement metrics
- Discord bot commands
- CLI commands
- Template-based output formatting

Just skip the `summarize_users.py` and `summarize_channels.py` scripts.

## Full LLM Analysis Report

For comprehensive details including architecture diagrams, complete data flows, and source code analysis, see: `LLM_ANALYSIS_REPORT.md`
