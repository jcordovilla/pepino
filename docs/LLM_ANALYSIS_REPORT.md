# Pepino LLM-Based Analysis Workflow Report

## Executive Summary

Pepino is a Discord analytics platform that integrates LLM capabilities for generating narrative summaries of user and channel activity. The system uses **local Ollama LLMs** (specifically `deepseek-r1:8b` by default) rather than cloud-based APIs, with data preprocessing through spaCy NLP models and traditional statistical analysis.

---

## 1. LLM Architecture Overview

### LLM Service Architecture
- **Type**: Local inference server (Ollama-compatible API)
- **Default Model**: `deepseek-r1:8b` 
- **Connection Method**: HTTP POST requests to `/api/generate` endpoint
- **Default Base URL**: `http://localhost:11434`
- **Environment Variables**: 
  - `OLLAMA_HOST`
  - `OLLAMA_BASE_URL`
  - `OLLAMA_URL`

### Where LLM is Used

The LLM is integrated in **two standalone scripts** that consume pre-analyzed data:

1. **`scripts/summarize_users.py`** - User Activity Summaries
2. **`scripts/summarize_channels.py`** - Channel Activity Summaries

### Important: No LLM in Core Analysis

The main analysis engines (`UserAnalyzer`, `ChannelAnalyzer`, `TopicAnalyzer`) do **NOT** use LLMs. They use:
- Statistical aggregation
- spaCy NLP for concept extraction
- BERTopic for topic modeling
- Vector embeddings for semantic analysis
- Traditional text processing

---

## 2. User Activity LLM Workflow

### File Location
**`/Users/jose/mylab/pepino/scripts/summarize_users.py`** (Lines 1-544)

### Complete Workflow Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. USER INVOKES COMMAND                                          │
│    poetry run python scripts/summarize_users.py \                │
│    --input stats/user_analysis_all_20251110-232006.json \        │
│    --output stats/user_summaries.json \                          │
│    --model deepseek-r1:8b \                                      │
│    --max-words 120                                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. LOAD USER ANALYSIS JSON                                       │
│    _load_user_entries() - Line 118                               │
│    - Reads aggregated JSON from pepino analyze users --all       │
│    - Filters out bot accounts                                    │
│    - Extracts metadata: display_name, message_count, channels    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. SELECT USERS TO SUMMARIZE                                     │
│    _select_users() - Line 154                                    │
│    - If --users specified: filter to requested users             │
│    - Otherwise: sort by message count and select top N (default) │
│    - Default limit: 100 users                                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. DATABASE MESSAGE FETCHING (for each user)                     │
│    _fetch_user_messages() - Line 207                             │
│    - Query SQLite messages table (data/discord_messages.db)      │
│    - Fetch up to 500 recent messages by user                     │
│    - Filter: content NOT NULL, TRIM(content) != ''              │
│    - Order by timestamp DESC                                     │
│                                                                  │
│    _fetch_long_messages() - Line 241                             │
│    - Fetch longer-form messages (280+ chars)                     │
│    - Limit: 20 samples                                           │
│    - Order by LENGTH DESC then timestamp DESC                    │
│    - Used as context examples in LLM prompt                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. LIGHTWEIGHT NLP PREPROCESSING                                 │
│    _analyze_messages() - Line 283                                │
│    - Tokenization with _tokenize() - Line 277                    │
│    - Regex word extraction: [a-zA-Z][a-zA-Z0-9_+-]{2,}         │
│    - Stop words filtering (115 English common words)             │
│    - Compute:                                                    │
│      * Average message length (characters)                       │
│      * Word frequency (Counter.most_common(15))                  │
│      * Bigram extraction (pairs of consecutive words)            │
│      * Bigram frequency (Counter.most_common(10))                │
│    - Returns: {total_messages_analyzed, average_length,          │
│                 top_words, top_bigrams}                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. BUILD LLM CONTEXT                                              │
│    _prepare_user_context() - Line 319                            │
│    Builds structured text prompt with:                           │
│                                                                  │
│    User Metadata:                                                │
│    • Display name, username, user ID                             │
│                                                                  │
│    Activity Metrics:                                             │
│    • Total messages, active channels, active days                │
│    • Avg message length, first/last message dates                │
│                                                                  │
│    Top Channels: Top 5 channels by message count                 │
│                                                                  │
│    Model Topics: Top topics from BERTopic analysis               │
│                                                                  │
│    Message Highlights:                                           │
│    • Messages analyzed count                                     │
│    • Average message length                                      │
│    • 3 long-form message samples (truncated at 320 chars)       │
│                                                                  │
│    Returns: Formatted multi-line string                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. LLM API CALL                                                   │
│    _call_llm() - Line 388                                        │
│                                                                  │
│    System Prompt (Line 46-50):                                   │
│    "You are an expert community analyst. Write compact prose     │
│     summaries (under {max_words} words) that describe a Discord  │
│     member's focus areas, tone, and engagement. Use concrete     │
│     evidence (message volumes, activity spans, topics, style,    │
│     recommendations, key channels) when available. Avoid bullet  │
│     points. Do not speculate or fabricate. Be critical and       │
│     objective."                                                  │
│                                                                  │
│    User Prompt (Line 395-401):                                   │
│    "Summarize the following Discord user analytics in under      │
│     {max_words} words. Emphasize participation style, recurring  │
│     themes, and observed impact. Analytics:\n{context}"          │
│                                                                  │
│    HTTP Request:                                                 │
│    POST {base_url}/api/generate                                  │
│                                                                  │
│    Payload (Line 405-411):                                       │
│    {                                                             │
│      "model": "deepseek-r1:8b",                                  │
│      "prompt": "{system_prompt}\n\n{user_prompt}",              │
│      "stream": false,                                            │
│      "options": {                                                │
│        "temperature": 0.6                                        │
│      }                                                           │
│    }                                                             │
│                                                                  │
│    Response Processing (Line 414-419):                           │
│    - HTTP POST with 300s timeout                                 │
│    - Extract "response" field from JSON                          │
│    - Strip whitespace and return                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. OUTPUT FORMATTING & STORAGE                                   │
│    main() - Line 471                                             │
│    For each user, create result dictionary:                      │
│    {                                                             │
│      "user_id": author_id,                                       │
│      "username": username,                                       │
│      "display_name": entry.get("_display_name"),                 │
│      "summary": summary_text,           ← LLM OUTPUT            │
│      "generated_at": ISO 8601 timestamp,                         │
│      "model": "deepseek-r1:8b",                                  │
│      "statistics": {                                             │
│        "message_count": ...,                                     │
│        "channels_active": ...                                    │
│      },                                                          │
│      "messages_analyzed": ...,                                   │
│      "long_message_samples": [...]                               │
│    }                                                             │
│                                                                  │
│    Write to JSON file (Line 534-537):                            │
│    {                                                             │
│      "summaries": [array of results]                             │
│    }                                                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    OUTPUT COMPLETE
```

---

## 3. Channel Activity LLM Workflow

### File Location
**`/Users/jose/mylab/pepino/scripts/summarize_channels.py`** (Lines 1-419)

### Workflow (Similar Pattern to Users)

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. LOAD CHANNEL ANALYSIS JSON                                    │
│    _load_channel_data() - Line 84                                │
│    - Reads aggregated JSON from pepino analyze channels --all    │
│    - Indexes by channel ID or name                               │
│    - Resolves parent category names via DB queries               │
│    - Creates aliases for fuzzy matching                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. SELECT CHANNELS TO SUMMARIZE                                  │
│    _select_channels() - Line 275                                 │
│    - If --channels specified: match requested channels           │
│    - If --sample N: randomly select N channels                   │
│    - Otherwise: process all channels                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. BUILD LLM CONTEXT FOR CHANNEL                                 │
│    _prepare_channel_context() - Line 158                         │
│    Builds structured text with:                                  │
│                                                                  │
│    • Channel ID, name, parent category                           │
│    • Statistics:                                                 │
│      - Total/human/bot message counts                            │
│      - Unique users, average message length                      │
│      - First/last message dates                                  │
│    • Engagement Metrics:                                         │
│      - Replies per post, reaction rates                          │
│      - Posts with reactions, reply counts                        │
│    • Health Metrics:                                             │
│      - Weekly active members, lurkers                            │
│      - Participation rate, inactive users                        │
│    • Top Topics: Most discussed items (with frequencies)         │
│    • Content Clusters & Keywords (if available)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. LLM API CALL                                                   │
│    _call_llm() - Line 242                                        │
│                                                                  │
│    System Prompt (Line 37-42):                                   │
│    "You are an expert community analyst. Craft concise (under    │
│     100 words) narrative summaries describing the purpose,       │
│     recurring content, and engagement level of Discord channels. │
│     Use an informative but friendly tone. Avoid bullet points.   │
│     Mention concrete signals (message counts, top contributors,  │
│     recent activity, engagement stats) when available. Do not    │
│     fabricate data."                                             │
│                                                                  │
│    User Prompt (Line 250-254):                                   │
│    "Summarize the following Discord channel analytics in less    │
│     than {max_words} words. Focus on the channel's purpose,      │
│     recurring content, and engagement level. Channel analytics:  │
│     {context}"                                                   │
│                                                                  │
│    HTTP POST to Ollama /api/generate endpoint                    │
│    Model: deepseek-r1:8b (configurable)                          │
│    Temperature: 0.6                                              │
│    Stream: false                                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. OUTPUT FORMATTING & STORAGE                                   │
│    Results per channel (Line 391-401):                           │
│    {                                                             │
│      "channel_key": key identifier,                              │
│      "channel_id": Discord channel ID,                           │
│      "channel_name": display name,                               │
│      "parent_id": category ID,                                   │
│      "parent_name": category name,                               │
│      "summary": summary_text,       ← LLM OUTPUT                 │
│      "generated_at": ISO 8601 timestamp,                         │
│      "model": "deepseek-r1:8b"                                   │
│    }                                                             │
│                                                                  │
│    Write to JSON (auto-generated or specified path)              │
│    Also echoes each summary to stdout                            │
└──────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Flow to LLM: Detailed Inputs

### Data Sources for User Summaries

| Data Type | Source | Query/Method | Example |
|-----------|--------|--------------|---------|
| **User Metadata** | JSON input file | `entry.get("_display_name")` | "alice_dev" |
| **Statistics** | JSON input file | `statistics.get("message_count")` | 847 |
| **Channel Activity** | JSON input file | `entry.get("channel_activity")` | [{channel_name, message_count}...] |
| **Topics** | JSON input file | `entry.get("top_topics")` | [{topic, frequency, relevance_score}...] |
| **Recent Messages** | SQLite DB | Query: `SELECT content FROM messages WHERE author_name = ? ... LIMIT 500` | Raw message text |
| **Long Messages** | SQLite DB | Query: `SELECT content ... WHERE LENGTH(content) >= 280 ... LIMIT 20` | Long-form text samples |
| **Tokenized Stats** | In-Memory Processing | Counter on tokens from messages | top_words, top_bigrams |

### Data Sources for Channel Summaries

| Data Type | Source | Example |
|-----------|--------|---------|
| **Channel Metadata** | JSON input file | channel_id, channel_name, parent_id |
| **Statistics** | JSON / Full Analysis | total_messages, unique_users, bot/human split |
| **Engagement Metrics** | Full Analysis | replies_per_post, reaction_rate, posts_with_reactions |
| **Health Metrics** | Full Analysis | weekly_active, lurkers, participation_rate |
| **Topics** | Full Analysis | [{term/topic, frequency}...] |

---

## 5. LLM API Integration Details

### Ollama API Endpoint

**Endpoint**: `POST {base_url}/api/generate`

**Default Base URL**: `http://localhost:11434`

**Request Payload Structure**:
```json
{
  "model": "deepseek-r1:8b",
  "prompt": "system prompt + user prompt + context",
  "stream": false,
  "options": {
    "temperature": 0.6
  }
}
```

**Response Structure**:
```json
{
  "model": "deepseek-r1:8b",
  "created_at": "2024-11-23T...",
  "response": "Generated summary text here...",
  "done": true,
  "context": [...],
  "total_duration": 12345678,
  "load_duration": 1234567,
  "prompt_eval_duration": 234567,
  "eval_duration": 8901234,
  "eval_count": 150
}
```

**Configuration**:
- **Model Parameter**: Configurable via `--model` flag (default: `deepseek-r1:8b`)
- **Temperature**: Fixed at `0.6` (balances creativity and consistency)
- **Timeout**: 300 seconds (5 minutes)
- **Base URL Environment Variables** (in order of precedence):
  1. `OLLAMA_HOST`
  2. `OLLAMA_BASE_URL`
  3. `OLLAMA_URL`
  4. Default: `http://localhost:11434`

---

## 6. Prompt Engineering

### System Prompt for Users (summarize_users.py)

```
You are an expert community analyst. Write compact prose summaries (under {max_words} words) 
that describe a Discord member's focus areas, tone, and engagement. Use concrete evidence 
(message volumes, activity spans, topics, style, recommendations, key channels) when available. 
Avoid bullet points. Do not speculate or fabricate. Be critical and objective.
```

**Key Characteristics**:
- Asks for narrative prose (not bullets)
- Emphasizes evidence-based analysis
- Sets word limit constraint
- Requests objectivity and criticism

### User-Specific Prompt Instructions

```
Summarize the following Discord user analytics in under {max_words} words.
Emphasize participation style, recurring themes, and observed impact.
Analytics:
{context}
```

### System Prompt for Channels (summarize_channels.py)

```
You are an expert community analyst. Craft concise (under 100 words) narrative 
summaries describing the purpose, recurring content, and engagement level of 
Discord channels. Use an informative but friendly tone. Avoid bullet points. 
Mention concrete signals (message counts, top contributors, recent activity, 
engagement stats) when available. Do not fabricate data.
```

### Channel-Specific Prompt Instructions

```
Summarize the following Discord channel analytics in less than {max_words} words.
Focus on the channel's purpose, recurring content, and engagement level.
Channel analytics:
{context}
```

---

## 7. Prompt Context Examples

### Sample User Context Sent to LLM

```
User: alice_dev
Handle: alice_dev
User ID: 123456789
Activity Metrics:
  Messages: 847
  Active Channels: 12
  Active Days: 45
  Avg Message Length: 95.3
  First Message: 2024-10-01T08:15:00Z
  Last Message: 2024-11-20T18:42:00Z
Top Channels:
  general: 234 msgs
  dev-help: 187 msgs
  announcements: 156 msgs
  random: 98 msgs
  projects: 72 msgs
Model Topics:
  Python Development (45)
  API Design (32)
  Testing Practices (28)
  Database Optimization (15)
Message Highlights:
  Sampled Messages: 500
  Average Length: 112.5 characters
  Long-form Samples:
    [1] I've been using pytest fixtures extensively and found that parametrizing...
    [2] When designing REST APIs, consider hypermedia as the engine of application...
    [3] Database normalization is crucial for scaling, but in some cases denormalization...
```

### Sample Channel Context Sent to LLM

```
Channel: dev-announcements
Channel ID: 987654321
Parent: Dev (ID: 567890)
Statistics:
  total_messages: 2345
  unique_users: 89
  unique_human_users: 78
  avg_message_length: 156
  human_messages: 2100
  bot_messages: 245
  first_message: 2024-01-15T10:00:00Z
  last_message: 2024-11-22T17:30:00Z
Engagement Metrics:
  human_replies_per_post: 1.8
  human_posts_with_reactions: 45%
  human_replies: 567
  human_original_posts: 1210
Health Metrics:
  weekly_active: 42
  inactive_users: 12
  total_channel_members: 156
  lurkers: 45
  human_participation_rate: 50%
Content Analysis:
  top_topics: Release Planning (98), Architecture (72), DevOps (65), Testing (54)
```

---

## 8. Result Processing and Storage

### Output Format: User Summaries JSON

**File**: `stats/user_summaries.json`

```json
{
  "summaries": [
    {
      "user_id": "123456789",
      "username": "alice_dev",
      "display_name": "Alice Developer",
      "summary": "Alice is an active Python developer who frequently discusses API design and testing practices. With 847 messages across 12 channels over 45 days, she demonstrates sustained engagement. Her contributions are technical and evidence-driven, focusing on best practices in development. Most active in #dev-help and #general, averaging 95 characters per message. Her participation suggests a mentoring role, with emphasis on architectural and testing discussions.",
      "generated_at": "2024-11-23T14:32:00.123456Z",
      "model": "deepseek-r1:8b",
      "statistics": {
        "message_count": 847,
        "channels_active": 12
      },
      "messages_analyzed": 500,
      "long_message_samples": [
        "I've been using pytest fixtures...",
        "When designing REST APIs..."
      ]
    }
  ]
}
```

### Output Format: Channel Summaries JSON

**File**: `stats/channel_summaries_TIMESTAMP.json`

```json
{
  "summaries": [
    {
      "channel_key": "987654321",
      "channel_id": "987654321",
      "channel_name": "dev-announcements",
      "parent_id": "567890",
      "parent_name": "Dev",
      "summary": "Dev Announcements is the primary hub for release planning and architectural discussions in the dev category. With 2,345 messages from 89 unique users over 11 months, it shows strong engagement with 50% participation rate. Topics include Release Planning (98 mentions), Architecture (72), and DevOps (65). The channel averages 156 characters per message and maintains healthy participation with 42 weekly active members and approximately 50% human participation rate.",
      "generated_at": "2024-11-23T14:32:00.123456Z",
      "model": "deepseek-r1:8b"
    }
  ]
}
```

---

## 9. Non-LLM Analysis Components (Core System)

### UserAnalyzer (`src/pepino/analysis/user_analyzer.py`)

**LLM Status**: ❌ NO LLM USAGE

**What it does**:
- Statistical analysis of message counts, channels, activity patterns
- Time-of-day activity patterns
- Topic extraction via BERTopic (NOT LLM-based)
- Fallback keyword extraction for topics
- No direct LLM calls

**Key Methods**:
- `analyze()` - Basic user statistics
- `analyze_enhanced()` - Enhanced with semantic analysis via NLP
- `_get_user_statistics_via_repository()` - Database queries
- `_analyze_time_patterns_via_repository()` - Time-based patterns
- `_get_user_topics_via_repository()` - Topic extraction

### ChannelAnalyzer (`src/pepino/analysis/channel_analyzer.py`)

**LLM Status**: ❌ NO LLM USAGE

**What it does**:
- Channel message statistics and user engagement
- Time patterns (hourly, daily, weekly)
- Health metrics (participation rate, lurkers, etc.)
- Engagement metrics (replies, reactions)
- Recent activity tracking

**Key Methods**:
- `analyze()` - Comprehensive channel analysis
- `_get_channel_statistics_via_repository()` - Basic stats
- `_analyze_time_patterns_via_repository()` - Time patterns
- `get_channel_health()` - Health scoring

### NLPService (`src/pepino/analysis/nlp_analyzer.py`)

**LLM Status**: ❌ NO LLM USAGE - Uses spaCy instead

**What it does**:
- Concept extraction (noun chunks, named entities, compounds)
- Sentiment analysis (lexicon-based)
- Named entity recognition
- Key phrase extraction
- Text complexity analysis

**Models Used**:
- spaCy: `en_core_web_sm` (small English model)
- Not LLM-based, uses rule-based NLP

**Key Features**:
- AI/Tech keyword detection (lines 136-189)
- Acronym normalization (GPT → GPT, LLM → LLM)
- Stop word filtering
- Noun phrase extraction
- Entity type detection (PERSON, ORG, PRODUCT, etc.)

### TopicAnalyzer (`src/pepino/analysis/topic_analyzer.py`)

**LLM Status**: ❌ NO LLM USAGE - Uses BERTopic

**What it does**:
- Multi-stage topic modeling pipeline
- UMAP dimensionality reduction
- HDBSCAN clustering
- Coherence-based quality filtering
- spaCy-based domain analysis

**Models Used**:
- BERTopic with `all-mpnet-base-v2` embeddings
- spaCy for domain analysis
- NOT LLM-based

---

## 10. Templates (Output Formatting)

### Discord Templates

**User Analysis**: `/templates/outputs/discord/user_analysis.md.j2`
- Shows statistics, channel activity, time patterns
- Displays topics with frequency/relevance scores
- Shows semantic analysis (entities, tech terms, concepts)
- Uses custom Jinja2 filters for formatting

**Channel Analysis**: `/templates/outputs/discord/channel_analysis.md.j2`
- Channel stats with bot/human message split
- Engagement metrics
- Top contributors
- Peak activity hours
- Health metrics
- Topics discussed

### CLI Templates

Similar structure but formatted for terminal output (`.txt.j2` files)

### Template Engine Features

**File**: `/Users/jose/mylab/pepino/src/pepino/templates/template_engine.py`

**Capabilities**:
- Analyzer integration (UserAnalyzer, ChannelAnalyzer, etc.)
- NLP helper functions (sentiment, concepts, entities)
- Chart generation (activity graphs, pie charts, word clouds)
- Custom Jinja2 filters
- Message data access from database

**NLP Functions in Templates**:
- `extract_concepts(text)` - Extract key concepts
- `analyze_sentiment(text)` - Sentiment classification
- `get_named_entities(text)` - Entity extraction
- `extract_key_phrases(text)` - Phrase extraction
- `analyze_complexity(text)` - Text complexity scoring

---

## 11. Configuration & Environment

### Settings (`src/pepino/config.py`)

**LLM-Related Settings**:
```python
# Embedding settings (for semantic analysis, NOT LLM)
embedding_model: str = "all-MiniLM-L6-v2"
embedding_batch_size: int = 32

# NLP settings (spaCy, NOT LLM)
nlp_model: str = "en_core_web_sm"
nlp_cache_size: int = 500

# Database
db_path: str = "data/discord_messages.db"

# Analysis
max_messages: int = 10000
topic_model_n_components: int = 5
```

**No cloud LLM API keys in configuration** - all processing is local

---

## 12. Command-Line Usage Examples

### Generate User Summaries

```bash
# Top 100 users with default settings
poetry run python scripts/summarize_users.py \
  --input stats/user_analysis_all_20251110-232006.json

# Specific users
poetry run python scripts/summarize_users.py \
  --input stats/user_analysis_all_20251110-232006.json \
  --users alice jane bob

# Custom model and word limit
poetry run python scripts/summarize_users.py \
  --input stats/user_analysis_all_20251110-232006.json \
  --model llama2:7b \
  --max-words 200 \
  --output stats/user_summaries.json

# 50 top users
poetry run python scripts/summarize_users.py \
  --input stats/user_analysis_all_20251110-232006.json \
  --limit 50
```

### Generate Channel Summaries

```bash
# Specific channels
poetry run python scripts/summarize_channels.py \
  --input stats/channel_analysis_all_20251110-162929.json \
  --channels general announcements dev-help

# Random sample of 5 channels
poetry run python scripts/summarize_channels.py \
  --input stats/channel_analysis_all_20251110-162929.json \
  --sample 5

# Custom model and save to file
poetry run python scripts/summarize_channels.py \
  --input stats/channel_analysis_all_20251110-162929.json \
  --model mistral:7b \
  --max-words 150 \
  --output stats/channel_summaries.json
```

---

## 13. Error Handling & Timeouts

### LLM Call Error Handling

**File**: `summarize_users.py` Line 414-415, `summarize_channels.py` Line 267-268

```python
response = requests.post(url, json=payload, timeout=300)
response.raise_for_status()  # Raises HTTPError if status is 4xx/5xx
```

**Timeout**: 300 seconds (5 minutes) per request

**Error Scenarios**:
- LLM service not running → ConnectionError
- Network timeout → Timeout exception
- HTTP error response → HTTPError (e.g., 500 Internal Server Error)
- Invalid JSON response → JSONDecodeError

**Current Behavior**: Errors bubble up to caller (no retry logic)

---

## 14. Complete Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                    USER REQUESTS ANALYSIS                         │
│  pepino analyze users --all                                       │
│  pepino analyze channels --all                                    │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────────┐
    │ CORE ANALYSIS ENGINES                    │
    │ (NO LLM - Uses Statistical + NLP)       │
    │                                          │
    │ ✓ UserAnalyzer                          │
    │ ✓ ChannelAnalyzer                       │
    │ ✓ TopicAnalyzer (BERTopic)              │
    │ ✓ TemporalAnalyzer                      │
    │ ✓ NLPService (spaCy)                    │
    │                                          │
    │ Data Sources:                            │
    │ • SQLite Database (discord_messages.db) │
    │ • Statistical aggregation               │
    │ • spaCy NLP processing                  │
    │ • Sentence Transformers embeddings      │
    └──────────────┬───────────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────────┐
    │ JSON EXPORT (CLI or Discord Command)     │
    │                                          │
    │ Output files:                            │
    │ • user_analysis_all_TIMESTAMP.json       │
    │ • channel_analysis_all_TIMESTAMP.json    │
    └──────────────┬───────────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────────┐
    │ USER RUNS SUMMARIZATION SCRIPT           │
    │ summarize_users.py                       │
    │ summarize_channels.py                    │
    │                                          │
    │ Input: JSON files                        │
    └──────────────┬───────────────────────────┘
                   │
                   ├─────────────────────────────────────┐
                   │                                     │
                   ▼                                     ▼
    ┌──────────────────────────┐    ┌─────────────────────────┐
    │ DATABASE MESSAGE QUERY   │    │ NLP PREPROCESSING       │
    │                          │    │                         │
    │ SELECT content FROM      │    │ • Tokenization          │
    │ messages WHERE           │    │ • Stop word filtering   │
    │ author_name = ?          │    │ • Word frequency        │
    │ LIMIT 500                │    │ • Bigram extraction     │
    │                          │    │                         │
    │ Returns: 500 messages    │    │ Returns: top_words,     │
    │                          │    │ top_bigrams, avg_len    │
    └──────────────┬───────────┘    └──────────────┬──────────┘
                   │                                │
                   └────────────────┬───────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────────┐
                    │ BUILD LLM CONTEXT                │
                    │ (Structured prompt)              │
                    │                                  │
                    │ User/Channel metadata            │
                    │ Statistics & metrics             │
                    │ Top topics/channels              │
                    │ Message samples & examples       │
                    │ NLP analysis results             │
                    └──────────────┬───────────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────┐
        │ OLLAMA API CALL                              │
        │                                              │
        │ POST http://localhost:11434/api/generate     │
        │                                              │
        │ Model: deepseek-r1:8b (default)              │
        │ Temperature: 0.6                             │
        │ Timeout: 300s                                │
        │                                              │
        │ Prompt:                                      │
        │ 1. System prompt (instructions)              │
        │ 2. User task prompt                          │
        │ 3. Context data                              │
        └──────────────┬───────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────────────┐
        │ LLM RESPONSE                                 │
        │                                              │
        │ Generated narrative summary:                 │
        │ "<120 words (users) or <100 words (channels)│
        │                                              │
        │ Examples:                                    │
        │ • User focus areas                           │
        │ • Engagement patterns                        │
        │ • Tone & style                               │
        │ • Channel purpose & content                  │
        │ • Community health                           │
        └──────────────┬───────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────────────────┐
        │ FORMAT & SAVE RESULTS                        │
        │                                              │
        │ JSON output with:                            │
        │ • Summary text (LLM output)                  │
        │ • Metadata (user/channel IDs, names)         │
        │ • Model info & timestamp                     │
        │ • Statistics & message samples               │
        │                                              │
        │ Output files:                                │
        │ • user_summaries.json                        │
        │ • channel_summaries_TIMESTAMP.json           │
        └──────────────────────────────────────────────┘
```

---

## 15. Summary of LLM Usage

### What Uses LLM
✓ **Standalone Scripts Only**
- `scripts/summarize_users.py` - Generate user narrative summaries
- `scripts/summarize_channels.py` - Generate channel narrative summaries

### What Doesn't Use LLM
✗ **Core Analysis** (statistical, not generative)
- UserAnalyzer - Statistics only
- ChannelAnalyzer - Statistics only  
- TopicAnalyzer - BERTopic clustering, NOT LLM
- NLPService - spaCy-based, NOT LLM
- TemporalAnalyzer - Time-based analysis, NOT LLM

✗ **Database Operations**
- All queries use SQLite directly

✗ **CLI Commands**
- `pepino analyze users|channels|topics|temporal` - No LLM
- `pepino list` commands - No LLM
- `pepino sync` - No LLM

✗ **Discord Bot Commands**
- Discord analysis commands use templates only, not LLM

### LLM Model Used
- **Default**: `deepseek-r1:8b`
- **Configurable**: Via `--model` CLI flag
- **Backend**: Ollama (local inference, not cloud API)
- **Temperature**: 0.6 (fixed)
- **Base URL**: http://localhost:11434 (configurable via env vars)

### Processing Pipeline Summary
```
Raw Analysis JSON → Message Database Query → NLP Preprocessing 
→ Context Building → Ollama LLM Call → Summary Generation 
→ JSON Output
```

---

## 16. Key Dependencies

### LLM-Related Packages
- **requests** 2.31.0+ - HTTP client for Ollama API calls
- **click** - CLI argument parsing

### NLP Packages (NOT LLM)
- **spacy** 3.7.2 - NLP model en_core_web_sm
- **sentence-transformers** - all-MiniLM-L6-v2 embeddings
- **bertopic** - Topic modeling with UMAP/HDBSCAN

### Core Packages
- **pydantic** - Data validation
- **discord.py** - Discord bot integration
- **sqlite3** - Built-in, no installation needed

---

## Conclusion

The Pepino analytics platform uses **local Ollama LLMs** exclusively for generating narrative summaries of user and channel activity. The core analysis work (statistics, NLP, topic modeling) is performed using traditional ML/statistical methods, not LLMs. The LLM layer serves as a post-processing step to convert structured analytical data into human-readable prose summaries, with the default model being `deepseek-r1:8b` running locally without any cloud API dependencies.
