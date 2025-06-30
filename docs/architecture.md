# Architecture Overview

## System Design

Pepino is a modern Discord analytics platform built with a **layered service architecture** using Python 3.12+ and Poetry for dependency management. The system implements the **Repository Pattern** for clean data access and improved testability, with robust configuration management and centralized logging.

## Core Architecture

```
┌─────────────────┐        ┌─────────────────┐
│   Discord Bot   │        │   CLI Interface │
│   (Real-time)   │        │   (Batch Mode)  │
└─────────┬───────┘        └─────────┬───────┘
          │                          │
          └──────────┬───────────────┘──────────────┬
                     │                              │
        ┌────────────▼─────────────┐    ┌───────────▼──────────┐
        │    Analysis Services     │    │   Discord Data Sync  │
        │  (Business Logic Layer)  │    │         (ETL)        │
        └────────────┬─────────────┘    └───────────┬──────────┘
                     │                              │
                     └────────────┬─────────────────┘
                                  │
                     ┌────────────▼─────────────┐
                     │    Repository Layer      │
                     │   (Data Access Objects)  │
                     │   + Base Filter Support  │
                     └────────────┬─────────────┘
                                  │
                     ┌────────────▼─────────────┐
                     │     Database Manager     │
                     │     (Infrastructure)     │
                     └──────────────────────────┘
```

## Package Structure

### Core Packages
| Package | Purpose | Key Components |
|---------|---------|----------------|
| **`pepino.discord`** | Discord integration | Bot, slash commands, message sync |
| **`pepino.cli`** | Command-line interface | Click-based analysis and data sync terminal commands |
| **`pepino.analysis`** | Analytics engine | Analytic helper functions for reports, charts |
| **`pepino.data`** | Data management | Database, repositories, models |
| **`pepino.templates`** | Template engine | Jinja2 template processing engine with analyzer integration and chart generation |

### Utility Modules
| Module | Purpose | Key Components |
|--------|---------|----------------|
| **`pepino.config`** | Configuration Management | Unified Settings, Environment Variables, SQL Validation, Base Filter Management |
| **`pepino.logging_config`** | Logging Infrastructure | Centralized Logger Factory, Console & File Logging, Colored Output, File Rotation |

## Recent Architecture Improvements

### 1. Configuration Robustness
- **Base Filter Validation**: SQL syntax validation with automatic cleanup
- **Field Validators**: Comprehensive validation for all config fields
- **Error Prevention**: Prevents SQL injection and malformed queries
- **Environment Integration**: Seamless .env file support

### 2. Repository Layer Enhancements
- **Consistent Base Filtering**: All sync methods apply base filter automatically
- **Bot Exclusion**: Proper filtering of bot messages and test channels
- **Query Optimization**: Efficient SQL queries with proper indexing
- **Data Integrity**: Ensures consistent data filtering across all operations

### 3. Error Handling Improvements
- **Input Validation**: Comprehensive validation for all user inputs
- **Graceful Failures**: System continues operation even with invalid data
- **Descriptive Logging**: Clear error messages for debugging
- **User-Friendly Output**: Helpful error messages for end users

### 4. Logging Modernization
- **Centralized Logger Factory**: All modules use `get_logger(__name__)`
- **Consistent Formatting**: Uniform log output across all components
- **Configurable Levels**: Log level controlled via environment/config
- **Dual Output**: Console and file logging with rotation

## Analysis Engine Architecture

### Current Analysis Modules
```
┌─────────────────────────────────────────────────────────┐
│                  Analysis Modules                       │
├─────────────────┬─────────────────┬─────────────────────┤
│ UserAnalyzer    │ ChannelAnalyzer │ TopicAnalyzer       │
│                 │                 │                     │
├─────────────────┼─────────────────┼─────────────────────┤
│ TemporalAnalyzer│ ConversationSvc │ EmbeddingService    │
│                 │                 │                     │
└─────────────────┴─────────────────┴─────────────────────┘
```

Each analyzer follows the same pattern:
- **Repository Injection**: Uses repository pattern for data access
- **Base Filter Support**: Automatic bot/exclusion filtering
- **Error Handling**: Comprehensive validation and graceful failures
- **Centralized Logging**: Consistent logging via `get_logger(__name__)`

## Design Patterns

### Repository Pattern with Base Filtering
```python
class ChannelRepository:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.base_filter = settings.base_filter.strip()  # Validated SQL
    
    def get_channel_message_statistics(self, channel_name: str, days: Optional[int] = None):
        query = f"""
        SELECT COUNT(*) as total_messages, ...
        FROM messages 
        WHERE channel_name = ? AND {self.base_filter}
        """
        # Automatically excludes bots and test channels
```

**Benefits**: 
- **Consistent Filtering**: All queries automatically exclude unwanted data
- **SQL Safety**: Validated base filter prevents injection
- **Maintainable**: Single source of truth for filtering logic

### Configuration as Code with Validation
```python
class Settings(BaseSettings):
    base_filter: str = Field(
        default="author_id != 'sesh' AND channel_name NOT LIKE '%test%'...",
        description="Base filter for excluding bots and test channels"
    )
    
    @field_validator("base_filter")
    @classmethod
    def validate_base_filter(cls, v):
        """Ensure base filter is properly formatted for SQL usage."""
        if not v or not v.strip():
            return "1=1"  # Always true condition
        cleaned = v.strip()
        if cleaned.upper().startswith("AND "):
            cleaned = cleaned[4:]  # Remove leading "AND "
        return cleaned
```

**Benefits**: 
- **Type Safety**: Pydantic validation ensures data integrity
- **SQL Safety**: Automatic cleanup prevents syntax errors
- **Environment Integration**: Seamless .env file support

### Centralized Logging Pattern
```python
from pepino.logging_config import get_logger

logger = get_logger(__name__)

class ChannelAnalyzer:
    def analyze(self, channel_name: str):
        logger.info(f"Starting channel analysis for: {channel_name}")
        # All modules use consistent logging format
```

**Benefits**:
- **Consistent Format**: Uniform log output across all modules
- **Configurable**: Log level controlled via environment
- **Dual Output**: Console and file logging with rotation
- **Easy Debugging**: Clear module identification in logs

## Key Strengths

### 1. **Robust Configuration Management**
- **SQL Validation**: Base filter automatically validated and cleaned
- **Environment Integration**: Seamless .env file support
- **Type Safety**: Pydantic validation for all config fields
- **Error Prevention**: Prevents common configuration mistakes

### 2. **Consistent Data Filtering**
- **Base Filter Application**: All repository methods automatically filter data
- **Bot Exclusion**: Proper filtering of bot messages and test channels
- **Query Safety**: Validated SQL prevents injection attacks
- **Maintainable**: Single source of truth for filtering logic

### 3. **Comprehensive Error Handling**
- **Input Validation**: All user inputs validated before processing
- **Graceful Failures**: System continues operation with invalid data
- **Descriptive Logging**: Clear error messages for debugging
- **User-Friendly Output**: Helpful error messages for end users

### 4. **Modern Logging Infrastructure**
- **Centralized Factory**: All modules use `get_logger(__name__)`
- **Consistent Formatting**: Uniform log output across components
- **Configurable Levels**: Log level controlled via environment/config
- **Dual Output**: Console and file logging with rotation

### 5. **Repository Pattern Benefits**
- **Testability**: Easy to mock repository methods for unit tests
- **Data Access Abstraction**: Business logic doesn't depend on SQL
- **Query Reuse**: Common queries centralized in repositories
- **Database Agnostic**: Can switch databases without changing analyzers

### 6. **Async-First Design**
- **Full async/await**: End-to-end async support
- **Database Pooling**: Efficient connection management
- **Concurrent Analysis**: Multiple analyses can run simultaneously
- **Non-blocking Operations**: UI remains responsive during analysis

## Testing Strategy

### Repository Mocking Approach
Pepino uses **repository mocking** instead of database fixtures for fast, reliable, and isolated tests. This eliminates database dependencies while providing comprehensive coverage of business logic.

```python
@pytest.mark.asyncio
async def test_analyzer_method():
    """Test analyzer with mocked repository"""
    mock_db_manager = MagicMock()
    analyzer = SomeAnalyzer(mock_db_manager)
    
    with patch.object(analyzer.repo, 'some_method', new_callable=AsyncMock) as mock_method:
        mock_method.return_value = test_data
        result = await analyzer.analyze()
        assert result["success"] is True
        mock_method.assert_called_once_with(expected_params)
```

### Testing Benefits
- **Speed**: Tests execute in milliseconds instead of seconds
- **Isolation**: Each test is completely independent
- **Reliability**: No database state dependencies or cleanup needed
- **Simplicity**: Focus on business logic rather than data setup

## Sync-Enabled Analysis Architecture

### Round-Trip Workflow Design

The system supports **sync-then-analyze workflows** for real-time Discord analysis:

```
Discord API → Data Freshness Check → Intelligent Sync → Analysis Execution → Results
     ↑                                      ↓
     └─────── Single Token Authentication ──┘
```

#### Workflow Components

1. **Data Freshness Check**: Configurable staleness detection
2. **Intelligent Sync**: Incremental updates with timeout protection  
3. **Analysis Execution**: Fresh data processing with chart generation
4. **Error Resilience**: Fallback to existing data if sync fails

#### Benefits
- **Always Fresh Data**: Analysis uses most recent Discord data
- **User Control**: Choose sync vs. existing data analysis
- **Single Token Operation**: No separate bot/sync token management
- **Transparent Process**: Clear feedback at each workflow step

**For detailed sync operations, see [bot_operations.md](bot_operations.md)**

## Analysis Engine

### Data Flow with Repository Pattern
```
Discord API → Extractors → Database → Repositories → Analyzers → CLI/Bot
                                    ↓
                              Base Filter Applied
                                    ↓
                              Clean Data Output
```