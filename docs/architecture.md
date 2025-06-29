# Architecture Overview

## System Design

Pepino is a modern Discord analytics platform built with a **layered service architecture** using Python 3.12+ and Poetry for dependency management. The system now implements the **Repository Pattern** for clean data access and improved testability.

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
| **`pepino.config`** | Configuration Management | Unified Settings, Environment Variables, Validation |
| **`pepino.logging_config`** | Logging Infrastructure | Console & File Logging, Colored Output, File Rotation |

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
- **Async Operations**: Full async/await support
- **Error Handling**: Consistent error response format
- **Configurable Filtering**: Base filter support for data scoping

## Design Patterns

### Repository Pattern
```python
class MessageRepository:
    async def get_messages_by_channel(self, channel_name: str) -> List[Message]
    async def get_top_users(self, limit: int) -> List[Dict]
    async def get_activity_trends(self, days: int) -> Dict
    async def get_temporal_analysis_data(self, **kwargs) -> List[Dict]
    async def get_messages_for_topic_analysis(self, **kwargs) -> List[str]
```

**Benefits**: Database abstraction, testability, query reuse, easy mocking

### Analysis Service Pattern
```python
class UserAnalyzer:
    def __init__(self, db_manager, base_filter: Optional[str] = None):
        self.db_manager = db_manager
        self.base_filter = base_filter or Settings().base_filter
        self.user_repo = UserRepository(self.db_manager)
    
    async def analyze(self, **kwargs) -> Dict[str, Any]:
        # Business logic using repository methods
        user_info = await self.user_repo.find_user_by_name(user_name)
        stats = await self.user_repo.get_user_statistics_by_id(user_id)
        return {"success": True, "user_info": user_info, "statistics": stats}
```

**Benefits**: Business logic isolation, dependency injection, consistent interfaces

### Configuration as Code
```python
class Settings(BaseSettings):
    discord_token: Optional[str] = None
    max_messages: int = 10000
    base_filter: str = "channel_name NOT LIKE '%test%'..."
    
    model_config = SettingsConfigDict(env_file=".env")
```

**Benefits**: Type safety, validation, environment-specific configs

## Key Strengths

### 1. **Repository Pattern Benefits**
- **Testability**: Easy to mock repository methods for unit tests
- **Data Access Abstraction**: Business logic doesn't depend on SQL
- **Query Reuse**: Common queries centralized in repositories
- **Database Agnostic**: Can switch databases without changing analyzers

### 2. **Modern Testing Architecture**
- **Fast Execution**: Tests run in milliseconds with repository mocking
- **No Database Dependencies**: Tests don't require database setup
- **Isolated Testing**: Each analyzer tested independently
- **Comprehensive Coverage**: All edge cases and error conditions tested

### 3. **Async-First Design**
- **Full async/await**: End-to-end async support
- **Database Pooling**: Efficient connection management
- **Concurrent Analysis**: Multiple analyses can run simultaneously
- **Non-blocking Operations**: UI remains responsive during analysis

### 4. **Type Safety**
- **Pydantic Models**: Runtime validation and type safety
- **Type Hints**: Full typing throughout codebase
- **MyPy Compatible**: Static type checking
- **IDE Support**: Better autocomplete and error detection

### 5. **Modular Architecture**
- **Single Responsibility**: Each analyzer handles one domain
- **Loose Coupling**: Analyzers interact through well-defined interfaces
- **High Cohesion**: Related functionality grouped together
- **Easy Extension**: New analyzers follow established patterns

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

### Test Categories
- **Analysis Module Tests**: 28 test cases covering user, channel, topic, and temporal analysis
- **Repository Mocking**: Uses `unittest.mock.AsyncMock` for all repository methods
- **Edge Cases**: Easy to test error conditions and data scenarios
- **Coverage Quality**: Realistic mock data represents actual scenarios

### Test Performance
- **Test Suite**: 32 test files execute in under 2 seconds
- **No Database Setup**: Tests start immediately without fixtures
- **Parallel Execution**: Tests can run concurrently
- **Memory Efficient**: Mocked data is lightweight

**For detailed testing guidelines and examples, see [testing.md](testing.md)**

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
```