# Development Guide

This guide covers the architecture, design patterns, and development practices for Pepino.

## 🏗️ Architecture Overview

Pepino uses a **modular service architecture** with the repository pattern, built with Python 3.12+ and Poetry.

### Core Architecture

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

### Package Structure

```
src/pepino/
├── analysis/              # Analysis services
│   ├── services/         # Specialized analysis services
│   ├── helpers/          # Analysis helper functions
│   └── templates/        # Report templates
├── data/                 # Data layer
│   ├── repositories/     # Data access objects
│   └── database/         # Database management
├── discord_bot/          # Discord integration
├── cli/                  # Command-line interface
└── config.py             # Configuration management
```

## 🔧 Analysis Services Architecture

### Modularized Services

The original monolithic `AnalysisService` (800+ lines) was split into specialized services:

```
BaseAnalysisService (common functionality)
├── ChannelAnalysisService (channel operations)
├── UserAnalysisService (user operations)  
├── TopicAnalysisService (topic operations)
├── TemporalAnalysisService (temporal operations)
├── ServerAnalysisService (server operations)
└── DatabaseAnalysisService (database operations)

UnifiedAnalysisService (orchestrates all services)
```

### Service Benefits

1. **Single Responsibility**: Each service focuses on one domain
2. **Better Testability**: Independent testing of specialized services
3. **Easier Maintenance**: Changes isolated to specific domains
4. **Cleaner Organization**: Related functionality grouped together
5. **Backward Compatibility**: UnifiedAnalysisService maintains original interface
6. **Easy Extension**: New domains can be added without modifying existing code

### Service Contracts

All services follow the same pattern:
- **Constructor**: Takes optional `db_path` and `base_filter` parameters
- **Context Manager**: Supports `with` statements for resource management
- **Lazy Initialization**: Services only create dependencies when needed
- **Error Handling**: Consistent error handling across all services
- **Template Rendering**: Common template rendering functionality

## 🗄️ Repository Pattern

### Data Access Layer

The repository pattern provides clean data access with automatic filtering:

```python
class ChannelRepository:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.base_filter = settings.analysis_base_filter_sql.strip()
    
    def get_channel_message_statistics(self, channel_name: str, days: Optional[int] = None):
        query = f"""
        SELECT COUNT(*) as total_messages, ...
        FROM messages 
        WHERE channel_name = ? AND {self.base_filter}
        """
        # Automatically excludes bots and test channels
```

### Repository Benefits

- **Consistent Filtering**: All queries automatically exclude unwanted data
- **SQL Safety**: Validated base filter prevents injection
- **Testability**: Easy to mock repository methods for unit tests
- **Data Access Abstraction**: Business logic doesn't depend on SQL
- **Query Reuse**: Common queries centralized in repositories
- **Database Agnostic**: Can switch databases without changing analyzers

## ⚙️ Configuration Management

### Pydantic Settings

Centralized configuration with validation:

```python
class Settings(BaseSettings):
    discord_token: Optional[str] = None
    guild_id: Optional[str] = None
    database_url: str = "sqlite:///data/discord_messages.db"
    max_messages: int = 10000
    
    @field_validator("base_filter")
    @classmethod
    def validate_base_filter(cls, v):
        """Ensure base filter is properly formatted for SQL usage."""
        if not v or not v.strip():
            return "1=1"
        cleaned = v.strip()
        if cleaned.upper().startswith("AND "):
            cleaned = cleaned[4:]
        return cleaned
```

### Configuration Benefits

- **Type Safety**: Pydantic validation ensures data integrity
- **SQL Safety**: Automatic cleanup prevents syntax errors
- **Environment Integration**: Seamless .env file support
- **Error Prevention**: Prevents common configuration mistakes

## 📝 Logging Infrastructure

### Professional Logging

Structured logging with rotation and colored output:

```python
from pepino.logging_config import get_logger

logger = get_logger(__name__)

class ChannelAnalyzer:
    def analyze(self, channel_name: str):
        logger.info(f"Starting channel analysis for: {channel_name}")
```

### Logging Features

- **Colored Console Output**: Level-based color coding
- **File Rotation**: Automatic rotation with size limits (10MB, 5 backups)
- **Multiple Log Files**: Separate logs for application, errors, and Discord
- **Structured Formatting**: Detailed logging with module, function, and line information
- **Third-party Logger Management**: Reduced noise from external libraries

## 🔄 Data Sync Architecture

### Complete Sync Workflow

```
CLI/Bot → SyncManager → DiscordClient → Extractors → Repositories → Database
    ↓           ↓            ↓            ↓            ↓
SyncLogger → SyncLogger → SyncLogger → SyncLogger → SyncLogger
```

### Sync Components

1. **SyncManager**: Orchestrates the entire sync process
2. **DiscordClient**: Handles Discord API communication
3. **Extractors**: Extract message data and metadata
4. **Repositories**: Save data to database
5. **SyncLogger**: Track all sync operations and errors

### Sync Features

- **Guild and Channel Discovery**: Automatic discovery of accessible guilds and channels
- **Batch Processing**: Configurable batch sizes for efficient data processing
- **Error Recovery**: Graceful handling of API rate limits and errors
- **Progress Tracking**: Real-time sync progress with detailed logging
- **Data Validation**: Comprehensive data validation before database storage

## 🧪 Testing Strategy

### Repository Mocking

Fast, reliable tests using repository mocking:

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

- **Speed**: Tests execute quickly with repository mocking
- **Isolation**: Each test is completely independent
- **Reliability**: No database state dependencies or cleanup needed
- **Simplicity**: Focus on business logic rather than data setup

## 🚀 Adding New Features

### Adding New Analysis Types

1. **Create analyzer** in `src/pepino/analysis/helpers/`
2. **Add service** in `src/pepino/analysis/services/`
3. **Create templates** in `src/pepino/analysis/templates/`
4. **Add CLI command** in `src/pepino/cli/`
5. **Add Discord command** in `src/pepino/discord_bot/`
6. **Write tests** in `tests/unit/analysis/`

### Example: New Analysis Service

```python
class NewAnalysisService(BaseAnalysisService):
    def _create_analyzers(self) -> Dict[str, Any]:
        return {'new_analyzer': NewAnalyzer(self.data_facade)}
    
    def new_analysis_method(self, **kwargs) -> str:
        return self.render_template("new_analysis", **kwargs)
```

## 🔧 Development Workflow

### Code Quality

```bash
# Format code
poetry run black src/ tests/
poetry run isort src/ tests/

# Type checking
poetry run mypy src/

# Linting
poetry run flake8 src/ tests/

# Run tests
poetry run pytest
```

### Testing

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src/pepino

# Run specific test categories
poetry run pytest tests/unit/analysis/     # Unit tests
poetry run pytest tests/component/         # Integration tests
poetry run pytest tests/smoke/             # Template smoke tests
```

## 📊 Key Design Patterns

### 1. **Repository Pattern with Base Filtering**
- Consistent data filtering across all operations
- SQL injection prevention
- Single source of truth for filtering logic

### 2. **Configuration as Code with Validation**
- Type safety with Pydantic validation
- SQL safety with automatic cleanup
- Environment integration

### 3. **Centralized Logging Pattern**
- Consistent format across all modules
- Configurable levels
- Dual output (console and file)

### 4. **Service Layer Architecture**
- Business logic separation
- Single responsibility principle
- Easy testing and maintenance

### 5. **Async-First Design**
- Full async/await support
- Efficient connection management
- Non-blocking operations

## 🎯 Best Practices

### Code Organization
- **Single Responsibility**: Each class/module has one clear purpose
- **Dependency Injection**: Use dependency injection for testability
- **Error Handling**: Comprehensive error handling with graceful failures
- **Documentation**: Clear docstrings and type hints

### Performance
- **Lazy Loading**: Services only create dependencies when needed
- **Batch Processing**: Use batch operations for database operations
- **Connection Pooling**: Efficient database connection management
- **Caching**: Consider caching for expensive operations

### Security
- **Input Validation**: Validate all user inputs
- **SQL Safety**: Use parameterized queries only
- **Token Protection**: Never commit sensitive tokens
- **Permission Scoping**: Use minimum required permissions

This architecture provides a solid foundation for building scalable, maintainable Discord analytics applications. 🚀 