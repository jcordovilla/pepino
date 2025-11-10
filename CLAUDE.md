# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

### Development Setup
```bash
# Complete development environment setup
make dev-setup

# Install dependencies only
poetry install

# Install development dependencies  
poetry install --dev
```

### Code Quality (Run before commits)
```bash
# Format and run fast tests (quick development cycle)
make dev

# Full quality checks (lint + format + type-check)
make lint

# Fix formatting issues
make format

# Type checking only
make type-check

# Full CI pipeline simulation
make ci
```

### Testing
```bash
# Run all tests
make test

# Fast tests without coverage
make test-fast

# Tests with coverage report
make test-cov

# Unit tests only
make test-unit

# Integration tests only
make test-integration
```

### Application Commands
```bash
# Run the Discord bot
poetry run pepino start

# CLI usage examples
poetry run pepino analyze users alice_dev --days 30
poetry run pepino sync run --progress
poetry run pepino list channels --format json
```

### Debugging and Logs
```bash
# View recent logs
make logs

# Follow logs in real-time
make logs-follow

# View error logs
make logs-errors

# Run in debug mode
make debug
```

## Architecture Overview

### Core Architecture Pattern
Pepino follows a **layered hexagonal architecture** with these key layers:

- **Domain Layer**: Data models (`src/pepino/data/models/`)
- **Data Access**: Repositories with base filtering (`src/pepino/data/repositories/`)
- **Analysis Engine**: Specialized analyzers (`src/pepino/analysis/`)
- **Interfaces**: Discord bot (`src/pepino/discord/`) and CLI (`src/pepino/cli/`)
- **Infrastructure**: Configuration, templates, logging

### Key Patterns

#### Repository Pattern
All data access goes through repositories in `src/pepino/data/repositories/`:
- `MessageRepository`: Message data with sophisticated filtering
- `UserRepository`: User-specific operations
- `ChannelRepository`: Channel data and statistics
- `EmbeddingRepository`: Vector embeddings for semantic analysis

#### Data Facade Pattern
`AnalysisDataFacade` (`src/pepino/analysis/data_facade.py`) provides:
- Centralized repository management with dependency injection
- Transaction support across operations
- Base filter management for consistent data filtering
- Clean interface for analyzers

#### Analyzer Strategy Pattern
Specialized analyzers in `src/pepino/analysis/`:
- `TopicAnalyzer`: Hybrid BERTopic + spaCy approach for topic modeling
- `UserAnalyzer`: User behavior and activity patterns
- `ChannelAnalyzer`: Channel statistics and engagement
- `TemporalAnalyzer`: Time-based activity analysis
- `EmbeddingAnalyzer`: Semantic similarity operations

### Database Schema
Comprehensive Discord data model in `src/pepino/data/database/schema.py`:
- **Messages table**: 100+ fields capturing complete Discord message data
- **Analysis tables**: Embeddings, topics, statistics, temporal data
- **Full-text search**: FTS5 virtual table for content search
- **Optimized indexes**: Performance-tuned for common query patterns

### Template Engine
Advanced Jinja2 integration (`src/pepino/templates/template_engine.py`):
- **Analyzer helpers**: Direct access to analyzers within templates
- **NLP functions**: Sentiment analysis, entity recognition in templates
- **Chart generation**: Matplotlib integration for visualizations
- **Unified output**: Same templates for CLI and Discord bot

## Discord Bot Architecture

### Command Structure
- **Slash commands** in `src/pepino/discord/commands/analysis.py`
- **Autocomplete functions** with fuzzy matching for user/channel names
- **Async thread pool execution** for CPU-intensive analysis
- **Template-driven responses** with rich formatting

### Data Sync Pipeline
Sophisticated sync system in `src/pepino/discord/sync/`:
- **Incremental sync**: Only updates stale data
- **Resilient handling**: Rate limits, permissions, errors
- **Batch processing**: Configurable batch sizes for efficiency
- **Extractors**: Specialized extractors for messages, emojis, interactions

## CLI Architecture

### Command Categories
- **Analysis**: `pepino analyze users|channels|topics|temporal`
- **Sync**: `pepino sync run|status`
- **Lists**: `pepino list channels|users|stats` 
- **Performance**: `pepino performance metrics|benchmark|profile`
- **Testing**: `pepino test data|analysis|templates|dependencies`

### Output Formats
CLI supports multiple formats: JSON, CSV, formatted text via templates

## Analysis Engine Deep Dive

### Topic Analysis Pipeline
Multi-stage pipeline in `TopicAnalyzer`:
1. **Quality assessment**: Determines BERTopic vs spaCy approach
2. **BERTopic integration**: UMAP + HDBSCAN clustering with `all-mpnet-base-v2`
3. **spaCy NLP**: Domain-specific pattern extraction
4. **Discord optimization**: Custom analysis for conversational data
5. **Post-processing**: Quality filtering, topic merging, coherence scoring

### Key Features
- **250+ Discord stop words**: Optimized for conversational data
- **Temporal analysis**: Emerging topic detection over time
- **Quality thresholds**: Strict filtering for meaningful results
- **Semantic embeddings**: Vector operations for similarity analysis

## Development Guidelines

### Code Organization
- **Type safety**: Full type hints with Pydantic models
- **Async/sync hybrid**: Discord bot is async, CLI is sync with thread pools
- **Error handling**: Graceful degradation with user-friendly messages
- **Configuration**: Pydantic settings with environment integration

### Testing Strategy
- **Unit tests**: Individual component testing
- **Integration tests**: End-to-end workflow testing
- **Fixtures**: Comprehensive test data in `tests/fixtures/`
- **Coverage**: Target 95%+ coverage with `make test-cov`

### Database Operations
- **Migrations**: Use scripts in `scripts/` directory
- **Backups**: `make backup-db` before schema changes
- **Transactions**: Always use data facade for multi-repository operations

### Adding New Features

#### New Analyzer
1. Create analyzer class in `src/pepino/analysis/`
2. Implement base analyzer interface
3. Register with data facade
4. Add templates for output formatting
5. Create CLI command and Discord command

#### New Discord Command
1. Add command in `src/pepino/discord/commands/`
2. Create corresponding template in `templates/outputs/discord/`
3. Test with both CLI and Discord bot interfaces

#### New Data Model
1. Add Pydantic model in `src/pepino/data/models/`
2. Update database schema if needed
3. Add repository methods
4. Update data facade if cross-repository access needed

### Performance Considerations
- **Database queries**: Use repository filters, not manual filtering
- **Large datasets**: Use pagination and streaming where possible
- **Analysis operations**: Use thread pools for CPU-intensive work
- **Memory usage**: Monitor with analysis timeout settings

### Security and Privacy
- **Bot tokens**: Always use environment variables
- **User data**: Follow Discord TOS for data handling
- **Database**: No real Discord data in tests or documentation
- **Logs**: Sensitive information excluded from logs

## Common Tasks

### Adding a New Analysis Command
```bash
# Example: Adding sentiment analysis
# 1. Create analyzer
touch src/pepino/analysis/sentiment_analyzer.py
# 2. Add CLI command  
# Edit src/pepino/cli/commands.py
# 3. Add Discord command
# Edit src/pepino/discord/commands/analysis.py
# 4. Create templates
touch templates/outputs/cli/sentiment_analysis.txt.j2
touch templates/outputs/discord/sentiment_analysis.md.j2
```

### Database Schema Changes
```bash
# 1. Backup database
make backup-db
# 2. Create migration script
touch scripts/migrate_v2_to_v3.py
# 3. Test migration
python scripts/migrate_v2_to_v3.py --dry-run
# 4. Apply migration
python scripts/migrate_v2_to_v3.py
```

### Debugging Analysis Issues
```bash
# 1. Check data availability
poetry run pepino list channels
# 2. Run analysis with debug
poetry run pepino analyze channels --channel general --debug
# 3. Check logs
make logs-errors
# 4. Validate templates
poetry run pepino test templates
```

This codebase emphasizes clean architecture, comprehensive testing, and maintainable code patterns. Always run `make quality` before committing, and use the existing patterns for consistency.