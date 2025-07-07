# Changelog

All notable changes to Pepino will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Sync-enabled analysis workflow** - Fresh data analysis with automatic sync
- **Chart generation** - Visual charts for user activity and channel analysis
- **Detailed analysis commands** - Comprehensive user, topic, and temporal analysis
- **Template contract tests** - Automated template rendering validation
- **Repository pattern** - Clean data access with base filtering
- **Professional logging** - Structured logging with rotation and colored output

### Changed
- **Modularized analysis services** - Split monolithic AnalysisService into specialized services
- **Configuration management** - Unified Pydantic settings with validation
- **Testing strategy** - Repository mocking for fast, reliable tests
- **Documentation structure** - Streamlined OSS documentation

### Fixed
- **Top contributors analysis** - Now shows real channel activity and top messages
- **Template rendering** - Fixed CLI vs Discord template alignment
- **Reply counting logic** - Only shows "no direct replies" when truly no replies exist
- **Data filtering** - Consistent bot and test channel exclusion

## [0.2.0] - 2024-12-XX

### Added
- **Discord bot integration** - Slash commands for real-time analysis
- **CLI interface** - Command-line analysis tools
- **Data synchronization** - Discord message sync with logging
- **Template engine** - Jinja2-based report generation
- **Basic analysis features** - User, channel, and topic analysis

### Changed
- **Architecture refactor** - Migrated from monolithic script to modular package
- **Dependency management** - Switched to Poetry with pyproject.toml
- **Database access** - Implemented repository pattern
- **Configuration** - Centralized settings management

### Fixed
- **Data consistency** - Proper message and user data handling
- **Error handling** - Comprehensive error management
- **Performance** - Optimized database queries

## [0.1.0] - 2024-XX-XX

### Added
- **Initial release** - Basic Discord analytics functionality
- **Message analysis** - Core message processing and statistics
- **User tracking** - User activity and engagement metrics
- **Channel analysis** - Channel-specific statistics and trends

---

## Version History

### Version 0.2.0
- **Major architectural improvements** with modular service design
- **Professional OSS structure** with comprehensive testing
- **Enhanced user experience** with Discord bot and CLI interfaces
- **Production-ready** with proper error handling and logging

### Version 0.1.0
- **Foundation release** with core analytics capabilities
- **Basic Discord integration** for message processing
- **Simple analysis features** for user and channel insights

## Migration Notes

### From 0.1.0 to 0.2.0
- **Breaking changes** in import structure (see [migration guide](docs/migration.md))
- **New configuration** system with .env file support
- **Updated CLI commands** with new syntax and options
- **Enhanced Discord bot** with slash commands instead of prefix commands

## Contributing

When adding entries to this changelog, please follow these guidelines:

- **Use present tense** ("Add feature" not "Added feature")
- **Reference issues and pull requests** when applicable
- **Group changes** by type (Added, Changed, Deprecated, Removed, Fixed, Security)
- **Keep entries concise** but descriptive
- **Include breaking changes** prominently 