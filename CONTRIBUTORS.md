# Contributing to Pepino

Thank you for your interest in contributing to Pepino! This guide provides the essential information to get started with development.

## 🚀 Quick Start

### Prerequisites
- **Python 3.12+**
- **Poetry** (dependency management)
- **Git**

### Development Setup
```bash
# Clone the repository
git clone https://github.com/your-username/pepino.git
cd pepino

# Complete setup (installs everything)
make dev-setup

# Or manual setup
poetry install
poetry run python scripts/install_spacy.py
cp .env.example .env
# Edit .env with your Discord token

# Verify setup
poetry run pytest
```

## 📚 Essential Documentation

Before diving into development, review these guides:

- **[Development Guide](docs/development.md)** - Architecture, design patterns, and technical details
- **[Testing Guide](docs/testing.md)** - Testing strategy, patterns, and guidelines  
- **[Operations Guide](docs/operations.md)** - Setup, configuration, and deployment

## 🔧 Development Workflow

### 1. Create Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes
- Follow existing code patterns
- Add tests for new functionality
- Update documentation if needed

### 3. Code Quality Checks
```bash
# Format and lint
make lint-fix

# Type checking
poetry run mypy src/

# Run tests
poetry run pytest
```

### 4. Commit Changes
```bash
git add .
git commit -m "feat: add new analysis feature

- Added user engagement metrics
- Updated templates for new data
- Added comprehensive tests"
```

### 5. Submit Pull Request
- **Clear title** describing the change
- **Detailed description** of what and why
- **Link related issues** if applicable
- **Include test results** and coverage

## 📋 Contribution Guidelines

### Code Style
- **Black** for code formatting
- **isort** for import sorting
- **Type hints** for all function parameters and returns
- **Docstrings** for public methods

### Testing Requirements
- **Mock repositories, not databases** for fast execution
- **Test both success and failure cases**
- **Use realistic test data** that matches production patterns
- **Maintain good test coverage** on analysis modules

**Note**: Current test coverage is around 28%. Focus on improving coverage for core analysis modules first.

### Adding New Analysis Types
1. **Create analyzer** in `src/pepino/analysis/helpers/`
2. **Add service** in `src/pepino/analysis/services/`
3. **Create templates** in `src/pepino/analysis/templates/`
4. **Add CLI command** in `src/pepino/cli/`
5. **Add Discord command** in `src/pepino/discord_bot/`
6. **Write tests** in `tests/unit/analysis/`

## 🐛 Bug Reports

When reporting bugs, please include:
- **Environment details** (OS, Python version, dependencies)
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Error messages** and stack traces
- **Sample data** if applicable

## 💡 Feature Requests

For new features:
- **Describe the problem** you're solving
- **Explain the proposed solution**
- **Consider implementation complexity**
- **Discuss alternatives** if applicable



## 🤝 Community Guidelines

### Code of Conduct
- **Be respectful** and inclusive
- **Help others** learn and grow
- **Focus on the code** and ideas
- **Assume good intentions**

### Communication
- **GitHub Issues** for bugs and features
- **GitHub Discussions** for questions and ideas
- **Pull Requests** for code contributions
- **Clear and constructive** feedback

## 🎯 Getting Help

- **Check existing issues** before creating new ones
- **Search documentation** for answers
- **Ask in Discussions** for general questions
- **Be specific** about your problem

## 📖 Additional Resources

### Development Tools
- **Poetry** - Dependency management
- **Pytest** - Testing framework
- **Black** - Code formatting
- **MyPy** - Type checking
- **Flake8** - Linting

### Discord Development
- [Discord.py Documentation](https://discordpy.readthedocs.io/)
- [Discord Developer Portal](https://discord.com/developers/docs)
- [Discord Bot Permissions](https://discord.com/developers/docs/topics/permissions)

Thank you for contributing to Pepino! 🚀 