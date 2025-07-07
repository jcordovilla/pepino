# Contributing to Pepino

Thank you for your interest in contributing to Pepino! This guide will help you get started with development and understand our contribution process.

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

# Install dependencies
poetry install

# Install optional dependencies
poetry run python scripts/install_spacy.py

# Set up environment
cp .env.example .env
# Edit .env with your Discord token

# Run tests to verify setup
poetry run pytest
```

## 🏗️ Architecture Overview

Pepino uses a **modular service architecture** with the repository pattern:

```
src/pepino/
├── analysis/          # Analysis services (user, channel, topic, etc.)
├── data/             # Data layer (repositories, database)
├── discord_bot/      # Discord bot functionality
├── cli/              # Command-line interface
└── config.py         # Configuration management
```

### Key Design Patterns
- **Repository Pattern**: Clean data access with base filtering
- **Service Layer**: Business logic separation
- **Template Engine**: Jinja2-based report generation
- **Async/Await**: Full async support throughout

## 🧪 Testing Strategy

We use **repository mocking** for fast, reliable tests:

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

### Testing Guidelines
- **Mock repositories, not databases** for fast execution
- **Test both success and failure cases**
- **Use realistic test data** that matches production patterns
- **Maintain good test coverage** on analysis modules

**Note**: Current test coverage is around 28%. Focus on improving coverage for core analysis modules first.

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

### Commit Messages
Use [Conventional Commits](https://www.conventionalcommits.org/):
```
feat: add new user analysis feature
fix: resolve template rendering issue
docs: update installation instructions
test: add coverage for edge cases
refactor: simplify repository pattern
```

### Pull Request Process
1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes with tests
4. **Run** all quality checks
5. **Submit** a pull request
6. **Address** review feedback
7. **Merge** when approved

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

## 🏛️ Project Structure

### Key Directories
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
└── cli/                  # Command-line interface

tests/
├── unit/                 # Unit tests with mocking
├── component/            # Integration tests
└── smoke/                # Template rendering tests
```

### Adding New Analysis Types
1. **Create analyzer** in `src/pepino/analysis/helpers/`
2. **Add service** in `src/pepino/analysis/services/`
3. **Create templates** in `src/pepino/analysis/templates/`
4. **Add CLI command** in `src/pepino/cli/`
5. **Add Discord command** in `src/pepino/discord_bot/`
6. **Write tests** in `tests/unit/analysis/`

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

## 📚 Resources

### Documentation
- [Architecture Guide](docs/development.md) - System design and patterns
- [Testing Guide](docs/testing.md) - Testing strategy and guidelines
- [Operations Guide](docs/operations.md) - Setup and deployment

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

## 🎯 Getting Help

- **Check existing issues** before creating new ones
- **Search documentation** for answers
- **Ask in Discussions** for general questions
- **Be specific** about your problem

Thank you for contributing to Pepino! 🚀 