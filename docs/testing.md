# Testing Guide

## Overview

Pepino uses **repository mocking** for fast, reliable, and isolated tests. This document provides comprehensive testing guidelines and strategies.

## 🧪 Testing Strategy

### Repository Mocking Approach

We use **repository mocking** instead of database fixtures for fast, reliable, and isolated tests. This eliminates database dependencies while providing comprehensive coverage of business logic.

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

## Writing New Tests

### Basic Test Pattern
```python
@pytest.mark.asyncio
async def test_new_feature():
    """Test description"""
    # Setup
    mock_db_manager = MagicMock()
    analyzer = SomeAnalyzer(mock_db_manager)
    
    # Mock repository methods
    with patch.object(analyzer.repo, 'method', new_callable=AsyncMock) as mock_method:
        mock_method.return_value = expected_data
        
        # Execute and verify
        result = await analyzer.analyze()
        assert result["success"] is True
```

### Testing Both Success and Failure Cases
```python
# Success case
@pytest.mark.asyncio
async def test_analyzer_success():
    mock_db_manager = MagicMock()
    analyzer = UserAnalyzer(mock_db_manager)
    
    with patch.object(analyzer.user_repo, 'find_user_by_name', new_callable=AsyncMock) as mock_find:
        mock_find.return_value = {"author_id": "123", "display_name": "Test User"}
        
        result = await analyzer.analyze(user_name="test_user")
        assert result["success"] is True

# Failure case
@pytest.mark.asyncio
async def test_analyzer_repository_error():
    mock_db_manager = MagicMock()
    analyzer = UserAnalyzer(mock_db_manager)
    
    with patch.object(analyzer.user_repo, 'find_user_by_name', new_callable=AsyncMock) as mock_method:
        mock_method.side_effect = Exception("Database error")
        
        result = await analyzer.analyze(user_name="test_user")
        assert result["success"] is False
        assert "Database error" in result["error"]
```

### Verifying Repository Calls
```python
# Verify method was called with correct parameters
mock_method.assert_called_once_with(
    expected_param1,
    expected_param2,
    limit=10
)

# Verify method was called specific number of times
assert mock_method.call_count == 2
```

## Common Testing Patterns

### Multiple Repository Mocking
```python
with patch.object(analyzer.message_repo, 'get_stats', new_callable=AsyncMock) as mock_stats, \
     patch.object(analyzer.user_repo, 'get_users', new_callable=AsyncMock) as mock_users:
    
    mock_stats.return_value = stats_data
    mock_users.return_value = users_data
    
    result = await analyzer.analyze()
```

### Conditional Mocking
```python
# Mock different return values based on parameters
def mock_find_user(name):
    if name == "existing_user":
        return {"author_id": "123", "display_name": "User"}
    return None

mock_method.side_effect = mock_find_user
```

### Parameterized Testing
```python
@pytest.mark.parametrize("granularity,expected_calls", [
    ("hour", 1),
    ("day", 1),
    ("week", 1)
])
@pytest.mark.asyncio
async def test_temporal_analyzer_granularity(granularity, expected_calls):
    # Test implementation
    pass
```

## Test Data Design

### Mock Data Principles
Mock data should be:
- **Realistic**: Match actual data structures
- **Predictable**: Enable deterministic testing
- **Comprehensive**: Cover various scenarios

```python
# Example: Mock user data
mock_user_info = {
    "author_id": "user123",
    "display_name": "Alice Smith",
    "author_name": "Alice"
}

mock_stats = {
    "total_messages": 150,
    "channels_active": 5,
    "avg_message_length": 45.5,
    "active_days": 25,
    "first_message": "2024-01-01T12:00:00Z",
    "last_message": "2024-12-01T12:00:00Z"
}
```

## Running Tests

### Basic Execution
```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src/pepino

# Run specific test file
poetry run pytest tests/unit/analysis/test_users.py

# Watch mode for development
poetry run ptw
```

### Test Categories

```bash
# Unit tests (fast, mocked)
poetry run pytest tests/unit/

# Component tests (integration)
poetry run pytest tests/component/

# Smoke tests (template rendering)
poetry run pytest tests/smoke/

# All tests with coverage
poetry run pytest --cov=src/pepino --cov-report=html
```

### Coverage Reports
```bash
# Generate HTML coverage report
poetry run pytest --cov=src/pepino --cov-report=html

# View coverage in browser
open htmlcov/index.html

# Generate coverage summary
poetry run pytest --cov=src/pepino --cov-report=term-missing
```

### Coverage Improvement Plan

Current coverage is around 28%. To improve coverage:

1. **Priority Areas**:
   - Analysis services (core business logic)
   - Repository layer (data access)
   - CLI commands (user interface)

2. **Testing Strategy**:
   - Focus on unit tests for business logic
   - Add integration tests for data flow
   - Include edge cases and error handling

3. **Coverage Goals**:
   - Analysis modules: 80%+
   - Repository layer: 70%+
   - CLI interface: 60%+

## 🔍 Test Types

### Unit Tests (`tests/unit/`)
Fast, isolated tests using repository mocking:
- **Purpose**: Test individual components in isolation
- **Speed**: Execute quickly
- **Coverage**: Good coverage on analysis modules
- **Pattern**: Mock repositories, test business logic

### Component Tests (`tests/component/`)
Integration tests with realistic data:
- **Purpose**: Test component interactions and data flow
- **Data**: Generated test data based on real patterns
- **Validation**: Ensure mathematical correctness of reports
- **Coverage**: End-to-end functionality testing

### Smoke Tests (`tests/smoke/`)
Template rendering validation:
- **Purpose**: Ensure templates render consistently
- **Golden Outputs**: Compare against expected results
- **Coverage**: All CLI and Discord templates
- **Regression Detection**: Catch template changes

## 📊 Test Data

### Mock Data Design
Mock data should be:
- **Realistic**: Match actual data structures
- **Predictable**: Enable deterministic testing
- **Comprehensive**: Cover various scenarios

```python
# Example: Mock user data
mock_user_info = {
    "author_id": "user123",
    "display_name": "Alice Smith",
    "author_name": "Alice"
}

mock_stats = {
    "total_messages": 150,
    "channels_active": 5,
    "avg_message_length": 45.5,
    "active_days": 25,
    "first_message": "2024-01-01T12:00:00Z",
    "last_message": "2024-12-01T12:00:00Z"
}
```

### Component Test Data
Component tests use generated test data based on real Discord patterns:
- **Messages**: 1000+ realistic messages with proper distribution
- **Channels**: Top 10 channels from actual data
- **Users**: Top 20 users from actual data
- **Temporal Patterns**: Based on actual activity patterns

## Best Practices

### 1. Mock at the Right Level
- **Mock repositories, not database connections**
- **Mock external dependencies, not internal methods**
- **Keep mocks simple and focused**

### 2. Use Realistic Test Data
- **Match actual data structures**
- **Include edge cases in test data**
- **Use consistent data patterns**

### 3. Verify Behavior, Not Implementation
- **Test return values and side effects**
- **Verify repository methods are called correctly**
- **Don't test internal implementation details**

### 4. Keep Tests Independent
- **Each test should be self-contained**
- **No shared state between tests**
- **Clear setup and teardown**

## Development Workflow

### Pre-commit Testing
```bash
# Run before committing changes
poetry run pytest                     # Run tests
poetry run pytest --cov=src/pepino   # Check coverage
```

### Testing New Features
1. **Write Tests First**: Create tests using repository mocking
2. **Implement Feature**: Add analyzer or repository methods
3. **Verify Coverage**: Ensure comprehensive test coverage
4. **Integration Test**: Test with actual CLI/Discord commands

This testing approach ensures high code quality while keeping the development cycle fast and reliable. 