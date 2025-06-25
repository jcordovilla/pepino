# Testing Guidelines

## Overview

Pepino uses **repository mocking** for fast, reliable, and isolated tests. This document provides detailed guidelines for writing and maintaining tests.

**📋 For testing strategy overview, see [architecture.md](architecture.md)**

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
poetry run pytest tests/test_analysis/test_users.py

# Watch mode for development
poetry run ptw
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