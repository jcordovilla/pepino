# Component Integration Tests

This directory contains comprehensive component integration tests that validate the validity of numbers in reports and functional correctness of the Pepino Discord analytics system.

**NOTE:** Component tests are now located in the `component/` subdirectory and are split by purpose:
- `component/test_discord_commands.py`: Discord bot command integration tests
- `component/test_analysis_service.py`: Analysis service integration tests
- `component/test_data_operations.py`: Data operations service integration tests
- `component/test_data_generator.py`: Test data generator
- `component/test_cleanup.py`: Artifact cleanup logic
- `component/conftest.py`: Shared fixtures for component tests

## Overview

The component tests are designed to:

1. **Bring up a test SQLite database** with generated test data based on the actual data in `discord_messages.db`
2. **Simulate Discord bot received slash commands** and test the results
3. **Test the analysis and data operation services** as primary core interfaces
4. **Validate that delivered data is correct** and mathematically accurate

## Test Components

### 1. Test Data Generator (`test_data_generator.py`)

Generates realistic test data based on patterns extracted from the actual Discord database.

**Features:**
- Extracts channel statistics, user statistics, and temporal patterns from real data
- Generates test messages with realistic distribution across channels and users
- Creates test database with proper schema and relationships
- Maintains data consistency with actual Discord patterns

**Usage:**
```bash
# Generate test data from actual database (part of make test-component)
cd tests/component && python test_data_generator.py --source-db ../../discord_messages.db
```

### 2. Component Integration Tests (`test_component_integration.py`)

Comprehensive pytest-based tests that validate all system components.

**Test Categories:**
- **Database Population**: Validates test data generation and insertion
- **Analysis Service Tests**: Tests all analysis service methods
- **Discord Bot Command Tests**: Simulates slash command execution
- **Data Operations Tests**: Validates data operations service
- **Number Validity Tests**: Ensures mathematical correctness of reports
- **Data Consistency Tests**: Validates consistency across services
- **Error Handling Tests**: Tests error scenarios and edge cases
- **Performance Tests**: Validates performance with larger datasets

**Usage:**
```bash
# Run component integration tests (part of make test-component)
cd tests/component && pytest
```

### 3. Analysis Service Validation (`test_analysis_validation.py`)

Direct validation of analysis service against the actual database.

**Validation Areas:**
- Channel analysis accuracy
- User analysis accuracy
- Pulsecheck functionality
- Data consistency across services
- Number validity in reports

**Usage:**
```bash
# Validate analysis service against actual database (part of make test-component)
cd tests/component && python test_analysis_validation.py
```

### 4. Test Runner (`make test-component`)

Orchestrates the entire testing process with detailed reporting.

**Process:**
1. Generates test data from actual database
2. Validates analysis service against real database
3. Runs automated component tests
4. Cleans up all artifacts automatically

## Running the Tests

### Quick Start

```bash
# Run complete component integration tests (generates data, validates against real DB, runs all tests)
make test-component
```

### Manual Execution

```bash
# Run the complete workflow manually
cd tests/component
python test_data_generator.py --source-db ../../discord_messages.db
python test_analysis_validation.py
pytest
```

### Individual Test Execution

```bash
# Run specific test file
cd tests/component && pytest test_analysis_service.py -v

# Run specific test method
cd tests/component && pytest test_analysis_service.py::test_analysis_service_pulsecheck -v

# Run Discord command tests only
cd tests/component && pytest test_discord_commands.py -v
```

## Test Data

### Generated Test Data

The test data generator creates:

- **Messages**: 1000+ realistic messages with proper distribution
- **Channels**: Top 10 channels from actual data
- **Users**: Top 20 users from actual data
- **Temporal Patterns**: Based on actual activity patterns
- **Content Patterns**: Based on actual message characteristics

### Data Validation

The tests validate:

- **Message Counts**: Ensure reported numbers match actual database counts
- **User Activity**: Validate user contribution rankings
- **Channel Activity**: Validate channel popularity rankings
- **Temporal Accuracy**: Ensure date-based filtering works correctly
- **Data Relationships**: Validate foreign key relationships

## Validation Criteria

### Number Validity

Tests ensure that:

1. **Message counts** in reports match actual database counts
2. **User rankings** reflect actual message activity
3. **Channel rankings** reflect actual message distribution
4. **Percentage calculations** are mathematically correct
5. **Date filtering** produces accurate results

### Functional Correctness

Tests validate that:

1. **Analysis Service** methods return expected data
2. **Discord Bot Commands** execute successfully
3. **Data Operations Service** provides consistent results
4. **Error Handling** works for edge cases
5. **Performance** remains acceptable with larger datasets

### Data Consistency

Tests ensure that:

1. **Multiple services** return consistent data
2. **Different analysis methods** produce compatible results
3. **Database queries** are consistent across repositories
4. **Template rendering** includes all expected data

## Test Reports

### Generated Reports

The test runner generates:

- **Console Output**: Real-time test progress and results
- **Component Test Report**: Detailed test execution summary
- **Analysis Validation Results**: JSON file with validation details
- **Test Data Statistics**: Information about generated test data

### Report Contents

Reports include:

- **Test Execution Summary**: Duration, success/failure status
- **Test Data Validation**: Database content verification
- **Manual Validation Results**: Service functionality checks
- **Channel Distribution**: Top channels and message counts
- **Recommendations**: Suggestions for improvement

## Troubleshooting

### Common Issues

1. **Database Not Found**: Ensure `discord_messages.db` exists in project root
2. **Import Errors**: Ensure all dependencies are installed (`poetry install`)
3. **Test Data Generation Fails**: Check database schema and permissions
4. **Analysis Service Errors**: Verify database path and configuration

### Debug Mode

```bash
# Run tests with verbose output
pytest test_component_integration.py -v -s

# Run specific test with debug output
python test_analysis_validation.py --debug

# Check test data generation
python test_data_generator.py --verbose
```

## Integration with CI/CD

### GitHub Actions

The component tests can be integrated into CI/CD pipelines:

```yaml
- name: Run Component Tests
  run: |
    make test-component
    make test-analysis-validation
```

### Pre-commit Hooks

Add to pre-commit configuration:

```yaml
- repo: local
  hooks:
    - id: component-tests
      name: Component Integration Tests
      entry: make test-component
      language: system
      pass_filenames: false
```

## Performance Considerations

### Test Data Size

- **Default**: 1000 messages (suitable for most testing)
- **Large Dataset**: 10,000+ messages (performance testing)
- **Memory Usage**: ~50MB for default test data

### Execution Time

- **Test Data Generation**: 5-10 seconds
- **Component Tests**: 30-60 seconds
- **Analysis Validation**: 10-20 seconds
- **Total**: 1-2 minutes

### Optimization

- **Parallel Execution**: Tests can run in parallel where possible
- **Database Caching**: Test database is reused across test runs
- **Selective Testing**: Run specific test categories as needed

## Contributing

### Adding New Tests

1. **Test Data**: Add realistic test data patterns to `test_data_generator.py`
2. **Component Tests**: Add new test methods to `TestComponentIntegration`
3. **Validation**: Add validation logic to `AnalysisValidator`
4. **Documentation**: Update this README with new test descriptions

### Test Guidelines

1. **Realistic Data**: Use patterns from actual Discord data
2. **Comprehensive Coverage**: Test all major functionality
3. **Edge Cases**: Include error conditions and boundary cases
4. **Performance**: Ensure tests complete within reasonable time
5. **Documentation**: Document test purpose and validation criteria

## Dependencies

### Required Packages

- `pytest`: Test framework
- `pytest-asyncio`: Async test support
- `colorama`: Colored console output
- `sqlite3`: Database operations
- `discord.py`: Discord bot simulation

### Optional Packages

- `pytest-cov`: Coverage reporting
- `pytest-watch`: Test watching
- `pytest-xdist`: Parallel test execution

## Support

For issues with component tests:

1. **Check Logs**: Review test output and error messages
2. **Validate Data**: Ensure test data generation succeeded
3. **Database Issues**: Verify database schema and permissions
4. **Dependencies**: Ensure all required packages are installed
5. **Configuration**: Check test configuration and paths 