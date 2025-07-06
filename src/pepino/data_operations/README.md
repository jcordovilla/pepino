# Data Operations Package

The `data_operations` package provides a clean interface for Discord data synchronization and export operations, separate from analysis operations.

## Overview

This package handles:
- **Data Synchronization**: Sync Discord data to local database via `discord_sync` module
- **Data Export**: Export data in various formats (CSV, JSON, Excel)
- **Database Management**: Clear database, get sync status

## Package Structure

```
data_operations/
├── service.py              # Main service interface
├── exporters.py            # Data export functionality
├── sync_operations.py      # Sync operations wrapper
└── discord_sync/           # Discord-specific sync implementation
    ├── sync_manager.py     # Discord sync orchestration
    ├── discord_client.py   # Discord API client
    └── models.py           # Sync data models
```

## Quick Start

### Basic Usage

```python
from pepino.data_operations.service import data_operations_service

# Using context manager
with data_operations_service() as service:
    # Export data
    result = service.export_data(table="messages", format="csv")
    print(result)
    
    # Get sync status
    status = await service.get_sync_status()
    print(status)
```

### CLI Usage

```bash
# Export data (legacy command - use data export instead)
pepino export-data --table messages --format csv --output data.csv
pepino export-data --format json --output all_data.json

# Data operations
pepino data tables                    # List available tables
pepino data schema messages           # Show table schema
pepino data export messages --format csv  # Export specific table

# Data operations
pepino data sync --force               # Force sync
pepino data sync --full --clear        # Full re-sync
pepino data sync-status                # Show sync status
pepino data clear --confirm            # Clear database
```

## API Reference

### DataOperationsService

The main service class for data operations.

#### Sync Operations

```python
# Run sync operations
result = await service.sync_data(force=False, full=False, clear_existing=False)

# Get sync status
status = await service.get_sync_status()

# Clear database
service.clear_database()
```

#### Export Operations

```python
# Export data
result = service.export_data(
    table="messages",           # Specific table or None for all
    output_path="data.csv",     # File path or None for stdout
    format="csv",               # csv, json, excel, text
    include_metadata=True       # Include schema and metadata
)

# Export specific table
result = service.export_table(
    table="messages",
    output_path="messages.csv",
    format="csv",
    filters=None               # Optional filters
)

# Get available tables
tables = service.get_available_tables()

# Get table schema
schema = service.get_table_schema("messages")
```

## Supported Formats

### CSV Export
```python
result = service.export_data(table="messages", format="csv")
# Returns CSV string or writes to file
```

### JSON Export
```python
result = service.export_data(table="messages", format="json")
# Returns JSON with metadata and data
```

### Excel Export
```python
result = service.export_data(table="messages", format="excel")
# Requires pandas and openpyxl
```

### Text Export
```python
result = service.export_data(table="messages", format="text")
# Returns formatted text output
```

## Available Tables

- **messages**: Discord messages with metadata
- **users**: Discord users and their attributes  
- **channels**: Discord channels and their guilds
- **sync_logs**: Sync operation logs

## Error Handling

The service provides comprehensive error handling:

```python
try:
    result = service.export_data(table="messages", format="csv")
except Exception as e:
    print(f"Export failed: {e}")
```

## Performance Considerations

- **Large datasets**: Use file output instead of stdout
- **Memory usage**: Export tables individually for large databases
- **Sync operations**: Can take several minutes for full syncs

## Examples

See `examples/data_operations_example.py` for comprehensive usage examples.

## Integration with Analysis

The data operations package is separate from analysis operations:

```python
# Data operations (sync/export)
from pepino.data_operations.service import data_operations_service

# Analysis operations (reports/insights)  
from pepino.analysis.service import analysis_service

# Use both together
with data_operations_service() as data_service:
    with analysis_service() as analysis_service:
        # Export data
        data_service.export_data(table="messages", format="csv")
        
        # Generate analysis
        analysis_service.pulsecheck(channel="general")
``` 