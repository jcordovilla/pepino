"""
Data Exporters

Handles data export operations with support for multiple formats using Pydantic for structured exports.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

from pydantic import BaseModel

from pepino.data.database.manager import DatabaseManager
from pepino.data.repositories import MessageRepository, UserRepository, ChannelRepository, SyncRepository

logger = logging.getLogger(__name__)


class ExportData(BaseModel):
    """Structured export data model."""
    
    table: str
    record_count: int
    exported_at: datetime
    data: List[Dict[str, Any]]
    schema: Optional[Dict[str, Any]] = None


class DataExporter:
    """
    Handles data export operations with support for multiple formats using Pydantic.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize data exporter.
        
        Args:
            db_path: Database path
        """
        self.db_path = db_path
        self._db_manager = None
        
    @property
    def db_manager(self) -> DatabaseManager:
        """Get database manager instance (lazy initialization)."""
        if self._db_manager is None:
            self._db_manager = DatabaseManager(self.db_path)
        return self._db_manager
    

    
    def get_available_tables(self) -> List[str]:
        """
        Get list of available tables for export.
        
        Returns:
            List of available table names
        """
        return ['messages', 'users', 'channels', 'sync_logs']
    
    def get_table_schema(self, table: str) -> Dict[str, Any]:
        """
        Get schema information for a specific table.
        
        Args:
            table: Table name
            
        Returns:
            Table schema information
        """
        schemas = {
            'messages': {
                'columns': ['id', 'message_id', 'channel_id', 'channel_name', 'author_id', 
                           'author_name', 'content', 'timestamp', 'edited_timestamp', 
                           'attachments', 'embeds', 'reactions', 'mentions'],
                'description': 'Discord messages with metadata'
            },
            'users': {
                'columns': ['id', 'user_id', 'user_name', 'user_display_name', 
                           'user_joined_at', 'user_roles', 'is_bot'],
                'description': 'Discord users and their attributes'
            },
            'channels': {
                'columns': ['id', 'channel_id', 'channel_name', 'guild_id', 'guild_name'],
                'description': 'Discord channels and their guilds'
            },
            'sync_logs': {
                'columns': ['id', 'sync_start_time', 'sync_end_time', 'total_messages_synced',
                           'guilds_synced', 'completed_at'],
                'description': 'Sync operation logs'
            }
        }
        
        return schemas.get(table, {'columns': [], 'description': 'Unknown table'})
    
    def export_data(self, table: Optional[str] = None, output_path: Optional[str] = None,
                   format: str = "csv", include_metadata: bool = True) -> str:
        """
        Export data from the database.
        
        Args:
            table: Specific table to export (None for all)
            output_path: Output file path (None for stdout)
            format: Output format (csv, json, excel)
            include_metadata: Include export metadata
            
        Returns:
            Export result information
        """
        if table:
            return self.export_table(table, output_path, format, include_metadata=include_metadata)
        else:
            return self.export_all_tables(output_path, format, include_metadata=include_metadata)
    
    def export_table(self, table: str, output_path: Optional[str] = None,
                    format: str = "csv", filters: Optional[Dict[str, Any]] = None,
                    include_metadata: bool = True) -> str:
        """
        Export specific table with optional filters.
        
        Args:
            table: Table name to export
            output_path: Output file path (None for stdout)
            format: Output format (csv, json, excel)
            filters: Optional filters to apply
            include_metadata: Include export metadata
            
        Returns:
            Export result information
        """
        try:
            # Get data from repository
            data = self._get_table_data(table, filters)
            
            if not data:
                return f"No data found for table '{table}'"
            
            # Prepare output
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                if format == "csv":
                    self._export_to_csv(data, output_path, table)
                elif format == "json":
                    self._export_to_json(data, output_path, table, include_metadata)
                elif format == "excel":
                    self._export_to_excel(data, output_path, table)
                else:
                    raise ValueError(f"Unsupported format: {format}")
                
                return f"Data exported to {output_path}"
            else:
                # Return formatted data for stdout
                if format == "csv":
                    return self._format_as_csv(data)
                elif format == "json":
                    return self._format_as_json(data, include_metadata)
                else:
                    return self._format_as_text(data, table)
                    
        except Exception as e:
            logger.error(f"Export failed for table '{table}': {e}")
            return f"Export failed: {e}"
    
    def export_all_tables(self, output_path: Optional[str] = None, 
                         format: str = "json", include_metadata: bool = True) -> str:
        """
        Export all tables.
        
        Args:
            output_path: Output file path (None for stdout)
            format: Output format (csv, json, excel)
            include_metadata: Include export metadata
            
        Returns:
            Export result information
        """
        all_data = {}
        
        for table in self.get_available_tables():
            try:
                data = self._get_table_data(table)
                if data:
                    all_data[table] = data
            except Exception as e:
                logger.warning(f"Could not export table '{table}': {e}")
                all_data[table] = []
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == "json":
                self._export_all_to_json(all_data, output_path, include_metadata)
            elif format == "excel":
                self._export_all_to_excel(all_data, output_path)
            else:
                raise ValueError(f"Unsupported format for all tables: {format}")
            
            return f"All tables exported to {output_path}"
        else:
            if format == "json":
                return self._format_all_as_json(all_data, include_metadata)
            else:
                return self._format_all_as_text(all_data)
    
    def _get_table_data(self, table: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Get data from specific table with optional filters."""
        try:
            if table == "messages":
                repo = MessageRepository(self.db_manager)
                # Get recent messages as a sample for export
                messages = repo.get_recent_messages(limit=10000, days_back=365)
                # Convert to dict format for export
                return [msg if isinstance(msg, dict) else msg.__dict__ for msg in messages]
            elif table == "users":
                repo = UserRepository(self.db_manager)
                # Get available users and their statistics
                users = repo.get_available_users(limit=1000)
                # Convert to dict format for export
                return [{"user_name": user} for user in users]
            elif table == "channels":
                repo = ChannelRepository(self.db_manager)
                # Get all channels
                channels = repo.get_all_channels()
                # Convert to dict format for export
                return [channel.__dict__ if hasattr(channel, '__dict__') else {"channel_name": channel} for channel in channels]
            elif table == "sync_logs":
                repo = SyncRepository(self.db_manager)
                return repo.get_all_sync_logs(limit=100)
            else:
                raise ValueError(f"Unknown table: {table}")
        except Exception as e:
            logger.error(f"Error getting data for table '{table}': {e}")
            return []
    
    def _export_to_csv(self, data: List[Dict[str, Any]], output_path: Path, table: str):
        """Export data to CSV file."""
        if not data:
            return
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if data:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
    
    def _export_to_json(self, data: List[Dict[str, Any]], output_path: Path, 
                       table: str, include_metadata: bool):
        """Export data to JSON file using Pydantic."""
        export_data = ExportData(
            table=table,
            exported_at=datetime.now(),
            record_count=len(data),
            data=data,
            schema=self.get_table_schema(table) if include_metadata else None
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(export_data.model_dump_json(indent=2))
    
    def _export_to_excel(self, data: List[Dict[str, Any]], output_path: Path, table: str):
        """Export data to Excel file."""
        try:
            import pandas as pd
            df = pd.DataFrame(data)
            df.to_excel(output_path, index=False, sheet_name=table)
        except ImportError:
            raise ValueError("pandas is required for Excel export")
    
    def _export_all_to_json(self, all_data: Dict[str, List[Dict[str, Any]]], 
                           output_path: Path, include_metadata: bool):
        """Export all tables to JSON file."""
        export_data = {
            'exported_at': datetime.now().isoformat(),
            'tables': {}
        }
        
        for table, data in all_data.items():
            table_data = {
                'record_count': len(data),
                'data': data
            }
            
            if include_metadata:
                table_data['schema'] = self.get_table_schema(table)
            
            export_data['tables'][table] = table_data
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    def _export_all_to_excel(self, all_data: Dict[str, List[Dict[str, Any]]], output_path: Path):
        """Export all tables to Excel file."""
        try:
            import pandas as pd
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                for table, data in all_data.items():
                    if data:
                        df = pd.DataFrame(data)
                        df.to_excel(writer, sheet_name=table, index=False)
        except ImportError:
            raise ValueError("pandas and openpyxl are required for Excel export")
    
    def _format_as_csv(self, data: List[Dict[str, Any]]) -> str:
        """Format data as CSV string."""
        if not data:
            return ""
        
        import io
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)
        return output.getvalue()
    
    def _format_as_json(self, data: List[Dict[str, Any]], include_metadata: bool) -> str:
        """Format data as JSON string using Pydantic."""
        export_data = ExportData(
            table="",  # Not needed for stdout
            exported_at=datetime.now() if include_metadata else None,
            record_count=len(data),
            data=data,
            schema=None
        )
        
        return export_data.model_dump_json(indent=2)
    
    def _format_as_text(self, data: List[Dict[str, Any]], table: str) -> str:
        """Format data as text."""
        if not data:
            return f"No data in table '{table}'"
        
        lines = [f"Table: {table}", f"Records: {len(data)}", ""]
        
        # Show first few records as example
        for i, record in enumerate(data[:5]):
            lines.append(f"Record {i+1}:")
            for key, value in record.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        if len(data) > 5:
            lines.append(f"... and {len(data) - 5} more records")
        
        return "\n".join(lines)
    
    def _format_all_as_json(self, all_data: Dict[str, List[Dict[str, Any]]], 
                           include_metadata: bool) -> str:
        """Format all data as JSON string using Pydantic."""
        tables_data = {}
        for table, data in all_data.items():
            tables_data[table] = ExportData(
                table=table,
                exported_at=datetime.now() if include_metadata else None,
                record_count=len(data),
                data=data,
                schema=None
            )
        
        export_data = {
            'exported_at': datetime.now().isoformat() if include_metadata else None,
            'tables': {k: v.model_dump() for k, v in tables_data.items()}
        }
        
        return json.dumps(export_data, indent=2, ensure_ascii=False)
    
    def _format_all_as_text(self, all_data: Dict[str, List[Dict[str, Any]]]) -> str:
        """Format all data as text."""
        lines = ["Database Export Summary", "=" * 30, ""]
        
        for table, data in all_data.items():
            lines.append(f"{table}: {len(data)} records")
        
        lines.append("")
        lines.append("Use --format json for detailed data")
        
        return "\n".join(lines)
    

    
    def close(self):
        """Close data exporter and clean up resources."""
        if self._db_manager:
            self._db_manager.close_connections()
        self._db_manager = None
        logger.debug("DataExporter closed") 