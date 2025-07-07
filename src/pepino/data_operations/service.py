"""
Data Operations Service

Provides a clean, encapsulated interface for data synchronization and export operations.
Separate from analysis operations to maintain clean separation of concerns.
"""

import logging
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from pepino.config import Settings
from pepino.data_operations.exporters import DataExporter

logger = logging.getLogger(__name__)


class DataOperationsService:
    """
    Public interface for data operations (sync and export).
    
    Provides dedicated methods for data synchronization and export operations,
    hiding internal complexity while maintaining clean separation from analysis.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize data operations service.
        
        Args:
            db_path: Optional database path (uses settings default if None)
        """
        self.settings = Settings()
        self.db_path = db_path or self.settings.database_sqlite_path
        
        # Lazy initialization - only create when needed
        self._sync_manager = None
        self._data_exporter = None
        
        logger.debug(f"DataOperationsService initialized for {self.db_path}")
    
    @property
    def sync_manager(self):
        """Get sync manager instance (lazy initialization)."""
        if self._sync_manager is None:
            # Lazy import - only when actually needed for sync operations
            from pepino.data_operations.discord_sync.sync_manager import SyncManager
            self._sync_manager = SyncManager(self.db_path)
        return self._sync_manager
    
    @property
    def data_exporter(self) -> DataExporter:
        """Get data exporter instance (lazy initialization)."""
        if self._data_exporter is None:
            self._data_exporter = DataExporter(self.db_path)
        return self._data_exporter
    
    # Sync Operations
    async def sync_data(self, force: bool = False, full: bool = False, 
                       clear_existing: bool = False) -> Dict[str, Any]:
        """
        Synchronize Discord data.
        
        Args:
            force: Force sync even if data is fresh
            full: Complete re-sync (re-downloads everything)
            clear_existing: Clear existing data before sync
            
        Returns:
            Sync result information
        """
        if full:
            result = await self.sync_manager.run_full_sync(clear_existing=clear_existing)
        else:
            result = await self.sync_manager.run_incremental_sync(force=force)
        
        # Convert Pydantic model to dict for service layer
        return {
            'sync_performed': result.sync_performed,
            'new_messages': getattr(result, 'new_messages', 0),
            'updated_channels': getattr(result, 'updated_channels', 0),
            'duration': result.duration,
            'error': getattr(result, 'error', None),
            'last_sync': getattr(result, 'last_sync', None),
            'reason': getattr(result, 'reason', None)
        }
    
    async def get_sync_status(self) -> Dict[str, Any]:
        """
        Get current sync status and statistics.
        
        Returns:
            Sync status information
        """
        last_sync = self.sync_manager.get_last_sync_time()
        is_stale = self.sync_manager.is_data_stale()
        
        return {
            'last_sync': last_sync.isoformat() if last_sync else None,
            'is_stale': is_stale,
            'status': 'Fresh' if not is_stale else 'Stale'
        }
    
    def clear_database(self) -> None:
        """Clear the database for a fresh start using SQLite CASCADE DELETE."""
        from pepino.data.database.manager import DatabaseManager
        
        logger.info("🗑️ Clearing database for fresh start...")
        try:
            db_manager = DatabaseManager(self.db_path)
            
            # Enable foreign key constraints
            db_manager.execute_query("PRAGMA foreign_keys = ON", fetch_one=False, fetch_all=False)
            
            # Clear all data using CASCADE DELETE approach
            # This is more efficient than individual repository methods
            # Clear child tables first, then parent tables to avoid foreign key conflicts
            tables_to_clear = [
                "message_temporal_stats",  # Child table - clear first
                "channel_members",         # Independent table
                "sync_logs",              # Independent table
                "messages"                # Parent table - clear last
            ]
            
            for table in tables_to_clear:
                try:
                    db_manager.execute_query(f"DELETE FROM {table}", fetch_one=False, fetch_all=False)
                    logger.info(f"✅ Cleared table: {table}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not clear table {table}: {e}")
            
            # Reset any auto-increment sequences if they exist
            try:
                db_manager.execute_query("DELETE FROM sqlite_sequence", fetch_one=False, fetch_all=False)
            except Exception:
                pass  # sqlite_sequence table might not exist
            
            db_manager.close_connections()
            logger.info("✅ Database cleared successfully using CASCADE DELETE")
            
        except Exception as e:
            logger.error(f"❌ Error clearing database: {e}")
            raise
    
    # Export Operations
    def export_data(self, table: Optional[str] = None, output_path: Optional[str] = None,
                   format: str = "csv", include_metadata: bool = True) -> str:
        """
        Export data from the database.
        
        Args:
            table: Specific table to export (messages, users, channels, None for all)
            output_path: Output file path (None for stdout)
            format: Output format (csv, json, excel)
            include_metadata: Include export metadata
            
        Returns:
            Export result information
        """
        return self.data_exporter.export_data(
            table=table,
            output_path=output_path,
            format=format,
            include_metadata=include_metadata
        )
    
    def export_table(self, table: str, output_path: Optional[str] = None,
                    format: str = "csv", filters: Optional[Dict[str, Any]] = None) -> str:
        """
        Export specific table with optional filters.
        
        Args:
            table: Table name to export
            output_path: Output file path (None for stdout)
            format: Output format (csv, json, excel)
            filters: Optional filters to apply
            
        Returns:
            Export result information
        """
        return self.data_exporter.export_table(
            table=table,
            output_path=output_path,
            format=format,
            filters=filters
        )
    
    def get_available_tables(self) -> List[str]:
        """
        Get list of available tables for export.
        
        Returns:
            List of available table names
        """
        return self.data_exporter.get_available_tables()
    
    def get_table_schema(self, table: str) -> Dict[str, Any]:
        """
        Get schema information for a specific table.
        
        Args:
            table: Table name
            
        Returns:
            Table schema information
        """
        return self.data_exporter.get_table_schema(table)
    
    def get_available_channels(self) -> List[str]:
        """
        Get list of available channels for analysis.
        
        Returns:
            List of available channel names
        """
        from pepino.data.database.manager import DatabaseManager
        from pepino.data.repositories import ChannelRepository
        
        # Create a temporary database manager and repository
        db_manager = DatabaseManager(self.db_path)
        try:
            channel_repo = ChannelRepository(db_manager)
            return channel_repo.get_available_channels()
        finally:
            db_manager.close_connections()
    
    def close(self):
        """Close the data operations service and clean up resources."""
        if self._sync_manager:
            self._sync_manager.close()
        if self._data_exporter:
            self._data_exporter.close()
        logger.debug("DataOperationsService closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience function for common operations
@contextmanager
def data_operations_service(db_path: Optional[str] = None):
    """
    Context manager for data operations.
    
    Args:
        db_path: Optional database path (uses settings default if None)
        
    Yields:
        DataOperationsService instance
    """
    settings = Settings()
    actual_db_path = db_path or settings.database_sqlite_path
    service = DataOperationsService(db_path=actual_db_path)
    try:
        yield service
    finally:
        service.close() 