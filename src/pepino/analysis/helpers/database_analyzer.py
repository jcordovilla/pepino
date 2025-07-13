"""
Database Analyzer 

Synchronous database analysis using the data facade pattern for repository management.
Provides comprehensive database statistics and health reporting with proper separation of concerns.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

from .data_facade import AnalysisDataFacade, get_analysis_data_facade
from ..models import (
    DatabaseAnalysisResponse, 
    DatabaseInfo, 
    TableStatistics, 
    DatabaseSummary,
    AnalysisErrorResponse
)
from pepino.logging_config import get_logger

logger = get_logger(__name__)


class DatabaseAnalyzer:
    """
    Analyzes database statistics and health using the data facade pattern for centralized repository management.
    
    Features:
    - Database file information and health metrics
    - Table statistics and row counts
    - Data summary and activity metrics
    - Performance indicators
    
    All database operations are abstracted through the data facade for proper
    separation of concerns and dependency injection support.
    """
    
    def __init__(self, data_facade: Optional[AnalysisDataFacade] = None):
        """Initialize database analyzer with data facade."""
        if data_facade is None:
            self.data_facade = get_analysis_data_facade()
            self._owns_facade = True
        else:
            self.data_facade = data_facade
            self._owns_facade = False
            
        logger.info("DatabaseAnalyzer initialized with data facade pattern")
    
    def analyze(self) -> DatabaseAnalysisResponse:
        """
        Analyze database statistics and health.
        
        Returns:
            DatabaseAnalysisResponse with analysis results
        """
        try:
            logger.info("Starting database analysis")
            
            # Get database info
            database_info = self._get_database_info()
            
            # Get table statistics
            table_stats = self._get_table_statistics()
            
            # Get summary
            summary = self._get_summary()
            
            return DatabaseAnalysisResponse(
                database_info=database_info,
                table_stats=table_stats,
                summary=summary
            )
            
        except Exception as e:
            logger.error(f"Database analysis failed: {e}")
            return AnalysisErrorResponse(
                success=False,
                error=f"Database analysis failed: {str(e)}",
                plugin="DatabaseAnalyzer"
            )
    
    def _get_database_info(self) -> DatabaseInfo:
        """Get database file information."""
        try:
            db_path = self.data_facade.db_manager.db_path
            
            if not db_path or not os.path.exists(db_path):
                return DatabaseInfo(
                    file_path=str(db_path) if db_path else "unknown",
                    size_mb=0.0,
                    last_modified=None
                )
            
            # Get file stats
            stat = os.stat(db_path)
            size_mb = stat.st_size / (1024 * 1024)
            last_modified = datetime.fromtimestamp(stat.st_mtime).isoformat()
            
            return DatabaseInfo(
                file_path=str(db_path),
                size_mb=round(size_mb, 2),
                last_modified=last_modified
            )
            
        except Exception as e:
            logger.error(f"Failed to get database info: {e}")
            return DatabaseInfo(
                file_path="unknown",
                size_mb=0.0,
                last_modified=None
            )
    
    def _get_table_statistics(self) -> List[TableStatistics]:
        """Get statistics for all tables using repository methods."""
        try:
            # Get actual tables from database using repository
            actual_tables = self.data_facade.database_repository.get_table_names()
            # Filter out sqlite_sequence table
            actual_tables = [table for table in actual_tables if table != 'sqlite_sequence']
            
            table_stats = []
            
            for table in actual_tables:
                try:
                    # Get row count using repository methods where available
                    row_count = self._get_table_row_count(table)
                    
                    # Debug logging for messages table
                    if table == 'messages':
                        logger.info(f"Messages table row count: {row_count}")
                    
                    # Get last insert (for messages table)
                    last_insert = None
                    if table == 'messages':
                        last_insert_val = self._get_messages_last_insert()
                        if isinstance(last_insert_val, str):
                            last_insert = last_insert_val
                        elif last_insert_val is not None:
                            try:
                                last_insert = last_insert_val.isoformat()
                            except Exception:
                                last_insert = str(last_insert_val)
                    
                    # Estimate table size (rough calculation)
                    size_mb = row_count * 0.001  # Rough estimate
                    
                    table_stats.append(TableStatistics(
                        table_name=table,
                        row_count=row_count,
                        size_mb=round(size_mb, 2),
                        last_insert=last_insert
                    ))
                    
                except Exception as e:
                    logger.error(f"Failed to get stats for table {table}: {e}")
                    table_stats.append(TableStatistics(
                        table_name=table,
                        row_count=0,
                        size_mb=0.0,
                        last_insert=None
                    ))
            
            return table_stats
            
        except Exception as e:
            logger.error(f"Failed to get table statistics: {e}")
            return []
    
    def _get_table_row_count(self, table_name: str) -> int:
        """Get row count for a table using repository methods."""
        try:
            if table_name == 'messages':
                # For table statistics, use raw count without base filter
                return self.data_facade.database_repository.get_table_row_count(table_name)
            elif table_name == 'users':
                # Use user repository to get distinct user count
                return self.data_facade.message_repository.get_distinct_user_count()
            elif table_name == 'channels':
                # Use channel repository to get distinct channel count
                return self.data_facade.channel_repository.get_distinct_channel_count()
            else:
                # Use database repository for other tables
                return self.data_facade.database_repository.get_table_row_count(table_name)
        except Exception as e:
            logger.debug(f"Failed to get row count for table {table_name}: {e}")
            return 0
    
    def _get_messages_last_insert(self) -> Optional[str]:
        """Get last message timestamp using repository methods."""
        try:
            # Get recent messages and find the latest timestamp
            recent_messages = self.data_facade.message_repository.get_recent_messages(limit=1)
            if recent_messages:
                return recent_messages[0].timestamp
            return None
        except Exception as e:
            logger.debug(f"Failed to get last message timestamp: {e}")
            return None
    
    def _get_summary(self) -> DatabaseSummary:
        """Get database summary using repository methods."""
        try:
            # Get total messages using message repository
            total_messages = self.data_facade.message_repository.get_total_message_count()
            
            # Get total unique users using message repository
            total_users = self.data_facade.message_repository.get_distinct_user_count()
            
            # Get total unique channels using channel repository
            total_channels = self.data_facade.channel_repository.get_distinct_channel_count()
            
            # Get date range using message repository
            date_range = self._get_date_range()
            
            # Get most active channel using channel repository
            most_active_channel = self._get_most_active_channel()
            
            # Get most active user using user repository
            most_active_user = self._get_most_active_user()
            
            # Calculate averages
            avg_messages_per_day = None
            avg_messages_per_user = None
            avg_messages_per_channel = None
            
            if total_messages > 0:
                if total_users > 0:
                    avg_messages_per_user = round(total_messages / total_users, 1)
                if total_channels > 0:
                    avg_messages_per_channel = round(total_messages / total_channels, 1)
                
                # Calculate average messages per day if we have date range
                if date_range and date_range.get('start') and date_range.get('end'):
                    try:
                        start_date = datetime.fromisoformat(date_range['start'].replace('Z', '+00:00'))
                        end_date = datetime.fromisoformat(date_range['end'].replace('Z', '+00:00'))
                        days_diff = (end_date - start_date).days + 1
                        if days_diff > 0:
                            avg_messages_per_day = round(total_messages / days_diff, 1)
                    except:
                        pass
            
            return DatabaseSummary(
                total_messages=total_messages,
                total_users=total_users,
                total_channels=total_channels,
                date_range=date_range,
                most_active_channel=most_active_channel,
                most_active_user=most_active_user,
                avg_messages_per_day=avg_messages_per_day,
                avg_messages_per_user=avg_messages_per_user,
                avg_messages_per_channel=avg_messages_per_channel
            )
            
        except Exception as e:
            logger.error(f"Failed to get summary: {e}")
            return DatabaseSummary(
                total_messages=0,
                total_users=0,
                total_channels=0
            )
    
    def _get_date_range(self) -> Optional[Dict[str, str]]:
        """Get message date range using repository methods."""
        try:
            # Get message statistics which includes first and last message timestamps
            stats = self.data_facade.message_repository.get_message_statistics()
            if stats and stats.get('first_message') and stats.get('last_message'):
                return {
                    'start': stats['first_message'],
                    'end': stats['last_message']
                }
            return None
        except Exception as e:
            logger.debug(f"Failed to get date range: {e}")
            return None
    
    def _get_most_active_channel(self) -> Optional[Dict[str, Any]]:
        """Get most active channel using channel repository."""
        try:
            # Get top channels by message count
            top_channels = self.data_facade.channel_repository.get_top_channels_by_message_count(limit=1)
            if top_channels:
                channel = top_channels[0]
                return {
                    'name': channel['channel_name'],
                    'message_count': channel['message_count']
                }
            return None
        except Exception as e:
            logger.debug(f"Failed to get most active channel: {e}")
            return None
    
    def _get_most_active_user(self) -> Optional[Dict[str, Any]]:
        """Get most active user using user repository."""
        try:
            # Get top users by message count
            top_users = self.data_facade.user_repository.get_top_users_by_message_count(limit=1)
            if top_users:
                user = top_users[0]
                return {
                    'name': user['author_name'],
                    'message_count': user['message_count']
                }
            return None
        except Exception as e:
            logger.debug(f"Failed to get most active user: {e}")
            return None
    
    def __del__(self):
        """Cleanup if we own the facade."""
        if hasattr(self, '_owns_facade') and self._owns_facade and hasattr(self, 'data_facade'):
            try:
                self.data_facade.close()
            except:
                pass 