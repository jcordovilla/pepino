"""
CLI Input Validation and Error Handling

Provides comprehensive input validation, sanitization, and user-friendly error handling
for CLI commands.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

import click

from pepino.logging_config import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors with user-friendly messages."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        self.message = message
        self.field = field
        self.value = value
        super().__init__(self.message)


class InputValidator:
    """
    Comprehensive input validator for CLI commands.
    
    Provides validation for:
    - Channel names
    - User names
    - Numeric parameters
    - File paths
    - Date ranges
    - Database paths
    """
    
    @staticmethod
    def validate_channel_name(channel_name: Optional[str]) -> Optional[str]:
        """
        Validate and sanitize channel name.
        
        Args:
            channel_name: Channel name to validate
            
        Returns:
            Sanitized channel name or None if invalid
            
        Raises:
            ValidationError: If channel name is invalid
        """
        if channel_name is None:
            return None
        
        if not isinstance(channel_name, str):
            raise ValidationError("Channel name must be a string", "channel_name", channel_name)
        
        # Remove leading/trailing whitespace
        channel_name = channel_name.strip()
        
        if not channel_name:
            raise ValidationError("Channel name cannot be empty", "channel_name", channel_name)
        
        # Check for invalid characters (basic Discord channel name validation)
        if len(channel_name) > 100:
            raise ValidationError("Channel name too long (max 100 characters)", "channel_name", channel_name)
        
        # Check for common invalid patterns
        invalid_patterns = [
            r'^[^a-zA-Z0-9\-_]+$',  # Only special characters
            r'^[0-9]+$',            # Only numbers
            r'^[^a-zA-Z0-9]*$',     # No alphanumeric characters
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, channel_name):
                raise ValidationError(f"Channel name '{channel_name}' contains invalid characters", "channel_name", channel_name)
        
        return channel_name
    
    @staticmethod
    def validate_user_name(user_name: Optional[str]) -> Optional[str]:
        """
        Validate and sanitize user name.
        
        Args:
            user_name: User name to validate
            
        Returns:
            Sanitized user name or None if invalid
            
        Raises:
            ValidationError: If user name is invalid
        """
        if user_name is None:
            return None
        
        if not isinstance(user_name, str):
            raise ValidationError("User name must be a string", "user_name", user_name)
        
        # Remove leading/trailing whitespace
        user_name = user_name.strip()
        
        if not user_name:
            raise ValidationError("User name cannot be empty", "user_name", user_name)
        
        # Check length
        if len(user_name) > 32:
            raise ValidationError("User name too long (max 32 characters)", "user_name", user_name)
        
        return user_name
    
    @staticmethod
    def validate_numeric_range(value: Union[int, float], min_val: Optional[Union[int, float]] = None, 
                              max_val: Optional[Union[int, float]] = None, field_name: str = "value") -> Union[int, float]:
        """
        Validate numeric value within a range.
        
        Args:
            value: Numeric value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            field_name: Name of the field for error messages
            
        Returns:
            Validated numeric value
            
        Raises:
            ValidationError: If value is outside allowed range
        """
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{field_name} must be a number", field_name, value)
        
        if min_val is not None and value < min_val:
            raise ValidationError(f"{field_name} must be at least {min_val}", field_name, value)
        
        if max_val is not None and value > max_val:
            raise ValidationError(f"{field_name} must be at most {max_val}", field_name, value)
        
        return value
    
    @staticmethod
    def validate_file_path(file_path: Optional[str], must_exist: bool = False, 
                          create_parents: bool = False) -> Optional[Path]:
        """
        Validate file path.
        
        Args:
            file_path: File path to validate
            must_exist: Whether the file must exist
            create_parents: Whether to create parent directories
            
        Returns:
            Path object or None if invalid
            
        Raises:
            ValidationError: If path is invalid
        """
        if file_path is None:
            return None
        
        if not isinstance(file_path, str):
            raise ValidationError("File path must be a string", "file_path", file_path)
        
        file_path = file_path.strip()
        
        if not file_path:
            raise ValidationError("File path cannot be empty", "file_path", file_path)
        
        path = Path(file_path)
        
        # Check if file exists (if required)
        if must_exist and not path.exists():
            raise ValidationError(f"File does not exist: {file_path}", "file_path", file_path)
        
        # Create parent directories if requested
        if create_parents and path.parent != path:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValidationError(f"Cannot create directory for {file_path}: {e}", "file_path", file_path)
        
        return path
    
    @staticmethod
    def validate_database_path(db_path: str) -> Path:
        """
        Validate database path.
        
        Args:
            db_path: Database path to validate
            
        Returns:
            Validated database path
            
        Raises:
            ValidationError: If database path is invalid
        """
        if not isinstance(db_path, str):
            raise ValidationError("Database path must be a string", "db_path", db_path)
        
        db_path = db_path.strip()
        
        if not db_path:
            raise ValidationError("Database path cannot be empty", "db_path", db_path)
        
        path = Path(db_path)
        
        # Check if parent directory exists and is writable
        if path.parent != path and not path.parent.exists():
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ValidationError(f"Cannot create database directory: {e}", "db_path", db_path)
        
        return path
    
    @staticmethod
    def validate_days_back(days: int, max_days: int = 365) -> int:
        """
        Validate days back parameter.
        
        Args:
            days: Number of days to look back
            max_days: Maximum allowed days
            
        Returns:
            Validated days value
            
        Raises:
            ValidationError: If days is invalid
        """
        return InputValidator.validate_numeric_range(days, 1, max_days, "days_back")
    
    @staticmethod
    def validate_limit(limit: int, max_limit: int = 1000) -> int:
        """
        Validate limit parameter.
        
        Args:
            limit: Limit value to validate
            max_limit: Maximum allowed limit
            
        Returns:
            Validated limit value
            
        Raises:
            ValidationError: If limit is invalid
        """
        return InputValidator.validate_numeric_range(limit, 1, max_limit, "limit")
    
    @staticmethod
    def validate_threshold(threshold: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """
        Validate threshold parameter (0.0 to 1.0).
        
        Args:
            threshold: Threshold value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Validated threshold value
            
        Raises:
            ValidationError: If threshold is invalid
        """
        return InputValidator.validate_numeric_range(threshold, min_val, max_val, "threshold")


class ErrorHandler:
    """
    Centralized error handler for CLI commands.
    
    Provides consistent error handling and user-friendly error messages.
    """
    
    @staticmethod
    def handle_validation_error(error: ValidationError, ctx: Optional[click.Context] = None) -> None:
        """
        Handle validation errors with user-friendly messages.
        
        Args:
            error: ValidationError to handle
            ctx: Click context for additional error handling
        """
        error_msg = f"❌ Validation Error: {error.message}"
        
        if error.field:
            error_msg += f" (Field: {error.field})"
        
        if error.value is not None:
            error_msg += f" (Value: {error.value})"
        
        click.echo(error_msg, err=True)
        
        if ctx and ctx.get('verbose'):
            logger.error(f"Validation error: {error.message}", exc_info=True)
    
    @staticmethod
    def handle_database_error(error: Exception, operation: str, ctx: Optional[click.Context] = None) -> None:
        """
        Handle database-related errors.
        
        Args:
            error: Database error to handle
            operation: Name of the operation that failed
            ctx: Click context for additional error handling
        """
        error_msg = f"❌ Database Error during {operation}: {str(error)}"
        
        # Provide helpful suggestions based on error type
        if "no such table" in str(error).lower():
            error_msg += "\n💡 Tip: Try running 'pepino sync run' to populate the database"
        elif "database is locked" in str(error).lower():
            error_msg += "\n💡 Tip: Another process might be using the database. Try again in a moment."
        elif "disk full" in str(error).lower():
            error_msg += "\n💡 Tip: Check available disk space"
        
        click.echo(error_msg, err=True)
        
        if ctx and ctx.get('verbose'):
            logger.error(f"Database error during {operation}: {error}", exc_info=True)
    
    @staticmethod
    def handle_template_error(error: Exception, template_name: str, ctx: Optional[click.Context] = None) -> None:
        """
        Handle template-related errors.
        
        Args:
            error: Template error to handle
            template_name: Name of the template that failed
            ctx: Click context for additional error handling
        """
        error_msg = f"❌ Template Error for '{template_name}': {str(error)}"
        
        # Provide helpful suggestions based on error type
        if "not found" in str(error).lower():
            error_msg += f"\n💡 Tip: Check if template '{template_name}' exists in templates directory"
        elif "syntax error" in str(error).lower():
            error_msg += f"\n💡 Tip: Check template syntax in '{template_name}'"
        elif "undefined" in str(error).lower():
            error_msg += f"\n💡 Tip: Check required variables in template '{template_name}'"
        
        click.echo(error_msg, err=True)
        
        if ctx and ctx.get('verbose'):
            logger.error(f"Template error for {template_name}: {error}", exc_info=True)
    
    @staticmethod
    def handle_analysis_error(error: Exception, analysis_type: str, ctx: Optional[click.Context] = None) -> None:
        """
        Handle analysis-related errors.
        
        Args:
            error: Analysis error to handle
            analysis_type: Type of analysis that failed
            ctx: Click context for additional error handling
        """
        error_msg = f"❌ Analysis Error during {analysis_type}: {str(error)}"
        
        # Provide helpful suggestions based on error type
        if "no data" in str(error).lower() or "empty" in str(error).lower():
            error_msg += f"\n💡 Tip: No data found for {analysis_type}. Try a different channel or time period."
        elif "timeout" in str(error).lower():
            error_msg += f"\n💡 Tip: Analysis timed out. Try reducing the data scope or time period."
        elif "memory" in str(error).lower():
            error_msg += f"\n💡 Tip: Analysis requires too much memory. Try reducing the data scope."
        
        click.echo(error_msg, err=True)
        
        if ctx and ctx.get('verbose'):
            logger.error(f"Analysis error during {analysis_type}: {error}", exc_info=True)
    
    @staticmethod
    def handle_generic_error(error: Exception, operation: str, ctx: Optional[click.Context] = None) -> None:
        """
        Handle generic errors.
        
        Args:
            error: Generic error to handle
            operation: Name of the operation that failed
            ctx: Click context for additional error handling
        """
        error_msg = f"❌ Error during {operation}: {str(error)}"
        
        click.echo(error_msg, err=True)
        
        if ctx and ctx.get('verbose'):
            logger.error(f"Generic error during {operation}: {error}", exc_info=True)
    
    @staticmethod
    def show_success_message(operation: str, details: Optional[str] = None) -> None:
        """
        Show success message.
        
        Args:
            operation: Name of the successful operation
            details: Optional additional details
        """
        message = f"✅ {operation} completed successfully"
        if details:
            message += f": {details}"
        click.echo(message)
    
    @staticmethod
    def show_warning_message(message: str) -> None:
        """
        Show warning message.
        
        Args:
            message: Warning message to display
        """
        click.echo(f"⚠️  Warning: {message}", err=True)
    
    @staticmethod
    def show_info_message(message: str) -> None:
        """
        Show info message.
        
        Args:
            message: Info message to display
        """
        click.echo(f"ℹ️  {message}")


def validate_cli_inputs(**kwargs) -> Dict[str, Any]:
    """
    Validate multiple CLI inputs at once.
    
    Args:
        **kwargs: Input parameters to validate
        
    Returns:
        Dictionary of validated inputs
        
    Raises:
        ValidationError: If any input is invalid
    """
    validator = InputValidator()
    validated = {}
    
    for key, value in kwargs.items():
        if value is None:
            continue
        
        try:
            if key == 'channel_name':
                validated[key] = validator.validate_channel_name(value)
            elif key == 'user_name':
                validated[key] = validator.validate_user_name(value)
            elif key == 'days_back':
                validated[key] = validator.validate_days_back(value)
            elif key == 'limit':
                validated[key] = validator.validate_limit(value)
            elif key == 'threshold':
                validated[key] = validator.validate_threshold(value)
            elif key == 'db_path':
                validated[key] = validator.validate_database_path(value)
            elif key == 'output_file':
                validated[key] = validator.validate_file_path(value, create_parents=True)
            else:
                # For unknown parameters, just store as-is
                validated[key] = value
                
        except ValidationError as e:
            # Re-raise with field name
            raise ValidationError(e.message, key, e.value)
    
    return validated 