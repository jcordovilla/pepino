"""
Professional logging configuration for Pepino Discord Analytics.

Provides centralized, structured logging with proper formatters, handlers,
and configuration management.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional

from pepino.data.config import Settings


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""

    # Color codes
    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        # Add color to level name
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            )
        return super().format(record)


def setup_logging(
    settings: Optional[Settings] = None,
    console_output: bool = True,
    file_output: bool = True,
    structured: bool = False,
) -> None:
    """
    Set up professional logging configuration.
    
    Args:
        settings: Application settings (optional)
        console_output: Whether to enable console logging
        file_output: Whether to enable file logging
        structured: Whether to use structured (JSON) logging format
    """
    if settings is None:
        settings = Settings()

    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Define formatters
    if structured:
        # Structured logging format for production
        console_format = (
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s", '
            '"module": "%(module)s", "function": "%(funcName)s", "line": %(lineno)d}'
        )
        file_format = console_format
    else:
        # Human-readable format for development
        console_format = "%(asctime)s [%(levelname)-8s] %(name)-20s: %(message)s"
        file_format = (
            "%(asctime)s [%(levelname)-8s] %(name)-20s [%(module)s.%(funcName)s:%(lineno)d]: %(message)s"
        )

    # Configure logging
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console": {
                "()": ColoredFormatter,
                "format": console_format,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "file": {
                "format": file_format,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "detailed": {
                "format": (
                    "%(asctime)s [%(levelname)-8s] %(name)-20s "
                    "[%(process)d:%(thread)d] [%(module)s.%(funcName)s:%(lineno)d]: "
                    "%(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {},
        "loggers": {
            "pepino": {
                "level": settings.log_level,
                "handlers": [],
                "propagate": False,
            },
            "discord": {
                "level": "INFO",
                "handlers": [],
                "propagate": False,
            },
            "root": {
                "level": "WARNING",
                "handlers": [],
            },
        },
    }

    # Add console handler if enabled
    if console_output:
        config["handlers"]["console"] = {
            "class": "logging.StreamHandler",
            "level": settings.log_level,
            "formatter": "console",
            "stream": "ext://sys.stdout",
        }
        config["loggers"]["pepino"]["handlers"].append("console")
        config["loggers"]["discord"]["handlers"].append("console")
        config["loggers"]["root"]["handlers"].append("console")

    # Add file handlers if enabled
    if file_output:
        # Main application log
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(logs_dir / "pepino.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8",
        }

        # Error log
        config["handlers"]["error_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": str(logs_dir / "errors.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8",
        }

        # Discord-specific log
        config["handlers"]["discord_file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "detailed",
            "filename": str(logs_dir / "discord.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 3,
            "encoding": "utf8",
        }

        # Add file handlers to loggers
        config["loggers"]["pepino"]["handlers"].extend(["file", "error_file"])
        config["loggers"]["discord"]["handlers"].extend(["discord_file", "error_file"])

    # Apply configuration
    logging.config.dictConfig(config)

    # Log startup message
    logger = logging.getLogger("pepino.logging")
    logger.info(f"Logging initialized - Level: {settings.log_level}")
    if file_output:
        logger.info(f"Log files: {logs_dir.absolute()}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a properly configured logger for a module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    # Ensure the logger is under the pepino hierarchy
    if not name.startswith("pepino.") and name != "pepino":
        name = f"pepino.{name}"
    
    return logging.getLogger(name)


def configure_third_party_loggers(settings: Optional[Settings] = None) -> None:
    """Configure third-party library loggers to reduce noise."""
    if settings is None:
        settings = Settings()

    # Reduce noise from common libraries
    noisy_loggers = {
        "urllib3": "WARNING",
        "requests": "WARNING",
        "aiohttp": "WARNING",
        "discord.client": "INFO",
        "discord.gateway": "WARNING",
        "discord.http": "WARNING",
        "matplotlib": "WARNING",
        "PIL": "WARNING",
        "asyncio": "WARNING",
    }

    for logger_name, level in noisy_loggers.items():
        logging.getLogger(logger_name).setLevel(getattr(logging, level))


# Context managers for temporary log level changes
class temp_log_level:
    """Context manager for temporarily changing log level."""
    
    def __init__(self, logger_name: str, level: str):
        self.logger = logging.getLogger(logger_name)
        self.level = getattr(logging, level.upper())
        self.original_level = None
    
    def __enter__(self):
        self.original_level = self.logger.level
        self.logger.setLevel(self.level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)


def setup_cli_logging(verbose: bool = False) -> None:
    """Set up logging specifically for CLI operations."""
    settings = Settings()
    
    # Adjust log level based on verbosity
    if verbose:
        settings.log_level = "DEBUG"
    else:
        settings.log_level = "INFO"
    
    # Set up logging with console output only for CLI
    setup_logging(
        settings=settings,
        console_output=True,
        file_output=True,  # Still log to files for debugging
        structured=False,
    )
    
    # Configure third-party loggers
    configure_third_party_loggers(settings)


def setup_bot_logging() -> None:
    """Set up logging specifically for Discord bot operations."""
    settings = Settings()
    
    # Set up full logging for bot operations
    setup_logging(
        settings=settings,
        console_output=True,
        file_output=True,
        structured=False,
    )
    
    # Configure third-party loggers
    configure_third_party_loggers(settings) 