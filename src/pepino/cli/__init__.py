"""
CLI package for Discord analytics.

Provides command-line interface for analysis operations with service integration.
"""

from .commands import cli
from .models import (
    ChannelAnalysisParams,
    OutputFormat,
    TemporalAnalysisParams,
    TopicsAnalysisParams,
    UserAnalysisParams,
)

# Service integration for CLI operations
from ..analysis.service import AnalysisService, analysis_service

__all__ = [
    "cli",
    "AnalysisService",
    "analysis_service",
    "OutputFormat",
    "UserAnalysisParams",
    "ChannelAnalysisParams",
    "TopicsAnalysisParams",
    "TemporalAnalysisParams",
]
