"""
Unified configuration management using Pydantic Settings.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Unified application settings with automatic environment variable loading."""

    # ============================================================================
    # DISCORD BOT CONFIGURATION
    # ============================================================================
    
    # Core Discord Bot Settings
    discord_bot_token: Optional[str] = Field(
        default=None, description="Discord bot authentication token"
    )
    discord_guild_id: Optional[str] = Field(
        default=None, description="Discord guild/server ID"
    )
    discord_bot_command_prefix: str = Field(
        default="!", description="Bot command prefix"
    )
    discord_bot_message_content_intent: bool = Field(
        default=True, description="Enable message content intent for bot"
    )
    discord_bot_members_intent: bool = Field(
        default=True, description="Enable members intent for bot"
    )
    
    # Discord Bot Performance Settings
    discord_bot_interaction_timeout_seconds: float = Field(
        default=25.0, description="Discord interaction timeout in seconds"
    )
    discord_bot_cache_ttl_seconds: int = Field(
        default=300, description="Autocomplete cache duration in seconds (5 minutes)"
    )
    discord_bot_cache_max_items: int = Field(
        default=50, description="Maximum items in autocomplete cache"
    )
    discord_bot_message_character_limit: int = Field(
        default=2000, description="Discord message character limit"
    )
    discord_bot_message_chunk_size: int = Field(
        default=1900, description="Discord message chunk size for long messages"
    )

    # ============================================================================
    # DATABASE CONFIGURATION
    # ============================================================================
    
    database_url: str = Field(
        default="sqlite:///discord_messages.db", description="Database connection URL"
    )
    database_sqlite_path: str = Field(
        default="data/discord_messages.db", description="SQLite database file path"
    )

    # ============================================================================
    # LOGGING CONFIGURATION
    # ============================================================================
    
    logging_level: str = Field(default="INFO", description="Application logging level")
    logging_debug_mode: bool = Field(default=False, description="Enable debug mode")

    # ============================================================================
    # DATA SYNCHRONIZATION CONFIGURATION
    # ============================================================================
    
    # Sync Performance Settings
    sync_batch_size: int = Field(
        default=100, description="Number of messages to sync in each batch"
    )
    sync_delay_seconds: int = Field(
        default=1, description="Delay between sync batches in seconds"
    )
    sync_max_retries: int = Field(
        default=3, description="Maximum number of sync retry attempts"
    )
    sync_timeout_seconds: int = Field(
        default=300, description="Maximum time for sync operation in seconds"
    )
    
    # Sync Behavior Settings
    sync_auto_threshold_hours: int = Field(
        default=1, description="Auto-sync if data older than X hours"
    )
    sync_allow_force: bool = Field(
        default=True, description="Allow users to force sync operations"
    )
    sync_show_progress: bool = Field(
        default=True, description="Show sync progress to users"
    )

    # ============================================================================
    # ANALYSIS CONFIGURATION
    # ============================================================================
    
    # Analysis Limits and Filters
    analysis_max_messages_total: int = Field(
        default=10000, description="Maximum total messages to analyze"
    )
    analysis_max_messages_per_run: int = Field(
        default=800, description="Maximum messages per analysis run"
    )
    analysis_min_message_length: int = Field(
        default=50, description="Minimum message length for analysis"
    )
    analysis_max_results: int = Field(
        default=1000, description="Maximum results returned per analysis"
    )
    analysis_cache_ttl_seconds: int = Field(
        default=3600, description="Analysis cache time-to-live in seconds (1 hour)"
    )
    
    # Analysis Content Filtering
    analysis_base_filter_sql: str = Field(
        default="author_id != 'sesh' AND author_id != '1362434210895364327' AND author_name != 'sesh' AND LOWER(author_name) != 'pepe' AND LOWER(author_name) != 'pepino' AND channel_name NOT LIKE '%test%' AND channel_name NOT LIKE '%playground%' AND channel_name NOT LIKE '%pg%'",
        description="Base SQL filter for excluding bots and test channels from analysis"
    )
    
    # Topic Analysis Settings
    analysis_topic_model_components: int = Field(
        default=5, description="Number of components for topic modeling"
    )

    # ============================================================================
    # NLP CONFIGURATION
    # ============================================================================
    
    nlp_spacy_model: str = Field(
        default="en_core_web_sm", description="spaCy model for NLP processing"
    )
    nlp_nltk_data_packages: Union[List[str], str] = Field(
        default=["punkt", "stopwords"], description="NLTK data packages to load"
    )
    nlp_cache_size: int = Field(
        default=500, description="NLP processing cache size"
    )

    # ============================================================================
    # EMBEDDING CONFIGURATION
    # ============================================================================
    
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2", description="Sentence transformer model for embeddings"
    )
    embedding_batch_size: int = Field(
        default=32, description="Batch size for embedding generation"
    )
    embedding_cache_size: int = Field(
        default=1000, description="Embedding cache size"
    )

    # ============================================================================
    # VISUALIZATION CONFIGURATION
    # ============================================================================
    
    # Chart Settings
    visualization_chart_dpi: int = Field(
        default=300, description="Chart resolution in DPI"
    )
    visualization_chart_format: str = Field(
        default="png", description="Chart output format"
    )
    visualization_temp_directory: str = Field(
        default="temp", description="Temporary directory for chart generation"
    )
    
    # Chart Colors
    visualization_chart_colors: Dict[str, str] = Field(
        default={
            "primary": "#5865F2",
            "secondary": "#4752C4",
            "accent": "#FF6B6B",
            "success": "#4ECDC4",
            "warning": "#FFEAA7",
        },
        description="Chart color scheme for visualizations",
    )

    # ============================================================================
    # EXPORT CONFIGURATION
    # ============================================================================
    
    export_default_format: str = Field(
        default="json", description="Default export format"
    )
    export_max_rows: int = Field(
        default=10000, description="Maximum rows to export"
    )

    # ============================================================================
    # FIELD VALIDATORS
    # ============================================================================

    @field_validator("nlp_nltk_data_packages")
    @classmethod
    def validate_nltk_data(cls, v):
        """Handle nltk_data field gracefully."""
        if isinstance(v, str):
            # If it's an empty string, return default
            if not v.strip():
                return ["punkt", "stopwords"]
            # If it's a comma-separated string, split it
            if "," in v:
                return [item.strip() for item in v.split(",") if item.strip()]
            # If it's a single value, wrap in list
            return [v.strip()]
        elif isinstance(v, list):
            return v
        else:
            # Fallback to default
            return ["punkt", "stopwords"]

    @field_validator("analysis_min_message_length")
    @classmethod
    def validate_min_message_length(cls, v):
        """Ensure minimum message length is non-negative."""
        if v < 0:
            raise ValueError("analysis_min_message_length must be non-negative")
        return v

    @field_validator("analysis_max_messages_per_run")
    @classmethod
    def validate_max_messages_per_analysis(cls, v):
        """Ensure max messages per analysis is positive."""
        if v < 1:
            raise ValueError("analysis_max_messages_per_run must be positive")
        return v

    @field_validator("analysis_topic_model_components")
    @classmethod
    def validate_topic_model_components(cls, v):
        """Ensure topic model components is positive."""
        if v < 1:
            raise ValueError("analysis_topic_model_components must be positive")
        return v

    @field_validator("visualization_chart_dpi")
    @classmethod
    def validate_chart_dpi(cls, v):
        """Ensure chart DPI is at least 72."""
        if v < 72:
            raise ValueError("visualization_chart_dpi must be at least 72")
        return v

    @field_validator("logging_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"logging_level must be one of {valid_levels}")
        return v.upper()

    @field_validator("analysis_base_filter_sql")
    @classmethod
    def validate_base_filter(cls, v):
        """Ensure base filter is properly formatted for SQL usage."""
        if not v or not v.strip():
            return "1=1"  # Always true condition when no filter is needed
        # Strip whitespace and ensure it doesn't start with AND
        cleaned = v.strip()
        if cleaned.upper().startswith("AND "):
            cleaned = cleaned[4:]  # Remove leading "AND "
        return cleaned

    def validate_required(self) -> bool:
        """Validate required configuration."""
        if not self.discord_bot_token:
            raise ValueError("DISCORD_BOT_TOKEN is required")
        return True

    def to_dict(self) -> Dict:
        """Convert settings to dictionary."""
        return self.model_dump()

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )


# Global settings instance
settings = Settings() 