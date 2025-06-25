"""
Unified configuration management using Pydantic Settings.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Unified application settings with automatic environment variable loading."""

    # Discord Bot Configuration
    discord_token: Optional[str] = None
    command_prefix: str = Field(default="!", description="Bot command prefix")
    message_content_intent: bool = Field(
        default=True, description="Enable message content intent"
    )
    members_intent: bool = Field(default=True, description="Enable members intent")

    # Database Configuration
    database_url: str = Field(
        default="sqlite:///data/discord_messages.db", description="Database URL"
    )
    db_path: str = Field(
        default="discord_messages.db", description="SQLite database path"
    )

    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    debug: bool = Field(default=False, description="Debug mode")

    # Analysis Configuration
    max_messages: int = Field(default=10000, description="Maximum messages to analyze")
    min_message_length: int = Field(
        default=50, description="Minimum message length for analysis"
    )
    max_messages_per_analysis: int = Field(
        default=800, description="Max messages per analysis"
    )
    topic_model_n_components: int = Field(
        default=5, description="Topic model components"
    )

    # Visualization Configuration
    chart_dpi: int = Field(default=300, description="Chart DPI")
    chart_format: str = Field(default="png", description="Chart format")
    temp_directory: str = Field(default="temp", description="Temporary directory")

    # Chart Colors
    chart_colors: Dict[str, str] = Field(
        default={
            "primary": "#5865F2",
            "secondary": "#4752C4",
            "accent": "#FF6B6B",
            "success": "#4ECDC4",
            "warning": "#FFEAA7",
        },
        description="Chart color scheme",
    )

    # NLP Configuration
    spacy_model: str = Field(default="en_core_web_sm", description="spaCy model")
    nltk_data: Union[List[str], str] = Field(
        default=["punkt", "stopwords"], description="NLTK data"
    )

    # Optional API Keys
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    huggingface_api_key: Optional[str] = Field(
        default=None, description="HuggingFace API key"
    )

    # Base Filter for Analysis
    base_filter: str = Field(
        default="""
            author_id != 'sesh' 
            AND author_id != '1362434210895364327'
            AND author_name != 'sesh'
            AND LOWER(author_name) != 'pepe'
            AND LOWER(author_name) != 'pepino'
            AND channel_name NOT LIKE '%test%' 
            AND channel_name NOT LIKE '%playground%' 
            AND channel_name NOT LIKE '%pg%'
        """,
        description="Base filter for excluding bots and test channels",
    )

    # Sync settings
    sync_batch_size: int = 100
    sync_delay_seconds: int = 1
    sync_max_retries: int = 3

    # Analysis settings
    analysis_cache_ttl: int = 3600  # 1 hour
    analysis_max_results: int = 1000

    # Sync settings
    auto_sync_threshold_hours: int = Field(
        default=1, description="Auto-sync if data older than X hours"
    )
    sync_timeout_seconds: int = Field(
        default=300, description="Max time for sync operation"
    )
    allow_force_sync: bool = Field(
        default=True, description="Allow users to force sync"
    )
    sync_feedback_enabled: bool = Field(
        default=True, description="Show sync progress to users"
    )

    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    embedding_cache_size: int = 1000

    # NLP settings
    nlp_model: str = "en_core_web_sm"
    nlp_cache_size: int = 500

    # Export settings
    export_default_format: str = "json"
    export_max_rows: int = 10000

    @field_validator("nltk_data")
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

    @field_validator("min_message_length")
    @classmethod
    def validate_min_message_length(cls, v):
        """Ensure minimum message length is non-negative."""
        if v < 0:
            raise ValueError("min_message_length must be non-negative")
        return v

    @field_validator("max_messages_per_analysis")
    @classmethod
    def validate_max_messages_per_analysis(cls, v):
        """Ensure max messages per analysis is positive."""
        if v < 1:
            raise ValueError("max_messages_per_analysis must be positive")
        return v

    @field_validator("topic_model_n_components")
    @classmethod
    def validate_topic_model_components(cls, v):
        """Ensure topic model components is positive."""
        if v < 1:
            raise ValueError("topic_model_n_components must be positive")
        return v

    @field_validator("chart_dpi")
    @classmethod
    def validate_chart_dpi(cls, v):
        """Ensure chart DPI is at least 72."""
        if v < 72:
            raise ValueError("chart_dpi must be at least 72")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()

    def validate_required(self) -> bool:
        """Validate required configuration."""
        if not self.discord_token:
            raise ValueError("DISCORD_TOKEN is required")
        return True

    def to_dict(self) -> Dict:
        """Convert settings to dictionary."""
        return self.model_dump()

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )


# Global settings instance
settings = Settings()
