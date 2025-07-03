"""
CLI commands for Discord analytics.
"""

import asyncio
import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

import click

from pepino.logging_config import get_logger, setup_cli_logging

from pepino.analysis.channel_analyzer import ChannelAnalyzer
from pepino.analysis.conversation_analyzer import ConversationService
from pepino.analysis.embedding_analyzer import EmbeddingService
from pepino.analysis.nlp_analyzer import NLPService
from pepino.analysis.similarity_analyzer import SimilarityService
from pepino.analysis.temporal_analyzer import TemporalAnalyzer
from pepino.analysis.topic_analyzer import TopicAnalyzer
from pepino.analysis.user_analyzer import UserAnalyzer
from pepino.config import Settings
from pepino.data.database.manager import DatabaseManager
from pepino.data.repositories import ChannelRepository, MessageRepository

from . import persistence
from .mixins import CLIAnalysisMixin

logger = get_logger(__name__)


class CLIAnalysisCommands(CLIAnalysisMixin):
    """
    CLI Analysis Commands with template integration.
    
    Uses mixin pattern for consistent template rendering and output formatting.
    """
    
    def __init__(self):
        super().__init__()
        logger.info("CLI Analysis Commands initialized with template support")
    
    def analyze_users(
        self,
        ctx_obj: Dict[str, Any],
        user: Optional[str],
        limit: int,
        output: Optional[str],
        output_format: str,
    ):
        """Analyze user activity with template integration."""
        try:
            from .persistence import analyze_user

            data = analyze_user(user, limit, ctx_obj.get("db_path"))

            # Handle response based on analysis type
            if user and "user_analysis" in data:
                result = data["user_analysis"]
                self.handle_analysis_result(
                    result, 
                    "User analysis", 
                    "user_analysis.txt.j2",
                    output, 
                    output_format
                )
            else:
                # Handle list of top users with template
                result = data.get("top_users", [])
                enhanced_data = {
                    "top_users": result,
                    "limit": limit,
                    "total_users": len(result)
                }
                self.handle_analysis_result(
                    enhanced_data,
                    "Top users analysis", 
                    "top_users.txt.j2",
                    output, 
                    output_format
                )

        except RuntimeError as e:
            self.show_template_error("User analysis", str(e))
        except Exception as e:
            self.show_template_error("User analysis", f"Unexpected error: {e}")
            if ctx_obj.get("verbose"):
                raise
    
    def analyze_channels(
        self,
        ctx_obj: Dict[str, Any],
        channel: Optional[str],
        limit: int,
        output: Optional[str],
        output_format: str,
    ):
        """Analyze channel activity with template integration."""
        try:
            from .persistence import analyze_channel

            data = analyze_channel(channel, limit, ctx_obj.get("db_path"))

            # Handle response based on analysis type
            if channel and "channel_analysis" in data:
                result = data["channel_analysis"]
                self.handle_analysis_result(
                    result, 
                    "Channel analysis", 
                    "channel_analysis.txt.j2",
                    output, 
                    output_format
                )
            else:
                # Handle list of top channels with template
                result = data.get("top_channels", [])
                enhanced_data = {
                    "top_channels": result,
                    "limit": limit,
                    "total_channels": len(result)
                }
                self.handle_analysis_result(
                    enhanced_data,
                    "Top channels analysis", 
                    "top_channels.txt.j2",
                    output, 
                    output_format
                )

        except RuntimeError as e:
            self.show_template_error("Channel analysis", str(e))
        except Exception as e:
            self.show_template_error("Channel analysis", f"Unexpected error: {e}")
            if ctx_obj.get("verbose"):
                raise
    
    def analyze_topics(
        self,
        ctx_obj: Dict[str, Any],
        channel: Optional[str],
        n_topics: int,
        days_back: int,
        output: Optional[str],
        output_format: str,
    ):
        """Analyze topics with template integration and NLP capabilities."""
        try:
            from .persistence import get_database_manager
            from ..analysis.data_facade import get_analysis_data_facade
            from ..analysis.topic_analyzer import TopicAnalyzer
            from ..config import Settings

            # Enhanced topic analysis with NLP capabilities
            settings = Settings()
            db_path = ctx_obj.get("db_path") or settings.db_path

            with get_database_manager(db_path) as db_manager:
                # Create data facade with proper base filter
                data_facade = get_analysis_data_facade(db_manager, settings.base_filter)
                
                # Create topic analyzer
                topic_analyzer = TopicAnalyzer(data_facade)
                
                # Perform analysis
                analysis_result = topic_analyzer.analyze(
                    channel_name=channel, 
                    top_n=n_topics, 
                    days_back=days_back
                )
                
                if not analysis_result:
                    self.show_template_error("Topic analysis", "No analysis result returned")
                    return
                
                # Check if we got an error response
                if hasattr(analysis_result, 'error') and analysis_result.error:
                    self.show_template_error("Topic analysis", analysis_result.error)
                    return
                
                # Check if we have topics
                if not hasattr(analysis_result, 'topics') or not analysis_result.topics:
                    self.show_template_error("Topic analysis", "No topics found for the specified criteria")
                    return
                
                # Get recent messages for NLP analysis context (increased limit for better concept extraction)
                recent_messages = []
                try:
                    if channel:
                        messages_data = data_facade.message_repository.get_messages_by_channel(channel, limit=500)
                    else:
                        messages_data = data_facade.message_repository.get_recent_messages(limit=500, days_back=days_back)
                    
                    if messages_data:
                        recent_messages = [
                            {
                                'id': msg.get('id'),
                                'content': msg.get('content', ''),
                                'author': msg.get('username') or msg.get('author', ''),
                                'timestamp': msg.get('timestamp', ''),
                                'channel': msg.get('channel_name', channel or 'all')
                            }
                            for msg in messages_data
                        ]
                except Exception as e:
                    if ctx_obj.get("verbose"):
                        self.show_template_error("Message context loading", f"Could not fetch messages for NLP analysis: {e}")
                
                # Prepare enhanced template data with NLP capabilities
                template_data = {
                    'channel_name': channel,
                    'days_back': days_back,
                    'n_topics': n_topics,
                    'analysis': analysis_result,
                    'topics': analysis_result.topics,
                    'message_count': analysis_result.message_count,
                    'capabilities_used': analysis_result.capabilities_used if hasattr(analysis_result, 'capabilities_used') else ['topic_analysis']
                }
                
                # Add domain analysis data if available from hybrid approach
                if hasattr(analysis_result, '_domain_analysis'):
                    template_data['_domain_analysis'] = analysis_result._domain_analysis
                
                # Convert Pydantic model to dict for compatibility
                result_dict = analysis_result.model_dump() if hasattr(analysis_result, "model_dump") else analysis_result
                final_data = {**template_data, **result_dict}
                
                # Handle template rendering with NLP context
                self.handle_analysis_result(
                    final_data,
                    "Topic analysis", 
                    "topic_analysis.txt.j2",
                    output, 
                    output_format,
                    data_facade=data_facade,
                    messages=recent_messages
                )

        except RuntimeError as e:
            self.show_template_error("Topic analysis", str(e))
        except Exception as e:
            self.show_template_error("Topic analysis", f"Unexpected error: {e}")
            if ctx_obj.get("verbose"):
                raise
    
    def analyze_temporal(
        self,
        ctx_obj: Dict[str, Any],
        channel: Optional[str],
        days_back: int,
        granularity: str,
        output: Optional[str],
        output_format: str,
    ):
        """Analyze temporal patterns with template integration."""
        try:
            from .persistence import analyze_temporal

            data = analyze_temporal(
                channel, days_back, granularity, ctx_obj.get("db_path")
            )

            # Handle Pydantic response models
            if "temporal_analysis" in data:
                result = data["temporal_analysis"]
                self.handle_analysis_result(
                    result, 
                    "Temporal analysis", 
                    "temporal_analysis.txt.j2",
                    output, 
                    output_format
                )
            else:
                self.handle_output(data, output, output_format)

        except RuntimeError as e:
            self.show_template_error("Temporal analysis", str(e))
        except Exception as e:
            self.show_template_error("Temporal analysis", f"Unexpected error: {e}")
            if ctx_obj.get("verbose"):
                raise


# Global CLI analysis commands instance
_cli_analysis = CLIAnalysisCommands()


def validate_db_path(ctx, param, value):
    """Click callback to validate database path."""
    if not value:
        return value

    try:
        # Convert to Path object for better handling
        path = Path(value)

        # Resolve to absolute path
        abs_path = path.resolve()

        # Check if parent directory exists or can be created
        parent_dir = abs_path.parent
        if not parent_dir.exists():
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
                click.echo(f"Created database directory: {parent_dir}", err=True)
            except PermissionError:
                raise click.BadParameter(
                    f"Cannot create database directory: {parent_dir} (permission denied)"
                )
            except OSError as e:
                raise click.BadParameter(
                    f"Cannot create database directory: {parent_dir} ({e})"
                )

        # Check write permissions on parent directory
        if not parent_dir.is_dir():
            raise click.BadParameter(
                f"Database path parent is not a directory: {parent_dir}"
            )

        # Check write permissions using os.access
        import os

        if not os.access(parent_dir, os.W_OK):
            raise click.BadParameter(
                f"No write permission for database directory: {parent_dir}"
            )

        # If database file exists, check if it's readable
        if abs_path.exists():
            if not abs_path.is_file():
                raise click.BadParameter(
                    f"Database path exists but is not a file: {abs_path}"
                )
            if not os.access(abs_path, os.R_OK):
                raise click.BadParameter(
                    f"Cannot read existing database file: {abs_path}"
                )

        return str(abs_path)

    except click.BadParameter:
        raise
    except Exception as e:
        raise click.BadParameter(f"Invalid database path '{value}': {e}")


@click.group()
@click.option(
    "--db-path",
    default="data/discord_messages.db",
    callback=validate_db_path,
    help="Database path",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def cli(ctx, db_path: str, verbose: bool):
    """Discord Analytics CLI - Analyze Discord server data."""
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db_path
    ctx.obj["verbose"] = verbose

    # Set up professional logging
    setup_cli_logging(verbose=verbose)


@cli.group(name="analyze")
@click.pass_context
def analyze(ctx):
    """Analyze Discord data and generate insights."""
    pass


@analyze.command(name="users")
@click.option("--user", "-u", help="Specific user to analyze")
@click.option(
    "--limit", "-l", default=10, help="Number of top users to show (default: 10)"
)
@click.option("--output", "-o", help="Output file (JSON or CSV)")
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json", "csv"]),
    help="Output format",
)
@click.pass_context
def analyze_users(
    ctx, user: Optional[str], limit: int, output: Optional[str], output_format: str
):
    """Analyze user activity and statistics."""
    _cli_analysis.analyze_users(ctx.obj, user, limit, output, output_format)


@analyze.command(name="channels")
@click.option("--channel", "-c", help="Specific channel to analyze")
@click.option(
    "--limit", "-l", default=10, help="Number of top channels to show (default: 10)"
)
@click.option("--output", "-o", help="Output file (JSON or CSV)")
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json", "csv"]),
    help="Output format",
)
@click.pass_context
def analyze_channels(
    ctx, channel: Optional[str], limit: int, output: Optional[str], output_format: str
):
    """Analyze channel activity and statistics."""
    _cli_analysis.analyze_channels(ctx.obj, channel, limit, output, output_format)


@analyze.command(name="topics")
@click.option("--channel", "-c", help="Specific channel to analyze (optional)")
@click.option(
    "--topics",
    "-t",
    "n_topics",
    default=20,
    help="Number of top topics to show (default: 20)",
)
@click.option(
    "--days", "-d", "days_back", default=30, help="Days to look back (default: 30)"
)
@click.option("--output", "-o", help="Output file (JSON or CSV)")
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json", "csv"]),
    help="Output format",
)
@click.pass_context
def analyze_topics(
    ctx,
    channel: Optional[str],
    n_topics: int,
    days_back: int,
    output: Optional[str],
    output_format: str,
):
    """Analyze trending topics and keywords."""
    _cli_analysis.analyze_topics(ctx.obj, channel, n_topics, days_back, output, output_format)


@analyze.command(name="temporal")
@click.option("--channel", "-c", help="Specific channel to analyze (optional)")
@click.option(
    "--days", "-d", "days_back", default=30, help="Days to look back (default: 30)"
)
@click.option(
    "--granularity",
    "-g",
    default="day",
    type=click.Choice(["hour", "day", "week"]),
    help="Time granularity",
)
@click.option("--output", "-o", help="Output file (JSON or CSV)")
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json", "csv"]),
    help="Output format",
)
@click.pass_context
def analyze_temporal(
    ctx,
    channel: Optional[str],
    days_back: int,
    granularity: str,
    output: Optional[str],
    output_format: str,
):
    """Analyze temporal activity patterns."""
    _cli_analysis.analyze_temporal(ctx.obj, channel, days_back, granularity, output, output_format)


@analyze.command(name="conversations")
@click.option("--channel", help="Channel to analyze conversations for")
@click.option("--output", type=click.Path(), help="Output file path")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "csv", "text"]),
    default="text",
    help="Output format",
)
@click.pass_context
def analyze_conversations(
    ctx, channel: Optional[str], output: Optional[str], output_format: str
):
    """Analyze conversation patterns and engagement."""
    _analyze_conversations(ctx.obj, channel, output, output_format)


@analyze.command(name="similar")
@click.option("--query", required=True, help="Text to find similar messages for")
@click.option("--limit", default=10, help="Number of similar messages to find")
@click.option("--threshold", default=0.5, help="Similarity threshold")
@click.option("--output", type=click.Path(), help="Output file path")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "csv", "text"]),
    default="text",
    help="Output format",
)
@click.pass_context
def find_similar(
    ctx,
    query: str,
    limit: int,
    threshold: float,
    output: Optional[str],
    output_format: str,
):
    """Find messages similar to the given query."""
    _find_similar(ctx.obj, query, limit, threshold, output, output_format)


@analyze.command(name="embeddings")
@click.option("--batch-size", default=100, help="Batch size for processing")
@click.option("--output", type=click.Path(), help="Output file path")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "csv", "text"]),
    default="text",
    help="Output format",
)
@click.pass_context
def generate_embeddings(
    ctx, batch_size: int, output: Optional[str], output_format: str
):
    """Generate embeddings for all messages."""
    _generate_embeddings(ctx.obj, batch_size, output, output_format)


@analyze.command(name="sentiment")
@click.option("--channel", help="Specific channel to analyze (optional)")
@click.option("--limit", default=100, help="Number of recent messages to analyze")
@click.option("--output", type=click.Path(), help="Output file path")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "csv", "text"]),
    default="text",
    help="Output format",
)
@click.pass_context
def analyze_sentiment(
    ctx, channel: Optional[str], limit: int, output: Optional[str], output_format: str
):
    """Analyze sentiment and extract entities from messages using NLP."""
    try:
        from .persistence import analyze_sentiment_async

        data = analyze_sentiment_async(
            channel_name=channel, 
            days_back=30, 
            db_path=ctx.obj.get("db_path")
        )

        _write_output(data, output, output_format)
        
        if data.get("error"):
            click.echo(f"❌ {data['error']}", err=True)
        else:
            click.echo(
                f"✅ Sentiment analysis completed! Analyzed {data['total_messages']} messages."
            )

    except Exception as e:
        click.echo(f"❌ Unexpected error analyzing sentiment: {e}", err=True)
        if ctx.obj.get("verbose"):
            raise


@analyze.command(name="duplicates")
@click.option("--channel", help="Specific channel to analyze (optional)")
@click.option(
    "--threshold",
    default=0.9,
    help="Similarity threshold for duplicate detection (0.0-1.0)",
)
@click.option("--output", type=click.Path(), help="Output file path")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "csv", "text"]),
    default="text",
    help="Output format",
)
@click.pass_context
def detect_duplicates(
    ctx,
    channel: Optional[str],
    threshold: float,
    output: Optional[str],
    output_format: str,
):
    """Detect duplicate messages using similarity analysis."""
    try:
        from .persistence import detect_duplicates_async

        data = detect_duplicates_async(
            channel_name=channel,
            similarity_threshold=threshold,
            db_path=ctx.obj.get("db_path")
        )

        _write_output(data, output, output_format)
        
        if data.get("error"):
            click.echo(f"❌ {data['error']}", err=True)
        else:
            click.echo(
                f"✅ Duplicate detection completed! Found {data['total_duplicate_pairs']} duplicate pairs."
            )

    except Exception as e:
        click.echo(f"❌ Error detecting duplicates: {e}", err=True)
        if ctx.obj.get("verbose"):
            raise


@cli.command()
@click.option("--table", help="Specific table to export (messages, users, channels)")
@click.option("--output", type=click.Path(), help="Output file path")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "csv"]),
    default="csv",
    help="Output format",
)
@click.pass_context
def export_data(ctx, table: Optional[str], output: Optional[str], output_format: str):
    """Export data from the database."""
    _export_data(ctx.obj, table, output, output_format)


@cli.command(name="start")
@click.option("--token", help="Discord bot token (overrides environment variable)")
@click.option("--prefix", default="!", help="Bot command prefix (default: !)")
@click.option("--debug", is_flag=True, help="Run in debug mode")
@click.pass_context
def start(ctx, token: Optional[str], prefix: str, debug: bool):
    """Start the Discord bot for interactive analysis."""
    try:
        from pepino.discord.bot import run_bot as start_bot

        # Update settings if command line options provided
        settings = Settings()

        if token:
            settings.discord_token = token
        if prefix != "!":
            settings.command_prefix = prefix
        if debug:
            settings.debug = True
            settings.log_level = "DEBUG"

        # Validate required settings
        if not settings.discord_token:
            click.echo(
                "❌ Error: DISCORD_TOKEN is required. Set it via environment variable or --token option.",
                err=True,
            )
            click.echo(
                "   Example: export DISCORD_TOKEN='your_bot_token_here'", err=True
            )
            ctx.exit(1)

        click.echo("🤖 Starting Discord bot...")
        click.echo(f"   Command prefix: {settings.command_prefix}")
        click.echo(f"   Database path: {settings.db_path}")
        click.echo(f"   Debug mode: {settings.debug}")
        click.echo(f"   Log level: {settings.log_level}")
        click.echo("   Press Ctrl+C to stop the bot")
        click.echo("")

        # Start the bot
        start_bot()

    except KeyboardInterrupt:
        click.echo("\n🛑 Bot stopped by user")
    except ValueError as e:
        click.echo(f"❌ Configuration error: {e}", err=True)
        ctx.exit(1)
    except Exception as e:
        click.echo(f"❌ Failed to start bot: {e}", err=True)
        if debug:
            import traceback

            traceback.print_exc()
        ctx.exit(1)


@cli.group(name="sync")
@click.pass_context
def sync(ctx):
    """Sync Discord data and manage sync operations."""
    pass


@sync.command(name="run")
@click.option("--force", is_flag=True, help="Force sync even if data is fresh")
@click.option("--full", is_flag=True, help="Complete re-sync (re-downloads everything)")
@click.option(
    "--clear", is_flag=True, help="Clear existing data before sync (use with --full)"
)
@click.option("--timeout", default=300, help="Sync timeout in seconds (default: 300)")
@click.pass_context
def sync_run(ctx, force: bool, full: bool, clear: bool, timeout: int):
    """Sync Discord data (smart: only updates if data is stale)."""
    if full:
        # Full sync mode
        timeout = (
            timeout if timeout != 300 else 600
        )  # Increase default timeout for full sync
        asyncio.run(
            _sync_data(
                ctx.obj,
                force=True,
                timeout=timeout,
                incremental=False,
                clear_existing=clear,
            )
        )
    else:
        # Incremental sync mode
        if clear:
            click.echo("⚠️  --clear option only works with --full flag", err=True)
            ctx.exit(1)
        asyncio.run(_sync_data(ctx.obj, force, timeout, incremental=True))


@sync.command(name="status")
@click.pass_context
def sync_status(ctx):
    """Check sync status and data freshness."""
    asyncio.run(_sync_status(ctx.obj))


@cli.group(name="list")
@click.pass_context
def list_cmd(ctx):
    """List available data for analysis and automation."""
    pass


@list_cmd.command(name="channels")
@click.option("--limit", "-l", default=0, help="Number of channels to show (0 for all)")
@click.option("--output", "-o", help="Output file (JSON or CSV)")
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json", "csv"]),
    help="Output format",
)
@click.pass_context
def list_channels(ctx, limit: int, output: Optional[str], output_format: str):
    """List all available channels for analysis."""
    _list_channels(ctx.obj, limit, output, output_format)


@list_cmd.command(name="users")
@click.option("--limit", "-l", default=0, help="Number of users to show (0 for all)")
@click.option("--output", "-o", help="Output file (JSON or CSV)")
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json", "csv"]),
    help="Output format",
)
@click.pass_context
def list_users(ctx, limit: int, output: Optional[str], output_format: str):
    """List all available users for analysis."""
    _list_users(ctx.obj, limit, output, output_format)


@list_cmd.command(name="stats")
@click.option("--output", "-o", help="Output file (JSON or CSV)")
@click.option(
    "--format",
    "output_format",
    default="text",
    type=click.Choice(["text", "json", "csv"]),
    help="Output format",
)
@click.pass_context
def list_stats(ctx, output: Optional[str], output_format: str):
    """Show database statistics for automation planning."""
    _list_stats(ctx.obj, output, output_format)


def _analyze_conversations(
    ctx_obj: Dict[str, Any],
    channel: Optional[str],
    output: Optional[str],
    output_format: str,
):
    """Analyze conversation patterns."""
    try:
        conversation_service = ConversationService()

        # Analyze conversations
        conversations = conversation_service.analyze_conversations(
            ctx_obj["db_path"], channel
        )

        # Calculate engagement
        engagement = conversation_service.calculate_engagement(
            ctx_obj["db_path"], channel
        )

        data = {"conversations": conversations, "engagement": engagement}

        _write_output(data, output, output_format)
        click.echo("✅ Conversation analysis completed successfully!")

    except Exception as e:
        click.echo(f"❌ Error analyzing conversations: {e}")
        if ctx_obj["verbose"]:
            raise


def _find_similar(
    ctx_obj: Dict[str, Any],
    query: str,
    limit: int,
    threshold: float,
    output: Optional[str],
    output_format: str,
):
    """Find similar messages."""
    try:
        embedding_service = EmbeddingService()
        embedding_service.initialize()

        similar_messages = embedding_service.find_similar_messages(
            ctx_obj["db_path"], query, limit, threshold
        )

        data = {
            "query": query,
            "similar_messages": similar_messages,
            "total_found": len(similar_messages),
        }

        _write_output(data, output, output_format)
        click.echo(f"✅ Found {len(similar_messages)} similar messages!")

    except Exception as e:
        click.echo(f"❌ Error finding similar messages: {e}")
        if ctx_obj["verbose"]:
            raise


def _generate_embeddings(
    ctx_obj: Dict[str, Any], batch_size: int, output: Optional[str], output_format: str
):
    """Generate embeddings for messages."""
    try:
        embedding_service = EmbeddingService()
        embedding_service.initialize()

        with click.progressbar(length=100, label="Generating embeddings") as bar:
            processed_count = embedding_service.batch_process_messages(
                ctx_obj["db_path"], batch_size=batch_size
            )
            bar.update(100)

        # Get statistics
        stats = embedding_service.get_embedding_statistics(ctx_obj["db_path"])

        data = {
            "embedding_stats": stats,
            "processed": processed_count,
            "total_messages": stats.get("total_messages", 0),
            "embedded_messages": stats.get("embedded_messages", 0),
            "coverage_percentage": stats.get("coverage_percentage", 0),
        }

        _write_output(data, output, output_format)
        click.echo(f"✅ Generated embeddings for {processed_count} messages!")

    except Exception as e:
        click.echo(f"❌ Error generating embeddings: {e}")
        if ctx_obj["verbose"]:
            raise


def _export_data(
    ctx_obj: Dict[str, Any],
    table: Optional[str],
    output: Optional[str],
    output_format: str,
):
    """Export data from database."""
    try:
        if table:
            # Export specific table
            data = persistence.export_table_data_async(table, output, output_format, ctx_obj["db_path"])
        else:
            # Export all tables
            data = persistence.export_all_tables_async(ctx_obj["db_path"])

        _write_output(data, output, output_format)
        click.echo("✅ Data export completed successfully!")

    except Exception as e:
        click.echo(f"❌ Error exporting data: {e}")
        if ctx_obj.get("verbose"):
            raise


def _format_channel_analysis_for_cli(result: Dict[str, Any]) -> str:
    """Format comprehensive channel analysis for CLI display (Discord-style)."""
    channel_info = result["channel_info"]
    stats = result["statistics"]
    top_users = result.get("top_users", [])
    engagement = result.get("engagement_metrics")
    peak_activity = result.get("peak_activity")
    recent_activity = result.get("recent_activity", [])
    health_metrics = result.get("health_metrics")
    top_topics = result.get("top_topics", [])
    total_human_members = result.get("total_human_members", 0)

    def format_timestamp(timestamp: str) -> str:
        try:
            if not timestamp:
                return "Unknown"
            from datetime import datetime
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M")
        except:
            return str(timestamp)[:16] if timestamp else "Unknown"

    output = f"📊 Channel Analysis: #{channel_info['channel_name']}\n\n"
    
    # Basic Statistics with Human/Bot breakdown
    output += "Basic Statistics:\n"
    output += f"• Total Messages: {stats['total_messages']:,}\n"
    if stats.get('human_messages', 0) > 0 and stats.get('bot_messages', 0) > 0:
        human_pct = (stats['human_messages'] / stats['total_messages'] * 100)
        bot_pct = (stats['bot_messages'] / stats['total_messages'] * 100)
        output += f"• Human Messages: {stats['human_messages']:,} ({human_pct:.1f}%)\n"
        output += f"• Bot Messages: {stats['bot_messages']:,} ({bot_pct:.1f}%)\n"
    output += f"• Total Unique Users: {stats['unique_users']:,}\n"
    if stats.get('unique_human_users', 0) > 0:
        percentage_str = ""
        if total_human_members and total_human_members > 0:
            percentage = (stats['unique_human_users'] / total_human_members * 100)
            percentage_str = f" ({percentage:.2f}%)"
        output += f"• Unique Human Users: {stats['unique_human_users']:,}{percentage_str}\n"
    output += f"• Average Message Length: {stats['avg_message_length']:.1f} characters\n"
    output += f"• First Message: {format_timestamp(stats['first_message'])}\n"
    output += f"• Last Message: {format_timestamp(stats['last_message'])}\n\n"

    # Human Engagement Metrics
    if engagement:
        output += "📈 Human Engagement Metrics:\n"
        output += f"• Average Replies per Original Post: {engagement['replies_per_post']:.2f}\n"
        output += f"• Posts with Reactions: {engagement['reaction_rate']:.1f}% ({engagement['posts_with_reactions']}/{stats.get('human_messages', stats['total_messages'])})\n"
        output += f"• Total Replies: {engagement['total_replies']:,} | Original Posts: {engagement['original_posts']:,}\n"
        output += f"• Note: Bot messages excluded from engagement calculations\n\n"

    # Top Contributors
    if top_users:
        output += "👥 Top Human Contributors:\n"
        for user in top_users[:5]:
            display_name = user.get('display_name') or user.get('author_name', 'Unknown')
            output += f"• {display_name} ({user['message_count']:,} messages)\n"
        output += "\n"

    # Peak Activity Times
    if peak_activity:
        if peak_activity.get('peak_hours'):
            output += "Peak Activity Hours:\n"
            for hour_data in peak_activity['peak_hours'][:3]:
                output += f"• {hour_data['hour']}: {hour_data['messages']:,} messages\n"
            output += "\n"
        
        if peak_activity.get('peak_days'):
            output += "Peak Activity Days:\n"
            for day_data in peak_activity['peak_days'][:3]:
                output += f"• {day_data['day']}: {day_data['messages']:,} messages\n"
            output += "\n"

    # Recent Activity (Last 7 Days)
    if recent_activity:
        output += "Recent Activity (Last 7 Days):\n"
        for activity in recent_activity[:7]:
            output += f"• {activity['date']}: {activity['messages']:,} messages\n"
        output += "\n"

    # Channel Health Metrics
    if health_metrics:
        output += "📈 Channel Health Metrics:\n"
        output += f"• Weekly Active Users: {health_metrics['weekly_active']:,}\n"
        output += f"• Inactive Users: {health_metrics['inactive_users']:,}\n"
        if health_metrics.get('total_channel_members', 0) > 0:
            output += f"• Total Channel Members: {health_metrics['total_channel_members']:,}\n"
            output += f"• Participation Rate: {(health_metrics['participation_rate'] * 100):.1f}%\n"
        output += "\n"

    # Top Topics Discussed
    if top_topics:
        output += "🔍 Top Topics:\n"
        for i, topic in enumerate(top_topics[:5], 1):
            if topic and topic.strip():  # Only show non-empty topics
                output += f"• {topic}\n"

    return output


def _write_output(data: Dict[str, Any], output: Optional[str], output_format: str):
    """Write output to file or stdout."""
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_format == "json":
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        elif output_format == "csv":
            _write_csv(data, output_path)
        else:  # text
            with open(output_path, "w") as f:
                _write_text(data, f)
    else:
        if output_format == "json":
            click.echo(json.dumps(data, indent=2, default=str))
        elif output_format == "csv":
            _write_csv(data, None)
        else:  # text
            _write_text(data, None)


def _write_csv(data: Dict[str, Any], output_path: Optional[Path]):
    """Write data as CSV."""
    if "tables" in data:
        # Multiple tables
        for table_name, table_data in data["tables"].items():
            if output_path:
                csv_path = output_path.parent / f"{table_name}.csv"
            else:
                csv_path = None

            if csv_path:
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=table_data["columns"])
                    writer.writeheader()
                    writer.writerows(table_data["rows"])
            else:
                # Write to stdout
                import sys

                writer = csv.DictWriter(sys.stdout, fieldnames=table_data["columns"])
                writer.writeheader()
                writer.writerows(table_data["rows"])
    else:
        # Single table or analysis result
        if output_path:
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                _flatten_dict_to_csv(data, writer)
        else:
            import sys

            writer = csv.writer(sys.stdout)
            _flatten_dict_to_csv(data, writer)


def _flatten_dict_to_csv(data: Dict[str, Any], writer):
    """Flatten nested dictionary to CSV format."""

    def flatten_dict(d, parent_key="", sep="_"):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(
                            flatten_dict(item, f"{new_key}_{i}", sep=sep).items()
                        )
                    else:
                        items.append((f"{new_key}_{i}", item))
            else:
                items.append((new_key, v))
        return dict(items)

    flattened = flatten_dict(data)
    writer.writerow(flattened.keys())
    writer.writerow(flattened.values())


def _write_text(data: Dict[str, Any], output_file):
    """Write data as formatted text."""

    def write_text_recursive(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                output_file.write(f"{'  ' * indent}{key}:\n")
                write_text_recursive(value, indent + 1)
            elif isinstance(value, list):
                output_file.write(f"{'  ' * indent}{key}:\n")
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        output_file.write(f"{'  ' * (indent + 1)}[{i}]:\n")
                        write_text_recursive(item, indent + 2)
                    else:
                        output_file.write(f"{'  ' * (indent + 1)}[{i}]: {item}\n")
            else:
                output_file.write(f"{'  ' * indent}{key}: {value}\n")

    if output_file:
        write_text_recursive(data)
    else:
        # Write to stdout
        import sys

        for key, value in data.items():
            if isinstance(value, dict):
                click.echo(f"{key}:")
                for k, v in value.items():
                    click.echo(f"  {k}: {v}")
            elif isinstance(value, list):
                click.echo(f"{key}:")
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        click.echo(f"  [{i}]:")
                        for k, v in item.items():
                            click.echo(f"    {k}: {v}")
                    else:
                        click.echo(f"  [{i}]: {item}")
            else:
                click.echo(f"{key}: {value}")


async def _sync_data(
    ctx_obj: Dict[str, Any],
    force: bool,
    timeout: int,
    incremental: bool = True,
    clear_existing: bool = False,
):
    """Perform data sync operation."""
    try:
        import asyncio

        from pepino.discord.sync.sync_manager import SyncManager

        # Create sync manager
        sync_manager = SyncManager(ctx_obj["db_path"])

        click.echo("🔄 Starting Discord data sync...")
        if incremental and not force:
            click.echo("   Mode: Incremental (only syncs if data is stale)")
        elif incremental:
            click.echo("   Mode: Incremental (forced)")
        else:
            click.echo("   Mode: Full sync")
            if clear_existing:
                click.echo("   ⚠️  Clearing existing data first")

        # Run sync with timeout
        try:
            if incremental:
                result = await asyncio.wait_for(
                    sync_manager.run_incremental_sync(force=force), timeout=timeout
                )
            else:
                result = await asyncio.wait_for(
                    sync_manager.run_full_sync(clear_existing=clear_existing),
                    timeout=timeout,
                )

            # Display results
            if result.sync_performed:
                click.echo(f"✅ Sync completed successfully!")
                click.echo(f"   New messages: {getattr(result, 'new_messages', 0)}")
                if hasattr(result, "updated_channels"):
                    click.echo(f"   Updated channels: {result.updated_channels}")
                click.echo(f"   Duration: {result.duration:.1f}s")

                if hasattr(result, "last_sync") and result.last_sync:
                    click.echo(f"   Last sync: {result.last_sync}")
            else:
                reason = getattr(
                    result, "reason", getattr(result, "error", "Unknown reason")
                )
                click.echo(f"ℹ️  Sync not performed: {reason}")

        except asyncio.TimeoutError:
            click.echo(f"⏰ Sync operation timed out after {timeout} seconds")
            ctx_obj["exit_code"] = 1

    except Exception as e:
        click.echo(f"❌ Error during sync: {e}")
        if ctx_obj.get("verbose"):
            import traceback

            traceback.print_exc()
        ctx_obj["exit_code"] = 1


async def _sync_status(ctx_obj: Dict[str, Any]):
    """Check sync status."""
    try:
        from datetime import datetime, timezone

        from pepino.discord.sync.sync_manager import SyncManager

        # Create sync manager
        sync_manager = SyncManager(ctx_obj["db_path"])

        # Get last sync info
        last_sync = await sync_manager.get_last_sync_time()
        is_stale = await sync_manager.is_data_stale()

        click.echo("📊 Sync Status")
        click.echo("=" * 40)

        if last_sync:
            click.echo(f"Last sync: {last_sync}")

            # Calculate time since last sync
            now = datetime.now(timezone.utc)
            time_diff = now - last_sync
            hours = int(time_diff.total_seconds() / 3600)
            minutes = int((time_diff.total_seconds() % 3600) / 60)

            click.echo(f"Time since last sync: {hours}h {minutes}m ago")

            if is_stale:
                click.echo("🟡 Data is stale (sync recommended)")
            else:
                click.echo("🟢 Data is fresh")
        else:
            click.echo("❌ No sync records found")
            click.echo("🔄 Initial sync required")

        # Get database info
        try:
            # Get database statistics using persistence layer
            stats = await persistence.get_database_statistics_async(ctx_obj["db_path"])

            click.echo(f"\nDatabase Statistics:")
            click.echo(f"Messages: {stats['message_count']:,}")
            click.echo(f"Channels: {stats['channel_count']}")
            click.echo(f"Users: {stats['user_count']}")

        except Exception as e:
            click.echo(f"⚠️  Could not retrieve database statistics: {e}")

    except Exception as e:
        click.echo(f"❌ Error checking sync status: {e}")
        if ctx_obj.get("verbose"):
            import traceback

            traceback.print_exc()


def _list_channels(
    ctx_obj: Dict[str, Any], limit: int, output: Optional[str], output_format: str
):
    """List all available channels for analysis."""
    try:
        from .persistence import get_database_manager
        
        with get_database_manager(ctx_obj.get("db_path")) as db_manager:
            settings = Settings()
            from pepino.analysis.data_facade import get_analysis_data_facade
            data_facade = get_analysis_data_facade(db_manager, settings.base_filter)
            channel_analyzer = ChannelAnalyzer(data_facade)

            # Get all channels
            channels = channel_analyzer.get_available_channels()

            if not channels:
                click.echo("❌ No channels found in database")
                return

            # Apply limit if specified
            if limit > 0:
                channels = channels[:limit]

            # Prepare data for output
            if output_format in ["json", "csv"]:
                # Rich format with metadata - use data facade instead of direct queries
                channel_data = []
                for channel in channels:
                    try:
                        # Get basic channel statistics using data facade repository
                        stats_data = data_facade.channel_repository.get_channel_message_statistics(channel)
                        
                        if stats_data:
                            channel_data.append(
                                {
                                    "name": channel,
                                    "message_count": stats_data.get("total_messages", 0),
                                    "unique_users": stats_data.get("unique_users", 0),
                                    "avg_message_length": round(stats_data.get("avg_message_length", 0.0), 2),
                                }
                            )
                        else:
                            channel_data.append(
                                {
                                    "name": channel,
                                    "message_count": 0,
                                    "unique_users": 0,
                                    "avg_message_length": 0.0,
                                }
                            )
                    except:
                        # Fallback for channels without stats
                        channel_data.append(
                            {
                                "name": channel,
                                "message_count": 0,
                                "unique_users": 0,
                                "avg_message_length": 0.0,
                            }
                        )

                data = {"channels": channel_data, "total_count": len(channels)}
            else:
                # Simple text format - just names
                data = {"channels": channels, "total_count": len(channels)}

            # Handle output based on format
            if output_format == "text" and not output:
                # Use template for console text output - need rich data like JSON format
                channel_data = []
                for channel in channels:
                    try:
                        # Get basic channel statistics using data facade repository
                        stats_data = data_facade.channel_repository.get_channel_message_statistics(channel)
                        
                        if stats_data:
                            channel_data.append(
                                {
                                    "channel_name": channel,
                                    "name": channel,
                                    "message_count": stats_data.get("total_messages", 0),
                                    "unique_users": stats_data.get("unique_users", 0),
                                    "avg_message_length": stats_data.get("avg_message_length", 0.0),
                                }
                            )
                        else:
                            channel_data.append(
                                {
                                    "channel_name": channel,
                                    "name": channel,
                                    "message_count": 0,
                                    "unique_users": 0,
                                    "avg_message_length": 0.0,
                                }
                            )
                    except:
                        # Fallback for channels without stats
                        channel_data.append(
                            {
                                "channel_name": channel,
                                "name": channel,
                                "message_count": 0,
                                "unique_users": 0,
                                "avg_message_length": 0.0,
                            }
                        )
                
                template_data = {
                    "items": channel_data,
                    "total_count": len(channel_analyzer.get_available_channels()),  # Total in DB
                    "showing_count": len(channels),
                    "has_more": limit > 0 and len(channels) >= limit
                }
                
                try:
                    from pepino.templates.template_engine import TemplateEngine
                    template_engine = TemplateEngine()
                    result = template_engine.render_template("outputs/cli/channel_list.txt.j2", **template_data)
                    click.echo(result)
                except Exception as e:
                    # Fallback to simple output and show error if verbose
                    if ctx_obj.get("verbose"):
                        click.echo(f"Template error: {e}", err=True)
                    click.echo(f"📺 Found {len(channels)} channels:")
                    for i, channel in enumerate(channels, 1):
                        click.echo(f"{i:3d}. {channel}")
                    if limit > 0 and len(channels) == limit:
                        click.echo(f"\n*Showing first {limit} channels. Use --limit 0 to see all.*")
            else:
                # File output or non-text format
                _write_output(data, output, output_format)
                if output_format != "text":
                    click.echo(f"✅ Listed {len(channels)} channels")

    except Exception as e:
        click.echo(f"❌ Error listing channels: {e}", err=True)
        if ctx_obj.get("verbose"):
            raise


def _list_users(
    ctx_obj: Dict[str, Any], limit: int, output: Optional[str], output_format: str
):
    """List all available users for analysis."""
    try:
        from .persistence import get_database_manager
        
        with get_database_manager(ctx_obj.get("db_path")) as db_manager:
            settings = Settings()
            from pepino.analysis.data_facade import get_analysis_data_facade
            data_facade = get_analysis_data_facade(db_manager, settings.base_filter)
            user_analyzer = UserAnalyzer(data_facade)

            # Get all users
            users = user_analyzer.get_available_users()

            if not users:
                click.echo("❌ No users found in database")
                return

            # Apply limit if specified
            if limit > 0:
                users = users[:limit]

            # Prepare data for output
            if output_format in ["json", "csv"]:
                # Rich format with metadata
                user_data = []
                for user in users:
                    try:
                        # Get basic user statistics using data facade repository
                        stats_data = data_facade.user_repository.get_user_message_statistics(user)
                        
                        if stats_data:
                            user_data.append(
                                {
                                    "name": user,
                                    "display_name": user,
                                    "author_id": "",  # We don't have easy access to author_id in this context
                                    "message_count": stats_data.get("total_messages", 0),
                                    "channels_active": stats_data.get("channels_active", 0),
                                }
                            )
                        else:
                            user_data.append(
                                {
                                    "name": user,
                                    "display_name": user,
                                    "author_id": "",
                                    "message_count": 0,
                                    "channels_active": 0,
                                }
                            )
                    except:
                        # Fallback for users without stats
                        user_data.append(
                            {
                                "name": user,
                                "display_name": user,
                                "author_id": "",
                                "message_count": 0,
                                "channels_active": 0,
                            }
                        )

                data = {"users": user_data, "total_count": len(users)}
            else:
                # Simple text format - just names
                data = {"users": users, "total_count": len(users)}

            _write_output(data, output, output_format)

            # Console output
            if output_format == "text":
                click.echo(f"👥 Found {len(users)} users:")
                for i, user in enumerate(users, 1):
                    click.echo(f"{i:3d}. {user}")
                if limit > 0 and len(users) == limit:
                    click.echo(
                        f"\n*Showing first {limit} users. Use --limit 0 to see all.*"
                    )
            else:
                click.echo(f"✅ Listed {len(users)} users")

    except Exception as e:
        click.echo(f"❌ Error listing users: {e}", err=True)
        if ctx_obj.get("verbose"):
            raise


def _list_stats(
    ctx_obj: Dict[str, Any], output: Optional[str], output_format: str
):
    """Show database statistics for automation planning."""
    try:
        from .persistence import get_database_manager
        
        with get_database_manager(ctx_obj.get("db_path")) as db_manager:
            settings = Settings()
            from pepino.analysis.data_facade import get_analysis_data_facade
            data_facade = get_analysis_data_facade(db_manager, settings.base_filter)

            # Get basic counts using data facade repository
            channels = data_facade.channel_repository.get_available_channels()
            user_count = data_facade.message_repository.get_distinct_user_count()
            message_count = data_facade.message_repository.get_total_message_count()

            # Prepare stats data
            stats_data = {
                "database_path": ctx_obj.get("db_path", "data/discord_messages.db"),
                "total_channels": len(channels),
                "total_users": user_count,
                "total_messages": message_count,
                "base_filter": settings.base_filter,
                "channels": channels[:10]
                if output_format == "text"
                else channels,  # Sample for text, all for structured
            }

            _write_output({"stats": stats_data}, output, output_format)

            # Console output
            if output_format == "text":
                click.echo("📊 Database Statistics:")
                click.echo(f"   Database: {stats_data['database_path']}")
                click.echo(f"   Channels: {stats_data['total_channels']:,}")
                click.echo(f"   Users: {stats_data['total_users']:,}")
                click.echo(f"   Messages: {stats_data['total_messages']:,}")
                click.echo(f"   Filter: {stats_data['base_filter'][:100]}...")
                click.echo(
                    f"\n📺 Sample Channels ({min(10, len(channels))} of {len(channels)}):"
                )
                for i, channel in enumerate(channels[:10], 1):
                    click.echo(f"   {i:2d}. {channel}")
            else:
                click.echo(
                    f"✅ Database stats: {message_count:,} messages, {len(channels)} channels, {user_count} users"
                )

    except Exception as e:
        click.echo(f"❌ Error getting database stats: {e}", err=True)
        if ctx_obj.get("verbose"):
            raise
