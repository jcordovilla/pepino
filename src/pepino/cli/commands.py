"""
CLI Commands for Pepino Discord Analytics

Provides command-line interface for analyzing Discord data with template integration.
"""

import json
import logging
import sys
import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import click

from pepino.analysis.services import UnifiedAnalysisService, analysis_service
from pepino.data_operations.service import data_operations_service

logger = logging.getLogger(__name__)


def validate_and_normalize_channel(channel_name: str, available_channels: List[str]) -> str:
    """Validate and normalize channel name, providing helpful error messages."""
    if not channel_name:
        return channel_name
    
    # Normalize channel name (remove # prefix if present)
    normalized = channel_name.lstrip('#')
    
    # Check if normalized channel exists
    if normalized in available_channels:
        return normalized
    
    # If not found, try with # prefix
    with_hash = f"#{normalized}"
    if with_hash in available_channels:
        return with_hash
    
    # If still not found, provide helpful error message
    similar_channels = []
    for available in available_channels:
        if normalized.lower() in available.lower() or available.lower() in normalized.lower():
            similar_channels.append(available)



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
    """Pepino Discord Analytics CLI."""
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db_path
    ctx.obj["verbose"] = verbose





@cli.group(name="analyze")
@click.pass_context
def analyze(ctx):
    """analyze [pulsecheck|top-channels|top-contributors]"""
    """Analyze Discord data."""
    pass


@cli.command(name="start")
@click.option("--token", help="Discord bot token (overrides environment variable)")
@click.option("--prefix", default="!", help="Bot command prefix (default: !)")
@click.option("--debug", is_flag=True, help="Run in debug mode")
@click.pass_context
def start(ctx, token: Optional[str], prefix: str, debug: bool):
    """Start the Discord bot."""
    import os
    if token:
        os.environ["DISCORD_TOKEN"] = token
    if prefix:
        os.environ["DISCORD_PREFIX"] = prefix
    if debug:
        os.environ["PEPINO_DEBUG"] = "1"
    from pepino.discord_bot.bot import run_bot
    try:
        run_bot()
    except KeyboardInterrupt:
        click.echo("Bot stopped by user")
    except Exception as e:
        click.echo(f"❌ Bot failed to start: {e}")
        if ctx.obj.get("verbose"):
            raise


@cli.group(name="list")
@click.pass_context
def list_cmd(ctx):
    """List available data."""
    pass


@cli.group(name="data")
@click.pass_context
def data_cmd(ctx):
    """Data operations (export, schema, etc.)."""
    pass


@data_cmd.command(name="tables")
@click.pass_context
def list_tables(ctx):
    """List available tables for export."""
    with data_operations_service(ctx.obj["db_path"]) as service:
        tables = service.get_available_tables()
        click.echo("📋 Available Tables:")
        for table in tables:
            schema = service.get_table_schema(table)
            click.echo(f"• {table}: {schema.get('description', 'No description')}")


@data_cmd.command(name="schema")
@click.argument("table", required=False)
@click.pass_context
def show_schema(ctx, table: Optional[str]):
    """Show schema for a specific table or all tables if none specified."""
    with data_operations_service(ctx.obj["db_path"]) as service:
        if table:
            # Show schema for specific table
            schema = service.get_table_schema(table)
            if schema.get('columns'):
                click.echo(f"📋 Schema for table '{table}':")
                click.echo(f"Description: {schema.get('description', 'No description')}")
                click.echo("Columns:")
                for column in schema['columns']:
                    click.echo(f"  • {column}")
            else:
                click.echo(f"❌ Table '{table}' not found")
        else:
            # Show schemas for all tables
            tables = service.get_available_tables()
            click.echo("📋 Available Tables and Schemas:")
            click.echo("=" * 50)
            
            for table_name in tables:
                schema = service.get_table_schema(table_name)
                click.echo(f"\n📋 {table_name}:")
                click.echo(f"Description: {schema.get('description', 'No description')}")
                click.echo("Columns:")
                for column in schema.get('columns', []):
                    click.echo(f"  • {column}")


@data_cmd.command(name="export")
@click.argument("table")
@click.option("--output", "-o", help="Output file path")
@click.option("--format", "output_format", type=click.Choice(["csv", "json", "excel"]), default="csv", help="Output format")
@click.option("--no-metadata", is_flag=True, help="Exclude metadata from export")
@click.pass_context
def export_table(ctx, table: str, output: Optional[str], output_format: str, no_metadata: bool):
    """Export a specific table."""
    with data_operations_service(ctx.obj["db_path"]) as service:
        result = service.export_table(
            table=table,
            output_path=output,
            format=output_format
        )
        click.echo(result)


@data_cmd.command(name="export-all")
@click.option("--output", "-o", help="Output file path")
@click.option("--format", "output_format", type=click.Choice(["json", "excel"]), default="json", help="Output format")
@click.option("--no-metadata", is_flag=True, help="Exclude metadata from export")
@click.option("--tables", help="Comma-separated list of tables to export (e.g., messages,users,channels)")
@click.pass_context
def export_all_tables(ctx, output: Optional[str], output_format: str, no_metadata: bool, tables: Optional[str]):
    """Export all tables or specified tables."""
    with data_operations_service(ctx.obj["db_path"]) as service:
        # Parse tables if specified
        table_list = None
        if tables:
            table_list = [table.strip() for table in tables.split(",")]
            # Validate tables exist
            available_tables = service.get_available_tables()
            invalid_tables = [t for t in table_list if t not in available_tables]
            if invalid_tables:
                click.echo(f"❌ Invalid tables: {', '.join(invalid_tables)}")
                click.echo(f"Available tables: {', '.join(available_tables)}")
                return
        
        # Export specified tables or all tables
        if table_list:
            # Export specific tables
            all_data = {}
            for table in table_list:
                try:
                    result = service.export_table(table=table, output_path=None, format=output_format)
                    # Parse the result to get the data
                    if "exported to" not in result:  # If it's not a file export, it's data
                        # Parse the JSON string back to an object
                        if output_format == "json":
                            try:
                                parsed_result = json.loads(result)
                                all_data[table] = parsed_result
                            except json.JSONDecodeError:
                                # If it's not valid JSON, store as string
                                all_data[table] = {"raw_data": result}
                        else:
                            all_data[table] = {"raw_data": result}
                except Exception as e:
                    click.echo(f"⚠️ Could not export table '{table}': {e}")
            
            # Write to file if specified
            if output:
                output_path = Path(output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                if output_format == "json":
                    export_data = {
                        'exported_at': datetime.now().isoformat(),
                        'tables': all_data
                    }
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(export_data, f, indent=2, ensure_ascii=False)
                    click.echo(f"Specified tables exported to {output_path}")
                else:
                    click.echo("Excel format not supported for selective table export")
            else:
                click.echo(json.dumps(all_data, indent=2))
        else:
            # Export all tables (original behavior)
            result = service.export_data(
                table=None,  # None means export all tables
                output_path=output,
                format=output_format,
                include_metadata=not no_metadata
            )
            click.echo(result)


@data_cmd.command(name="clear")
@click.option("--confirm", is_flag=True, help="Confirm database clearing")
@click.pass_context
def data_clear(ctx, confirm: bool):
    """Clear the database for a fresh start."""
    if not confirm:
        click.echo("⚠️  This will delete all data! Use --confirm to proceed.")
        return
    
    with data_operations_service(ctx.obj["db_path"]) as service:
        service.clear_database()
        click.echo("🗑️ Database cleared successfully!")


@data_cmd.command(name="sync-status")
@click.pass_context
def data_sync_status(ctx):
    """Show sync status and statistics."""
    import asyncio

    async def show_status():
        with data_operations_service(ctx.obj["db_path"]) as service:
            status = await service.get_sync_status()
            
            click.echo("📊 Sync Status:")
            click.echo(f"• Last Sync: {status.get('last_sync', 'Never')}")
            click.echo(f"• Status: {status.get('status', 'Unknown')}")
            click.echo(f"• Data Stale: {'Yes' if status.get('is_stale') else 'No'}")

    try:
        asyncio.run(show_status())
    except Exception as e:
        click.echo(f"❌ Failed to get sync status: {e}")
        if ctx.obj.get("verbose"):
            raise


@data_cmd.command(name="sync")
@click.option("--force", is_flag=True, help="Force sync even if data is fresh")
@click.option("--full", is_flag=True, help="Complete re-sync (re-downloads everything)")
@click.option(
    "--clear", is_flag=True, help="Clear existing data before sync (use with --full)"
)
@click.pass_context
def data_sync(ctx, force: bool, full: bool, clear: bool):
    """Synchronize Discord data."""
    import asyncio

    async def run_sync():
        with data_operations_service(ctx.obj["db_path"]) as service:
            result = await service.sync_data(force=force, full=full, clear_existing=clear)
            
            if result.get('sync_performed'):
                click.echo("✅ Sync completed successfully!")
                if result.get('new_messages'):
                    click.echo(f"📝 New messages: {result['new_messages']}")
                if result.get('duration'):
                    click.echo(f"⏱️ Duration: {result['duration']:.1f}s")
            else:
                click.echo(f"ℹ️ {result.get('reason', 'No sync performed')}")
    
    try:
        asyncio.run(run_sync())
    except Exception as e:
        click.echo(f"❌ Sync failed: {e}")
        if ctx.obj.get("verbose"):
            raise


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
    with analysis_service() as service:
        output_str = service.list_channels(output_format="cli")
        
        if output:
            with open(output, "w") as f:
                f.write(output_str)
            click.echo(f"Output written to: {output}")
        else:
            click.echo(output_str)


@analyze.command('top-contributors')
@click.option('--channel', help='Channel name to analyze (default: all channels)')
@click.option('--limit', default=10, help='Number of top users to show (default: 10)')
@click.option('--end-date', type=click.DateTime(formats=['%Y-%m-%d']), default=datetime.now, help='End date for analysis (default: today)')
@click.pass_context
def top_contributors(ctx, channel, limit, end_date):
    """Show top contributors analysis in CLI format."""
    days_back = 30  # Default to 30 days for contributor analysis
    
    # Get available channels for validation using data operations service
    with data_operations_service(ctx.obj["db_path"]) as data_service:
        if channel:
            available_channels = data_service.get_available_channels()
            
            # Validate and normalize channel name
            try:
                normalized_channel = validate_and_normalize_channel(channel, available_channels)
            except click.BadParameter as e:
                click.echo(f"❌ {e}")
                return
    
    # Run analysis using analysis service
    with analysis_service() as service:
        if channel:
            # Use the new public interface
            output = service.top_contributors(channel_name=normalized_channel, limit=limit, 
                                            days_back=days_back, end_date=end_date, output_format="cli")
        else:
            # Use the new public interface
            output = service.top_contributors(limit=limit, days_back=days_back, 
                                            end_date=end_date, output_format="cli")
        
        click.echo(output)


@analyze.command('pulsecheck')
@click.option('--channel', help='Channel name to analyze (default: all channels)')
@click.option('--end-date', type=click.DateTime(formats=['%Y-%m-%d']), default=datetime.now, help='End date for analysis (default: today)')
@click.pass_context
def pulsecheck(ctx, channel, end_date):
    """Show weekly channel analysis in CLI format."""
    days_back = 7  # Fixed for weekly analysis
    
    # Get available channels for validation using data operations service
    with data_operations_service(ctx.obj["db_path"]) as data_service:
        if channel:
            available_channels = data_service.get_available_channels()
            
            # Validate and normalize channel name
            try:
                normalized_channel = validate_and_normalize_channel(channel, available_channels)
            except click.BadParameter as e:
                click.echo(f"❌ {e}")
                return
    
    # Run analysis using analysis service
    with analysis_service() as service:
        # Use the unified pulsecheck method that handles both single channel and all channels
        output = service.pulsecheck(
            channel_name=normalized_channel if channel else None, 
            days_back=days_back, 
            end_date=end_date, 
            output_format="cli"
        )
        click.echo(output)


@analyze.command('top-channels')
@click.option('--end-date', type=click.DateTime(formats=['%Y-%m-%d']), default=datetime.now, help='End date for analysis (default: today)')
@click.option('--limit', default=5, help='Number of top channels to show (default: 5)')
@click.option('--days-back', default=7, help='Number of days to look back (default: 7)')
@click.pass_context
def top_channels(ctx, end_date, limit, days_back):
    """Show top channels summary report with most active channels and key insights."""
    
    with analysis_service(ctx.obj["db_path"]) as service:
        # Use the new public interface
        output = service.top_channels(limit=limit, days_back=days_back, 
                                    end_date=end_date, output_format="cli")
        click.echo(output)


@analyze.command('database-stats')
@click.pass_context
def database_stats(ctx):
    """Show database statistics and health report."""
    try:
        with analysis_service(ctx.obj["db_path"]) as service:
            result = service.database_stats(output_format="cli")
            click.echo(result)
    except Exception as e:
        click.echo(f"❌ Database analysis failed: {e}")
        if ctx.obj.get("verbose"):
            raise


@analyze.command('detailed-user')
@click.argument('username')
@click.option('--days-back', type=int, default=30, help='Number of days to look back (default: 30)')
@click.option('--format', 'output_format', type=click.Choice(['cli', 'discord']), default='cli', help='Output format')
@click.pass_context
def detailed_user(ctx, username, days_back, output_format):
    """Show detailed user analysis (new system, unique template)."""
    try:
        with analysis_service(ctx.obj["db_path"]) as service:
            result = service.detailed_user_analysis(username, days_back, output_format)
            click.echo(result)
    except Exception as e:
        click.echo(f"❌ Detailed user analysis failed: {e}")
        if ctx.obj.get("verbose"):
            raise


@analyze.command('detailed-topic')
@click.option('--channel', 'channel_name', help='Channel name to analyze (default: all channels)')
@click.option('--n-topics', type=int, default=10, help='Number of topics to extract (default: 10)')
@click.option('--days-back', type=int, help='Number of days to look back')
@click.option('--format', 'output_format', type=click.Choice(['cli', 'discord']), default='cli', help='Output format')
@click.pass_context
def detailed_topic(ctx, channel_name, n_topics, days_back, output_format):
    """Show detailed topic analysis (new system, unique template)."""
    try:
        with analysis_service(ctx.obj["db_path"]) as service:
            result = service.detailed_topic_analysis(channel_name, n_topics, days_back, output_format)
            click.echo(result)
    except Exception as e:
        click.echo(f"❌ Detailed topic analysis failed: {e}")
        if ctx.obj.get("verbose"):
            raise


@analyze.command('detailed-temporal')
@click.option('--channel', 'channel_name', help='Channel name to analyze (default: all channels)')
@click.option('--days-back', type=int, help='Number of days to look back')
@click.option('--granularity', type=click.Choice(['hourly', 'daily', 'weekly']), default='daily', help='Time granularity (default: daily)')
@click.option('--format', 'output_format', type=click.Choice(['cli', 'discord']), default='cli', help='Output format')
@click.pass_context
def detailed_temporal(ctx, channel_name, days_back, granularity, output_format):
    """Show detailed temporal analysis (new system, unique template)."""
    try:
        with analysis_service(ctx.obj["db_path"]) as service:
            result = service.detailed_temporal_analysis(channel_name, days_back, granularity, output_format)
            click.echo(result)
    except Exception as e:
        click.echo(f"❌ Detailed temporal analysis failed: {e}")
        if ctx.obj.get("verbose"):
            raise


@analyze.command('activity-trends')
@click.option('--channel', 'channel_name', help='Channel name to analyze (default: all channels)')
@click.option('--days-back', type=int, help='Number of days to look back')
@click.option('--format', 'output_format', type=click.Choice(['cli', 'discord']), default='cli', help='Output format')
@click.pass_context
def activity_trends(ctx, channel_name, days_back, output_format):
    """Show activity trends analysis with chart generation."""
    try:
        with analysis_service(ctx.obj["db_path"]) as service:
            result = service.activity_trends_analysis(channel_name, days_back, output_format)
            click.echo(result)
    except Exception as e:
        click.echo(f"❌ Activity trends analysis failed: {e}")
        if ctx.obj.get("verbose"):
            raise


@analyze.command('server-overview')
@click.option('--days-back', type=int, help='Number of days to look back (default: all time)')
@click.option('--format', 'output_format', type=click.Choice(['cli', 'discord']), default='discord', help='Output format')
@click.pass_context
def server_overview(ctx, days_back, output_format):
    """Show comprehensive server overview analysis."""
    try:
        with analysis_service(ctx.obj["db_path"]) as service:
            result = service.server_overview_analysis(days_back, output_format)
            click.echo(result)
    except Exception as e:
        click.echo(f"❌ Server overview analysis failed: {e}")
        if ctx.obj.get("verbose"):
            raise


if __name__ == "__main__":
    cli()
