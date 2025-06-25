"""
CLI entry point for Discord analytics.
"""

import sys
from pathlib import Path

import click

# Handle both relative and absolute imports
try:
    from .commands import cli
except ImportError:
    # If relative import fails, try absolute import
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from pepino.cli.commands import cli


def main():
    """Main CLI entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
