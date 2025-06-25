"""Entry point for running pepino as a module"""

import sys


def main():
    """Main entry point - routes to CLI."""
    from pepino.cli.commands import cli

    cli()


if __name__ == "__main__":
    main()
