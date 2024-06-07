import argparse

from dotenv import load_dotenv

from . import cli_commands
from .logging_config import logger

if load_dotenv():
    logger.info("Parsed .env file successfully")
else:
    logger.info("Did not find .env file")


def main() -> None:

    parser = argparse.ArgumentParser(description="RAGu-late CLI tool.")

    # Subparsers for the main commands
    subparsers = parser.add_subparsers(title="commands", dest="command", required=True)

    cli_commands.setup_download(subparsers=subparsers)
    cli_commands.setup_ingest(subparsers=subparsers)
    cli_commands.setup_query(subparsers=subparsers)
    cli_commands.setup_compare(subparsers=subparsers)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the appropriate function based on the command
    args.func(args)


if __name__ == "__main__":
    main()
