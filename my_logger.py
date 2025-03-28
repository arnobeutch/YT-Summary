"""Log utility for script."""

import argparse
import logging
import sys

log = logging.getLogger(__name__)

def initialize_logger(args: argparse.Namespace) -> None:
    """Initialize logger for script."""
    log.setLevel(logging.INFO if not args.verbose else logging.DEBUG)
    log_screen_handler = logging.StreamHandler(stream=sys.stdout)
    log.addHandler(log_screen_handler)
    log.propagate = False
