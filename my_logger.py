"""Log utility for script."""

import logging
import sys

import my_parser
from main import log


def initionalize_logger()->None:
    """Initialize logger for script."""
    log.setLevel(logging.INFO if not my_parser.args.verbose else logging.DEBUG)
    log_screen_handler = logging.StreamHandler(stream=sys.stdout)
    log.addHandler(log_screen_handler)
    log.propagate = False
