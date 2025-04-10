"""Log utility for script."""

import logging
import sys

my_logger = logging.getLogger(__name__)

def initialize_logger() -> None:
    """Initialize logger for script."""
    my_logger.setLevel(logging.INFO)
    log_screen_handler = logging.StreamHandler(stream=sys.stdout)
    my_logger.addHandler(log_screen_handler)
    my_logger.propagate = False
