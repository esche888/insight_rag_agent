# logging_config.py
import logging
import sys
from typing import Optional

FORMAT = '%(asctime)s | %(levelname)-7s [%(module)+14s:%(lineno)-5d] %(funcName)-20s | %(message)s'
DATEFMT = "%H:%M:%S"
# DATEFMT = "%Y-%m-%d %H:%M:%S"

def configure_logging(level: int = logging.INFO, logfile: Optional[str] = None):
    """ Configure the root logger with a console handler and optional file handler.
        Calling this multiple times is safe because we check for existing handlers. """

    root = logging.getLogger()
    root.setLevel(level)
    if not root.handlers:
        formatter = logging.Formatter(FORMAT, datefmt=DATEFMT)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

        if logfile:
            file_handler = logging.FileHandler(logfile)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)

    # return root for convenience
    return root