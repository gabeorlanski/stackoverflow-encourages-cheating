"""
Logging handlers
"""

import logging
from logging.handlers import RotatingFileHandler
import tqdm

__all__ = [
    "TQDMLoggingHandler",
    "CompactFileHandler"
]


class TQDMLoggingHandler(logging.Handler):
    """
    Console Handler that works with tqdm.

    All credit goes to: https://stackoverflow.com/questions/38543506/change-logging-print
    -function-to-tqdm-write-so-logging-doesnt-interfere-wit
    """

    def __init__(self, level=logging.NOTSET, fmt: logging.Formatter = None):
        super().__init__(level)
        self.setFormatter(
            fmt if fmt is not None else logging.Formatter('%(levelname)8s: %(message)s'))

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


class CompactFileHandler(RotatingFileHandler):
    """
    Wrapper to reduce repetition
    """
    def __init__(self, file: str, level: int = logging.NOTSET, fmt: logging.Formatter = None):
        """
        Initialize the handler
        Args:
            file: file to write to
            level: The logging level
            fmt: Logging formatter
        """
        # Clear file
        with open(file, 'w', encoding='utf-8') as f:
            f.write('')
        super(CompactFileHandler, self).__init__(file, mode='w', encoding='utf-8',
                                                 maxBytes=10485760, backupCount=3)
        self.setLevel(level)
        self.setFormatter(
            fmt if fmt is not None else logging.Formatter('%(levelname)8s: %(message)s'))
