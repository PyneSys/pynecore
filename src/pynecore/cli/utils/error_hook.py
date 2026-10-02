import sys
import traceback
from pathlib import Path


def setup_global_error_logging(log_path: Path):
    """
    Sets up a global error logging mechanism that writes uncaught exceptions to a logfile.

    :param log_path: Path to the log file where uncaught exceptions will be written.
    """
    # Creating the log file directory
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Remove last error file if exists
    log_path.unlink(missing_ok=True)

    # Save the original excepthook
    original_excepthook = sys.excepthook

    # Exception handler – saves the raw stack trace to a file
    def log_and_reraise(exc_type, exc_value, tb):
        tb_str = ''.join(traceback.format_exception(exc_type, exc_value, tb))
        log_path.write_text(tb_str, encoding="utf-8")
        original_excepthook(exc_type, exc_value, tb)

    # Set the new excepthook
    sys.excepthook = log_and_reraise
