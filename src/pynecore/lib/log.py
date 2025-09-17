from typing import Any
import os
import logging
from datetime import datetime
from pathlib import Path

from .string import format as _format
from .. import lib

# Try to import rich, but don't fail if not available
try:
    import rich
    import rich.logging
except ImportError:
    rich = None

__all__ = 'info', 'warning', 'error', 'logger'

if os.environ.get("PYNE_NO_COLOR_LOG", "") == "1":
    rich = None

# Custom RichHandler that uses Pine Script time and timezone
if rich:
    class PineRichHandler(rich.logging.RichHandler):
        """Custom RichHandler that formats time using syminfo.timezone"""

        # noinspection PyProtectedMember
        def render(self, *, record, traceback, message_renderable):
            """Override render to use Pine Script time and timezone"""
            from ..types import NA
            from datetime import UTC
            from rich.text import Text
            from rich.table import Table

            # Get the datetime in the correct timezone
            if lib._time:
                tz = lib.syminfo.timezone
                if not tz or isinstance(tz, NA):
                    # No timezone, use UTC
                    log_time = datetime.fromtimestamp(lib._time / 1000, UTC)
                else:
                    # Use specified timezone
                    log_time = lib._get_dt(lib._time, tz)
            else:
                # No Pine time, use current time
                log_time = datetime.fromtimestamp(record.created)

            # Format the time
            path = Path(record.pathname).name
            level = self.get_level_text(record)
            time_format = None if self.formatter is None else self.formatter.datefmt

            # Create custom log output with colored bar_index
            if hasattr(lib, 'bar_index') and lib.bar_index is not None:
                # Format the base time
                time_str = log_time.strftime(time_format or "[%Y-%m-%d %H:%M:%S]")

                # Create table for log output
                output = Table.grid(padding=(0, 1))
                output.expand = True

                # Add columns - no style for first two columns to allow Text styling
                output.add_column()  # Time column
                output.add_column()  # Bar index column
                output.add_column(style="log.level", width=8)  # Level column
                output.add_column(ratio=1, style="log.message", overflow="fold")  # Message column

                # Build row with styled Text objects
                row = [
                    Text(time_str, style="log.time"),
                    Text(f"bar: {lib.bar_index:6}", style="cyan"),
                    level,
                    message_renderable,
                ]

                output.add_row(*row)
                return output
            else:
                # Standard rendering without bar_index
                log_renderable = self._log_render(
                    self.console,
                    [message_renderable] if not traceback else [message_renderable, traceback],
                    log_time=log_time,
                    time_format=time_format,
                    level=level,
                    path=path,
                    line_no=record.lineno,
                    link_path=record.pathname if self.enable_link_path else None,
                )
                return log_renderable


# noinspection PyProtectedMember
class PineLogFormatter(logging.Formatter):
    """Custom formatter that mimics Pine Script log format"""

    def formatTime(self, record: logging.LogRecord, datefmt: str = None) -> str:
        """Format the time using syminfo.timezone"""
        from datetime import UTC
        from ..types import NA

        # Get the appropriate datetime
        if lib._time:
            tz = lib.syminfo.timezone
            if not tz or isinstance(tz, NA):
                # No timezone, use UTC
                dt = datetime.fromtimestamp(lib._time / 1000, UTC)
            else:
                # Use specified timezone
                dt = lib._get_dt(lib._time, tz)
        else:
            # No Pine time, use record's time
            dt = datetime.fromtimestamp(record.created)

        # Format the datetime
        if datefmt:
            time_str = dt.strftime(datefmt)
        else:
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]

        # Add bar_index to the time string
        if hasattr(lib, 'bar_index') and lib.bar_index is not None:
            time_str = f"{time_str} bar: {lib.bar_index:6}"

        return time_str

    def format(self, record: logging.LogRecord) -> Any:
        """Format log record in Pine style: [timestamp]: message"""
        if record.args:
            msg = _format(record.msg, *record.args)
        else:
            msg = str(record.msg)

        record.args = ()

        record.created = lib._time / 1000 if lib._time else record.created
        if rich:
            return msg

        record.msg = msg
        return super().format(record)


# Create logger
logger = logging.getLogger("pyne_core_logger")
# Remove existing handlers before adding new one
if logger.hasHandlers():
    logger.handlers.clear()

logger.setLevel(logging.INFO)
if rich:
    handler = PineRichHandler(  # noqa
        show_time=True,
        show_level=True,
        omit_repeated_times=False,
        markup=False,
        show_path=False,
    )
else:
    handler = logging.StreamHandler()
handler.setFormatter(PineLogFormatter(
    "%(asctime)s %(levelname)-7s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S%z]")
)
logger.addHandler(handler)


# noinspection PyPep8Naming,PyUnusedLocal
def info(formatString: str, *args: Any, **kwargs: Any) -> None:
    """
    Print an info message to the console.

    :param formatString: Message format string
    :param args: Arguments to format the message
    :param kwargs: Additional arguments (unused)
    """
    logger.info(formatString, *args)


# noinspection PyPep8Naming,PyUnusedLocal
def warning(formatString: str, *args: Any, **kwargs: Any) -> None:
    """
    Print a warning message to the console.

    :param formatString: Message format string
    :param args: Arguments to format the message
    :param kwargs: Additional arguments (unused)
    """
    logger.warning(formatString, *args)


# noinspection PyPep8Naming,PyUnusedLocal
def error(formatString: str, *args: Any, **kwargs: Any) -> None:
    """
    Print an error message to the console.

    :param formatString: Message format string
    :param args: Arguments to format the message
    :param kwargs: Additional arguments (unused)
    """
    logger.error(formatString, *args)
