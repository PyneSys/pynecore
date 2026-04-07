"""
Alert

This is a callable module, so the module itself is both a function and a namespace
"""
from __future__ import annotations
from ..core.callable_module import CallableModule

from ..types.alert import AlertEnum


#
# Module object
#

class AlertModule(CallableModule):
    #
    # Constants
    #

    freq_all = AlertEnum('all')
    freq_once_per_bar = AlertEnum('once_per_bar')
    freq_once_per_bar_close = AlertEnum('once_per_bar_close')


#
# Callable module function
#

def alert(
        message: str,
        freq: AlertEnum = AlertModule.freq_once_per_bar  # noqa
) -> None:
    """
    Display alert message. Uses rich formatting if available, falls back to print.

    :param message: Alert message to display
    :param freq: Alert frequency (currently ignored)
    """
    try:
        # Try to use typer for nice colored output
        import typer
        typer.secho(f"🚨 ALERT: {message}", fg=typer.colors.BRIGHT_YELLOW, bold=True)
    except ImportError:
        # Fallback to simple print
        print(f"🚨 ALERT: {message}")


#
# Module initialization
#

AlertModule(__name__)
