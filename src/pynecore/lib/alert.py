"""
Alert

The module is both a function (``alert(...)``) and a namespace (``alert.freq_once_per_bar``);
call sites are routed to :func:`alert` by the module property AST transformer.
"""
from ..types.alert import AlertEnum


#
# Constants
#

freq_all = AlertEnum('all')
freq_once_per_bar = AlertEnum('once_per_bar')
freq_once_per_bar_close = AlertEnum('once_per_bar_close')


#
# Module function
#

def alert(
        message: str,
        freq: AlertEnum = freq_once_per_bar
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
