from ..core.module_property import module_property

__all__ = [
    'isfirst',
    'islast',
    'isconfirmed',
    'ishistory',
    'islastconfirmedhistory',
    'isnew',
    'isrealtime'
]

# Dynamic state variables set by ScriptRunner during live mode
_is_live_phase = False
_is_confirmed = True
_is_new_bar = False
_is_last_confirmed_history = False

isfirst = True
""" Returns true if current bar is first bar in barset, false otherwise."""

islast = False
""" Returns true if current bar is the last bar in barset, false otherwise. """


@module_property
def isconfirmed() -> bool:
    """
    Returns true if the script is calculating the last (closing) update of the current bar

    :return: True if the script is calculating the last (closing) update of the current bar
    """
    return _is_confirmed


@module_property
def ishistory() -> bool:
    """
    Returns true if script is calculating on historical bars, false otherwise.

    :return: True if script is calculating on historical bars, false otherwise
    """
    return not _is_live_phase


@module_property
def islastconfirmedhistory() -> bool:
    """
    Returns true if script is executing on the dataset's last bar when market is closed, or script
    is executing on the bar immediately preceding the real-time bar, if market is open.

    :return: True if script is executing on the dataset's last bar when market is closed, or script
             is executing on the bar immediately preceding the real-time bar, if market is open
    """
    return _is_last_confirmed_history


@module_property
def isnew() -> bool:
    """
    Returns true if script is currently calculating on new bar, false otherwise.

    :return: True if script is currently calculating on new bar, false otherwise
    """
    return _is_new_bar


@module_property
def isrealtime() -> bool:
    """
    Returns true if script is calculating on real-time bars, false otherwise.

    :return: True if script is calculating on real-time bars, false otherwise
    """
    return _is_live_phase
