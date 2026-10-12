"""
Argument type check shared by ``str.format`` and ``log.*``.
"""
from typing import Any

from ..types.na import NA
from ..utils.sequence_view import SequenceView

# TradingView accepts int, float, bool and string format arguments and arrays of them; a
# matrix, map, color, drawing or user type is a compile error there (measured, CE10122).
_FORMAT_SCALARS = (str, int, float, NA)


def check_format_args(args: tuple[Any, ...]) -> None:
    """
    Reject a ``str.format`` / ``log.*`` argument that cannot be formatted.

    Only int, float, bool and string values and arrays of them can be formatted. An empty
    array always passes: its element type is not known at run time.

    :param args: The format arguments, ``args[0]`` being ``{0}``
    :raises TypeError: If an argument or an array element has any other type
    """
    for position, value in enumerate(args):
        if value is None or isinstance(value, _FORMAT_SCALARS):
            continue
        if isinstance(value, (list, SequenceView)):
            for item in value:
                if not (item is None or isinstance(item, _FORMAT_SCALARS)):
                    raise TypeError(
                        f"Format argument {{{position}}} is an array of "
                        f"{type(item).__name__}; only int, float, bool and string values "
                        f"and arrays of them can be formatted")
            continue
        raise TypeError(
            f"Format argument {{{position}}} is a {type(value).__name__}; only int, float, "
            f"bool and string values and arrays of them can be formatted")
