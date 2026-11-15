"""
Argument type check shared by ``str.format`` and ``log.*``.
"""
from typing import Any

from ..errorcodes import WrongOperandTypeError
from ..types.na import NA
from ..utils.sequence_view import SequenceView

# TradingView accepts int, float, bool and string format arguments and arrays of them; a
# matrix, map, color, drawing or user type is a compile error there (measured, CE10122).
_FORMAT_SCALARS = (str, int, float, NA)


def check_format_args(function: str, args: tuple[Any, ...]) -> None:
    """
    Reject a ``str.format`` / ``log.*`` argument that cannot be formatted.

    Only int, float, bool and string values and arrays of them can be formatted. An empty
    array always passes: its element type is not known at run time.

    :param function: The Pine name of the formatting call, for the error message
    :param args: The format arguments, ``args[0]`` being ``{0}``
    :raises WrongOperandTypeError: If an argument or an array element has any other type
    """
    for position, value in enumerate(args):
        if value is None or isinstance(value, _FORMAT_SCALARS):
            continue
        if isinstance(value, (list, SequenceView)):
            for item in value:
                if not (item is None or isinstance(item, _FORMAT_SCALARS)):
                    raise WrongOperandTypeError(
                        operation=function,
                        operand=f"an array of {type(item).__name__} as argument {{{position}}}")
            continue
        raise WrongOperandTypeError(
            operation=function, operand=f"a {type(value).__name__} as argument {{{position}}}")
