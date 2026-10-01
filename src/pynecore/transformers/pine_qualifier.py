"""
The Pine qualifier lattice and the rules that move values along it.

Pine gives every value a QUALIFIER besides its type: ``const`` (known at compile
time), ``simple`` (known before the first bar: inputs, the symbol, the chart) and
``series`` (may change from bar to bar). Most of the runtime never needs it, but
some built-ins measurably run a DIFFERENT machine depending on it, so the type
inference derives it alongside the type, in the same walk, and records what the
machines need.

MEASURED LAW (BINANCE:BTCUSDT 30m, probes ``qual``/``qual2``/``lrna``/``lrnb``,
every probed value matched): ``ta.highest``/``ta.lowest``/``ta.highestbars``/
``ta.lowestbars`` and ``ta.linreg`` keep a ``length + 1`` ring when ``length`` is
const, input or simple, and a forward-filled per-bar history when it is series --
decided by the qualifier, not the value: a series length that is 8 on every bar
still runs the history machine. The variable rules, measured on the same probes:

- a literal, an input, ``syminfo.*``/``timeframe.*``/``chart.*`` and a variable never
  reassigned, or reassigned unconditionally to a non-series value, are NOT series;
- a variable reassigned under a series condition, a ``var`` variable reassigned
  anywhere, anything reading ``bar_index``, a price, a history subscript or a
  ``ta.*`` result IS series;
- a function parameter takes the qualifier of the argument: TradingView
  specializes the function per call.

This module holds the vocabulary and the rules; the walking is done by
``pine_type_infer``.
"""
import ast
from collections.abc import Iterable
from typing import Final

__all__ = [
    'CONST', 'SIMPLE', 'SERIES', 'NOT_READ', 'lib_value_qualifier', 'lib_call_qualifier',
    'builtin_call_qualifier', 'WINDOW_CALLS', 'SERIES_LENGTH_KEYWORD', 'window_length',
    'SERIES_LEN_ATTR', 'SERIES_LENS_ATTR', 'get_series_len', 'set_series_len',
    'get_series_lens', 'set_series_lens',
]

CONST: Final = 0
SIMPLE: Final = 1
SERIES: Final = 2

#: A binding nothing has read yet -- above every qualifier, so ``min`` keeps the
#: lowest one a read has seen
NOT_READ: Final = 3

#: Namespaces whose values are known before the first bar
_SIMPLE_NAMESPACES: Final = frozenset({'syminfo', 'timeframe', 'chart'})

#: Namespaces whose values change from bar to bar
_SERIES_NAMESPACES: Final = frozenset({
    'barstate', 'ta', 'strategy', 'strategy.opentrades', 'strategy.closedtrades',
    'dividends', 'earnings',
})

#: Built-ins whose result is ``const`` when every argument is (MEASURED on
#: TradingView by feeding each call into a const-only slot)
_CONST_PRESERVING: Final = frozenset({
    'color.new', 'color.rgb',
    'math.abs', 'math.acos', 'math.asin', 'math.atan', 'math.ceil', 'math.cos', 'math.exp',
    'math.floor', 'math.log', 'math.log10', 'math.max', 'math.min', 'math.pow', 'math.round',
    'math.sign', 'math.sin', 'math.sqrt', 'math.tan',
    'str.contains', 'str.endswith', 'str.length', 'str.lower', 'str.pos', 'str.repeat',
    'str.replace', 'str.startswith', 'str.substring', 'str.tonumber', 'str.trim', 'str.upper',
})

#: Built-ins whose result is ``simple`` when no argument is series: they read the
#: chart, the symbol or the clock of the script run, or format a value
_SIMPLE_PRESERVING: Final = frozenset({
    'na', 'nz', 'int', 'float', 'bool', 'string', 'color',
    'math.avg', 'math.round_to_mintick', 'str.tostring', 'str.format', 'str.split',
    'timestamp', 'timeframe.in_seconds', 'timeframe.from_seconds',
})

#: Namespaces whose functions are all ``simple`` when no argument is series
_SIMPLE_CALL_NAMESPACES: Final = frozenset({'ticker', 'syminfo'})

#: Python builtins a Pyne script calls that only convert or combine their arguments
_PLAIN_BUILTINS: Final = frozenset({
    'int', 'float', 'bool', 'str', 'abs', 'min', 'max', 'round', 'range',
})

#: The built-ins whose machine depends on the qualifier of ``length``. A
#: ``length``-only form (``ta.highest(length)``) has the length first.
WINDOW_CALLS: Final = frozenset({
    'ta.highest', 'ta.lowest', 'ta.highestbars', 'ta.lowestbars', 'ta.linreg',
})

#: The keyword the window built-ins take their machine from
SERIES_LENGTH_KEYWORD: Final = '_series_length'

#: Attribute a window call carries its verdict under, where every context agrees
SERIES_LEN_ATTR: Final = '_pine_series_len'

#: Attribute a window call carries its PER-CONTEXT verdicts under, present only
#: where the contexts a shared body was analysed in disagree
SERIES_LENS_ATTR: Final = '_pine_series_lens'


def lib_value_qualifier(name: str, ty: str) -> int:
    """
    The qualifier of a lib value read.

    :param name: The registry key, e.g. ``'close'`` or ``'syminfo.mintick'``
    :param ty: Its registry type
    :return: The qualifier
    """
    namespace, _, member = name.rpartition('.')
    if not namespace:
        # The bare built-in variables: prices, time and the bar clock
        return SERIES
    head = namespace.split('.', 1)[0]
    if head in _SIMPLE_NAMESPACES:
        return SIMPLE
    if namespace == 'session':
        return SERIES if member.startswith('is') else CONST
    if namespace in _SERIES_NAMESPACES:
        # Their enum members (``strategy.long``, ``dividends.gross``) are constants
        if ty.startswith('o:lib#') and namespace != 'ta':
            return CONST
        if name == 'strategy.account_currency':
            return SIMPLE
        return SERIES
    if namespace == 'dayofweek' and member == 'dayofweek':
        return SERIES
    return CONST


def lib_call_qualifier(name: str, arguments: Iterable[int], argc: int | None) -> int:
    """
    The qualifier of a lib call's result.

    :param name: The registry key of the callee
    :param arguments: The qualifier of every argument passed
    :param argc: The argument count, None when an unpacking hides it
    :return: The qualifier
    """
    strongest = max(arguments, default=CONST)
    if name.startswith('input.') or name == 'input':
        return max(strongest, SIMPLE)
    if name == 'timestamp' and argc == 1:
        # Only the single date-string form is const
        return strongest
    if name in _CONST_PRESERVING:
        return strongest
    if name in _SIMPLE_PRESERVING or name.split('.', 1)[0] in _SIMPLE_CALL_NAMESPACES:
        return max(strongest, SIMPLE)
    return SERIES


def builtin_call_qualifier(name: str, arguments: Iterable[int]) -> int | None:
    """
    The qualifier of a Python builtin call, when the callee is one.

    :param name: The callee as written
    :param arguments: The qualifier of every argument passed
    :return: The qualifier, or None when the callee is no such builtin
    """
    if name not in _PLAIN_BUILTINS:
        return None
    return max(arguments, default=CONST)


def window_length(name: str, node: ast.Call) -> ast.expr | None:
    """
    The ``length`` argument of a window call.

    :param name: The registry key of the callee, one of ``WINDOW_CALLS``
    :param node: The call node
    :return: The argument expression, or None when the call does not spell one
    """
    for keyword in node.keywords:
        if keyword.arg == 'length':
            return keyword.value
    positional = [arg for arg in node.args if not isinstance(arg, ast.Starred)]
    if len(positional) != len(node.args):
        return None
    if name != 'ta.linreg' and len(positional) == 1 \
            and not any(keyword.arg == 'source' for keyword in node.keywords):
        return positional[0]  # the ``ta.highest(length)`` form
    return positional[1] if len(positional) >= 2 else None


def get_series_len(node: ast.Call) -> bool | None:
    """
    Read a window call's verdict.

    :param node: The call node
    :return: True for a series length, False for a ring one, None when unstamped
    """
    return getattr(node, SERIES_LEN_ATTR, None)


def set_series_len(node: ast.Call, value: bool | None) -> ast.Call:
    """
    Stamp a window call's verdict, ``None`` included.

    :param node: The call node
    :param value: The verdict every context agrees on, or None
    :return: ``node``
    """
    setattr(node, SERIES_LEN_ATTR, value)
    return node


def get_series_lens(node: ast.Call) -> dict[int, bool] | None:
    """
    Read the per-context verdicts of a window call the contexts disagree on.

    :param node: The call node
    :return: context id -> verdict, or None when one verdict holds for all
    """
    return getattr(node, SERIES_LENS_ATTR, None)


def set_series_lens(node: ast.Call, value: dict[int, bool] | None) -> ast.Call:
    """
    Stamp the per-context verdicts of a window call, ``None`` included.

    :param node: The call node
    :param value: context id -> verdict, or None
    :return: ``node``
    """
    setattr(node, SERIES_LENS_ATTR, value)
    return node
