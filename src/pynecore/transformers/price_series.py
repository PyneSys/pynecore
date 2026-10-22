"""
The builtin price series, by the names a script reads them under.

A leaf module: the passes that recognize these names (the library-series pass, the
input pass) import it without importing each other.
"""

__all__ = ['BUILTIN_PRICE_SERIES']

#: Builtin price series accepted by generic ``input()`` as source names
BUILTIN_PRICE_SERIES = frozenset({
    'open', 'high', 'low', 'close', 'volume',
    'bid', 'ask',
    'hl2', 'hlc3', 'ohlc4', 'hlcc4',
})
