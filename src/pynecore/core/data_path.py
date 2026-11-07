"""
Data file naming

A data file is named after its symbol and timeframe, and the symbol may itself
contain a dot (a perpetual like ``BTCUSD.P``): in ``bybit_BTCUSD.P_1D`` the
``.P_1D`` tail is part of the name, not an extension. :class:`pathlib.Path`
treats everything after the last dot as the suffix, so ``with_suffix`` would
cut such a name down to ``bybit_BTCUSD``. These helpers only treat a known data
format extension as the extension and append anything else by name.
"""
from pathlib import Path

__all__ = ['DATA_EXTENSIONS', 'data_extension', 'with_data_extension']

DATA_EXTENSIONS: tuple[str, ...] = ('.ohlcv', '.csv', '.txt', '.json')
"""The data file extensions PyneCore reads, in bare-name lookup order."""


def data_extension(path: str | Path) -> str:
    """
    Return the known data extension of a path.

    :param path: A data file path or a bare data name.
    :return: The extension (e.g. ``'.ohlcv'``) as written in the path, or ``''``
             when the path does not end in a known data extension.
    """
    suffix = Path(path).suffix
    return suffix if suffix.lower() in DATA_EXTENSIONS else ''


def with_data_extension(path: str | Path, extension: str) -> Path:
    """
    Return the path with its data extension replaced, or the extension appended.

    The extension is replaced only if it is a known data extension. Any other
    dotted tail belongs to the name and is kept:
    ``bybit_BTCUSD.P_1D`` -> ``bybit_BTCUSD.P_1D.ohlcv``, while
    ``bybit_BTCUSD.P_1D.csv`` -> ``bybit_BTCUSD.P_1D.ohlcv``.

    :param path: A data file path or a bare data name.
    :param extension: The new extension including the leading dot
                      (e.g. ``'.ohlcv'``, ``'.toml'``, ``'.extra.csv'``).
    :return: The resulting path.
    """
    p = Path(path)
    name = p.name
    current = data_extension(p)
    if current:
        name = name[:-len(current)]
    return p.with_name(name + extension)
