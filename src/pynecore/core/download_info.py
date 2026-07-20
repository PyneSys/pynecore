"""
Persist the originating provider string of a download in the syminfo TOML.

``pyne data download`` appends a ``[download]`` section to the ``.toml`` saved
next to the ``.ohlcv`` file::

    [download]
    provider = "ccxt:BYBIT:BTC/USDT:USDT@1D"

The provider string is the canonical CLI form, so the file can be re-downloaded
or resumed without reconstructing it from ``SymInfo`` fields (``prefix`` and the
flattened filename are both lossy). ``SymInfo.load_toml`` reads only the
``[symbol]`` section, so older pynecore versions ignore this section;
``SymInfo.save_toml`` preserves it verbatim when rewriting the file.
"""
from __future__ import annotations

import re
from pathlib import Path

__all__ = ['read_download_provider', 'write_download_provider', 'extract_download_section']

# [download] up to (not including) the next section header or EOF
_SECTION_RE = re.compile(r'(?ms)^\[download\][^\n]*\n.*?(?=^\[|\Z)')


def read_download_provider(toml_path: Path) -> str | None:
    """
    Read the persisted provider string from a syminfo TOML.

    :param toml_path: Path to the ``.toml`` next to the ``.ohlcv`` file.
    :return: The provider string, or None if the file or section is missing
        or unparsable.
    """
    import tomllib
    try:
        with open(toml_path, 'rb') as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    value = data.get('download', {}).get('provider')
    return value if isinstance(value, str) and value else None


def extract_download_section(text: str) -> str | None:
    """
    Extract the ``[download]`` section verbatim from TOML text.

    :param text: Full TOML file content.
    :return: The section text (without trailing whitespace), or None.
    """
    m = _SECTION_RE.search(text)
    return m.group(0).rstrip() if m else None


def write_download_provider(toml_path: Path, provider_string: str) -> None:
    """
    Write (or replace) the ``[download]`` section in a syminfo TOML.

    Missing files are left untouched: the section rides along with the symbol
    info and makes no sense on its own.

    :param toml_path: Path to the ``.toml`` next to the ``.ohlcv`` file.
    :param provider_string: Canonical provider string to persist.
    """
    if not toml_path.exists():
        return
    text = toml_path.read_text(encoding='utf-8')
    text = _SECTION_RE.sub('', text).rstrip()
    section = f'[download]\nprovider = "{provider_string}"'
    toml_path.write_text(f'{text}\n\n{section}\n', encoding='utf-8')
