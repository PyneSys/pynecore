"""Write workdir/config/plugins/binance_broker.toml from .env — file to file,
credentials are never printed.

Reads ``Binance_Testnet_Api_Key`` / ``Binance_Testnet_Secret`` (testnet,
default) or ``Binance_Api_Key`` / ``Binance_Secret`` with ``--mainnet``
(also sets ``allow_mainnet = true`` — only do this deliberately).

Usage: ``.venv/bin/python plugins/binance/tools/write_config_from_env.py``
"""
from __future__ import annotations

import sys
from pathlib import Path

ENV_PATH = Path('.env')
CONFIG_PATH = Path('workdir/config/plugins/binance_broker.toml')


def read_env() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, _, value = line.partition('=')
            values[key.strip()] = value.strip()
    return values


def main() -> None:
    mainnet = '--mainnet' in sys.argv
    env = read_env()
    prefix = 'Binance_' if mainnet else 'Binance_Testnet_'
    api_key = env.get(f'{prefix}Api_Key', '')
    secret = env.get(f'{prefix}Secret', '')
    if not api_key or not secret:
        print(f"MISSING: {prefix}Api_Key / {prefix}Secret not found in .env")
        sys.exit(1)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        "# Binance broker plugin config — written by write_config_from_env.py.\n"
        "# workdir/ is gitignored; keys stay out of the repo.\n"
        f'apiKey = "{api_key}"\n'
        f'secret = "{secret}"\n'
        f"sandbox = {'false' if mainnet else 'true'}\n"
        f"allow_mainnet = {'true' if mainnet else 'false'}\n"
        "stop_slippage_ticks = 10\n"
        "poll_interval = 2.0\n"
    )
    print(f"wrote {CONFIG_PATH} (mode={'MAINNET' if mainnet else 'testnet'}, "
          f"apiKey length={len(api_key)})")


if __name__ == '__main__':
    main()
