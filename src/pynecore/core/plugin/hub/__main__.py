"""
Run a provider hub: ``python -m pynecore.core.plugin.hub [--config-dir DIR]``.

The token comes from the environment (:data:`~pynecore.core.plugin.hub.TOKEN_ENV`),
so it never shows in the process list. Once the hub listens it prints one JSON line,
``{"host": ..., "port": ...}``, on its standard output. It runs until its standard
input closes: a front end that starts it keeps the pipe open and the hub ends with
the front end, however that one ended.
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import threading
from pathlib import Path

from . import HUB_ENV, TOKEN_ENV
from .server import ProviderHub


async def _run(config_dir: Path | None, token: str) -> None:
    hub = ProviderHub(token, config_dir=config_dir)
    port = await hub.start()
    sys.stdout.write(json.dumps({"host": "127.0.0.1", "port": port}) + "\n")
    sys.stdout.flush()

    loop = asyncio.get_running_loop()
    front_end_gone = asyncio.Event()

    def watch_stdin() -> None:
        try:
            while sys.stdin.buffer.read(4096):
                pass
        except (OSError, ValueError):
            pass
        loop.call_soon_threadsafe(front_end_gone.set)

    threading.Thread(target=watch_stdin, name="provider-hub-stdin", daemon=True).start()
    try:
        await front_end_gone.wait()
    finally:
        await hub.close()


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m pynecore.core.plugin.hub",
                                     description="Serve data providers to other processes.")
    parser.add_argument("--config-dir", type=Path, default=None,
                        help="directory of the plugin configs and the symbol map")
    args = parser.parse_args()
    token = os.environ.get(TOKEN_ENV, "")
    if not token:
        parser.error(f"the hub's token must be set in {TOKEN_ENV}")
    # The hub holds the real providers: it must never look for a hub itself
    os.environ.pop(HUB_ENV, None)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    try:
        asyncio.run(_run(args.config_dir, token))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
