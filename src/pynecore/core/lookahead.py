"""
Opt-in reproduction of TradingView's future-leaks, for measurement only.

PyneCore never reproduces lookahead. TradingView leaks the future through more
than one channel, and PyneCore closes every one of them:

- ``request.*`` with ``barmerge.lookahead_on``: on an open higher-timeframe
  period TradingView serves the period's FINAL values on every chart bar of that
  period; PyneCore serves the period as it has built so far (see
  :class:`pynecore.core.security.Lookahead`).
- A body running mid-bar — a ``calc_on_order_fills`` re-execution, or a
  ``calc_on_every_tick`` history pass — reads the CLOSED bar's extremes and close
  on TradingView, values the bar could not yet have had; PyneCore shows the bar
  as it had built at that point of the emulator's path (see
  ``_set_path_bar`` in :mod:`pynecore.core.script_runner`).

``PYNE_ALLOW_LOOKAHEAD=1`` restores TradingView's behaviour on every one of those
channels, so a script can be measured BOTH ways: identical output proves it does
not depend on data its bars could not have had, and different output proves it
does. That difference is the only reliable repaint test — static inspection
cannot tell a legitimate ``close[1]`` idiom from a future read, nor can it tell
whether a re-executing body ever looks at a price at all.

It is a MEASUREMENT switch, never a trading one. Leave it off for anything that
places orders: with it on, a backtest reports profits the market never offered.
"""
import os

__all__ = ['ALLOW_LOOKAHEAD']

ALLOW_LOOKAHEAD = os.environ.get("PYNE_ALLOW_LOOKAHEAD", "").strip().lower() in (
    "1", "true", "yes", "on",
)
