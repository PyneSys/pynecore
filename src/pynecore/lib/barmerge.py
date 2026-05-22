"""
``barmerge`` namespace — gap and lookahead modes for ``request.security()``.

Gap modes
---------

- ``gaps_off``: forward-fill missing values (default)
- ``gaps_on``: emit ``na`` between security periods

Lookahead modes
---------------

PyneCore exposes three lookahead-mode constants. Pine v6's ``lookahead_off``
and ``lookahead_on`` are kept for source-level compatibility with TradingView
scripts; ``lookahead_last_closed`` is a PyneSys-native alternative.

- ``lookahead_off`` (default): TV-faithful closed-bar behavior — the security
  context advances only to the bar that has CLOSED at or before the chart
  bar's time. In historical mode this matches TradingView exactly. In live
  mode every HTF period close is shipped to the subprocess via the chart-side
  ``HTFAggregator`` (the static ``.ohlcv`` file cannot grow at runtime); no
  developing-bar exposure. Note: PyneCore's ``lookahead_off`` is intentionally
  repaint-free even in live mode — it does not mirror TV's live developing
  exposure for ``lookahead_off + close[0]``.

- ``lookahead_last_closed`` (PyneSys-native): always returns the most recently
  closed security bar. In historical mode it is functionally equivalent to
  ``lookahead_off``; in live mode it uses the same closed-bar transport as
  ``lookahead_off`` and stays repaint-free (no in-progress bar). Preferred
  when you want explicit "last closed" semantics without depending on the TV
  ``close[1]`` idiom.

- ``lookahead_on``: TV-compatible. In live mode the security subprocess steps
  into the containing HTF bar with ``barstate.isconfirmed=False`` and OHLCV
  aggregated from chart-timeframe data — matching TradingView's developing-bar
  semantics, so the TV idiom ``request.security(..., lookahead_on)[1]`` returns
  the latest closed value as it does on TV. In historical mode it falls back
  to closed-only semantics (equivalent to ``lookahead_off``), so historical
  backtests never expose a developing close.

Cross-symbol HTF ``lookahead_*`` in live mode is bounded by chart-symbol
aggregation: only same-symbol HTF contexts get the live HTF transport.
Cross-symbol HTF falls back to the static ``.ohlcv`` file (adequate for
historical/backtest, inert in live). Cross-symbol HTF ``lookahead_on``
specifically raises ``NotImplementedError`` at startup because the developing
phase would deliver wrong-instrument OHLCV.
"""
from ..types.barmerge import BarMerge

#
# Constants
#

gaps_off = BarMerge()
gaps_on = BarMerge()
lookahead_off = BarMerge()
lookahead_on = BarMerge()
lookahead_last_closed = BarMerge()
