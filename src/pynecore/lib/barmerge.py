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

- ``lookahead_on``: the security subprocess steps into the containing HTF bar
  with ``barstate.isconfirmed=False`` and OHLCV aggregated from chart-timeframe
  data, in historical and live mode alike. The TV idiom
  ``request.security(sym, tf, close[1], lookahead_on)`` returns the latest
  closed value exactly as it does on TV.

  A bare ``close`` reads the containing period as it has built up to the
  current chart bar. This is a **deliberate divergence**: TradingView hands
  back the period's FINAL close on every chart bar of the period, which is
  future data on all but the period's last bar. PyneCore never reproduces
  lookahead, so a historical backtest cannot see a value the bar could not
  have known.

Cross-symbol HTF ``lookahead_*`` in live mode is bounded by chart-symbol
aggregation: only same-symbol HTF contexts get the live HTF transport. Cross-symbol
HTF ``lookahead_off`` / ``lookahead_last_closed`` read closed bars from the
security's own data feed. Cross-symbol HTF ``lookahead_on`` returns ``na`` for
the current chart bar inside an open HTF period (the developing bar cannot be
aggregated from the wrong instrument); ``close[1]`` at the period boundary
still delivers the just-closed cross-symbol HTF close, preserving the TV
``lookahead_on + close[1]`` idiom.
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
