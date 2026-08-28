"""DNSE broker configuration (Phase 0a, #66).

Extracted from ``broker.py`` so the config type is importable without the
1,500-line broker module, and so ``DNSEProvider`` can be generic in its
config (the structural "base classes are mutually incompatible" pyright
error came from the provider pinning ``DNSEConfig`` while ``BrokerPlugin``
was parameterized with ``DNSEBrokerConfig``). ``broker.py`` re-exports this
name — every existing ``from pynecore_dnse.broker import DNSEBrokerConfig``
keeps working.
"""
from dataclasses import dataclass

from .provider import DNSEConfig


@dataclass
class DNSEBrokerConfig(DNSEConfig):
    """:ivar account_no: DNSE trading account. Empty = resolve via ``/accounts``.
    :ivar trading_token: Bootstrap token; used only if the state file is absent.
    :ivar token_file: State file written by the OTP minter (the live source).
    :ivar stop_slippage_ticks: Fallback offset (in ticks) applied *through* a stop's
        trigger when pricing the LO it emits, used only when the strategy declares no
        ``strategy(slippage=)``. DNSE has no stop-market order, so a triggered stop
        posts a limit; pricing it at the trigger means a gap through never fills. The
        strategy's own ``slippage x 2`` takes precedence; this is the floor so the
        Pine default of 0 cannot silently recreate a never-filling stop.
    """

    account_no: str = ""
    trading_token: str = ""
    token_file: str = "workdir/state/dnse_trading_token.json"
    stop_slippage_ticks: int = 3
    #: Order-book poll period (seconds). One cycle = 2 requests (NORMAL + STOP
    #: books) and is how fast a fill becomes visible. DNSE allows 100,000
    #: Get-Orders req/hour PER API KEY: 0.5 s = 4 req/s = 14,400/h = 14% of the
    #: limit for ONE strategy. The budget is shared, so N concurrent strategies
    #: cost N x that — 10 at 0.5 s would EXCEED the hourly limit (raise this, or
    #: pool the poll, before running a fleet).
    order_poll_interval: float = 0.5
    #: Bar poll period (seconds), one request per cycle per (symbol, timeframe).
    #: Binding constraint here is the DAILY quota (100,000 Get-OHLC/day): one
    #: instance at 3 s over a ~6 h session = 7,200 (7%); ten instances = 72%.
    #: Keep >= 1 s always, and >= 3 s when several instances share one key.
    bar_poll_interval: float = 3.0
    #: Market-data feed mode (#37). "ohlc" (default) = today's closed-bar
    #: delivery, byte-identical path. "tick" = poll ``/trades/latest`` and
    #: synthesize the developing bar (is_closed=False) between closes; the
    #: venue's official closed bar stays authoritative at rollover when it
    #: arrives within ``tick_close_timeout``.
    feed_mode: str = "ohlc"
    #: Tick-mode poll period (seconds). /trades/latest has its OWN 10,000/h
    #: bucket (guide-ratelimits.md): 2 s = 1,800/h = 18% per strategy — the
    #: fleet ceiling is FIVE tick-mode strategies per key at this default
    #: (six breach the hour). At the 1 s floor it drops to TWO (#40's panel
    #: figure — three breach). Corrected 2026-08-26: the TWO had been carried
    #: over from the 1 s analysis when the default moved to 2 s.
    tick_poll_interval: float = 2.0
    #: Seconds to wait for the official closed bar at rollover before closing
    #: the SYNTHESIZED bar loudly instead: the session-final candle is
    #: withheld ~+903 s at the close (Live-L4-T03) and the emit-ordering
    #: guard (#37 panel) forbids forming N+1 before closed N — an unbounded
    #: wait would stall the feed through every session close.
    tick_close_timeout: float = 20.0

