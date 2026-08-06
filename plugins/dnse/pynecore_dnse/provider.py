"""DNSE data provider plugin for PyneCore (v2).

Historical OHLCV over ``GET /price/ohlc`` + symbol metadata + the two-level
symbol resolution (``VN30F1M`` symbolType alias → dated KRX contract). Built on
the vendored openapi-sdk via the plugin's :class:`DNSEClient` wrapper.

    type=DERIVATIVE symbol=VN30F1M resolution=15  ->  17 bars/trading day
    (09:00-11:30 + 13:00-14:45 ICT), payload {t,o,h,l,c,v,nextTime}

The continuous alias ``VN30F1M`` is accepted directly by ``/price/ohlc``; dated
contract codes are what orders/streams want (resolved via ``/market/instruments``).
Live streaming + orders live in :class:`DNSEBroker` (this class is data-only).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Callable

from pynecore.core.plugin import override
from pynecore.core.plugin.provider import ProviderPlugin
from pynecore.core.plugin.live_provider import LiveProviderConfig
from pynecore.core.syminfo import SymInfo, SymInfoInterval, SymInfoSession
from pynecore.types.ohlcv import OHLCV

from .client import DNSEClient

#: TradingView timeframe -> DNSE resolution. DNSE offers 1 3 5 15 30 1H 1D 1W,
#: so the mapping is NOT identity above 30 minutes.
_TV_TO_DNSE = {"1": "1", "3": "3", "5": "5", "15": "15", "30": "30",
               "60": "1H", "1D": "1D", "D": "1D", "1W": "1W", "W": "1W"}
_DNSE_TO_TV = {"1": "1", "3": "3", "5": "5", "15": "15", "30": "30",
               "1H": "60", "1D": "1D", "1W": "1W"}

#: VN30F1M trading sessions, Asia/Ho_Chi_Minh. Two blocks per weekday around
#: the lunch break — the 17-bars-per-day figure confirms these boundaries.
_MORNING = (time(9, 0), time(11, 30))
_AFTERNOON = (time(13, 0), time(14, 45))


@dataclass
class DNSEConfig(LiveProviderConfig):
    """Credentials and endpoint for the DNSE OpenAPI.

    :ivar api_key: DNSE API key (also sent as the ``x-api-key`` header).
    :ivar api_secret: HMAC-SHA256 signing secret.
    :ivar base_url: REST base URL.

    Inherits ``symbol_map`` from :class:`LiveProviderConfig`.
    """

    api_key: str = ""
    api_secret: str = ""
    base_url: str = "https://openapi.dnse.com.vn"
    #: WebSocket origin. Kept for the live-vs-test endpoint banner (and any future
    #: market-data stream); v2's core flow is REST-only.
    ws_url: str = "wss://ws-openapi.dnse.com.vn"
    #: IGNORED: the ``version`` header is pinned to 2026-07-23 inside the
    #: :class:`DNSEClient` wrapper (a floating date silently breaks conditional
    #: cancels). Retained only so an existing config file with this key still loads.
    api_version: str = "2026-07-23"


class DNSEProvider(ProviderPlugin[DNSEConfig]):
    """DNSE market-data provider (Vietnam stocks, derivatives and indices)."""

    plugin_name = "DNSE"
    Config = DNSEConfig

    def __init__(self, *, symbol=None, timeframe=None, ohlcv_dir=None, config=None):
        super().__init__(symbol=symbol, timeframe=timeframe,
                         ohlcv_dir=ohlcv_dir, config=config)
        self._client: DNSEClient | None = None

    # --- helpers ---

    @property
    def client(self) -> DNSEClient:
        if self._client is None:
            assert self.config is not None, "DNSEProvider requires config"
            if not self.config.api_key or not self.config.api_secret:
                raise ValueError(
                    "DNSE api_key / api_secret missing — set them in "
                    "workdir/config/plugins/dnse.toml"
                )
            # The wrapper pins the API version (2026-07-23) and verifies TLS;
            # api_version from config is intentionally not forwarded.
            self._client = DNSEClient(self.config.api_key, self.config.api_secret,
                                      base_url=self.config.base_url)
            import logging
            level = logging.INFO if not self.is_production else logging.WARNING
            logging.getLogger(__name__).log(level, self.announce_endpoints())
        return self._client

    #: Official DNSE hosts. Anything else means a fake/local venue, which the
    #: plugin announces loudly — the dangerous mistake is believing you are on
    #: a fake when you are pointed at production, so the banner is derived from
    #: the endpoints themselves rather than a separate flag that could disagree.
    PRODUCTION_HOSTS = ("openapi.dnse.com.vn", "ws-openapi.dnse.com.vn")

    @property
    def is_production(self) -> bool:
        """Whether any endpoint points at the real DNSE."""
        assert self.config is not None
        # ANY production endpoint means production. `all()` would label a
        # production-REST + localhost-WS config as a test venue while every
        # order went to the real exchange — the exact mistake this guards.
        return any(host in url for host, url in
                   zip(self.PRODUCTION_HOSTS, (self.config.base_url, self.config.ws_url)))

    def announce_endpoints(self) -> str:
        """One-line endpoint banner, logged before the first request."""
        assert self.config is not None
        mixed = (("openapi.dnse.com.vn" in self.config.base_url)
                 != ("ws-openapi.dnse.com.vn" in self.config.ws_url))
        tag = ("MIXED - PRODUCTION REST/WS + LOCAL" if mixed
               else "LIVE DNSE" if self.is_production else "TEST VENUE (not DNSE)")
        return f"[{tag}] rest={self.config.base_url} ws={self.config.ws_url}"

    #: VN30F1M has a lunch break and a 14:45 close, so quiet stretches are
    #: normal and long. The framework default (3 bars) would reconnect-churn
    #: across every session gap; CCXT raises it to 30 for the same reason.
    feed_timeout_bars: int | None = 40

    def resolve_contract(self, symbol: str | None = None) -> str:
        """Map a ``symbolType`` alias to the tradable KRX contract code.

        DNSE's symbol model is two-level: ``VN30F1M`` is a *symbolType* accepted
        by ``/price/ohlc``, while orders and the streaming channels want the
        dated contract (``41I1G8000``). ``/market/instruments`` carries both, so
        the alias resolves to the front month here and is cached per instance.

        A symbol that is not a known alias is returned unchanged — stocks such
        as ``HPG`` are already their own tradable code.
        """
        wanted = (symbol or self.symbol or "").upper()
        cache = getattr(self, "_contract_cache", None)
        if cache is None:
            cache = self._contract_cache = {}
        if wanted in cache:
            return cache[wanted]

        status, body = self.client.get_instruments(limit=200)
        resolved = wanted
        if status == 200 and isinstance(body, dict):
            for row in body.get("data") or []:
                if row.get("symbolType") == wanted:
                    resolved = row["symbol"]
                    break
        cache[wanted] = resolved
        return resolved

    @override
    def normalize_symbol(self, symbol: str) -> str:
        """Streaming wants the tradable contract code, not the alias.

        Called by the live runner before ``watch_ohlcv``. Historical methods keep
        using ``self.symbol`` (the alias), which is what ``/price/ohlc`` accepts —
        the two paths genuinely differ.
        """
        return self.resolve_contract(symbol)

    def _secdef(self, symbol: str) -> dict:
        """Security definition for ``symbol``, or ``{}``.

        ``/price/{symbol}/secdef`` returns a LIST (one row per board) and is
        empty for a symbolType alias, so derivatives are looked up by their
        resolved contract code.
        """
        cache = getattr(self, "_secdef_cache", None)
        if cache is None:
            cache = self._secdef_cache = {}
        if symbol in cache:
            return cache[symbol]
        lookup = symbol
        if symbol.upper().startswith("VN30F"):
            lookup = self.resolve_contract(symbol)
        status, body = self.client.get_security_definition(lookup)
        if status == 200 and isinstance(body, list) and body:
            row = body[0]
        elif status == 200 and isinstance(body, dict):
            row = body
        else:
            row = {}
        cache[symbol] = row
        return row

    def _reference_price(self, symbol: str) -> float:
        """Basic (reference) price, used to pick the stock tick band."""
        row = self._secdef(symbol)
        for key in ("basicPrice", "ceilingPrice", "floorPrice"):
            value = row.get(key)
            if value:
                return float(value)
        return 20.0  # mid-band fallback -> 0.05 tick

    @property
    def market_type(self) -> str:
        """DNSE bar type for the current symbol.

        ``marketType`` is a property of the instrument, not of the environment.
        Prefer the venue's own ``securityGroupId`` (``FU`` = futures,
        ``ST`` = stock); fall back to the VN30F prefix when secdef is silent.
        """
        symbol = (self.symbol or "").upper()
        group = (self._secdef(symbol).get("securityGroupId") or "").upper()
        if group:
            return "DERIVATIVE" if group == "FU" else "STOCK"
        return "DERIVATIVE" if symbol.startswith("VN30F") else "STOCK"

    # --- timeframe conversion ---

    @classmethod
    @override
    def to_tradingview_timeframe(cls, timeframe: str) -> str:
        return _DNSE_TO_TV.get(timeframe, timeframe)

    @classmethod
    @override
    def to_exchange_timeframe(cls, timeframe: str) -> str:
        try:
            return _TV_TO_DNSE[timeframe]
        except KeyError:
            raise ValueError(
                f"DNSE has no resolution for timeframe {timeframe!r}; "
                f"supported: {', '.join(_TV_TO_DNSE)}"
            ) from None

    # --- symbol metadata ---

    @override
    def get_list_of_symbols(self, *args, **kwargs) -> list[str]:
        return [self.symbol] if self.symbol else []

    @override
    def update_symbol_info(self) -> SymInfo:
        """Build SymInfo for the current symbol.

        Sessions are hard-coded from the confirmed VN30F1M schedule rather than
        discovered, because the opening hours gate the idle-bar synth and the
        feed-staleness watchdog — a wrong value there produces synthetic bars
        through the lunch break.
        """
        symbol = (self.symbol or "").upper()
        derivative = self.market_type == "DERIVATIVE"

        opening_hours, session_starts, session_ends = [], [], []
        for day in range(1, 6):  # Mon-Fri
            opening_hours.append(SymInfoInterval(day=day, start=_MORNING[0], end=_MORNING[1]))
            opening_hours.append(SymInfoInterval(day=day, start=_AFTERNOON[0], end=_AFTERNOON[1]))
            session_starts.append(SymInfoSession(day=day, time=_MORNING[0]))
            session_ends.append(SymInfoSession(day=day, time=_AFTERNOON[1]))

        # DNSE quotes stocks in THOUSANDS of VND (HPG 22,150 VND -> 22.15), so
        # tick sizes must be expressed in the same unit. HOSE bands, in VND:
        #   < 10,000 -> 10d   |   10,000-49,950 -> 50d   |   >= 50,000 -> 100d
        # which in thousand-units is 0.01 / 0.05 / 0.10.
        if derivative:
            mintick, minmove, pricescale = 0.1, 1, 10
        else:
            reference = self._reference_price(symbol)
            mintick = 0.01 if reference < 10 else 0.05 if reference < 50 else 0.10
            pricescale, minmove = 100, round(mintick * 100)

        return SymInfo(
            prefix="DNSE",
            description=f"{symbol} (DNSE {self.market_type.lower()})",
            ticker=symbol,
            currency="VND",
            basecurrency=None,
            period=self.timeframe or "15",
            type="futures" if derivative else "stock",
            mintick=mintick,
            pricescale=pricescale,
            minmove=minmove,
            # VN30 index futures: 100,000 VND per index point. Stocks are
            # quoted in thousands, so one price unit is 1,000 VND.
            pointvalue=100_000 if derivative else 1_000,
            mincontract=1,
            timezone="Asia/Ho_Chi_Minh",
            volumetype="base",
            taker_fee=0.0,
            maker_fee=0.0,
            opening_hours=opening_hours,
            session_starts=session_starts,
            session_ends=session_ends,
        )

    # --- history ---

    @override
    def download_ohlcv(self, time_from: datetime, time_to: datetime,
                       on_progress: Callable[[datetime], None] | None = None,
                       limit: int | None = None, with_extra: bool = False):
        """Fetch candles and persist them via ``save_ohlcv_data``.

        ``/price/ohlc`` returns TradingView-UDF-style parallel arrays
        ``{t,o,h,l,c,v}`` with ``t`` in unix SECONDS; PyneCore stores
        milliseconds.
        """
        assert self.timeframe is not None, "timeframe required"
        resolution = self.to_exchange_timeframe(self.timeframe)

        from_ts = int(time_from.replace(tzinfo=time_from.tzinfo or timezone.utc).timestamp())
        to_ts = int(time_to.replace(tzinfo=time_to.tzinfo or timezone.utc).timestamp())

        status, body = self.client.get_ohlc(self.market_type, {
            "symbol": self.symbol, "resolution": resolution,
            "from": from_ts, "to": to_ts,
        })
        if status != 200 or not isinstance(body, dict):
            raise RuntimeError(f"DNSE OHLC request failed: HTTP {status} {body}")

        times = body.get("t") or []
        for index, ts in enumerate(times):
            self.save_ohlcv_data(OHLCV(
                timestamp=int(ts) * 1000,
                open=float(body["o"][index]),
                high=float(body["h"][index]),
                low=float(body["l"][index]),
                close=float(body["c"][index]),
                volume=float(body["v"][index]),
            ))
            if on_progress is not None and index % 200 == 0:
                # The download runner compares against a naive ``from``; an
                # aware datetime here raises "can't subtract offset-naive and
                # offset-aware datetimes".
                progress_at = datetime.fromtimestamp(int(ts), timezone.utc).replace(tzinfo=None)
                if time_to.tzinfo is not None:
                    progress_at = progress_at.replace(tzinfo=timezone.utc)
                on_progress(progress_at)

        if on_progress is not None:
            on_progress(time_to)
