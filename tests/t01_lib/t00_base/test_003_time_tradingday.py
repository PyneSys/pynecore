"""
@pyne
"""
from pynecore.lib import script, plot, time, time_tradingday


@script.indicator(title="Trading day", shorttitle="ttd")
def main():
    plot(time, "t")
    plot(time_tradingday, "ttd")


def __test_time_tradingday__(runner, log):
    """ time_tradingday is 00:00 UTC of the trading day; it rolls on overnight sessions """
    from datetime import datetime, UTC, time as dt_time

    from pynecore.types.ohlcv import OHLCV
    from pynecore.core.syminfo import SymInfoInterval, SymInfoSession

    def midnight_utc_ms(year: int, month: int, day: int) -> int:
        return int(datetime(year, month, day, tzinfo=UTC).timestamp() * 1000)

    def bar(iso: str) -> OHLCV:
        ts = int(datetime.fromisoformat(iso).replace(tzinfo=UTC).timestamp())
        return OHLCV(timestamp=ts, open=1.0, high=1.0, low=1.0, close=1.0, volume=1.0)

    # Forex-style symbol: exchange timezone New York, a 24h session that opens at
    # 17:00 every trading day (start == end), closed on Saturday. This is exactly how
    # TradingView reports EURUSD opening hours (weekday: 0=Mon ... 6=Sun).
    ny_days = (0, 1, 2, 3, 4, 6)
    override = {
        "period": "60",
        "type": "forex",
        "timezone": "America/New_York",
        "opening_hours": [SymInfoInterval(day=d, start=dt_time(17, 0), end=dt_time(17, 0)) for d in ny_days],
        "session_starts": [SymInfoSession(day=d, time=dt_time(17, 0)) for d in ny_days],
        "session_ends": [SymInfoSession(day=d, time=dt_time(17, 0)) for d in ny_days],
    }

    # Bars given in UTC; the comment shows the exchange-timezone (New York, EST) clock.
    bars = [
        "2025-01-05T22:00:00",  # Sun 17:00 NY -> session open -> Monday's trading day
        "2025-01-05T23:00:00",  # Sun 18:00 NY -> Monday's trading day
        "2025-01-06T14:00:00",  # Mon 09:00 NY -> still Monday's trading day
        "2025-01-06T21:00:00",  # Mon 16:00 NY -> still Monday's trading day
        "2025-01-06T22:00:00",  # Mon 17:00 NY -> session open -> Tuesday's trading day
        "2025-01-06T23:00:00",  # Mon 18:00 NY -> Tuesday's trading day
    ]
    expected = [
        midnight_utc_ms(2025, 1, 6),
        midnight_utc_ms(2025, 1, 6),
        midnight_utc_ms(2025, 1, 6),
        midnight_utc_ms(2025, 1, 6),
        midnight_utc_ms(2025, 1, 7),
        midnight_utc_ms(2025, 1, 7),
    ]

    candles = [bar(iso) for iso in bars]
    results = [_plot["ttd"] for _candle, _plot in runner(iter(candles), syminfo_override=override).run_iter()]

    assert results == expected, f"time_tradingday mismatch: {results} != {expected}"
    # Guard against the function-isolation regression that froze the value to bar 0.
    assert len(set(results)) >= 2, "time_tradingday must update per bar, not freeze"
