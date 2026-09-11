"""
@pyne

Open-trade profit percentages follow each trade's current mark and entry cost.
"""
import pytest

from pynecore.lib import bar_index, input, na, plot, script, strategy
from pynecore.types.ohlcv import OHLCV
from pynecore.types.strategy import Commission


@script.strategy("Open trade percent", initial_capital=1000000, pyramiding=2,
                 default_qty_type=strategy.fixed, default_qty_value=4)
def main(quantity=input.float(4.0, "Quantity"), protect=input.bool(False, "Protect")):
    cycle = bar_index % 40
    if cycle == 0:
        strategy.entry("L1", strategy.long, qty=quantity)
    if cycle == 20:
        strategy.entry("S1", strategy.short, qty=quantity)
    if not protect:
        if cycle == 4:
            strategy.entry("L2", strategy.long, qty=quantity * 0.75)
        if cycle == 8:
            strategy.close("L1", qty=quantity / 2)
        if cycle == 12:
            strategy.close_all()
        if cycle == 24:
            strategy.entry("S2", strategy.short, qty=quantity * 0.75)
        if cycle == 28:
            strategy.close("S1", qty=quantity / 2)
        if cycle == 32:
            strategy.close_all()

    has_open = strategy.opentrades > 0
    has_second = strategy.opentrades > 1
    stop_triggered = has_open and strategy.opentrades.profit_percent(0) < -1.0
    if protect and stop_triggered:
        strategy.close_all()
    plot(strategy.opentrades, "open_count")
    plot(strategy.opentrades.profit(0) if has_open else na, "profit0")
    plot(strategy.opentrades.profit_percent(0) if has_open else na, "percent0")
    plot(strategy.opentrades.entry_price(0) if has_open else na, "entry0")
    plot(strategy.opentrades.size(0) if has_open else na, "size0")
    plot(strategy.opentrades.commission(0) if has_open else na, "fee0")
    plot(strategy.opentrades.profit(1) if has_second else na, "profit1")
    plot(strategy.opentrades.profit_percent(1) if has_second else na, "percent1")
    plot(strategy.opentrades.entry_price(1) if has_second else na, "entry1")
    plot(strategy.opentrades.size(1) if has_second else na, "size1")
    plot(strategy.opentrades.commission(1) if has_second else na, "fee1")
    plot(1 if stop_triggered else 0, "stop_triggered")


__test_helper_commissions = (
    ("zero", strategy.commission.percent, 0.0),
    ("percent", strategy.commission.percent, 1.0),
    ("contract", strategy.commission.cash_per_contract, 0.05),
    ("order", strategy.commission.cash_per_order, 0.12),
)


def __test_helper_bars():
    prices = {
        1: (100.0, 100.0),
        2: (100.0, 110.0),
        3: (110.0, 90.0),
        5: (120.0, 100.0),
        6: (100.0, 110.0),
        7: (110.0, 90.0),
        9: (100.0, 110.0),
        10: (110.0, 90.0),
    }
    for index in range(40):
        opening, closing = prices.get(index % 20, (100.0, 100.0))
        yield OHLCV(timestamp=1704067200 + index * 300, open=opening,
                    high=max(opening, closing), low=min(opening, closing),
                    close=closing, volume=10.0)


@pytest.mark.parametrize("name, commission_type, commission_value", __test_helper_commissions)
@pytest.mark.parametrize("quantity", (2.0, 4.0))
@pytest.mark.parametrize("pointvalue", (1.0, 50.0))
def __test_open_profit_percent_marks_each_trade__(
        runner, name: str, commission_type: Commission, commission_value: float,
        quantity: float, pointvalue: float):
    """Long and short percentages update before and after a partial close."""
    run = runner(__test_helper_bars(), syminfo_override={"pointvalue": pointvalue},
                 inputs={"quantity": quantity})
    run.script.commission_type = commission_type
    run.script.commission_value = commission_value
    checked = 0
    for index, (candle, values, _closed) in enumerate(run.run_iter()):
        cycle = index % 20
        expected_count = 2 if 5 <= cycle <= 12 else 1 if 1 <= cycle <= 4 else 0
        assert values["open_count"] == expected_count
        for trade_num in range(expected_count):
            size = quantity if trade_num == 0 else quantity * 0.75
            if trade_num == 0 and cycle >= 9:
                size /= 2
            entry = 100.0 if trade_num == 0 else 120.0
            if name in ("zero", "percent"):
                entry_fee = size * entry * pointvalue * commission_value * 0.01
                exit_fee = size * candle.close * pointvalue * commission_value * 0.01
            elif name == "contract":
                entry_fee = exit_fee = size * commission_value
            else:
                entry_fee = commission_value
                if trade_num == 0 and cycle >= 9:
                    entry_fee /= 2
                exit_fee = commission_value
            signed_size = size if index < 20 else -size
            expected_profit = signed_size * (candle.close - entry) * pointvalue - entry_fee - exit_fee
            expected_percent = expected_profit / (size * entry * pointvalue + entry_fee) * 100.0
            assert values[f"size{trade_num}"] == signed_size
            assert values[f"entry{trade_num}"] == pytest.approx(entry, abs=1e-12)
            assert values[f"fee{trade_num}"] == pytest.approx(entry_fee, abs=1e-12)
            assert values[f"profit{trade_num}"] == pytest.approx(expected_profit, abs=1e-10)
            assert values[f"percent{trade_num}"] == pytest.approx(expected_percent, abs=1e-10)
            checked += 1
    assert checked == 40


@pytest.mark.parametrize("name, commission_type, commission_value", __test_helper_commissions)
def __test_open_profit_percent_matches_tradingview__(
        csv_reader, runner, name: str, commission_type: Commission, commission_value: float):
    """Open-trade fields match the same TradingView chart in all commission modes."""
    with csv_reader("strategy_opentrades_profit_percent.csv", subdir="data") as source:
        run = runner(source, syminfo_override={
            "prefix": "CAPITALCOM", "ticker": "EURUSD", "currency": "USD",
            "period": "240", "mintick": 0.00001, "pricescale": 100000,
        })
        run.script.commission_type = commission_type
        run.script.commission_value = commission_value
        bars = 0
        checked = 0
        for candle, values, _closed in run.run_iter():
            expected = candle.extra_fields
            assert values["open_count"] == expected[f"{name}_open_count"]
            for trade_num in range(int(values["open_count"])):
                for field in ("profit", "percent", "entry", "size", "fee"):
                    key = f"{field}{trade_num}"
                    assert values[key] == pytest.approx(expected[f"{name}_{key}"],
                                                        rel=1e-10, abs=1e-10)
                    checked += 1
            bars += 1
        assert bars == 80
        assert checked == 400


def __test_open_profit_percent_triggers_protection__(runner):
    """A percentage-based stop closes an actual losing position."""
    run = runner(__test_helper_bars(), inputs={"protect": True})
    outcomes = [(dict(values), len(closed)) for _candle, values, closed in run.run_iter()]
    assert outcomes[2][0]["percent0"] == pytest.approx(10.0)
    assert outcomes[3][0]["percent0"] == pytest.approx(-10.0)
    assert outcomes[3][0]["stop_triggered"] == 1
    assert outcomes[4][0]["open_count"] == 0
    assert outcomes[4][1] == 1
