"""
The script declarations take positional arguments in Pine's documented order.

MEASURED on TradingView (CAPITALCOM:EURUSD 60): ``strategy("t", "t", false,
format.inherit, 2, scale.right, 0, false, false, 300, 0, strategy.fixed, 3)`` compiles
and trades 3 units, so the 10th argument is ``max_bars_back`` and the 13th
``default_qty_value``. Pine-only parameters a later release added come last, where a
call can only name them.
"""
import inspect

from pynecore.core.script import Script
from pynecore.lib import format, scale, strategy

#: Pine v6 ``strategy()`` parameters, in the reference's positional order
__test_helper_PINE_STRATEGY = (
    'title', 'shorttitle', 'overlay', 'format', 'precision', 'scale', 'pyramiding',
    'calc_on_order_fills', 'calc_on_every_tick', 'max_bars_back',
    'backtest_fill_limits_assumption', 'default_qty_type', 'default_qty_value',
    'initial_capital', 'currency', 'slippage', 'commission_type', 'commission_value',
    'process_orders_on_close', 'close_entries_rule', 'margin_long', 'margin_short',
    'explicit_plot_zorder', 'max_lines_count', 'max_labels_count', 'max_boxes_count',
    'calc_bars_count', 'risk_free_rate', 'use_bar_magnifier', 'fill_orders_on_standard_ohlc',
    'max_polylines_count', 'dynamic_requests', 'behind_chart',
)

#: Pine v6 ``indicator()`` parameters, in the reference's positional order
__test_helper_PINE_INDICATOR = (
    'title', 'shorttitle', 'overlay', 'format', 'precision', 'scale', 'max_bars_back',
    'timeframe', 'timeframe_gaps', 'explicit_plot_zorder', 'max_lines_count',
    'max_labels_count', 'max_boxes_count', 'calc_bars_count', 'max_polylines_count',
    'dynamic_requests', 'behind_chart',
)


def __test_helper_positional(method) -> tuple[str, ...]:
    parameters = list(inspect.signature(method).parameters.values())
    return tuple(p.name for p in parameters
                 if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD)


def __test_strategy_positional_order_is_pine__():
    """ ``strategy()`` binds its positional arguments as Pine does """
    names = __test_helper_positional(Script.strategy)
    assert names[:len(__test_helper_PINE_STRATEGY)] == __test_helper_PINE_STRATEGY


def __test_indicator_positional_order_is_pine__():
    """ ``indicator()`` binds its positional arguments as Pine does """
    names = __test_helper_positional(Script.indicator)
    assert names[:len(__test_helper_PINE_INDICATOR)] == __test_helper_PINE_INDICATOR


def __test_measured_positional_strategy_call__():
    """ The measured TradingView call: the 10th is max_bars_back, the 13th the quantity """
    decorator = Script.strategy('pos probe', 'pp', False, format.inherit, 2, scale.right, 0,
                                False, False, 300, 0, strategy.fixed, 3)
    main = decorator(lambda: None)
    declared = main.script
    assert declared.max_bars_back == 300
    assert declared.backtest_fill_limits_assumption == 0
    assert declared.default_qty_type == strategy.fixed
    assert declared.default_qty_value == 3
    assert declared.calc_on_every_history_tick is False
