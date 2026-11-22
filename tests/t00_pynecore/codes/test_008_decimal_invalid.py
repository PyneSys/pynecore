"""
The decimal arithmetic of the strategy, the number formatter and the candle
aggregation stops with R0034 under its ``function`` and ``why``: the kernel's own
name, replaced by the outermost strategy entry point on the stack, and the
strategy's ``strategy()`` options of no known value are R0042.
"""
import math
from decimal import Decimal
from types import SimpleNamespace

import pytest

from pynecore import errorcodes, lib
from pynecore.core.aggregator import (VOLUME_ERROR_MAGNIFIER, VOLUME_ERROR_SECURITY,
                                      _merge_candles)
from pynecore.lib import format as pine_format, string, strategy, syminfo
from pynecore.types.ohlcv import OHLCV


def __test_helper_raised(call, *args) -> errorcodes.DecimalInvalidError:
    """The R0034 error ``call(*args)`` raises."""
    with pytest.raises(errorcodes.DecimalInvalidError) as raised:
        call(*args)
    assert raised.value.code is errorcodes.DECIMAL_INVALID
    return raised.value


def __test_the_kernels_name_themselves__(monkeypatch):
    """Each kernel raises under its own name with its reason"""
    monkeypatch.setattr(syminfo, '_size_round_factor', 1e4, raising=False)
    error = __test_helper_raised(strategy._book_commission,  # noqa
                                 [Decimal(0), 0.0, [], 0.0], 1.0, math.inf, 0.0, True)
    assert error.params == {'function': '_book_commission', 'why': 'infinity times zero'}
    error = __test_helper_raised(strategy._apply_booking,  # noqa
                                 [Decimal(0), 0.0, [0.0, 1.0], 0.0], math.inf, 1.0)
    assert error.params == {'function': '_apply_booking', 'why': 'infinity times zero'}
    error = __test_helper_raised(strategy._apply_booking,  # noqa
                                 [Decimal(0), 0.0, [0.0, 0.0], 0.0], 1.0, 0.0)
    assert error.params == {'function': '_apply_booking', 'why': 'division by zero'}
    error = __test_helper_raised(strategy._sig10_money, math.inf)  # noqa
    assert error.params == {'function': '_sig10_money', 'why': 'an infinite or NaN operand'}
    error = __test_helper_raised(strategy._money_ticks, 100.0, 0.0)  # noqa
    assert error.params == {'function': '_money_ticks', 'why': 'division by zero'}
    assert str(error) == "`_money_ticks`: the decimal calculation has no result (division by zero)."


def __test_the_outermost_entry_point_names_the_error__(monkeypatch):
    """An entry point on the stack replaces the kernel's name, the outermost one wins"""
    strategy._decimal_error('', '')  # noqa  (fills the entry-point table)

    def inner():
        return strategy._sig10_money(math.inf)  # noqa

    def outer():
        return inner()

    monkeypatch.setitem(strategy._DEC_ENTRY_POINTS, inner.__code__, 'strategy.exit')  # noqa
    assert __test_helper_raised(inner).params['function'] == 'strategy.exit'
    monkeypatch.setitem(strategy._DEC_ENTRY_POINTS, outer.__code__, 'strategy.entry')  # noqa
    assert __test_helper_raised(outer).params['function'] == 'strategy.entry'


def __test_the_entry_points_are_named__():
    """The table names the strategy entry points a decimal error is reported under"""
    strategy._decimal_error('', '')  # noqa
    assert set(strategy._DEC_ENTRY_POINTS.values()) == {  # noqa
        'strategy.entry', 'strategy.exit', 'strategy.order', 'strategy.close',
        'strategy.close_all', 'strategy.default_entry_qty', 'strategy.margin_liquidation_price',
        'process_orders', 'process_orders_at_close', 'settle_immediate_closes',
        'process_orders_magnified', 'process_deferred_margin_call', 'mark_to_last_fill'}


def __test_currency_formatting_past_28_digits__():
    """The cents of a currency format do not fit the thread context from 1e26 on"""
    why = 'quantizing to cents needs more than 28 digits'
    error = __test_helper_raised(string.tostring, 1e27, pine_format.price)
    assert error.params == {'function': 'str.tostring', 'why': why}
    error = __test_helper_raised(string.format, '{0,number,currency}', 1e27)
    assert error.params == {'function': 'str.format', 'why': why}


def __test_volumes_of_both_infinities__():
    """The caller of the candle merge names the error"""
    candles = [OHLCV(timestamp=0, open=1.0, high=1.0, low=1.0, close=1.0, volume=v)
               for v in (math.inf, -math.inf)]
    for volume_error in (VOLUME_ERROR_MAGNIFIER, VOLUME_ERROR_SECURITY):
        error = __test_helper_raised(_merge_candles, candles, 0, False, volume_error)
        assert (error.params['function'], error.params['why']) == volume_error
    assert VOLUME_ERROR_MAGNIFIER == (
        'strategy', 'the sub-bar volumes add up to infinity minus infinity')
    assert VOLUME_ERROR_SECURITY == (
        'request.security', 'the summed volumes hold both +inf and -inf')


def __test_an_unknown_default_qty_type_is_r0042__(monkeypatch):
    """A ``default_qty_type`` of no known value stops the sizing"""
    monkeypatch.setattr(lib, '_script', SimpleNamespace(default_qty_type='bogus'))
    with pytest.raises(errorcodes.InvalidOptionError) as raised:
        strategy._default_entry_budget(1.0, 1.0)  # noqa
    assert isinstance(raised.value, ValueError)
    assert raised.value.params['option'] == 'default_qty_type'
