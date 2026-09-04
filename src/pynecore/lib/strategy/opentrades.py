from ...types.na import NA, na_float, na_int
from ...types import PyneFloat, PyneInt, PyneStr
from ... import lib

from ...core.module_property import module_property


#
# Functions
#


def _trade_index(trade_num: int) -> int:
    """
    Normalize a trade number into a list index.

    Pine's ``int`` is a static type only, so an int-TYPED expression may arrive
    carrying a fractional value; this consuming slot truncates it. An ``na``
    trade number becomes -1, which every accessor already answers with ``na``
    instead of reaching the subscript with a non-integer.

    :param trade_num: Trade number of the trade, possibly fractional or ``na``
    :return: Integer index, or -1 when there is none
    """
    if not (trade_num == trade_num):  # is_na_arg
        return -1
    return int(trade_num)


# noinspection PyProtectedMember
def commission(trade_num: int) -> PyneFloat:
    """
    Returns the sum of entry and exit fees paid in the open trade, expressed in strategy.account_currency

    :param trade_num: The trade number of the open trade. The number of the first trade is zero
    :return: The sum of entry and exit fees paid in the open trade, expressed in strategy.account_currency
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return na_float
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        return lib._script.position.open_trades[trade_num].commission
    except (IndexError, AssertionError):
        return 0.0


# noinspection PyProtectedMember
def entry_bar_index(trade_num: int) -> PyneInt:
    """
    Returns the bar_index of the open trade's entry

    :param trade_num: The trade number of the open trade. The number of the first trade is zero
    :return: The bar_index of the open trade's entry
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return na_int
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        return float(lib._script.position.open_trades[trade_num].entry_bar_index)
    except (IndexError, AssertionError):
        return na_int


# noinspection PyProtectedMember
def entry_comment(trade_num: int) -> PyneStr:
    """
    Returns the comment message of the open trade's entry

    :param trade_num: The trade number of the open trade. The number of the first trade is zero
    :return: The comment message of the open trade's entry
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return NA(str)
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        comment = lib._script.position.open_trades[trade_num].entry_comment
        return comment if comment is not None else NA(str)
    except (IndexError, AssertionError):
        return NA(str)


# noinspection PyProtectedMember
def entry_id(trade_num: int) -> PyneStr:
    """
    Returns the id of the open trade's entry

    :param trade_num: The trade number of the open trade. The number of the first trade is zero
    :return: The id of the open trade's entry
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return NA(str)
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        entry_id_ = lib._script.position.open_trades[trade_num].entry_id
        return entry_id_ if entry_id_ is not None else NA(str)
    except (IndexError, AssertionError):
        return NA(str)


# noinspection PyProtectedMember
def entry_price(trade_num: int) -> PyneFloat:
    """
    Returns the price of the open trade's entry

    :param trade_num: The trade number of the open trade. The number of the first trade is zero
    :return: The price of the open trade's entry
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return na_float
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        return lib._script.position.open_trades[trade_num].entry_price
    except (IndexError, AssertionError):
        return na_float


# noinspection PyProtectedMember
def entry_time(trade_num: int) -> PyneInt:
    """
    Returns the time of the open trade's entry (UNIX)

    :param trade_num: The trade number of the open trade. The number of the first trade is zero
    :return: The time of the open trade's entry (UNIX)
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return na_int
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        return float(lib._script.position.open_trades[trade_num].entry_time)
    except (IndexError, AssertionError):
        return na_int


# noinspection PyProtectedMember
def max_drawdown(trade_num: int) -> PyneFloat:
    """
    Returns the maximum drawdown of the open trade

    :param trade_num: The trade number of the open trade. The number of the first trade is zero
    :return: The maximum drawdown of the open trade
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return na_float
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        return lib._script.position.open_trades[trade_num].max_drawdown
    except (IndexError, AssertionError):
        return 0.0


# noinspection PyProtectedMember
def max_drawdown_percent(trade_num: int) -> PyneFloat:
    """
    Returns the maximum drawdown percentage of the open trade

    :param trade_num: The trade number of the open trade. The number of the first trade is zero
    :return: The maximum drawdown percentage of the open trade
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return na_float
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        return lib._script.position.open_trades[trade_num].max_drawdown_percent
    except (IndexError, AssertionError):
        return 0.0


# noinspection PyProtectedMember
def max_runup(trade_num: int) -> PyneFloat:
    """
    Returns the maximum runup of the open trade

    :param trade_num: The trade number of the open trade. The number of the first trade is zero
    :return: The maximum runup of the open trade
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return na_float
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        return lib._script.position.open_trades[trade_num].max_runup
    except (IndexError, AssertionError):
        return 0.0


# noinspection PyProtectedMember
def max_runup_percent(trade_num: int) -> PyneFloat:
    """
    Returns the maximum runup percentage of the open trade

    :param trade_num: The trade number of the open trade. The number of the first trade is zero
    :return: The maximum runup percentage of the open trade
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return na_float
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        return lib._script.position.open_trades[trade_num].max_runup_percent
    except (IndexError, AssertionError):
        return 0.0


# noinspection PyProtectedMember
def profit(trade_num: int) -> PyneFloat:
    """
    Returns the profit of the open trade expressed in strategy.account_currency
    :param trade_num: The trade number of the open trade. The number of the first trade is zero
    :return: The profit of the open trade expressed in strategy.account_currency
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return na_float
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        return lib._script.position.open_trades[trade_num].profit
    except (IndexError, AssertionError):
        return 0.0


# noinspection PyProtectedMember
def profit_percent(trade_num: int) -> PyneFloat:
    """
    Returns the profit percentage of the open trade
    :param trade_num: The trade number of the open trade. The number of the first trade is zero
    :return: The profit percentage of the open trade
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return na_float
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        return lib._script.position.open_trades[trade_num].profit_percent
    except (IndexError, AssertionError):
        return 0.0


# noinspection PyProtectedMember
def size(trade_num: int) -> PyneFloat:
    """
    Returns the size and direction (<0 short >0 long) of the open trade

    :param trade_num: The trade number of the open trade. The number of the first trade is zero
    :return: The size and direction (<0 short >0 long) of the open trade
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return 0.0
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        return lib._script.position.open_trades[trade_num].size
    except (IndexError, AssertionError):
        return 0.0


#
# Module property
#

# noinspection PyProtectedMember
@module_property
def opentrades() -> PyneInt:
    """
    Number of market position entries, which were not closed and remain opened.

    :return: The number of open trades
    """
    if lib._script is None or lib._script.position is None:
        return 0.0
    position = lib._script.position
    # A Pine int is a double at runtime
    return float(len(position.open_trades))


# noinspection PyProtectedMember
@module_property
def capital_held() -> PyneFloat:
    """
    The capital the open trades hold, in currency units.

    :return: The summed entry value of the open trades, 0 while flat, na when the strategy
             requires no margin at all
    """
    # Measured on TradingView (BINANCE:BTCUSDT 1D, pyramiding 3): the value is the
    # sum of |size| * entry price over the open trades. It does not follow the
    # market, and a short trade contributes positively. TradingView keeps a running
    # accumulator instead of re-summing, so its flat state carries a float residue
    # (3.6e-12 measured) where this fresh sum gives an exact 0.0.
    # The margin percentages do NOT scale the value: margin_long=50/margin_short=25 and
    # margin_long=50/margin_short=0 both report the full entry value. Only the fully
    # unfunded configuration is special -- with margin_long=0 AND margin_short=0
    # TradingView reports na on every bar, flat ones included.
    # The pointvalue factor is not separately measured -- the probe symbol has
    # pointvalue 1 -- it follows how the engine scales its other monetary values.
    if lib._script is None or lib._script.position is None:
        return 0.0
    if lib._script.margin_long <= 0.0 and lib._script.margin_short <= 0.0:
        return na_float
    # Imported here rather than at module level: the strategy package imports this
    # module while it is still executing, long before it defines this function.
    from . import _account_point_value
    pv = _account_point_value()
    total = 0.0
    for trade in lib._script.position.open_trades:
        total += abs(trade.size) * trade.entry_price * pv
    return total
