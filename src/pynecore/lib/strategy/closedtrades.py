from ...types.na import NA, na_float, na_int
from ...types import PyneFloat, PyneInt, PyneStr
from ... import lib

from ...core.module_property import module_property


#
# Functions
#


# noinspection PyProtectedMember
def _trade_index(trade_num: int) -> int:
    """
    Normalize a global trade number into a retained-list index.

    Pine's ``int`` is a static type only, so an int-TYPED expression may arrive
    carrying a fractional value; this consuming slot truncates it. An ``na``
    trade number identifies global trade zero. Each accessor handles unavailable
    trades before indexing the retained list.

    :param trade_num: Trade number of the trade, possibly fractional or ``na``
    :return: Retained-list index, or a negative index for an unavailable trade
    """
    if not (trade_num == trade_num):  # is_na_arg
        index = 0
    else:
        index = int(trade_num)
    if lib._script is not None and lib._script.position is not None:
        position = lib._script.position
        index -= position.closed_trades_count - len(position.closed_trades)
    return index


# noinspection PyProtectedMember
def commission(trade_num: int) -> PyneFloat:
    """
    Returns the sum of entry and exit fees paid in the closed trade, expressed in strategy.account_currency

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The sum of entry and exit fees paid in the closed trade, expressed in strategy.account_currency
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return 0.0
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        return lib._script.position.closed_trades[trade_num].commission
    except (IndexError, AssertionError):
        return 0.0


# noinspection PyProtectedMember
def entry_bar_index(trade_num: int) -> PyneInt:
    """
    Returns the bar_index of the closed trade's entry

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The bar_index of the closed trade's entry
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return na_int
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        return float(lib._script.position.closed_trades[trade_num].entry_bar_index)
    except (IndexError, AssertionError):
        return na_int


# noinspection PyProtectedMember
def entry_comment(trade_num: int) -> PyneStr:
    """
    Returns the comment message of the closed trade's entry

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The comment message of the closed trade's entry
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return NA(str)
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        comment = lib._script.position.closed_trades[trade_num].entry_comment
        return comment if comment is not None else NA(str)
    except (IndexError, AssertionError):
        return NA(str)


# noinspection PyProtectedMember
def entry_id(trade_num: int) -> PyneStr:
    """
    Returns the id of the closed trade's entry

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The id of the closed trade's entry
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return NA(str)
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        entry_id_ = lib._script.position.closed_trades[trade_num].entry_id
        return entry_id_ if entry_id_ is not None else NA(str)
    except (IndexError, AssertionError):
        return NA(str)


# noinspection PyProtectedMember
def entry_price(trade_num: int) -> PyneFloat:
    """
    Returns the price of the closed trade's entry

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The price of the closed trade's entry
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return na_float
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        return lib._script.position.closed_trades[trade_num].entry_price
    except (IndexError, AssertionError):
        return na_float


# noinspection PyProtectedMember
def entry_time(trade_num: int) -> PyneInt:
    """
    Returns the time of the closed trade's entry (UNIX)

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The time of the closed trade's entry
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return na_int
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        return float(lib._script.position.closed_trades[trade_num].entry_time)
    except (IndexError, AssertionError):
        return na_int


# noinspection PyProtectedMember
def exit_bar_index(trade_num: int) -> PyneInt:
    """
    Returns the bar_index of the closed trade's exit

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The bar_index of the closed trade's exit
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return na_int
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        return float(lib._script.position.closed_trades[trade_num].exit_bar_index)
    except (IndexError, AssertionError):
        return na_int


# noinspection PyProtectedMember
def exit_comment(trade_num: int) -> PyneStr:
    """
    Returns the comment message of the closed trade's exit

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The comment message of the closed trade's exit
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return NA(str)
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        return lib._script.position.closed_trades[trade_num].exit_comment
    except (IndexError, AssertionError):
        return NA(str)


# noinspection PyProtectedMember
def exit_id(trade_num: int) -> PyneStr:
    """
    Returns the id of the closed trade's exit

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The id of the closed trade's exit
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return NA(str)
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        exit_id_ = lib._script.position.closed_trades[trade_num].exit_id
        return exit_id_ if exit_id_ is not None else NA(str)
    except (IndexError, AssertionError):
        return NA(str)


# noinspection PyProtectedMember
def exit_price(trade_num: int) -> PyneFloat:
    """
    Returns the price of the closed trade's exit

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The price of the closed trade's exit
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return na_float
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        return lib._script.position.closed_trades[trade_num].exit_price
    except (IndexError, AssertionError):
        return na_float


# noinspection PyProtectedMember
def exit_time(trade_num: int) -> PyneInt:
    """
    Returns the time of the closed trade's exit (UNIX)
    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The time of the closed trade's exit
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return na_int
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        return float(lib._script.position.closed_trades[trade_num].exit_time)
    except (IndexError, AssertionError):
        return na_int


# noinspection PyProtectedMember
def max_drawdown(trade_num: int) -> PyneFloat:
    """
    Returns the maximum drawdown of the closed trade

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The maximum drawdown of the closed trade
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return 0.0
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        return lib._script.position.closed_trades[trade_num].max_drawdown
    except (IndexError, AssertionError):
        return 0.0


# noinspection PyProtectedMember
def max_drawdown_percent(trade_num: int) -> PyneFloat:
    """
    Returns the maximum drawdown percent of the closed trade

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The maximum drawdown percent of the closed trade
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return 0.0
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        return lib._script.position.closed_trades[trade_num].max_drawdown_percent
    except (IndexError, AssertionError):
        return 0.0


# noinspection PyProtectedMember
def max_runup(trade_num: int) -> PyneFloat:
    """
    Returns the maximum runup of the closed trade

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The maximum runup of the closed trade
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return 0.0
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        return lib._script.position.closed_trades[trade_num].max_runup
    except (IndexError, AssertionError):
        return 0.0


# noinspection PyProtectedMember
def max_runup_percent(trade_num: int) -> PyneFloat:
    """
    Returns the maximum runup percent of the closed trade

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The maximum runup percent of the closed trade
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return 0.0
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        return lib._script.position.closed_trades[trade_num].max_runup_percent
    except (IndexError, AssertionError):
        return 0.0


# noinspection PyProtectedMember
def profit(trade_num: int) -> PyneFloat:
    """
    Returns the profit of the closed trade

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The profit of the closed trade
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return 0.0
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        return lib._script.position.closed_trades[trade_num].profit
    except (IndexError, AssertionError):
        return 0.0


# noinspection PyProtectedMember
def profit_percent(trade_num: int) -> PyneFloat:
    """
    Returns the profit percent of the closed trade

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The profit percent of the closed trade
    """
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return 0.0
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        return lib._script.position.closed_trades[trade_num].profit_percent
    except (IndexError, AssertionError):
        return 0.0


# noinspection PyProtectedMember
def size(trade_num: int) -> PyneFloat:
    trade_num = _trade_index(trade_num)
    if trade_num < 0:
        return 0.0
    try:
        assert lib._script is not None
        assert lib._script.position is not None
        return lib._script.position.closed_trades[trade_num].size
    except (IndexError, AssertionError):
        return 0.0


#
# Module property
#

# noinspection PyProtectedMember
@module_property
def closedtrades() -> PyneInt:
    """
    Number of trades, which were closed for the whole trading range.

    :return: The number of closed trades
    """
    if lib._script is None or lib._script.position is None:
        return 0.0
    position = lib._script.position
    # A Pine int is a double at runtime
    return float(position.closed_trades_count)


# noinspection PyProtectedMember
@module_property
def first_index() -> PyneInt:
    """
    The trade number of the oldest trade still listed in the List of Trades.

    :return: The index of the oldest retained closed trade
    """
    if lib._script is None or lib._script.position is None:
        return 0.0
    position = lib._script.position
    return float(position.closed_trades_count - len(position.closed_trades))
