from ...types.na import NA, NA
from ... import lib

__all__ = [
    "commission", "entry_bar_index", "entry_comment", "entry_id", "entry_price", "entry_time",
    "exit_bar_index", "exit_comment", "exit_id", "exit_price", "exit_time", "max_drawdown",
    "max_runup", "profit", "size"
]


# noinspection PyProtectedMember
def commission(trade_num: int) -> float | NA:
    """
    Returns the sum of entry and exit fees paid in the closed trade, expressed in strategy.account_currency

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The sum of entry and exit fees paid in the closed trade, expressed in strategy.account_currency
    """
    if trade_num < 0:
        return NA
    try:
        return lib._script.position.closed_trades[trade_num].commission
    except IndexError:
        return 0.0


# noinspection PyProtectedMember
def entry_bar_index(trade_num: int) -> int | NA:
    """
    Returns the bar_index of the closed trade's entry

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The bar_index of the closed trade's entry
    """
    if trade_num < 0:
        return NA
    try:
        return lib._script.position.closed_trades[trade_num].entry_bar_index
    except IndexError:
        return NA


# noinspection PyProtectedMember
def entry_comment(trade_num: int) -> str | NA:
    """
    Returns the comment message of the closed trade's entry

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The comment message of the closed trade's entry
    """
    if trade_num < 0:
        return NA
    try:
        return lib._script.position.open_trades[trade_num].entry_comment
    except IndexError:
        return NA


# noinspection PyProtectedMember
def entry_id(trade_num: int) -> str | NA:
    """
    Returns the id of the closed trade's entry

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The id of the closed trade's entry
    """
    if trade_num < 0:
        return NA
    try:
        return lib._script.position.closed_trades[trade_num].entry_id
    except IndexError:
        return NA


# noinspection PyProtectedMember
def entry_price(trade_num: int) -> float | NA:
    """
    Returns the price of the closed trade's entry

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The price of the closed trade's entry
    """
    if trade_num < 0:
        return NA
    try:
        return lib._script.position.closed_trades[trade_num].entry_price
    except IndexError:
        return NA


# noinspection PyProtectedMember
def entry_time(trade_num: int) -> int | NA:
    """
    Returns the time of the closed trade's entry (UNIX)

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The time of the closed trade's entry
    """
    if trade_num < 0:
        return NA
    try:
        return lib._script.position.closed_trades[trade_num].entry_time
    except IndexError:
        return NA


# noinspection PyProtectedMember
def exit_bar_index(trade_num: int) -> int | NA:
    """
    Returns the bar_index of the closed trade's exit

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The bar_index of the closed trade's exit
    """
    if trade_num < 0:
        return NA
    try:
        return lib._script.position.closed_trades[trade_num].exit_bar_index
    except IndexError:
        return NA


# noinspection PyProtectedMember
def exit_comment(trade_num: int) -> str | NA:
    """
    Returns the comment message of the closed trade's exit

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The comment message of the closed trade's exit
    """
    if trade_num < 0:
        return NA
    try:
        return lib._script.position.open_trades[trade_num].exit_comment
    except IndexError:
        return NA


# noinspection PyProtectedMember
def exit_id(trade_num: int) -> str | NA:
    """
    Returns the id of the closed trade's exit

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The id of the closed trade's exit
    """
    if trade_num < 0:
        return NA
    try:
        return lib._script.position.closed_trades[trade_num].exit_id
    except IndexError:
        return NA


# noinspection PyProtectedMember
def exit_price(trade_num: int) -> float | NA:
    """
    Returns the price of the closed trade's exit

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The price of the closed trade's exit
    """
    if trade_num < 0:
        return NA
    try:
        return lib._script.position.closed_trades[trade_num].exit_price
    except IndexError:
        return NA


# noinspection PyProtectedMember
def exit_time(trade_num: int) -> int | NA:
    """
    Returns the time of the closed trade's exit (UNIX)
    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The time of the closed trade's exit
    """
    if trade_num < 0:
        return NA
    try:
        return lib._script.position.closed_trades[trade_num].exit_time
    except IndexError:
        return NA


# noinspection PyProtectedMember
def max_drawdown(trade_num: int) -> float | NA:
    """
    Returns the maximum drawdown of the closed trade

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The maximum drawdown of the closed trade
    """
    if trade_num < 0:
        return NA
    try:
        return lib._script.position.closed_trades[trade_num].max_drawdown
    except IndexError:
        return 0.0


# noinspection PyProtectedMember
def max_drawdown_percent(trade_num: int) -> float | NA:
    """
    Returns the maximum drawdown percent of the closed trade

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The maximum drawdown percent of the closed trade
    """
    if trade_num < 0:
        return NA
    try:
        return lib._script.position.closed_trades[trade_num].max_drawdown_percent
    except IndexError:
        return 0.0


# noinspection PyProtectedMember
def max_runup(trade_num: int) -> float | NA:
    """
    Returns the maximum runup of the closed trade

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The maximum runup of the closed trade
    """
    if trade_num < 0:
        return NA
    try:
        return lib._script.position.closed_trades[trade_num].max_runup
    except IndexError:
        return 0.0


# noinspection PyProtectedMember
def max_runup_percent(trade_num: int) -> float | NA:
    """
    Returns the maximum runup percent of the closed trade

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The maximum runup percent of the closed trade
    """
    if trade_num < 0:
        return NA
    try:
        return lib._script.position.closed_trades[trade_num].max_runup_percent
    except IndexError:
        return 0.0


# noinspection PyProtectedMember
def profit(trade_num: int) -> float:
    """
    Returns the profit of the closed trade

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The profit of the closed trade
    """
    if trade_num < 0:
        return NA
    try:
        return lib._script.position.closed_trades[trade_num].profit
    except IndexError:
        return 0.0


# noinspection PyProtectedMember
def profit_percent(trade_num: int) -> float:
    """
    Returns the profit percent of the closed trade

    :param trade_num: The trade number of the closed trade. The number of the first trade is zero
    :return: The profit percent of the closed trade
    """
    if trade_num < 0:
        return NA
    try:
        return lib._script.position.closed_trades[trade_num].profit_percent
    except IndexError:
        return 0.0


# noinspection PyProtectedMember
def size(trade_num: int) -> float:
    if trade_num < 0:
        return NA
    try:
        return lib._script.position.closed_trades[trade_num].size
    except IndexError:
        return 0.0
