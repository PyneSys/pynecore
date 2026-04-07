from ...types.na import NA
from ...types import PyneFloat, PyneInt, PyneStr
from ... import lib

from ...core.module_property import module_property
from ...core.callable_module import CallableModule


#
# Module object
#


class ClosedTradesModule(CallableModule):
    #
    # Functions
    #
    # noinspection PyProtectedMember
    @staticmethod
    def commission(trade_num: int) -> PyneFloat:
        """
        Returns the sum of entry and exit fees paid in the closed trade, expressed in strategy.account_currency

        :param trade_num: The trade number of the closed trade. The number of the first trade is zero
        :return: The sum of entry and exit fees paid in the closed trade, expressed in strategy.account_currency
        """
        if trade_num < 0:
            return NA(float)
        try:
            assert lib._script is not None
            assert lib._script.position is not None
            return lib._script.position.closed_trades[trade_num].commission
        except (IndexError, AssertionError):
            return 0.0

    # noinspection PyProtectedMember
    @staticmethod
    def entry_bar_index(trade_num: int) -> PyneInt:
        """
        Returns the bar_index of the closed trade's entry

        :param trade_num: The trade number of the closed trade. The number of the first trade is zero
        :return: The bar_index of the closed trade's entry
        """
        if trade_num < 0:
            return NA(int)
        try:
            assert lib._script is not None
            assert lib._script.position is not None
            return lib._script.position.closed_trades[trade_num].entry_bar_index
        except (IndexError, AssertionError):
            return NA(int)

    # noinspection PyProtectedMember
    @staticmethod
    def entry_comment(trade_num: int) -> PyneStr:
        """
        Returns the comment message of the closed trade's entry

        :param trade_num: The trade number of the closed trade. The number of the first trade is zero
        :return: The comment message of the closed trade's entry
        """
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
    @staticmethod
    def entry_id(trade_num: int) -> PyneStr:
        """
        Returns the id of the closed trade's entry

        :param trade_num: The trade number of the closed trade. The number of the first trade is zero
        :return: The id of the closed trade's entry
        """
        if trade_num < 0:
            return NA(str)
        try:
            assert lib._script is not None
            assert lib._script.position is not None
            entry_id = lib._script.position.closed_trades[trade_num].entry_id
            return entry_id if entry_id is not None else NA(str)
        except (IndexError, AssertionError):
            return NA(str)

    # noinspection PyProtectedMember
    @staticmethod
    def entry_price(trade_num: int) -> PyneFloat:
        """
        Returns the price of the closed trade's entry

        :param trade_num: The trade number of the closed trade. The number of the first trade is zero
        :return: The price of the closed trade's entry
        """
        if trade_num < 0:
            return NA(float)
        try:
            assert lib._script is not None
            assert lib._script.position is not None
            return lib._script.position.closed_trades[trade_num].entry_price
        except (IndexError, AssertionError):
            return NA(float)

    # noinspection PyProtectedMember
    @staticmethod
    def entry_time(trade_num: int) -> PyneInt:
        """
        Returns the time of the closed trade's entry (UNIX)

        :param trade_num: The trade number of the closed trade. The number of the first trade is zero
        :return: The time of the closed trade's entry
        """
        if trade_num < 0:
            return NA(int)
        try:
            assert lib._script is not None
            assert lib._script.position is not None
            return lib._script.position.closed_trades[trade_num].entry_time
        except (IndexError, AssertionError):
            return NA(int)

    # noinspection PyProtectedMember
    @staticmethod
    def exit_bar_index(trade_num: int) -> PyneInt:
        """
        Returns the bar_index of the closed trade's exit

        :param trade_num: The trade number of the closed trade. The number of the first trade is zero
        :return: The bar_index of the closed trade's exit
        """
        if trade_num < 0:
            return NA(int)
        try:
            assert lib._script is not None
            assert lib._script.position is not None
            return lib._script.position.closed_trades[trade_num].exit_bar_index
        except (IndexError, AssertionError):
            return NA(int)

    # noinspection PyProtectedMember
    @staticmethod
    def exit_comment(trade_num: int) -> PyneStr:
        """
        Returns the comment message of the closed trade's exit

        :param trade_num: The trade number of the closed trade. The number of the first trade is zero
        :return: The comment message of the closed trade's exit
        """
        if trade_num < 0:
            return NA(str)
        try:
            assert lib._script is not None
            assert lib._script.position is not None
            return lib._script.position.closed_trades[trade_num].exit_comment
        except (IndexError, AssertionError):
            return NA(str)

    # noinspection PyProtectedMember
    @staticmethod
    def exit_id(trade_num: int) -> PyneStr:
        """
        Returns the id of the closed trade's exit

        :param trade_num: The trade number of the closed trade. The number of the first trade is zero
        :return: The id of the closed trade's exit
        """
        if trade_num < 0:
            return NA(str)
        try:
            assert lib._script is not None
            assert lib._script.position is not None
            exit_id = lib._script.position.closed_trades[trade_num].exit_id
            return exit_id if exit_id is not None else NA(str)
        except (IndexError, AssertionError):
            return NA(str)

    # noinspection PyProtectedMember
    @staticmethod
    def exit_price(trade_num: int) -> PyneFloat:
        """
        Returns the price of the closed trade's exit

        :param trade_num: The trade number of the closed trade. The number of the first trade is zero
        :return: The price of the closed trade's exit
        """
        if trade_num < 0:
            return NA(float)
        try:
            assert lib._script is not None
            assert lib._script.position is not None
            return lib._script.position.closed_trades[trade_num].exit_price
        except (IndexError, AssertionError):
            return NA(float)

    # noinspection PyProtectedMember
    @staticmethod
    def exit_time(trade_num: int) -> PyneInt:
        """
        Returns the time of the closed trade's exit (UNIX)
        :param trade_num: The trade number of the closed trade. The number of the first trade is zero
        :return: The time of the closed trade's exit
        """
        if trade_num < 0:
            return NA(int)
        try:
            assert lib._script is not None
            assert lib._script.position is not None
            return lib._script.position.closed_trades[trade_num].exit_time
        except (IndexError, AssertionError):
            return NA(int)

    # noinspection PyProtectedMember
    @staticmethod
    def max_drawdown(trade_num: int) -> PyneFloat:
        """
        Returns the maximum drawdown of the closed trade

        :param trade_num: The trade number of the closed trade. The number of the first trade is zero
        :return: The maximum drawdown of the closed trade
        """
        if trade_num < 0:
            return NA(float)
        try:
            assert lib._script is not None
            assert lib._script.position is not None
            return lib._script.position.closed_trades[trade_num].max_drawdown
        except (IndexError, AssertionError):
            return 0.0

    # noinspection PyProtectedMember
    @staticmethod
    def max_drawdown_percent(trade_num: int) -> PyneFloat:
        """
        Returns the maximum drawdown percent of the closed trade

        :param trade_num: The trade number of the closed trade. The number of the first trade is zero
        :return: The maximum drawdown percent of the closed trade
        """
        if trade_num < 0:
            return NA(float)
        try:
            assert lib._script is not None
            assert lib._script.position is not None
            return lib._script.position.closed_trades[trade_num].max_drawdown_percent
        except (IndexError, AssertionError):
            return 0.0

    # noinspection PyProtectedMember
    @staticmethod
    def max_runup(trade_num: int) -> PyneFloat:
        """
        Returns the maximum runup of the closed trade

        :param trade_num: The trade number of the closed trade. The number of the first trade is zero
        :return: The maximum runup of the closed trade
        """
        if trade_num < 0:
            return NA(float)
        try:
            assert lib._script is not None
            assert lib._script.position is not None
            return lib._script.position.closed_trades[trade_num].max_runup
        except (IndexError, AssertionError):
            return 0.0

    # noinspection PyProtectedMember
    @staticmethod
    def max_runup_percent(trade_num: int) -> PyneFloat:
        """
        Returns the maximum runup percent of the closed trade

        :param trade_num: The trade number of the closed trade. The number of the first trade is zero
        :return: The maximum runup percent of the closed trade
        """
        if trade_num < 0:
            return NA(float)
        try:
            assert lib._script is not None
            assert lib._script.position is not None
            return lib._script.position.closed_trades[trade_num].max_runup_percent
        except (IndexError, AssertionError):
            return 0.0

    # noinspection PyProtectedMember
    @staticmethod
    def profit(trade_num: int) -> PyneFloat:
        """
        Returns the profit of the closed trade

        :param trade_num: The trade number of the closed trade. The number of the first trade is zero
        :return: The profit of the closed trade
        """
        if trade_num < 0:
            return NA(float)
        try:
            assert lib._script is not None
            assert lib._script.position is not None
            return lib._script.position.closed_trades[trade_num].profit
        except (IndexError, AssertionError):
            return 0.0

    # noinspection PyProtectedMember
    @staticmethod
    def profit_percent(trade_num: int) -> PyneFloat:
        """
        Returns the profit percent of the closed trade

        :param trade_num: The trade number of the closed trade. The number of the first trade is zero
        :return: The profit percent of the closed trade
        """
        if trade_num < 0:
            return NA(float)
        try:
            assert lib._script is not None
            assert lib._script.position is not None
            return lib._script.position.closed_trades[trade_num].profit_percent
        except (IndexError, AssertionError):
            return 0.0

    # noinspection PyProtectedMember
    @staticmethod
    def size(trade_num: int) -> PyneFloat:
        if trade_num < 0:
            return 0.0
        try:
            assert lib._script is not None
            assert lib._script.position is not None
            return lib._script.position.closed_trades[trade_num].size
        except (IndexError, AssertionError):
            return 0.0


#
# Callable module function
#

# noinspection PyProtectedMember
@module_property
def closedtrades() -> int:
    """
    Number of trades, which were closed for the whole trading range.

    :return: The number of closed trades
    """
    if lib._script is None or lib._script.position is None:
        return 0
    position = lib._script.position
    return len(position.closed_trades)


#
# Module initialization
#

ClosedTradesModule(__name__)
