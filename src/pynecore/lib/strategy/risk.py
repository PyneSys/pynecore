from ...types.na import NA
from ...types import PyneStr
from .. import strategy

from ... import lib


# noinspection PyShadowingBuiltins,PyProtectedMember
def max_drawdown(
        value: float | int,
        type: strategy.QtyType = strategy.percent_of_equity,
        alert_message: PyneStr = NA(str)
) -> None:
    """
    The purpose of this rule is to determine maximum drawdown. The rule affects the whole strategy.
    Once the maximum drawdown value is reached, all pending orders are cancelled, all open positions
    are closed and no new orders can be placed.

    :param value: The maximum drawdown value
    :param type: The type of the value
    :param alert_message: The alert message
    """
    if lib._script is None or lib._script.position is None:
        return

    lib._script.position.risk_max_drawdown_value = value
    lib._script.position.risk_max_drawdown_type = type
    lib._script.position.risk_max_drawdown_alert = None if isinstance(alert_message, NA) else str(alert_message)


# noinspection PyProtectedMember
def allow_entry_in(value: strategy.direction.Direction) -> None:
    """
    This function can be used to specify in which market direction the strategy.entry function is
    allowed to open positions.

    :param value: The allowed direction
    """
    if lib._script is None or lib._script.position is None:
        return

    lib._script.position.risk_allowed_direction = value


# noinspection PyProtectedMember
def max_cons_loss_days(count: int, alert_message: PyneStr = NA(str)) -> None:
    """
    The purpose of this rule is to determine the maximum number of consecutive losing days.
    Once the maximum number of consecutive losing days is reached, all pending orders are cancelled,
    all open positions are closed and no new orders can be placed

    :param count: The maximum number of consecutive losing days
    :param alert_message: The alert message
    """
    if lib._script is None or lib._script.position is None:
        return

    lib._script.position.risk_max_cons_loss_days = count
    lib._script.position.risk_max_cons_loss_days_alert = None if isinstance(alert_message, NA) else str(alert_message)


# noinspection PyProtectedMember
def max_intraday_filled_orders(count: int, alert_message: PyneStr = NA(str)) -> None:
    """
    The purpose of this rule is to determine the maximum number of intraday filled orders

    :param count: The maximum number of intraday filled orders
    :param alert_message: The alert message
    """
    if lib._script is None or lib._script.position is None:
        return

    lib._script.position.risk_max_intraday_filled_orders = count
    lib._script.position.risk_max_intraday_filled_orders_alert = (
        None if isinstance(alert_message, NA) else str(alert_message)
    )


# noinspection PyShadowingBuiltins,PyProtectedMember
def max_intraday_loss(value: float | int, type: strategy.QtyType = strategy.percent_of_equity,
                      alert_message: PyneStr = NA(str)) -> None:
    """
    The purpose of this rule is to determine the maximum intraday loss. The rule affects the whole strategy.
    Once the maximum intraday loss value is reached, all pending orders are cancelled, all open positions
    are closed and no new orders can be placed

    :param value: The maximum intraday loss value
    :param type: The type of the value
    :param alert_message: The alert message
    """
    if lib._script is None or lib._script.position is None:
        return

    lib._script.position.risk_max_intraday_loss_value = value
    lib._script.position.risk_max_intraday_loss_type = type
    lib._script.position.risk_max_intraday_loss_alert = None if isinstance(alert_message, NA) else str(alert_message)


# noinspection PyProtectedMember
def max_position_size(contracts: int | float):
    """
    The purpose of this rule is to determine maximum size of a market position

    :param contracts: The maximum size of a market position
    """
    if lib._script is None or lib._script.position is None:
        return

    lib._script.position.risk_max_position_size = abs(contracts)
