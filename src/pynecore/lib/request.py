from ..types.footprint import Footprint


# noinspection PyUnusedLocal
def security(*args, **kwargs):
    """
    Request data from another symbol/timeframe.

    :raises NotImplementedError: Not yet implemented in PyneCore
    """
    raise NotImplementedError("request.security() is not yet implemented in PyneCore")


# noinspection PyUnusedLocal
def security_lower_tf(*args, **kwargs):
    """
    Request intrabar data from a lower timeframe.

    :raises NotImplementedError: Not yet implemented in PyneCore
    """
    raise NotImplementedError("request.security_lower_tf() is not yet implemented in PyneCore")


# noinspection PyUnusedLocal
def currency_rate(*args, **kwargs) -> float:
    """
    Get the currency conversion rate.

    :raises NotImplementedError: Not yet implemented in PyneCore
    """
    raise NotImplementedError("request.currency_rate() is not yet implemented in PyneCore")


# noinspection PyUnusedLocal
def dividends(*args, **kwargs) -> float:
    """
    Request dividend data for a symbol.

    :raises NotImplementedError: Not yet implemented in PyneCore
    """
    raise NotImplementedError("request.dividends() is not yet implemented in PyneCore")


# noinspection PyUnusedLocal
def splits(*args, **kwargs) -> float:
    """
    Request stock split data for a symbol.

    :raises NotImplementedError: Not yet implemented in PyneCore
    """
    raise NotImplementedError("request.splits() is not yet implemented in PyneCore")


# noinspection PyUnusedLocal
def earnings(*args, **kwargs) -> float:
    """
    Request earnings data for a symbol.

    :raises NotImplementedError: Not yet implemented in PyneCore
    """
    raise NotImplementedError("request.earnings() is not yet implemented in PyneCore")


# noinspection PyUnusedLocal
def financial(*args, **kwargs) -> float:
    """
    Request financial data from FactSet.

    :raises NotImplementedError: Not yet implemented in PyneCore
    """
    raise NotImplementedError("request.financial() is not yet implemented in PyneCore")


# noinspection PyUnusedLocal
def economic(*args, **kwargs) -> float:
    """
    Request economic data.

    :raises NotImplementedError: Not yet implemented in PyneCore
    """
    raise NotImplementedError("request.economic() is not yet implemented in PyneCore")


# noinspection PyUnusedLocal
def quandl(*args, **kwargs) -> float:
    """
    Request data from Quandl/Nasdaq.

    :raises NotImplementedError: Not yet implemented in PyneCore
    """
    raise NotImplementedError("request.quandl() is not yet implemented in PyneCore")


# noinspection PyUnusedLocal
def seed(*args, **kwargs):
    """
    Request data from user-maintained GitHub repositories.

    :raises NotImplementedError: Not yet implemented in PyneCore
    """
    raise NotImplementedError("request.seed() is not yet implemented in PyneCore")


# noinspection PyUnusedLocal
def footprint(ticks_per_row: int, va_percent: int) -> Footprint:
    """
    Request volume footprint data for the current bar.

    :param ticks_per_row: Number of ticks per footprint row
    :param va_percent: Value Area percentage
    :return: Footprint object with volume data
    :raises NotImplementedError: Not yet implemented in PyneCore
    """
    raise NotImplementedError("request.footprint() is not yet implemented in PyneCore")
