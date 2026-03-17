from ..types.footprint import Footprint
from ..types.volume_row import VolumeRow


def buy_volume(id: Footprint) -> float:
    """
    Total buy volume for the bar.

    :param id: Footprint object
    :return: Buy volume
    """
    return id.buy_volume()


def sell_volume(id: Footprint) -> float:
    """
    Total sell volume for the bar.

    :param id: Footprint object
    :return: Sell volume
    """
    return id.sell_volume()


def delta(id: Footprint) -> float:
    """
    Volume delta (buy - sell) for the bar.

    :param id: Footprint object
    :return: Volume delta
    """
    return id.delta()


def vah(id: Footprint) -> VolumeRow:
    """
    Value Area High row.

    :param id: Footprint object
    :return: VolumeRow for VAH
    """
    return id.vah()


def val(id: Footprint) -> VolumeRow:
    """
    Value Area Low row.

    :param id: Footprint object
    :return: VolumeRow for VAL
    """
    return id.val()


def poc(id: Footprint) -> VolumeRow:
    """
    Point of Control row.

    :param id: Footprint object
    :return: VolumeRow for POC
    """
    return id.poc()
