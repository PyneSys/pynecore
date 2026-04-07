from ..types.footprint import Footprint
from ..types.volume_row import VolumeRow


# noinspection PyShadowingBuiltins
def buy_volume(id: Footprint) -> float:
    """
    Total buy volume for the bar.

    :param id: Footprint object
    :return: Buy volume
    """
    return id.buy_volume()


# noinspection PyShadowingBuiltins
def sell_volume(id: Footprint) -> float:
    """
    Total sell volume for the bar.

    :param id: Footprint object
    :return: Sell volume
    """
    return id.sell_volume()


# noinspection PyShadowingBuiltins
def delta(id: Footprint) -> float:
    """
    Volume delta (buy - sell) for the bar.

    :param id: Footprint object
    :return: Volume delta
    """
    return id.delta()


# noinspection PyShadowingBuiltins
def vah(id: Footprint) -> VolumeRow:
    """
    Value Area High row.

    :param id: Footprint object
    :return: VolumeRow for VAH
    """
    return id.vah()


# noinspection PyShadowingBuiltins
def val(id: Footprint) -> VolumeRow:
    """
    Value Area Low row.

    :param id: Footprint object
    :return: VolumeRow for VAL
    """
    return id.val()


# noinspection PyShadowingBuiltins
def poc(id: Footprint) -> VolumeRow:
    """
    Point of Control row.

    :param id: Footprint object
    :return: VolumeRow for POC
    """
    return id.poc()
