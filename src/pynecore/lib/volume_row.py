from ..types.volume_row import VolumeRow


# noinspection PyShadowingBuiltins
def up_price(id: VolumeRow) -> float:
    """
    Upper price boundary of the row.

    :param id: VolumeRow object
    :return: Upper price
    """
    return id.up_price()


# noinspection PyShadowingBuiltins
def down_price(id: VolumeRow) -> float:
    """
    Lower price boundary of the row.

    :param id: VolumeRow object
    :return: Lower price
    """
    return id.down_price()


# noinspection PyShadowingBuiltins
def buy_volume(id: VolumeRow) -> float:
    """
    Buy volume for this row.

    :param id: VolumeRow object
    :return: Buy volume
    """
    return id.buy_volume()


# noinspection PyShadowingBuiltins
def sell_volume(id: VolumeRow) -> float:
    """
    Sell volume for this row.

    :param id: VolumeRow object
    :return: Sell volume
    """
    return id.sell_volume()


# noinspection PyShadowingBuiltins
def delta(id: VolumeRow) -> float:
    """
    Volume delta (buy - sell) for this row.

    :param id: VolumeRow object
    :return: Volume delta
    """
    return id.delta()


# noinspection PyShadowingBuiltins
def total_volume(id: VolumeRow) -> float:
    """
    Total volume for this row.

    :param id: VolumeRow object
    :return: Total volume
    """
    return id.total_volume()
