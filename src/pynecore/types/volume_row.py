from dataclasses import dataclass


@dataclass(slots=True)
class VolumeRow:
    """
    Represents a single price row within a footprint bar. Contains volume data
    for a specific price level, including buy/sell volume and price boundaries.

    Returned by footprint methods such as ``footprint.vah()``, ``footprint.val()``,
    and ``footprint.poc()``.
    """

    def up_price(self) -> float:
        """Upper price boundary of the row."""
        raise NotImplementedError("VolumeRow.up_price() is not yet implemented in PyneCore")

    def down_price(self) -> float:
        """Lower price boundary of the row."""
        raise NotImplementedError("VolumeRow.down_price() is not yet implemented in PyneCore")

    def buy_volume(self) -> float:
        """Buy volume for this row."""
        raise NotImplementedError("VolumeRow.buy_volume() is not yet implemented in PyneCore")

    def sell_volume(self) -> float:
        """Sell volume for this row."""
        raise NotImplementedError("VolumeRow.sell_volume() is not yet implemented in PyneCore")

    def delta(self) -> float:
        """Volume delta (buy - sell) for this row."""
        raise NotImplementedError("VolumeRow.delta() is not yet implemented in PyneCore")

    def total_volume(self) -> float:
        """Total volume for this row."""
        raise NotImplementedError("VolumeRow.total_volume() is not yet implemented in PyneCore")
