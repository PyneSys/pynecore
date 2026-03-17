from dataclasses import dataclass

from .volume_row import VolumeRow


@dataclass(slots=True)
class Footprint:
    """
    Stores volume footprint data for a bar, including total buy/sell volume,
    delta, and key price level rows (VAH, VAL, POC).

    Obtained via ``request.footprint()``.
    """

    def buy_volume(self) -> float:
        """Total buy volume for the bar."""
        raise NotImplementedError("Footprint.buy_volume() is not yet implemented in PyneCore")

    def sell_volume(self) -> float:
        """Total sell volume for the bar."""
        raise NotImplementedError("Footprint.sell_volume() is not yet implemented in PyneCore")

    def delta(self) -> float:
        """Volume delta (buy - sell) for the bar."""
        raise NotImplementedError("Footprint.delta() is not yet implemented in PyneCore")

    def vah(self) -> VolumeRow:
        """Value Area High row."""
        raise NotImplementedError("Footprint.vah() is not yet implemented in PyneCore")

    def val(self) -> VolumeRow:
        """Value Area Low row."""
        raise NotImplementedError("Footprint.val() is not yet implemented in PyneCore")

    def poc(self) -> VolumeRow:
        """Point of Control row."""
        raise NotImplementedError("Footprint.poc() is not yet implemented in PyneCore")
