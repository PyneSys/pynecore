from dataclasses import dataclass

from .volume_row import VolumeRow
from ..codes import NotAvailableError


@dataclass(slots=True, eq=False)
class Footprint:
    """
    Stores volume footprint data for a bar, including total buy/sell volume,
    delta, and key price level rows (VAH, VAL, POC).

    Obtained via ``request.footprint()``.
    """

    def buy_volume(self) -> float:
        """Total buy volume for the bar."""
        raise NotAvailableError(function='footprint.buy_volume')

    def sell_volume(self) -> float:
        """Total sell volume for the bar."""
        raise NotAvailableError(function='footprint.sell_volume')

    def delta(self) -> float:
        """Volume delta (buy - sell) for the bar."""
        raise NotAvailableError(function='footprint.delta')

    def total_volume(self) -> float:
        """Total volume (buy + sell) for the bar."""
        raise NotAvailableError(function='footprint.total_volume')

    def vah(self) -> VolumeRow:
        """Value Area High row."""
        raise NotAvailableError(function='footprint.vah')

    def val(self) -> VolumeRow:
        """Value Area Low row."""
        raise NotAvailableError(function='footprint.val')

    def poc(self) -> VolumeRow:
        """Point of Control row."""
        raise NotAvailableError(function='footprint.poc')
