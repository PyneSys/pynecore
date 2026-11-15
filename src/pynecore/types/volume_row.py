from dataclasses import dataclass

from ..codes import NotAvailableError


@dataclass(slots=True, eq=False)
class VolumeRow:
    """
    Represents a single price row within a footprint bar. Contains volume data
    for a specific price level, including buy/sell volume and price boundaries.

    Returned by footprint methods such as ``footprint.vah()``, ``footprint.val()``,
    and ``footprint.poc()``.
    """

    def up_price(self) -> float:
        """Upper price boundary of the row."""
        raise NotAvailableError(function='volume_row.up_price')

    def down_price(self) -> float:
        """Lower price boundary of the row."""
        raise NotAvailableError(function='volume_row.down_price')

    def buy_volume(self) -> float:
        """Buy volume for this row."""
        raise NotAvailableError(function='volume_row.buy_volume')

    def sell_volume(self) -> float:
        """Sell volume for this row."""
        raise NotAvailableError(function='volume_row.sell_volume')

    def delta(self) -> float:
        """Volume delta (buy - sell) for this row."""
        raise NotAvailableError(function='volume_row.delta')

    def total_volume(self) -> float:
        """Total volume for this row."""
        raise NotAvailableError(function='volume_row.total_volume')
