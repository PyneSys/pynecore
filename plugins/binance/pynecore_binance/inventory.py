"""Binance spot inventory port — the surface
:class:`~pynecore.core.broker.spot_inventory.SpotInventoryManager` drives.

Spot has no venue position object, so core synthesizes the Pine position from
a persisted ledger of THIS BOT's fills. This port feeds that ledger from
Binance ``myTrades`` (id-cursored, ``cursor_scope='product'``) and answers the
balance invariant from ``GET /api/v3/account``.

Attribution: ``myTrades`` rows carry ``orderId`` but not ``clientOrderId``.
The plugin resolves ``orderId -> clientOrderId`` (cached; ``fetch_order`` for
ids it did not place in this process) and a fill counts as ours only when that
client id is one the bot minted. Foreign fills (manual UI trades, other bots)
are excluded — they belong to the frozen baseline, not the ledger.
"""
from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from pynecore.core.broker.spot_inventory import SpotExecution, SpotExecutionBatch

if TYPE_CHECKING:
    from .broker import BinanceBroker

__all__ = ['BinanceSpotPort', 'execution_from_trade']

#: One ``myTrades`` page; Binance allows up to 1000.
_TRADES_PAGE_LIMIT = 500

#: Spot fills settle atomically and balances are exact decimal strings, but a
#: BNB-fee row leaves the base untouched while dust conversion may not — keep
#: zero slack and let the settlement grace absorb in-flight reads.
_SETTLEMENT_GRACE_S = 15.0


def execution_from_trade(trade: dict, *, base_asset: str, quote_asset: str,
                         client_order_id: str) -> SpotExecution:
    """Convert one ccxt unified trade row into a ledger :class:`SpotExecution`.

    Sign rules (validated by ``SpotExecution.__post_init__``): buy → base>0,
    quote<=0; sell → base<0, quote>=0. A base-currency fee subtracts from the
    base delta, a quote-currency fee from the quote delta, a third-currency
    fee (BNB) touches neither.
    """
    side = str(trade.get('side') or '')
    amount = Decimal(str(trade['amount']))
    cost = Decimal(str(trade.get('cost')
                       or amount * Decimal(str(trade['price']))))
    fee = trade.get('fee') or {}
    fee_amount = Decimal(str(fee.get('cost') or 0))
    fee_currency = str(fee.get('currency') or '')

    base_delta = amount if side == 'buy' else -amount
    quote_delta = -cost if side == 'buy' else cost
    if fee_amount:
        if fee_currency == base_asset:
            base_delta -= fee_amount
        elif fee_currency == quote_asset:
            quote_delta -= fee_amount

    trade_id = str(trade['id'])
    return SpotExecution(
        fill_id=trade_id,
        side=side,
        base_delta=base_delta,
        quote_delta=quote_delta,
        price=Decimal(str(trade['price'])),
        fee_amount=fee_amount,
        fee_currency=fee_currency,
        ts_ms=int(trade['timestamp']),
        exchange_order_id=str(trade.get('order') or '') or None,
        client_order_id=client_order_id,
        venue_seq=int(trade_id) if trade_id.isdigit() else None,
    )


class BinanceSpotPort:
    """:class:`SpotInventoryPort` implementation over Binance spot REST."""

    cursor_scope = 'product'
    base_tolerance = Decimal(0)
    settlement_grace_s = _SETTLEMENT_GRACE_S

    def __init__(self, plugin: 'BinanceBroker', market: dict) -> None:
        self._plugin = plugin
        self.product_id = str(market['id'])            # e.g. "BTCUSDT"
        self.base_asset = str(market['base'])
        self.quote_asset = str(market['quote'])
        qty_step = plugin.qty_step
        self.position_dust_threshold = (
            Decimal(str(qty_step)) if qty_step > 0 else Decimal(0))

    async def fetch_executions(self, cursor: str | None) -> SpotExecutionBatch:
        """Bot fills after trade-id ``cursor`` (exclusive), oldest first.

        ``cursor=None`` (first startup) anchors the watermark at the venue's
        newest trade id and returns an EMPTY batch — the account's prior
        history is foreign baseline, not the bot's ledger.
        """
        import asyncio
        plugin = self._plugin
        if cursor is None:
            newest = await asyncio.to_thread(plugin.newest_trade_id)
            return SpotExecutionBatch(next_cursor=str(newest))
        try:
            watermark = int(cursor)
        except ValueError:
            # Unparsable cursor = scope/format drift; fail closed.
            return SpotExecutionBatch(conclusive=False)

        trades = await asyncio.to_thread(
            plugin.fetch_trades_after, watermark, _TRADES_PAGE_LIMIT)
        executions: list[SpotExecution] = []
        last_id = watermark
        for trade in trades:
            last_id = max(last_id, int(trade['id']))
            client_order_id = await asyncio.to_thread(
                plugin.client_order_id_for_trade, trade)
            if not client_order_id:
                continue                                # foreign fill
            executions.append(execution_from_trade(
                trade, base_asset=self.base_asset, quote_asset=self.quote_asset,
                client_order_id=client_order_id))
        return SpotExecutionBatch(
            executions=tuple(executions),
            next_cursor=str(last_id),
            has_more=len(trades) >= _TRADES_PAGE_LIMIT,
        )

    async def fetch_base_balance(self) -> Decimal:
        """TOTAL owned base asset — free + locked in open orders."""
        import asyncio
        return await asyncio.to_thread(
            self._plugin.total_asset_balance, self.base_asset)
