"""Binance spot plugin for PyneCore — data provider + broker on ccxt transport.

Entry points (``pyne.plugin``):

* ``binance``        — :class:`~pynecore_binance.provider.BinanceProvider`
* ``binance_broker`` — :class:`~pynecore_binance.broker.BinanceBroker`

Run: ``pyne run <script>.py binance:BTC/USDT@480 [--live | --broker]``.
The broker refuses mainnet unless ``allow_mainnet = true`` is set explicitly
in ``workdir/config/plugins/binance_broker.toml`` — testnet (``sandbox = true``
with keys from https://testnet.binance.vision) is the default proving ground.
"""
