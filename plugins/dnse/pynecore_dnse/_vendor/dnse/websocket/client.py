"""
TradingClient - High-level async WebSocket client for real-time trading data.

This module provides the main TradingClient class that handles:
- WebSocket connection management with automatic reconnection
- HMAC authentication
- Channel subscriptions (market data and private channels)
- Event-driven message handling
- Heartbeat monitoring
- Graceful shutdown
"""

import asyncio
import logging
import time
from typing import Optional, Callable, List, Dict, Any

from .auth import AuthManager
from .connection import WebSocketConnection
from .encoding import MessageEncoder, MessageDecoder
from .exceptions import (
    AuthenticationError,
    ConnectionError,
    SubscriptionError,
    ConnectionClosed,
)
from .models import Trade, Quote, Ohlc, Order, AccountUpdate, ExpectedPrice, SecurityDefinition, TradeExtra, \
    MarketIndex, ForeignInvestor, Position, EstimatedMarketIndex, Session

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

DEFAULT_BOARDS = ["G1", "G3", "G4", "G7", "T1", "T2", "T3", "T4", "T6"]

_MSG_TYPE_MAP = {
    "t": ("trade", Trade, None),
    "te": ("trade_extra", TradeExtra, None),
    "e": ("expected_price", ExpectedPrice, None),
    "sd": ("security_definition", SecurityDefinition, None),
    "q": ("quote", Quote, None),
    "b": ("ohlc", Ohlc, None),
    "bc": ("ohlc_closed", Ohlc, None),
    "do": ("order_event", Order, "order"),
    "eo": ("order_event", Order, "order"),
    "dp": ("position_event", Position, "position"),
    "ep": ("position_event", Position, "position"),
    "mi": ("market_index", MarketIndex, None),
    "emi": ("estimated_market_index", EstimatedMarketIndex, "marketIndex"),
    "a": ("account", AccountUpdate, None),
    "f": ("foreign", ForeignInvestor, None),
    "s": ("session", Session, None),
}


class TradingClient:
    """
    Async WebSocket client for real-time trading data.

    Features:
    - Automatic reconnection with re-authentication and re-subscription
    - HMAC-SHA256 authentication
    - Support for JSON and MessagePack encoding
    - Event-driven architecture with callback handlers
    - Heartbeat monitoring with pong tracking
    - Context manager support (async with)
    - Async iterator support (async for)
    """

    def __init__(
            self,
            api_key: str,
            api_secret: str,
            base_url: str = "wss://ws-openapi.dnse.com.vn",
            encoding: str = "json",  # or "json"
            auto_reconnect: bool = True,
            max_retries: int = 10,
            heartbeat_interval: float = 25.0,
            timeout: float = 60.0,
    ):
        """
        Initialize trading client.

        Args:
            api_key: API key for authentication
            api_secret: API secret for HMAC signature
            base_url: WebSocket gateway URL
            encoding: Message encoding ("json" or "msgpack")
            auto_reconnect: Enable automatic reconnection
            max_retries: Maximum reconnection attempts
            heartbeat_interval: Seconds between heartbeat pings
            timeout: Connection timeout in seconds
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.encoding = encoding
        self.auto_reconnect = auto_reconnect
        self.max_retries = max_retries
        self.heartbeat_interval = heartbeat_interval
        self.timeout = timeout

        # Internal state
        self._connection: Optional[WebSocketConnection] = None
        self._auth_manager = AuthManager(api_key, api_secret)
        self._encoder = MessageEncoder(encoding)
        self._decoder = MessageDecoder(encoding)
        self._event_handlers: Dict[str, List[Callable]] = {}
        self._subscriptions: Dict[str, Dict[str, Any]] = {}
        self._is_authenticated = False
        self._session_id: Optional[str] = None
        # Per-event queues: only created when needed (async iterator usage)
        self._queues: Dict[str, asyncio.Queue] = {}
        # Internal dispatch queues: 1 queue per worker, symbol hashed to worker
        self._dispatch_queues: List[asyncio.Queue] = []
        self._is_running = False
        self._last_pong_time: float = 0.0
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._message_handler_task: Optional[asyncio.Task] = None
        self._dispatch_worker_tasks: List[asyncio.Task] = []
        self._num_workers: int = 6  # 1 worker per symbol slot

    async def connect(self) -> None:
        """
        Establish WebSocket connection and authenticate.

        Raises:
            ConnectionError: Failed to connect
            AuthenticationError: Authentication failed
            asyncio.TimeoutError: Connection timeout
        """
        url = f"{self.base_url}/v1/stream?encoding={self.encoding}"

        logger.info(f"Connecting to {url}")

        self._connection = WebSocketConnection(
            url=url,
            timeout=self.timeout,
            heartbeat_interval=self.heartbeat_interval,
            auto_reconnect=self.auto_reconnect,
            max_retries=self.max_retries,
        )

        await self._connection.connect()

        # Wait for welcome message
        welcome = await asyncio.wait_for(
            self._connection.receive(), timeout=self.timeout
        )

        welcome_data = self._decoder.decode(welcome)
        self._session_id = welcome_data.get("session_id") or welcome_data.get("sid")

        logger.info(f"Connected! Session ID: {self._session_id}")

        # Authenticate
        await self._authenticate()

        # Start background tasks
        self._is_running = True
        self._last_pong_time = time.time()

        # Init 1 queue per worker slot
        self._dispatch_queues = [asyncio.Queue() for _ in range(self._num_workers)]

        # Start dispatch workers (1 per slot, same symbol → same worker → ordered)
        self._dispatch_worker_tasks = [
            asyncio.create_task(self._dispatch_worker(i))
            for i in range(self._num_workers)
        ]

        # Start message handler (receive + decode only)
        self._message_handler_task = asyncio.create_task(self._message_handler())

        # Start heartbeat
        if self.heartbeat_interval > 0:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _authenticate(self) -> None:
        """
        Perform HMAC authentication.

        Raises:
            AuthenticationError: Authentication failed
            asyncio.TimeoutError: Authentication timeout
        """
        auth_msg = self._auth_manager.create_auth_message()
        encoded = self._encoder.encode(auth_msg)

        await self._connection.send(encoded)

        # Wait for auth response
        response = await asyncio.wait_for(
            self._connection.receive(), timeout=self.timeout
        )

        data = self._decoder.decode(response)
        action = data.get("action") or data.get("a")

        if action == "auth_success":
            self._is_authenticated = True
            logger.info("Authentication successful")
        elif action in ("auth_error", "error"):
            error_msg = data.get("message") or data.get("msg", "Unknown error")
            raise AuthenticationError(f"Authentication failed: {error_msg}")
        else:
            raise AuthenticationError(f"Unexpected response: {action}")

    async def subscribe_trades(
            self, symbols: List[str], on_trade: Optional[Callable[[Trade], None]] = None, encoding="json", board_id=None
    ) -> None:
        boards = [board_id] if board_id is not None else DEFAULT_BOARDS

        for board in boards:
            channel = f"tick.{board}.json"
            if encoding == "msgpack":
                channel = f"tick.{board}.msgpack"
            await self._subscribe_channel(channel, symbols)

        if on_trade:
            _handler = self._make_filtered_handler(board_id, "boardId", on_trade) if board_id is not None else on_trade
            self.on("trade", _handler)

    async def subscribe_trade_extra(
            self, symbols: List[str], on_trade_extra: Optional[Callable[[TradeExtra], None]] = None, encoding="json",
            board_id=None
    ) -> None:
        boards = [board_id] if board_id is not None else DEFAULT_BOARDS

        for board in boards:
            channel = f"tick_extra.{board}.json"
            if encoding == "msgpack":
                channel = f"tick_extra.{board}.msgpack"
            await self._subscribe_channel(channel, symbols)

        if on_trade_extra:
            _handler = self._make_filtered_handler(board_id, "boardId",
                                                   on_trade_extra) if board_id is not None else on_trade_extra
            self.on("trade_extra", _handler)

    async def subscribe_expected_price(
            self, symbols: List[str], on_expected_price: Optional[Callable[[ExpectedPrice], None]] = None,
            encoding="json", board_id=None
    ) -> None:
        boards = [board_id] if board_id is not None else DEFAULT_BOARDS

        for board in boards:
            channel = f"expected_price.{board}.json"
            if encoding == "msgpack":
                channel = f"expected_price.{board}.msgpack"
            await self._subscribe_channel(channel, symbols)

        if on_expected_price:
            _handler = self._make_filtered_handler(board_id, "boardId",
                                                   on_expected_price) if board_id is not None else on_expected_price
            self.on("expected_price", _handler)

    async def subscribe_order_event(
            self, market_type="STOCK",
            on_order_event: Optional[Callable[[Order], None]] = None,
            encoding="json"
    ) -> None:
        channel = f"order.{market_type}.{encoding}"
        await self._subscribe_channel(channel, [])

        if on_order_event:
            self.on("order_event", on_order_event)

    async def subscribe_broker_order_event(
            self,
            investor_id: str,
            market_type="STOCK",
            on_order_event: Optional[Callable[[Order], None]] = None,
            encoding="json"
    ) -> None:
        channel = f"order.broker.{market_type}.{investor_id}.{encoding}"
        await self._subscribe_channel(channel, [])

        if on_order_event:
            self.on("order_event", on_order_event)

    async def subscribe_position_event(
            self, market_type="STOCK",
            on_position_event: Optional[Callable[[Position], None]] = None,
            encoding="json"
    ) -> None:
        channel = f"position.{market_type}.{encoding}"
        await self._subscribe_channel(channel, [])

        if on_position_event:
            self.on("position_event", on_position_event)

    async def subscribe_broker_position_event(
            self,
            investor_id: str,
            market_type="STOCK",
            on_position_event: Optional[Callable[[Position], None]] = None,
            encoding="json"
    ) -> None:
        channel = f"position.broker.{market_type}.{investor_id}.{encoding}"
        await self._subscribe_channel(channel, [])

        if on_position_event:
            self.on("position_event", on_position_event)

    async def subscribe_sec_def(
            self, symbols: List[str], on_sec_def: Optional[Callable[[SecurityDefinition], None]] = None,
            encoding="json", board_id=None
    ) -> None:
        boards = [board_id] if board_id is not None else DEFAULT_BOARDS

        for board in boards:
            channel = f"security_definition.{board}.json"
            if encoding == "msgpack":
                channel = f"security_definition.{board}.msgpack"
            await self._subscribe_channel(channel, symbols)

        if on_sec_def:
            _handler = self._make_filtered_handler(board_id, "boardId",
                                                   on_sec_def) if board_id is not None else on_sec_def
            self.on("security_definition", _handler)

    async def subscribe_market_index(
            self, market_index: str, on_market_index: Optional[Callable[[MarketIndex], None]] = None, encoding="json"
    ) -> None:
        channel = f"market_index.{market_index}.json"
        if encoding == "msgpack":
            channel = f"market_index.{market_index}.msgpack"
        await self._subscribe_channel(channel, [])

        if on_market_index:
            self.on("market_index", on_market_index)

    async def subscribe_estimated_market_index(
            self, estimated_market_index: str,
            on_estimated_market_index: Optional[Callable[[EstimatedMarketIndex], None]] = None, encoding="json"
    ) -> None:
        channel = f"estimated_market_index.{estimated_market_index}.json"
        if encoding == "msgpack":
            channel = f"estimated_market_index.{estimated_market_index}.msgpack"
        await self._subscribe_channel(channel, [])

        if on_estimated_market_index:
            self.on("estimated_market_index", on_estimated_market_index)

    async def subscribe_quotes(
            self, symbols: List[str], on_quote: Optional[Callable[[Quote], None]] = None, encoding="json", board_id=None
    ) -> None:
        boards = [board_id] if board_id is not None else ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]

        for board in boards:
            channel = f"top_price.{board}.json"
            if encoding == "msgpack":
                channel = f"top_price.{board}.msgpack"
            await self._subscribe_channel(channel, symbols)

        if on_quote:
            _handler = self._make_filtered_handler(board_id, "boardId", on_quote) if board_id is not None else on_quote
            self.on("quote", _handler)

    async def subscribe_foreign_trading(
            self, symbols: List[str], board_id: str = "*",
            on_trade: Optional[Callable[[ForeignInvestor], None]] = None,
            encoding="json"
    ) -> None:
        ext = "msgpack" if encoding == "msgpack" else "json"
        channel = f"foreign.{board_id}.{ext}"
        await self._subscribe_channel(channel, symbols)

        if on_trade:
            self.on("foreign", on_trade)

    async def subscribe_ohlc(
            self,
            symbols: List[str],
            resolution: Optional[str] = None,
            on_ohlc: Optional[Callable[[Ohlc], None]] = None, encoding="json"
    ) -> None:
        # If resolution is None, subscribe to all resolutions
        if resolution is None:
            all_resolutions = ["1", "3", "5", "15", "30", "1H", "1D", "1W"]
            for res in all_resolutions:
                channel = "ohlc." + res + ".json"
                if encoding == "msgpack":
                    channel = "ohlc." + res + ".msgpack"
                await self._subscribe_channel(channel, symbols)
        else:
            channel = "ohlc." + resolution + ".json"
            if encoding == "msgpack":
                channel = "ohlc." + resolution + ".msgpack"
            await self._subscribe_channel(channel, symbols)

        if on_ohlc:
            self.on("ohlc", on_ohlc)

    async def subscribe_session(
            self,
            product_group_id: str,
            board_id: str = "*",
            on_session: Optional[Callable[[Session], None]] = None, encoding="json"
    ) -> None:
        channel = f"session.{product_group_id}.{board_id}.json"

        if encoding == "msgpack":
            channel = f"session.{product_group_id}.{board_id}.msgpack"
        await self._subscribe_channel(channel, [])

        if on_session:
            self.on("session", on_session)

    async def subscribe_ohlc_closed(
            self,
            symbols: List[str],
            resolution: Optional[str] = None,
            on_ohlc: Optional[Callable[[Ohlc], None]] = None, encoding="json"
    ) -> None:
        # If resolution is None, subscribe to all resolutions
        if resolution is None:
            all_resolutions = ["1", "3", "5", "15", "30", "1H", "1D", "1W"]
            for res in all_resolutions:
                channel = "ohlc_closed." + res + ".json"
                if encoding == "msgpack":
                    channel = "ohlc_closed." + res + ".msgpack"
                await self._subscribe_channel(channel, symbols)
        else:
            channel = "ohlc_closed." + resolution + ".json"
            if encoding == "msgpack":
                channel = "ohlc_closed." + resolution + ".msgpack"
            await self._subscribe_channel(channel, symbols)

        if on_ohlc:
            self.on("ohlc_closed", on_ohlc)

    async def subscribe_orders(
            self, on_order: Optional[Callable[[Order], None]] = None
    ) -> None:
        await self._subscribe_channel("orders", [])

        if on_order:
            self.on("order", on_order)

    async def subscribe_positions(
            self, on_position: Optional[Callable[[Position], None]] = None
    ) -> None:

        await self._subscribe_channel("positions", [])

        if on_position:
            self.on("position", on_position)

    async def subscribe_account(
            self, on_account: Optional[Callable[[AccountUpdate], None]] = None
    ) -> None:
        await self._subscribe_channel("account", [])

        if on_account:
            self.on("account", on_account)

    async def _subscribe_channel(
            self, channel: str, symbols: List[str], **kwargs
    ) -> None:
        if not self._is_authenticated:
            raise SubscriptionError("Must authenticate before subscribing")

        subscribe_msg = {
            "action": "subscribe",
            "channels": [{"name": channel, "symbols": symbols, **kwargs}],
        }

        encoded = self._encoder.encode(subscribe_msg)
        await self._connection.send(encoded)

        # Store subscription for reconnection
        self._subscriptions[channel] = {"symbols": symbols, "kwargs": kwargs}

        logger.info(f"Subscribed to {channel}: {symbols}")

    async def unsubscribe(self, channel: str, symbols: List[str]) -> None:
        unsubscribe_msg = {
            "action": "unsubscribe",
            "channels": [{"name": channel, "symbols": symbols}],
        }

        encoded = self._encoder.encode(unsubscribe_msg)
        await self._connection.send(encoded)

        # Update local state
        if channel in self._subscriptions:
            stored_symbols = self._subscriptions[channel].get("symbols", [])
            for symbol in symbols:
                if symbol in stored_symbols:
                    stored_symbols.remove(symbol)

            # Remove channel if no symbols left
            if not stored_symbols:
                del self._subscriptions[channel]

        logger.info(f"Unsubscribed from {channel}: {symbols}")

    def _make_filtered_handler(self, board_id: Optional[str], attr: str, handler: Callable) -> Callable:
        """Wrap handler to only fire when obj.{attr} == board_id."""

        def _filtered(obj, _b=board_id, _a=attr, _cb=handler):
            if getattr(obj, _a, None) == _b:
                _cb(obj)

        return _filtered

    def on(self, event: str, handler: Callable) -> None:
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)

    async def _message_handler(self) -> None:
        reconnect_attempt = 0
        max_reconnect_delay = 60

        while self._is_running:
            try:
                async for message in self._connection:
                    data = self._decoder.decode(message)
                    data["_receivedAt"] = time.time()

                    # Hash symbol → worker index để đảm bảo cùng 1 mã
                    symbol = data.get("Symbol") or data.get("symbol") or ""
                    worker_idx = hash(symbol) % self._num_workers
                    await self._dispatch_queues[worker_idx].put(data)

                    reconnect_attempt = 0
            except ConnectionClosed as e:
                logger.warning(f"Connection closed: {e}")

                # Handle reconnection if enabled and error is recoverable
                if self.auto_reconnect and e.recoverable:
                    reconnect_attempt += 1
                    logger.info(f"Attempting to reconnect (attempt {reconnect_attempt}/{self.max_retries})...")

                    # Emit reconnecting event
                    self._emit("reconnecting", {
                        "attempt": reconnect_attempt,
                        "max_retries": self.max_retries,
                        "delay": 0,
                        "error": str(e),
                    })

                    try:
                        await self._handle_reconnection()
                        reconnect_attempt = 0
                    except Exception as reconnect_error:
                        logger.error(f"Reconnection failed: {reconnect_error}")
                        self._emit("error", reconnect_error)
                        break
                else:
                    self._emit("error", e)
                    break
            except Exception as e:
                logger.error(f"Message handler error: {e}")
                self._emit("error", e)

                if not self.auto_reconnect:
                    break

                if self._is_connection_error(e):
                    reconnect_attempt += 1

                    if reconnect_attempt > self.max_retries:
                        logger.error(f"Max reconnection attempts ({self.max_retries}) exceeded")
                        self._emit("max_reconnect_exceeded", reconnect_attempt)
                        break

                    delay = min(2 ** (reconnect_attempt - 1), max_reconnect_delay)
                    logger.info(
                        f"Connection error detected. Reconnecting in {delay}s (attempt {reconnect_attempt}/{self.max_retries})...")

                    self._emit("reconnecting", {
                        "attempt": reconnect_attempt,
                        "max_retries": self.max_retries,
                        "delay": delay,
                        "error": str(e),
                    })

                    await asyncio.sleep(delay)

                    try:
                        await self._handle_reconnection()
                        logger.info("Reconnection successful after connection error")
                        reconnect_attempt = 0
                    except Exception as reconnect_error:
                        logger.error(f"Reconnection failed: {reconnect_error}")
                        continue
                else:
                    logger.error(f"Non-recoverable error: {e}")
                    break

    async def _dispatch_worker(self, worker_idx: int) -> None:
        """
        Each worker owns 1 queue. Symbol is hashed to worker_idx,
        so same symbol always processed by same worker → message order guaranteed.
        """
        q = self._dispatch_queues[worker_idx]
        while self._is_running:
            try:
                data = await asyncio.wait_for(q.get(), timeout=1.0)
                await self._dispatch_message(data)
                q.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Dispatch worker[{worker_idx}] error: {e}")

    def _is_connection_error(self, error: Exception) -> bool:
        """
        Check if an exception is related to connection issues.

        Args:
            error: The exception to check

        Returns:
            True if the error is connection-related and potentially recoverable
        """
        import websockets.exceptions

        # Connection-related exception types
        connection_error_types = (
            ConnectionError,  # Our custom ConnectionError
            OSError,  # Network-level errors (includes socket errors)
            ConnectionResetError,
            ConnectionRefusedError,
            ConnectionAbortedError,
            BrokenPipeError,
            TimeoutError,
            asyncio.TimeoutError,
            websockets.exceptions.WebSocketException,
            websockets.exceptions.ConnectionClosed,
            websockets.exceptions.ConnectionClosedError,
            websockets.exceptions.ConnectionClosedOK,
        )

        if isinstance(error, connection_error_types):
            return True

        # Check error message for common connection-related patterns
        error_msg = str(error).lower()
        connection_keywords = [
            'connection',
            'network',
            'socket',
            'timeout',
            'reset',
            'refused',
            'closed',
            'broken pipe',
            'eof',
            'disconnect',
        ]

        return any(keyword in error_msg for keyword in connection_keywords)

    async def _dispatch_message(self, data: Dict[str, Any]) -> None:
        action = data.get("action") or data.get("a")
        msg_type = data.get("T")

        if action == "subscribed":
            logger.debug(f"Subscription confirmed: {data}")
        elif action == "ping":
            logger.info("Received ping from server, sending pong")
            await self._connection.send(self._encoder.encode({"action": "pong"}))
        elif action == "pong":
            self._last_pong_time = time.time()
        elif action == "error":
            error_msg = data.get("message") or data.get("msg")
            logger.error(f"Server error: {error_msg}")
            self._emit("error", Exception(error_msg))
        elif msg_type in _MSG_TYPE_MAP:
            event, model_cls, field = _MSG_TYPE_MAP[msg_type]
            if field is not None and field != "":
                obj = model_cls.from_dict(data[field])
                obj.receivedAt = data["_receivedAt"]
            else:
                obj = model_cls.from_dict(data)
            self._emit(event, obj)
            # Only push to queue if no callback registered (using async iterator)
            if event not in self._event_handlers:
                if event in self._queues:
                    await self._queues[event].put(obj)
                if "*" in self._queues:
                    await self._queues["*"].put(obj)

    def _emit(self, event: str, data: Any) -> None:
        if event not in self._event_handlers:
            return
        for handler in self._event_handlers[event]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    asyncio.ensure_future(handler(data))
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Handler error for {event}: {e}")

    def queue(self, event: str) -> asyncio.Queue:
        """
        Get (or create) a dedicated queue for a specific event type.
        Use this when consuming messages via async iterator instead of callbacks.

        Example:
            q = client.queue("quote")
            while True:
                quote = await q.get()
        """
        if event not in self._queues:
            self._queues[event] = asyncio.Queue()
        return self._queues[event]

    async def _heartbeat_loop(self) -> None:
        while self._is_running and self._connection and self._connection.is_connected:
            try:
                ping_msg = self._encoder.encode({"action": "ping"})
                await self._connection.send(ping_msg)
                logger.debug("Sent heartbeat ping")
                await asyncio.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                break

    async def _handle_reconnection(self) -> None:
        """
        Handle reconnection after connection drop.

        Performs:
        1. Re-establish connection
        2. Re-authenticate
        3. Re-subscribe to all previous channels
        4. Emit reconnected event
        """
        logger.info("Starting reconnection process...")

        # Store subscriptions before reconnect
        previous_subscriptions = self._subscriptions.copy()

        # Reset authentication state
        self._is_authenticated = False

        # Reconnect
        await self._connection.connect()

        # Wait for welcome message
        welcome = await asyncio.wait_for(
            self._connection.receive(), timeout=self.timeout
        )

        welcome_data = self._decoder.decode(welcome)
        self._session_id = welcome_data.get("session_id") or welcome_data.get("sid")

        logger.info(f"Reconnected! New Session ID: {self._session_id}")

        # Re-authenticate
        await self._authenticate()

        # Re-subscribe to all previous channels
        for channel, sub_data in previous_subscriptions.items():
            symbols = sub_data.get("symbols", [])
            kwargs = sub_data.get("kwargs", {})
            await self._subscribe_channel(channel, symbols, **kwargs)
            logger.info(f"Re-subscribed to {channel}: {symbols}")

        # Update pong time
        self._last_pong_time = time.time()

        # Emit reconnected event
        self._emit("reconnected", {"session_id": self._session_id})
        logger.info("Reconnection complete")

    @property
    def is_healthy(self) -> bool:
        if not self._connection or not self._connection.is_connected:
            return False

        if not self._is_authenticated:
            return False

        # Check if we received a pong recently
        if self.heartbeat_interval > 0:
            time_since_pong = time.time() - self._last_pong_time
            max_pong_delay = self.heartbeat_interval * 2
            if time_since_pong > max_pong_delay:
                logger.warning(
                    f"No pong received for {time_since_pong:.1f}s (max: {max_pong_delay:.1f}s)"
                )
                return False

        return True

    async def disconnect(self) -> None:
        """
        Close connection gracefully.

        Stops all background tasks and closes the WebSocket connection.
        """
        logger.info("Disconnecting...")

        # Stop background tasks
        self._is_running = False

        # Cancel dispatch workers
        for worker in self._dispatch_worker_tasks:
            if not worker.done():
                worker.cancel()
                try:
                    await worker
                except asyncio.CancelledError:
                    pass
        self._dispatch_worker_tasks.clear()

        # Cancel tasks
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._message_handler_task and not self._message_handler_task.done():
            self._message_handler_task.cancel()
            try:
                await self._message_handler_task
            except asyncio.CancelledError:
                pass

        # Close connection
        if self._connection:
            await self._connection.close()

        self._is_authenticated = False
        logger.info("Disconnected")

    async def __aenter__(self):
        """Context manager support - connect on entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup - disconnect on exit."""
        await self.disconnect()

    def __aiter__(self):
        """Allow async iteration over all messages via a shared queue."""
        # Ensure a general queue exists for mixed iteration
        self.queue("*")
        return self

    async def __anext__(self):
        q = self._queues.get("*")
        while self._is_running:
            try:
                return await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
        raise StopAsyncIteration
