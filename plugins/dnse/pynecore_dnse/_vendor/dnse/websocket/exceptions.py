class TradingWebSocketError(Exception):
    """Base exception for all SDK errors"""
    pass


class ConnectionError(TradingWebSocketError):
    """Failed to establish connection"""
    pass


class ConnectionClosed(TradingWebSocketError):
    """Connection was closed"""

    def __init__(self, message: str, recoverable: bool = False):
        """
        Initialize ConnectionClosed exception.

        Args:
            message: Error message
            recoverable: Whether this is a recoverable error (should retry)
        """
        super().__init__(message)
        self.recoverable = recoverable


class AuthenticationError(TradingWebSocketError):
    """Authentication failed"""
    pass


class SubscriptionError(TradingWebSocketError):
    """Subscription failed"""
    pass


class EncodingError(TradingWebSocketError):
    """Message encoding/decoding failed"""
    pass
