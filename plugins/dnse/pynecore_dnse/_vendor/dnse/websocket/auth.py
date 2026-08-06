import hmac
import hashlib
import time
from typing import Dict, Any


class AuthManager:
    """
    HMAC-SHA256 authentication manager.

    Handles signature generation and nonce creation for WebSocket authentication.
    """

    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize auth manager.

        Args:
            api_key: API key
            api_secret: API secret for HMAC signature
        """
        self.api_key = api_key
        self.api_secret = api_secret

    def create_auth_message(self) -> Dict[str, Any]:
        timestamp = int(time.time())  # second
        nonce = str(int(time.time() * 1000000))  # microseconds for uniqueness

        signature = self.compute_signature(timestamp, nonce)

        return {
            "action": "auth",
            "api_key": self.api_key,
            "signature": signature,
            "timestamp": timestamp,
            "nonce": nonce
        }

    def compute_signature(self, timestamp: int, nonce: str) -> str:
        """
        Compute HMAC-SHA256 signature.

        Args:
            timestamp: Unix timestamp in second
            nonce: Unique nonce string

        Returns:
            Hex-encoded HMAC signature
        """
        # Message format: {api_key}:{timestamp}:{nonce}
        message = f"{self.api_key}:{timestamp}:{nonce}"

        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        return signature
