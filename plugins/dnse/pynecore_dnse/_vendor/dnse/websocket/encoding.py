import json
from typing import Dict, Any
import msgpack
from .exceptions import EncodingError


class MessageEncoder:
    """Encode messages for WebSocket transmission"""

    def __init__(self, encoding: str = "msgpack"):
        """
        Initialize encoder.

        Args:
            encoding: "json" or "msgpack"
        """
        if encoding not in ("json", "msgpack"):
            raise ValueError(f"Invalid encoding: {encoding}. Must be 'json' or 'msgpack'")

        self.encoding = encoding

    def encode(self, data: Dict[str, Any]) -> bytes:
        """
        Encode message.

        Args:
            data: Message dict

        Returns:
            Encoded bytes

        Raises:
            EncodingError: Encoding failed
        """
        try:
            if self.encoding == "json":
                return json.dumps(data).encode('utf-8')
            else:  # msgpack
                return msgpack.packb(data)
        except Exception as e:
            raise EncodingError(f"Failed to encode message: {e}")


class MessageDecoder:
    """Decode messages from WebSocket"""

    def __init__(self, encoding: str = "msgpack"):
        """
        Initialize decoder.

        Args:
            encoding: "json" or "msgpack"
        """
        if encoding not in ("json", "msgpack"):
            raise ValueError(f"Invalid encoding: {encoding}. Must be 'json' or 'msgpack'")

        self.encoding = encoding

    def decode(self, data: bytes) -> Dict[str, Any]:
        """
        Decode message.

        Args:
            data: Encoded bytes

        Returns:
            Decoded dict

        Raises:
            EncodingError: Decoding failed
        """
        try:
            if self.encoding == "json":
                return json.loads(data.decode('utf-8'))
            else:  # msgpack
                return msgpack.unpackb(data, raw=False)
        except Exception as e:
            raise EncodingError(f"Failed to decode message: {e}")
