"""
Custom exceptions for PyneCore API client.
"""

from typing import Any


class APIError(Exception):
    """Base exception for API-related errors."""

    def __init__(self, message: str = "", status_code: int | None = None,
                 response_data: dict[str, Any] | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data or {}


class AuthError(APIError):
    """Authentication-related errors (401, invalid token, etc.)."""
    pass


class RateLimitError(APIError):
    """Rate limiting errors (429)."""

    def __init__(self, message: str, retry_after: int | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class CompilationError(APIError):
    """Compilation-related errors (400, 422)."""

    def __init__(self, message: str, validation_errors: list | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.validation_errors = validation_errors or []


class NetworkError(APIError):
    """Network-related errors (timeouts, connection issues)."""
    pass


class ServerError(APIError):
    """Server-side errors (500, 502, etc.)."""
    pass
