"""
Data models for PyneCore API responses.
"""

from dataclasses import dataclass
from typing import Any
from datetime import datetime


@dataclass
class TokenValidationResponse:
    """Response from token validation endpoint."""
    valid: bool
    message: str
    user_id: str | None = None
    token_type: str | None = None
    expiration: datetime | None = None
    expires_at: datetime | None = None
    expires_in: int | None = None
    raw_response: dict[str, Any] | None = None


@dataclass
class CompileResponse:
    """Response from script compilation endpoint."""
    success: bool
    compiled_code: str | None = None
    error_message: str | None = None
    error: str | None = None
    validation_errors: list[dict[str, Any]] | None = None
    warnings: list[str] | None = None
    details: list[str] | None = None
    status_code: int | None = None
    raw_response: dict[str, Any] | None = None

    @property
    def has_validation_errors(self) -> bool:
        """Check if response contains validation errors."""
        return bool(self.validation_errors)

    @property
    def is_rate_limited(self) -> bool:
        """Check if response indicates rate limiting."""
        return self.status_code == 429

    @property
    def is_auth_error(self) -> bool:
        """Check if response indicates authentication error."""
        return self.status_code == 401
