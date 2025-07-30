"""Data models for PyneCore API responses."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime


@dataclass
class TokenValidationResponse:
    """Response from token validation endpoint."""
    valid: bool
    message: str
    user_id: Optional[str] = None
    token_type: Optional[str] = None
    expiration: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    expires_in: Optional[int] = None
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class CompileResponse:
    """Response from script compilation endpoint."""
    success: bool
    compiled_code: Optional[str] = None
    error_message: Optional[str] = None
    error: Optional[str] = None
    validation_errors: Optional[List[Dict[str, Any]]] = None
    warnings: Optional[List[str]] = None
    details: Optional[List[str]] = None
    status_code: Optional[int] = None
    raw_response: Optional[Dict[str, Any]] = None

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