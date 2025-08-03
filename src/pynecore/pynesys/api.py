"""
PyneCore API client
"""
from typing import Any

import json
import base64

from datetime import datetime

from dataclasses import dataclass

import urllib.request
import urllib.parse
import urllib.error


#
# API Response Models
#

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
class UsageLimits:
    """Usage limits for daily and hourly periods."""
    limit: int
    used: int
    remaining: int
    reset_at: datetime


@dataclass
class UsageResponse:
    """Response from account usage endpoint."""
    daily: UsageLimits
    hourly: UsageLimits
    api_keys: dict[str, Any]
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
        """Check if response indicates an authentication error."""
        return self.status_code == 401


#
# Exceptions
#

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


#
# API Client
#

class APIClient:
    """
    API Client for interacting with PyneSys API
    """

    def __init__(self, api_key: str, base_url: str = "https://api.pynesys.io", timeout: int = 30):
        """
        Initialize the API client.

        :param api_key: PyneSys API key
        :param base_url: Base URL for the API
        :param timeout: Request timeout in seconds
        """
        if api_key is None or not api_key.strip():
            raise ValueError("API key is required")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _make_request(
            self,
            method: str,
            endpoint: str,
            data: dict[str, Any] | None = None,
            headers: dict[str, str] | None = None
    ) -> urllib.request.Request:
        """
        Create a urllib request object.

        :param method: HTTP method (GET, POST, etc.)
        :param endpoint: API endpoint
        :param data: Request data
        :param headers: Additional headers
        :return: Configured request object
        """
        url = f"{self.base_url}/{endpoint}"

        # Default headers
        req_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "PyneCore-API-Client",
        }

        if headers:
            req_headers.update(headers)

        # Handle data encoding
        encoded_data = None
        if data and method != "GET":
            if "Content-Type" in req_headers and "json" in req_headers["Content-Type"]:
                encoded_data = json.dumps(data).encode('utf-8')
            else:
                encoded_data = urllib.parse.urlencode(data).encode('utf-8')
        elif data and method == "GET":
            # For GET requests, add data as query parameters
            query_string = urllib.parse.urlencode(data)
            url = f"{url}?{query_string}"

        request = urllib.request.Request(
            url,
            data=encoded_data,
            headers=req_headers,
            method=method
        )

        return request

    def verify_token(self) -> TokenValidationResponse:
        """
        Verify API token validity.

        :return: TokenValidationResponse with validation details
        :raises AuthError: If token is invalid
        :raises NetworkError: If network request fails
        :raises APIError: For other API errors
        """
        try:
            request = self._make_request(
                "GET",
                "auth/verify-token",
                data={"token": self.api_key}
            )

            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode('utf-8'))
                return TokenValidationResponse(
                    valid=data.get("valid", False),
                    message=data.get("message", ""),
                    user_id=data.get("user_id"),
                    token_type=data.get("token_type"),
                    expiration=(datetime.fromisoformat(data["expiration"].replace("Z", "+00:00"))
                                if data.get("expiration") else None),
                    expires_at=(datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00"))
                                if data.get("expires_at") else None),
                    expires_in=data.get("expires_in"),
                    raw_response=data,
                )

        except urllib.error.HTTPError as e:
            self._handle_http_error(e)
        except urllib.error.URLError as e:
            raise NetworkError(f"Network error during token verification: {e}")
        except Exception as e:
            if not isinstance(e, APIError):
                raise APIError(f"Unexpected error during token verification: {e}")
            else:
                raise

        raise APIError("Unexpected error during token verification.")

    def verify_token_local(self) -> TokenValidationResponse:
        """
        Verify JWT token locally without server request.
        
        :return: TokenValidationResponse with validation details
        :raises AuthError: If token format is invalid or expired
        """
        try:
            # JWT tokens have format: header.payload.signature
            parts = self.api_key.split('.')
            if len(parts) != 3:
                return TokenValidationResponse(
                    valid=False,
                    message="Invalid JWT format: must have 3 parts separated by dots"
                )

            header_b64, payload_b64, signature_b64 = parts

            # Decode header
            try:
                # Add padding if needed
                header_b64 += '=' * (4 - len(header_b64) % 4)
                header_data = json.loads(base64.urlsafe_b64decode(header_b64).decode('utf-8'))
            except (ValueError, json.JSONDecodeError):
                return TokenValidationResponse(
                    valid=False,
                    message="Invalid JWT header format"
                )

            # Decode payload
            try:
                # Add padding if needed
                payload_b64 += '=' * (4 - len(payload_b64) % 4)
                payload_data = json.loads(base64.urlsafe_b64decode(payload_b64).decode('utf-8'))
            except (ValueError, json.JSONDecodeError):
                return TokenValidationResponse(
                    valid=False,
                    message="Invalid JWT payload format"
                )

            # Check expiration - try both 'exp' (standard) and 'e' (custom format)
            exp = payload_data.get('exp') or payload_data.get('e')
            if exp:
                exp_time = datetime.fromtimestamp(exp)
                if datetime.now() >= exp_time:
                    return TokenValidationResponse(
                        valid=False,
                        message="Token has expired",
                        expiration=exp_time,
                        expires_at=exp_time
                    )

            # Extract user info
            user_id = payload_data.get('s')  # Based on the image, 's' contains user ID

            return TokenValidationResponse(
                valid=True,
                message="Token is valid",
                user_id=user_id,
                token_type=header_data.get('typ', 'JWT'),
                expiration=datetime.fromtimestamp(exp) if exp else None,
                expires_at=datetime.fromtimestamp(exp) if exp else None,
                raw_response={
                    'header': header_data,
                    'payload': payload_data
                }
            )

        except Exception as e:
            return TokenValidationResponse(
                valid=False,
                message=f"Token validation error: {str(e)}"
            )

    def get_usage(self) -> UsageResponse:
        """
        Get current usage statistics and limits for the authenticated user.

        :return: UsageResponse with usage details
        :raises AuthError: If authentication fails
        :raises NetworkError: If network request fails
        :raises APIError: For other API errors
        """
        try:
            request = self._make_request("GET", "account/usage")

            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode('utf-8'))

                # Parse daily usage
                daily_data = data["daily"]
                daily = UsageLimits(
                    limit=daily_data["limit"],
                    used=daily_data["used"],
                    remaining=daily_data["remaining"],
                    reset_at=datetime.fromisoformat(daily_data["reset_at"].replace("Z", "+00:00"))
                )

                # Parse hourly usage
                hourly_data = data["hourly"]
                hourly = UsageLimits(
                    limit=hourly_data["limit"],
                    used=hourly_data["used"],
                    remaining=hourly_data["remaining"],
                    reset_at=datetime.fromisoformat(hourly_data["reset_at"].replace("Z", "+00:00"))
                )

                return UsageResponse(
                    daily=daily,
                    hourly=hourly,
                    api_keys=data.get("api_keys", {}),
                    raw_response=data
                )

        except urllib.error.HTTPError as e:
            self._handle_http_error(e)
        except urllib.error.URLError as e:
            raise NetworkError(f"Network error during usage retrieval: {e}")
        except Exception as e:
            if not isinstance(e, APIError):
                raise APIError(f"Unexpected error during usage retrieval: {e}")
            else:
                raise

        raise APIError("Unexpected error during usage retrieval.")

    def compile_script(
            self,
            script: str,
            strict: bool = False
    ) -> CompileResponse:
        """
        Compile Pine Script to Python via API.

        :param script: Pine Script code to compile
        :param strict: Whether to use strict compilation mode
        :return: CompileResponse with compiled code or error details
        :raises AuthError: If authentication fails
        :raises RateLimitError: If rate limit is exceeded
        :raises CompilationError: If compilation fails
        :raises NetworkError: If network request fails
        :raises APIError: For other API errors
        """
        try:
            # Prepare form data
            data = {
                "script": script,
                "strict": str(strict).lower()
            }

            request = self._make_request(
                "POST",
                "compiler/compile",
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )

            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                # Success - return compiled code
                compiled_code = response.read().decode('utf-8')
                return CompileResponse(
                    success=True,
                    compiled_code=compiled_code,
                    status_code=200
                )

        except urllib.error.HTTPError as e:
            # Handle error responses
            return self._handle_compile_http_error(e)
        except urllib.error.URLError as e:
            raise NetworkError(f"Network error during compilation: {e}")
        except Exception as e:
            if not isinstance(e, APIError):
                raise APIError(f"Unexpected error during compilation: {e}")
            else:
                raise

    @staticmethod
    def _handle_http_error(error: urllib.error.HTTPError, message: str = None) -> None:
        """
        Handle HTTP error responses.

        :param error: HTTPError object
        :param message: Optional pre-extracted error message
        :raises: Appropriate exception based on status code
        """
        status_code = error.code

        if message is None:
            try:
                error_content = error.read().decode('utf-8')
                error_data = json.loads(error_content)
                message = error_data.get("message", error_content)
            except (json.JSONDecodeError, ValueError):
                message = error.reason or f"HTTP {status_code} error"

        if status_code == 401:
            raise AuthError(message, status_code=status_code)
        elif status_code == 429:
            retry_after = error.headers.get("Retry-After")
            raise RateLimitError(
                message,
                status_code=status_code,
                retry_after=int(retry_after) if retry_after else None
            )
        elif status_code >= 500:
            raise ServerError(message, status_code=status_code)
        else:
            raise APIError(message, status_code=status_code)

    def _handle_compile_http_error(self, error: urllib.error.HTTPError) -> CompileResponse:
        """
        Handle compilation error responses.

        :param error: HTTPError object
        :return: CompileResponse with error details
        :raises CompilationError: For compilation-related errors (422)
        :raises: Other exceptions for authentication, rate limiting, etc.
        """
        status_code = error.code

        try:
            error_content = error.read().decode('utf-8')
            error_data = json.loads(error_content)
        except (json.JSONDecodeError, ValueError):
            error_data = {}
            error_content = error.reason or f"HTTP {status_code} error"

        # Extract error message
        if "detail" in error_data and isinstance(error_data["detail"], list):
            # Validation error format (422)
            validation_errors = error_data["detail"]
            error_message = "Validation errors occurred"
        elif "detail" in error_data and isinstance(error_data["detail"], dict):
            # Structured error format (400) - pass the complete JSON for parsing
            validation_errors = None
            error_message = error_content  # Pass the full JSON response
        else:
            validation_errors = None
            error_message = error_data.get("message", error_content)

        # For compilation errors (422), raise CompilationError
        if status_code == 422:
            raise CompilationError(error_message, status_code=status_code, validation_errors=validation_errors)

        # For other errors, use the general error handler with the extracted message
        else:
            self._handle_http_error(error, error_message)

        # This should never be reached
        return CompileResponse(
            success=False,
            error_message=error_message,
            validation_errors=validation_errors,
            status_code=status_code,
            raw_response=error_data
        )
