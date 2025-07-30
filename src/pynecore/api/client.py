"""PyneCore API client for PyneSys compiler service."""

import asyncio
from typing import Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime
import json

if TYPE_CHECKING:
    import httpx
else:
    try:
        import httpx
    except ImportError:
        httpx = None

from .exceptions import (
    APIError,
    AuthError,
    RateLimitError,
    CompilationError,
    NetworkError,
    ServerError,
)
from .models import TokenValidationResponse, CompileResponse


class PynesysAPIClient:
    """Client for interacting with PyneSys API."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.pynesys.io",
        timeout: int = 30,
    ):
        """Initialize the API client.
        
        Args:
            api_key: PyneSys API key
            base_url: Base URL for the API
            timeout: Request timeout in seconds
        """
        if httpx is None:
            raise ImportError(
                "httpx is required for API functionality. "
                "Install it with: pip install httpx"
            )
        
        if not api_key or not api_key.strip():
            raise ValueError("API key is required")
        
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional["httpx.AsyncClient"] = None
    
    def compile_script_sync(self, script: str, strict: bool = False) -> CompileResponse:
        """Synchronous wrapper for compile_script.
        
        Args:
            script: Pine Script code to compile
            strict: Enable strict compilation mode
            
        Returns:
            CompileResponse with compilation results
        """
        return asyncio.run(self.compile_script(script, strict))
    
    def verify_token_sync(self) -> TokenValidationResponse:
        """Synchronous wrapper for verify_token.
        
        Returns:
            TokenValidationResponse with token validation results
        """
        return asyncio.run(self.verify_token())
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_client(self):
        """Ensure HTTP client is initialized."""
        if self._client is None and httpx is not None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": "PyneCore-API-Client",
                },
            )
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def verify_token(self) -> TokenValidationResponse:
        """Verify API token validity.
        
        Returns:
            TokenValidationResponse with validation details
            
        Raises:
            AuthError: If token is invalid
            NetworkError: If network request fails
            APIError: For other API errors
        """
        await self._ensure_client()
        
        try:
            if self._client is None:
                raise APIError("HTTP client not initialized")
                
            response = await self._client.get(
                f"{self.base_url}/auth/verify-token",
                params={"token": self.api_key}
            )
            
            if response.status_code == 200:
                data = response.json()
                return TokenValidationResponse(
                    valid=data.get("valid", False),
                    message=data.get("message", ""),
                    user_id=data.get("user_id"),
                    token_type=data.get("token_type"),
                    expiration=self._parse_datetime(data.get("expiration")),
                    expires_at=self._parse_datetime(data.get("expires_at")),
                    expires_in=data.get("expires_in"),
                    raw_response=data,
                )
            else:
                self._handle_api_error(response)
                # This should never be reached due to _handle_api_error raising
                raise APIError("Unexpected API response")
                
        except Exception as e:
            if httpx and isinstance(e, httpx.RequestError):
                raise NetworkError(f"Network error during token verification: {e}")
            elif not isinstance(e, APIError):
                raise APIError(f"Unexpected error during token verification: {e}")
            else:
                raise
    
    async def compile_script(
        self,
        script: str,
        strict: bool = False
    ) -> CompileResponse:
        """Compile Pine Script to Python via API.
        
        Args:
            script: Pine Script code to compile
            strict: Whether to use strict compilation mode
            
        Returns:
            CompileResponse with compiled code or error details
            
        Raises:
            AuthError: If authentication fails
            RateLimitError: If rate limit is exceeded
            CompilationError: If compilation fails
            NetworkError: If network request fails
            APIError: For other API errors
        """
        await self._ensure_client()
        
        try:
            # Prepare form data
            data = {
                "script": script,
                "strict": str(strict).lower()
            }
            
            if self._client is None:
                raise APIError("HTTP client not initialized")
                
            response = await self._client.post(
                f"{self.base_url}/compiler/compile",
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            if response.status_code == 200:
                # Success - return compiled code
                compiled_code = response.text
                return CompileResponse(
                    success=True,
                    compiled_code=compiled_code,
                    status_code=200
                )
            else:
                # Handle error responses
                return self._handle_compile_error(response)
                
        except Exception as e:
            if httpx and isinstance(e, httpx.RequestError):
                raise NetworkError(f"Network error during compilation: {e}")
            elif not isinstance(e, APIError):
                raise APIError(f"Unexpected error during compilation: {e}")
            else:
                raise
    
    def _handle_api_error(self, response: "httpx.Response") -> None:
        """Handle API error responses.
        
        Args:
            response: HTTP response object
            
        Raises:
            Appropriate exception based on status code
        """
        status_code = response.status_code
        
        try:
            error_data = response.json()
            message = error_data.get("message", response.text)
        except (json.JSONDecodeError, ValueError):
            message = response.text or f"HTTP {status_code} error"
        
        if status_code == 401:
            raise AuthError(message, status_code=status_code)
        elif status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(
                message,
                status_code=status_code,
                retry_after=int(retry_after) if retry_after else None
            )
        elif status_code >= 500:
            raise ServerError(message, status_code=status_code)
        else:
            raise APIError(message, status_code=status_code)
    
    def _handle_compile_error(self, response: "httpx.Response") -> CompileResponse:
        """Handle compilation error responses.
        
        Args:
            response: HTTP response object
            
        Returns:
            CompileResponse with error details
            
        Raises:
            CompilationError: For compilation-related errors (422)
            Other exceptions: For authentication, rate limiting, etc.
        """
        status_code = response.status_code
        
        try:
            error_data = response.json()
        except (json.JSONDecodeError, ValueError):
            error_data = {}
        
        # Extract error message
        if "detail" in error_data and isinstance(error_data["detail"], list):
            # Validation error format (422)
            validation_errors = error_data["detail"]
            error_message = "Validation errors occurred"
        else:
            validation_errors = None
            error_message = error_data.get("message", response.text or f"HTTP {status_code} error")
        
        # For compilation errors (422), raise CompilationError
        if status_code == 422:
            raise CompilationError(error_message, status_code=status_code, validation_errors=validation_errors)
        
        # For other errors, use the general API error handler
        self._handle_api_error(response)
        
        # This should never be reached
        return CompileResponse(
            success=False,
            error_message=error_message,
            validation_errors=validation_errors,
            status_code=status_code,
            raw_response=error_data
        )
    
    def _parse_datetime(self, dt_str: Optional[str]) -> Optional[datetime]:
        """Parse datetime string from API response.
        
        Args:
            dt_str: Datetime string from API
            
        Returns:
            Parsed datetime object or None
        """
        if not dt_str:
            return None
        
        try:
            # Try common datetime formats
            for fmt in [
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d"
            ]:
                try:
                    return datetime.strptime(dt_str, fmt)
                except ValueError:
                    continue
            
            # If no format matches, return None
            return None
            
        except Exception:
            return None


# Synchronous wrapper for convenience
class SyncPynesysAPIClient:
    """Synchronous wrapper for PynesysAPIClient."""
    
    def __init__(self, *args, **kwargs):
        self._async_client = PynesysAPIClient(*args, **kwargs)
    
    def verify_token(self) -> TokenValidationResponse:
        """Synchronous token verification."""
        return asyncio.run(self._async_client.verify_token())
    
    def compile_script(self, script: str, strict: bool = False) -> CompileResponse:
        """Synchronous script compilation."""
        return asyncio.run(self._async_client.compile_script(script, strict))
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        asyncio.run(self._async_client.close())