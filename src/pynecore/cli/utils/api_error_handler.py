"""Centralized API error handling utilities for CLI commands."""

import re

from rich.console import Console
from typer import Exit

from pynecore.pynesys.api import APIError, AuthError, RateLimitError, CompilationError


class APIErrorHandler:
    """Context manager that provides centralized API error handling."""

    def __init__(self, console: Console | None = None):
        if not console:
            console = Console()
        self.console = console

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            return False

        if exc_type == CompilationError:
            self._handle_compilation_error(exc_value)
        elif exc_type == AuthError:
            self._handle_auth_error(exc_value)
        elif exc_type == RateLimitError:
            self._handle_rate_limit_error(exc_value)
        elif exc_type == APIError:
            self._handle_api_error(exc_value)
        else:
            return False  # Let other exceptions propagate

        raise Exit(1)

    def _handle_compilation_error(self, e: CompilationError):
        """Handle compilation-specific errors."""
        self.console.print(f"[red]Oops! Compilation encountered an issue:[/red] {str(e)}")
        if e.validation_errors:
            self.console.print("[red]Validation errors:[/red]")
            for error in e.validation_errors:
                self.console.print(f"  [red]• {error}[/red]")

    def _handle_auth_error(self, e: AuthError):
        """Handle authentication errors."""
        self.console.print(f"[red]Authentication issue:[/red] {str(e)}")
        self.console.print("[yellow]To fix:[/yellow] Check [cyan]api_key[/cyan] in [cyan]api.toml[/cyan] "
                           "in your working directory")

    def _handle_rate_limit_error(self, e: RateLimitError):
        """Handle rate limit errors."""
        self.console.print("[red]Rate Limit Exceeded:[/red] You've hit your compilation limit")
        if e.retry_after:
            self.console.print(f"[yellow]Please try again in {e.retry_after} seconds[/yellow]")
        self.console.print(
            "[yellow]To increase your limits, consider upgrading your subscription at "
            "[link=https://pynesys.io]https://pynesys.io[/link]")

    def _handle_api_error(self, e: APIError):
        """Handle general API errors with specific status code handling."""
        error_msg = str(e).lower()

        # Handle specific API error scenarios based on HTTP status codes
        has_400_error = "400" in error_msg or "bad request" in error_msg
        has_structured_error = (
            '"detail":' in error_msg and 
            '"error":' in error_msg and 
            '"line":' in error_msg
        )
        
        if has_400_error or has_structured_error:
            if "compilation fails" in error_msg or "script is too large" in error_msg:
                self.console.print("[red]Script Issue:[/red] Your Pine Script couldn't be compiled")
                self.console.print("[yellow]Common fixes:[/yellow]")
                self.console.print("  • Check if your script is too large (try breaking it into smaller parts)")
                self.console.print("  • Verify your Pine Script syntax is correct")
                self.console.print("  • Make sure you're using Pine Script v6 syntax")
            else:
                # Try to parse structured error response (JSON-like format)
                error_str = str(e)
                structured_error_parsed = False

                # Check for structured JSON error format
                has_error_and_line = (
                    ("'error':" in error_str and "'line':" in error_str) or
                    ('"error":' in error_str and '"line":' in error_str)
                )
                
                if has_error_and_line:
                    # Pattern for JSON format: {"detail":{"error":"...","line":...,"file":"..."}}
                    json_pattern = (
                        r'\{"detail":\{"status":"error","error":"([^"]+)",' +
                        r'"line":(\d+),"file":"([^"]+)"\}\}'
                    )
                    match = re.search(json_pattern, error_str)
                    if match:
                        error_msg, line_num, file_name = match.groups()
                        self.console.print(f"[red]Script Error:[/red] {error_msg}")
                        self.console.print(
                            f"[yellow]Location:[/yellow] Line {line_num} in {file_name}"
                        )
                        self.console.print(
                            "[yellow]Quick fix:[/yellow] Check the variable declaration and spelling"
                        )
                        structured_error_parsed = True

                # Fallback to generic error display if structured parsing failed
                if not structured_error_parsed:
                    self.console.print(f"[red]Script Error:[/red] {str(e)}")
                    self.console.print("[yellow]Common causes:[/yellow]")
                    self.console.print("  • Pine Script syntax errors")
                    self.console.print("  • Unsupported Pine Script features")
                    self.console.print("  • Incorrect variable declarations or usage")

        elif "401" in error_msg or "authentication" in error_msg or "no permission" in error_msg:
            self.console.print("[red]Authentication Failed:[/red] Your API credentials aren't working")
            self.console.print("[yellow]Quick fixes:[/yellow]")
            self.console.print("  • Check if your API key is valid and active")
            self.console.print("  • Verify your token type is allowed for compilation")
            self.console.print(
                "Get a new API key at [link=https://pynesys.io]https://pynesys.io[/link]")
            self.console.print(
                "Then run [cyan]'pyne api configure'[/cyan] to update your configuration")

        elif "404" in error_msg or "not found" in error_msg:
            self.console.print("[red]Not Found:[/red] The API endpoint or user wasn't found")
            self.console.print("[yellow]This might indicate:[/yellow]")
            self.console.print("  • Your account may not exist or be accessible")
            self.console.print("  • There might be a temporary service issue")

        elif "422" in error_msg or "validation error" in error_msg:
            self.console.print("[red]Validation Error:[/red] Your request data has validation issues")
            self.console.print("[yellow]Common causes:[/yellow]")
            self.console.print("  • Invalid Pine Script syntax or structure")
            self.console.print("  • Missing required parameters")
            self.console.print("  • Incorrect data format")
            self.console.print(f"[dim]Details: {str(e)}[/dim]")

        elif "429" in error_msg or "rate limit" in error_msg or "too many requests" in error_msg:
            self.console.print("[red]Rate Limit Exceeded:[/red] You've hit your compilation limit")
            self.console.print("[yellow]What you can do:[/yellow]")
            self.console.print("  • Wait a bit before trying again")
            self.console.print("  • Consider upgrading your plan for higher limits")

        elif "500" in error_msg or "server" in error_msg or "internal" in error_msg:
            self.console.print("[red]Server Error:[/red] Something went wrong on our end")
            self.console.print("  • This is a temporary server issue")
            self.console.print("  • Please try again in a few moments")

        elif "unsupported pinescript version" in error_msg:
            self.console.print("[red]Version Issue:[/red] Your Pine Script version isn't supported")
            if "version 5" in error_msg:
                self.console.print("[yellow]Pine Script v5 → v6 Migration:[/yellow]")
                self.console.print("  • Update your script to Pine Script version 6")
                self.console.print("  • Most v5 scripts need minimal changes")
            else:
                self.console.print("[yellow]Only Pine Script version 6 is currently supported[/yellow]")

        elif "api key" in error_msg:
            self.console.print("[red]API Key Issue:[/red] There's a problem with your API key")
            self.console.print("Get your API key at [link=https://pynesys.io]https://pynesys.io[/link]")
            self.console.print(
                "Then run [cyan]'pyne api configure'[/cyan] to set up your configuration")

        else:
            # Generic API error fallback
            self.console.print(f"[red]API Error:[/red] {str(e)}")
