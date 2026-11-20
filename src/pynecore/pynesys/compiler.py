"""
Core compilation service for programmatic use.
"""
import hashlib
import json
from pathlib import Path

from pynecore.pynesys.api import (APIClient, APIError, AuthError, RateLimitError, CompilationError, UsageResponse,
                                  TokenValidationResponse, CompileResponse)
from pynecore.utils.file_utils import is_updated


class PyneComp:
    """
    Compiler through the PyneSys API.
    """

    api_client: APIClient

    def __init__(self, api_key, base_url="https://api.pynesys.io", timeout=30):
        """
        Initialize the compilation service.

        :param api_key: PyneSys API key
        :param base_url: Base URL for the API
        :param timeout: Request timeout in seconds
        """
        self.api_client = APIClient(api_key=api_key, base_url=base_url, timeout=timeout)

    def compile(self, pine_path: Path, output_path: Path | None = None,
                force: bool = False, strict: bool = False) -> Path:
        """
        Compile a .pine file to Python.

        :param pine_path: Path to the .pine file
        :param output_path: Optional output path (defaults to .py extension)
        :param force: Force recompilation even if file hasn't changed
        :param strict: No effect; accepted so existing callers keep working
        :return: Path to the compiled .py file
        :raises FileNotFoundError: If pine file doesn't exist
        :raises CompilationError: If compilation fails
        :raises APIError: If API request fails
        """
        del strict
        # Validate input file
        if not pine_path.exists():
            raise FileNotFoundError(f"Pine file not found: {pine_path}")

        if pine_path.suffix != '.pine':
            raise ValueError(f"This file format isn't supported: {pine_path.suffix}. "
                             f"Only .pine files can be compiled!")

        # Determine output path
        resolved_output: Path = output_path if output_path is not None else pine_path.with_suffix('.py')

        # Check if compilation is needed (unless forced)
        if not force and not self.needs_compilation(pine_path, resolved_output):
            return resolved_output

        # Read Pine Script content
        try:
            with open(pine_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
        except IOError as e:
            raise IOError(f"Error reading Pine file {pine_path}: {e}")

        # Compile via API
        try:
            response = self.api_client.compile_script(script_content, sourcemap=True)

            if not response.success:
                raise CompilationError(
                    f"Compilation failed: {response.error_message}",
                    status_code=response.status_code,
                    validation_errors=response.validation_errors
                )

            # Write compiled code to output file
            assert response.compiled_code is not None
            resolved_output.parent.mkdir(parents=True, exist_ok=True)
            with open(resolved_output, 'w', encoding='utf-8') as f:
                f.write(response.compiled_code)
            self.write_sourcemap(resolved_output, response, script_content)

            # No need to update tracking info with mtime approach
            return resolved_output

        except (APIError, AuthError, RateLimitError, CompilationError):
            # Re-raise API-related errors as-is
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise APIError(f"Unexpected error during compilation: {e}")

    def get_usage(self) -> UsageResponse:
        """
        Get current usage statistics and limits for the authenticated user.
        """
        return self.api_client.get_usage()

    def validate_api_key(self) -> TokenValidationResponse:
        """
        Validate API key.
        """
        return self.api_client.verify_token_local()

    @staticmethod
    def write_sourcemap(output_path: Path, response: CompileResponse, pine: str) -> None:
        """
        Write the compile's sourcemap beside the output as ``<output>.map``, in the form
        PyneIDE writes it: stamped with the hashes of the Python code and the Pine source, so a
        later edit of the ``.py`` invalidates it. A response without a sourcemap (an older
        server) removes an existing map instead: a stale map must not outlive its code.

        :param output_path: The compiled ``.py`` file
        :param response: The successful compile response
        :param pine: The Pine Script source
        """
        map_path = output_path.with_name(output_path.name + '.map')
        if response.sourcemap is None:
            map_path.unlink(missing_ok=True)
            return
        stored = dict(response.sourcemap)
        # The file as written, which on Windows has the platform's line endings
        stored['py_sha256'] = hashlib.sha256(output_path.read_bytes()).hexdigest()
        stored['pine_sha256'] = hashlib.sha256(pine.encode('utf-8')).hexdigest()
        pine_version = stored.get('pine_version')
        if isinstance(pine_version, int) and pine_version < 6:
            # The mappings point at the lines of the v6 script the compiler converted it to
            stored['converted_source'] = True
        map_path.write_text(json.dumps(stored), encoding='utf-8')

    @staticmethod
    def needs_compilation(pine_file_path: Path, output_file_path: Path) -> bool:
        """
        Check if a .pine file needs compilation using modification time comparison.

        :param pine_file_path: Path to the .pine file
        :param output_file_path: Path to the compiled .py file
        :return: True if compilation is needed, False otherwise
        """
        return is_updated(pine_file_path, output_file_path)
