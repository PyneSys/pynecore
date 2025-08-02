"""
Core compilation service for programmatic use.
"""

import subprocess
import sys
from pathlib import Path

from pynecore.pynesys.api import APIClient
from pynecore.pynesys.api import APIError, AuthError, RateLimitError, CompilationError
from pynecore.utils.file_utils import is_updated


class PyneComp:
    """
    COmpiler through the PyneSys API.
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
        :param strict: Enable strict compilation mode
        :return: Path to the compiled .py file
        :raises FileNotFoundError: If pine file doesn't exist
        :raises CompilationError: If compilation fails
        :raises APIError: If API request fails
        """
        # Validate input file
        if not pine_path.exists():
            raise FileNotFoundError(f"Pine file not found: {pine_path}")

        if pine_path.suffix != '.pine':
            raise ValueError(f"This file format isn't supported: {pine_path.suffix}. "
                             f"Only .pine files can be compiled!")

        # Determine output path
        if output_path is None:
            output_path = pine_path.with_suffix('.py')

        # Check if compilation is needed (unless forced)
        if not force and not self.needs_compilation(pine_path, output_path):
            return output_path

        # Read Pine Script content
        try:
            with open(pine_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
        except IOError as e:
            raise IOError(f"Error reading Pine file {pine_path}: {e}")

        # Compile via API
        try:
            response = self.api_client.compile_script(script_content, strict=strict)

            if not response.success:
                raise CompilationError(
                    f"Compilation failed: {response.error_message}",
                    status_code=response.status_code,
                    validation_errors=response.validation_errors
                )

            # Write compiled code to output file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(response.compiled_code)

            # No need to update tracking info with mtime approach
            return output_path

        except (APIError, AuthError, RateLimitError, CompilationError):
            # Re-raise API-related errors as-is
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise APIError(f"Unexpected error during compilation: {e}")

    @staticmethod
    def needs_compilation(pine_file_path: Path, output_file_path: Path) -> bool:
        """
        Check if a .pine file needs compilation using modification time comparison.

        :param pine_file_path: Path to the .pine file
        :param output_file_path: Path to the compiled .py file
        :return: True if compilation is needed, False otherwise
        """
        return is_updated(pine_file_path, output_file_path)

    def compile_and_run(
            self,
            pine_file_path: Path,
            script_args: list | None = None,
            force: bool = False,
            strict: bool = False,
            output_file_path: Path | None = None
    ) -> int:
        """
        Compile a .pine file and run the resulting Python script.

        :param pine_file_path: Path to the .pine file
        :param script_args: Arguments to pass to the compiled script
        :param force: Force recompilation even if file hasn't changed
        :param strict: Enable strict compilation mode
        :param output_file_path: Optional output path (defaults to .py extension)
        :return: Exit code from the executed script
        :raises FileNotFoundError: If pine file doesn't exist
        :raises CompilationError: If compilation fails
        :raises APIError: If API request fails
        """
        # Compile the file
        compiled_file = self.compile(
            pine_path=pine_file_path,
            output_path=output_file_path,
            force=force,
            strict=strict
        )

        # Prepare command to run the compiled script
        cmd = [sys.executable, str(compiled_file)]
        if script_args:
            cmd.extend(script_args)

        # Run the compiled script
        try:
            result = subprocess.run(cmd, check=False)
            return result.returncode
        except Exception as e:
            raise RuntimeError(f"Error executing compiled script: {e}")
