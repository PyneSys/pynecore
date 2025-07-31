"""Core compilation service for programmatic use.

The CLI should use this service rather than implementing compilation logic directly.
This ensures all compilation functionality is available programmatically.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from ..api.client import PynesysAPIClient
from ..api.config import APIConfig, ConfigManager
from ..api.exceptions import APIError, AuthError, RateLimitError, CompilationError
from ..utils.mtime_utils import file_needs_compilation


class CompilationService:
    """Core compilation service for programmatic use.
    
    The CLI should use this service rather than implementing compilation logic directly.
    This ensures all compilation functionality is available programmatically.
    """
    
    def __init__(self, api_client: Optional[PynesysAPIClient] = None, config: Optional[APIConfig] = None):
        """Initialize the compilation service.
        
        Args:
            api_client: Optional pre-configured API client
            config: Optional API configuration (will load from default if not provided)
        """
        if api_client:
            self.api_client = api_client
        else:
            # Load config if not provided
            if not config:
                config = ConfigManager.load_config()
            self.api_client = PynesysAPIClient(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=config.timeout
            )
    
    def compile_file(
        self,
        pine_file_path: Path,
        output_file_path: Optional[Path] = None,
        force: bool = False,
        strict: bool = False
    ) -> Path:
        """Compile a .pine file to Python.
        
        Args:
            pine_file_path: Path to the .pine file
            output_file_path: Optional output path (defaults to .py extension)
            force: Force recompilation even if file hasn't changed
            strict: Enable strict compilation mode
            
        Returns:
            Path to the compiled .py file
            
        Raises:
            FileNotFoundError: If pine file doesn't exist
            CompilationError: If compilation fails
            APIError: If API request fails
        """
        # Validate input file
        if not pine_file_path.exists():
            raise FileNotFoundError(f"Pine file not found: {pine_file_path}")
        
        if pine_file_path.suffix != '.pine':
            raise ValueError(f"This file format isn't supported: {pine_file_path.suffix}. Only .pine files can be compiled! ✨ Try using a .pine file instead.")
        
        # Determine output path
        if output_file_path is None:
            output_file_path = pine_file_path.with_suffix('.py')
        
        # Check if compilation is needed (unless forced)
        if not force and not self.needs_compilation(pine_file_path, output_file_path):
            return output_file_path
        
        # Read Pine Script content
        try:
            with open(pine_file_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
        except IOError as e:
            raise IOError(f"Error reading Pine file {pine_file_path}: {e}")
        
        # Compile via API
        try:
            response = self.api_client.compile_script_sync(script_content, strict=strict)
            
            if not response.success:
                raise CompilationError(
                    f"Compilation failed: {response.error_message}",
                    status_code=response.status_code,
                    validation_errors=response.validation_errors
                )
            
            # Write compiled code to output file
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(response.compiled_code)
            
            # No need to update tracking info with mtime approach
            
            return output_file_path
            
        except (APIError, AuthError, RateLimitError, CompilationError) as e:
            # Re-raise API-related errors as-is
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise APIError(f"Unexpected error during compilation: {e}")
    
    def needs_compilation(self, pine_file_path: Path, output_file_path: Path) -> bool:
        """Check if a .pine file needs compilation using modification time comparison.
        
        Args:
            pine_file_path: Path to the .pine file
            output_file_path: Path to the compiled .py file
            
        Returns:
            True if compilation is needed, False otherwise
        """
        return file_needs_compilation(pine_file_path, output_file_path)
    
    def compile_and_run(
        self,
        pine_file_path: Path,
        script_args: Optional[list] = None,
        force: bool = False,
        strict: bool = False,
        output_file_path: Optional[Path] = None
    ) -> int:
        """Compile a .pine file and run the resulting Python script.
        
        Args:
            pine_file_path: Path to the .pine file
            script_args: Arguments to pass to the compiled script
            force: Force recompilation even if file hasn't changed
            strict: Enable strict compilation mode
            output_file_path: Optional output path (defaults to .py extension)
            
        Returns:
            Exit code from the executed script
            
        Raises:
            FileNotFoundError: If pine file doesn't exist
            CompilationError: If compilation fails
            APIError: If API request fails
        """
        # Compile the file
        compiled_file = self.compile_file(
            pine_file_path=pine_file_path,
            output_file_path=output_file_path,
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
    



def create_compilation_service(
    api_key: Optional[str] = None,
    config_path: Optional[Path] = None
) -> CompilationService:
    """Factory function to create a CompilationService instance.
    
    Args:
        api_key: Optional API key override
        config_path: Optional path to config file
        
    Returns:
        Configured CompilationService instance
        
    Raises:
        ValueError: If no valid configuration found
    """
    if api_key:
        # Create API client with provided key
        api_client = PynesysAPIClient(api_key=api_key)
        return CompilationService(api_client=api_client)
    else:
        # Load configuration from file
        config = ConfigManager.load_config(config_path)
        return CompilationService(config=config)