"""Configuration management for PyneSys API."""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

# Try to import tomllib (Python 3.11+) or tomli for TOML support
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None

# Simple TOML writer function to avoid tomli_w dependency
def _write_toml(data: Dict[str, Any], file_path: Path) -> None:
    """Write data to TOML file using raw Python."""
    lines = []
    
    def _format_value(value: Any) -> str:
        if isinstance(value, str):
            return f'"{value}"'
        elif isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, (int, float)):
            return str(value)
        else:
            return f'"{str(value)}"'
    
    def _write_section(section_name: str, section_data: Dict[str, Any]) -> None:
        lines.append(f"[{section_name}]")
        for key, value in section_data.items():
            if isinstance(value, str) and "#" in key:
                # Handle comments
                lines.append(f"{key} = {_format_value(value)}")
            else:
                lines.append(f"{key} = {_format_value(value)}")
        lines.append("")  # Empty line after section
    
    # Add header comment
    lines.append("# PyneCore API Configuration")
    lines.append("# This is the default configuration file for PyneCore API integration")
    lines.append("")
    
    # Write sections
    for key, value in data.items():
        if isinstance(value, dict):
            _write_section(key, value)
        else:
            lines.append(f"{key} = {_format_value(value)}")
    
    # Write to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


@dataclass
class APIConfig:
    """Configuration for PyneSys API client."""
    
    api_key: str
    base_url: str = "https://api.pynesys.io"
    timeout: int = 30
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APIConfig":
        """Create config from dictionary."""
        # Handle both flat format and [api] section format
        if "api" in data:
            api_data = data["api"]
            api_key = api_data.get("pynesys_api_key") or api_data.get("api_key")
            base_url = api_data.get("base_url", "https://api.pynesys.io")
            timeout = api_data.get("timeout", 30)
        else:
            api_key = data.get("pynesys_api_key") or data.get("api_key")
            base_url = data.get("base_url", "https://api.pynesys.io")
            timeout = data.get("timeout", 30)
        
        if not api_key:
            raise ValueError("API key is required (pynesys_api_key or api_key)")
        
        return cls(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_env(cls) -> "APIConfig":
        """Create config from environment variables."""
        api_key = os.getenv("PYNESYS_API_KEY")
        if not api_key:
            raise ValueError("PYNESYS_API_KEY environment variable is required")
        
        return cls(
            api_key=api_key,
            base_url=os.getenv("PYNESYS_BASE_URL", "https://api.pynesys.io"),
            timeout=int(os.getenv("PYNESYS_TIMEOUT", "30"))
        )
    
    @classmethod
    def from_file(cls, config_path: Path) -> "APIConfig":
        """Load configuration from TOML file.
        
        Args:
            config_path: Path to TOML configuration file
            
        Returns:
            APIConfig instance
            
        Raises:
            ValueError: If file doesn't exist or has invalid format
        """
        if not config_path.exists():
            raise ValueError(f"Configuration file not found: {config_path}")
        
        try:
            content = config_path.read_text()
            import tomllib
            data = tomllib.loads(content)
            return cls.from_dict(data)
            
        except Exception as e:
            raise ValueError(f"Failed to parse TOML configuration file {config_path}: {e}")
    
    def save_to_file(self, config_path: Path) -> None:
        """Save configuration to TOML file.
        
        Args:
            config_path: Path to save TOML configuration file
        """
        # Create parent directories if they don't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as TOML with [api] section using raw Python
        data = {
            "api": {
                "pynesys_api_key": self.api_key,
                "timeout": self.timeout
            }
        }
        _write_toml(data, config_path)


class ConfigManager:
    """Manages API configuration loading and saving."""
    
    DEFAULT_CONFIG_PATH = Path("workdir/config/api.toml")
    DEFAULT_FALLBACK_CONFIG_PATH = Path.home() / ".pynecore" / "api.toml"
    
    @classmethod
    def load_config(cls, config_path: Optional[Path] = None) -> APIConfig:
        """Load configuration from various sources.
        
        Priority order:
        1. Provided config_path
        2. Environment variables
        3. Default config file
        
        Args:
            config_path: Optional path to config file
            
        Returns:
            APIConfig instance
            
        Raises:
            ValueError: If no valid configuration found
        """
        # Try provided config path first
        if config_path and config_path.exists():
            return APIConfig.from_file(config_path)
        
        # Try environment variables
        try:
            return APIConfig.from_env()
        except ValueError:
            pass
        
        # Try default config file first, then fallback locations
        if cls.DEFAULT_CONFIG_PATH.exists():
            return APIConfig.from_file(cls.DEFAULT_CONFIG_PATH)
        elif cls.DEFAULT_FALLBACK_CONFIG_PATH.exists():
            return APIConfig.from_file(cls.DEFAULT_FALLBACK_CONFIG_PATH)
        
        raise ValueError(
            f"No configuration file found. Tried:\n"
            f"  - {cls.DEFAULT_CONFIG_PATH} (default)\n"
            f"  - {cls.DEFAULT_FALLBACK_CONFIG_PATH} (fallback)\n"
            f"\nUse 'pyne api configure' to set up your API configuration."
        )
    
    @classmethod
    def save_config(cls, config: APIConfig, config_path: Optional[Path] = None) -> None:
        """Save configuration to file.
        
        Args:
            config: APIConfig instance to save
            config_path: Optional path to save config file (defaults to DEFAULT_CONFIG_PATH)
        """
        path = config_path or cls.DEFAULT_CONFIG_PATH
        config.save_to_file(path)
    
    @classmethod
    def get_default_config_path(cls) -> Path:
        """Get the default configuration file path."""
        return cls.DEFAULT_CONFIG_PATH