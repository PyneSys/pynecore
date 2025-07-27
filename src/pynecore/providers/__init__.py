from pathlib import Path
from typing import Dict, Type, Optional

from .ccxt import CCXTProvider
from .capitalcom import CapitalComProvider
from .provider import Provider

# Built-in providers
_builtin_providers = {
    'ccxt': CCXTProvider,
    'capitalcom': CapitalComProvider,
}

# Cache for loaded plugin providers
_plugin_providers: Dict[str, Type[Provider]] = {}


def get_available_providers() -> tuple[str, ...]:
    """Get list of all available providers (built-in + plugins)
    
    Returns:
        Tuple of provider names
    """
    # Get built-in providers
    builtin = set(_builtin_providers.keys())
    
    # Get plugin providers
    plugin_names = set()
    try:
        # Import here to avoid circular imports
        from ..cli.plugin_manager import plugin_manager
        plugin_names = set(plugin_manager.get_available_providers())
    except ImportError:
        # Plugin manager not available (e.g., in minimal installations)
        pass
    
    return tuple(sorted(builtin | plugin_names))


def get_provider_class(provider_name: str) -> Optional[Type[Provider]]:
    """Get provider class by name
    
    Args:
        provider_name: Name of the provider
        
    Returns:
        Provider class or None if not found
    """
    # Check built-in providers first
    if provider_name in _builtin_providers:
        return _builtin_providers[provider_name]
    
    # Check cached plugin providers
    if provider_name in _plugin_providers:
        return _plugin_providers[provider_name]
    
    # Try to load from plugins
    try:
        from ..cli.plugin_manager import plugin_manager
        provider_class = plugin_manager.load_plugin(provider_name)
        if provider_class:
            # Cache the loaded provider
            _plugin_providers[provider_name] = provider_class
            return provider_class
    except ImportError:
        # Plugin manager not available
        pass
    
    return None


# Legacy support - dynamically generate available_providers
# This maintains backward compatibility with existing code
available_providers = get_available_providers()


# Re-export for convenience
__all__ = [
    'Provider',
    'CCXTProvider', 
    'CapitalComProvider',
    'available_providers',
    'get_available_providers',
    'get_provider_class',
]
