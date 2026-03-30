from pynecore.core.plugin import get_available_plugin_names

available_providers = tuple(get_available_plugin_names('pyne.provider'))
