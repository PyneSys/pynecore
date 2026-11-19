"""
Provider hub: one process holds the data providers of several runs.

A provider is expensive to hold: its SDK, the venue's market list and its
connection. A front end that runs many live processes at once (one per chart, per
script and per ``request.security`` context) pays that in every one of them. With a
hub the providers live once, in the hub's process, and every other process reaches
them through a :class:`~pynecore.core.plugin.hub.remote.RemoteProvider`: a
:class:`~pynecore.core.plugin.live_provider.LiveProviderPlugin` that forwards each
call to the hub. The live logic of the core (closed-bar recovery, reconnection, the
boundary watchdog) runs in every process as before, on the forwarded provider.

A process uses the hub when :data:`HUB_ENV` names its address and
:data:`TOKEN_ENV` its token: :func:`~pynecore.core.plugin.load_plugin` then returns
the remote class of a provider plugin instead of the plugin itself. Child processes
inherit the two variables, so the ``request.security`` contexts of a run use the
same hub. The hub serves market data only: no order of a broker plugin can be
reached through it.

Start one with ``python -m pynecore.core.plugin.hub``.
"""

HUB_ENV = "PYNE_PROVIDER_HUB"
"""Environment variable with the hub's address, ``host:port``."""

TOKEN_ENV = "PYNE_PROVIDER_HUB_TOKEN"
"""Environment variable with the token every connection to the hub starts with."""
