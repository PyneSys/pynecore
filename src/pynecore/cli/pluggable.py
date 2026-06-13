"""
Click Command subclass that supports dynamic parameter injection by plugins.

Typer generates Click commands internally.  By passing ``cls=PluggableCommand``
to ``@app.command()``, plugins can register extra ``--flags`` that appear in
``--help`` and are parsed alongside built-in parameters.

Plugin parameters are separated from core parameters before the callback is
invoked, so the original function signature does not need to change.  The
injected values are stored on ``ctx.plugin_params``.

Typer rebuilds the whole Click command tree on every invocation
(``typer.main.get_command`` is not cached), so registrations cannot live on a
single command instance — they would be lost the next time the tree is built.
The registry is therefore class-level, keyed by command name, and every rebuilt
:class:`PluggableCommand` reads its injected parameters back from it.
"""

import click
from typer.core import TyperCommand


class PluggableCommand(TyperCommand):
    """
    A Typer-compatible Click command that allows plugins to inject parameters.

    Usage::

        @app.command(cls=PluggableCommand)
        def run(ctx: typer.Context, script: Path = ...):
            live = ctx.plugin_params.get('live', False)

    After the command is registered, call :meth:`register_plugin_param` to add
    plugin-provided options/arguments.
    """

    # Command-name -> injected params. Class-level so it survives Typer
    # rebuilding the command tree on each invocation. Command names are unique
    # leaves (e.g. "run", "download"), so the leaf name is a safe key.
    _plugin_param_registry: dict[str, list[click.Parameter]] = {}

    def register_plugin_param(self, param: click.Parameter) -> bool:
        """
        Register a plugin-provided parameter for this command.

        Checks both parameter names and option strings (e.g. ``--from``, ``-f``)
        against the core parameters and already-registered plugin parameters to
        prevent conflicts.

        :param param: A Click Parameter (typically ``click.Option``).
        :return: ``False`` if the name or any option string conflicts.
        """
        registered = self._plugin_param_registry.setdefault(self.name or "", [])
        all_params = [*self.params, *registered]

        existing_names = {p.name for p in all_params}
        if param.name in existing_names:
            return False

        existing_opts = {opt for p in all_params for opt in getattr(p, 'opts', ())}
        new_opts = set(getattr(param, 'opts', ()))
        if existing_opts & new_opts:
            return False

        registered.append(param)
        return True

    def _plugin_params(self) -> list[click.Parameter]:
        """Injected parameters registered for this command name."""
        return self._plugin_param_registry.get(self.name or "", [])

    def get_params(self, ctx: click.Context) -> list[click.Parameter]:
        """Return core params + plugin params + help option."""
        rv = [*self.params, *self._plugin_params()]
        help_option = self.get_help_option(ctx)
        if help_option is not None:
            rv.append(help_option)
        return rv

    def invoke(self, ctx: click.Context) -> None:
        """Pop plugin params from ctx.params before calling the callback."""
        ctx.plugin_params = {}
        for p in self._plugin_params():
            if p.name in ctx.params:
                ctx.plugin_params[p.name] = ctx.params.pop(p.name)
        return super().invoke(ctx)
