"""
Click Command subclass that supports dynamic parameter injection by plugins.

Typer generates Click commands internally.  By passing ``cls=PluggableCommand``
to ``@app.command()``, plugins can register extra ``--flags`` that appear in
``--help`` and are parsed alongside built-in parameters.

Plugin parameters are separated from core parameters before the callback is
invoked, so the original function signature does not need to change.  The
injected values are stored on ``ctx.plugin_params``.
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._plugin_params: list[click.Parameter] = []

    def register_plugin_param(self, param: click.Parameter) -> bool:
        """
        Register a plugin-provided parameter.

        Checks both parameter names and option strings (e.g. ``--from``, ``-f``)
        to prevent conflicts.

        :param param: A Click Parameter (typically ``click.Option``).
        :return: ``False`` if the name or any option string conflicts.
        """
        all_params = [*self.params, *self._plugin_params]

        existing_names = {p.name for p in all_params}
        if param.name in existing_names:
            return False

        existing_opts = {opt for p in all_params for opt in getattr(p, 'opts', ())}
        new_opts = set(getattr(param, 'opts', ()))
        if existing_opts & new_opts:
            return False

        self._plugin_params.append(param)
        return True

    def get_params(self, ctx: click.Context) -> list[click.Parameter]:
        """Return core params + plugin params + help option."""
        rv = [*self.params, *self._plugin_params]
        help_option = self.get_help_option(ctx)
        if help_option is not None:
            rv.append(help_option)
        return rv

    def invoke(self, ctx: click.Context) -> None:
        """Pop plugin params from ctx.params before calling the callback."""
        ctx.plugin_params = {}
        for p in self._plugin_params:
            if p.name in ctx.params:
                ctx.plugin_params[p.name] = ctx.params.pop(p.name)
        return super().invoke(ctx)
