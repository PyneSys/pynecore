"""
Typer Command subclass that supports dynamic parameter injection by plugins.

Typer builds the command tree from its own ``TyperCommand``/``TyperOption``
classes.  By passing ``cls=PluggableCommand`` to ``@app.command()``, plugins
can register extra ``--flags`` that appear in ``--help`` and are parsed
alongside built-in parameters.

Plugins describe their options with :class:`~pynecore.core.plugin.CLIOption`
and this module turns them into the parser objects Typer expects.  Plugins
never construct those objects themselves: Click is not a PyneCore dependency,
and since Typer 0.26 it is not a Typer dependency either — Typer vendors a
reduced fork of it, whose parser cannot handle foreign ``click.Option``
instances.

Plugin parameters are separated from core parameters before the callback is
invoked, so the original function signature does not need to change.  The
injected values are stored on ``ctx.plugin_params``.

Typer rebuilds the whole command tree on every invocation
(``typer.main.get_command`` is not cached), so registrations cannot live on a
single command instance — they would be lost the next time the tree is built.
The registry is therefore class-level, keyed by command name, and every rebuilt
:class:`PluggableCommand` reads its injected parameters back from it.
"""

from typing import Any, TypeAlias

from typer.core import TyperCommand, TyperOption
# Typer has no public choice type; ``typer.main`` only re-exports this one
# noinspection PyProtectedMember
from typer._types import TyperChoice

from ..core.plugin import CLIOption

__all__ = ['PluggableCommand']

# The parser context class is Typer-version dependent — Click's own before
# Typer 0.26, Typer's vendored fork from 0.26 on — so it cannot be named
# portably here.
Context: TypeAlias = Any


def _build_option(spec: CLIOption) -> TyperOption:
    """
    Build the parser object Typer expects from a backend-agnostic option spec.

    :param spec: The plugin-provided option description.
    :return: A Typer option ready to be added to a command.
    """
    param_type: Any = TyperChoice(list(spec.choices)) if spec.choices is not None else spec.type
    return TyperOption(
        param_decls=list(spec.decls),
        help=spec.help or None,
        default=spec.default,
        # ``None`` leaves the flag/value decision to Typer's own inference
        is_flag=spec.is_flag or None,
        type=param_type,
        metavar=spec.metavar,
        required=spec.required,
        multiple=spec.multiple,
        hidden=spec.hidden,
        envvar=spec.envvar,
        rich_help_panel=spec.rich_help_panel,
    )


class PluggableCommand(TyperCommand):
    """
    A Typer command that allows plugins to inject parameters.

    Usage::

        @app.command(cls=PluggableCommand)
        def run(ctx: typer.Context, script: Path = ...):
            live = ctx.plugin_params.get('live', False)

    After the command is registered, call :meth:`register_plugin_param` to add
    plugin-provided options.
    """

    # Command-name -> injected params. Class-level so it survives Typer
    # rebuilding the command tree on each invocation. Command names are unique
    # leaves (e.g. "run", "download"), so the leaf name is a safe key.
    _plugin_param_registry: dict[str, list[TyperOption]] = {}

    def register_plugin_param(self, spec: CLIOption) -> bool:
        """
        Register a plugin-provided parameter for this command.

        Checks both parameter names and option strings (e.g. ``--from``, ``-f``)
        against the core parameters and already-registered plugin parameters to
        prevent conflicts.

        :param spec: The option to inject.
        :return: ``False`` if the name or any option string conflicts.
        """
        param = _build_option(spec)
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

    def _plugin_params(self) -> list[TyperOption]:
        """Injected parameters registered for this command name."""
        return self._plugin_param_registry.get(self.name or "", [])

    def get_params(self, ctx: Context) -> list[Any]:
        """Return core params + plugin params + help option."""
        rv = [*self.params, *self._plugin_params()]
        help_option = self.get_help_option(ctx)
        if help_option is not None:
            rv.append(help_option)
        return rv

    def invoke(self, ctx: Context) -> None:
        """Pop plugin params from ctx.params before calling the callback."""
        ctx.plugin_params = {}
        for p in self._plugin_params():
            if p.name in ctx.params:
                ctx.plugin_params[p.name] = ctx.params.pop(p.name)
        return super().invoke(ctx)
