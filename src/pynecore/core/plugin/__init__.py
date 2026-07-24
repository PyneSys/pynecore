"""
Plugin base class, discovery, and loading via Python entry points.

All PyneCore plugins register under a single entry point group
(``pyne.plugin``) in their ``pyproject.toml``.  The plugin class hierarchy
determines capabilities:

- ``ProviderPlugin(Plugin)`` — offline OHLCV data provider
- ``LiveProviderPlugin(ProviderPlugin)`` — offline + WebSocket/streaming data
- ``CLIPlugin(Plugin)`` — CLI commands and parameter hooks

Multiple inheritance combines capabilities::

    class YahooPlugin(ProviderPlugin, CLIPlugin): ...          # offline only
    class BinancePlugin(LiveProviderPlugin, CLIPlugin): ...    # offline + live

Plugin metadata (name, version) comes from the package's ``pyproject.toml``
via :mod:`importlib.metadata`, not from class attributes.

Example ``pyproject.toml``::

    [project.entry-points."pyne.plugin"]
    myexchange = "mypackage:MyExchangeProvider"

Discovery::

    plugins = discover_plugins()
    cls = load_plugin("capitalcom")
"""

import inspect
import re
import sys
from typing import TypeVar, Generic, Any

# noinspection PyProtectedMember
from importlib.metadata import entry_points, EntryPoint

if sys.version_info >= (3, 12):
    # noinspection PyUnusedImports
    from typing import override
else:
    def override(func):
        """Marks a method as overriding a base class method (polyfill for <3.12)."""
        return func

PLUGIN_GROUP = 'pyne.plugin'

ConfigT = TypeVar('ConfigT')


class Plugin(Generic[ConfigT]):
    """
    Minimal base class for all PyneCore plugins.

    Generic over the config dataclass type.  Plugin authors should inherit
    from a concrete subclass: :class:`ProviderPlugin`, :class:`LiveProviderPlugin`,
    :class:`~pynecore.core.plugin.broker.BrokerPlugin`, :class:`CLIPlugin`,
    or a combination via multiple inheritance.

    Example::

        class MyProvider(ProviderPlugin[MyConfig]):
            Config = MyConfig
    """

    Config: type[ConfigT] | None = None
    """Override with a ``@dataclass`` for plugin configuration."""

    plugin_name: str = ""
    """Optional display name override.  If empty, the entry point name is used."""

    plugin_params: dict[str, Any] = {}
    """Values of plugin-injected CLI flags for the running command, populated by
    the core CLI before invoking the plugin (see :class:`CLIPlugin.cli_params`).
    Each plugin reads only its own keys; empty when no command set it."""


class PluginNotFoundError(ImportError):
    """Raised when a requested plugin is not installed."""


class PluginNameConflictError(ImportError):
    """Raised when the requested plugin name is provided by several packages.

    Entry point names are not unique — any package may register the same
    ``pyne.plugin`` name — so loading one of them by guesswork could run
    foreign code under a trusted name.
    """


class ProviderError(Exception):
    """Base class for *expected*, user-actionable data-provider failures.

    Covers conditions a user can act on — missing/invalid credentials, a
    connection that could not be opened, an unknown broker — as opposed to
    programming errors. The ``pyne data`` CLI catches this (alongside
    :class:`NotImplementedError`) on its listing/download paths and prints a
    one-line ``Error: ...`` instead of a traceback. A plugin's own error
    hierarchy should subclass it so those failures surface cleanly.

    :cvar retryable: Whether the same operation could plausibly succeed if
        retried later — a *transient* connectivity / broker-unavailable fault,
        as opposed to a permanent misconfiguration (unknown symbol, bad
        credentials, wrong account mode). Defaults to ``False`` so an unknown
        failure fails fast rather than looping forever; genuine transient
        faults set it via :class:`TransientProviderError` or a ``retryable``
        property keyed on an error code.
    """

    retryable: bool = False


class TransientProviderError(ProviderError):
    """A *transient* provider failure — the operation may succeed on retry.

    Connectivity drops, request-routing failures (broker maintenance), socket
    timeouts. A long-running ``--broker`` / ``--live`` run should wait and
    retry rather than halt; a one-shot backtest still fails fast. See
    :func:`is_retryable_provider_error`.
    """

    retryable: bool = True


def is_retryable_provider_error(exc: BaseException) -> bool:
    """Whether ``exc`` is a transient, retry-worthy provider failure.

    Walks the ``__cause__`` / ``__context__`` chain so a transient error stays
    classifiable even after being re-wrapped in another exception (the
    original wire fault is often nested under a higher-level
    :class:`ProviderError`). Returns ``True`` as soon as any link is a
    :class:`ProviderError` whose ``retryable`` flag is set.

    :param exc: The caught exception.
    :return: ``True`` if waiting and retrying the operation could plausibly
        succeed, ``False`` for permanent / unknown failures.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ProviderError) and current.retryable:
            return True
        current = current.__cause__ or current.__context__
    return False


def get_plugin_package(ep: EntryPoint) -> str:
    """
    Return the distribution (PyPI package) name providing an entry point.

    :param ep: The entry point of the plugin.
    :return: Package name, or ``"(unknown package)"`` if the distribution is
        not resolvable.
    """
    if ep.dist is None:
        return '(unknown package)'
    return ep.dist.metadata['Name'] or ep.dist.name or '(unknown package)'


def discover_plugin_entry_points() -> dict[str, list[EntryPoint]]:
    """
    Return every installed plugin entry point, grouped by plugin name.

    Entry point names are **not** unique: any package may register e.g.
    ``bybit`` under the ``pyne.plugin`` group, so a name may map to several
    entry points.  Conflicts are reported here, never resolved silently.

    :return: Mapping of plugin name to all entry points declaring it, each
        list ordered deterministically by providing package.
    """
    grouped: dict[str, list[EntryPoint]] = {}
    for ep in entry_points(group=PLUGIN_GROUP):
        grouped.setdefault(ep.name, []).append(ep)
    for eps in grouped.values():
        eps.sort(key=lambda ep: (get_plugin_package(ep), ep.value))
    return grouped


def discover_plugins() -> dict[str, EntryPoint]:
    """
    Return all installed plugins, one entry point per name.

    On a name conflict a deterministic winner is picked (first by providing
    package name) so listings stay stable.  This function never raises on a
    conflict on purpose: two unrelated third-party plugins colliding must not
    take down every plugin lookup — that would halt a running bot.  Only
    :func:`load_plugin` refuses to guess, and only for the requested name.

    :return: Mapping of plugin name to a single :class:`EntryPoint`.
    """
    return {name: eps[0] for name, eps in discover_plugin_entry_points().items()}


def load_plugin(name: str) -> type:
    """
    Load and return a plugin class by name.

    :param name: Plugin name as declared in the entry point.
    :return: The plugin class.
    :raises PluginNotFoundError: If no plugin with the given name is installed.
    :raises PluginNameConflictError: If several installed packages declare the
        same plugin name — loading either one would be a guess.
    """
    grouped = discover_plugin_entry_points()
    candidates = grouped.get(name)
    if not candidates:
        raise PluginNotFoundError(
            f"Plugin '{name}' is not installed.\n"
            f"Available plugins: {', '.join(sorted(grouped)) or '(none)'}"
        )
    if len(candidates) > 1:
        packages = ', '.join(get_plugin_package(ep) for ep in candidates)
        raise PluginNameConflictError(
            f"Plugin name '{name}' is ambiguous — declared by {len(candidates)} "
            f"installed packages: {packages}\n"
            f"Uninstall all but the one you want to use."
        )
    return candidates[0].load()


def get_available_plugin_names() -> list[str]:
    """
    Return a sorted list of all available plugin names.

    :return: Sorted list of plugin names.
    """
    return sorted(discover_plugins())


def get_plugin_metadata(ep: EntryPoint) -> dict[str, str]:
    """
    Extract plugin metadata from its package distribution.

    :param ep: The entry point of the plugin.
    :return: Dict with ``name``, ``package``, ``version``, ``description``,
        ``requires_pynecore``, ``min_pynecore``.
    """
    assert ep.dist is not None
    meta = ep.dist.metadata
    specifier = parse_pynecore_requirement(ep)
    return {
        'name': ep.name,
        'package': get_plugin_package(ep),
        'version': meta['Version'] or '',
        'description': meta['Summary'] or '',
        'requires_pynecore': specifier,
        'min_pynecore': min_pynecore_version(specifier),
    }


def get_plugin_summary(cls: type) -> str:
    """
    Return the first paragraph of the plugin class docstring.

    The first paragraph is the text up to the first blank line, with
    internal newlines collapsed to single spaces — suitable for a
    one-line summary in listings.

    :param cls: The plugin class.
    :return: First-paragraph summary, or ``""`` if no docstring.
    """
    doc = inspect.getdoc(cls) or ""
    first_para = doc.split("\n\n", 1)[0].strip()
    return " ".join(first_para.split())


def get_plugin_description(cls: type) -> str:
    """
    Return the full normalized plugin class docstring.

    Uses :func:`inspect.getdoc`, which strips the uniform leading
    indentation per PEP 257.

    :param cls: The plugin class.
    :return: Normalized docstring, or ``""`` if no docstring.
    """
    return inspect.getdoc(cls) or ""


PYNECORE_PACKAGE = 'pynesys-pynecore'

_REQUIREMENT_RE = re.compile(
    r'^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)'  # PEP 508 distribution name
    r'\s*(?:\[(?P<extras>[^]]*)])?'  # optional extras
    r'\s*(?:\((?P<paren>[^)]*)\)|(?P<bare>[^;]*))?'  # specifier, optionally parenthesized
)

_SPECIFIER_RE = re.compile(r'^\s*(?P<op>===|~=|==|!=|<=|>=|<|>)\s*(?P<version>\S+?)\s*$')

# Operators that place a *lower* bound on the version
_LOWER_BOUND_OPS = frozenset(('>=', '==', '===', '~='))


def normalize_package_name(name: str) -> str:
    """
    Normalize a distribution name for comparison (PEP 503).

    ``PyneSys_PyneCore`` and ``pynesys-pynecore`` are the same project.

    :param name: Raw distribution name.
    :return: Lowercase name with runs of ``-``, ``_``, ``.`` collapsed to ``-``.
    """
    return re.sub(r'[-_.]+', '-', name).strip().lower()


def parse_pynecore_requirement(ep: EntryPoint) -> str:
    """
    Return the plugin's declared PyneCore version constraint.

    Reads the ``Requires-Dist`` list and returns the whole specifier set of the
    :data:`PYNECORE_PACKAGE` requirement — ``">=6.6.0"``, ``">=6.6.0,<7"``,
    ``"~=6.6"`` — not just a lower bound, since an upper bound is meaningful:
    PyneCore versions are ``PineVersion.Major.Minor``, so ``<7`` means "Pine v6
    only".  Extras, parenthesized specifiers, environment markers, arbitrary
    whitespace and non-normalized package names are all handled.

    :param ep: The entry point of the plugin.
    :return: Normalized specifier set (e.g. ``">=6.6.0,<7"``), or ``""`` if the
        package does not declare a PyneCore dependency.
    """
    assert ep.dist is not None
    for req in ep.dist.requires or ():
        m = _REQUIREMENT_RE.match(req.split(';', 1)[0])
        if m is None or normalize_package_name(m.group('name')) != PYNECORE_PACKAGE:
            continue
        raw = m.group('paren') or m.group('bare') or ''
        clauses = (_SPECIFIER_RE.match(c) for c in raw.split(',') if c.strip())
        return ','.join(f"{c.group('op')}{c.group('version')}" for c in clauses if c)
    return ''


def min_pynecore_version(specifier: str) -> str:
    """
    Return the lowest PyneCore version allowed by a specifier set.

    Of the lower-bounding clauses (``>=``, ``==``, ``===``, ``~=``) the highest
    version wins; upper bounds and exclusions are ignored.  Versions are
    compared component-wise on their numeric release parts, so ``6.10.0`` is
    correctly greater than ``6.9.0`` — a plain string compare is wrong here.

    :param specifier: Specifier set as returned by
        :func:`parse_pynecore_requirement`.
    :return: Version string (e.g. ``"6.6.0"``), or ``""`` if the specifier set
        has no lower bound.
    """
    best = ''
    for clause in specifier.split(','):
        m = _SPECIFIER_RE.match(clause)
        if m is None or m.group('op') not in _LOWER_BOUND_OPS:
            continue
        version = m.group('version').rstrip('*').rstrip('.')
        if not version:
            continue
        if not best or _version_key(version) > _version_key(best):
            best = version
    return best


def _version_key(version: str) -> tuple[int, ...]:
    """
    Return a comparable key for the numeric release part of a version.

    Pre/post/local suffixes (``6.6.0rc1``, ``6.6.0+local``) are cut off — this
    is used for ordering lower bounds, not for full PEP 440 semantics.

    :param version: Version string.
    :return: Tuple of the leading integer components.
    """
    release = re.match(r'\d+(?:\.\d+)*', version)
    return tuple(int(p) for p in release.group(0).split('.')) if release else ()


# Plugin type subclasses — import after Plugin is defined to avoid circular imports
from .provider import ProviderPlugin, Broker
from .live_provider import LiveProviderPlugin, LiveProviderConfig, PluginSymbol
from .cli import CLIPlugin
