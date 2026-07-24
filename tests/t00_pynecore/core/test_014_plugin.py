"""
Tests for the plugin discovery and loading system.
"""

from dataclasses import dataclass
from importlib.metadata import EntryPoint

from pynecore.core.plugin import (
    PLUGIN_GROUP,
    parse_pynecore_requirement,
    min_pynecore_version,
    normalize_package_name,
    Plugin,
    ProviderPlugin,
    ProviderError,
    TransientProviderError,
    is_retryable_provider_error,
    LiveProviderConfig,
    LiveProviderPlugin,
    PluginSymbol,
    CLIPlugin,
    discover_plugins,
    load_plugin,
    get_available_plugin_names,
    get_plugin_metadata,
    PluginNotFoundError,
    PluginNameConflictError,
    discover_plugin_entry_points,
)


def __test_discover_plugins_returns_dict__():
    """discover_plugins returns a dict of installed plugins"""
    result = discover_plugins()
    assert isinstance(result, dict)


def __test_discover_ccxt__():
    """discover_plugins finds CCXT provider via pyne.plugin entry point"""
    result = discover_plugins()
    assert "ccxt" in result, (
        f"CCXT not found in pyne.plugin entry points. "
        f"Available: {list(result.keys())}. "
        f"Make sure pynecore is installed with: pip install -e pynecore/"
    )


def __test_load_plugin_ccxt__():
    """load_plugin loads the CCXTProvider class"""
    from pynecore.providers.ccxt import CCXTProvider

    cls = load_plugin("ccxt")
    assert cls is CCXTProvider


def __test_load_plugin_not_found__():
    """load_plugin raises PluginNotFoundError for missing plugin"""
    try:
        load_plugin("nonexistent_provider_xyz")
        assert False, "Should have raised PluginNotFoundError"
    except PluginNotFoundError as e:
        assert "nonexistent_provider_xyz" in str(e)
        assert "Available plugins:" in str(e)
        # No install command may be guessed from the plugin name: the package
        # name is not derivable from it and the guessed names are unclaimed
        # on PyPI (typosquat bait).
        assert "pip install" not in str(e)


def __test_name_conflict_only_fails_the_requested_name__(monkeypatch):
    """A colliding entry-point name is listed, but only breaks its own load.

    Two packages may declare the same ``pyne.plugin`` name. Discovery must
    survive that (a blanket failure would take down every plugin lookup and
    halt a running bot); only loading the ambiguous name itself may fail.
    """
    from pynecore.core import plugin as plugin_module

    def _both(ep_name: str, value: str, dist_name: str) -> EntryPoint:
        ep = EntryPoint(ep_name, value, PLUGIN_GROUP)
        dist = _FakeDist(None)
        dist.name = dist_name
        dist.metadata = {'Name': dist_name}
        object.__setattr__(ep, 'dist', dist)
        return ep

    fake_eps = [
        _both('dup', 'pkg_z:X', 'zz-plugin'),
        _both('dup', 'pkg_a:Y', 'aa-plugin'),
        _both('solo', 'pkg_s:Z', 'solo-plugin'),
    ]
    monkeypatch.setattr(plugin_module, 'entry_points', lambda group=None: fake_eps)

    grouped = discover_plugin_entry_points()
    assert [p.value for p in grouped['dup']] == ['pkg_a:Y', 'pkg_z:X']  # deterministic order
    assert len(grouped['solo']) == 1

    # The single-winner view never raises, so unrelated plugins stay loadable.
    assert set(discover_plugins()) == {'dup', 'solo'}

    try:
        load_plugin('dup')
        assert False, "Should have raised PluginNameConflictError"
    except PluginNameConflictError as e:
        assert 'aa-plugin' in str(e) and 'zz-plugin' in str(e)


def __test_get_available_plugin_names__():
    """get_available_plugin_names returns sorted list including ccxt"""
    names = get_available_plugin_names()
    assert isinstance(names, list)
    assert "ccxt" in names
    assert names == sorted(names)


def __test_ccxt_is_provider__():
    """CCXTProvider inherits from Plugin and Provider"""
    cls = load_plugin("ccxt")
    assert issubclass(cls, Plugin)
    assert issubclass(cls, ProviderPlugin)


def __test_plugin_metadata__():
    """get_plugin_metadata extracts metadata from pyproject.toml"""
    plugins = discover_plugins()
    assert "ccxt" in plugins
    meta = get_plugin_metadata(plugins["ccxt"])
    assert meta['name'] == 'ccxt'
    assert meta['version']  # should be non-empty
    assert meta['package'] == 'pynesys-pynecore'


class _FakeDist:
    """Minimal stand-in for an installed distribution."""

    def __init__(self, requires: list[str] | None):
        self.requires = requires
        self.name = 'fake-plugin'
        self.metadata = {'Name': 'fake-plugin'}


def _fake_entry_point(requires: list[str] | None) -> EntryPoint:
    ep = EntryPoint('fake', 'fake_module:FakePlugin', PLUGIN_GROUP)
    object.__setattr__(ep, 'dist', _FakeDist(requires))
    return ep


def __test_parse_pynecore_requirement_forms__():
    """The PyneCore dependency is recognised in every PEP 508 spelling."""
    cases = {
        'pynesys-pynecore>=6.6.0': '>=6.6.0',
        'pynesys-pynecore >= 6.6.0': '>=6.6.0',           # whitespace
        'pynesys-pynecore>=6.6.0,<7': '>=6.6.0,<7',       # upper bound kept
        'pynesys-pynecore~=6.6': '~=6.6',                 # compatible release
        'pynesys-pynecore (>=6.6.0,<7)': '>=6.6.0,<7',    # parenthesized
        'pynesys_pynecore>=6.6.0': '>=6.6.0',             # non-normalized name
        'PyneSys.PyneCore>=6.6': '>=6.6',
        'pynesys-pynecore[all]>=6.6.0': '>=6.6.0',        # extras
        "pynesys-pynecore>=6.6.0 ; python_version >= '3.11'": '>=6.6.0',  # marker
        'pynesys-pynecore': '',                           # no constraint
    }
    for requirement, expected in cases.items():
        assert parse_pynecore_requirement(_fake_entry_point([requirement])) == expected, requirement


def __test_parse_pynecore_requirement_ignores_others__():
    """Only the PyneCore dependency is read, and a missing one yields ''."""
    assert parse_pynecore_requirement(_fake_entry_point(['httpx>=1.0'])) == ''
    assert parse_pynecore_requirement(_fake_entry_point([])) == ''
    assert parse_pynecore_requirement(_fake_entry_point(None)) == ''
    ep = _fake_entry_point(['httpx>=1.0', 'pynesys-pynecore>=6.6.0'])
    assert parse_pynecore_requirement(ep) == '>=6.6.0'


def __test_min_pynecore_version__():
    """The lower bound is picked numerically, not lexically."""
    # PyneCore versions are PineVersion.Major.Minor — three numeric components,
    # so a string compare would rank 6.9.0 above 6.10.0.
    assert min_pynecore_version('>=6.9.0,>=6.10.0') == '6.10.0'
    assert min_pynecore_version('>=6.6.0,<7') == '6.6.0'
    assert min_pynecore_version('==6.6.1') == '6.6.1'
    assert min_pynecore_version('~=6.6') == '6.6'
    assert min_pynecore_version('==6.6.*') == '6.6'
    # Upper bounds and exclusions carry no lower bound.
    assert min_pynecore_version('<7') == ''
    assert min_pynecore_version('!=6.6.0') == ''
    assert min_pynecore_version('') == ''


def __test_normalize_package_name__():
    """PEP 503 normalization makes the distribution name comparable."""
    assert normalize_package_name('PyneSys_PyneCore') == 'pynesys-pynecore'
    assert normalize_package_name(' pynesys.pynecore ') == 'pynesys-pynecore'
    assert normalize_package_name('pynesys--pynecore') == 'pynesys-pynecore'


def __test_plugin_base_defaults__():
    """Plugin base class has minimal attributes, CLIPlugin has CLI defaults"""
    assert Plugin.Config is None
    assert Plugin.plugin_name == ""
    assert CLIPlugin.cli() is None
    assert CLIPlugin.cli_params('run') == []


def __test_resolve_symbol_with_map__():
    """resolve_symbol() consults config.symbol_map first."""

    @dataclass
    class _Cfg(LiveProviderConfig):
        pass

    class _StubProvider(ProviderPlugin):
        Config = _Cfg

        @classmethod
        def to_tradingview_timeframe(cls, timeframe: str) -> str:
            return timeframe

        @classmethod
        def to_exchange_timeframe(cls, timeframe: str) -> str:
            return timeframe

        def get_list_of_symbols(self, *args, **kwargs):
            return []

        def update_symbol_info(self):  # type: ignore[override]
            raise NotImplementedError

        def download_ohlcv(self, time_from, time_to, on_progress=None, limit=None):
            raise NotImplementedError

        def normalize_symbol(self, symbol: str) -> str:
            # CCXT-style: returns ``self.symbol`` regardless of input —
            # i.e. an instance-bound normalizer. The base ``resolve_symbol``
            # deliberately bypasses this for unmapped keys.
            return self.symbol or symbol

    cfg = _Cfg(symbol_map={"FX:EURUSD": "EURUSD_NATIVE"})
    p = _StubProvider(symbol="ANY", timeframe="1D", config=cfg)
    # Mapped key wins
    assert p.resolve_symbol("FX:EURUSD") == "EURUSD_NATIVE"
    # Unmapped key forwards the pine key unchanged — never the instance's
    # own symbol, which would silently misroute cross-symbol requests.
    assert p.resolve_symbol("FX:GBPUSD") == "FX:GBPUSD"


def __test_plugin_symbol_is_picklable__():
    """PluginSymbol is a frozen dataclass picklable across spawn."""
    import pickle

    cfg = LiveProviderConfig(symbol_map={"FX:X": "Y"})
    ps = PluginSymbol(
        provider_name="cc",
        symbol="EURUSD",
        timeframe="1h",
        config=cfg,
    )
    blob = pickle.dumps(ps)
    rt = pickle.loads(blob)
    assert rt == ps
    assert rt.config.symbol_map == {"FX:X": "Y"}


def __test_live_provider_config_default_symbol_map__():
    """LiveProviderConfig default symbol_map is an empty mutable dict."""
    a = LiveProviderConfig()
    b = LiveProviderConfig()
    # Each instance gets its own dict (default_factory), not a shared one.
    a.symbol_map["k"] = "v"
    assert b.symbol_map == {}


def __test_provider_error_not_retryable_by_default__():
    """A plain ProviderError is permanent: retry must not be attempted."""
    assert ProviderError("bad symbol").retryable is False
    assert is_retryable_provider_error(ProviderError("bad symbol")) is False


def __test_transient_provider_error_is_retryable__():
    """TransientProviderError marks a retry-worthy connectivity fault."""
    assert TransientProviderError("link dropped").retryable is True
    assert is_retryable_provider_error(TransientProviderError("link dropped")) is True


def __test_is_retryable_walks_cause_chain__():
    """A transient error wrapped in a plain ProviderError is still retryable."""
    transient = TransientProviderError("maintenance")
    try:
        try:
            raise transient
        except TransientProviderError as inner:
            raise ProviderError("download failed") from inner
    except ProviderError as outer:
        assert is_retryable_provider_error(outer) is True


def __test_is_retryable_walks_context_chain__():
    """A transient error in the implicit (__context__) chain is still retryable."""
    try:
        try:
            raise TransientProviderError("maintenance")
        except TransientProviderError:
            # No ``from`` — links via __context__, not __cause__.
            raise ProviderError("download failed")
    except ProviderError as outer:
        assert is_retryable_provider_error(outer) is True


def __test_is_retryable_non_provider_error_is_false__():
    """A non-ProviderError exception is never classified retryable."""
    assert is_retryable_provider_error(ValueError("boom")) is False
