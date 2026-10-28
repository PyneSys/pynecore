"""The native ``pine_qualifier`` port and the compiled-in registries against Python.

Every ported function is called through ``_native._qualifier`` and compared with its
Python original over every lib registry key and the edge spellings; every section of
the registries compiled into the extension (``_native._data``) must equal what the
Python steps read.
"""
import ast
import itertools

from pynecore.transformers import pine_qualifier as qualifier
from pynecore.transformers.lib_registry import lib_registry, lib_types
from pynecore.transformers.module_property import module_properties

from .test_137_native_transform import __test_helper_require_native

#: Names the registry does not list, spelled to reach every branch of the rules
__test_helper_NAMES = [
    '', 'close', '.x', 'a.', 'a..b', 'session.isfirstbar', 'session.regular', 'session.',
    'strategy.long', 'strategy.account_currency', 'strategy.opentrades.size', 'ta.tr',
    'dayofweek.dayofweek', 'dayofweek.monday', 'syminfo.mintick', 'syminfo.x.y', 'timeframe',
    'timeframe.period', 'chart.bg_color', 'dividends.gross', 'earnings.actual', 'barstate.isnew',
    'input', 'input.int', 'inputs.int', 'timestamp', 'color.new', 'math.avg', 'ticker.new',
    'ticker', 'str.tostring', 'x.y.z', 'é.ő', 'int', 'abs', 'range', 'len',
]

#: Types a value read is asked about
__test_helper_TYPES = ['i', 'f', 'o:lib#Direction', 'o:lib#', 'o:/x.py#Mode', '?', '']

#: Calls spelling the window length every way
__test_helper_CALLS = [
    'ta.highest(close, 5)', 'ta.highest(5)', 'ta.highest(length=5)', 'ta.highest(close, length=n)',
    'ta.highest(source=close)', 'ta.highest(*args)', 'ta.highest(close, *args)',
    'ta.highest(5, source=close)', 'ta.highest()', 'ta.highest(**kw)', 'ta.highest(a, b, c)',
    'ta.highest(close, 5, length=6)', 'ta.highest(close, **kw, length=6)',
]


def __test_helper_native(name, *args):
    from pynecore.transformers import _native

    return _native._qualifier(name, *args)


def __test_helper_data(section):
    from pynecore.transformers import _native

    return _native._data(section)


def __test_native_qualifier_tables_match__():
    """The qualifiers, attribute names and name sets are the Python module's."""
    __test_helper_require_native()
    for name, value in __test_helper_native('constants').items():
        assert getattr(qualifier, name) == value, name
    for name, value in __test_helper_native('sets').items():
        assert set(getattr(qualifier, name)) == set(value) and len(value) == len(set(value)), name


def __test_native_value_and_call_qualifiers_match__():
    """Every lib name and every edge spelling, with every argument qualifier mix."""
    __test_helper_require_native()
    names = list(lib_types()) + __test_helper_NAMES
    checked = 0
    for name in names:
        entry = lib_types().get(name, {})
        for ty in [entry.get('ty') or 'f', *__test_helper_TYPES]:
            assert __test_helper_native('lib_value_qualifier', name, ty) \
                == qualifier.lib_value_qualifier(name, ty), (name, ty)
            checked += 1
        for count in range(3):
            for arguments in itertools.product(range(4), repeat=count):
                for argc in (None, count, 1):
                    assert __test_helper_native('lib_call_qualifier', name, list(arguments), argc) \
                        == qualifier.lib_call_qualifier(name, arguments, argc), (name, arguments)
                    checked += 1
                assert __test_helper_native('builtin_call_qualifier', name, list(arguments)) \
                    == qualifier.builtin_call_qualifier(name, arguments), (name, arguments)
    assert checked > 10_000


def __test_native_window_length_matches__():
    """The ``length`` argument of every window call form, as the very node Python picks."""
    __test_helper_require_native()
    for source in __test_helper_CALLS:
        call = ast.parse(source, mode='eval').body
        assert isinstance(call, ast.Call)
        for name in sorted(qualifier.WINDOW_CALLS) + ['ta.sma']:
            assert __test_helper_native('window_length', name, call) \
                is qualifier.window_length(name, call), (name, source)


def __test_native_series_len_stamps_match__():
    """The window verdict stamps read and write what the Python accessors do."""
    __test_helper_require_native()
    for verdict in (None, True, False):
        call = ast.parse('ta.lowest(low, n)', mode='eval').body
        assert isinstance(call, ast.Call)
        assert __test_helper_native('get_series_len', call) is qualifier.get_series_len(call)
        qualifier.set_series_len(call, verdict)
        assert __test_helper_native('get_series_len', call) is verdict
        stamped = __test_helper_native('set_series_len', call, verdict)
        assert vars(stamped)[qualifier.SERIES_LEN_ATTR] is verdict
    for verdicts in (None, {}, {0: True, 3: False}):
        call = ast.parse('ta.lowest(low, n)', mode='eval').body
        assert isinstance(call, ast.Call)
        assert __test_helper_native('get_series_lens', call) == qualifier.get_series_lens(call)
        qualifier.set_series_lens(call, verdicts)
        assert __test_helper_native('get_series_lens', call) == verdicts
        stamped = __test_helper_native('set_series_lens', call, verdicts)
        assert qualifier.get_series_lens(stamped) == verdicts


def __test_native_registries_match__():
    """Every compiled-in registry section is the one the Python steps read."""
    __test_helper_require_native()
    registry = lib_registry()
    names = {name: (entry['kind'], entry.get('ty'), bool(entry.get('callable', False)),
                    entry.get('ret'), [impl.get('ret') for impl in entry.get('impls', ())])
             for name, entry in lib_types().items()}
    assert __test_helper_data('names') == names
    assert __test_helper_data('classes') == {name: list(fields.items())
                                            for name, fields in registry['classes'].items()}
    for section in ('routes', 'scalar_classes', 'star_exports'):
        assert __test_helper_data(section) == registry[section], section
    assert __test_helper_data('module_properties') == {
        module: {member: entry['type'] for member, entry in members.items()}
        for module, members in module_properties().items()}
