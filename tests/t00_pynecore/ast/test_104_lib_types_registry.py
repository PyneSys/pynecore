"""
Freshness guard for lib_types.json.

The Pine type inference resolves every lib name through this registry, so a
stale one silently changes which overload a call site pins and which slots the
inference believes are int-typed. The committed JSON must always match what
scripts/lib_type_collector.py generates from the current lib source.

The spot checks below are the second half of the guard: they pin the handful
of entries the inference leans on hardest, so a collector change that keeps
the file self-consistent but flips a type still fails here.
"""
import importlib.util
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_JSON_PATH = _REPO_ROOT / 'src' / 'pynecore' / 'transformers' / 'lib_types.json'


def _load_collector():
    collector_path = _REPO_ROOT / 'scripts' / 'lib_type_collector.py'
    spec = importlib.util.spec_from_file_location('lib_type_collector', collector_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def __test_lib_types_json_is_current__():
    """The committed registry is what the collector produces today"""
    module = _load_collector()
    collector = module.LibTypeCollector(project_src=_REPO_ROOT / 'src')
    generated = collector.collect()
    committed = json.loads(_JSON_PATH.read_text())

    assert generated == committed, \
        "lib_types.json is stale — rerun scripts/lib_type_collector.py"


def __test_schema_version_matches_the_collector__():
    """A shape change must bump the version the consumers pin"""
    module = _load_collector()
    committed = json.loads(_JSON_PATH.read_text())
    assert committed['v'] == module.SCHEMA_VERSION


def __test_load_bearing_entries__():
    """The names the inference leans on carry the types it expects"""
    names = json.loads(_JSON_PATH.read_text())['names']

    # Builtin series and counters: the int/float split the whole pass exists for
    assert names['bar_index'] == {'kind': 'value', 'ty': 'i'}
    assert names['last_bar_index'] == {'kind': 'value', 'ty': 'i'}
    assert names['time'] == {'kind': 'value', 'ty': 'i'}
    for source in ('open', 'high', 'low', 'close', 'volume'):
        assert names[source] == {'kind': 'value', 'ty': 'f'}, source

    # Rolling machines return floats whatever their length is
    assert names['ta.sma']['ret'] == 'f'
    assert names['ta.ema']['ret'] == 'f'

    # An overload group must survive as a group: the pin picks between these
    assert names['ta.highest']['kind'] == 'overloads'
    assert len(names['ta.highest']['impls']) == 2
    assert names['math.round']['kind'] == 'overloads'

    # A module that defines its own name is reachable the way a script spells it
    assert 'plot' in names and names['plot']['kind'] == 'function'
    # ... and a name re-exported from a private machine module is present
    assert names['math.sum']['kind'] == 'function'
    assert names['math.random']['kind'] == 'function'

    # Known non-scalars are KNOWN, not typing failures: an array-using script
    # must not be pushed out of the typed subset
    assert names['array.new_float']['ret'] == 'o'
    assert names['matrix.new']['ret'] == 'o'


def __test_a_none_default_records_what_its_annotation_takes__():
    """
    A ``None`` default is decided by the ANNOTATION, not by its type character.

    ``int`` and ``int | None`` are the same Pine type and only the second one
    takes the ``None`` the selector binds into it, so the registry carries the
    annotation's answer alongside the default's own character.
    """
    names = json.loads(_JSON_PATH.read_text())['names']

    # ``table.clear(.., end_column: int = None, end_row: int = None)``
    assert names['table.clear']['default_ty'] == ['0', '0']
    assert names['table.clear']['default_none_ok'] == [False, False]

    # ``barcolor(.., show_last: int | None = None, title: str | None = None, ..)``
    assert names['barcolor']['default_ty'] == ['0', 'i', 'b', '0', '0', '0']
    assert names['barcolor']['default_none_ok'] == [True, False, False, True, True, True]

    # A group with no ``None`` default carries no flags at all
    assert 'default_none_ok' not in names['ta.highest']['impls'][0]


def __test_union_annotations_do_not_collapse__():
    """
    A conflicting union member must not be swallowed by an absence marker.

    ``int | float | str | bool | NA`` is a genuine conflict and has to come out
    UNKNOWN. Treating every unrecognized member as "absent" let the LAST member
    win instead, which typed ``str.tostring``'s value parameter as bool.
    """
    names = json.loads(_JSON_PATH.read_text())['names']
    assert names['string.tostring']['params'][0] == '?'
    assert names['string.tostring']['ret'] == 's'
