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
    # ... ``time`` is a module property: read as a value, callable as well
    assert names['time'] == {'kind': 'value', 'ty': 'i', 'callable': True}
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
    # must not be pushed out of the typed subset -- and a container carries its
    # ELEMENT type, so ``array.get`` on one of these can be answered
    assert names['array.new_float']['ret'] == 'a:f'
    assert names['array.new_int']['ret'] == 'a:i'
    assert names['array.new_line']['ret'] == 'a:o:lib#Line'
    assert names['array.sort_indices']['ret'] == 'a:i'
    # ... while a container the annotation spells without a payload is a plain
    # object: ``matrix.new -> Matrix`` says nothing about what it holds
    assert names['matrix.new']['ret'] == 'o'
    assert names['map.new']['ret'] == 'o'

    # An object-returning lib name carries the CLASS it returns, under the
    # module key reserved for the lib
    assert names['line.new']['impls'][0]['ret'] == 'o:lib#Line'
    assert names['chart.point.new']['ret'] == 'o:lib#ChartPoint'


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


#: The lib names a compiled script reads as VALUES, with the type each carries.
#: One entry per SHAPE the collector has to recognize, not a copy of the corpus:
#: a constant published by an annotated assignment, by a bare constructor call,
#: by a plain literal, and by an alias to another namespace's constant -- plus a
#: namespace instance whose methods are reachable through it. A shape that stops
#: being collected takes every script that mentions it out of the typed subset,
#: silently, which is what this guard is for.
_CONSTANT_COVERAGE = {
    # ``data_window: Display = Display()`` -- object-typed annotation
    'display.all': 'o:lib#Display',
    'display.data_window': 'o:lib#Display',
    'display.none': 'o:lib#Display',
    'display.pane': 'o:lib#Display',
    # ``white = Color('#FFFFFF')`` -- unannotated constructor call, and the one
    # class whose instances are a Pine scalar
    'color.black': 'c',
    'color.blue': 'c',
    'color.gray': 'c',
    'color.green': 'c',
    'color.orange': 'c',
    'color.red': 'c',
    'color.white': 'c',
    'color.yellow': 'c',
    # ... the same shape over every enum namespace, each carrying the class it
    # is an instance of, so two constants of DIFFERENT namespaces no longer
    # join to one anonymous object
    'barmerge.gaps_off': 'o:lib#BarMerge',
    'barmerge.lookahead_off': 'o:lib#BarMerge',
    'extend.both': 'o:lib#Extend',
    'extend.none': 'o:lib#Extend',
    'extend.right': 'o:lib#Extend',
    'format.mintick': 'o:lib#Format',
    'hline.style_dashed': 'o:lib#HLineEnum',
    'hline.style_dotted': 'o:lib#HLineEnum',
    'hline.style_solid': 'o:lib#HLineEnum',
    'line.style_dashed': 'o:lib#LineEnum',
    'line.style_dotted': 'o:lib#LineEnum',
    'line.style_solid': 'o:lib#LineEnum',
    'location.absolute': 'o:lib#Location',
    'plot.style_columns': 'o:lib#PlotEnum',
    'plot.style_cross': 'o:lib#PlotEnum',
    'position.top_right': 'o:lib#Position',
    'shape.circle': 'o:lib#Shape',
    'size.normal': 'o:lib#Size',
    'text.align_center': 'o:lib#AlignEnum',
    'xloc.bar_index': 'o:lib#XLoc',
    'xloc.bar_time': 'o:lib#XLoc',
    'yloc.price': 'o:lib#YLoc',
    # ``islast = False`` -- a plain literal assignment
    'barstate.isfirst': 'b',
    'barstate.islast': 'b',
    'barstate.isrealtime': 'b',
    # ``long = direction.long`` -- an alias into another namespace
    'strategy.long': 'o:lib#Direction',
    'strategy.short': 'o:lib#Direction',
    # An annotated primitive, and a module property read as a value
    'syminfo.mintick': 'f',
    'syminfo.pricescale': 'i',
    'syminfo.tickerid': 's',
    'bar_index': 'i',
    # ``point = _ChartPoint()`` -- a namespace that is an instance
    'chart.point': 'o:lib#_ChartPoint',
}

#: The lib calls a script makes as STATEMENTS. A body with no ``return
#: <value>`` returns nothing, and reading that as unknown pushed every script
#: that plots or trades out of the typed subset.
_VOID_COVERAGE = (
    'alertcondition',
    'line.delete',
    'runtime.error',
    'strategy.cancel',
    'strategy.close',
    'strategy.close_all',
    'strategy.entry',
    'strategy.exit',
    'strategy.order',
)


def __test_every_constant_shape_is_collected__():
    """Each way a lib namespace publishes a constant reaches the registry"""
    names = json.loads(_JSON_PATH.read_text())['names']
    missing = [name for name in _CONSTANT_COVERAGE if name not in names]
    assert not missing, f"the registry lost these constants: {missing}"
    wrong = {name: names[name] for name, ty in _CONSTANT_COVERAGE.items()
             if names[name] != {'kind': 'value', 'ty': ty}}
    assert not wrong, wrong


def __test_a_statement_call_returns_void__():
    """A lib function with no returned value is VOID, not unknown"""
    names = json.loads(_JSON_PATH.read_text())['names']
    wrong = {name: names[name]['ret'] for name in _VOID_COVERAGE
             if names[name]['ret'] != 'v'}
    assert not wrong, wrong
    # ... while one that DOES return a value keeps that value's type
    assert names['plot']['ret'] == 'o:lib#Plot'
    assert names['chart.point.from_index']['ret'] == 'o:lib#ChartPoint'
    assert names['chart.point.new']['params'] == ['i', 'i', 'f']


def __test_a_builtin_class_publishes_its_fields__():
    """
    A ``chart.point`` knows its class, so ``p.price`` has the field's type.

    A builtin class says what it holds in the type package rather than in a
    module interface, so the registry is where the inference can read it --
    the same extraction as the returns.
    """
    classes = json.loads(_JSON_PATH.read_text())['classes']

    # The one Pine scripts actually read fields off
    assert classes['ChartPoint'] == {'index': 'i', 'time': 'i', 'price': 'f'}
    # A field whose own type is a class carries the class id
    assert classes['Line']['xloc'] == 'o:lib#XLoc'
    assert classes['Line']['color'] == 'c'


def __test_a_container_read_is_left_unknown__():
    """
    Element typing is a separate decision, and guessing one is worse than none.

    ``array.get`` returns whatever the array holds, which the ANNOTATION
    cannot say -- the element type comes from the array at the call site, and
    the inference reads it there (``LIB_TYPE_OVERRIDES``'s ``'elem0'``). The
    registry itself must keep saying nothing: a guess baked in here would
    reach the enclosing overload pin and select an implementation the runtime
    would not.
    """
    names = json.loads(_JSON_PATH.read_text())['names']
    for name in ('array.get', 'array.pop', 'array.last', 'array.shift', 'array.remove'):
        assert names[name]['ret'] == '?', name
