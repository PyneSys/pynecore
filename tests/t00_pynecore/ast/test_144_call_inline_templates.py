"""
Freshness guard for call_inline_templates.json.

``CallInlineTransformer`` copies the body of a trivial lib wrapper into the call
site from this registry, never from the lib itself, so a stale one silently
emits a body the wrapper no longer has. The committed JSON must always match
what scripts/call_inline_collector.py derives from the current lib sources.

The lib sources are not part of the pipeline hash and this file is, so this
guard is what turns an edit of an inlined body -- or of the line it stands on,
whose location the emission carries -- into a regenerated file and with it into
retransformed script bytecode.
"""
import importlib.util
import json
from pathlib import Path
from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[3]
_JSON_PATH = _REPO_ROOT / 'src' / 'pynecore' / 'transformers' / 'call_inline_templates.json'

_REGENERATE = ("call_inline_templates.json is stale — regenerate it with "
               "`python scripts/call_inline_collector.py` from the pynecore checkout root "
               "(venv active), and commit the result")


def _load_collector() -> ModuleType:
    collector_path = _REPO_ROOT / 'scripts' / 'call_inline_collector.py'
    spec = importlib.util.spec_from_file_location('call_inline_collector', collector_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def __test_call_inline_templates_are_current__():
    """The committed registry is what the collector derives today, and every
    difference is named: the function whose template changed, the callee path
    that moved, the loop iterable and the varargs arities"""
    generated = _load_collector().collect()
    committed = json.loads(_JSON_PATH.read_text(encoding='utf-8'))

    changes = []
    for section in ('functions', 'callees'):
        now, then = generated[section], committed.get(section, {})
        for key in sorted(now.keys() | then.keys()):
            if now.get(key) == then.get(key):
                continue
            if key not in then:
                changes.append(f"{section}[{key!r}]: added")
            elif key not in now:
                changes.append(f"{section}[{key!r}]: removed")
            elif section == 'functions' and now[key].get('text') != then[key].get('text'):
                changes.append(f"{section}[{key!r}]: {then[key].get('text')!r} -> "
                               f"{now[key].get('text')!r}")
            elif section == 'functions':
                changes.append(f"{section}[{key!r}]: same expression, other source "
                               f"locations or structure")
            else:
                changes.append(f"{section}[{key!r}]: {then[key]!r} -> {now[key]!r}")
    for section in ('loop_ranges', 'varargs_arities', 'v'):
        if generated[section] != committed.get(section):
            changes.append(f"{section}: {committed.get(section)!r} -> {generated[section]!r}")
    assert not changes, _REGENERATE + ". Changed:\n  " + "\n  ".join(changes)
    assert generated == committed, _REGENERATE


def __test_schema_version_matches_the_collector__():
    """A shape change must bump the version"""
    committed = json.loads(_JSON_PATH.read_text(encoding='utf-8'))
    assert committed['v'] == _load_collector().SCHEMA_VERSION


def __test_load_bearing_entries__():
    """The templates the corpus leans on hardest carry the expected expressions"""
    registry = json.loads(_JSON_PATH.read_text(encoding='utf-8'))
    functions = registry['functions']

    assert functions['pynecore.lib.math.abs']['text'] == (
        '__inl·na_float__ if not number == number else __inl·py_builtins__.abs(number)')
    assert functions['pynecore.lib.array.get']['text'] == (
        '__inl·array_na_element__(id) if not index == index else id[__inl·py_int__(index)]')
    assert functions['pynecore.lib.math.max'] == {
        'kind': 'varargs', 'builtin': 'max', 'na_anchor': 'math_na_of_operands',
        'builtins_anchor': 'py_builtins'}
    assert functions['pynecore.lib.math.min']['builtin'] == 'min'
    # A parameter read is a slot of the call site's arguments, not a name
    assert functions['pynecore.lib.math.abs']['body']['test']['operand']['left'] == {'param': 0}
    # ... and a node copied from the wrapper keeps the wrapper's source location
    assert len(functions['pynecore.lib.math.abs']['body']['test']['loc']) == 4

    assert registry['callees']['pynecore.lib.math.abs'] == 'pynecore.lib.math.abs'
    assert set(registry['callees'].values()) == set(functions)
    assert {'pynecore.core.pine_range.pine_range', 'pynecore.pine_range'} \
        <= set(registry['loop_ranges'])
    assert registry['varargs_arities'] == [2, 3]
