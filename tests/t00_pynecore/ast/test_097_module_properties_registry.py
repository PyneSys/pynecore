"""
Freshness guard for module_properties.json.

The ModulePropertyTransformer raises on names missing from the registry, so a
stale registry breaks valid scripts. The committed JSON must always match what
scripts/module_property_collector.py generates from the current lib source.
"""
import importlib.util
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def __test_module_properties_json_is_current__():
    collector_path = _REPO_ROOT / 'scripts' / 'module_property_collector.py'
    spec = importlib.util.spec_from_file_location('module_property_collector', collector_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    collector = module.ModulePropertyCollector(project_src=_REPO_ROOT / 'src')
    generated = collector.collect()

    json_path = _REPO_ROOT / 'src' / 'pynecore' / 'transformers' / 'module_properties.json'
    committed = json.loads(json_path.read_text())

    assert generated == committed, \
        "module_properties.json is stale — rerun scripts/module_property_collector.py"


def __test_registry_excludes_typing_machinery__():
    """
    Implementation-only names must NOT be accepted public lib names: a registry
    entry suppresses the transformer's unknown-name error, so typing machinery
    leaking in makes e.g. ``ta.cast`` silently valid in user scripts.
    """
    json_path = _REPO_ROOT / 'src' / 'pynecore' / 'transformers' / 'module_properties.json'
    committed = json.loads(json_path.read_text())

    banned = {
        'cast', 'overload', 'TypeVar', 'TypeAlias', 'TYPE_CHECKING', 'Literal',
        'Any', 'Callable', 'module_property', 'module_function_property',
        # TypeVar instances used across lib modules
        'T', 'TFI', 'TFIB', 'TKey', 'TValue', 'Number',
    }
    offenders = sorted(
        f'{module}.{name}'
        for module, names in committed.items()
        for name in names if name in banned
    )
    assert not offenders, f"typing machinery leaked into the registry: {offenders}"
