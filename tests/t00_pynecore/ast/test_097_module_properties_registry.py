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
