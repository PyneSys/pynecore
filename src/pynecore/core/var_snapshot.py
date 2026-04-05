"""
Snapshot/restore mechanism for var globals during calc_on_order_fills re-execution.

Only var (not varip) globals are rolled back before each script re-execution after a fill.
Handles multiple modules (main script + libraries) since each has its own __dict__.
"""
from copy import copy, deepcopy
from dataclasses import replace as dataclass_replace


class VarSnapshot:
    """Snapshots and restores var globals for calc_on_order_fills rollback."""

    __slots__ = ('_targets', '_snapshots')

    def __init__(self, script_module, registered_libraries):
        self._targets: list[tuple[dict, list[str]]] = []
        self._snapshots: list[dict[str, object]] = []

        self._collect_module(script_module.__dict__)
        for _title, main_func in registered_libraries:
            lib_globals = main_func.__globals__
            if '__persistent_function_vars__' in lib_globals:
                self._collect_module(lib_globals)

    def _collect_module(self, globals_dict: dict):
        all_vars = globals_dict.get('__persistent_function_vars__', {})
        varip_vars = globals_dict.get('__varip_function_vars__', {})

        varip_set: set[str] = set()
        for scope_vars in varip_vars.values():
            varip_set.update(scope_vars)

        var_names = [
            name for scope_vars in all_vars.values()
            for name in scope_vars if name not in varip_set
        ]
        if var_names:
            self._targets.append((globals_dict, var_names))

    @property
    def has_vars(self) -> bool:
        return len(self._targets) > 0

    def save(self):
        """Snapshot all var globals across all modules (called at bar start)."""
        self._snapshots = [
            {name: _copy_value(g[name]) for name in names if name in g}
            for g, names in self._targets
        ]

    def restore(self):
        """Restore all var globals to saved snapshot (rollback)."""
        for (g, _names), snapshot in zip(self._targets, self._snapshots):
            for name, value in snapshot.items():
                g[name] = _copy_value(value)


def _copy_value(value):
    """Copy a value for snapshot/restore — same pattern as function_isolation.py"""
    if isinstance(value, (int, float, bool, str, type(None))):
        return value
    if isinstance(value, (dict, list)):
        return deepcopy(value)
    try:
        return dataclass_replace(value)
    except TypeError:
        return copy(value)
