"""
The plain double-underscore names the transform emits into user code.

Most names the transformers inject carry the reserved middle dot (see
``PYNE_RESERVED_NAME_CHAR`` in ``core/import_hook.py``). The ones below do not:
runtime helpers, module-level protocol records and generated temporaries that
other tools (the runtime, the security children, the ahead-of-time compiler)
address by these exact spellings. User code that binds or reads one of them
would silently collide with the emission -- a truthiness temporary rebinding a
module variable of the same name, a user ``__state__`` shadowing the state
vector -- so the loader rejects Pyne code that spells one (see
``_reject_pipeline_names`` in ``core/import_hook.py``).

A leaf module (``re`` only): the loader imports it at module level.
"""
import re

__all__ = ['PIPELINE_NAME', 'is_pipeline_name']

#: Every plain double-underscore name the transform can emit. The ``__pyne_*__``
#: names are the protocol namespace (module records, function attributes); the
#: numbered ones are per-module temporaries
PIPELINE_NAME = re.compile(
    r'__(?:'
    r'pyne_\w+'
    # Generated temporaries: comparison and truthiness walrus targets, lazy-context
    # history hoists, per-context ``main`` slices
    r'|cmp\d+|bool\d+|hist_\d+|sec_main_\d+'
    # Slot-state plumbing
    r'|state|attach_layout|resolve_slot|bind_any|bind_loop|bind_slot|bind_pinned'
    r'|loop_state|slot_state|dyn_default'
    # The request.security protocol
    r'|security_contexts|active_security|same_context'
    r'|sec_read|sec_write|sec_wait|sec_signal|ltf_unzip'
    r')__')


def is_pipeline_name(name: str) -> bool:
    """Whether ``name`` is one of the names the transform emits.

    :param name: An identifier, as Python binds it (NFKC-normalized).
    :return: True for a name Pyne code may not spell.
    """
    return PIPELINE_NAME.fullmatch(name) is not None
