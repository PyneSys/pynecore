"""
The name of the state vector parameter the slot lowering emits.

A leaf module: a pass that only has to recognize the parameter (a slot read
``__state__[7]`` among the arguments it binds) does not depend on the slot
layout model in ``slot_layout.py``, which re-exports the name.
"""
__all__ = ['DEFAULT_STATE_PARAM']

#: The hidden parameter of a function with slot state. A function that contains
#: nested definitions gets the scope-qualified ``__state·<scope>__`` instead
DEFAULT_STATE_PARAM = '__state__'
