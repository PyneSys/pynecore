"""
What the transform's emission depends on besides the tree: the reserved separator
of the names it injects, and the env switch that changes what it emits.

The loader (``core/import_hook.py``) and the transformers read both. They live in
this leaf module (``os`` only) rather than in the loader, so a transformer depends
on these few definitions and not on the whole import hook: the native ports pin
the sources of their Python steps by digest (see ``transformers/native.py``), and
a loader edit must not take them out of use.
"""
import os

__all__ = ['PYNE_RESERVED_NAME_CHAR', 'SECURITY_SLICE_ENV', 'env_switch_on',
           'security_slice_disabled']

# Nearly everything the transformers inject into script scope is named with a
# Unicode middle dot: the scope-qualified state parameters and slot constants
# (``__state·main__``, ``__slot·main·x__``), the generated temporaries
# (``__st·__``, ``__cnt·0__``) and the aliased runtime helper imports
# (``__resolve_slot·__``). The separator is a legal identifier character in
# Python (``Other_ID_Continue``), so the namespace is only collision-free while
# scripts stay out of it — a script name spelled with it would shadow or clobber
# an injected one and break the emission in ways no transformer can detect. The
# few plain double-underscore names the transform emits are listed in
# ``transformers/pipeline_names.py`` and reserved the same way.
PYNE_RESERVED_NAME_CHAR = '·'

#: Env switch turning the per-context ``main()`` slicing off (see
#: ``transformers/security_slice.py``). It changes the EMITTED tree, so the loader
#: mixes it into the pipeline digest (``import_hook._get_transform_pipeline_hash``)
#: — a chart and its security children can never end up on bytecode built under
#: the other setting.
SECURITY_SLICE_ENV = 'PYNE_NO_SECURITY_SLICE'

_TRUTHY = frozenset({'1', 'true', 'yes', 'on'})


def env_switch_on(name: str) -> bool:
    """Whether an env switch is set to a truthy word (``1``, ``true``, ``yes``, ``on``).

    :param name: Name of the environment variable.
    :return: True when the variable holds a truthy word, case and surrounding
        whitespace ignored.
    """
    return os.environ.get(name, '').strip().lower() in _TRUTHY


def security_slice_disabled() -> bool:
    """Whether ``PYNE_NO_SECURITY_SLICE`` asks for the unsliced child behaviour.

    :return: True when no backward slice of ``main()`` may be emitted.
    """
    return env_switch_on(SECURITY_SLICE_ENV)
