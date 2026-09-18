"""
@pyne

A ``Matrix`` in a var slot rolls back exactly like a Pine array (a list) or a
Pine map (a dict): ``instance_state._copy_value`` deep-copies it, so neither the
row storage nor a mutable element of a discarded re-execution reaches the
restored baseline.
"""
from pynecore.core import instance_state
from pynecore.lib import matrix as matrix_lib
from pynecore.types.matrix import Matrix


class Holder:
    """Mutable element standing in for a UDT instance stored in a cell."""

    def __init__(self, value: float):
        self.value = value


def __test_matrix_copy_value_is_deep__():
    """`_copy_value` hands back a matrix with its own rows AND own elements."""
    original = Matrix(2, 2, 0.0)
    copied = instance_state._copy_value(original)
    assert copied is not original
    assert copied.data is not original.data
    matrix_lib.set(copied, 0, 0, 5.0)
    assert matrix_lib.get(original, 0, 0) == 0.0

    holder = Holder(1.0)
    with_element: Matrix = Matrix(1, 1)
    matrix_lib.set(with_element, 0, 0, holder)
    element_copy = instance_state._copy_value(with_element)
    assert matrix_lib.get(element_copy, 0, 0) is not holder


def __test_matrix_retick_does_not_leak_into_baseline__():
    """A same-bar re-tick's matrix mutation is rolled back in root and child."""
    root_key = 'test_106_root'
    try:
        child_layout = {'init': (None,), 'series': (), 'varip': (), 'children': ()}
        root_layout = {'init': (None, None), 'series': (), 'varip': (),
                       'children': ((1, 'child', False),)}

        instance_state.discard_root(root_key)
        root = instance_state.create_root(root_key, root_layout)
        child = instance_state._make_state(child_layout)
        root[1] = child

        root_matrix = Matrix(2, 2, 0.0)
        child_matrix: Matrix = Matrix(1, 1)
        element = Holder(1.0)
        matrix_lib.set(child_matrix, 0, 0, element)
        root[0] = root_matrix
        child[0] = child_matrix

        var_snapshot = instance_state.RootVarSnapshot(keys=[root_key])
        child_snapshot = instance_state.RootChildSnapshot(keys=[root_key])
        var_snapshot.save()
        child_snapshot.save()

        matrix_lib.set(root_matrix, 0, 0, 111.0)
        matrix_lib.get(child_matrix, 0, 0).value = 222.0

        var_snapshot.restore()
        child_snapshot.restore()

        assert matrix_lib.get(root[0], 0, 0) == 0.0
        assert matrix_lib.get(root[1][0], 0, 0).value == 1.0
    finally:
        instance_state.discard_root(root_key)
