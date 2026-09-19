"""
@pyne

Pine arrays, maps, matrices and UDT instances are references. A rollback of the
var slots (a calc_on_order_fills re-execution, a live intra-bar tick, a
``request.security`` developing re-tick) restores the OBJECTS the variables name,
so two variables naming one object still name one object afterwards, and a
mutation that reached a nested object is undone as well.
"""
from pynecore.core import instance_state
from pynecore.core.pine_udt import udt
from pynecore.lib import matrix as matrix_lib
from pynecore.types.matrix import Matrix


@udt
class Node:
    value: float = 0.0
    items: list | None = None
    link: 'Node | None' = None


class __test_helper_Rollback:
    """A root vector with three var slots and one child instance with two."""

    key = 'test_107_root'

    def __init__(self):
        child_layout = {'init': (None, None), 'series': (), 'varip': (), 'children': ()}
        root_layout = {'init': (None, None, None, None), 'series': (), 'varip': (),
                       'children': ((3, 'child', False),)}
        instance_state.discard_root(self.key)
        self.root = instance_state.create_root(self.key, root_layout)
        self.child = instance_state._make_state(child_layout)
        self.root[3] = self.child
        self._var = self._children = None

    def save(self):
        self._var = instance_state.RootVarSnapshot(keys=[self.key])
        self._children = instance_state.RootChildSnapshot(keys=[self.key])
        self._var.save()
        self._children.save()

    def restore(self):
        self._var.restore()
        self._children.restore()

    def close(self):
        instance_state.discard_root(self.key)


def __test_two_root_slots_keep_naming_one_array__():
    """ Two var slots naming one array name one array after a rollback """
    state = __test_helper_Rollback()
    try:
        shared = [1.0]
        state.root[0] = state.root[1] = shared
        state.save()
        state.root[0].append(9.0)
        state.restore()
        assert state.root[0] is state.root[1]
        assert state.root[0] == [1.0]
        state.root[0].append(2.0)
        assert state.root[1] == [1.0, 2.0]
    finally:
        state.close()


def __test_a_root_slot_and_a_function_instance_keep_naming_one_array__():
    """ The alias survives across the var and the child snapshot """
    state = __test_helper_Rollback()
    try:
        shared = [1.0]
        state.root[0] = state.child[0] = shared
        state.save()
        state.child[0].append(9.0)
        state.restore()
        assert state.root[0] is state.root[3][0]
        assert state.root[0] == [1.0]
    finally:
        state.close()


def __test_two_slots_keep_naming_one_matrix_and_one_map__():
    """ Matrices and maps are references exactly like arrays """
    state = __test_helper_Rollback()
    try:
        shared_matrix = Matrix(2, 2, 0.0)
        shared_map = {'a': 1.0}
        state.root[0] = state.root[1] = shared_matrix
        state.root[2] = state.child[0] = shared_map
        state.save()
        matrix_lib.set(state.root[0], 0, 0, 5.0)
        matrix_lib.add_row(state.root[0])
        state.root[2]['b'] = 2.0
        state.restore()
        assert state.root[0] is state.root[1]
        assert matrix_lib.get(state.root[0], 0, 0) == 0.0
        assert state.root[0].rows == 2
        assert state.root[2] is state.root[3][0]
        assert state.root[2] == {'a': 1.0}
    finally:
        state.close()


def __test_a_mutation_of_a_udt_field_array_is_rolled_back__():
    """ A discarded execution's push into an array held by a UDT field is undone """
    state = __test_helper_Rollback()
    try:
        state.root[0] = Node(1.0, [1.0])
        state.save()
        state.root[0].items.append(9.0)
        state.root[0].value = 5.0
        state.restore()
        assert state.root[0].items == [1.0]
        assert state.root[0].value == 1.0
    finally:
        state.close()


def __test_a_udt_in_an_array_and_in_a_slot_stays_one_object__():
    """ A UDT instance reachable through an array element and a variable stays shared """
    state = __test_helper_Rollback()
    try:
        node = Node(1.0)
        state.root[0] = node
        state.root[1] = [node]
        state.save()
        state.root[0].value = 7.0
        state.restore()
        assert state.root[0] is state.root[1][0]
        assert state.root[0].value == 1.0
    finally:
        state.close()


def __test_a_udt_cycle_is_rolled_back__():
    """ UDT instances linking to each other roll back without recursing forever """
    state = __test_helper_Rollback()
    try:
        first, second = Node(1.0), Node(2.0)
        first.link, second.link = second, first
        state.root[0] = first
        state.save()
        second.value = 9.0
        first.link = None
        state.restore()
        assert state.root[0].link is second
        assert second.link is first
        assert second.value == 2.0
    finally:
        state.close()


def __test_a_rebound_slot_gets_its_object_back__():
    """ A slot the discarded execution pointed at a new array names the old one again """
    state = __test_helper_Rollback()
    try:
        shared = [1.0]
        state.root[0] = state.root[1] = shared
        state.save()
        state.root[1] = [5.0]
        state.restore()
        assert state.root[1] is shared
        assert state.root[0] is state.root[1]
    finally:
        state.close()


def __test_a_second_rollback_restores_the_same_baseline__():
    """ The baseline is not consumed or polluted by a restore """
    state = __test_helper_Rollback()
    try:
        state.root[0] = [Node(1.0, [1.0])]
        state.save()
        for _ in range(2):
            state.root[0][0].items.append(9.0)
            state.root[0].append(Node(3.0))
            state.restore()
            assert len(state.root[0]) == 1
            assert state.root[0][0].items == [1.0]
    finally:
        state.close()
