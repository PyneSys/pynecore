"""
@pyne
"""
__pyne_slot_layout__ = {'t2': {'init': (None, None), 'series': ((0, None), (1, None)), 'varip': (), 'children': (), 'names': ('s', 's1')}, 'main·t': {'init': (None,), 'series': ((0, None),), 'varip': (), 'children': (), 'names': ('s',)}}

def t2(__state__, s: float, s1: float):
    s = __state__[0].add(s)
    s1 = __state__[1].add(s1)
    s = __state__[0].set(s + 1)
    print(s, __state__[0][1], s1, __state__[1][10])
    return __state__[0][2]
t2.__pyne_layout__ = __pyne_slot_layout__['t2']

def main():

    def t(__state__, s: float):
        s = __state__[0].add(s)
        s = __state__[0].set(s + 1)
        print(s, __state__[0][1])
        return s
    t.__pyne_layout__ = __pyne_slot_layout__['main·t']
    s: float = 1
