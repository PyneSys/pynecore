"""
Positional binding of the input.* functions against the Pine v6 reference
orders. Deliberately NOT a @pyne file: the InputTransformer must stay out of
the way so the raw runtime binding is what is exercised (explicit ``_id``).

v6 orders under test:
- input.int/float range form:   defval, title, minval, maxval, step, tooltip,
                                inline, group, confirm, display, active
- input.int/float options form: defval, title, options, tooltip, inline, group,
                                confirm, display, active
- input.source (unique):        defval, title, tooltip, inline, group, display,
                                active, confirm  -- confirm LAST
"""
import pytest

from pynecore.core.script import input as pyne_input, inputs


@pytest.fixture(autouse=True)
def _clean_inputs():
    saved = dict(inputs)
    inputs.clear()
    yield
    inputs.clear()
    inputs.update(saved)


def __test_int_range_form_positional__():
    """The 3rd..11th positionals bind to minval..active in the range form"""
    value = pyne_input.int(5, 'len', 1, 500, 2, 'tip', 'ln', 'grp', True, None, False, _id='t')
    data = inputs['t']
    assert value == 5
    assert (data.minval, data.maxval, data.step) == (1, 500, 2)
    assert (data.tooltip, data.inline, data.group) == ('tip', 'ln', 'grp')
    assert data.confirm is True
    assert data.options is None


def __test_int_options_form_positional__():
    """A tuple/list 3rd positional selects the options form: 4th binds to tooltip"""
    value = pyne_input.int(5, 'len', (1, 2, 3), 'tip', 'ln', 'grp', True, _id='t')
    data = inputs['t']
    assert value == 5
    assert data.options == (1, 2, 3)
    assert (data.tooltip, data.inline, data.group) == ('tip', 'ln', 'grp')
    assert data.confirm is True
    assert data.minval is None and data.maxval is None and data.step is None


def __test_float_both_forms__():
    """input.float shares the dual overload binding"""
    pyne_input.float(1.5, 'f', 0.1, 9.9, 0.1, _id='r')
    assert (inputs['r'].minval, inputs['r'].maxval, inputs['r'].step) == (0.1, 9.9, 0.1)
    pyne_input.float(1.5, 'f', (1.0, 1.5), _id='o')
    assert inputs['o'].options == (1.0, 1.5)


def __test_numeric_positional_keyword_clash_raises__():
    """A positional minval plus keyword minval is a TypeError, like plain Python"""
    with pytest.raises(TypeError, match="multiple values"):
        pyne_input.int(5, 't', 1, minval=2, _id='t')


def __test_source_confirm_is_last_positional__():
    """input.source order: tooltip, inline, group, display, active, confirm (confirm 8th)"""
    pyne_input.source('close', 'src', 'tip', 'ln', 'grp', None, None, True, _id='s')
    data = inputs['s']
    assert (data.tooltip, data.inline, data.group) == ('tip', 'ln', 'grp')
    assert data.confirm is True


def __test_active_accepted_everywhere_and_ignored__():
    """active is UI-only: accepted positionally and by keyword, never stored"""
    pyne_input.bool(True, 'b', 'tip', 'ln', 'grp', False, None, True, _id='b')
    assert inputs['b'].confirm is False
    pyne_input.int(1, 'i', active=False, _id='i')
    pyne_input('x', 'g', 'tip', 'ln', 'grp', None, True, _id='g')
    assert inputs['g'].tooltip == 'tip'
