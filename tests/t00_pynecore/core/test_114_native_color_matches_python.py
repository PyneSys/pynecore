"""Color creation preserves packed values, NA propagation and object independence.

The native wheel gate requires the compiled implementation. Source installs without
the extension still exercise the optimized Python path and its fallback selection.
"""
import math
import os
import random
import subprocess
import sys

import pytest

from pynecore.lib import color
from pynecore.types.color import Color
from pynecore.types.na import NA


def __test_helper_implementation(backend):
    if backend == 'python':
        return color.PYTHON_IMPLEMENTATIONS['new']
    try:
        from pynecore.core._native_color import new
    except ImportError:
        if os.environ.get('PYNE_REQUIRE_NATIVE_COLOR'):
            pytest.fail('the native color extension is not installed')
        pytest.skip('native color extension not built')
    return new


def __test_native_color_matches_python__():
    """Native alpha rounding matches Python across sweeps and rounding boundaries."""
    native = __test_helper_implementation('native')
    python = __test_helper_implementation('python')
    if os.environ.get('PYNE_REQUIRE_NATIVE_COLOR'):
        assert color.new is native

    rnd = random.Random(20261007)
    transparencies = [i / 10 for i in range(-100, 1101)]
    transparencies += [rnd.uniform(-20, 120) for _ in range(10_000)]
    transparencies += [0, 30, 90, 100, 10 ** 1000, -(10 ** 1000),
                      math.inf, -math.inf, math.nan, NA(float), NA(Color)]
    for alpha in range(255):
        boundary = (1 - (alpha + 0.5) / 255) * 100
        transparencies += [math.nextafter(boundary, -math.inf), boundary,
                           math.nextafter(boundary, math.inf)]
    for base in (color.red, Color('#01234567'), '#abcdef', '#ABCDEF00',
                 '##abcdef12', NA(Color)):
        for transp in transparencies:
            got, want = native(base, transp), python(base, transp)
            assert type(got) is type(want), (base, transp)
            if isinstance(want, Color):
                assert got.value == want.value, (base, transp, got, want)


@pytest.mark.parametrize('backend', ('python', 'native'))
def __test_new_colors_are_independent__(backend):
    """Changing a returned color leaves both its input and sibling result untouched."""
    new = __test_helper_implementation(backend)
    base = Color('#01234567')
    first = new(color=base, transp=30)
    second = new(color=base, transp=30)
    assert type(first) is Color
    assert first is not base and first is not second
    assert first.value == second.value == 0x012345B3
    first.a = 0
    second.t = 90
    assert base.value == 0x01234567
    assert first.value == 0x01234500
    assert second.value == 0x01234519


@pytest.mark.parametrize('backend', ('python', 'native'))
def __test_new_color_object_avoids_hex_constructor__(backend, monkeypatch):
    """A Color input needs no hexadecimal parsing to produce another Color."""
    new = __test_helper_implementation(backend)
    base = Color('#01234567')

    def forbid_hex(self, hexstr):
        raise AssertionError('the object path must not parse a hexadecimal string')

    monkeypatch.setattr(Color, '__init__', forbid_hex)
    assert new(base, 30).value == 0x012345B3


@pytest.mark.parametrize('backend', ('python', 'native'))
def __test_new_color_string_and_na_inputs__(backend):
    """Strings retain their RGB; an NA argument propagates before string parsing."""
    new = __test_helper_implementation(backend)
    assert new('#012345', 30).value == 0x012345B3
    assert new('#01234500', 30).value == 0x012345B3
    assert new(color.red).value == color.red.value
    assert new(NA(Color), 30) is NA(Color)
    assert new('invalid', math.nan) is NA(Color)
    with pytest.raises(ValueError):
        new('invalid', 30)


@pytest.mark.parametrize('mode', ('disabled', 'unavailable'))
def __test_color_falls_back_to_python__(mode):
    """A disabled or unavailable native extension keeps color.new operational."""
    env = os.environ.copy()
    env.pop('PYNE_NO_NATIVE_COLOR', None)
    if mode == 'disabled':
        env['PYNE_NO_NATIVE_COLOR'] = '1'
    setup = "import sys; sys.modules['pynecore.core._native_color'] = None\n" \
        if mode == 'unavailable' else ''
    subprocess.run([sys.executable, '-B', '-c', setup + (
        "from pynecore.lib import color\n"
        "assert color.new is color.PYTHON_IMPLEMENTATIONS['new']\n"
        "assert color.new(color.red, 30).value == 0xF23645B3\n"
    )], env=env, check=True, capture_output=True, text=True)
