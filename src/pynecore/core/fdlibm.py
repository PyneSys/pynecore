"""Bit-exact Python port of fdlibm 5.3 sin/cos/tan/exp/log/pow/asin/acos/atan.

Ported from the SunSoft fdlibm sources (netlib.org/fdlibm), which carry the
following notice that must be preserved:

    ====================================================
    Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.

    Developed at SunSoft, a Sun Microsystems, Inc. business.
    Permission to use, copy, modify, and distribute this
    software is freely granted, provided that this notice
    is preserved.
    ====================================================

Python floats are IEEE binary64 and Python arithmetic maps 1:1 onto the C
double operations, so the port is bit-exact by construction; it is verified
against Java StrictMath (the same fdlibm lineage) over millions of arguments.

``tan``, ``log`` and ``pow`` serve the parse-time constant fold only (see
``transformers.const_fold``): at runtime the venue computes them with its Intel-LIBM
intrinsics, so they have no compiled twin in ``_native_math``.
"""
import math as _math
import os as _os
from struct import pack as _pack, unpack as _unpack

__all__ = ['sin', 'cos', 'exp', 'asin', 'acos', 'atan', 'log', 'tan', 'pow']

_M32 = 0xFFFFFFFF


def _f2b(f: float) -> int:
    return _unpack('<Q', _pack('<d', f))[0]


def _b2f(b: int) -> float:
    return _unpack('<d', _pack('<Q', b))[0]


def _hi(x: float) -> int:
    """Signed high word of x (C's __HI)."""
    h = (_f2b(x) >> 32) & _M32
    return h - 0x100000000 if h >= 0x80000000 else h


def _lo(x: float) -> int:
    """Unsigned low word of x (C's __LO)."""
    return _f2b(x) & _M32


def _set_hi(x: float, hi: int) -> float:
    return _b2f(((hi & _M32) << 32) | (_f2b(x) & _M32))


def _set_lo(x: float, lo: int) -> float:
    return _b2f((_f2b(x) & 0xFFFFFFFF00000000) | (lo & _M32))


# --- __kernel_sin / __kernel_cos (k_sin.c, k_cos.c) --------------------------

_S1 = -1.66666666666666324348e-01
_S2 = 8.33333333332248946124e-03
_S3 = -1.98412698298579493134e-04
_S4 = 2.75573137070700676789e-06
_S5 = -2.50507602534068634195e-08
_S6 = 1.58969099521155010221e-10

_C1 = 4.16666666666666019037e-02
_C2 = -1.38888888888741095749e-03
_C3 = 2.48015872894767294178e-05
_C4 = -2.75573143513906633035e-07
_C5 = 2.08757232129817482790e-09
_C6 = -1.13596475577881948265e-11


def _kernel_sin(x: float, y: float, iy: int) -> float:
    ix = _hi(x) & 0x7fffffff
    if ix < 0x3e400000:                 # |x| < 2**-27
        return x
    z = x * x
    v = z * x
    r = _S2 + z * (_S3 + z * (_S4 + z * (_S5 + z * _S6)))
    if iy == 0:
        return x + v * (_S1 + z * r)
    return x - ((z * (0.5 * y - v * r) - y) - v * _S1)


def _kernel_cos(x: float, y: float) -> float:
    ix = _hi(x) & 0x7fffffff
    if ix < 0x3e400000:                 # |x| < 2**-27
        return 1.0
    z = x * x
    r = z * (_C1 + z * (_C2 + z * (_C3 + z * (_C4 + z * (_C5 + z * _C6)))))
    if ix < 0x3FD33333:                 # |x| < 0.3
        return 1.0 - (0.5 * z - (z * r - x * y))
    if ix > 0x3fe90000:                 # |x| > 0.78125
        qx = 0.28125
    else:
        qx = _b2f(((ix - 0x00200000) & _M32) << 32)     # x/4, low word zero
    hz = 0.5 * z - qx
    a = 1.0 - qx
    return a - (hz - (z * r - x * y))


# --- __ieee754_rem_pio2 (e_rem_pio2.c) + __kernel_rem_pio2 (k_rem_pio2.c) ----

_TWO_OVER_PI = (
    0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62,
    0x95993C, 0x439041, 0xFE5163, 0xABDEBB, 0xC561B7, 0x246E3A,
    0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129,
    0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C, 0x7026B4, 0x5F7E41,
    0x3991D6, 0x398353, 0x39F49C, 0x845F8B, 0xBDF928, 0x3B1FF8,
    0x97FFDE, 0x05980F, 0xEF2F11, 0x8B5A0A, 0x6D1F6D, 0x367ECF,
    0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5,
    0xF17B3D, 0x0739F7, 0x8A5292, 0xEA6BFB, 0x5FB11F, 0x8D5D08,
    0x560330, 0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3,
    0x91615E, 0xE61B08, 0x659985, 0x5F14A0, 0x68408D, 0xFFD880,
    0x4D7327, 0x310606, 0x1556CA, 0x73A8C9, 0x60E27B, 0xC08C6B,
)

_NPIO2_HW = (
    0x3FF921FB, 0x400921FB, 0x4012D97C, 0x401921FB, 0x401F6A7A, 0x4022D97C,
    0x4025FDBB, 0x402921FB, 0x402C463A, 0x402F6A7A, 0x4031475C, 0x4032D97C,
    0x40346B9C, 0x4035FDBB, 0x40378FDB, 0x403921FB, 0x403AB41B, 0x403C463A,
    0x403DD85A, 0x403F6A7A, 0x40407E4C, 0x4041475C, 0x4042106C, 0x4042D97C,
    0x4043A28C, 0x40446B9C, 0x404534AC, 0x4045FDBB, 0x4046C6CB, 0x40478FDB,
    0x404858EB, 0x404921FB,
)

_PIO2 = (
    1.57079625129699707031e+00,
    7.54978941586159635335e-08,
    5.39030252995776476554e-15,
    3.28200341580791294123e-22,
    1.27065575308067607349e-29,
    1.22933308981111328932e-36,
    2.73370053816464559624e-44,
    2.16741683877804819444e-51,
)

_TWO24 = 1.67772160000000000000e+07
_TWON24 = 5.96046447753906250000e-08

_INVPIO2 = 6.36619772367581382433e-01
_PIO2_1 = 1.57079632673412561417e+00
_PIO2_1T = 6.07710050650619224932e-11
_PIO2_2 = 6.07710050630396597660e-11
_PIO2_2T = 2.02226624879595063154e-21
_PIO2_3 = 2.02226624871116645580e-21
_PIO2_3T = 8.47842766036889956997e-32

_INIT_JK = (2, 3, 4, 6)


def _kernel_rem_pio2(x: list[float], e0: int, nx: int, prec: int) -> tuple[int, float, float]:
    """Returns (n & 7, y0, y1) for prec in (1, 2)."""
    iq = [0] * 20
    f = [0.0] * 20
    fq = [0.0] * 20
    q = [0.0] * 20

    jk: int = _INIT_JK[prec]
    jp = jk
    jx = nx - 1
    jv = (e0 - 3) // 24 if e0 >= 3 else 0   # C int division truncates; e0>=?
    if jv < 0:
        jv = 0
    q0 = e0 - 24 * (jv + 1)

    j = jv - jx
    m = jx + jk
    for i in range(m + 1):
        f[i] = 0.0 if j < 0 else float(_TWO_OVER_PI[j])
        j += 1

    for i in range(jk + 1):
        fw = 0.0
        for j in range(jx + 1):
            fw += x[j] * f[jx + i - j]
        q[i] = fw

    jz = jk
    while True:                                     # recompute:
        i = 0
        j = jz
        z = q[jz]
        while j > 0:
            fw = float(int(_TWON24 * z))
            iq[i] = int(z - _TWO24 * fw)
            z = q[j - 1] + fw
            i += 1
            j -= 1

        z = _math.ldexp(z, q0)
        z -= 8.0 * _math.floor(z * 0.125)
        n = int(z)
        z -= float(n)
        ih = 0
        if q0 > 0:
            i = iq[jz - 1] >> (24 - q0)
            n += i
            iq[jz - 1] -= i << (24 - q0)
            ih = iq[jz - 1] >> (23 - q0)
        elif q0 == 0:
            ih = iq[jz - 1] >> 23
        elif z >= 0.5:
            ih = 2

        if ih > 0:
            n += 1
            carry = 0
            for i in range(jz):
                j = iq[i]
                if carry == 0:
                    if j != 0:
                        carry = 1
                        iq[i] = 0x1000000 - j
                else:
                    iq[i] = 0xffffff - j
            if q0 == 1:
                iq[jz - 1] &= 0x7fffff
            elif q0 == 2:
                iq[jz - 1] &= 0x3fffff
            if ih == 2:
                z = 1.0 - z
                if carry != 0:
                    z -= _math.ldexp(1.0, q0)

        if z == 0.0:
            j = 0
            for i in range(jz - 1, jk - 1, -1):
                j |= iq[i]
            if j == 0:                              # need recomputation
                k = 1
                while iq[jk - k] == 0:
                    k += 1
                for i in range(jz + 1, jz + k + 1):
                    f[jx + i] = float(_TWO_OVER_PI[jv + i])
                    fw = 0.0
                    for j in range(jx + 1):
                        fw += x[j] * f[jx + i - j]
                    q[i] = fw
                jz += k
                continue                            # goto recompute
        break

    if z == 0.0:
        jz -= 1
        q0 -= 24
        while iq[jz] == 0:
            jz -= 1
            q0 -= 24
    else:
        z = _math.ldexp(z, -q0)
        if z >= _TWO24:
            fw = float(int(_TWON24 * z))
            iq[jz] = int(z - _TWO24 * fw)
            jz += 1
            q0 += 24
            iq[jz] = int(fw)
        else:
            iq[jz] = int(z)

    fw = _math.ldexp(1.0, q0)
    for i in range(jz, -1, -1):
        q[i] = fw * float(iq[i])
        fw *= _TWON24

    for i in range(jz, -1, -1):
        fw = 0.0
        k = 0
        while k <= jp and k <= jz - i:
            fw += _PIO2[k] * q[i + k]
            k += 1
        fq[jz - i] = fw

    fw = 0.0
    for i in range(jz, -1, -1):
        fw += fq[i]
    y0 = fw if ih == 0 else -fw
    fw = fq[0] - fw
    for i in range(1, jz + 1):
        fw += fq[i]
    y1 = fw if ih == 0 else -fw
    return n & 7, y0, y1


def _rem_pio2(x: float) -> tuple[int, float, float]:
    """__ieee754_rem_pio2: returns (n, y0, y1)."""
    hx = _hi(x)
    ix = hx & 0x7fffffff
    if ix <= 0x3fe921fb:                    # |x| ~<= pi/4
        return 0, x, 0.0
    if ix < 0x4002d97c:                     # |x| < 3pi/4
        if hx > 0:
            z = x - _PIO2_1
            if ix != 0x3ff921fb:
                y0 = z - _PIO2_1T
                y1 = (z - y0) - _PIO2_1T
            else:
                z -= _PIO2_2
                y0 = z - _PIO2_2T
                y1 = (z - y0) - _PIO2_2T
            return 1, y0, y1
        z = x + _PIO2_1
        if ix != 0x3ff921fb:
            y0 = z + _PIO2_1T
            y1 = (z - y0) + _PIO2_1T
        else:
            z += _PIO2_2
            y0 = z + _PIO2_2T
            y1 = (z - y0) + _PIO2_2T
        return -1, y0, y1
    if ix <= 0x413921fb:                    # |x| ~<= 2^19*(pi/2)
        t = abs(x)
        n = int(t * _INVPIO2 + 0.5)
        fn = float(n)
        r = t - fn * _PIO2_1
        w = fn * _PIO2_1T
        if n < 32 and ix != _NPIO2_HW[n - 1]:
            y0 = r - w
        else:
            j = ix >> 20
            y0 = r - w
            i = j - ((_hi(y0) >> 20) & 0x7ff)
            if i > 16:                      # 2nd iteration, good to 118 bits
                t2 = r
                w = fn * _PIO2_2
                r = t2 - w
                w = fn * _PIO2_2T - ((t2 - r) - w)
                y0 = r - w
                i = j - ((_hi(y0) >> 20) & 0x7ff)
                if i > 49:                  # 3rd iteration, 151 bits
                    t3 = r
                    w = fn * _PIO2_3
                    r = t3 - w
                    w = fn * _PIO2_3T - ((t3 - r) - w)
                    y0 = r - w
        y1 = (r - y0) - w
        if hx < 0:
            return -n, -y0, -y1
        return n, y0, y1
    if ix >= 0x7ff00000:                    # inf or NaN
        return 0, x - x, x - x
    # all other (large) arguments
    lo = _lo(x)
    e0 = (ix >> 20) - 1046
    z = _b2f((((ix - ((e0 << 20) & _M32)) & _M32) << 32) | lo)
    tx = [0.0, 0.0, 0.0]
    for i in range(2):
        tx[i] = float(int(z))
        z = (z - tx[i]) * _TWO24
    tx[2] = z
    nx = 3
    while tx[nx - 1] == 0.0:
        nx -= 1
    n, y0, y1 = _kernel_rem_pio2(tx, e0, nx, 2)
    if hx < 0:
        return -n, -y0, -y1
    return n, y0, y1


# --- sin, cos (s_sin.c, s_cos.c) ---------------------------------------------

def sin(x: float) -> float:
    ix = _hi(x) & 0x7fffffff
    if ix <= 0x3fe921fb:
        return _kernel_sin(x, 0.0, 0)
    if ix >= 0x7ff00000:
        return x - x
    n, y0, y1 = _rem_pio2(x)
    n &= 3
    if n == 0:
        return _kernel_sin(y0, y1, 1)
    if n == 1:
        return _kernel_cos(y0, y1)
    if n == 2:
        return -_kernel_sin(y0, y1, 1)
    return -_kernel_cos(y0, y1)


def cos(x: float) -> float:
    ix = _hi(x) & 0x7fffffff
    if ix <= 0x3fe921fb:
        return _kernel_cos(x, 0.0)
    if ix >= 0x7ff00000:
        return x - x
    n, y0, y1 = _rem_pio2(x)
    n &= 3
    if n == 0:
        return _kernel_cos(y0, y1)
    if n == 1:
        return -_kernel_sin(y0, y1, 1)
    if n == 2:
        return -_kernel_cos(y0, y1)
    return _kernel_sin(y0, y1, 1)


# --- exp (e_exp.c) -----------------------------------------------------------

_HALF = (0.5, -0.5)
_HUGE = 1.0e+300
_TWOM1000 = 9.33263618503218878990e-302
_O_THRESHOLD = 7.09782712893383973096e+02
_U_THRESHOLD = -7.45133219101941108420e+02
_LN2HI = (6.93147180369123816490e-01, -6.93147180369123816490e-01)
_LN2LO = (1.90821492927058770002e-10, -1.90821492927058770002e-10)
_INVLN2 = 1.44269504088896338700e+00
_EP1 = 1.66666666666666019037e-01
_EP2 = -2.77777777770155933842e-03
_EP3 = 6.61375632143793436117e-05
_EP4 = -1.65339022054652515390e-06
_EP5 = 4.13813679705723846039e-08


def exp(x: float) -> float:
    hx = _hi(x)
    xsb = (hx >> 31) & 1
    hx &= 0x7fffffff

    hi = lo = 0.0
    if hx >= 0x40862E42:                    # |x| >= 709.78...
        if hx >= 0x7ff00000:
            if ((hx & 0xfffff) | _lo(x)) != 0:
                return x + x                # NaN
            return x if xsb == 0 else 0.0   # exp(+-inf)
        if x > _O_THRESHOLD:
            return _HUGE * _HUGE            # overflow
        if x < _U_THRESHOLD:
            return _TWOM1000 * _TWOM1000    # underflow

    if hx > 0x3fd62e42:                     # |x| > 0.5 ln2
        if hx < 0x3FF0A2B2:                 # |x| < 1.5 ln2
            hi = x - _LN2HI[xsb]
            lo = _LN2LO[xsb]
            k = 1 - xsb - xsb
        else:
            k = int(_INVLN2 * x + _HALF[xsb])
            t = float(k)
            hi = x - t * _LN2HI[0]
            lo = t * _LN2LO[0]
        x = hi - lo
    elif hx < 0x3e300000:                   # |x| < 2**-28
        return 1.0 + x
    else:
        k = 0

    t = x * x
    c = x - t * (_EP1 + t * (_EP2 + t * (_EP3 + t * (_EP4 + t * _EP5))))
    if k == 0:
        return 1.0 - ((x * c) / (c - 2.0) - x)
    y = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi)
    if k >= -1021:
        return _set_hi(y, _hi(y) + (k << 20))
    y = _set_hi(y, _hi(y) + ((k + 1000) << 20))
    return y * _TWOM1000


# --- asin, acos (e_asin.c, e_acos.c) -----------------------------------------

_PIO2_HI = 1.57079632679489655800e+00
_PIO2_LO = 6.12323399573676603587e-17
_PIO4_HI = 7.85398163397448278999e-01
_PI_ACOS = 3.14159265358979311600e+00
_PS0 = 1.66666666666666657415e-01
_PS1 = -3.25565818622400915405e-01
_PS2 = 2.01212532134862925881e-01
_PS3 = -4.00555345006794114027e-02
_PS4 = 7.91534994289814532176e-04
_PS5 = 3.47933107596021167570e-05
_QS1 = -2.40339491173441421878e+00
_QS2 = 2.02094576023350569471e+00
_QS3 = -6.88283971605453293030e-01
_QS4 = 7.70381505559019352791e-02


def asin(x: float) -> float:
    hx = _hi(x)
    ix = hx & 0x7fffffff
    if ix >= 0x3ff00000:                    # |x| >= 1
        if ((ix - 0x3ff00000) | _lo(x)) == 0:
            return x * _PIO2_HI + x * _PIO2_LO
        return _math.nan                    # (x-x)/(x-x)
    if ix < 0x3fe00000:                     # |x| < 0.5
        if ix < 0x3e400000:                 # |x| < 2**-27
            return x
        t = x * x
        p = t * (_PS0 + t * (_PS1 + t * (_PS2 + t * (_PS3 + t * (_PS4 + t * _PS5)))))
        q = 1.0 + t * (_QS1 + t * (_QS2 + t * (_QS3 + t * _QS4)))
        w = p / q
        return x + x * w
    # 1 > |x| >= 0.5
    w = 1.0 - abs(x)
    t = w * 0.5
    p = t * (_PS0 + t * (_PS1 + t * (_PS2 + t * (_PS3 + t * (_PS4 + t * _PS5)))))
    q = 1.0 + t * (_QS1 + t * (_QS2 + t * (_QS3 + t * _QS4)))
    s = _math.sqrt(t)
    if ix >= 0x3FEF3333:                    # |x| > 0.975
        w = p / q
        t = _PIO2_HI - (2.0 * (s + s * w) - _PIO2_LO)
    else:
        w = _set_lo(s, 0)
        c = (t - w * w) / (s + w)
        r = p / q
        p = 2.0 * s * r - (_PIO2_LO - 2.0 * c)
        q = _PIO4_HI - 2.0 * w
        t = _PIO4_HI - (p - q)
    return t if hx > 0 else -t


def acos(x: float) -> float:
    hx = _hi(x)
    ix = hx & 0x7fffffff
    if ix >= 0x3ff00000:                    # |x| >= 1
        if ((ix - 0x3ff00000) | _lo(x)) == 0:
            if hx > 0:
                return 0.0
            return _PI_ACOS + 2.0 * _PIO2_LO
        return _math.nan                    # (x-x)/(x-x)
    if ix < 0x3fe00000:                     # |x| < 0.5
        if ix <= 0x3c600000:                # |x| < 2**-57
            return _PIO2_HI + _PIO2_LO
        z = x * x
        p = z * (_PS0 + z * (_PS1 + z * (_PS2 + z * (_PS3 + z * (_PS4 + z * _PS5)))))
        q = 1.0 + z * (_QS1 + z * (_QS2 + z * (_QS3 + z * _QS4)))
        r = p / q
        return _PIO2_HI - (x - (_PIO2_LO - x * r))
    if hx < 0:                              # x < -0.5
        z = (1.0 + x) * 0.5
        p = z * (_PS0 + z * (_PS1 + z * (_PS2 + z * (_PS3 + z * (_PS4 + z * _PS5)))))
        q = 1.0 + z * (_QS1 + z * (_QS2 + z * (_QS3 + z * _QS4)))
        s = _math.sqrt(z)
        r = p / q
        w = r * s - _PIO2_LO
        return _PI_ACOS - 2.0 * (s + w)
    # x > 0.5
    z = (1.0 - x) * 0.5
    s = _math.sqrt(z)
    df = _set_lo(s, 0)
    c = (z - df * df) / (s + df)
    p = z * (_PS0 + z * (_PS1 + z * (_PS2 + z * (_PS3 + z * (_PS4 + z * _PS5)))))
    q = 1.0 + z * (_QS1 + z * (_QS2 + z * (_QS3 + z * _QS4)))
    r = p / q
    w = r * s + c
    return 2.0 * (df + w)


# --- atan (s_atan.c) ---------------------------------------------------------

_ATAN_HI = (
    4.63647609000806093515e-01,             # atan(0.5) head
    7.85398163397448278999e-01,             # atan(1.0) head
    9.82793723247329054082e-01,             # atan(1.5) head
    1.57079632679489655800e+00,             # atan(inf) head
)
_ATAN_LO = (
    2.26987774529616870924e-17,
    3.06161699786838301793e-17,
    1.39033110312309984516e-17,
    6.12323399573676603587e-17,
)
_AT0 = 3.33333333333329318027e-01
_AT1 = -1.99999999998764832476e-01
_AT2 = 1.42857142725034663711e-01
_AT3 = -1.11111104054623557880e-01
_AT4 = 9.09088713343650656196e-02
_AT5 = -7.69187620504482999495e-02
_AT6 = 6.66107313738753120669e-02
_AT7 = -5.83357013379057348645e-02
_AT8 = 4.97687799461593236017e-02
_AT9 = -3.65315727442169155270e-02
_AT10 = 1.62858201153657823623e-02


def atan(x: float) -> float:
    hx = _hi(x)
    ix = hx & 0x7fffffff
    if ix >= 0x44100000:                    # |x| >= 2**66
        if ix > 0x7ff00000 or (ix == 0x7ff00000 and _lo(x) != 0):
            return x + x                    # NaN
        if hx > 0:
            return _ATAN_HI[3] + _ATAN_LO[3]
        return -_ATAN_HI[3] - _ATAN_LO[3]
    if ix < 0x3fdc0000:                     # |x| < 0.4375
        if ix < 0x3e200000:                 # |x| < 2**-29
            return x
        id_ = -1
    else:
        x = _math.fabs(x)
        if ix < 0x3ff30000:                 # |x| < 1.1875
            if ix < 0x3fe60000:             # 7/16 <= |x| < 11/16
                id_ = 0
                x = (2.0 * x - 1.0) / (2.0 + x)
            else:                           # 11/16 <= |x| < 19/16
                id_ = 1
                x = (x - 1.0) / (x + 1.0)
        elif ix < 0x40038000:               # |x| < 2.4375
            id_ = 2
            x = (x - 1.5) / (1.0 + 1.5 * x)
        else:                               # 2.4375 <= |x| < 2**66
            id_ = 3
            x = -1.0 / x

    # the sum of aT[i] * z**(i+1) broken into its odd and even halves
    z = x * x
    w = z * z
    s1 = z * (_AT0 + w * (_AT2 + w * (_AT4 + w * (_AT6 + w * (_AT8 + w * _AT10)))))
    s2 = w * (_AT1 + w * (_AT3 + w * (_AT5 + w * (_AT7 + w * _AT9))))
    if id_ < 0:
        return x - x * (s1 + s2)
    z = _ATAN_HI[id_] - ((x * (s1 + s2) - _ATAN_LO[id_]) - x)
    return -z if hx < 0 else z


# --- log (e_log.c) -----------------------------------------------------------

_LN2_HI = 6.93147180369123816490e-01
_LN2_LO = 1.90821492927058770002e-10
_TWO54 = 1.80143985094819840000e+16
_LG1 = 6.666666666666735130e-01
_LG2 = 3.999999999940941908e-01
_LG3 = 2.857142874366239149e-01
_LG4 = 2.222219843214978396e-01
_LG5 = 1.818357216161805012e-01
_LG6 = 1.531383769920937332e-01
_LG7 = 1.479819860511658591e-01


def log(x: float) -> float:
    hx = _hi(x)
    lx = _lo(x)
    k = 0
    if hx < 0x00100000:                     # x < 2**-1022
        if ((hx & 0x7fffffff) | lx) == 0:
            return -_math.inf               # log(+-0) = -inf
        if hx < 0:
            return _math.nan                # log(-#) = NaN
        k -= 54
        x *= _TWO54                         # subnormal: scale up x
        hx = _hi(x)
    if hx >= 0x7ff00000:
        return x + x
    k += (hx >> 20) - 1023
    hx &= 0x000fffff
    i = (hx + 0x95f64) & 0x100000
    x = _set_hi(x, hx | (i ^ 0x3ff00000))  # normalize x or x/2
    k += i >> 20
    f = x - 1.0
    if (0x000fffff & (2 + hx)) < 3:         # |f| < 2**-20
        if f == 0.0:
            if k == 0:
                return 0.0
            dk = float(k)
            return dk * _LN2_HI + dk * _LN2_LO
        r = f * f * (0.5 - 0.33333333333333333 * f)
        if k == 0:
            return f - r
        dk = float(k)
        return dk * _LN2_HI - ((r - dk * _LN2_LO) - f)
    s = f / (2.0 + f)
    dk = float(k)
    z = s * s
    i = hx - 0x6147a
    w = z * z
    j = 0x6b851 - hx
    t1 = w * (_LG2 + w * (_LG4 + w * _LG6))
    t2 = z * (_LG1 + w * (_LG3 + w * (_LG5 + w * _LG7)))
    i |= j
    r = t2 + t1
    if i > 0:
        hfsq = 0.5 * f * f
        if k == 0:
            return f - (hfsq - s * (hfsq + r))
        return dk * _LN2_HI - ((hfsq - (s * (hfsq + r) + dk * _LN2_LO)) - f)
    if k == 0:
        return f - s * (f - r)
    return dk * _LN2_HI - ((s * (f - r) - dk * _LN2_LO) - f)


# --- tan (k_tan.c, s_tan.c) --------------------------------------------------

_T = (
    3.33333333333334091986e-01,
    1.33333333333201242699e-01,
    5.39682539762260521377e-02,
    2.18694882948595424599e-02,
    8.86323982359930005737e-03,
    3.59207910759131235356e-03,
    1.45620945432529025516e-03,
    5.88041240820264096874e-04,
    2.46463134818469906812e-04,
    7.81794442939557092300e-05,
    7.14072491382608190305e-05,
    -1.85586374855275456654e-05,
    2.59073051863633712884e-05,
)
_PIO4 = 7.85398163397448278999e-01
_PIO4LO = 3.06161699786838301793e-17


def _kernel_tan(x: float, y: float, iy: int) -> float:
    hx = _hi(x)
    ix = hx & 0x7fffffff
    if ix < 0x3e300000:                     # |x| < 2**-28
        if ((ix | _lo(x)) | (iy + 1)) == 0:
            return _math.inf                # 1 / |x| of a zero
        if iy == 1:
            return x
        # compute -1 / (x+y) carefully
        w = x + y
        z = _set_lo(w, 0)
        v = y - (z - x)
        a = _div(-1.0, w)
        t = _set_lo(a, 0)
        s = 1.0 + t * z
        return t + a * (s + t * v)
    if ix >= 0x3FE59428:                    # |x| >= 0.6744
        if hx < 0:
            x = -x
            y = -y
        z = _PIO4 - x
        w = _PIO4LO - y
        x = z + w
        y = 0.0
    z = x * x
    w = z * z
    r = _T[1] + w * (_T[3] + w * (_T[5] + w * (_T[7] + w * (_T[9] + w * _T[11]))))
    v = z * (_T[2] + w * (_T[4] + w * (_T[6] + w * (_T[8] + w * (_T[10] + w * _T[12])))))
    s = z * x
    r = y + z * (s * (r + v) + y)
    r += _T[0] * s
    w = x + r
    if ix >= 0x3FE59428:
        v = float(iy)
        return float(1 - ((hx >> 30) & 2)) * (v - 2.0 * (x - (w * w / (w + v) - r)))
    if iy == 1:
        return w
    # compute -1.0 / (x+r) accurately
    z = _set_lo(w, 0)
    v = r - (z - x)
    a = -1.0 / w
    t = _set_lo(a, 0)
    s = 1.0 + t * z
    return t + a * (s + t * v)


def tan(x: float) -> float:
    ix = _hi(x) & 0x7fffffff
    if ix <= 0x3fe921fb:
        return _kernel_tan(x, 0.0, 1)
    if ix >= 0x7ff00000:
        return x - x
    n, y0, y1 = _rem_pio2(x)
    return _kernel_tan(y0, y1, 1 - ((n & 1) << 1))


# --- pow (e_pow.c) -----------------------------------------------------------

_BP = (1.0, 1.5)
_DP_H = (0.0, 5.84962487220764160156e-01)
_DP_L = (0.0, 1.35003920212974897128e-08)
_TWO53 = 9007199254740992.0
_TINY = 1.0e-300
_PL1 = 5.99999999999994648725e-01
_PL2 = 4.28571428578550184252e-01
_PL3 = 3.33333329818377432918e-01
_PL4 = 2.72728123808534006489e-01
_PL5 = 2.30660745775561754067e-01
_PL6 = 2.06975017800338417784e-01
_LG2_P = 6.93147180559945286227e-01
_LG2_H = 6.93147182464599609375e-01
_LG2_L = -1.90465429995776804525e-09
_OVT = 8.0085662595372944372e-17
_CP = 9.61796693925975554329e-01
_CP_H = 9.61796700954437255859e-01
_CP_L = -7.02846165095275826516e-09
_IVLN2 = 1.44269504088896338700e+00
_IVLN2_H = 1.44269502162933349609e+00
_IVLN2_L = 1.92596299112661746887e-08


def _div(a: float, b: float) -> float:
    """C's ``a / b``, where a zero divisor gives an infinity or a nan instead of raising."""
    if b != 0.0:
        return a / b
    if a != a or a == 0.0:
        return _math.nan
    return _math.copysign(_math.inf, a) * _math.copysign(1.0, b)


def pow(x: float, y: float) -> float:
    hx = _hi(x)
    lx = _lo(x)
    hy = _hi(y)
    ly = _lo(y)
    ix = hx & 0x7fffffff
    iy = hy & 0x7fffffff

    if (iy | ly) == 0:                      # x**0 = 1
        return 1.0
    if ix > 0x7ff00000 or (ix == 0x7ff00000 and lx != 0) \
            or iy > 0x7ff00000 or (iy == 0x7ff00000 and ly != 0):
        return x + y                        # +-NaN

    # yisint: 0 -- y is not an integer, 1 -- odd integer, 2 -- even integer
    yisint = 0
    if hx < 0:
        if iy >= 0x43400000:
            yisint = 2
        elif iy >= 0x3ff00000:
            k = (iy >> 20) - 0x3ff
            if k > 20:
                j = ly >> (52 - k)
                if (j << (52 - k)) == ly:
                    yisint = 2 - (j & 1)
            elif ly == 0:
                j = iy >> (20 - k)
                if (j << (20 - k)) == iy:
                    yisint = 2 - (j & 1)

    if ly == 0:
        if iy == 0x7ff00000:                # y is +-inf
            if ((ix - 0x3ff00000) | lx) == 0:
                return y - y                # inf**+-1 is NaN
            if ix >= 0x3ff00000:            # (|x|>1)**+-inf = inf,0
                return y if hy >= 0 else 0.0
            return -y if hy < 0 else 0.0    # (|x|<1)**-,+inf = inf,0
        if iy == 0x3ff00000:                # y is +-1
            return _div(1.0, x) if hy < 0 else x
        if hy == 0x40000000:                # y is 2
            return x * x
        if hy == 0x3fe00000 and hx >= 0:    # y is 0.5, x >= +0
            return _math.sqrt(x)

    ax = abs(x)
    if lx == 0 and (ix == 0x7ff00000 or ix == 0 or ix == 0x3ff00000):
        z = ax                              # x is +-0, +-inf, +-1
        if hy < 0:
            z = _div(1.0, z)
        if hx < 0:
            if ((ix - 0x3ff00000) | yisint) == 0:
                z = _math.nan               # (-1)**non-int is NaN
            elif yisint == 1:
                z = -z                      # (x<0)**odd = -(|x|**odd)
        return z

    n = (hx >> 31) + 1
    if (n | yisint) == 0:
        return _math.nan                    # (x<0)**(non-int) is NaN

    s = 1.0                                 # sign of the result
    if (n | (yisint - 1)) == 0:
        s = -1.0                            # (-ve)**(odd int)

    if iy > 0x41e00000:                     # |y| > 2**31
        if iy > 0x43f00000:                 # |y| > 2**64, must o/uflow
            if ix <= 0x3fefffff:
                return _HUGE * _HUGE if hy < 0 else _TINY * _TINY
            if ix >= 0x3ff00000:
                return _HUGE * _HUGE if hy > 0 else _TINY * _TINY
        # over/underflow if x is not close to one
        if ix < 0x3fefffff:
            return s * _HUGE * _HUGE if hy < 0 else s * _TINY * _TINY
        if ix > 0x3ff00000:
            return s * _HUGE * _HUGE if hy > 0 else s * _TINY * _TINY
        # |1-x| <= 2**-20: log(x) by x-x^2/2+x^3/3-x^4/4
        t = ax - 1.0
        w = (t * t) * (0.5 - t * (0.3333333333333333333333 - t * 0.25))
        u = _IVLN2_H * t
        v = t * _IVLN2_L - w * _IVLN2
        t1 = _set_lo(u + v, 0)
        t2 = v - (t1 - u)
    else:
        n = 0
        if ix < 0x00100000:                 # subnormal x
            ax *= _TWO53
            n -= 53
            ix = _hi(ax)
        n += (ix >> 20) - 0x3ff
        j = ix & 0x000fffff
        ix = j | 0x3ff00000                 # normalize ix
        if j <= 0x3988E:                    # |x| < sqrt(3/2)
            k = 0
        elif j < 0xBB67A:                   # |x| < sqrt(3)
            k = 1
        else:
            k = 0
            n += 1
            ix -= 0x00100000
        ax = _set_hi(ax, ix)

        # ss = s_h+s_l = (x-1)/(x+1) or (x-1.5)/(x+1.5)
        u = ax - _BP[k]
        v = 1.0 / (ax + _BP[k])
        ss = u * v
        s_h = _set_lo(ss, 0)
        # t_h = ax+bp[k] High
        t_h = _set_hi(0.0, ((ix >> 1) | 0x20000000) + 0x00080000 + (k << 18))
        t_l = ax - (t_h - _BP[k])
        s_l = v * ((u - s_h * t_h) - s_h * t_l)
        # log(ax)
        s2 = ss * ss
        r = s2 * s2 * (_PL1 + s2 * (_PL2 + s2 * (_PL3 + s2 * (_PL4 + s2 * (_PL5 + s2 * _PL6)))))
        r += s_l * (s_h + ss)
        s2 = s_h * s_h
        t_h = _set_lo(3.0 + s2 + r, 0)
        t_l = r - ((t_h - 3.0) - s2)
        # u+v = ss*(1+...)
        u = s_h * t_h
        v = s_l * t_h + t_l * ss
        # 2/(3log2)*(ss+...)
        p_h = _set_lo(u + v, 0)
        p_l = v - (p_h - u)
        z_h = _CP_H * p_h
        z_l = _CP_L * p_h + p_l * _CP + _DP_L[k]
        # log2(ax) = (ss+..)*2/(3*log2) = n + dp_h + z_h + z_l
        t = float(n)
        t1 = _set_lo(((z_h + z_l) + _DP_H[k]) + t, 0)
        t2 = z_l - (((t1 - t) - _DP_H[k]) - z_h)

    # split up y into y1+y2 and compute (y1+y2)*(t1+t2)
    y1 = _set_lo(y, 0)
    p_l = (y - y1) * t1 + y * t2
    p_h = y1 * t1
    z = p_l + p_h
    j = _hi(z)
    i = _lo(z)
    if j >= 0x40900000:                     # z >= 1024
        if ((j - 0x40900000) | i) != 0:
            return s * _HUGE * _HUGE        # overflow
        if p_l + _OVT > z - p_h:
            return s * _HUGE * _HUGE
    elif (j & 0x7fffffff) >= 0x4090cc00:    # z <= -1075
        if (((j & _M32) - 0xc090cc00) | i) != 0:
            return s * _TINY * _TINY        # underflow
        if p_l <= z - p_h:
            return s * _TINY * _TINY

    # 2**(p_h+p_l)
    i = j & 0x7fffffff
    k = (i >> 20) - 0x3ff
    n = 0
    if i > 0x3fe00000:                      # |z| > 0.5: n = [z+0.5]
        n = j + (0x00100000 >> (k + 1))
        k = ((n & 0x7fffffff) >> 20) - 0x3ff
        t = _set_hi(0.0, n & ~(0x000fffff >> k))
        n = ((n & 0x000fffff) | 0x00100000) >> (20 - k)
        if j < 0:
            n = -n
        p_h -= t
    t = _set_lo(p_l + p_h, 0)
    u = t * _LG2_H
    v = (p_l - (t - p_h)) * _LG2_P + t * _LG2_L
    z = u + v
    w = v - (z - u)
    t = z * z
    t1 = z - t * (_EP1 + t * (_EP2 + t * (_EP3 + t * (_EP4 + t * _EP5))))
    r = (z * t1) / (t1 - 2.0) - (w + z * w)
    z = 1.0 - (r - z)
    j = _hi(z) + (n << 20)
    if (j >> 20) <= 0:
        z = _math.ldexp(z, n)               # subnormal output
    else:
        z = _set_hi(z, _hi(z) + (n << 20))
    return s * z


#: The pure-Python implementations. ``_native_math``, when built, rebinds the public
#: names above to compiled twins of these (bit-identical, see that module); this keeps
#: the originals reachable for the tests that hold the two paths against each other.
PYTHON_IMPLEMENTATIONS = {'sin': sin, 'cos': cos, 'exp': exp, 'asin': asin, 'acos': acos, 'atan': atan}

if not _os.environ.get('PYNE_NO_NATIVE_MATH'):
    try:
        from . import _native_math  # noqa: F401 -- importing it installs the twins
    except ImportError:
        pass
