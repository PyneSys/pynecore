"""Bit-exact Python port of fdlibm 5.3 sin/cos/exp/asin/acos.

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
"""
import math as _math
from struct import pack as _pack, unpack as _unpack

__all__ = ['sin', 'cos', 'exp', 'asin', 'acos']

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

    jk = _INIT_JK[prec]
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
    k = 0
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
