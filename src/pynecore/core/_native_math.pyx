# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: infer_types=False, initializedcheck=False
"""Compiled twins of :mod:`pynecore.core.pine_math` and :mod:`pynecore.core.fdlibm`.

Every function here is the SAME algorithm as its pure-Python original, operation for
operation, only compiled: the Python modules stay the reference implementation and the
fallback wherever this extension is not built (source installs without a compiler,
WebAssembly runtimes), and they swap these in at import when it is.

Bit-identity with the originals rests on three things:

- the C compiler must evaluate every expression exactly as written: no fused
  multiply-add contraction and no reassociation. The build passes ``-ffp-contract=off``
  (``/fp:precise`` on MSVC) and never ``-ffast-math``; clang contracts ``a * b + c``
  into an FMA by default, which silently changes last bits on arm64;
- integer steps use fixed-width unsigned arithmetic wherever the original masks to 32
  or 64 bits, and 64-bit signed arithmetic otherwise, so no Python big-int semantics
  leak in;
- the rare paths that need wider-than-64-bit integers (the fixed-point 2/pi reduction
  of huge sin/cos arguments) call back into the Python original instead of being
  re-implemented.

The numeric tables are copied from the Python modules at import, so they have one
source of truth.
"""
from libc.stdint cimport uint64_t, int64_t, uint32_t, int32_t
from libc.string cimport memcpy
from libc.math cimport fabs, copysign, sqrt

from . import pine_math as _pm
from . import fdlibm as _fd

# The pure-Python originals the compiled code falls back to on its rare wide-integer
# paths. Captured here, before ``_install`` below rebinds the public names to the
# compiled twins -- calling ``_fd.sin`` afterwards would recurse into ``f_sin``.
_py_fd_sin = _fd.PYTHON_IMPLEMENTATIONS['sin']
_py_fd_cos = _fd.PYTHON_IMPLEMENTATIONS['cos']

__all__ = ['cos', 'sin', 'exp', 'log', 'log10', 'pow',
           'f_sin', 'f_cos', 'f_exp', 'f_asin', 'f_acos', 'f_atan']

cdef uint64_t M64 = 0xFFFFFFFFFFFFFFFFULL


cdef inline uint64_t f2b(double f) noexcept nogil:
    cdef uint64_t b
    memcpy(&b, &f, 8)
    return b


cdef inline double b2f(uint64_t b) noexcept nogil:
    cdef double f
    memcpy(&f, &b, 8)
    return f


# --- pine_math: cos / sin ---------------------------------------------------------

cdef double PI32INV = _pm._PI32INV
cdef double P1 = _pm._P1
cdef double P2 = _pm._P2
cdef double P3 = _pm._P3
cdef double SC1_LO = _pm._SC1_LO
cdef double SC1_HI = _pm._SC1_HI
cdef double SC2_LO = _pm._SC2_LO
cdef double SC2_HI = _pm._SC2_HI
cdef double SC3_LO = _pm._SC3_LO
cdef double SC3_HI = _pm._SC3_HI
cdef double SC4_LO = _pm._SC4_LO
cdef double SC4_HI = _pm._SC4_HI

cdef double CT0F[64]
cdef double CT8[64]
cdef double CT16[64]
cdef double CT24[64]

cdef int _i
for _i in range(64):
    CT0F[_i] = _pm._CT0F[_i]
    CT8[_i] = _pm._CT8[_i]
    CT16[_i] = _pm._CT16[_i]
    CT24[_i] = _pm._CT24[_i]


cdef double sincos_poly(double x, int64_t n, int j, double corr) noexcept nogil:
    cdef double nf = <double>n
    cdef double p1n = P1 * nf
    cdef double p2n = P2 * nf
    cdef double rr = x - p1n
    cdef double r = rr - p2n
    cdef double c = (rr - r) - p2n
    cdef double m = (nf * P3 - c) - corr
    cdef double r2 = r * r
    cdef double r4 = r2 * r2
    cdef double c0 = CT0F[j]
    cdef double c8 = CT8[j]
    cdef double q = c0 + CT24[j]
    cdef double c8r_q = c8 * r - q
    cdef double c24r = CT24[j] * r
    cdef double poly_lo = ((SC2_LO * r2 + SC1_LO) + (SC4_LO * rr * r + SC3_LO) * r4) * (q * r * r2)
    cdef double poly_hi = ((SC2_HI * r2 + SC1_HI) + (SC4_HI * rr * r + SC3_HI) * r4) * (c8 * r2)
    cdef double t4 = r * c0
    cdef double x3 = c8 + c24r
    cdef double m_corr = m * c8r_q + CT16[j]
    cdef double s = t4 + x3
    cdef double lo5 = c8 - x3
    cdef double x3s = x3 - s
    cdef double lo0 = c24r + lo5
    cdef double x3t = x3s + t4
    cdef double total = lo0 + m_corr
    total = total + x3t
    total = total + poly_lo
    total = total + poly_hi
    return total + s


cdef inline int64_t round_n(double x) noexcept nogil:
    return <int64_t>(x * PI32INV + copysign(0.5, x))


def cos(double x):
    """Bit-exact venue-runtime cos (compiled :func:`pine_math.cos`)."""
    cdef uint64_t x_bits = f2b(x)
    cdef uint32_t band = <uint32_t>(<uint32_t>((x_bits >> 32) & 2147418112u) - 808452096u)
    cdef int64_t n
    if band > 281346048u:
        if <uint32_t>(band - 281346048u) < 0x80000000u:
            return _pm._reduce_huge(x_bits, 1865232, True)
        return 1.0 - fabs(x)
    n = round_n(x)
    return sincos_poly(x, n, <int>((n + 16) & 63), 0.0)


def sin(double x):
    """Bit-exact venue-runtime sin (compiled :func:`pine_math.sin`)."""
    cdef uint64_t x_bits = f2b(x)
    cdef uint32_t band = <uint32_t>(<uint32_t>((x_bits >> 32) & 2147418112u) - 808452096u)
    cdef int64_t n
    if band > 281346048u:
        if <uint32_t>(band - 281346048u) < 0x80000000u:
            return _pm._reduce_huge(x_bits, 1865216, False)
        if (band >> 20) == 3325u:
            return x * b2f(0x3fefffffffffffffULL)
        return x
    n = round_n(x)
    return sincos_poly(x, n, <int>(n & 63), 0.0)


# --- pine_math: exp ----------------------------------------------------------------

cdef double E_LOG2_64 = _pm._E_LOG2_64
cdef double E_LN2_64_HEAD = _pm._E_LN2_64_HEAD
cdef double E_LN2_64_TAIL = _pm._E_LN2_64_TAIL
cdef double E_HALF = _pm._E_HALF
cdef double E_P3_LO = _pm._E_P3_LO
cdef double E_P3_HI = _pm._E_P3_HI
cdef double E_P5_LO = _pm._E_P5_LO
cdef double E_P5_HI = _pm._E_P5_HI
cdef double E_SHIFTER = _pm._E_SHIFTER
cdef double E_XMAX = _pm._E_XMAX
cdef double E_XMIN = _pm._E_XMIN
cdef uint64_t E_INF_BITS = _pm._E_INF_BITS

cdef double ET_LO[64]
cdef uint64_t ET_HI_BITS[64]
for _i in range(64):
    ET_LO[_i] = _pm._ET_LO[_i]
    ET_HI_BITS[_i] = _pm._ET_HI_BITS[_i]


cdef double c_exp(double x) noexcept nogil:
    cdef uint64_t x_bits = f2b(x)
    cdef uint32_t hi32 = <uint32_t>(x_bits >> 32)
    cdef uint32_t top15 = <uint32_t>((x_bits >> 48) & 32767u)
    cdef uint32_t mag
    cdef double s, nd, y, y2, y3, p_lo, p_hi, res2, acc
    cdef int64_t n64, n
    cdef int j
    cdef uint64_t scale_bits, res2_bits, mask
    cdef uint32_t shift, delta
    cdef double res2_adj, scale3, acc2, dropped, low_part, saved, result, a, bpart
    cdef uint64_t a_bits, b_bits, neg

    if <uint32_t>((16527u - top15) | (top15 - 15504u)) >= 0x80000000u:
        mag = hi32 & 2147483647u
        if mag >= 1083179008u:
            if mag < 2146435072u:
                if hi32 >= 0x80000000u:
                    return E_XMIN * E_XMIN
                return E_XMAX * E_XMAX
            if mag > 2146435072u or (x_bits & 0xFFFFFFFFULL) != 0:
                return x + x
            if hi32 == 2146435072u:
                return b2f(E_INF_BITS)
            return 0.0
        return x + 1.0

    s = x * E_LOG2_64 + E_SHIFTER
    nd = s - E_SHIFTER
    n64 = <int64_t>nd
    j = <int>(n64 & 63)
    n = n64 >> 6
    y = x - nd * E_LN2_64_HEAD
    y = y - nd * E_LN2_64_TAIL
    scale_bits = ((<uint64_t>n64 & 0xffffffc0ULL) + 0x0000ffc0ULL) << 46
    y2 = y * y
    y3 = y * y2
    p_lo = y3 * y2 * (E_P5_LO + E_P3_LO * y)
    p_hi = y3 * (E_P5_HI + E_P3_HI * y)
    res2_bits = ET_HI_BITS[j] | scale_bits
    res2 = b2f(res2_bits)
    acc = (y + ET_LO[j]) + p_lo
    acc = p_hi + acc
    acc = acc + E_HALF * y2
    if 0 <= n + 894 <= 1916:
        return acc * res2 + res2

    shift = <uint32_t>(-1022 - n)
    mask = (M64 << shift) if shift < 64u else 0
    delta = <uint32_t>((<uint64_t>(n >> 1) & 0xFFFFULL) << 20)
    res2_adj = b2f(res2_bits - (<uint64_t>delta << 32))
    scale3 = b2f(<uint64_t>(<uint32_t>(delta + 0x3ff00000u)) << 32)
    acc2 = acc * res2_adj
    if shift <= 52u:
        dropped = b2f(f2b(res2_adj) & mask)
        low_part = res2_adj - dropped
        acc2 = acc2 + low_part
        if n >= 1023:
            return (acc2 + dropped) * scale3
        neg = (f2b(acc2) >> 48) & 32768u
        if (shift | neg) == 0:
            return (acc2 + dropped) * scale3
        saved = acc2
        result = (acc2 + dropped) * scale3
        if (f2b(result) >> 48) & 32752u:
            return result
        a = saved * scale3
        bpart = dropped * scale3
        a_bits = f2b(a)
        b_bits = f2b(bpart)
        if (a_bits ^ b_bits) >> 63:
            return b2f(b_bits - (a_bits & 0x7FFFFFFFFFFFFFFFULL))
        return b2f(b_bits + (a_bits & 0x7FFFFFFFFFFFFFFFULL))
    return (acc2 + res2_adj) * scale3


def exp(double x):
    """Bit-exact venue-runtime exp (compiled :func:`pine_math.exp`)."""
    return c_exp(x)


# --- pine_math: log ----------------------------------------------------------------

cdef double L_LOG2_HI = _pm._L_LOG2_HI
cdef double L_LOG2_LO = _pm._L_LOG2_LO
cdef double L_C7 = _pm._L_C7
cdef double L_C6 = _pm._L_C6
cdef double L_C5 = _pm._L_C5
cdef double L_C4 = _pm._L_C4
cdef double L_C3 = _pm._L_C3
cdef double L_C2 = _pm._L_C2
cdef double L_INF = _pm._L_INF
cdef double L_NAN = _pm._L_NAN
cdef double L_MANT_SCALE = _pm._L_MANT_SCALE
cdef double L_ULP52 = _pm._L_ULP52
cdef double L_DENORM_SCALE = _pm._L_DENORM_SCALE
cdef int64_t L_BIAS = _pm._L_BIAS
cdef int64_t L_DENORM_BIAS = _pm._L_DENORM_BIAS

cdef uint32_t RCP[4096]
for _i in range(4096):
    RCP[_i] = _pm._RCP[_i]

cdef double L_THI[129]
cdef double L_TLO[129]
for _i in range(129):
    L_THI[_i] = _pm._L_THI[_i]
    L_TLO[_i] = _pm._L_TLO[_i]


cdef double c_log(double x) noexcept nogil:
    cdef uint64_t bits = f2b(x)
    cdef uint32_t head = <uint32_t>((bits >> 48) & 0xFFFFu)
    cdef uint32_t biased = <uint32_t>(head - 16u)
    cdef int64_t bias
    cdef uint32_t lane
    cdef double b, mant_hi, r, k, head_term, result, r2, tail
    cdef uint64_t fraction
    cdef int bin_index

    if biased >= 32736u:
        if head < 16u:
            if bits == 0:
                return -L_INF
            bits = f2b(x * L_DENORM_SCALE)
            head = <uint32_t>((bits >> 48) & 0xFFFFu)
            bias = L_DENORM_BIAS
        elif head < 32768u:
            return x + x
        elif (bits << 1) == 0:
            return -L_INF
        elif head >= 0xFFF0u and (bits & 0xFFFFFFFFFFFFFULL) != 0:
            return x + x
        else:
            return L_NAN
    else:
        head = biased
        bias = L_BIAS

    lane = RCP[(bits >> 40) & 0xFFFu] + 32768u
    b = b2f((<uint64_t>lane << 29) & 0xffffe00000000000ULL)

    fraction = bits & 0xFFFFFFFFFFFFFULL
    mant_hi = (1.0 + <double>(fraction & 0x000FE00000000000ULL) * L_ULP52) * L_MANT_SCALE
    r = ((1.0 + <double>fraction * L_ULP52) * L_MANT_SCALE - mant_hi) * b \
        + (mant_hi * b - 1.0)

    bin_index = <int>((lane >> 16) & 0xFFu)
    k = <double>(<int64_t>(head & 32752u) - bias)
    head_term = L_THI[bin_index] + k * L_LOG2_HI
    result = head_term + r

    r2 = r * r
    tail = r + (head_term - result)
    tail = tail + (k * L_LOG2_LO + L_TLO[bin_index])
    tail = tail + ((L_C6 * r + L_C5) * r + (L_C7 * r) * r2) * (r2 * r2)
    tail = tail + ((L_C3 * r + L_C2) + L_C4 * r2) * r2
    return result + tail


def log(double x):
    """``ln(x)`` as the venue's engine computes it (compiled :func:`pine_math.log`)."""
    return c_log(x)


# --- pine_math: log10 --------------------------------------------------------------

cdef double L10_LH = _pm._L10_LH
cdef float L10_LH_F32 = _pm._L10_LH_F32
cdef double L10_LE_TAIL = _pm._L10_LE_TAIL
cdef double L10_LOG2_HI = _pm._L10_LOG2_HI
cdef double L10_LOG2_LO = _pm._L10_LOG2_LO
cdef double L10_C0 = _pm._L10_C0
cdef double L10_C1 = _pm._L10_C1
cdef double L10_C2 = _pm._L10_C2
cdef double L10_C3 = _pm._L10_C3
cdef double L10_C4 = _pm._L10_C4
cdef double L10_C5 = _pm._L10_C5

cdef double L10_THI[129]
cdef double L10_TLO[129]
for _i in range(129):
    L10_THI[_i] = _pm._L10_THI[_i]
    L10_TLO[_i] = _pm._L10_TLO[_i]


cdef inline uint32_t scaled_rcp_lane(uint32_t rcp_bits, float scale, uint32_t half) noexcept nogil:
    """The reciprocal times ``scale`` rounded to float32, plus the grid's half-step."""
    cdef float rcp, prod
    cdef uint32_t bits
    memcpy(&rcp, &rcp_bits, 4)
    prod = rcp * scale
    memcpy(&bits, &prod, 4)
    return bits + half


cdef double c_log10(double x) noexcept nogil:
    cdef uint64_t bits = f2b(x)
    cdef uint32_t head = <uint32_t>((bits >> 48) & 0xFFFFu)
    cdef uint32_t biased = <uint32_t>(head - 16u)
    cdef int64_t bias
    cdef uint32_t lane
    cdef double b, mant, mant_hi, r, k, head_term, result, r2, tail
    cdef int bin_index

    if biased >= 32736u:
        if head < 16u:
            if bits == 0:
                return -L_INF
            bits = f2b(x * L_DENORM_SCALE)
            head = <uint32_t>((bits >> 48) & 0xFFFFu)
            bias = L_DENORM_BIAS
        elif head < 32768u:
            return x + x
        elif (bits << 1) == 0:
            return -L_INF
        elif head >= 0xFFF0u and (bits & 0xFFFFFFFFFFFFFULL) != 0:
            return x + x
        else:
            return L_NAN
    else:
        head = biased
        bias = L_BIAS

    lane = scaled_rcp_lane(RCP[(bits >> 40) & 0xFFFu], L10_LH_F32, 0x8000u)
    b = b2f((<uint64_t>lane << 29) & 0xffffe00000000000ULL)

    mant = (1.0 + <double>(bits & 0xFFFFFFFFFFFFFULL) * L_ULP52) * L_MANT_SCALE
    mant_hi = b2f(f2b(mant) & 0xfffffffff8000000ULL)
    r = (mant - mant_hi) * b + (mant_hi * b - L10_LH)

    bin_index = <int>((lane >> 16) & 0xFFu) - 94
    k = <double>(<int64_t>(head & 32752u) - bias)
    head_term = L10_THI[bin_index] + k * L10_LOG2_HI
    result = head_term + r

    r2 = r * r
    tail = r + (head_term - result)
    tail = tail + L10_LE_TAIL * r
    tail = tail + (k * L10_LOG2_LO + L10_TLO[bin_index])
    tail = tail + ((L10_C2 * r + L10_C4) * r + (L10_C0 * r) * r2) * (r2 * r2)
    tail = tail + ((L10_C3 * r + L10_C5) + L10_C1 * r2) * r2
    return result + tail


def log10(double x):
    """``log10(x)`` as the venue's engine computes it (compiled :func:`pine_math.log10`)."""
    return c_log10(x)


# --- pine_math: pow ----------------------------------------------------------------

cdef double P_LH = _pm._P_LH
cdef float P_LH_F32 = _pm._P_LH_F32
cdef double P_C0 = _pm._P_C0
cdef double P_C1 = _pm._P_C1
cdef double P_C2 = _pm._P_C2
cdef double P_C3 = _pm._P_C3
cdef double P_C4 = _pm._P_C4
cdef double P_C5 = _pm._P_C5
cdef double P_C6 = _pm._P_C6
cdef double P_C7 = _pm._P_C7
cdef double P_C8 = _pm._P_C8
cdef double P_C9 = _pm._P_C9
cdef double P_C10 = _pm._P_C10
cdef double P_C11 = _pm._P_C11
cdef double P_CH0 = _pm._P_CH0
cdef double P_CH1 = _pm._P_CH1
cdef double P_E0 = _pm._P_E0
cdef double P_E1 = _pm._P_E1
cdef double P_E2 = _pm._P_E2
cdef double P_E3 = _pm._P_E3
cdef double P_LN2 = _pm._P_LN2
cdef double P_SHIFTER = _pm._P_SHIFTER
cdef double P_INT_SHIFTER = _pm._P_INT_SHIFTER
cdef uint64_t P_MANT_SCALE = _pm._P_MANT_SCALE
cdef double P_TWO128 = _pm._P_TWO128
cdef uint64_t P_FRACTION = _pm._P_FRACTION
cdef double P_NAN = _pm._P_NAN

cdef double P_LOG_HI[513]
cdef double P_LOG_LO[513]
cdef uint64_t P_EXP_HI[256]
cdef uint64_t P_EXP_LO[256]
for _i in range(513):
    P_LOG_HI[_i] = _pm._P_LOG_HI[_i]
    P_LOG_LO[_i] = _pm._P_LOG_LO[_i]
for _i in range(256):
    P_EXP_HI[_i] = _pm._P_EXP_HI[_i]
    P_EXP_LO[_i] = _pm._P_EXP_LO[_i]


cdef inline uint32_t sar32(uint32_t v, int n) noexcept nogil:
    return <uint32_t>((<int32_t>v) >> n)


cdef inline int bit_length(uint64_t v) noexcept nogil:
    cdef int n = 0
    while v:
        v >>= 1
        n += 1
    return n


cdef inline uint64_t add_high_word(uint64_t bits, uint32_t add) noexcept nogil:
    return (<uint64_t>(<uint32_t>((bits >> 32) + add)) << 32) | (bits & 0xFFFFFFFFULL)


cdef double c_pow(double x, double y) noexcept nogil:
    cdef uint64_t xb = f2b(x)
    cdef uint64_t yb = f2b(y)
    cdef uint64_t head
    if yb == 0x4000000000000000ULL:
        return x * x
    if yb == 0x3FE0000000000000ULL and xb < 0x8000000000000000ULL:
        return sqrt(x)
    head = xb >> 48
    if 16 <= head < 32752:
        return pow_finite(xb, <int64_t>head - 16, y, yb, 0, False, False)
    if (head & 0x7FF0) == 0x7FF0:
        return pow_x_inf_nan(x, xb, y, yb)
    if head & 0x8000:
        return pow_x_negative(x, xb, y, yb)
    if xb == 0:
        return pow_x_zero(x, y, yb)
    return pow_finite(xb, 0, y, yb, 0, True, False)


cdef bint pow_odd_integer(double y, uint64_t yb) noexcept nogil:
    cdef uint64_t ey = (yb >> 52) & 2047
    cdef double rounded, fraction
    if ey > 1075 or ey < 1023:
        return False
    if ey == 1075:
        return (yb & 1) != 0
    rounded = y + P_INT_SHIFTER
    fraction = y + (P_INT_SHIFTER - rounded)
    return ((f2b(fraction) >> 48) & 0x7FF0) == 0 and (f2b(rounded) & 1) != 0


cdef double pow_x_inf_nan(double x, uint64_t xb, double y, uint64_t yb) noexcept nogil:
    cdef bint y_negative
    if xb & P_FRACTION:
        return 1.0 if (yb << 1) == 0 else x + x
    if (yb << 1) == 0:
        return 1.0
    if ((yb >> 48) & 0x7FF0) == 0x7FF0 and (yb & P_FRACTION):
        return y + y
    y_negative = (yb >> 63) != 0
    if (xb >> 63) and pow_odd_integer(y, yb):
        return -0.0 if y_negative else x
    if not (xb >> 63):
        return 0.0 if y_negative else x
    return 0.0 if y_negative else L_INF


cdef double pow_x_zero(double x, double y, uint64_t yb) noexcept nogil:
    if ((yb >> 48) & 0x7FF0) == 0x7FF0 and (yb & P_FRACTION):
        return y + y
    if (yb << 1) == 0:
        return 1.0
    if pow_odd_integer(y, yb):
        return copysign(L_INF, x) if (yb >> 63) else x
    return L_INF if (yb >> 63) else 0.0


cdef double pow_x_negative(double x, uint64_t xb, double y, uint64_t yb) noexcept nogil:
    cdef uint64_t ey
    cdef uint32_t e, low, sign
    cdef bint odd
    cdef double rounded, fraction
    cdef int64_t head
    if xb == 0x8000000000000000ULL:
        return pow_x_zero(x, y, yb)
    if (yb << 1) == 0:
        return 1.0
    ey = (yb >> 48) & 0x7FF0
    if ey == 0x7FF0:
        if yb & P_FRACTION:
            return y + y
        if xb == 0xBFF0000000000000ULL:
            return P_NAN
        e = <uint32_t>((<int64_t>((xb >> 48) & 0x7FF0)) - 16368)
        return L_INF if ((e ^ <uint32_t>(yb >> 48)) & 0x8000u) == 0 else 0.0
    if ey > 17200:
        odd = False
    elif ey >= 17184:
        low = <uint32_t>(yb & 0xFFFFFFFFULL)
        if ey > 17184:
            odd = (low & 1u) != 0
        elif low & 1u:
            return P_NAN
        else:
            odd = (low & 2u) != 0
    elif ey < 16368:
        return P_NAN
    else:
        rounded = y + P_INT_SHIFTER
        fraction = y + (P_INT_SHIFTER - rounded)
        if (f2b(fraction) >> 48) & 0x7FFF:
            return P_NAN
        odd = (f2b(rounded) & 1) != 0
    sign = 0x80000000u if odd else 0u
    head = <int64_t>((xb >> 48) & 0x7FFF) - 16
    if head < 0:
        return pow_finite(xb, 0, y, yb, sign, True, False)
    return pow_finite(xb, head, y, yb, sign, False, True)


cdef double pow_finite(uint64_t xb, int64_t head, double y, uint64_t yb, uint32_t sign,
                       bint denormal, bint negative) noexcept nogil:
    cdef uint64_t xs, mask
    cdef uint32_t lane, near_one, k32
    cdef double mant, mant_hi, b
    cdef int64_t k, e16
    cdef int split
    if denormal:
        xs = f2b(b2f(xb) * P_TWO128)
        lane = scaled_rcp_lane(RCP[(xs >> 40) & 0xFFFu], P_LH_F32, 0x2000u)
        mant = b2f((xs & P_FRACTION) | P_MANT_SCALE)
        mant_hi = b2f(f2b(mant) & 0xFFFFFC0000000000ULL)
        b = b2f((<uint64_t>lane << 29) & 0xFFFFF80000000000ULL)
        k = (<int64_t>((xs >> 48) & 0x7FF0) - 18416) >> 4
        return pow_series(mant_hi * b, mant - mant_hi, b, k, lane, xb, y, yb, sign)

    if negative:
        e16 = (head & 0x7FF0) - 16368
    else:
        e16 = <int64_t>((xb >> 48) & 0x7FF0) - 16368
    split = bit_length(<uint64_t>((e16 if e16 >= 0 else -e16) + 16)) - 1
    lane = scaled_rcp_lane(RCP[(xb >> 40) & 0xFFFu], P_LH_F32, 0x2000u)
    mask = (<uint64_t>(<uint32_t>(0xFFFFFFFFu << ((split - 4) & 31)))) << 32
    mant = b2f((xb & P_FRACTION) | P_MANT_SCALE)
    near_one = <uint32_t>(head - 16351)
    if near_one <= 1u:
        mant_hi = b2f((f2b(mant) + 0x80000000ULL) & mask)
        if ((yb >> 48) & 0x7FF0) >= 16560:
            return pow_near_one(mant, mant_hi, lane, near_one, xb, y, yb, sign)
    else:
        mant_hi = b2f(f2b(mant) & mask)
    b = b2f((<uint64_t>lane << 29) & 0xFFFFF80000000000ULL)
    k32 = sar32(near_one - 1u, 4)
    k = <int64_t>(<int32_t>k32)
    return pow_series(mant_hi * b, mant - mant_hi, b, k, lane, xb, y, yb, sign)


cdef double pow_series(double head_b, double mant_lo, double b, int64_t k, uint32_t lane,
                       uint64_t xb, double y, uint64_t yb, uint32_t sign) noexcept nogil:
    cdef double lo_b, u, r, log_hi, log_lo, r2, r3, r4, q1, q1h, q2, q2h
    cdef double s1, s1h, s2, s2h, s3, s3h, y_hi, y_lo, m1, p_lo, m2, shifted
    cdef double s_lo, s_hi, rounded, p, d, z, t0, t1
    cdef int k_bits, row
    cdef uint32_t scale, n, exponent
    lo_b = mant_lo * b
    u = head_b - P_LH
    k_bits = bit_length(<uint64_t>((k if k >= 0 else -k) + 1)) - 1
    r = lo_b + u
    row = <int>((lane & 0xFFC000u) >> 14) - 228
    log_hi = u + P_LOG_HI[row]
    log_lo = lo_b + P_LOG_LO[row]
    r2 = r * r
    log_hi = log_hi + <double>k
    r3 = r * r2
    q1 = P_C4 + P_C0 * r
    q1h = P_C5 + P_C1 * r
    r4 = r2 * r2
    q2 = P_C6 + P_C2 * r
    q2h = P_C7 + P_C3 * r
    s1 = q1 * r3
    s1h = q1h * r
    s2 = q2 * r3
    s2h = q2h * r
    scale = <uint32_t>((k_bits << 4) - 15872 + <int64_t>((yb >> 48) & 0x7FF0))
    s3 = r4 * s1
    s3h = r4 * s1h
    if scale >= 624u:
        return pow_wide(scale, log_hi, log_lo, s2, s2h, s3, s3h, xb, y, yb, sign)

    y_hi = b2f(yb & 0xFFFFFFF800000000ULL)
    y_lo = y - y_hi
    m1 = y_hi * log_hi
    p_lo = s2 + log_lo
    m2 = y_lo * log_hi
    shifted = P_SHIFTER + m1
    s_lo = s3 + p_lo
    s_hi = s3h + s2h
    n = <uint32_t>(f2b(shifted) & 0xFFFFFFFFULL)
    rounded = shifted - P_SHIFTER
    p = s_hi + s_lo
    d = m1 - rounded
    d = d + m2
    z = y * p
    exponent = ((n << 12) ^ sign) & 0xFFF00000u
    z = z + d
    t0 = b2f(add_high_word(P_EXP_HI[n & 255], exponent))
    t1 = b2f(add_high_word(P_EXP_LO[n & 255], exponent))
    return pow_exp2(z, t0, t1)


cdef double pow_exp2(double z, double t0, double t1) noexcept nogil:
    cdef double w, z2, z4, f, fh, g, gh
    w = P_LN2 * z
    z2 = z * z
    w = w * t0
    z4 = z2 * z2
    f = P_E2 + P_E0 * z
    fh = P_E3 + P_E1 * z
    w = w + t1
    g = (f * z4) * t0
    gh = (fh * z2) * t0
    return ((g + w) + gh) + t0


cdef double pow_wide(uint32_t scale, double log_hi, double log_lo, double s2, double s2h,
                     double s3, double s3h, uint64_t xb, double y, uint64_t yb,
                     uint32_t sign) noexcept nogil:
    cdef double p, ln2, log2x, carry, l_hi, y_hi, l_tail, y_lo, m1, z, m2, shifted, rounded
    cdef uint32_t n, half, rest, exponent
    cdef double factor
    if scale >> 31:
        if (<uint32_t>(scale + 384u)) >> 31:
            return 1.0
        p = (s3 + (s2 + log_lo)) + (s3h + s2h)
        ln2 = -P_LN2 if sign else P_LN2
        return (-1.0 if sign else 1.0) + (log_hi * y + y * p) * ln2
    if scale >= 752u:
        return pow_out_of_range(xb, y, yb, sign)

    p = (s3h + s2h) + (s3 + (s2 + log_lo))
    log2x = log_hi + p
    carry = log_hi - log2x
    l_hi = b2f(f2b(log2x) & 0xFFFFFFFFF8000000ULL)
    y_hi = b2f(yb & 0xFFFFFFFFF8000000ULL)
    l_tail = log2x - l_hi
    p = p + carry
    y_lo = y - y_hi
    m1 = y_hi * l_hi
    p = p + l_tail
    z = y * p
    m2 = y_lo * l_hi
    shifted = P_SHIFTER + m1
    n = <uint32_t>(f2b(shifted) & 0xFFFFFFFFULL)
    rounded = shifted - P_SHIFTER
    z = z + m2
    z = z + (m1 - rounded)
    if ((f2b(rounded) >> 48) & 0x7FFF) > 16529:
        return pow_out_of_range(xb, y, yb, sign)
    half = sar32(sar32(n, 8), 1)
    rest = sar32(n, 8) - half
    exponent = (half << 20) ^ sign
    factor = b2f((<uint64_t>((((rest << 4) + 16368u)) & 0xFFFFu)) << 48)
    return pow_scaled(z, n & 255, exponent, factor)


cdef double pow_scaled(double z, uint32_t j, uint32_t exponent, double factor) noexcept nogil:
    cdef uint64_t t0_bits = add_high_word(P_EXP_HI[j], exponent)
    cdef double t0 = b2f(t0_bits)
    cdef double t1 = b2f(add_high_word(P_EXP_LO[j], exponent))
    cdef double w, z2, z4, f, fh, g, gh, s, result, t0_hi, t0_lo
    cdef int64_t shift
    cdef uint32_t mask
    w = P_LN2 * z
    z2 = z * z
    w = w * t0
    z4 = z2 * z2
    f = P_E2 + P_E0 * z
    fh = P_E3 + P_E1 * z
    w = w + t1
    g = (f * z4) * t0
    gh = (fh * z2) * t0
    s = (g + w) + gh
    result = (s + t0) * factor
    if (f2b(result) >> 48) & 0x7FF0:
        return result
    shift = -31 - ((<int64_t>((f2b(factor) >> 48) & 0x7FF0)
                    + <int64_t>((t0_bits >> 48) & 0x7FF0) - 16368) >> 4)
    if shift <= 0:
        mask = 0xFFFFFFFFu
    elif shift > 20:
        return result
    else:
        mask = 0xFFFFFFFFu << shift
    t0_hi = b2f(t0_bits & (<uint64_t>mask << 32))
    t0_lo = (t0 - t0_hi) + s
    return t0_hi * factor + t0_lo * factor


cdef double pow_out_of_range(uint64_t xb, double y, uint64_t yb, uint32_t sign) noexcept nogil:
    cdef uint32_t e
    if ((yb >> 48) & 0x7FF0) == 0x7FF0:
        return pow_y_inf_nan(xb, y, yb)
    e = <uint32_t>(<int64_t>((xb >> 48) & 0x7FF0) - 16368)
    if (<uint32_t>(yb >> 48) ^ e) & 0x8000u:
        return -0.0 if sign else 0.0
    return -L_INF if sign else L_INF


cdef double pow_y_inf_nan(uint64_t xb, double y, uint64_t yb) noexcept nogil:
    cdef uint32_t e
    if (f2b(b2f(xb) - 1.0) >> 48) == 0:
        return P_NAN
    if yb & P_FRACTION:
        return y + y
    e = <uint32_t>(<int64_t>((xb >> 48) & 0x7FF0) - 16368)
    return L_INF if ((e ^ <uint32_t>(yb >> 48)) & 0x8000u) == 0 else 0.0


cdef double pow_near_one(double mant, double mant_hi, uint32_t lane, uint32_t near_one,
                         uint64_t xb, double y, uint64_t yb, uint32_t sign) noexcept nogil:
    cdef double b, k, lo_b, u, r, acc, acc_tail, h4, h0, h6, h1, v, x0, ur, x2, x5, x4, x1
    cdef double r2, p0, p0h, p5, p5h, r3, r4, l_hi, q, qh, y_hi, y_lo, m1, m2
    cdef double shifted, rounded, z, factor
    cdef int row
    cdef uint32_t e, ey, n, half, rest
    b = b2f((<uint64_t>lane << 29) & 0xFFFFF80000000000ULL)
    k = <double>(<int64_t>((near_one + 16351u) >> 4) - 1022)
    lo_b = (mant - mant_hi) * b
    u = mant_hi * b - P_LH
    r = lo_b + u
    row = <int>((lane & 0xFFC000u) >> 14) - 228
    acc = k + P_LOG_HI[row]
    acc_tail = 0.0 + P_LOG_LO[row]
    h4 = P_CH0 * u
    h0 = P_CH1 * u
    h6 = P_CH0 * lo_b
    h1 = P_CH1 * lo_b
    h4 = h4 * u
    v = u + h0
    x0 = acc
    ur = u + r
    acc = acc + v
    h6 = h6 * ur
    x0 = x0 - acc
    x2 = acc
    acc = acc + h4
    x0 = x0 + v
    x2 = x2 - acc
    h4 = h4 + x2
    x5 = acc
    acc = acc + lo_b
    h4 = h4 + x0
    x5 = x5 - acc
    h6 = h6 + h4
    x4 = acc
    x5 = x5 + lo_b
    acc = acc + h1
    x4 = x4 - acc
    h6 = h6 + x5
    x4 = x4 + h1
    x1 = acc
    acc = acc + acc_tail
    x1 = x1 - acc
    x1 = x1 + acc_tail
    h6 = h6 + x4
    h6 = h6 + x1
    r2 = r * r
    p0 = P_C0 * r + P_C4
    p0h = P_C1 * r + P_C5
    p5 = P_C10 + P_C8 * r
    p5h = P_C11 + P_C9 * r
    r3 = r * r2
    r4 = r2 * r2
    p0 = p0 * r3 * r4
    p0h = p0h * r * r4
    p5 = p5 * r3
    p5h = p5h * r
    l_hi = b2f(f2b(acc) & 0xFFFFFFFFF8000000ULL)
    p5 = p5 + h6
    acc = acc - l_hi
    q = p5 + p0
    qh = p5h + p0h
    e = <uint32_t>(<int64_t>((f2b(l_hi) >> 48) & 0x7FF0) - 16368)
    ey = <uint32_t>((yb >> 48) & 0x7FF0)
    if ey == 0x7FF0u:
        return pow_y_inf_nan(xb, y, yb)
    if <uint32_t>(ey + e) >= 16576u:
        if ((<uint32_t>(yb >> 48) ^ <uint32_t>(f2b(l_hi) >> 48)) & 0x8000u) == 0:
            return -L_INF if sign else L_INF
        return -0.0 if sign else 0.0
    y_hi = b2f(yb & 0xFFFFFFFF00000000ULL)
    q = q + qh
    y_lo = y - y_hi
    acc = acc + q
    m1 = y_hi * l_hi
    m2 = y_lo * l_hi + y * acc
    shifted = P_SHIFTER + m1
    n = <uint32_t>(f2b(shifted) & 0xFFFFFFFFULL)
    rounded = shifted - P_SHIFTER
    z = (m1 - rounded) + m2
    if ((f2b(rounded) >> 48) & 0x7FFF) > 16529:
        return pow_out_of_range(xb, y, yb, sign)
    half = (n >> 8) >> 1
    rest = (n >> 8) - half
    factor = b2f((<uint64_t>(((rest + 1023u) << 20) | sign)) << 32)
    return pow_scaled(z, n & 255, half << 20, factor)


# noinspection PyShadowingBuiltins
def pow(double x, double y):
    """``x ** y`` as the venue's engine computes it (compiled :func:`pine_math.pow`)."""
    return c_pow(x, y)


# --- fdlibm ------------------------------------------------------------------------

cdef inline int32_t hi_word(double x) noexcept nogil:
    return <int32_t>(<uint32_t>(f2b(x) >> 32))


cdef inline uint32_t lo_word(double x) noexcept nogil:
    return <uint32_t>(f2b(x) & 0xFFFFFFFFULL)


cdef inline double set_hi(double x, int64_t hi) noexcept nogil:
    return b2f((<uint64_t>(<uint32_t>hi) << 32) | (f2b(x) & 0xFFFFFFFFULL))


cdef inline double set_lo(double x, int64_t lo) noexcept nogil:
    return b2f((f2b(x) & 0xFFFFFFFF00000000ULL) | <uint64_t>(<uint32_t>lo))


cdef double S1 = _fd._S1
cdef double S2 = _fd._S2
cdef double S3 = _fd._S3
cdef double S4 = _fd._S4
cdef double S5 = _fd._S5
cdef double S6 = _fd._S6
cdef double C1 = _fd._C1
cdef double C2 = _fd._C2
cdef double C3 = _fd._C3
cdef double C4 = _fd._C4
cdef double C5 = _fd._C5
cdef double C6 = _fd._C6


cdef double kernel_sin(double x, double y, int iy) noexcept nogil:
    cdef int32_t ix = hi_word(x) & 0x7fffffff
    cdef double z, v, r
    if ix < 0x3e400000:
        return x
    z = x * x
    v = z * x
    r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)))
    if iy == 0:
        return x + v * (S1 + z * r)
    return x - ((z * (0.5 * y - v * r) - y) - v * S1)


cdef double kernel_cos(double x, double y) noexcept nogil:
    cdef int32_t ix = hi_word(x) & 0x7fffffff
    cdef double z, r, qx, hz, a
    if ix < 0x3e400000:
        return 1.0
    z = x * x
    r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))))
    if ix < 0x3FD33333:
        return 1.0 - (0.5 * z - (z * r - x * y))
    if ix > 0x3fe90000:
        qx = 0.28125
    else:
        qx = b2f(<uint64_t>(<uint32_t>(ix - 0x00200000)) << 32)
    hz = 0.5 * z - qx
    a = 1.0 - qx
    return a - (hz - (z * r - x * y))


cdef double INVPIO2 = _fd._INVPIO2
cdef double PIO2_1 = _fd._PIO2_1
cdef double PIO2_1T = _fd._PIO2_1T
cdef double PIO2_2 = _fd._PIO2_2
cdef double PIO2_2T = _fd._PIO2_2T
cdef double PIO2_3 = _fd._PIO2_3
cdef double PIO2_3T = _fd._PIO2_3T
cdef int32_t NPIO2_HW[32]
for _i in range(32):
    NPIO2_HW[_i] = _fd._NPIO2_HW[_i]


cdef int rem_pio2_medium(double x, double *y0, double *y1) noexcept nogil:
    """__ieee754_rem_pio2 for |x| <= 2^19*(pi/2); larger arguments stay in Python."""
    cdef int32_t hx = hi_word(x)
    cdef int32_t ix = hx & 0x7fffffff
    cdef double z, t, fn, r, w, t2, t3
    cdef int64_t n
    cdef int32_t j, i
    if ix <= 0x3fe921fb:
        y0[0] = x
        y1[0] = 0.0
        return 0
    if ix < 0x4002d97c:
        if hx > 0:
            z = x - PIO2_1
            if ix != 0x3ff921fb:
                y0[0] = z - PIO2_1T
                y1[0] = (z - y0[0]) - PIO2_1T
            else:
                z -= PIO2_2
                y0[0] = z - PIO2_2T
                y1[0] = (z - y0[0]) - PIO2_2T
            return 1
        z = x + PIO2_1
        if ix != 0x3ff921fb:
            y0[0] = z + PIO2_1T
            y1[0] = (z - y0[0]) + PIO2_1T
        else:
            z += PIO2_2
            y0[0] = z + PIO2_2T
            y1[0] = (z - y0[0]) + PIO2_2T
        return -1
    t = fabs(x)
    n = <int64_t>(t * INVPIO2 + 0.5)
    fn = <double>n
    r = t - fn * PIO2_1
    w = fn * PIO2_1T
    if n < 32 and ix != NPIO2_HW[n - 1]:
        y0[0] = r - w
    else:
        j = ix >> 20
        y0[0] = r - w
        i = j - ((hi_word(y0[0]) >> 20) & 0x7ff)
        if i > 16:
            t2 = r
            w = fn * PIO2_2
            r = t2 - w
            w = fn * PIO2_2T - ((t2 - r) - w)
            y0[0] = r - w
            i = j - ((hi_word(y0[0]) >> 20) & 0x7ff)
            if i > 49:
                t3 = r
                w = fn * PIO2_3
                r = t3 - w
                w = fn * PIO2_3T - ((t3 - r) - w)
                y0[0] = r - w
    y1[0] = (r - y0[0]) - w
    if hx < 0:
        y0[0] = -y0[0]
        y1[0] = -y1[0]
        return <int>(-n)
    return <int>n


def f_sin(double x):
    """Compiled :func:`fdlibm.sin`."""
    cdef int32_t ix = hi_word(x) & 0x7fffffff
    cdef double y0, y1
    cdef int n
    if ix <= 0x3fe921fb:
        return kernel_sin(x, 0.0, 0)
    if ix >= 0x7ff00000:
        return x - x
    if ix > 0x413921fb:
        return _py_fd_sin(x)
    n = rem_pio2_medium(x, &y0, &y1) & 3
    if n == 0:
        return kernel_sin(y0, y1, 1)
    if n == 1:
        return kernel_cos(y0, y1)
    if n == 2:
        return -kernel_sin(y0, y1, 1)
    return -kernel_cos(y0, y1)


def f_cos(double x):
    """Compiled :func:`fdlibm.cos`."""
    cdef int32_t ix = hi_word(x) & 0x7fffffff
    cdef double y0, y1
    cdef int n
    if ix <= 0x3fe921fb:
        return kernel_cos(x, 0.0)
    if ix >= 0x7ff00000:
        return x - x
    if ix > 0x413921fb:
        return _py_fd_cos(x)
    n = rem_pio2_medium(x, &y0, &y1) & 3
    if n == 0:
        return kernel_cos(y0, y1)
    if n == 1:
        return -kernel_sin(y0, y1, 1)
    if n == 2:
        return -kernel_cos(y0, y1)
    return kernel_sin(y0, y1, 1)


cdef double HUGE = _fd._HUGE
cdef double TWOM1000 = _fd._TWOM1000
cdef double O_THRESHOLD = _fd._O_THRESHOLD
cdef double U_THRESHOLD = _fd._U_THRESHOLD
cdef double LN2HI[2]
cdef double LN2LO[2]
cdef double HALF[2]
for _i in range(2):
    LN2HI[_i] = _fd._LN2HI[_i]
    LN2LO[_i] = _fd._LN2LO[_i]
    HALF[_i] = _fd._HALF[_i]
cdef double INVLN2 = _fd._INVLN2
cdef double EP1 = _fd._EP1
cdef double EP2 = _fd._EP2
cdef double EP3 = _fd._EP3
cdef double EP4 = _fd._EP4
cdef double EP5 = _fd._EP5


def f_exp(double x):
    """Compiled :func:`fdlibm.exp`."""
    cdef int32_t hx = hi_word(x)
    cdef int xsb = (hx >> 31) & 1
    cdef double hi = 0.0
    cdef double lo = 0.0
    cdef double t, c, y
    cdef int64_t k
    hx &= 0x7fffffff
    if hx >= 0x40862E42:
        if hx >= 0x7ff00000:
            if ((hx & 0xfffff) | lo_word(x)) != 0:
                return x + x
            return x if xsb == 0 else 0.0
        if x > O_THRESHOLD:
            return HUGE * HUGE
        if x < U_THRESHOLD:
            return TWOM1000 * TWOM1000

    if hx > 0x3fd62e42:
        if hx < 0x3FF0A2B2:
            hi = x - LN2HI[xsb]
            lo = LN2LO[xsb]
            k = 1 - xsb - xsb
        else:
            k = <int64_t>(INVLN2 * x + HALF[xsb])
            t = <double>k
            hi = x - t * LN2HI[0]
            lo = t * LN2LO[0]
        x = hi - lo
    elif hx < 0x3e300000:
        return 1.0 + x
    else:
        k = 0

    t = x * x
    c = x - t * (EP1 + t * (EP2 + t * (EP3 + t * (EP4 + t * EP5))))
    if k == 0:
        return 1.0 - ((x * c) / (c - 2.0) - x)
    y = 1.0 - ((lo - (x * c) / (2.0 - c)) - hi)
    if k >= -1021:
        return set_hi(y, <int64_t>hi_word(y) + k * 1048576)
    y = set_hi(y, <int64_t>hi_word(y) + (k + 1000) * 1048576)
    return y * TWOM1000


cdef double PIO2_HI = _fd._PIO2_HI
cdef double PIO2_LO = _fd._PIO2_LO
cdef double PIO4_HI = _fd._PIO4_HI
cdef double PI_ACOS = _fd._PI_ACOS
cdef double PS0 = _fd._PS0
cdef double PS1 = _fd._PS1
cdef double PS2 = _fd._PS2
cdef double PS3 = _fd._PS3
cdef double PS4 = _fd._PS4
cdef double PS5 = _fd._PS5
cdef double QS1 = _fd._QS1
cdef double QS2 = _fd._QS2
cdef double QS3 = _fd._QS3
cdef double QS4 = _fd._QS4
cdef double NAN_ = b2f(0x7ff8000000000000ULL)


def f_asin(double x):
    """Compiled :func:`fdlibm.asin`."""
    cdef int32_t hx = hi_word(x)
    cdef int32_t ix = hx & 0x7fffffff
    cdef double t, p, q, w, s, c, r
    if ix >= 0x3ff00000:
        if ((ix - 0x3ff00000) | <int32_t>lo_word(x)) == 0:
            return x * PIO2_HI + x * PIO2_LO
        return NAN_
    if ix < 0x3fe00000:
        if ix < 0x3e400000:
            return x
        t = x * x
        p = t * (PS0 + t * (PS1 + t * (PS2 + t * (PS3 + t * (PS4 + t * PS5)))))
        q = 1.0 + t * (QS1 + t * (QS2 + t * (QS3 + t * QS4)))
        w = p / q
        return x + x * w
    w = 1.0 - fabs(x)
    t = w * 0.5
    p = t * (PS0 + t * (PS1 + t * (PS2 + t * (PS3 + t * (PS4 + t * PS5)))))
    q = 1.0 + t * (QS1 + t * (QS2 + t * (QS3 + t * QS4)))
    s = sqrt(t)
    if ix >= 0x3FEF3333:
        w = p / q
        t = PIO2_HI - (2.0 * (s + s * w) - PIO2_LO)
    else:
        w = set_lo(s, 0)
        c = (t - w * w) / (s + w)
        r = p / q
        p = 2.0 * s * r - (PIO2_LO - 2.0 * c)
        q = PIO4_HI - 2.0 * w
        t = PIO4_HI - (p - q)
    return t if hx > 0 else -t


def f_acos(double x):
    """Compiled :func:`fdlibm.acos`."""
    cdef int32_t hx = hi_word(x)
    cdef int32_t ix = hx & 0x7fffffff
    cdef double z, p, q, r, s, w, df, c
    if ix >= 0x3ff00000:
        if ((ix - 0x3ff00000) | <int32_t>lo_word(x)) == 0:
            if hx > 0:
                return 0.0
            return PI_ACOS + 2.0 * PIO2_LO
        return NAN_
    if ix < 0x3fe00000:
        if ix <= 0x3c600000:
            return PIO2_HI + PIO2_LO
        z = x * x
        p = z * (PS0 + z * (PS1 + z * (PS2 + z * (PS3 + z * (PS4 + z * PS5)))))
        q = 1.0 + z * (QS1 + z * (QS2 + z * (QS3 + z * QS4)))
        r = p / q
        return PIO2_HI - (x - (PIO2_LO - x * r))
    if hx < 0:
        z = (1.0 + x) * 0.5
        p = z * (PS0 + z * (PS1 + z * (PS2 + z * (PS3 + z * (PS4 + z * PS5)))))
        q = 1.0 + z * (QS1 + z * (QS2 + z * (QS3 + z * QS4)))
        s = sqrt(z)
        r = p / q
        w = r * s - PIO2_LO
        return PI_ACOS - 2.0 * (s + w)
    z = (1.0 - x) * 0.5
    s = sqrt(z)
    df = set_lo(s, 0)
    c = (z - df * df) / (s + df)
    p = z * (PS0 + z * (PS1 + z * (PS2 + z * (PS3 + z * (PS4 + z * PS5)))))
    q = 1.0 + z * (QS1 + z * (QS2 + z * (QS3 + z * QS4)))
    r = p / q
    w = r * s + c
    return 2.0 * (df + w)


cdef double ATAN_HI[4]
cdef double ATAN_LO[4]
for _i in range(4):
    ATAN_HI[_i] = _fd._ATAN_HI[_i]
    ATAN_LO[_i] = _fd._ATAN_LO[_i]
cdef double AT0 = _fd._AT0
cdef double AT1 = _fd._AT1
cdef double AT2 = _fd._AT2
cdef double AT3 = _fd._AT3
cdef double AT4 = _fd._AT4
cdef double AT5 = _fd._AT5
cdef double AT6 = _fd._AT6
cdef double AT7 = _fd._AT7
cdef double AT8 = _fd._AT8
cdef double AT9 = _fd._AT9
cdef double AT10 = _fd._AT10


def f_atan(double x):
    """Compiled :func:`fdlibm.atan`."""
    cdef int32_t hx = hi_word(x)
    cdef int32_t ix = hx & 0x7fffffff
    cdef int id_
    cdef double z, w, s1, s2
    if ix >= 0x44100000:
        if ix > 0x7ff00000 or (ix == 0x7ff00000 and lo_word(x) != 0):
            return x + x
        if hx > 0:
            return ATAN_HI[3] + ATAN_LO[3]
        return -ATAN_HI[3] - ATAN_LO[3]
    if ix < 0x3fdc0000:
        if ix < 0x3e200000:
            return x
        id_ = -1
    else:
        x = fabs(x)
        if ix < 0x3ff30000:
            if ix < 0x3fe60000:
                id_ = 0
                x = (2.0 * x - 1.0) / (2.0 + x)
            else:
                id_ = 1
                x = (x - 1.0) / (x + 1.0)
        elif ix < 0x40038000:
            id_ = 2
            x = (x - 1.5) / (1.0 + 1.5 * x)
        else:
            id_ = 3
            x = -1.0 / x

    z = x * x
    w = z * z
    s1 = z * (AT0 + w * (AT2 + w * (AT4 + w * (AT6 + w * (AT8 + w * AT10)))))
    s2 = w * (AT1 + w * (AT3 + w * (AT5 + w * (AT7 + w * AT9))))
    if id_ < 0:
        return x - x * (s1 + s2)
    z = ATAN_HI[id_] - ((x * (s1 + s2) - ATAN_LO[id_]) - x)
    return -z if hx < 0 else z


def _install():
    """Rebind the public names of both Python modules to their compiled twins.

    Runs at the end of this module's import, whichever of the two Python modules pulled
    it in: by then both are fully defined (a circular import hands this module the one
    still finishing, and its functions all exist before its import line), and their
    ``PYTHON_IMPLEMENTATIONS`` keep the originals reachable.
    """
    _pm.cos = cos
    _pm.sin = sin
    _pm.exp = exp
    _pm.log = log
    _pm.log10 = log10
    _pm.pow = pow
    _fd.sin = f_sin
    _fd.cos = f_cos
    _fd.exp = f_exp
    _fd.asin = f_asin
    _fd.acos = f_acos
    _fd.atan = f_atan


_install()
