"""Runtime-exact transcendental functions for Pine-compatible execution.

Implements the cos/sin/exp/log/log10/pow algorithms that HotSpot JVMs before
JDK 19 use for the ``Math`` functions of the same names on x86-64 (the Intel
LIBM table-driven methods), so per-bar runtime results are bit-identical to the
venue engine. None of them is correctly rounded, and the logarithms and ``pow``
are where that shows: a correctly rounded logarithm disagrees with the venue
whenever the argument comes within about 1.2% of 1, and the platform's ``pow``
misses the venue's last bit on roughly one argument in 700.

This is an independent implementation written from the published algorithm
description: argument reduction against a fixed-point table of 2/pi (huge
arguments) or N = round(x * 32/pi) (normal range), a 64-entry split-precision
table of cos/sin values at multiples of pi/32, and a shared polynomial
reconstruction; ``log`` and ``log10`` reduce instead against a 129-bin table of
-log(B) for the bins an approximate hardware reciprocal selects and finish with
a short series, and ``pow`` takes ``log2(x)`` the same way from a 513-bin table,
then ``2**(y * log2(x))`` from a 256-entry one. The reciprocal is the venue
processor's own, measured bin by bin (see ``_RCP``). The numeric tables are the
published algorithm constants (mathematical values, reproduced as data).
Correctness is established by
bit-comparison against venue-engine oracles over hundreds of thousands of
arguments, including huge, tiny, denormal-result and overflow/underflow
boundary regions.
"""
import math as _math
import os as _os
from struct import pack as _pack, unpack as _unpack

__all__ = ['cos', 'sin', 'exp', 'log', 'log10', 'pow']

_M32 = 0xFFFFFFFF
_M64 = 0xFFFFFFFFFFFFFFFF


def _f2b(f: float) -> int:
    return _unpack('<Q', _pack('<d', f))[0]


def _b2f(b: int) -> float:
    return _unpack('<d', _pack('<Q', b))[0]


# --- table data (published algorithm constants) ------------------------------
# 64 entries of 32 bytes for angles k*pi/32: C_hl, S_hi, S_lo, sigma.
_CTABLE = bytes.fromhex(
    '000000000000000000000000000000000000000000000000000000000000f03f316d6d17'
    '2eb973bf2cb429bca617b93f000000e018273ebc000000000000f03ffb69140106ad93bf'
    '0ba6693cb8f8c83f000000c0196d62bc000000000000f03f5a229d93ea0ba6bf069fd52e'
    '0694d23f000000a08dd275bc000000000000f03fcf956b86a17cb3bf63a9aea6e27dd83f'
    '000000e0ed2c67bc000000000000f03f7912fa73683abebf3bf606385d2bde3f00000020'
    '890d5e3c000000000000f03f7479c55b6792c5bfc868ae393bc7e13f00000020dd258b3c'
    '000000000000f03ffda2ab53fe0dcdbfd61d0925f34ce43f000000206a07683c00000000'
    '0000f03f32effc997982ca3fcd3b7f669ea0e63f0000002034dd8bbc000000000000e03f'
    '58772494cc33c13f4117156b80bce83f00000020e1c582bc000000000000e03f878ce69a'
    'b373ac3fa3a10e29669bea3f000000e030f6393c000000000000e03f4e9c907f2c4a9dbf'
    'b1bd80f1b238ec3f00000080b1e076bc000000000000e03f755a45657508bebf468d32cf'
    '6b90ed3f00000020e657743c000000000000e03f2df8ac7631a0a43fda2dc656419fee3f'
    '000000e0b160873c000000000000d03fd567590e1f1dacbfb05cf7cf9762ef3f00000020'
    '1762753c000000000000d03f502f590f65a19bbf2625d1a38dd8ef3f00000040f67d88bc'
    '000000000000c03f0000000000000000000000000000f03f000000000000000000000000'
    '00000000502f590f65a19b3f2625d1a38dd8ef3f00000040f67d88bc000000000000c0bf'
    'd567590e1f1dac3fb05cf7cf9762ef3f000000201762753c000000000000d0bf2df8ac76'
    '31a0a4bfda2dc656419fee3f000000e0b160873c000000000000d0bf755a45657508be3f'
    '468d32cf6b90ed3f00000020e657743c000000000000e0bf4e9c907f2c4a9d3fb1bd80f1'
    'b238ec3f00000080b1e076bc000000000000e0bf878ce69ab373acbfa3a10e29669bea3f'
    '000000e030f6393c000000000000e0bf58772494cc33c1bf4117156b80bce83f00000020'
    'e1c582bc000000000000e0bf32effc997982cabfcd3b7f669ea0e63f0000002034dd8bbc'
    '000000000000e0bffda2ab53fe0dcd3fd61d0925f34ce43f000000206a07683c00000000'
    '0000f0bf7479c55b6792c53fc868ae393bc7e13f00000020dd258b3c000000000000f0bf'
    '7912fa73683abe3f3bf606385d2bde3f00000020890d5e3c000000000000f0bfcf956b86'
    'a17cb33f63a9aea6e27dd83f000000e0ed2c67bc000000000000f0bf5a229d93ea0ba63f'
    '069fd52e0694d23f000000a08dd275bc000000000000f0bffb69140106ad933f0ba6693c'
    'b8f8c83f000000c0196d62bc000000000000f0bf316d6d172eb9733f2cb429bca617b93f'
    '000000e018273ebc000000000000f0bf0000000000000000000000000000000000000000'
    '00000000000000000000f0bf316d6d172eb9733f2cb429bca617b9bf000000e018273e3c'
    '000000000000f0bffb69140106ad933f0ba6693cb8f8c8bf000000c0196d623c00000000'
    '0000f0bf5a229d93ea0ba63f069fd52e0694d2bf000000a08dd2753c000000000000f0bf'
    'cf956b86a17cb33f63a9aea6e27dd8bf000000e0ed2c673c000000000000f0bf7912fa73'
    '683abe3f3bf606385d2bdebf00000020890d5ebc000000000000f0bf7479c55b6792c53f'
    'c868ae393bc7e1bf00000020dd258bbc000000000000f0bffda2ab53fe0dcd3fd61d0925'
    'f34ce4bf000000206a0768bc000000000000f0bf32effc997982cabfcd3b7f669ea0e6bf'
    '0000002034dd8b3c000000000000e0bf58772494cc33c1bf4117156b80bce8bf00000020'
    'e1c5823c000000000000e0bf878ce69ab373acbfa3a10e29669beabf000000e030f639bc'
    '000000000000e0bf4e9c907f2c4a9d3fb1bd80f1b238ecbf00000080b1e0763c00000000'
    '0000e0bf755a45657508be3f468d32cf6b90edbf00000020e65774bc000000000000e0bf'
    '2df8ac7631a0a4bfda2dc656419feebf000000e0b16087bc000000000000d0bfd567590e'
    '1f1dac3fb05cf7cf9762efbf00000020176275bc000000000000d0bf502f590f65a19b3f'
    '2625d1a38dd8efbf00000040f67d883c000000000000c0bf000000000000000000000000'
    '0000f0bf00000000000000000000000000000000502f590f65a19bbf2625d1a38dd8efbf'
    '00000040f67d883c000000000000c03fd567590e1f1dacbfb05cf7cf9762efbf00000020'
    '176275bc000000000000d03f2df8ac7631a0a43fda2dc656419feebf000000e0b16087bc'
    '000000000000d03f755a45657508bebf468d32cf6b90edbf00000020e65774bc00000000'
    '0000e03f4e9c907f2c4a9dbfb1bd80f1b238ecbf00000080b1e0763c000000000000e03f'
    '878ce69ab373ac3fa3a10e29669beabf000000e030f639bc000000000000e03f58772494'
    'cc33c13f4117156b80bce8bf00000020e1c5823c000000000000e03f32effc997982ca3f'
    'cd3b7f669ea0e6bf0000002034dd8b3c000000000000e03ffda2ab53fe0dcdbfd61d0925'
    'f34ce4bf000000206a0768bc000000000000f03f7479c55b6792c5bfc868ae393bc7e1bf'
    '00000020dd258bbc000000000000f03f7912fa73683abebf3bf606385d2bdebf00000020'
    '890d5ebc000000000000f03fcf956b86a17cb3bf63a9aea6e27dd8bf000000e0ed2c673c'
    '000000000000f03f5a229d93ea0ba6bf069fd52e0694d2bf000000a08dd2753c00000000'
    '0000f03ffb69140106ad93bf0ba6693cb8f8c8bf000000c0196d623c000000000000f03f'
    '316d6d172eb973bf2cb429bca617b9bf000000e018273e3c000000000000f03f'
)

# 2/pi in fixed point: consecutive 32-bit windows selected by exponent.
_PI_INV_TABLE = bytes.fromhex(
    '00000000000000006e83f9a22915444ed15727fcc0dd34f5999562db4190433cab6351fe'
    '61c5bbde3a6e24b7e0d24d42ea2e49061c92d1091ceb1dfe3ea729b1f53582e88444bb2e'
    '26709ce9417e5fb439d69139f43953838b5f849c3b28f9bdff97f81f0f9805de8b112fef'
    '1f6d0a5acf7e366db709cb27663f464f2dea5f9ec7ba27757bf1e5ebf739073dea92528a'
    'b15ffb6b085d8d1f46300356ab6b7bfc21bccff0'
)

# exp: 64 entries of 16 bytes: 2^(j/64) split as T_lo double + T_hi mantissa.
_EXP_TBL = bytes.fromhex(
    '000000000000000000000000000000004d75030ebf7bad3c6080773e9a2c000013f66735'
    '52d28c3c748515d3b059000061c8e6614ef7603cc89b7518458700006c7b835da69a973c'
    '0f89f96c58b50000d19c2f703dbe3e3ca2d1d332ece30000d8bc631e6e51a33c505b12d0'
    '011301007b38f02654c5a43cdf2da9ae9a420100b63f52625351a93c7a517d3cb8720100'
    'bf53133f8c898b3c75cb6feb5ba301005f2f3a3ef7ec9a3caab9683187d401008dc3a644'
    '416f8a3cd68c62883b06020094a8a8e3fd8e963c3862756e7a380200f2e71f982b47803c'
    'dd7ce265456b020031ab096de1f7823ce1de1ff59d9e0200b30a0c7282378b3c0b03e4a6'
    '85d20200b6abb04d754d833c15b7310afe0603004af8d35d39dd8f3cff1664b2083c0300'
    '297d18cc8c2fa13ccaa93a37a77103008b5e8b7329d2a73cf69fe534dba703006d4c2aa7'
    '489f853c2234124ca6de030005929d2546b8a83c292ef7210a16040012acc260ed63433c'
    '2d896160084e04007903a1dae1cc6e3cd03cc1b5a2860400b0af7abbce90763c272a36d5'
    'dabf0400092a289bcc83a03ca62c9d76b2f9040007e7aac1b009a53c814f9d562b340500'
    '8ed7fd180535933cda27b536476f050009541ce2e163903c295448dd07ab050035c0642b'
    'e632943c4821ad156fe705000a8cf0998412a03c84553ab07e24060006dc730087f0993c'
    '24225582386206007155a00d4d8d993ccc3b7f669ea006008647ce86b92ba53c2e1a653c'
    'b2df0600ab0d6f209220a33c735fece8751f0700a6a7178e2261a03cc8674256eb5f0700'
    '869f1e46ac44a23c8601eb7314a10700556cd6abe1eb653c624ecf36f3e20700d067ffbb'
    '9ffe963c12ce4c9989250800df01c814141f953cec92449bd9680800b4eaf0c12fb78d3c'
    'dba02a42e5ac0800445ff35983f67b3c36771599aef108003c28069cba60a33ce4c5cdb0'
    '37370900aa62f920d1e8953c4f4ede9f827d090027ce912bfcaf713c90f0a38291c40900'
    'bd2e9a58346d9b3c64e55d7b660c0a008098b89a7c27953c5c253eb203550a00b35a736e'
    '8469843cbffd79556b9e0a008733cb92771a8c3cadd35a999fe80a00961d2ddc6624a23c'
    'fa154fb8a2330b00ae0595b12e11a13c465efbf2767f0b00cddd5f0ad7ff743cd2c14b90'
    '1ecc0b00b30caf30ae6e733c9c5285dd9b190c00ac5909d18fe0843c4bd1572ef1670c00'
    '6819926c2c6b673c6990efdc20b70c00b399df360970933c7b89074a2d070d00a7073da6'
    '85a3743c87a4fbdc18580d00ac92c1d5505a8e3c8532db03e6a90d0092974a1c73bb983c'
    '5e9b7b3397fc0d00d3883a6004b6743cf63f8be72e500e0026490992276f913cd990a4a2'
    'afa40e000820aa41bcc38e3c275a61ee1bfa0e00ee85d131a9648a3c40456e5b76500f00'
    '9dcd914d3b89773cd8909e81c1a70f00'
)


def _u32(table: bytes, off: int) -> int:
    return _unpack('<I', table[off:off + 4])[0]


def _u64(table: bytes, off: int) -> int:
    return _unpack('<Q', table[off:off + 8])[0]


# --- cos/sin scalar constants ------------------------------------------------

_PI32INV = _b2f(0x40245f306dc9c883)  # 32/pi
_P1 = _b2f(0x3fb921fb54400000)  # pi/32 head
_P2 = _b2f(0x3d90b4611a600000)  # pi/32 middle
_P3 = _b2f(0x3b63198a2e037073)  # pi/32 tail
_SC1_LO, _SC1_HI = _b2f(0xbfc5555555555555), _b2f(0xbfe0000000000000)
_SC2_LO, _SC2_HI = _b2f(0x3f81111111111111), _b2f(0x3fa5555555555555)
_SC3_LO, _SC3_HI = _b2f(0xbf2a01a01a01a01a), _b2f(0xbf56c16c16c16c17)
_SC4_LO, _SC4_HI = _b2f(0x3ec71de3a556c734), _b2f(0x3efa01a01a01a01a)
_PI_4_HEAD = _b2f(0x3fe921fb40000000)  # pi/4 split head
_PI_4_TAIL = _b2f(0x3e64442d18469899)  # pi/4 split tail

_CT0 = [_u64(_CTABLE, j * 32) for j in range(64)]
_CT8 = [_b2f(_u64(_CTABLE, j * 32 + 8)) for j in range(64)]
_CT16 = [_b2f(_u64(_CTABLE, j * 32 + 16)) for j in range(64)]
_CT24 = [_b2f(_u64(_CTABLE, j * 32 + 24)) for j in range(64)]
_CT0F = [_b2f(v) for v in _CT0]


def _sincos_poly(x: float, n: int, j: int, corr: float) -> float:
    """Shared table-polynomial reconstruction for angle x ~ n*(pi/32) + r.

    ``corr`` carries the low half of a double-double reduced argument (zero on
    the fast path).  Operation order is fixed: every add/mul below is a single
    IEEE double rounding and the sequence must not be reassociated.
    """
    nf = float(n)
    p1n = _P1 * nf
    p2n = _P2 * nf
    rr = x - p1n
    r = rr - p2n
    c = (rr - r) - p2n
    m = (nf * _P3 - c) - corr
    r2 = r * r
    r4 = r2 * r2
    c0 = _CT0F[j]
    c8 = _CT8[j]
    q = c0 + _CT24[j]
    c8r_q = c8 * r - q
    c24r = _CT24[j] * r
    poly_lo = ((_SC2_LO * r2 + _SC1_LO) + (_SC4_LO * rr * r + _SC3_LO) * r4) * (q * r * r2)
    poly_hi = ((_SC2_HI * r2 + _SC1_HI) + (_SC4_HI * rr * r + _SC3_HI) * r4) * (c8 * r2)
    t4 = r * c0
    x3 = c8 + c24r
    m_corr = m * c8r_q + _CT16[j]
    s = t4 + x3
    lo5 = c8 - x3
    x3s = x3 - s
    lo0 = c24r + lo5
    x3t = x3s + t4
    total = lo0 + m_corr
    total = total + x3t
    total = total + poly_lo
    total = total + poly_hi
    return total + s


def _round_n(x: float) -> int:
    """N = trunc(x*32/pi +- 0.5): round-half-away in the fast-path range."""
    return int(x * _PI32INV + _math.copysign(0.5, x))


def _reduce_huge(x_bits: int, quad_add: int, cvt64: bool) -> float:
    """cos/sin for |x| >= 90112: fixed-point reduction by 2/pi.

    The 53-bit significand is multiplied by a 224-bit window of the 2/pi
    table selected by the exponent; the top bits of the (partially truncated)
    product hold the quadrant and the fraction of the reduced angle.  The
    fraction is normalized, converted to a double-double and multiplied back
    by pi/4, then fed to the shared table polynomial with the quadrant folded
    into the table index.
    """
    exp16 = (x_bits >> 48) & 0x7FF0
    if exp16 == 0x7FF0:  # Inf or NaN
        return _b2f(x_bits) * -0.0

    off = ((exp16 - 16224) >> 7) & 0xFFFC
    m_lo = x_bits & _M32
    m_hi = (((x_bits >> 21) & 0x7FFFFFFF | 0x80000000) >> 11) & _M32
    w = [_u32(_PI_INV_TABLE, off + 4 * k) for k in range(7)]

    # partially truncated 53 x 224-bit product (terms m_hi*w0 and m_lo*w6
    # fall outside the kept window); bits >= 32 of the sum are exact
    s = m_lo * w[0] << 192
    for k in range(1, 6):
        s += ((m_hi << 32) + m_lo) * w[k] << (192 - 32 * k)
    s += m_hi * w[6] << 32
    prod = s >> 32
    low = prod & _M64  # bits [0:64)
    mid = (prod >> 64) & _M64  # bits [64:128)
    up32 = (prod >> 128) & _M32  # bits [128:160)
    top = (prod >> 160) & _M64  # bits [160:224)

    sign16 = 32768 if x_bits >> 63 else 0
    expo = ((x_bits >> 52) & 2047) - 1023
    point = off * 8 + 19 - expo  # binary-point offset
    e_ctr = point + 32
    sign_flip = 0

    # noinspection PyShadowingNames
    def _complement(borrow_hi, lo, md, tp):
        """(borrow_hi - fraction): three-limb negate with borrow chain."""
        lo2 = (0 - lo) & _M64
        cf = 1 if lo != 0 else 0
        t = md + cf
        md2 = (0 - t) & _M64
        cf = 1 if t != 0 else 0
        tp2 = (borrow_hi - tp - cf) & _M64
        return lo2, md2, tp2

    if point >= 1:  # binary point inside the top word
        sh = (29 - point) & 31
        t32 = ((top & _M32) << sh) & _M32
        quad_acc = t32
        frac29 = t32 & 0x1FFFFFFF
        f = (frac29 >> sh) & _M32
        top = ((f << 32) | up32) & _M64
        if frac29 & 0x10000000:  # fraction >= 1/2: complement, bump
            low, mid, top = _complement(((0x20000000 >> sh) & _M32) << 32,
                                        low, mid, top)
            sign_flip = 32768
            quad_acc = (quad_acc + 0x20000000) & _M32
        quad_base = quad_acc >> 29
    else:  # point below the top word
        sh = (-point) & 63
        full = (((top << 32) | up32) << sh) & _M64
        quad_acc = full
        top = (((full & _M32) >> (sh & 31)) & _M32)
        if full & 0x80000000:
            low, mid, top = _complement((0x100000000 >> sh) & _M64,
                                        low, mid, top)
            sign_flip = 32768
            quad_base = ((((quad_acc >> 3) & _M32) + 0x20000000) & _M32) >> 29
        else:
            quad_base = ((quad_acc >> 3) & _M32) >> 29

    # normalize: shift limbs up until the leading bit of `top` is bit 29
    zero_frac = False
    while True:
        if top == 0:
            e_ctr = (e_ctr + 64) & _M32
            top, mid, low = mid, low, 0
            if top != 0:
                continue
            e_ctr = (e_ctr + 64) & _M32
            top, mid = mid, low
            if top != 0:
                continue
            zero_frac = True
            break
        d = 29 - (top.bit_length() - 1)
        if d > 0:
            top = ((top << d) | (mid >> (64 - d))) & _M64
            mid = ((mid << d) | (low >> (64 - d))) & _M64
            e_ctr = (e_ctr + d) & _M32
        elif d != 0:
            mid = ((mid >> (-d)) | (top << (64 + d))) & _M64
            top >>= -d
            e_ctr = (e_ctr - (-d)) & _M32
        break

    if zero_frac:
        red = 0.0
        corr = 0.0
    else:
        e_bits = (((16368 - ((e_ctr << 4) & _M32)) & _M32) | sign16) ^ sign_flip
        scale1 = _b2f((e_bits & 0xFFFF) << 48)
        scale2 = _b2f(((e_bits - 1008) & 0xFFFF) << 48)
        f1 = float(top) * scale1
        f2 = float(mid >> 1) * scale2
        head = f1 * _PI_4_HEAD
        t = f1 + f2
        tail = _PI_4_TAIL * t + f2 * _PI_4_HEAD
        red = head + tail
        corr = tail + (head - red)

    s_ext = -1 if sign16 else 0
    quad = ((quad_base + s_ext) ^ s_ext) & _M32

    t = red * _PI32INV + _math.copysign(0.5, red)
    n = int(t)
    if not cvt64:
        n = ((n + 0x80000000) & _M32) - 0x80000000
    j = (n + quad_add + 8 * quad) & 63
    return _sincos_poly(red, n, j, corr)


def _sincos_poly_corr(x: float, n: int, j: int, corr: float) -> float:
    return _sincos_poly(x, n, j, corr)


def cos(x: float) -> float:
    """Bit-exact venue-runtime cos."""
    x_bits = _f2b(x)
    band = (((x_bits >> 32) & 2147418112) - 808452096) & _M32
    if band > 281346048:
        if band - 281346048 < 0x80000000:  # |x| >= 90112 (signed positive)
            return _reduce_huge(x_bits, 1865232, True)
        return 1.0 - abs(x)  # |x| < 2^-252
    n = _round_n(x)
    return _sincos_poly(x, n, (n + 16) & 63, 0.0)


def sin(x: float) -> float:
    """Bit-exact venue-runtime sin."""
    x_bits = _f2b(x)
    band = (((x_bits >> 32) & 2147418112) - 808452096) & _M32
    if band > 281346048:
        if band - 281346048 < 0x80000000:  # |x| >= 90112
            return _reduce_huge(x_bits, 1865216, False)
        if (band >> 20) == 3325:
            return x * _b2f(0x3fefffffffffffff)
        return x
    n = _round_n(x)
    return _sincos_poly(x, n, n & 63, 0.0)


# --- exp ---------------------------------------------------------------------

_E_LOG2_64 = _b2f(0x40571547652b82fe)  # 64/ln2
_E_LN2_64_HEAD = _b2f(0x3f862e42fefa0000)  # ln2/64 head
_E_LN2_64_TAIL = _b2f(0x3d1cf79abc9e3b3a)  # ln2/64 tail
_E_HALF = _b2f(0x3fdffffffffffffe)  # 0.5 - 1/4 ulp
_E_P3_LO, _E_P3_HI = _b2f(0x3f56c15ce3289860), _b2f(0x3fa55555555b9e25)
_E_P5_LO, _E_P5_HI = _b2f(0x3f811115c090cf0f), _b2f(0x3fc5555555548ba1)
_E_SHIFTER = _b2f(0x4338000000000000)  # 1.5 * 2^52
_E_XMAX = _b2f(0x7fefffffffffffff)
_E_XMIN = _b2f(0x0010000000000000)
_E_INF_BITS = 0x7ff0000000000000

_ET_LO = [_b2f(_u64(_EXP_TBL, j * 16)) for j in range(64)]
_ET_HI_BITS = [_u64(_EXP_TBL, j * 16 + 8) for j in range(64)]


def exp(x: float) -> float:
    """Bit-exact venue-runtime exp."""
    x_bits = _f2b(x)
    hi32 = (x_bits >> 32) & _M32
    top15 = (x_bits >> 48) & 32767
    if (((16527 - top15) | (top15 - 15504)) & _M32) >= 0x80000000:
        # outside the main range: special and small-argument handling
        mag = hi32 & 2147483647
        if mag >= 1083179008:  # |x| >= ~709.78 or non-finite
            if mag < 2146435072:
                if hi32 >= 0x80000000:  # underflow to zero
                    return _E_XMIN * _E_XMIN
                return _E_XMAX * _E_XMAX  # overflow to inf
            if mag > 2146435072 or (x_bits & _M32) != 0:
                return x + x  # NaN
            if hi32 == 2146435072:
                return _b2f(_E_INF_BITS)  # exp(+inf)
            return 0.0  # exp(-inf)
        return x + 1.0  # tiny |x|

    s = x * _E_LOG2_64 + _E_SHIFTER
    nd = s - _E_SHIFTER
    n64 = int(nd)
    j = n64 & 63
    n = n64 >> 6
    y = x - nd * _E_LN2_64_HEAD
    y = y - nd * _E_LN2_64_TAIL
    scale_bits = (((n64 & 0xffffffc0) + 0x0000ffc0) << 46) & _M64
    y2 = y * y
    y3 = y * y2
    p_lo = y3 * y2 * (_E_P5_LO + _E_P3_LO * y)
    p_hi = y3 * (_E_P5_HI + _E_P3_HI * y)
    res2_bits = _ET_HI_BITS[j] | scale_bits
    res2 = _b2f(res2_bits)
    acc = (y + _ET_LO[j]) + p_lo
    acc = p_hi + acc
    acc = acc + _E_HALF * y2
    if 0 <= n + 894 <= 1916:
        return acc * res2 + res2

    # result near the overflow/underflow boundary: split the 2^n scaling
    shift = (-1022 - n) & _M32
    mask = (_M64 << shift) & _M64 if shift < 64 else 0
    delta = (((n >> 1) & 0xFFFF) << 20) & _M32
    res2_adj = _b2f((res2_bits - (delta << 32)) & _M64)
    scale3 = _b2f((((delta + 0x3ff00000) & _M32) << 32))
    acc2 = acc * res2_adj
    if shift <= 52:
        dropped = _b2f(_f2b(res2_adj) & mask)
        low_part = res2_adj - dropped
        acc2 = acc2 + low_part
        if n >= 1023:  # overflow side
            return (acc2 + dropped) * scale3
        neg = (_f2b(acc2) >> 48) & 32768
        if (shift | neg) == 0:
            return (acc2 + dropped) * scale3
        saved = acc2
        result = (acc2 + dropped) * scale3
        if (_f2b(result) >> 48) & 32752:
            return result  # still a normal number
        # denormal result: redo the last step in fixed point to get the
        # correctly rounded significand
        a = saved * scale3
        bpart = dropped * scale3
        a_bits = _f2b(a)
        b_bits = _f2b(bpart)
        if (a_bits ^ b_bits) >> 63:
            return _b2f((b_bits - (a_bits & 0x7FFFFFFFFFFFFFFF)) & _M64)
        return _b2f((b_bits + (a_bits & 0x7FFFFFFFFFFFFFFF)) & _M64)
    # deep denormal
    return (acc2 + res2_adj) * scale3

# --- the processor's approximate reciprocal ---------------------------------

# log, log10 and pow all begin by asking the processor for an APPROXIMATE
# reciprocal of the argument's mantissa (RCPPS) and round that onto their own
# table's grid; the bin the rounding lands in decides the table row, and so the
# last bit of the result. The instruction is only specified to 1.5 * 2**-12, and
# vendors and generations answer differently: what matters is what the VENUE's
# processor answers. It depends only on the top 12 mantissa bits, so it is one
# float32 per bin of [1, 2). _rcp_model is the textbook answer -- the reciprocal
# of the bin's midpoint at 12 significant bits -- and _RCP_MEASURED overrides it
# in the 414 bins where the venue answers otherwise. Those answers lean up to
# 1.45 * 2**-12 off 1/midpoint and are not all on the 12-bit grid, so they are
# pinned by measurement, not a formula: every bin was probed with arguments
# crafted so that each table row the three functions could pick gives a
# different result (probes rcp_probe, rcp2, rcp3: 9000+ arguments), and each
# value below lies in the interval all of them allow. In 81 bins a log or log10
# row boundary stays open: no argument that tells its two sides apart turned up
# in 500000 tries.


def _rcp_model(j: int) -> int:
    midpoint = 1.0 + (j + 0.5) / 4096.0
    return ((_unpack('<I', _pack('<f', 1.0 / midpoint))[0] + 0x800) >> 12) << 12


_RCP_MEASURED: dict[int, int] = {
    8: 0x3f7f7000, 9: 0x3f7f6000, 19: 0x3f7ec000, 24: 0x3f7e78bb, 42: 0x3f7d7000,
    76: 0x3f7b6000, 87: 0x3f7ab000, 90: 0x3f7a8000, 110: 0x3f795000, 123: 0x3f788888,
    128: 0x3f784000, 140: 0x3f779000, 175: 0x3f75796c, 194: 0x3f747000, 203: 0x3f73f000,
    237: 0x3f71f000, 243: 0x3f71b000, 244: 0x3f71a000, 249: 0x3f714000, 255: 0x3f70f000,
    265: 0x3f706000, 274: 0x3f6fe000, 280: 0x3f6fa000, 286: 0x3f6f5000, 325: 0x3f6d3000,
    338: 0x3f6c7a1e, 350: 0x3f6be000, 351: 0x3f6bd000, 364: 0x3f6b2000, 377: 0x3f6a7000,
    394: 0x3f699000, 424: 0x3f680000, 431: 0x3f679000, 439: 0x3f674000, 451: 0x3f66a000,
    465: 0x3f65f000, 473: 0x3f657000, 493: 0x3f647000, 500: 0x3f643000, 508: 0x3f63b000,
    513: 0x3f637acf, 531: 0x3f629000, 542: 0x3f622000, 556: 0x3f617000, 563: 0x3f610000,
    570: 0x3f60c000, 621: 0x3f5e5000, 636: 0x3f5da000, 652: 0x3f5ce000, 658: 0x3f5c8000,
    680: 0x3f5b9000, 688: 0x3f5b2000, 702: 0x3f5a9000, 734: 0x3f592000, 742: 0x3f58c000,
    749: 0x3f587000, 779: 0x3f572000, 780: 0x3f571000, 796: 0x3f566000, 820: 0x3f555000,
    836: 0x3f54a000, 852: 0x3f53f000, 868: 0x3f534000, 893: 0x3f522000, 901: 0x3f51e000,
    918: 0x3f511000, 926: 0x3f50d000, 941: 0x3f503000, 976: 0x3f4ec000, 985: 0x3f4e5000,
    993: 0x3f4e1000, 1002: 0x3f4da000, 1010: 0x3f4d6000, 1027: 0x3f4ca7e5, 1037: 0x3f4c5000,
    1054: 0x3f4ba000, 1085: 0x3f4a7000, 1089: 0x3f4a4000, 1107: 0x3f499000, 1134: 0x3f487000,
    1143: 0x3f483000, 1144: 0x3f482000, 1152: 0x3f47c000, 1171: 0x3f472000, 1198: 0x3f460000,
    1207: 0x3f45c000, 1237: 0x3f449000, 1254: 0x3f43f000, 1268: 0x3f438000, 1273: 0x3f434000,
    1282: 0x3f430000, 1292: 0x3f429000, 1350: 0x3f409000, 1370: 0x3f3fe000, 1389: 0x3f3f3000,
    1409: 0x3f3e78bb, 1429: 0x3f3dd000, 1431: 0x3f3dc000, 1440: 0x3f3d7000, 1460: 0x3f3cc000,
    1491: 0x3f3ba000, 1499: 0x3f3b7000, 1522: 0x3f3ab000, 1532: 0x3f3a4000, 1534: 0x3f3a3000,
    1542: 0x3f3a0000, 1553: 0x3f399000, 1557: 0x3f397000, 1563: 0x3f395000, 1569: 0x3f392000,
    1596: 0x3f384000, 1605: 0x3f37e000, 1607: 0x3f37d000, 1617: 0x3f379000, 1639: 0x3f36e000,
    1641: 0x3f36d000, 1650: 0x3f367000, 1660: 0x3f363000, 1682: 0x3f35796c, 1704: 0x3f34d000,
    1714: 0x3f347bf6, 1726: 0x3f342000, 1738: 0x3f33b000, 1751: 0x3f335819, 1761: 0x3f331000,
    1789: 0x3f322000, 1795: 0x3f31f000, 1806: 0x3f31b000, 1812: 0x3f317000, 1826: 0x3f3105fe,
    1845: 0x3f307000, 1856: 0x3f302000, 1867: 0x3f2fd768, 1873: 0x3f2fb000, 1897: 0x3f2f0000,
    1909: 0x3f2e9000, 1914: 0x3f2e78bc, 1933: 0x3f2de000, 1944: 0x3f2da000, 1946: 0x3f2d8000,
    1948: 0x3f2d7000, 1968: 0x3f2cf000, 1975: 0x3f2cc000, 1981: 0x3f2c8000, 1986: 0x3f2c7000,
    1993: 0x3f2c4000, 1999: 0x3f2c1000, 2005: 0x3f2bd000, 2012: 0x3f2ba000, 2017: 0x3f2b9000,
    2024: 0x3f2b6000, 2036: 0x3f2af000, 2043: 0x3f2ac000, 2048: 0x3f2ab000, 2055: 0x3f2a7b80,
    2069: 0x3f2a1ac0, 2073: 0x3f2a0000, 2074: 0x3f2a0000, 2086: 0x3f299000, 2093: 0x3f296000,
    2111: 0x3f28e000, 2130: 0x3f287000, 2131: 0x3f287000, 2137: 0x3f284000, 2138: 0x3f284000,
    2153: 0x3f27c000, 2156: 0x3f27c000, 2163: 0x3f279000, 2169: 0x3f275000, 2182: 0x3f271000,
    2189: 0x3f26e000, 2196: 0x3f26a521, 2241: 0x3f25768b, 2248: 0x3f255000, 2262: 0x3f24e000,
    2267: 0x3f24d000, 2275: 0x3f24a000, 2279: 0x3f247000, 2281: 0x3f246000, 2284: 0x3f246000,
    2294: 0x3f242000, 2301: 0x3f23f000, 2308: 0x3f23b000, 2315: 0x3f238000, 2328: 0x3f234000,
    2330: 0x3f232000, 2342: 0x3f22d000, 2350: 0x3f22a000, 2357: 0x3f227000, 2370: 0x3f223000,
    2375: 0x3f221000, 2376: 0x3f221000, 2383: 0x3f21e000, 2390: 0x3f21b000, 2391: 0x3f21b000,
    2405: 0x3f214000, 2418: 0x3f210000, 2422: 0x3f20d000, 2446: 0x3f205000, 2469: 0x3f1fb9e3,
    2474: 0x3f1fa000, 2482: 0x3f1f7000, 2489: 0x3f1f3000, 2497: 0x3f1f0000, 2503: 0x3f1ef000,
    2518: 0x3f1e8b4d, 2519: 0x3f1e8b4d, 2533: 0x3f1e2000, 2547: 0x3f1de000, 2554: 0x3f1da000,
    2562: 0x3f1d7000, 2566: 0x3f1d7000, 2568: 0x3f1d6000, 2576: 0x3f1d3000, 2597: 0x3f1cb000,
    2598: 0x3f1cb000, 2605: 0x3f1c7a1e, 2606: 0x3f1c7a1e, 2614: 0x3f1c5000, 2621: 0x3f1c1000,
    2629: 0x3f1be000, 2635: 0x3f1bd000, 2643: 0x3f1ba000, 2648: 0x3f1b7000, 2658: 0x3f1b3000,
    2664: 0x3f1b15ad, 2673: 0x3f1af000, 2689: 0x3f1a8000, 2695: 0x3f1a7000, 2703: 0x3f1a4000,
    2711: 0x3f1a0000, 2714: 0x3f1a0000, 2719: 0x3f19d000, 2734: 0x3f199000, 2736: 0x3f197000,
    2742: 0x3f196000, 2749: 0x3f192000, 2764: 0x3f18e000, 2780: 0x3f187000, 2789: 0x3f184000,
    2795: 0x3f183000, 2804: 0x3f180000, 2818: 0x3f17b000, 2820: 0x3f179000, 2826: 0x3f17780a,
    2827: 0x3f17780a, 2835: 0x3f175000, 2844: 0x3f172000, 2852: 0x3f16e000, 2867: 0x3f16a000,
    2876: 0x3f167000, 2899: 0x3f15f000, 2908: 0x3f15c000, 2925: 0x3f155905, 2926: 0x3f155905,
    2931: 0x3f154000, 2940: 0x3f151000, 2949: 0x3f14e000, 2957: 0x3f14a000, 2966: 0x3f147000,
    2980: 0x3f142a6f, 2981: 0x3f142a6f, 2989: 0x3f13f000, 3005: 0x3f13b000, 3006: 0x3f13b000,
    3014: 0x3f137acf, 3023: 0x3f134000, 3039: 0x3f130000, 3048: 0x3f12d000, 3050: 0x3f12b000,
    3056: 0x3f129000, 3062: 0x3f127000, 3065: 0x3f126000, 3074: 0x3f123000, 3077: 0x3f122000,
    3081: 0x3f122000, 3099: 0x3f11b000, 3106: 0x3f11a000, 3115: 0x3f117000, 3116: 0x3f117000,
    3133: 0x3f1105ff, 3134: 0x3f1105ff, 3150: 0x3f10c000, 3159: 0x3f109000, 3194: 0x3f0fd768,
    3211: 0x3f0f7000, 3212: 0x3f0f7000, 3221: 0x3f0f4b2f, 3222: 0x3f0f4b2f, 3238: 0x3f0f0000,
    3247: 0x3f0ec000, 3256: 0x3f0e9000, 3257: 0x3f0e9000, 3263: 0x3f0e78bc, 3264: 0x3f0e78bc,
    3265: 0x3f0e78bc, 3274: 0x3f0e5000, 3280: 0x3f0e3000, 3300: 0x3f0dd000, 3310: 0x3f0da000,
    3312: 0x3f0d8000, 3315: 0x3f0d7000, 3319: 0x3f0d7000, 3328: 0x3f0d3000, 3338: 0x3f0d0000,
    3341: 0x3f0cf000, 3346: 0x3f0cf000, 3356: 0x3f0cc000, 3365: 0x3f0c8000, 3368: 0x3f0c7000,
    3372: 0x3f0c7000, 3382: 0x3f0c4000, 3392: 0x3f0c1000, 3393: 0x3f0c1000, 3402: 0x3f0bd590,
    3403: 0x3f0bd590, 3411: 0x3f0ba000, 3419: 0x3f0b9000, 3421: 0x3f0b7000, 3429: 0x3f0b6000,
    3430: 0x3f0b6000, 3448: 0x3f0af000, 3459: 0x3f0ac000, 3466: 0x3f0aa6f9, 3467: 0x3f0aa6f9,
    3475: 0x3f0a7b80, 3476: 0x3f0a7b80, 3485: 0x3f0a4000, 3486: 0x3f0a4000, 3496: 0x3f0a1ac0,
    3497: 0x3f0a1ac0, 3504: 0x3f0a0000, 3505: 0x3f0a0000, 3514: 0x3f09d000, 3515: 0x3f09d000,
    3528: 0x3f099000, 3529: 0x3f099000, 3542: 0x3f095000, 3553: 0x3f092000, 3562: 0x3f08e000,
    3573: 0x3f08b000, 3583: 0x3f088000, 3586: 0x3f087000, 3587: 0x3f087000, 3591: 0x3f087000,
    3593: 0x3f085000, 3594: 0x3f085000, 3611: 0x3f080000, 3622: 0x3f07d000, 3625: 0x3f07c000,
    3626: 0x3f07c000, 3630: 0x3f07c000, 3643: 0x3f077000, 3650: 0x3f075000, 3651: 0x3f075000,
    3661: 0x3f072000, 3669: 0x3f071000, 3680: 0x3f06e000, 3681: 0x3f06e000, 3691: 0x3f06a521,
    3692: 0x3f06a521, 3709: 0x3f066000, 3710: 0x3f066000, 3720: 0x3f063000, 3731: 0x3f060000,
    3741: 0x3f05c000, 3752: 0x3f059000, 3759: 0x3f05768b, 3760: 0x3f05768b, 3761: 0x3f05768b,
    3771: 0x3f055000, 3772: 0x3f055000, 3792: 0x3f04ea52, 3793: 0x3f04ea52, 3812: 0x3f04a000,
    3813: 0x3f04a000, 3819: 0x3f047000, 3827: 0x3f046000, 3833: 0x3f043000, 3834: 0x3f043000,
    3842: 0x3f042000, 3843: 0x3f042000, 3853: 0x3f03f000, 3854: 0x3f03f000, 3863: 0x3f03b000,
    3864: 0x3f03b000, 3879: 0x3f037000, 3897: 0x3f032000, 3898: 0x3f032000, 3906: 0x3f031000,
    3907: 0x3f031000, 3917: 0x3f02d000, 3937: 0x3f029000, 3940: 0x3f027000, 3959: 0x3f022000,
    3968: 0x3f021000, 3969: 0x3f021000, 3980: 0x3f01e000, 3991: 0x3f01b000, 3992: 0x3f01b000,
    4002: 0x3f0174b3, 4003: 0x3f0174b3, 4013: 0x3f014000, 4014: 0x3f014000, 4025: 0x3f011000,
    4026: 0x3f011000, 4034: 0x3f010000, 4035: 0x3f010000, 4041: 0x3f00d000, 4046: 0x3f00d000,
    4057: 0x3f009000, 4077: 0x3f00461c, 4078: 0x3f00461c, 4090: 0x3f002000,
}

_RCP = [_RCP_MEASURED.get(j, _rcp_model(j)) for j in range(4096)]


# --- log ---------------------------------------------------------------------

# Reduction table: for each of the 129 bins the reduction can land in, the head
# and tail halves of ``-ln(B)`` for that bin's reciprocal ``B``. Sixteen bytes
# per entry, addressed by the BYTE offset the reduction computes directly.
_LOG_TBL = bytes.fromhex(
    '0038fafe422ee63f3067c79357f32e3d001824aa82eee53fbe46da0c3802223d0048365c'
    '40afe53ffbc910ac63fa2d3d008cbb267a70e53fdd0333ff0b98093d007886262e32e53f'
    '3175255dc4cc053d00505a835af4e43ffbb8936d516c2ebd000c976ffdb6e43f1c544ced'
    '1571ef3c00a4e827157ae43faa604df96acb22bd0024f9f29f3de43ff75110484f98fdbc'
    '00cc25219c01e43f4cc7f03079ce26bd00c0360c08c6e33fc213fe7c36b702bd00781917'
    'e28ae33fa46955bb7a8b21bd008c9dad2850e33face627953fb8103d00083444da15e33f'
    '9ceda0c5934e27bd00e0b057f5dbe23f11dcb907e5a617bd00c00e6d78a2e23f2d8897e7'
    '2b6d203d00dc34116269e23f50622205f1610bbd00bcbed8b030e23f7b66486e06fc123d'
    '0018c65f63f8e13fd381fec942722abd0060ae4978c0e13f67e670eddeaccc3c003cf240'
    'ee88e13f5046abf84ecc143d0098f2f6c351e13f49ae93a297dd2ebd005cc723f81ae13f'
    'b2dc9dbb478625bd00cc118689e4e03f4217800798291c3d0054d0e276aee03f277e7e88'
    '6b481f3d00c43305bf78e03ffdf5ed412281263d000476be6043e03fe03995e75fc404bd'
    '0008b2e55a0ee03f1c7b72b1a33b05bd00487aaf58b3df3f3549163cfa85003d001803ee'
    'a74adf3f8b4a016fe5cd123d0010b456a1e2de3f5102475af4272f3d00b0ddc3427bde3f'
    '08bd7253506524bd0028271a8a14de3f38293207b22613bd00984c4875aedd3f6a61dc60'
    '2da41ebd00f8de460249dd3fa867a7e9af5b233d004806182fe4dc3fb0a6c73ec39707bd'
    '005845c7f97fdc3fae4952c1ddb629bd00a03f69601cdc3f80e1e87f80ec2c3d00e0801b'
    '61b9db3f6d660af45bd8273d00284604fa56db3f9519842d2595103d00d0485229f5da3f'
    '58447752c57c21bd00d88a3ced93da3f5d7aa7bef2361e3d00f824024433da3ff5799d7f'
    '45c6233d00f015ea2bd3d93fb0c0d0109e2726bd00581343a373d93ff0d902a5132315bd'
    '00f85b63a814d93f7d30e62eb56617bd00308ba839b6d83f7004e7e5e15a20bd00c86d77'
    '5558d83f8a7733336fd52f3d0018d83bfafad73f6a5612c8902027bd00f87c68269ed73f'
    '7817fd2e7dec293d0078c676d841d73fb360dc49098b2d3d0018afe60ee6d63f872d227c'
    '6521173d00689c3ec88ad63fa0eb5627d3a0203d00b03a0b0330d63f00ae31e723b62dbd'
    '006059dfbdd5d53fdc65a4082a0b0abd00d0c853f77bd53fef405deeedad1f3d00a03807'
    'ae22d53f59c7648170be2e3d0030179ee0c9d43fa4d80a1b89202ebd00c871c28d71d43f'
    '75d66709ce272fbd00e8d523b419d43f9de090ec36e4083d0030337752c2d33f5cbd06b6'
    '543b183d0010be76676bd33fc877f1b0cd6e113d0060d3e1f114d33fb83c21d37ae228bd'
    '0090dc7cf0bed23ff404504afa9c2a3d00d834116269d23fb6b35bdfc1932c3d00b80e6d'
    '4514d23feaba46bade870a3d00685a6399bfd13fb7bd4751eda62c3d00f8accb5c6bd13f'
    '8116a5f7cd9a2b3d00e827828e17d13f1cf0a5630e212cbd006061672dc4d03fe9ea3c16'
    '8b18273d00584d603871d03f914eed16db9cf83c00c82656ae1ed03f4ae985148cf016bd'
    '00b0b36c1c99cf3f30df0ccaeccb1b3d0000dde4adf5ce3f118ebb651521cabc0010e7ff'
    '0e53ce3f30f441602712c23c0090d4b03db1cd3f35b015f72aff2abd006065f23710cd3f'
    'e4f6b6757e4a08bd0010f0c6fb6fcc3fd22b96c572ecf1bc00e03b3887d0cb3fb6125459'
    'c44b2dbd00d05b57d831cb3faae1ac4e8d350cbd00e08a3ced93ca3f69215650437228bd'
    '00900807c4f6c93f7a8165684d90293d0000f7dc5a5ac93f6fffa05828f2073d000039eb'
    'afbec83fd12ce9aa543d07bd00a05165c123c83f831e639adb0d1e3d005044858d89c73f'
    '0543917010661cbd0070758b12f0c63fe1219ce58d1125bd00108cbe4e57c63f782e3c2c'
    '8bcf193d0040546b40bfc53f1c9868eb237012bd00b0a1e4e527c53fc77d69e5e833263d'
    '00b033833d91c43f78b6fd547983253d003099a545fbc33f4d356a7ed8d12cbd009015b0'
    'fc65c33f89724b23a82fc63c0080860c61d1c23fa1b481cb6c9d033d00c0492a713dc23f'
    '5cdfd38f230d103d00f0237e2baac13f349938448ea72c3d00e027828e17c13ff2072dce'
    '78ef213d00409eb59885c03f2c900970dde527bd00e0db3991e8bf3ffd0aa14fd63425bd'
    '00200a8339c7be3fe045e6af68c02dbd0040846327a7bd3f3317a71f40891a3d0040bc01'
    '5888bc3fd3ac5ac6d146263d0060ad8dc86abb3fe568f72b809013bd00c0b140764eba3f'
    'c80744b9b6420ebd0040595d5e33b93fda47bd3a5c11233d00e0402f7e19b83ff7fd6ff9'
    'dc800f3d00c0ea0ad300b73f32ed9da98d1eec3c00a0974d5ae9b53f1e1d5d3c06692cbd'
    '0080205d11d3b43fefe1f482253af5bc00e0d1a7f5bdb33fd74edba55ec82c3d002047a4'
    '04aab23f7d699caee8b620bd006046d13b97b13f9b9e0d565d3225bd00409eb59885b03f'
    '2c900970dde517bd00c006c031eaae3f7b3bc94f3e110ebd00c0ddcd73cbac3f0728d847'
    'f2681abd0000fbd0f2aeaa3f2eb43b351afc203d00c09f14aa94a83f7d265ad0957919bd'
    '00c0d4f2947ca63fa2af19ecfb9e02bd00002ed4ae66a43f28fdbd7573162cbd00008d2f'
    'f352a23f7bb621e09a3e283d0040e7895d41a03f53d7f15cc011013d008014ecd2639c3f'
    'f3b29e3fc678253d0000c9282549983f340c5a32baa02abd00009825a932943ffe378692'
    '3981093d008093585620903fd2f7e2065bdc23bd000089a34824883f40f674da775527bd'
    '000089751510803fe82b9d996bc710bd000058590508703f7bc631cbaf66213d00000000'
    '000000000000000000000080'
)

_L_LOG2_HI = _b2f(0x3fa62e42fefa3800)  # ln2 / 16, head
_L_LOG2_LO = _b2f(0x3ceef35793c76730)  # ln2 / 16, tail
# log1p series coefficients, in the two interleaved chains the algorithm keeps
_L_C7 = _b2f(0x3fc2492492492492)
_L_C6 = _b2f(0xbfc5555e3d6fb175)
_L_C5 = _b2f(0x3fc999999999999a)
_L_C4 = _b2f(0xbfd0000000000000)
_L_C3 = _b2f(0x3fd5555555555555)
_L_C2 = _b2f(0xbfe0000000000000)
_L_INF = _b2f(0x7ff0000000000000)
_L_NAN = _b2f(0x7ff8000000000000)
# The reduction carries the mantissa at 2**896 and meets a reciprocal near
# 2**-896, so their product lands next to 1 with the cancellation exposed.
_L_MANT_SCALE = _b2f(0x77f0000000000000)
_L_ULP52 = 2.0 ** -52
# 2**128 lifts a denormal into the normal range; the exponent term pays it back
# through the larger bias
_L_DENORM_SCALE = _b2f(0x47f0000000000000)
_L_BIAS = 16352
_L_DENORM_BIAS = 18416

_L_THI = [_b2f(_u64(_LOG_TBL, j * 16)) for j in range(129)]
_L_TLO = [_b2f(_u64(_LOG_TBL, j * 16 + 8)) for j in range(129)]


def log(x: float) -> float:
    """``ln(x)`` as the venue's engine computes it.

    The venue evaluates ``math.log`` with the JVM's x86 ``Math.log`` intrinsic,
    the Intel LIBM stub, and that stub is NOT correctly rounded: it reduces the
    argument against a 129-bin reciprocal table and finishes with a degree-7
    ``log1p`` series in plain doubles, so its last bit is its own. This ports the
    algorithm operation for operation, including the order its four partial sums
    are folded together, which is what decides that bit.

    MEASURED (BINANCE:BTCUSDT 30m, 120856 arguments — ``log`` of every ratio of
    two prices of the same bar over 30213 bars): exact on every one. A correctly
    rounded ``ln`` misses 6 of them and an exact reciprocal in the bin reduction
    misses another, all seven with the argument inside ``[0.988, 1.0021]``, where
    the reduction is at its weakest; a script that divides one nearly cancelling
    difference of logs by another turns that single ulp into a visible divergence.
    Exact as well on crafted arguments for every reciprocal bin (probes
    ``rcp_probe``/``rcp2``/``rcp3``) and on an independent holdout of 66678
    arguments on CAPITALCOM:BTCUSD 15m.

    The reciprocal instruction fills a second lane from a denormal, whose
    reciprocal is an infinity that contributes nothing, and its own lane can never
    overflow because the mantissa it reads is a float32 in ``[1, 2)`` — both are
    folded into the table lookup below.

    :param x: The argument.
    :return: Its natural logarithm, bit-identical to the venue's.
    """
    bits = _f2b(x)
    head = (bits >> 48) & 0xFFFF
    biased = (head - 16) & _M32
    if biased >= 32736:
        # zero, denormal, infinity, NaN or negative — everything the reduction
        # below cannot take. Only a positive denormal comes back out of here.
        if head < 16:
            if bits == 0:
                return -_L_INF
            bits = _f2b(x * _L_DENORM_SCALE)
            head = (bits >> 48) & 0xFFFF
            bias = _L_DENORM_BIAS
        elif head < 32768:
            return x + x  # +inf, positive NaN
        elif (bits << 1) & _M64 == 0:
            return -_L_INF  # -0.0
        elif head >= 0xFFF0 and bits & 0xFFFFFFFFFFFFF:
            return x + x  # negative NaN
        else:
            return _L_NAN  # -inf and every other negative
    else:
        head = biased
        bias = _L_BIAS

    # B, the bin's reciprocal: the processor's approximate reciprocal of the
    # mantissa (see _RCP), rounded onto the table's grid by the added half-bin.
    # Dividing here instead costs the last bit on roughly one argument in 20000.
    lane = (_RCP[(bits >> 40) & 0xFFF] + 32768) & _M32
    b = _b2f((lane << 29) & 0xffffe00000000000)

    # r = B * mantissa - 1, the mantissa split in two so the product's own
    # rounding cannot eat the cancellation
    fraction = bits & 0xFFFFFFFFFFFFF
    mant_hi = (1.0 + (fraction & 0x000FE00000000000) * _L_ULP52) * _L_MANT_SCALE
    r = ((1.0 + fraction * _L_ULP52) * _L_MANT_SCALE - mant_hi) * b \
        + (mant_hi * b - 1.0)

    bin_index = (lane >> 16) & 0xFF
    k = float((head & 32752) - bias)
    head_term = _L_THI[bin_index] + k * _L_LOG2_HI
    result = head_term + r

    r2 = r * r
    tail = r + (head_term - result)
    tail = tail + (k * _L_LOG2_LO + _L_TLO[bin_index])
    tail = tail + ((_L_C6 * r + _L_C5) * r + (_L_C7 * r) * r2) * (r2 * r2)
    tail = tail + ((_L_C3 * r + _L_C2) + _L_C4 * r2) * r2
    return result + tail


# --- log10 -------------------------------------------------------------------

# Reduction table: for each of the 129 bins, the head and tail halves of
# ``-log10(B / LH)`` for that bin's scaled reciprocal. Addressed like _LOG_TBL.
_L10_TBL = bytes.fromhex(
    '00789f501344d33f58b3121f31ef1f3d003433801824d33fd0d971c6bf42f5bc00501951'
    '4204d33fc3b0a4786a21183d0094c76f90e4d23f9d38fa80692890bc0040d08902c5d23f'
    '64f5c2755407043d001cdd4d98a5d23fc3b219d2841dfabc007caa6b5186d23fc1be9afd'
    'd3e61b3d008802942d67d23f55a489e25ede1ebd0064b8782c48d23f79d134679be71f3d'
    '00c8a3cc4d29d23fb8401a98ea34edbc00509c43910ad23f372739ccc39c1abd002c7592'
    'f6ebd13fe7afc903f8981e3d00dcf86e7dcdd13ff4e7da716ca8083d00dce48f25afd13f'
    'a18591ee1234ffbc0094e5acee90d13f53b3cac2d97ef13c0050927ed872d13fb2c15269'
    '1c52f13c004469bee254d13fca79cbca78dc0bbd00accb260d37d13fe14d1ff7bef801bd'
    '0008fa725719d13f0b91bf552b6e943c0060105fc1fbd03fc139e6394ba8143d00a802a8'
    '4aded03f5d1df3d3858317bd0020990bf3c0d03f6f1043382f601fbd00e86c48baa3d03f'
    '7c4919887a98ef3c0094e41da086d03f6704aa1cc7ae0f3d00cc304ca469d03f724342a4'
    'fc1816bd00004994c64cd03fd21765944b3818bd0040e8b70630d03f379c10e0aca619bd'
    '000c8a796413d03f64e82151f74c16bd0080ce38bfedcf3f1a4d214602c4bbbc0020e6c8'
    'efb4cf3f0332b9da76011e3d0028b02c5a7ccf3fe4a82e2a6ac8febc00a0eaeefd43cf3f'
    'a4498ec1a810f13c00e8b69bda0bcf3fc0c93c9299ce15bd00f093c0efd3ce3fe9514b4d'
    'c7041a3d00f858ec3c9cce3f59ad3c166082ac3c0070909a7d2dce3f4636a93fc0a1e43c'
    '0010313799bfcd3ffdd1ab329dea073d00b844678c52cd3fd4dfcb4de2081bbd00e86de3'
    '53e6cc3f7f7f7b0b038f1bbd00685077ec7acc3ffbc921a863c1133d0088ff005310cc3f'
    '76ca6b53e54e07bd0098717084a6cb3f6b9bdad716bf1fbd00d8f8c67d3dcb3fb30b22e2'
    '5d291a3d0058c1163cd5ca3f1e9124e72258f5bc00385382bc6dca3f7123986d7c56ac3c'
    '00e8193cfc06ca3f807dd18404a21d3d0080ef85f8a0c93f6a6a4654042200bd0020acb0'
    'ae3bc93f65fd01d60c84183d00b0b91b1cd7c83f6687f57b97f814bd00e8aa343e73c83f'
    '24acf63a455c0fbd0080d6761210c83fa1e10343809a1fbd0078f56a96adc73f46cbfb43'
    '3ec3f43c0010c5a6c74bc73fc5eaf0703b2e19bd0098abcca3eac63ffe3d09c015af0fbd'
    '00b8608b288ac63ffdd578deeea49ebc0070989d532ac63f6eea2b96844019bd00e8b0c9'
    '22cbc53f99d98d8801e21f3d004863e1936cc53fada7ad1688111b3d00c076c1a40ec53f'
    'b5b55941089cf0bc0060765153b1c43f233d39849ca8f6bc005069839d54c43fbb8b0b9f'
    '8c4b1c3d00588d5381f8c33f47f79df4999bf83c008013c8fc9cc33f34b803d59fb913bd'
    '0008dff00d42c33f86b311f0bed805bd006846e7b2e7c23fc27b9cf34eb91bbd0028d6cd'
    'e98dc23f9bd6e60505ed10bd00d815d0b034c23f9d6c9be267f91fbd00a84e2206dcc13f'
    'fc1177720db3ffbc00005401e883c13f5a6c7839573fc23c00984db2542cc13f42a305c9'
    '1d3a003d005883824ad5c03fc020999b5ab2033d00c02ac7c77ec03f246af24641faf03c'
    '00d835ddca28c03fdcd6d941654a033d00404752a4a6bf3f4964f644d3ca193d00d0a32d'
    'b8fcbe3f992983670f40183d0000a132ce53be3f1a3b0e9cfd62ffbc00706b55e3abbd3f'
    '136997023b24f8bc0080e897f404bd3f973779ec78051c3d00706409ff5ebc3f6505fc05'
    '9e791dbd006042c6ffb9bb3fedf5254623571f3d00d0aff7f315bb3f61ae5add1e7e1abd'
    '00b058d3d872ba3fd3e4143391bc173d00501f9babd0b93f4b514d9a9b8cf13c00e0d49c'
    '692fb93fab96447e6df9f13c00f0f431108fb83fe77964f51858163d008062bf9cefb73f'
    '6d48bf26a61311bd00b026b50c51b73f84331c1a8d89a93c00e0318e5db3b63f615387b3'
    'ac6106bd00e01dd08c16b63ffaac7c2a10df1bbd0030f20a987ab53f168886ffd046f03c'
    '0000ead87cdfb43fe7fb151529d51fbd00203bde3845b43f32a1596eeeae1f3d0090dfc8'
    'c9abb33f612332f1078819bd00105f502d13b33fabe6880880531e3d00d09b35617bb23f'
    '22bbbcdf2427febc00e09e4263e4b13f8cc5b46ed64dfebc0030674a314eb13f9bace14c'
    '91a61b3d0060b928c9b8b03fb813788c72380bbd0080f0c12824b03f2c8cbcc26beab53c'
    '00a0a1059c20af3f8ef1e87284dfe8bc00e0b5c06dfaad3f36f4de9f6473083d006041af'
    'c2d5ac3fa9c36810e727083d006035db96b2ab3fd3340a129f1a103d00a0fe5de690aa3f'
    '64d2deda92c314bd00c03460ad70a93fa9069d1c5e701bbd00604c19e851a83fd96a9983'
    'bc1701bd00c04acf9234a73f624aa9b142eaa5bc00407bd6a918a63fcad8ae759b1107bd'
    '00c0269129fea43f33d591528f65123d00404d6f0ee5a33fd95c2ccd705c1d3d008060ee'
    '54cda23f898400d102481a3d00e00099f9b6a13f9855fb543f5916bd0060bb06f9a1a03f'
    'b457ef646b6317bd000094b79f1c9f3f37476aee79d4b53c0000aa91f5f79c3f3c37163a'
    '1471083d00806b15edd59a3f4a556c83b000693c004076d47fb6983f7bf112ed74c9ffbc'
    '00c0de77a799963feae72c23bb351e3d0040bfbf5d7f943f6efa4fd8490a0e3d00c0c782'
    '9c67923f900e178df2d914bd0000d2ad5d52903f8ef8d98686b9de3c0000f186367f8c3f'
    '17a5e0b9aa9fe23c00805cb79e5e883fcb682554db7b1fbd0000b346e842843fd9e754b9'
    '87521e3d0000e6b7072c803f170bda2227fb19bd00008b6ce333783fef711282960f19bd'
    '000091293619703fa59134bc45cf1bbd00004a35e30f603f0a52ffc01cd719bd00000000'
    '000000000000000000000000'
)
_L10_LH = 0.43359375  # short approximation of log10(e) the reduction is scaled by
_L10_LH_F32 = _unpack('<f', _pack('<I', 0x3EDE0000))[0]
_L10_LE_TAIL = _b2f(0x3f5a7a6cbf2e4108)  # log10(e) / LH - 1
_L10_LOG2_HI = _b2f(0x3f934413509f7800)  # log10(2) / 16, head
_L10_LOG2_LO = _b2f(0x3cdfef311f12b358)  # log10(2) / 16, tail
_L10_C0 = _b2f(0x40358874c1a5f12e)
_L10_C1 = _b2f(0xc008930964d4ef0d)
_L10_C2 = _b2f(0xc025c917385593b1)
_L10_C3 = _b2f(0x3ffc6a02dc963467)
_L10_C4 = _b2f(0x4016ab9f7f9d3aa1)
_L10_C5 = _b2f(0xbff27af2dc77b115)

_L10_THI = [_b2f(_u64(_L10_TBL, j * 16)) for j in range(129)]
_L10_TLO = [_b2f(_u64(_L10_TBL, j * 16 + 8)) for j in range(129)]


def log10(x: float) -> float:
    """``log10(x)`` as the venue's engine computes it.

    The venue evaluates a runtime ``math.log10`` with the JVM's x86 ``Math.log10``
    intrinsic, the Intel LIBM stub that is ``log``'s sibling: the same 129-bin
    reduction, but against the reciprocal scaled by ``LH = 0.43359375`` -- a short
    approximation of ``log10(e)`` -- in float32, so ``r = B * m - LH`` already
    carries the base change, and a degree-6 series finishes it. Neither
    correctly rounded, nor the platform's ``log10``.

    MEASURED (BINANCE:BTCUSDT 30m, 363348 arguments -- prices, volumes, ranges
    and every ratio of two prices of the same bar, probe ``log10_probe``, plus
    crafted arguments for every reciprocal bin, probes ``rcp_probe``/``rcp2``/
    ``rcp3``, plus an independent holdout of 88904 arguments on CAPITALCOM:BTCUSD
    15m): exact on all of them. The platform's ``log10`` misses 212 of the first
    set.

    :param x: The argument.
    :return: Its base-10 logarithm, bit-identical to the venue's.
    """
    bits = _f2b(x)
    head = (bits >> 48) & 0xFFFF
    biased = (head - 16) & _M32
    if biased >= 32736:
        # the same special cases as log
        if head < 16:
            if bits == 0:
                return -_L_INF
            bits = _f2b(x * _L_DENORM_SCALE)
            head = (bits >> 48) & 0xFFFF
            bias = _L_DENORM_BIAS
        elif head < 32768:
            return x + x
        elif (bits << 1) & _M64 == 0:
            return -_L_INF
        elif head >= 0xFFF0 and bits & 0xFFFFFFFFFFFFF:
            return x + x
        else:
            return _L_NAN
    else:
        head = biased
        bias = _L_BIAS

    # B ~ LH / mantissa: the reciprocal times LH, rounded to float32 by the
    # instruction that multiplies it, then onto the table's grid
    rcp = _unpack('<f', _pack('<I', _RCP[(bits >> 40) & 0xFFF]))[0]
    lane = (_unpack('<I', _pack('<f', rcp * _L10_LH_F32))[0] + 0x8000) & _M32
    b = _b2f((lane << 29) & 0xffffe00000000000)

    # r = B * mantissa - LH with the mantissa split at 25 bits
    mant = (1.0 + (bits & 0xFFFFFFFFFFFFF) * _L_ULP52) * _L_MANT_SCALE
    mant_hi = _b2f(_f2b(mant) & 0xfffffffff8000000)
    r = (mant - mant_hi) * b + (mant_hi * b - _L10_LH)

    bin_index = ((lane >> 16) & 0xFF) - 94
    k = float((head & 32752) - bias)
    head_term = _L10_THI[bin_index] + k * _L10_LOG2_HI
    result = head_term + r

    r2 = r * r
    tail = r + (head_term - result)
    tail = tail + _L10_LE_TAIL * r
    tail = tail + (k * _L10_LOG2_LO + _L10_TLO[bin_index])
    tail = tail + ((_L10_C2 * r + _L10_C4) * r + (_L10_C0 * r) * r2) * (r2 * r2)
    tail = tail + ((_L10_C3 * r + _L10_C5) + _L10_C1 * r2) * r2
    return result + tail


# --- pow ---------------------------------------------------------------------

# log2 reduction table: for each of the 513 bins the reduction can land in, the
# head and tail halves of ``-log2(B / LH)`` for that bin's scaled reciprocal
_P_LOG_TBL = bytes.fromhex(
    '000000000000f03f0000000000000000000000200af0ef3f951f629656185b3e000000e0'
    '19e0ef3f9e6f91e5785232be000000002fd0ef3f62109a85b75f593e000000c049c0ef3f'
    '8ff145b2389c52be000000e069b0ef3fa78028ad301250be000000608fa0ef3f2024e7c8'
    'd17b593e00000080ba90ef3f00450cc3756c5dbe000000e0ea80ef3f433fc60218132e3e'
    '000000c02071ef3fccccd4b32ac544be000000005c61ef3f9713d9db6c7d4ebe000000a0'
    '9c51ef3f68cdc565c82d52be000000a0e241ef3f6c30d1460e845abe000000e02d32ef3f'
    '940e98d2af71503e000000a07e22ef3fdeba3a77e59158be000000a0d412ef3f6bf46bdc'
    'becc5cbe000000e02f03ef3ffa4772bc83ab2bbe0000008090f3ee3f461eaabc3bbb53be'
    '00000060f6e3ee3f2d686c5f19c654be0000008061d4ee3f68e34151866d4bbe000000e0'
    'd1c4ee3f768f67ecf69a36be0000008047b5ee3f551f304112432dbe00000060c2a5ee3f'
    'bda66d67d08d4dbe000000604296ee3fc491a85791f9513e000000a0c786ee3f1e49ebe4'
    'f99b573e000000205277ee3f2c4adcfde65633be000000c0e167ee3ff15b5bd7319544be'
    '000000807658ee3f8e3b42bde44ff53d000000601049ee3fb9510e339c28543e00000080'
    'af39ee3f5fa95186d6aa55be000000a0532aee3f08c7985ea9c42fbe000000e0fc1aee3f'
    '8d3289098c95233e00000040ab0bee3fbd2a64eed85d42be000000a05efced3f36d294c3'
    '6263523e0000002017eded3f8eaa04e147e24c3e000000c0d4dded3fe49b5a267ab75bbe'
    '0000004097ceed3f2fc5ca0eb17c4a3e000000e05ebfed3fb8b34c122470253e00000080'
    '2bb0ed3fbefed4e6ee3320be00000020fda0ed3f0ea0cc39bcda3dbe000000c0d391ed3f'
    '2a558aef903354be00000040af82ed3f0452e8b85038513e000000e08f73ed3f08fe593d'
    '28b75dbe000000407564ed3fd1eaa73a4b80583e000000c05f55ed3fa95ba3f8b09852be'
    '000000004f46ed3f15dd889adb8c5a3e000000404337ed3f90a1b0b03586593e00000080'
    '3c28ed3f953211e219115cbe000000803a19ed3f2817bfaf9c2e49be000000603d0aed3f'
    'f3cca4e40eb9193e0000002045fbec3fb8be3cba506b403e000000c051ecec3fdd7d0f11'
    '06680d3e0000004063ddec3f08d5d77d43895abe0000008079ceec3f71f2609b6a6750be'
    '0000008094bfec3f60d69a0b4f17593e00000060b4b0ec3f9c3d820072bf5b3e00000020'
    'd9a1ec3f89eca638f9384dbe000000800293ec3f8e7d0b3afddb533e000000c03084ec3f'
    '346b82c6c9c527be000000c06375ec3f8163700c533659be000000609b66ec3fc74ef37d'
    'b51a463e000000e0d757ec3fe8e7e540ae3d5cbe000000001949ec3f0f7702569d2155be'
    '000000c05e3aec3feb1179ec255d5a3e00000060a92bec3f25a29eb30bc053be00000080'
    'f81cec3f2e217a96df8d5a3e000000604c0eec3fbd980758ab535f3e00000000a5ffeb3f'
    'f62d28b874b846be0000002002f1eb3f29673ae33f96543e0000000064e2eb3f8ae8533b'
    'e1dc3abe00000060cad3eb3f845058c29fde5c3e0000008035c5eb3feec535a39cfd39be'
    '00000020a5b6eb3f4db0257315ba423e0000006019a8eb3f0f546415359f3a3e00000040'
    '9299eb3f92f5ff83ce6554be000000a00f8beb3fd363dab90a1a4bbe00000080917ceb3f'
    'a41e6f6d5776553e00000000186eeb3fbfa1805eb6db4d3e00000000a35feb3fb5ac9e1c'
    '7728593e000000a03251eb3fb3be406d8c8551be000000a0c642eb3f7bc640d7d27a423e'
    '000000405f34eb3feecce0a3c42f5cbe00000040fc25eb3f502b758ec2a33dbe000000c0'
    '9d17eb3fdee792a881b41f3e000000c04309eb3fe971ed21065236be00000020eefaea3f'
    'a380130e7b5b5c3e000000209decea3f0e643d3cd0bb5dbe0000006050deea3f15a7978f'
    'c58e3a3e0000002008d0ea3f3928ab238ae92f3e00000040c4c1ea3f0fd5bbf4f6d8543e'
    '000000e084b3ea3f4d7c75144c7748be000000c049a5ea3fea0e7b7cbb515b3e00000020'
    '1397ea3f13706ff50062383e000000e0e088ea3fbe8e42bef54a51be000000e0b27aea3f'
    '96440e8d65914f3e00000060896cea3fd5c5acdb3b065cbe00000020645eea3f70d9193f'
    '8c0c5abe000000204350ea3f6b3eea09dc65503e000000802642ea3f6c24df78f6055e3e'
    '000000400e34ea3fa0d457402b1b433e00000040fa25ea3fb57b8682be764b3e000000a0'
    'ea17ea3f0af4369439ad5abe00000020df09ea3fb353524b0b38463e00000000d8fbe93f'
    '6624c58f9b6f38be00000020d5ede93f44f3d322478353be00000060d6dfe93f2235c31a'
    '53bc5d3e00000000dcd1e93f1dffbdea0cfc403e000000e0e5c3e93f730ed3af635e58be'
    '000000e0f3b5e93f6a222fa5f9e843be0000002006a8e93f8d69b8ec365b51be00000080'
    '1c9ae93f9de8b4f22bb6483e00000020378ce93ffb889a7c4c41443e00000000567ee93f'
    '415701daba135dbe000000e07870e93f06ceda5f47b9513e00000000a062e93f94a06c95'
    '8587513e00000040cb54e93f1d4c1601575b5d3e000000c0fa46e93f67373be6e7844fbe'
    '000000402e39e93fa9c27ce5a3ed343e000000e0652be93f44b5758ca066573e000000c0'
    'a11de93f87d0d137b12a5ebe00000080e10fe93f20dc53a9f3a15f3e000000802502e93f'
    '69f3d3dbdbd6473e000000a06df4e83f89e99b1c0a2b5ebe000000a0b9e6e83f6ad7933c'
    '18865c3e000000e009d9e83f9afc82219eaa41be000000205ecbe83f9d53b3e6190d53be'
    '00000060b6bde83fc38ce54974b33bbe000000a012b0e83f8febcfa712c4563e00000000'
    '73a2e83f19bc528db829143e00000060d794e83f6c2cc34d4c6048be000000c03f87e83f'
    '568e860ce54e56be00000000ac79e83f28e8ae56d82f5e3e000000601c6ce83fecb8ea7c'
    '6533493e000000c0905ee83fdcdad478257f4fbe000000000951e83f8082cd0ca2e7313e'
    '000000408543e83f154eba347780323e000000800536e83f5a9770a6e5ee53be000000a0'
    '8928e83fb2771bf60aa243be000000a0111be83f3b64e613e55f5e3e000000c09d0de83f'
    'e894cc82f9f15fbe000000a02d00e83f5d9c0c8ae7b042be00000060c1f2e73f016fa122'
    'a09e5d3e0000002059e5e73f51d48cc36369503e000000c0f4d7e73f71bc0299d703453e'
    '0000004094cae73fc0a3f2deed983d3e000000a037bde73fb0ab49edffc1243e000000e0'
    'deafe73f70beb0e367c440be000000008aa2e73f3c199faf6cff5dbe000000e03895e73f'
    'b6f64cb7d08e25be000000a0eb87e73fc727911db05f343e00000040a27ae73f1dc22810'
    'bd1946be000000a05c6de73fe4b5b07ca2f1403e000000e01a60e73fadc41b2bbbe832be'
    '000000e0dc52e73f4ef639687bf5413e000000c0a245e73f7e1f12c40ac452be00000060'
    '6c38e73f722d85d66b4e5cbe000000c0392be73ff790d6918ff857be000000e00a1ee73f'
    '59217a62d52544be000000c0df10e73f3340a5507e2b423e00000060b803e73f915f0b3b'
    '57385d3e000000e094f6e63fa228d68490f051be0000000075e9e63f94886d30834d41be'
    '000000e058dce63faa24bf30ca5046be0000008040cfe63f698d62d407b05dbe000000c0'
    '2bc2e63f7be5aaa279d231be000000c01ab5e63f7edf0e864a4c2dbe000000800da8e63f'
    '419355f3987e5fbe000000e0039be63f9e8985a811205cbe000000e0fd8de63f376ddc2b'
    '824a223e000000a0fb80e63fb9d12ac156cf40be00000000fd73e63f59f6cd1b2d2ff5bd'
    '000000000267e63f0804f15de063563e000000c00a5ae63f680507a42fb140be00000000'
    '174de63f474cc5718b5e5f3e000000002740e63f837e4bbdd6ea423e000000a03a33e63f'
    'd28b5961d4484cbe000000c05126e63f618d536f0184543e000000a06c19e63f20413414'
    'f69a52be000000008b0ce63f87c582594f1e3ebe00000000adffe53fead451fe7a894cbe'
    '00000080d2f2e53fe1eb46fd002e553e000000a0fbe5e53f995669a471d45e3e00000060'
    '28d9e53fae18d180616b453e000000a058cce53f0b334c3029dc543e000000808cbfe53f'
    'dfdef20abda93abe000000e0c3b2e53f5892fc15379a47be000000c0fea5e53feac79292'
    '5086183e000000203d99e53f80d3b433936d5d3e000000207f8ce53fc716fd0261e92f3e'
    '000000a0c47fe53fb6ed054ab4554dbe000000a00d73e53fbb3a443d54695ebe00000000'
    '5a66e53feacf4a021be6503e00000000aa59e53f09dd9ecc035432be00000060fd4ce53f'
    '5069e21f0e505d3e000000605440e53f64e15a6cb4794abe000000c0ae33e53f87024b15'
    '711540be000000a00c27e53f01f473066be556be000000e06d1ae53f9c631b756952233e'
    '000000a0d20de53fed2b7b7c87c8de3d000000c03a01e53f174eabaf75755e3e00000060'
    'a6f4e43f6886302ed6ae593e0000008015e8e43f762a3ef384f151be000000e087dbe43f'
    '3e3e9f8301db573e000000c0fdcee43fbba7eda90f5e533e0000000077c2e43fa5668f2a'
    '51e45c3e000000c0f3b5e43f5624190518854ebe000000c073a9e43f1dcda74a4a78463e'
    '00000040f79ce43f5e02238ef24957be000000007e90e43f1502d318390f363e00000020'
    '0884e43ff3f2dc63fe005e3e000000c09577e43f092d1846d97351be000000a0266be43f'
    'aa620e8f81f248be000000e0ba5ee43f0cc47557d4aa56be000000605252e43f695fe20f'
    '71bd483e00000040ed45e43fc59e98e9970d593e000000808b39e43fe3ffd9b3bc9d473e'
    '000000202d2de43f2e4d8e3880ed5ebe000000e0d120e43f187c796f4c4b553e00000020'
    '7a14e43fb48b043112115bbe000000802508e43ff9a4fb2ec7eb483e00000040d4fbe33f'
    '1911205001b7403e0000004086efe33f2cb34d0ae81d553e000000a03be3e33f8b149c0c'
    'f6c150be00000020f4d6e33f479412c9a03f533e00000000b0cae33fa0b5e5aa8eb622be'
    '000000206fbee33f8a5e300208fc54be0000006031b2e33f5882907f05dc573e00000000'
    'f7a5e33f78af091a8b03083e000000e0bf99e33fc143064942be5dbe000000e08b8de33f'
    '24d78a5e722b3cbe000000205b81e33fb69671c6cf13173e000000a02d75e33f29e48261'
    '4cc13ebe000000400369e33faeb16eabc52c5a3e00000040dc5ce33f64c05dfe78585cbe'
    '00000040b850e33fe4b9a60b9b61513e000000809744e33faa61778553ff5f3e00000000'
    '7a38e33f8cd672f84d4f483e000000a05f2ce33fc2977e082e84523e000000804820e33f'
    'c0d0d673df3e50be000000803414e33fa156140cad725fbe000000a02308e33fd5a4a183'
    'cc655ebe000000e015fce23f90735a85386450be000000400bf0e23f878289a2a2223d3e'
    '000000e003e4e23f6ff6568bfda55abe00000080ffd7e23f9a11db523d2e3a3e00000060'
    'fecbe23fc0d4dde2696458be0000004000c0e23f10bf016b9d2b353e0000004005b4e23f'
    'df1c7ab0da5c5c3e000000800da8e23f68f8b5c7b36856be000000c0189ce23f62df5e18'
    '663d56be000000002790e23fcce129f7a0a9593e000000803884e23f27c7336489cc43be'
    '000000004d78e23f312678410c7530be000000a0646ce23fb71149910e2958be00000040'
    '7f60e23fe173cc3dcd6942be000000009d54e23f70bf512798695abe000000c0bd48e23f'
    'fbb9484200db4dbe00000080e13ce23f2ff85cf3711b563e000000600831e23f2d1a488e'
    'b98f513e000000603225e23fdc6eb95ac5af5fbe000000405f19e23f1139948019f807be'
    '000000408f0de23f6c2d6f388bba54be00000040c201e23fac6496f215b85ebe00000020'
    'f8f5e13f9033f0640c325e3e0000002031eae13f96f67f74a5f05e3e000000406ddee13f'
    '51eb9c3e278d5fbe00000020acd2e13f5eb5e04a21aa5f3e00000020eec6e13f5e9a5628'
    '4f8a593e0000002033bbe13f073eb3540a13463e000000207bafe13f78104f0293bf4dbe'
    '00000000c6a3e13ffa3b78b04892413e000000e01398e13f36b8022fb7024e3e000000c0'
    '648ce13fd4c9de284f06093e00000080b880e13f06f4cb45461f5b3e000000400f75e13f'
    '4c96d903790a5b3e000000006969e13f2b885b8b868023be000000a0c55de13ff8d6ba73'
    'a4fcf1bd000000202552e13f9c768553768d5e3e000000a08746e13f6bdc7616081d573e'
    '00000020ed3ae13f7f1cc4a8258a59be00000060552fe13ff0aae1c47752433e000000a0'
    'c023e13fe13836407caa21be000000c02e18e13f2b097a556b11d0bd000000c09f0ce13f'
    '669f777dba614a3e000000c01301e13f45c6092b6e585dbe0000002004eae03f46ad2cea'
    '7ca95a3e0000002000d3e03f540e1923a7f1503e000000a007bce03fa6a579139d6151be'
    '000000601aa5e03f4a3d6a9219f05c3e000000a0388ee03f5843c2a81e24353e00000020'
    '6277e03f7a7e3124fa2c513e000000009760e03f74f29cfdf3be55be00000000d749e03f'
    '9db489366dd236be000000402233e03fc4f62ef708cd54be000000a0781ce03f2d2d7023'
    'bf0059be00000000da05e03f4cc1593f0bd8573e000000408ddedf3f6d7667add4fa57be'
    '000000407cb1df3fe74a4f643be41e3e000000408184df3fd2343290861a503e00000040'
    '9c57df3f09e5e9af3e7c26be00000000cd2adf3f0bdadfb79b1448be0000004013fede3f'
    '5e30943ba74e5f3e000000806fd1de3f61da955d98c155be00000000e1a4de3fc9606940'
    '199ad9bd000000006878de3f39352fd2780c473e00000080044cde3f35c5ee8332123ebe'
    '00000040b61fde3fcbfffb3d717d4bbe000000407df3dd3fe0e41b7e8f8f5bbe00000040'
    '59c7dd3f87e8da46580435be000000804a9bdd3f49cc6eed45005fbe00000080506fdd3f'
    '3c889e2eda15293e000000806b43dd3f32cbbcf0c9684a3e000000809b17dd3f79c7bf9b'
    '6aa254be00000000e0ebdc3fab33ea7cb7c6433e0000004039c0dc3f06fd40e7c226553e'
    '00000040a794dc3f1aebad9e8d6d39be000000c02969dc3f5af9a8f0b20a5cbe00000080'
    'c03ddc3f3b69e26ee692093e000000c06b12dc3f81b5c65ab63428be000000402be7db3f'
    'ff26c28ca696353e00000000ffbbdb3fbb742af913583c3e00000000e790db3fc0649647'
    '44d650be00000000e365db3f5b9704508f2555be00000000f33adb3f9431b2e4078458be'
    '000000c01610db3f0a4d5de6267c523e000000804ee5da3fd6dd4f81a262593e00000040'
    '9abada3f13099de14e2f56be00000080f98fda3f06d0cf43ebfd4cbe000000406c65da3f'
    '4e0a6f68a8475e3e000000c0f23ada3f10d4007299115e3e000000c08c10da3f6e26d2ab'
    'd1e45e3e000000403ae6d93f2c8f6f39fbbf4d3e00000000fbbbd93fdd252be3543a5c3e'
    '00000040cf91d93f35401e43257945be00000080b667d93fd33ded7b1dc6403e00000000'
    'b13dd93f659344d71964303e00000080be13d93f91e74617fcfc563e00000040dfe9d83f'
    '8b02a9f3b94150be000000c012c0d83f500c84560ae226be000000405996d83f02317619'
    '66f451be00000080b26cd83f7cde32708a294dbe000000801e43d83fab9fb3deeb6143be'
    '000000409d19d83fe0cb015db32554be000000802ef0d73fa99ae93ca86f143e00000080'
    'd2c6d73fb962a2d1691a5abe000000c0889dd73f36c20686083a423e000000805174d73f'
    'b7e1d18f636a5a3e000000c02c4bd73f6a4591e4cac1423e000000401a22d73fd7a69944'
    '9aa6363e000000001af9d63f94df3752028f0fbe000000002cd0d63f6e2c48b6f7bc5abe'
    '0000000050a7d63f61fd1919e2ad57be00000000867ed63f4d997aaabd3f3fbe00000000'
    'ce55d63f4c01db6750c5333e00000000282dd63fb75628a8d10914be000000c09304d63f'
    '0d306a1e99d8553e0000008011dcd53f5cbd2212c0bf35be000000c0a0b3d53fd3c28d6e'
    '794d5d3e00000000428bd53fe6ace4e0037351be00000080f462d53fa8e006b30fdf5e3e'
    '000000c0b83ad53f54bc746559e85e3e000000808e12d53f072290ea88615f3e000000c0'
    '75ead43f791d919f3517513e000000806ec2d43f9773c7f943165bbe00000040789ad43f'
    '5892fc15379a473e000000809372d43fd94da0d5566e42be000000c0bf4ad43ff54240e0'
    'c6f7563e00000040fd22d43fc8f28b1d10885d3e000000004cfbd33feedda888541431be'
    '000000c0abd3d33f475e3b3e721b5dbe000000401cacd33f595dabc22bb0313e000000c0'
    '9d84d33f9e4be3d42fcb513e00000040305dd33ffb047217d78c2bbe00000080d335d33f'
    '828cd3fce15643be00000080870ed33fcc4af56424624ebe000000004ce7d23fd97579aa'
    'fec05d3e0000008021c0d23f3fab6d51a3ff50be000000400799d23f1373fb2ba274563e'
    '000000c0fd71d23f99fc4905295d383e000000c0044bd23f7330b6556d0c50be00000000'
    '1c24d23f3a95913f7799383e000000c043fdd13f713f54a1ab8734be000000c07bd6d13f'
    '7c86c84edca2f63d00000000c4afd13fbbe32843c0d9413e000000801c89d13f84da1c2e'
    '87dd3b3e000000408562d13fae31534b8e1253be00000000fe3bd13f64c1aeb998ac52be'
    '000000c08615d13f16131ed9300635be000000801fefd03f2cc1ac7c19523f3e00000040'
    'c8c8d03fb777e2bcc0303d3e0000000081a2d03f7d44632a771354be00000080497cd03f'
    'b583c4faec7257be000000c02156d03f70a5b836bdd44fbe000000c00930d03ff705e5ba'
    '880345be00000080010ad03fadae353efc3054be0000008011c8cf3fac7574706e80383e'
    '000000803f7ccf3ffc1718c9eacc40be000000808c30cf3fe9d505aeb81949be00000080'
    'f8e4ce3fe6c96cae940b53be000000008399ce3f8e3efe1e7e74573e000000002d4ece3f'
    'bfd978da08a659be00000000f502ce3f2e2cbe8aad354a3e00000000dcb7cd3f0d459514'
    'cc7208be00000080e16ccd3fa00bee86a0594fbe000000000522cd3f88a81ce8c302543e'
    '0000000047d7cc3fb924443bc3fd5d3e00000080a78ccc3f6cb505d3a62d203e00000000'
    '2642cc3f10699a391c2a48be00000080c2f7cb3f38797f74727358be000000807cadcb3f'
    'a046c26f3dd8503e000000005563cb3fe59b9eeebd355cbe000000804a19cb3fbcc01684'
    '4f6d543e000000005ecfca3f8ff0f74976da563e000000008f85ca3fe20dc35d0c395f3e'
    '00000000de3bca3fb683059569415ebe0000008049f2c93f53156333b1ae523e00000000'
    'd3a8c93fa69587de04a559be00000000795fc93f1ef46b07fe22513e000000803c16c93f'
    'e7c8142964d03d3e000000001dcdc83fa3ec303aaab421be000000801a84c83f5066a9b2'
    '445457be00000080343bc83fcbc07623c7742abe000000806bf2c73f53b6a0d8b68151be'
    '00000000bfa9c73f82782532b4784abe000000002f61c73fd98bee1e9dfe1bbe00000080'
    'bb18c73fc43c600cc9fd363e0000008064d0c63fcfb828372e541ebe000000802988c63f'
    '67409ac70f385c3e000000000b40c63f69ac9ef6840a553e0000008008f8c53fa480a7b7'
    '24925d3e0000008022b0c53f1efb9dad2f2455be000000005868c53fbe189b65a3fd4bbe'
    '00000080a920c53f3136ee6669d757be0000008016d9c43f1928c61ef727243e00000080'
    '9f91c43f6953c2de315443be00000000444ac43f4bfcaca8e8623cbe000000000403c43f'
    'ab3e1dcf9fa2fbbd00000080dfbbc33feaa3ab79c8b7f1bd00000080d674c33fda86d1b8'
    'cf3051be00000080e82dc33f52f1749db685223e0000000016e7c23fa97cae50203950be'
    '000000805ea0c23f2ed9ae6c243953be00000000c259c23f4e03b59c310e51be00000080'
    '4013c23f78d3c412430b54be00000080d9ccc13f068741cc7a88593e000000008e86c13f'
    '06411f92678e52be000000805c40c13f1e44693951805d3e0000000046fac03f5bef41d9'
    '79905f3e000000804ab4c03fb2813e5a917656be00000000696ec03fe7af669dfb434dbe'
    '00000000a228c03f62a1920a94f352be00000000eac5bf3fe5979820379e523e00000000'
    'c53abf3f7bbd58843128583e00000000d5afbe3fb8b4d8b84a6b48be000000001825be3f'
    'b6b7a3e0d2af5b3e00000000909abd3f0e71f22b2b3b383e000000003c10bd3fb76aeb73'
    '8dd756be000000001b86bc3ff5afce325adc32be000000002efcbb3fb74ce0bea4714abe'
    '000000007472bb3f7795ae352f14383e00000000eee8ba3fb4daadcbf09054be00000000'
    '9a5fba3f1411ce95717c593e000000007ad6b93f780f7c6d2dbc3a3e000000008d4db93f'
    '82a74128bc6c56be00000000d2c4b83fc629d46ef9ff3cbe000000004a3cb83fbb9fa4e4'
    '642955be00000000f4b3b73f1ed8932172fa42be00000000d02bb73f22c170dd8c7a523e'
    '00000000dfa3b63f548a1003930345be000000001f1cb63f5479ff304058563e00000000'
    '9294b53f0c46dddeb52254be00000000360db53f459f0f95f61353be000000000b86b43f'
    'b1dc2c58396d503e0000000012ffb33fa6d3167219a74a3e000000004a78b33ffd23a457'
    '9f9b5a3e00000000b4f1b23f418b137a18b450be000000004e6bb23fead7bf2f3ea5233e'
    '0000000019e5b13fcb3c9118c15f463e00000000155fb13f214ea27e4328043e00000000'
    '41d9b03f779c6d7c1ef6593e000000009e53b03f44fd4e11b7ca4c3e00000000569caf3f'
    '57f67717652f553e00000000d291ae3f6ab817c3e0615abe00000000ac87ad3ffb4e66b7'
    '4ef641be00000000e67dac3fa9033d5da007083e000000008074ab3feb383c74e12637be'
    '00000000786baa3ff153a20636d65a3e00000000d062a93f1b545fa37a185a3e00000000'
    '885aa83f46e4864b508150be000000009c52a73fcfca89258a93523e00000000104ba63f'
    'f2116bafcd5434be00000000e243a53fef6f5097c5de5fbe00000000103da43fd97d5fe7'
    'd38d38be000000009c36a33f329613a47751eabd000000008630a23f1e6f2d35d65a56be'
    '00000000cc2aa13fb79e4477c7d550be000000006e25a03f78da78742447403e00000000'
    'dc409e3f7fef9cf50a9d53be0000000090379c3f3cd41115c8c2533e00000000002f9a3f'
    '3cff8b9be1b343be000000002427983fa5221eadbdf0463e000000000020963f56930d13'
    'a05b473e000000009419943f83f8868f0b3d51be00000000dc13923fc80d4d91354353be'
    '00000000d80e903fe7e5732d75ba22be0000000010158c3f0ed7b7c55d9c593e00000000'
    'e00d883f7e85278ac8283dbe000000001008843f287376da3d1b533e00000000b003803f'
    'f3caba77e3045fbe000000005001783f20074bdfff8b5a3e0000000040fc6f3f718ec434'
    '99cd3fbe00000000c0f65f3faf18d21aa7784cbe00000000000000000000000000000080'
)
# exp2 table: 2**(j/256) for j in [0, 256), head and tail halves
_P_EXP_TBL = bytes.fromhex(
    '000000000000f03f000000000000703bbfbc5afa1a0bf03f719f60a7b2f684bc3533fba9'
    '3d16f03fb7cdb89a29619b3c81023b146821f03fb64ec50f31bf82bc6180773e9a2cf03f'
    '5d085b53839071bcccbb112ed437f03f1ae1adee1168653c857f6ee81543f03f6ec97719'
    '1ca390bcb154f6725f4ef03f8cd0a03a79c3843c748515d3b059f03f65b475a4e2738d3c'
    '891f3c0e0a65f03f97c399577bcb95bcdef6dd296b70f03f273cb1e2df918cbc36a8722b'
    'd47bf03f008745543423833cc89b75184587f03fff84b24bbe86613ce00766f6bd92f03f'
    'd13f0a80638096bc83f3c6ca3e9ef03f366131187848913c19391f9bc7a9f03f381d3d87'
    '6cd1853c0f89f96c58b5f03f0b61dc4a2ea6983c856ce445f1c0f03fef1cd20689f9943c'
    'f747722b92ccf03f714fe216dc1e903cec5d39233bd8f03f6a313fe44dc19bbca2d1d332'
    'ece3f03f527bc527173a403cc5a9df5fa5eff03f1b0254bcb99d94bc1bd3feaf66fbf03f'
    '7bbd4ec4ed9b6bbc3e23d7283007f13fd5fd9216eb468d3c515b12d00113f13f3a9b4439'
    '10c596bcb62a5eabdb1ef13f72fb03f754a49cbccc316cc0bd2af13fc7a56cb314b551bc'
    'ab04f214a836f13ff0dc48ba8f1067bce02da9ae9a42f13f9e36f19abf2f93bc2e314f93'
    '954ef13fab44bf39e8918bbc518ea5c8985af13f0aabeeb96a40823cc2c37154a466f13f'
    '321aea823bf2583c7b517d3cb872f13f768ad7b9419081bcc0bb9586d47ef13f645aace2'
    '3f9e703cea8d8c38f98af13f6c0f97d1231091bc2f5d37582697f13f087ef185ddaa943c'
    '75cb6feb5ba3f13fe468497b4c5b8e3c1c8a13f899aff13f8092b6a485bf973cd45c0484'
    'e0bbf13f07f62e35865399bc6b1c28952fc8f13fc9f810807709903caab9683187d4f13f'
    '3c64a2006e019e3c2740b45ee7e0f13fdeb68c08d8fd96bc1dd9fc2250edf13f8cb77b02'
    '98df91bc4dce3884c1f9f13f5caf97a024f59bbcd68c62883b06f23f95844a8175c78d3c'
    '19a87835be12f23fc9aafc2c2d59933c96dc7d91491ff23feea594947ea9823cd11279a2'
    'dd2bf23f9fd67755fb348d3c3862756e7a38f23f7305c7b67eb0993c0a1482fb1f45f23f'
    '96a91c91cccf8a3c3fa6b24fce51f23fa4f4f4be55c18a3c75ce1e71855ef23f2c1bc34a'
    'a2e1933cdd7ce265456bf23fd9e9409933bd823c29df1d340e78f23f6ce7f9057c069e3c'
    '8163f5e1df84f23f7e0d3f8c3a4c9abc70bb9175ba91f23fbd1c402872cc82bce1de1ff5'
    '9d9ef23f5512adafe812863c130fd1668aabf23fa7901619435799bc90d9dad07fb8f23f'
    'a41a38d6dc0a41bc2f1b77397ec5f23f2451eba6450195bc0b03e4a685d2f23fd541db54'
    '4702903c8915641f96dff23f98e1bcfbcf169d3c562f3ea9afecf23f8323d5450fca713c'
    '6b88bd4ad2f9f23f93da2b53553c65bc15b7310afe06f33fe48231d26af4863cfdb2eeed'
    '3214f33fd1fcf3f3a359893c31d84cfc7021f33f7c04188ee79c8a3c32eaa83bb82ef33f'
    '18f3b43ce8459cbcff1664b2083cf33fa65936842127933c2dfae3666249f33fa4810893'
    '755a83bcf19f925fc556f33f28464e5cee5c8bbc3b88dea23164f33f5eb86ca044318cbc'
    'cba93a37a771f33fe2ea42bfea3a96bc4a751e23267ff33f3cb2ce9ecaf599bc66d8056d'
    'ae8cf33fbd04993c8d959ebcef40711b409af33f34298efca5a999bcf79fe534dba7f33f'
    'e3f561d636e475bcf46cecbf7fb5f33f18ff6fe2664c953ce5a813c32dc3f33fc3295d37'
    'f8ff9ebc73e1ed44e5d0f33f714c288cd0e87f3c2234124ca6def33fbc9ef01109da8a3c'
    '75511cdf70ecf33fca9b8c7b63f68abc1c80ac0445faf33ff3f956f923d097bc24a067c3'
    '2208f43f48d0f4b6f8dd8b3c2a2ef7210a16f43f7892301c69f35ebc8a460927fb23f43f'
    'dd14b3c02d4698bc97a850d9f531f43f99795fe3ddc781bcd4b9843ffa3ff43f03c00497'
    'be80883c2d896160084ef43fd080ef047a9b483c32d2a742205cf43f8e1ffb82196468bc'
    '57001ded416af43f768a64d14b949c3c37328b666d78f43f335744edf0209cbcd03cc1b5'
    'a286f43ff06290b6a3c1733cd2ae92e1e194f43fa09e495e89b283bcded3d7f02aa3f43f'
    '56bed1f362cb993cd7b76dea7db1f43ff097287fb82581bc272a36d5dabff43fe242ecaf'
    '97437d3c14c117b841cef43f5dbd0a69295e903c0dddfd99b2dcf43f33786abcdbec983c'
    'ffabd8812debf43f527a5d2e7d2595bca72c9d76b2f9f43fe35759d209b394bcee31457f'
    '4108f53f5f46b7499b247a3c4266cfa2da16f53fef93bd6985768fbcef4e3fe87d25f53f'
    '71efef438d997cbc824f9d562b34f53fad3cb11dbe7a80bc27adf6f4e242f53f7e5f2d19'
    '6d92873c0f925dcaa451f53f9be5edef9c688dbcd210e9dd7060f53ff0eb8e166efb90bc'
    'da27b536476ff53fad931d012cbb993ccfc4e2db277ef53fc4b9578a8cb990bcfdc797d4'
    '128df53fe81d9a5be195823ccc07ff27089cf53f0fe667e4cee297bc295448dd07abf53f'
    'ad4746054c32963c037aa8fb11baf53f1a3e234ca1779bbcb746598a26c9f53fa2866981'
    '1b4b3c3c938b999045d8f53f4356b4a8a7d69cbc4821ad156fe7f53f5ee68030f9a69b3c'
    '71ebdc20a3f6f53f92cfcde3ddea89bc09dc76b9e105f63f47de569b42e293bcf4f6cde6'
    '2a15f63f274cb84a3e4b9e3c85553ab07e24f63f97b4407ec18393bcfd29191ddd33f63f'
    'e564b9be1047983c20c3cc344643f63f33899d753c488cbcb78fbcfeb952f63f093ea7c9'
    'd5e39abc252255823862f63f341c598709b69bbcf63308c7c171f63f34616c5832878ebc'
    '73a94cd45581f63f653ef744ae38603c38959eb1f490f63f5d44eb9abd04883ccd3b7f66'
    '9ea0f63f5664b21334dd9bbc3e1775fa52b0f63f0e9d9a2cf5387a3cbfda0b7512c0f63f'
    '0d0bff67568972bc4576d4dddccff63f0973f1b6a97a9c3c2f1a653cb2dff63fab883c68'
    '3abe6bbce53a599892eff63fb2c81a9e74b990bc849451f97dfff63ff60e86250f3c88bc'
    '872ef466740ff73f5fa65ad444d6593c745fece8751ff73f997a8886476e81bc8ad0ea86'
    '822ff73f722cd62ca00a92bc7481a5489a3ff73f3cd5656cd9a890bcfdcbd735bd4ff73f'
    '1c6e8a61fd47903cc9674256eb5ff73fd36d3157592490bc096eabb12470f73ff8479116'
    '77789b3c3f5dde4f6980f73f2d16020ab866983cf61cac38b990f73f3eddaa62a849933c'
    '8701eb7314a1f73f2f9904ee771584bcdbcf76097bb1f73f88dc6884b5eb9bbc32c13001'
    'edc1f73fd64d16d14c129f3cf086ff626ad2f73fb4b872fbdbbd913c624ecf36f3e2f73f'
    '7e7915ba025d703c91c4918487f3f73fae1193cf117f80bc121a3e542704f83f2b976d62'
    '867c92bcd906d1add214f83f4d1d150d3764943c13ce4c998925f83fd83215d41d4c9dbc'
    'f741b91e4c36f83fd52bdf319a9b993cadc723461a47f83ffbcd41a384d688bc215b9f17'
    'f457f83fd016b2f848a75bbced92449bd968f83fbaf6d49bf8c69fbc36a431d9ca79f83f'
    'bd47dbd2d7d2853c99668ad9c78af83f3ab57cf3c294993c0f5878a4d09bf83f2a207544'
    '95539d3cdba02a42e5acf83f274b8656f1e9963c7817d6ba05bef83f6e4443fc5ecb9e3c'
    '8c44b51632cff83faae3e9325ed570bcd966085e6ae0f83fe6b2c96f4a1197bc36771599'
    'aef1f83f6c97e3a213cc853c8a2c28d0fe02f93fd23ffe85ca92953cc6ff910b5b14f93f'
    '2425582e79d69dbce22faa53c325f93f7fdb39a65f4583bce5c5cdb03737f93fbc7eb581'
    'c75f67bce5985f2bb848f93f992d7d79d6c38dbc0f52c8cb445af93f39f0a5967c4b76bc'
    'b370769add6bf93f96c8197f96a55bbc504ede9f827df93fd1851b7c5b189dbca2227ae4'
    '338ff93fec784ca2daab7c3cba07ca70f1a0f93f32e6ce91bd7391bc0dfe534dbbb2f93f'
    '18d5f64d4ed89dbc90f0a38291c4f93fbef271b0467c7c3cd5b84b1974d6f93f3382dda3'
    'be1695bc2323e31963e8f93f6e4ce678ca24783c9ef2078d5efaf93fcefaf1aacea984bc'
    '65e55d7b660cfa3f33d51c5d495993bcbbb88eed7a1efa3f0ee78bee18669c3c332d4aec'
    '9b30fa3fab36dc7d5c30963cd80a4680c942fa3f20b19f5880a79abc5d253eb20355fa3f'
    'e1418ddb6e2f9dbc5260f48a4a67fa3f66036730560f653c58b330139e79fa3fc763c5ca'
    '7ecb9b3c592ec153fe8bfa3fa915bab267f894bcbffd79556b9efa3f31fdf70ec9fa903c'
    'ba6e3521e5b0fa3f4545e9da319c883c7af3d3bf6bc3fa3fd06ce7ca34928fbc74273c3a'
    'ffd5fa3fe5b8b1b63bef973cadd35a999fe8fa3f81cc5d34cda1973cfff222e64cfbfa3f'
    'cd5e310ffcb294bc66b68d29070efb3f25e4804cf5de9bbc52899a6cce20fb3fcc56074a'
    '02dd943cfb154fb8a233fb3f08d784305e8062bcb149b7158446fb3f907cdfe93d767fbc'
    '3a59e58d7259fb3fe36dbabbdf719cbc2ac5f1296e6cfb3f6e3f8852f3a8923c475efbf2'
    '767ffb3f3bac547e4f5875bce44927f28c92fb3fc665cb5416729bbc4a06a130b0a5fb3f'
    '2e29540ed3fc9ebc1f6f9ab7e0b8fb3f056269c9d1523fbcd2c14b901eccfb3f849e2d7a'
    'd03d823c07a2f3c369dffb3f535bea6023263cbc091ed75bc2f2fb3f739c6b3fcafd9ebc'
    '3db341612806fc3f34cafba15a8a8dbc9c5285dd9b19fc3fdd4850896510813c2c65fad9'
    '1c2dfc3fd7a5c81716e596bc7ad0ff5fab40fc3f0ac683e037459b3c22fbfa784754fc3f'
    'afb59324072f913c4bd1572ef167fc3fad3c48ff4d88923c33c98889a87bfc3f595525be'
    'bb768ebcb5e706946d8ffc3f445c8048bcac713cdbc4515740a3fc3ff5080dd1bef287bc'
    '6990efdc20b7fc3fdb49e9d1cb03753c75166d2e0fcbfc3f929000860f227dbcfac35d55'
    '0bdffc3f729d82533bd88dbc74ab5b5b15f3fc3f57ff6db8e9089abc7c89074a2d07fd3f'
    '9c7a794337bc9cbc68c9082b531bfd3fee369a213656953cf2890d08872ffd3f78859d71'
    '7b489dbcd6a1caeac843fd3f14165abf53db933c87a4fbdc1858fd3f07375bd702ed823c'
    'd3e662e8766cfd3fa065814a7ae85f3c9883c916e380fd3fe8dfed8bc11e91bc7560ff71'
    '5d95fd3fbef69abb2d059a3c8532db03e6a9fd3f32b56d6900239c3c15833ad67cbefd3f'
    'e48b6b92f1769bbc60b401f321d3fd3fc318f07857da923c58061c64d5e7fd3f8fba798e'
    '52a59cbc5f9b7b3397fcfd3f5c4b184fcda591bc177d196b6711fe3f447f5cbd29b572bc'
    '29a1f5144626fe3f96147a8127b697bc12ee163b333bfe3f8bc6fd31a4f499bcf63f8be7'
    '2e50fe3f8fcca980899e833c766d67243965fe3f35b72275f83f86bc834cc7fb517afe3f'
    'e28d0cca22d5923c40b7cd77798ffe3fb154b080940891bcda90a4a2afa4fe3f93289c17'
    '239c9ebc6eca7c86f4b9fe3ff2e493222f83943cf1678e2d48cffe3f8cad11b4f3939cbc'
    '108518a2aae4fe3f8d5687a48dc6913c275a61ee1bfafe3fb0b6a486f4c79d3c2a41b61c'
    '9c0fff3f451d1865002293bc97ba6b372b25ff3f438e0dbfa5a1933c7472dd48c93aff3f'
    'de37d83e5a5a79bc40456e5b7650ff3f8ba1d82de1d3993cf84488793266ff3f3e343935'
    '7ba39f3c14be9cadfd7bff3f0a3506d012bb9dbc893c2402d891ff3f89f679a7a82e61bc'
    'd8909e81c1a7ff3f1e93a5f35348873c14d59236babdff3fb68e0915736779bcf1718f2b'
    'c2d3ff3fe779659674eb623cd9232a6bd9e9ff3fe3fd427403a6743c'
)
_P_LH = 1.4453125  # short approximation of log2(e) the reduction is scaled by
_P_LH_F32 = _unpack('<f', _pack('<I', 0x3FB90000))[0]
# log2 series: four interleaved coefficient pairs for the normal path; the
# near-one path reuses the first two pairs and adds its own
_P_C0 = _b2f(0xbf8365786dc96112)
_P_C1 = _b2f(0xbf9b0301ee241472)
_P_C2 = _b2f(0xbfb528db9f95985a)
_P_C3 = _b2f(0xbfd619b6b3841d2a)
_P_C4 = _b2f(0x3f9004f2518775e3)
_P_C5 = _b2f(0x3fa76c9bac8349bb)
_P_C6 = _b2f(0x3fc4635e486ececc)
_P_C7 = _b2f(0xbf5dabe1161bb241)
_P_C8 = _b2f(0xbfb528db9f95985a)
_P_C9 = _b2f(0x3ef2531ef8b5787d)
_P_C10 = _b2f(0x3fc4635e486ececb)
_P_C11 = _b2f(0xbdd61bb2412055cc)
_P_CH0 = _b2f(0xbfd61a0000000000)
_P_CH1 = _b2f(0xbf5dabe100000000)
# exp2 series
_P_E0 = _b2f(0x3f55d87fe78a6731)
_P_E1 = _b2f(0x3fac6b08d704a0c0)
_P_E2 = _b2f(0x3f83b2ab6fba4e77)
_P_E3 = _b2f(0x3fcebfbdff82c58f)
_P_LN2 = _b2f(0x3fe62e42fefa39ef)
_P_SHIFTER = _b2f(0x42B8000000000000)  # 1.5 * 2**44: rounds to a multiple of 2**-8
_P_INT_SHIFTER = _b2f(0x4338000000000000)  # 1.5 * 2**52: rounds to an integer
_P_MANT_SCALE = 0x77F0000000000000  # 2**896, the mantissa's exponent in the reduction
_P_TWO128 = _b2f(0x47F0000000000000)
_P_FRACTION = 0xFFFFFFFFFFFFF
_P_NAN = _b2f(0x7FF8000000000000)

_P_LOG_HI = [_b2f(_u64(_P_LOG_TBL, j * 16)) for j in range(513)]
_P_LOG_LO = [_b2f(_u64(_P_LOG_TBL, j * 16 + 8)) for j in range(513)]
_P_EXP_HI = [_u64(_P_EXP_TBL, j * 16) for j in range(256)]
_P_EXP_LO = [_u64(_P_EXP_TBL, j * 16 + 8) for j in range(256)]


def _sar32(v: int, n: int) -> int:
    """Arithmetic right shift of a 32-bit two's-complement word, as a 32-bit word."""
    v &= _M32
    if v >> 31:
        v -= 1 << 32
    return (v >> n) & _M32


def _add_high_word(bits: int, add: int) -> int:
    """Add a 32-bit word to the high word of a double's bits, wrapping (paddd)."""
    return ((((bits >> 32) + add) & _M32) << 32) | (bits & _M32)


# noinspection PyShadowingBuiltins
def pow(x: float, y: float) -> float:
    """``x ** y`` as the venue's engine computes it.

    The venue evaluates a runtime ``math.pow`` with the JVM's x86 ``Math.pow``
    intrinsic, the Intel LIBM stub: ``log2(x)`` in head and tail halves -- a
    513-bin reciprocal reduction and a degree-8 series, with an extra-precision
    variant for ``x`` within 1/16 of 1 against ``|y| >= 2**12`` -- then
    ``2**(y * log2(x))`` from a 256-entry table and a degree-5 series. Not
    correctly rounded, and neither is the platform's ``pow``. This is the
    JDK 17 stub: ``y == 2`` returns ``x * x`` and ``y == 0.5`` returns
    ``sqrt(x)`` for a non-negative ``x`` before anything else.

    MEASURED (BINANCE:BTCUSDT 30m, 423906 arguments -- 14 exponent patterns over
    prices, ratios and volumes, probe ``pow_probe``, plus crafted arguments for
    every reciprocal bin, probes ``rcp2``/``rcp3``, plus an independent holdout of
    244486 arguments on CAPITALCOM:BTCUSD 15m): exact on all of them. The
    platform's ``pow`` misses 548 of the first set.

    :param x: The base.
    :param y: The exponent.
    :return: ``x ** y``, bit-identical to the venue's.
    """
    xb = _f2b(x)
    yb = _f2b(y)
    if yb == 0x4000000000000000:
        return x * x
    if yb == 0x3FE0000000000000 and xb < 0x8000000000000000:
        return _math.sqrt(x)
    head = xb >> 48
    if 16 <= head < 32752:
        return _pow_finite(xb, head - 16, y, yb, 0, False)
    if head & 0x7FF0 == 0x7FF0:
        return _pow_x_inf_nan(x, xb, y, yb)
    if head & 0x8000:
        return _pow_x_negative(x, xb, y, yb)
    if xb == 0:
        return _pow_x_zero(x, y, yb)
    return _pow_finite(xb, 0, y, yb, 0, True)


def _pow_odd_integer(y: float, yb: int) -> bool:
    """Whether ``y`` (finite, nonzero) is an odd integer, the way the stub tells."""
    ey = (yb >> 52) & 2047
    if ey > 1075 or ey < 1023:
        return False
    if ey == 1075:
        return bool(yb & 1)
    rounded = y + _P_INT_SHIFTER
    fraction = y + (_P_INT_SHIFTER - rounded)
    return (_f2b(fraction) >> 48) & 0x7FF0 == 0 and bool(_f2b(rounded) & 1)


def _pow_x_inf_nan(x: float, xb: int, y: float, yb: int) -> float:
    if xb & _P_FRACTION:
        return 1.0 if (yb << 1) & _M64 == 0 else x + x
    if (yb << 1) & _M64 == 0:
        return 1.0
    if (yb >> 48) & 0x7FF0 == 0x7FF0 and yb & _P_FRACTION:
        return y + y
    y_negative = yb >> 63
    if xb >> 63 and _pow_odd_integer(y, yb):
        return -0.0 if y_negative else x
    if not xb >> 63:
        return 0.0 if y_negative else x
    return 0.0 if y_negative else _L_INF


def _pow_x_zero(x: float, y: float, yb: int) -> float:
    if (yb >> 48) & 0x7FF0 == 0x7FF0 and yb & _P_FRACTION:
        return y + y
    if (yb << 1) & _M64 == 0:
        return 1.0
    if _pow_odd_integer(y, yb):
        return _math.copysign(_L_INF, x) if yb >> 63 else x
    return _L_INF if yb >> 63 else 0.0


def _pow_x_negative(x: float, xb: int, y: float, yb: int) -> float:
    if xb == 0x8000000000000000:
        return _pow_x_zero(x, y, yb)
    if (yb << 1) & _M64 == 0:
        return 1.0
    ey = (yb >> 48) & 0x7FF0
    if ey == 0x7FF0:
        if yb & _P_FRACTION:
            return y + y
        if xb == 0xBFF0000000000000:
            return _P_NAN
        # +inf when |x| > 1 meets +inf or |x| < 1 meets -inf, else +0
        e = (((xb >> 48) & 0x7FF0) - 16368) & _M32
        return _L_INF if ((e ^ (yb >> 48)) & 0x8000) == 0 else 0.0
    # a finite negative base needs an integer exponent; its parity is the sign
    if ey > 17200:
        odd = False
    elif ey >= 17184:
        low = yb & _M32
        if ey > 17184:
            odd = bool(low & 1)
        elif low & 1:
            return _P_NAN
        else:
            odd = bool(low & 2)
    elif ey < 16368:
        return _P_NAN
    else:
        rounded = y + _P_INT_SHIFTER
        fraction = y + (_P_INT_SHIFTER - rounded)
        if (_f2b(fraction) >> 48) & 0x7FFF:
            return _P_NAN
        odd = bool(_f2b(rounded) & 1)
    sign = 0x80000000 if odd else 0
    head = ((xb >> 48) & 0x7FFF) - 16
    if head < 0:
        return _pow_finite(xb, 0, y, yb, sign, True)
    return _pow_finite(xb, head, y, yb, sign, False, True)


def _pow_finite(xb: int, head: int, y: float, yb: int, sign: int, denormal: bool,
                negative: bool = False) -> float:
    """``|x| ** y`` for a finite nonzero ``x``, the result's sign bit given.

    ``head`` is the high 16 bits of ``|x|`` less 16. The split of the mantissa
    into head and tail halves depends on ``|log2(x)|``, and for a negative base
    the stub derives it from ``head`` one binade lower -- kept as it is.
    """
    if denormal:
        xs = _f2b(_b2f(xb) * _P_TWO128)
        rcp = _unpack('<f', _pack('<I', _RCP[(xs >> 40) & 0xFFF]))[0]
        lane = (_unpack('<I', _pack('<f', rcp * _P_LH_F32))[0] + 0x2000) & _M32
        mant = _b2f((xs & _P_FRACTION) | _P_MANT_SCALE)
        mant_hi = _b2f(_f2b(mant) & 0xFFFFFC0000000000)
        b = _b2f((lane << 29) & 0xFFFFF80000000000)
        k = (((xs >> 48) & 0x7FF0) - 18416) >> 4
        return _pow_series(mant_hi * b, mant - mant_hi, b, k, lane, xb, y, yb, sign)

    e16 = ((head if negative else xb >> 48) & 0x7FF0) - 16368
    split = (abs(e16) + 16).bit_length() - 1
    rcp = _unpack('<f', _pack('<I', _RCP[(xb >> 40) & 0xFFF]))[0]
    lane = (_unpack('<I', _pack('<f', rcp * _P_LH_F32))[0] + 0x2000) & _M32
    mask = ((_M32 << ((split - 4) & 31)) & _M32) << 32
    mant = _b2f((xb & _P_FRACTION) | _P_MANT_SCALE)
    near_one = (head - 16351) & _M32
    if near_one <= 1:
        # |x| in [1 - 1/16, 1 + 1/16): the head half is rounded, not truncated
        mant_hi = _b2f((_f2b(mant) + 0x80000000) & mask)
        if (yb >> 48) & 0x7FF0 >= 16560:
            return _pow_near_one(mant, mant_hi, lane, near_one, xb, y, yb, sign)
    else:
        mant_hi = _b2f(_f2b(mant) & mask)
    b = _b2f((lane << 29) & 0xFFFFF80000000000)
    k = _sar32(near_one - 1, 4)
    k = k - (1 << 32) if k >> 31 else k
    return _pow_series(mant_hi * b, mant - mant_hi, b, k, lane, xb, y, yb, sign)


def _pow_series(head_b: float, mant_lo: float, b: float, k: int, lane: int, xb: int,
                y: float, yb: int, sign: int) -> float:
    """``log2(x)`` from the reduction, then ``2**(y * log2(x))``."""
    lo_b = mant_lo * b
    u = head_b - _P_LH
    k_bits = (abs(k) + 1).bit_length() - 1
    r = lo_b + u
    row = ((lane & 0xFFC000) >> 14) - 228
    log_hi = u + _P_LOG_HI[row]
    log_lo = lo_b + _P_LOG_LO[row]
    r2 = r * r
    log_hi = log_hi + float(k)
    r3 = r * r2
    q1 = _P_C4 + _P_C0 * r
    q1h = _P_C5 + _P_C1 * r
    r4 = r2 * r2
    q2 = _P_C6 + _P_C2 * r
    q2h = _P_C7 + _P_C3 * r
    s1 = q1 * r3
    s1h = q1h * r
    s2 = q2 * r3
    s2h = q2h * r
    # the fast path holds only while |y * log2(x)| stays well inside the range
    scale = ((k_bits << 4) - 15872 + ((yb >> 48) & 0x7FF0)) & _M32
    s3 = r4 * s1
    s3h = r4 * s1h
    if scale >= 624:
        return _pow_wide(scale, log_hi, log_lo, s2, s2h, s3, s3h, xb, y, yb, sign)

    y_hi = _b2f(yb & 0xFFFFFFF800000000)
    y_lo = y - y_hi
    m1 = y_hi * log_hi
    p_lo = s2 + log_lo
    m2 = y_lo * log_hi
    shifted = _P_SHIFTER + m1
    s_lo = s3 + p_lo
    s_hi = s3h + s2h
    n = _f2b(shifted) & _M32
    rounded = shifted - _P_SHIFTER
    p = s_hi + s_lo
    d = m1 - rounded
    d = d + m2
    z = y * p
    exponent = (((n << 12) & _M32) ^ sign) & 0xFFF00000
    z = z + d
    t0 = _b2f(_add_high_word(_P_EXP_HI[n & 255], exponent))
    t1 = _b2f(_add_high_word(_P_EXP_LO[n & 255], exponent))
    return _pow_exp2(z, t0, t1)


def _pow_exp2(z: float, t0: float, t1: float) -> float:
    """``t0 * 2**z`` (plus the table tail ``t1``) with the stub's summation order."""
    w = _P_LN2 * z
    z2 = z * z
    w = w * t0
    z4 = z2 * z2
    f = _P_E2 + _P_E0 * z
    fh = _P_E3 + _P_E1 * z
    w = w + t1
    g = (f * z4) * t0
    gh = (fh * z2) * t0
    return ((g + w) + gh) + t0


def _pow_wide(scale: int, log_hi: float, log_lo: float, s2: float, s2h: float, s3: float,
              s3h: float, xb: int, y: float, yb: int, sign: int) -> float:
    """``|y * log2(x)|`` far from the fast path: tiny, huge, or near the limits."""
    if scale >> 31:
        if ((scale + 384) & _M32) >> 31:
            return 1.0
        p = (s3 + (s2 + log_lo)) + (s3h + s2h)
        ln2 = -_P_LN2 if sign else _P_LN2
        return (-1.0 if sign else 1.0) + (log_hi * y + y * p) * ln2
    if scale >= 752:
        return _pow_out_of_range(xb, y, yb, sign)

    # log2(x) re-split so that y's head times it is exact
    p = (s3h + s2h) + (s3 + (s2 + log_lo))
    log2x = log_hi + p
    carry = log_hi - log2x
    l_hi = _b2f(_f2b(log2x) & 0xFFFFFFFFF8000000)
    y_hi = _b2f(yb & 0xFFFFFFFFF8000000)
    l_tail = log2x - l_hi
    p = p + carry
    y_lo = y - y_hi
    m1 = y_hi * l_hi
    p = p + l_tail
    z = y * p
    m2 = y_lo * l_hi
    shifted = _P_SHIFTER + m1
    n = _f2b(shifted) & _M32
    rounded = shifted - _P_SHIFTER
    z = z + m2
    z = z + (m1 - rounded)
    if (_f2b(rounded) >> 48) & 0x7FFF > 16529:
        return _pow_out_of_range(xb, y, yb, sign)
    # the binary exponent goes in two halves: one into the table value, one
    # into a separate scale factor, so neither overflows on its own
    half = _sar32(_sar32(n, 8), 1)
    rest = (_sar32(n, 8) - half) & _M32
    exponent = (((half << 20) & _M32) ^ sign) & _M32
    factor = _b2f((((rest << 4) + 16368) & 0xFFFF) << 48)
    return _pow_scaled(z, n & 255, exponent, factor)


def _pow_scaled(z: float, j: int, exponent: int, factor: float) -> float:
    t0_bits = _add_high_word(_P_EXP_HI[j], exponent)
    t0 = _b2f(t0_bits)
    t1 = _b2f(_add_high_word(_P_EXP_LO[j], exponent))
    w = _P_LN2 * z
    z2 = z * z
    w = w * t0
    z4 = z2 * z2
    f = _P_E2 + _P_E0 * z
    fh = _P_E3 + _P_E1 * z
    w = w + t1
    g = (f * z4) * t0
    gh = (fh * z2) * t0
    s = (g + w) + gh
    result = (s + t0) * factor
    if (_f2b(result) >> 48) & 0x7FF0:
        return result
    # a subnormal result: redo the last product with the table value split, so
    # the scaling rounds once
    shift = -31 - (((((_f2b(factor) >> 48) & 0x7FF0)
                     + ((t0_bits >> 48) & 0x7FF0) - 16368)) >> 4)
    if shift <= 0:
        mask = _M32
    elif shift > 20:
        return result
    else:
        mask = (_M32 << shift) & _M32
    t0_hi = _b2f(t0_bits & (mask << 32))
    t0_lo = (t0 - t0_hi) + s
    return t0_hi * factor + t0_lo * factor


def _pow_out_of_range(xb: int, y: float, yb: int, sign: int) -> float:
    if (yb >> 48) & 0x7FF0 == 0x7FF0:
        return _pow_y_inf_nan(xb, y, yb)
    e = (((xb >> 48) & 0x7FF0) - 16368) & _M32
    if ((yb >> 48) ^ e) & 0x8000:
        return -0.0 if sign else 0.0
    return -_L_INF if sign else _L_INF


def _pow_y_inf_nan(xb: int, y: float, yb: int) -> float:
    if _f2b(_b2f(xb) - 1.0) >> 48 == 0:
        return _P_NAN
    if yb & _P_FRACTION:
        return y + y
    e = (((xb >> 48) & 0x7FF0) - 16368) & _M32
    return _L_INF if ((e ^ (yb >> 48)) & 0x8000) == 0 else 0.0


def _pow_near_one(mant: float, mant_hi: float, lane: int, near_one: int, xb: int, y: float,
                  yb: int, sign: int) -> float:
    """|x| within 1/16 of 1 against |y| >= 2**12: ``log2(x)`` in extra precision."""
    b = _b2f((lane << 29) & 0xFFFFF80000000000)
    k = float((((near_one + 16351) & _M32) >> 4) - 1022)
    lo_b = (mant - mant_hi) * b
    u = mant_hi * b - _P_LH
    r = lo_b + u
    row = ((lane & 0xFFC000) >> 14) - 228
    acc = k + _P_LOG_HI[row]
    acc_tail = 0.0 + _P_LOG_LO[row]
    h4 = _P_CH0 * u
    h0 = _P_CH1 * u
    h6 = _P_CH0 * lo_b
    h1 = _P_CH1 * lo_b
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
    p0 = _P_C0 * r + _P_C4
    p0h = _P_C1 * r + _P_C5
    p5 = _P_C10 + _P_C8 * r
    p5h = _P_C11 + _P_C9 * r
    r3 = r * r2
    r4 = r2 * r2
    p0 = p0 * r3 * r4
    p0h = p0h * r * r4
    p5 = p5 * r3
    p5h = p5h * r
    l_hi = _b2f(_f2b(acc) & 0xFFFFFFFFF8000000)
    p5 = p5 + h6
    acc = acc - l_hi
    q = p5 + p0
    qh = p5h + p0h
    e = (((_f2b(l_hi) >> 48) & 0x7FF0) - 16368) & _M32
    ey = (yb >> 48) & 0x7FF0
    if ey == 0x7FF0:
        return _pow_y_inf_nan(xb, y, yb)
    if ((ey + e) & _M32) >= 16576:
        if (((yb >> 48) ^ (_f2b(l_hi) >> 48)) & 0x8000) == 0:
            return -_L_INF if sign else _L_INF
        return -0.0 if sign else 0.0
    y_hi = _b2f(yb & 0xFFFFFFFF00000000)
    q = q + qh
    y_lo = y - y_hi
    acc = acc + q
    m1 = y_hi * l_hi
    m2 = y_lo * l_hi + y * acc
    shifted = _P_SHIFTER + m1
    n = _f2b(shifted) & _M32
    rounded = shifted - _P_SHIFTER
    z = (m1 - rounded) + m2
    if (_f2b(rounded) >> 48) & 0x7FFF > 16529:
        return _pow_out_of_range(xb, y, yb, sign)
    # unlike the wide path, the stub splits the exponent with logical shifts
    half = (n >> 8) >> 1
    rest = ((n >> 8) - half) & _M32
    factor = _b2f((((((rest + 1023) << 20) & _M32) | sign) << 32))
    return _pow_scaled(z, n & 255, (half << 20) & _M32, factor)


#: The pure-Python implementations. ``_native_math``, when built, rebinds the public
#: names above to compiled twins of these (bit-identical, see that module); this keeps
#: the originals reachable for the tests that hold the two paths against each other.
PYTHON_IMPLEMENTATIONS = {'cos': cos, 'sin': sin, 'exp': exp, 'log': log, 'log10': log10,
                          'pow': pow}

if not _os.environ.get('PYNE_NO_NATIVE_MATH'):
    try:
        from . import _native_math  # noqa: F401 -- importing it installs the twins
    except ImportError:
        pass
