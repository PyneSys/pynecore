"""Runtime-exact transcendental functions for Pine-compatible execution.

Implements the cos/sin/exp/log algorithms that HotSpot JVMs before JDK 19 use
for ``Math.cos``/``Math.sin``/``Math.exp``/``Math.log`` on x86-64 (the Intel
LIBM table-driven methods), so per-bar runtime results are bit-identical to the
venue engine. None of them is correctly rounded, and ``log`` is the one where
that shows: a correctly rounded logarithm disagrees with the venue whenever the
argument comes within about 1.2% of 1.

This is an independent implementation written from the published algorithm
description: argument reduction against a fixed-point table of 2/pi (huge
arguments) or N = round(x * 32/pi) (normal range), a 64-entry split-precision
table of cos/sin values at multiples of pi/32, and a shared polynomial
reconstruction; ``log`` reduces instead against a 129-bin table of -ln(B) for
the bins an approximate hardware reciprocal selects, and finishes with a
degree-7 ``log1p`` series. The numeric tables are the published algorithm constants
(mathematical values, reproduced as data). Correctness is established by
bit-comparison against venue-engine oracles over hundreds of thousands of
arguments, including huge, tiny, denormal-result and overflow/underflow
boundary regions.
"""
import math as _math
from struct import pack as _pack, unpack as _unpack

__all__ = ['cos', 'sin', 'exp', 'log']

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
_L_ULP23 = 2.0 ** -23
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

    The reciprocal instruction fills a second lane from a denormal, whose
    reciprocal is an infinity that contributes nothing, and its own lane can never
    overflow because the mantissa it reads is a float32 in ``[1, 2)`` — both are
    folded into the closed forms below.

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

    # B, the bin's reciprocal. The instruction that answers here is an
    # APPROXIMATE reciprocal: it indexes an 11-bit mantissa table and returns the
    # reciprocal of that interval's midpoint at 12 significant bits, so it leans
    # off 1/mantissa by up to 1.5 * 2**-12 -- exactly the error the instruction is
    # specified to. The added half-bin then rounds the answer onto the table's
    # grid, and a lean that carries across that boundary picks the neighbouring
    # bin. Dividing here instead costs the last bit on roughly one argument in
    # 20000.
    mantissa = (bits >> 29) & 0x7FFFFF
    midpoint = 1.0 + (((mantissa >> 12) << 12) + 0x800) * _L_ULP23
    rcp = _unpack('<I', _pack('<f', 1.0 / midpoint))[0]
    lane = ((((rcp + 0x800) >> 12) << 12) + 32768) & _M32
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
