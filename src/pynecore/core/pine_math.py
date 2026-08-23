"""Runtime-exact transcendental functions for Pine-compatible execution.

Implements the cos/sin/exp algorithm that HotSpot JVMs before JDK 19 use for
``Math.cos``/``Math.sin``/``Math.exp`` on x86-64 (the Intel LIBM table-driven
method), so per-bar runtime results are bit-identical to the venue engine.

This is an independent implementation written from the published algorithm
description: argument reduction against a fixed-point table of 2/pi (huge
arguments) or N = round(x * 32/pi) (normal range), a 64-entry split-precision
table of cos/sin values at multiples of pi/32, and a shared polynomial
reconstruction. The numeric tables are the published algorithm constants
(mathematical values, reproduced as data). Correctness is established by
bit-comparison against venue-engine oracles over hundreds of thousands of
arguments, including huge, tiny, denormal-result and overflow/underflow
boundary regions.
"""
import math as _math
from decimal import Decimal as _Decimal, localcontext as _localcontext
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

# Reduction table: for each of the 128 bins of the mantissa [1, 2) the double
# ``inv`` closest to 1 / (bin midpoint) and ``-ln(inv)`` as a head/tail double
# pair. The logs are taken of the ROUNDED ``inv``, so the table carries no
# approximation of its own — the only error left in the reduction is the one the
# series below makes.
_LOG_TBL = bytes.fromhex(
    '20e01fe01fe0ef3fa00bb1a20af06f3f57d3a6d51a82023c12fa01aa1ca1ef3f690a815f'
    '47dc873f721016bc4449273cb5dba0ac1063ef3f84a54643a4ce933f009d1548ad6528bc'
    'b50a2344f625ef3f9a91af27c09f9b3f86dc2992e60a39bc028e45f8c7e9ee3f7fd92389'
    'd9b0a13f4162dd44744d47bceb01ba7a80aeee3fd3e4c8af5b8aa53f406ec56985abfcbb'
    'e45097a51a74ee3ff2e3c80e835ca93fe917a4001db43e3c731adc79913aee3f560bdb8a'
    '6b27ad3f53ff954cf178403c1ee0011ee001ee3f71e498359875b03fdc229e99d206503c'
    'ca1da0dc01caed3f17140a2ff653b23f81d9014ed3f6213c8a7f1e23f292ed3fee46a6be'
    'dc2eb43f9b345336581155bcb2727580ac5ced3fc45037a95806b63fd336841d8b104fbc'
    '1a5bfca32c27ed3fd0127b6d76dab73fa27d4d6440224a3cc6bf445c6ef2ec3fae332046'
    '42abb93f8e4e181c9e094abce7cb01966dbeec3fa0edb02bc878bb3f3cb0f9610eef53bc'
    '428afb5a268bec3f5db36cd61343bd3ffa901d95dd905b3c86490dd19458ec3fa46211c0'
    '300abf3f9b75b7b864be483c1ca02e39b526ec3f6f59ca121567c03f679b47819bf352bc'
    '8b8d86ee83f5eb3fac4267848547c13f3af8d3f10944393c7b3e8865fdc4eb3fcd5a0a19'
    '6f26c23f77617f0e84ab6dbc23ff182b1e95eb3fd57fe418d703c33f4782651fe75a6bbc'
    '05eebee3e265eb3f2ac6ecb0c2dfc33ff4f7138c2ba66b3cce06d84a4837eb3fe5559af3'
    '36bac43faa983c437e766fbca422d9314b09eb3f852098d93893c53f1994baaa6ed1683c'
    '5e90947fe8dbea3f51ad7242cd6ac63f65513d9c1c2049bcfdeb872f1dafea3fa33740f5'
    'f840c73f26b3579dbfd9563c59e13051e682ea3fe95743a1c015c83f9efac5f8b741513c'
    '4a8a68074157ea3f416d88de28e9c83f40626ab99e58423ca01cc5872a2cea3f85fb7d2e'
    '36bbc93fe783ffc1391455bc1aa0011aa001ea3f192f88fcec8bca3feb398c91378c5abc'
    '2d686b179fd7e93fa6b58f9e515bcb3f5f1ee623805d6dbcda1055ea24aee93fc2188c55'
    '6829cc3fac24e03a8e10363cffc08e0d2f85e93fddc5094e35f6cc3f96b6557da039633c'
    'ae77e30bbb5ce93f7becaba0bcc1cd3ffb6d31338a695c3ce62c9b7fc634e93f605aaa52'
    '028cce3f35917f7374c03dbcd59001124f0de93f377b4b560a55cf3f3de72f20093a61bc'
    '3f37f17a52e6e83f1d50ad456c0ed03fadfef68f56b963bc3aff6280cebfe83f0d59cd5f'
    'b871d03f40efbdfc838b603c9c8901f6c099e83f4bb79a576bd4d03fc95e1e0e641f723c'
    'b992c0bc2774e83fb0a893028736d13f94a4db31c56c683c140678c2004fe83f6f23d42d'
    '0d98d13fb92e1b4f2e2c70bca0a482014a2ae83ff3a2489efff8d13f0d961834bf3f69bc'
    '061860800106e83f3a76df106059d23f3cbd0eaed8ee49bc1d4f5a5125e2e73f259db83a'
    '30b9d23f51ab14f6d75a58bc7c012e92b3bee73f854154c97118d33f6f36818959a36ebc'
    '8b39b66baa9be73f5cd8bf622677d33f88a0fb89752a603c0dc69a110879e73f10f7c1a5'
    '4fd5d33fcdd978e56836553c6d7501c2ca56e73f13e8042aef32d43f0692b5e2623268bc'
    '8dfe41c5f034e73fd00940800690d43f6ddfc5070dff4bbc097c9c6d7813e73f69026032'
    '97ecd43f5edc77d987aa71bc1760f21660f2e63f63d2adc3a248d53fee4618bfe78c75bc'
    '61c88126a6d1e63fe2cff4b02aa4d53f3d9aee7dcb6b7cbc3d1aa30a49b1e63fd493a770'
    '30ffd53fb7377e7d076370bcc0d0883a4791e63ff2e10373b559d63fc783fb8eafb01d3c'
    '1a6701369f71e63f3d943522bbb3d63f4d782633a957793ca34a3b854f52e63f369278e2'
    '420dd73f543914f79be97e3cdec08ab85633e63fcfdb39124e66d73ff8daf5645d6d6dbc'
    '94ae3168b314e63fbfaf370adebed73f5b1a80b93c7876bcfc2d293464f6e53f95d4a01d'
    'f416d83fe48fb45fc36d773ca5e2ecc367d8e53fa10b339a916ed83f459002d2c90077bc'
    '91fa47c6bcbae53f8bb458c8b7c5d83f6cfa05024b757d3caacc23f1619de53f3ea845eb'
    '671cd93fdfa3963beae3653c600558015680e53f59511341a372d93f278fb42df6a375bc'
    'e2527cba9763e53f6208dc026bc8d93fdf2b624911e8773cfe82bbe62547e53f95b9d564'
    'c01dda3fb8a08b6928017a3c4b05a856ff2ae53fe9d96b96a472da3ff1619fc6da29753c'
    'c5c411e1220fe53fe5b058c218c7da3fe3dee8adc782763c9b4cdd628ff3e43f5afcbd0e'
    '1e1bdb3fa64ed77d1aee50bc4c2cdcbe43d8e43f5cf33c9db56edb3fc495cd324352713c'
    'e18fa6dd3ebde43f0aad0d8be0c1db3fa899eae3e38553bc4a0176ad7fa2e43f27f015f1'
    '9f14dc3ff6397fde6868643c804801220588e43ff96fffe3f466dc3f89688b25472958bc'
    '66605934ce6de43fca7a4d74e0b8dc3feb5aef32bc5b7c3cca76c7e2d953e43f641e72ae'
    '630add3ff2402c11ceac643c4deeab30273ae43f84c6e29a7f5bdd3f963fb5071884643c'
    '51595e26b520e43f55592c3e35acdd3f04f2f2a365bc7abc66650ed18207e43fb5d50699'
    '85fcdd3f9863f999131e753c07afa5428feee33f047768a8714cde3fb952f0e0364c73bc'
    'c675aa91d9d5e33ff5619865fa9bde3fcf1e243850e47dbc552923d960bde33ff3dd40c6'
    '20ebde3f04841b14471e78bc22c87a3824a5e33f5d1e81bce539df3f73981321e200423c'
    '8e0866d3228de33fc19efe364a88df3f0084004fae18763cee45c9d15b75e33f7115f620'
    '4fd6df3f49a3d55958611bbcf82a9f5fce5de33f8aff25b1fa11e03f781721503704843c'
    '4613e0ac7946e33f3c63ceef9e38e03f4aba419ae2aa483cfa1d6aed5c2fe33f9c4526bd'
    '145fe03ffe9e4feeb835593cb6ebe9587718e33f0e454b885c85e03f18f3496e8285873c'
    '6002c42ac801e33fd214cebe76abe03f9ac0baca3629703c4bd1fea14eebe23fb8d6b9cc'
    '63d1e03fc3f3d0959511463ca0502d010ad5e23f7d499b1c24f7e03f9db19d3b44a88b3c'
    '11375a8ef9bee23ff8cc8717b81ce13f20993f560fc77d3c05c1f3921ca9e23f453d2425'
    '2042e13fec96a4b6e3e5873ca504b85b7293e23f0ea6abab5c67e13f8e7c47159cb17cbc'
    '4dcea138fa7de23f07cff50f6e8ce13f37e6f4f4baa679bc2701d67cb368e23f9ea27db5'
    '54b1e13fb54a125c0a77723cb277917e9d53e23f037067fe10d6e13f3d96473656277d3c'
    '5b601797b73ee23f4c09874ba3fae13f7632f41ef7428c3c2a12a022012ae23fecbe65fc'
    '0b1fe23f927c18c9f0246cbce65548807915e23f3439486f4b43e23f116d0fcfb8eb6bbc'
    '122001122001e23fe03034016267e23f27671c78956a78bc4cb87f3cf4ece13f8307f60d'
    '508be23fa3a9aaf4f313383cbd4a2e67f5d8e13fac4026f015afe23f5a67252932d87e3c'
    '59e01cfc22c5e13f9ddc2e01b4d2e23f55e364369dae893ce3baf2677cb1e13f46955099'
    '2af6e23f330130c6bcdc87bc9e11e019019ee13f6afea70f7a19e33f8f1297fb48636f3c'
    'db2b9083b08ae13f948932baa23ce33f1009fd66bac6813c84d61b198a77e13fa06ed3ed'
    'a45fe33f36640968d427773c0132fc508d64e13f7f7958fe8082e33f74b9a9d8f45667bc'
    'c9d5fda3b951e13ff9bd7e3e37a5e33f24118e1411ce753c2447348d0e3fe13f0632f7ff'
    'c7c7e33fd1be2570db803ebcacc0ed898b2ce13f5b2f6b9333eae33f529fec75e9664f3c'
    '2648a719301ae13feadc80487a0ce43f8927ff798b3c513c801001befb07e13fbf80df6d'
    '9c2ee43ff17ccdc211d484bca225b3faedf5e03f0abb33519a50e43fa584d27a1d7085bc'
    '1160825506e4e03fadaa333f7472e43f7e6b5ded0f933abc3a9e355644d2e03f07fca283'
    '2a94e43fb556ca888aa1723c71418b86a7c0e03f73e25669bdb5e43f72a7ee6ba0c782bc'
    'b5ec2e722fafe03f01fd393a2dd7e43f1b019c829a1a703c6083afa6db9de03fe926503f'
    '7af8e43f9deabcb1a88c86bce26575b3ab8ce03f4634bac0a419e53f7fa7e42821338a3c'
    'e2eab8299f7be03f7c9bb905ad3ae53fe294b8142c7267bcfb12799cb56ae03fce0bb454'
    '935be53f3da641e542f341bc867572a0ee59e03f91f136f3577ce53f6e7e37c4c5ea813c'
    'c56416cc4949e03f7fe8fa25fb9ce53f57e3ac2248b96bbcfc4782b7c638e03f731ce730'
    '7dbde53f448e2e354996ccbbe92977fc6428e03f23991457dedde53f85f25ed737fa403c'
    '377a51362418e03f1989d1da1efee53f83e43cde932e793c800001020408e03f6764a4fd'
    '3e1ee63f731484043692673c'
)

_L_INV = [_b2f(_u64(_LOG_TBL, j * 24)) for j in range(128)]
_L_HI = [_b2f(_u64(_LOG_TBL, j * 24 + 8)) for j in range(128)]
_L_LO = [_b2f(_u64(_LOG_TBL, j * 24 + 16)) for j in range(128)]

_L_LN2_HI = _b2f(0x3fe62e42fefa39ef)
_L_LN2_LO = _b2f(0x3c7abc9e3b39803f)
_L_THIRD = _b2f(0x3fd5555555555555)
_L_THIRD_LO = _b2f(0x3c75555555555555)
# The head/tail pair below carries the result to ~2**-74 relative (MEASURED over
# 60k random exponents: worst 2**-74.2). The bound here is that with a margin;
# a result closer than this to a rounding boundary takes the exact path.
_L_REL_ERR = 2.0 ** -71
_L_NEAR_LO = 0.9921875   # 1 - 2**-7: below this the table reduction is used
_L_NEAR_HI = 1.0078125   # 1 + 2**-7
_L_SPLIT = 134217729.0   # 2**27 + 1, Dekker's splitter
_L_INF = _b2f(0x7ff0000000000000)
_L_NAN = _b2f(0x7ff8000000000000)


def _log1p_dd(r: float) -> tuple[float, float]:
    """``log1p(r)`` as a head/tail double pair, for ``|r| <= 2**-7``."""
    # r^2 exactly (Dekker), then -r^2/2 (exact: a power-of-two scaling)
    p = r * r
    t = _L_SPLIT * r
    rh = t - (t - r)
    rl = r - rh
    pl = ((rh * rh - p) + 2.0 * rh * rl) + rl * rl
    hh = -0.5 * p
    hl = -0.5 * pl
    # r - r^2/2
    s = r + hh
    b = s - r
    e = (r - (s - b)) + (hh - b) + hl
    sh = s + e
    sl = e - (sh - s)
    # + r^3/3 — still 2**-24 of the result, so it needs the extra precision too
    c3 = p * r
    t = _L_SPLIT * p
    ph = t - (t - p)
    plo = p - ph
    c3l = ((ph * rh - c3) + ph * rl + plo * rh) + plo * rl + pl * r
    th = c3 * _L_THIRD
    t = _L_SPLIT * c3
    ch = t - (t - c3)
    cl = c3 - ch
    t2 = _L_SPLIT * _L_THIRD
    dh = t2 - (t2 - _L_THIRD)
    dl = _L_THIRD - dh
    tl = ((ch * dh - th) + ch * dl + cl * dh) + cl * dl + c3l * _L_THIRD + c3 * _L_THIRD_LO
    s = sh + th
    b = s - sh
    e = (sh - (s - b)) + (th - b) + sl + tl
    sh = s + e
    sl = e - (sh - s)
    # the remaining terms stay below 2**-30 and fit a plain double
    tail = c3 * r * (-0.25 + r * (0.2 + r * (-1.0 / 6.0 + r * (
        1.0 / 7.0 + r * (-0.125 + r * (1.0 / 9.0 + r * (-0.1)))))))
    s = sh + tail
    b = s - sh
    e = (sh - (s - b)) + (tail - b) + sl
    sh = s + e
    return sh, e - (sh - s)


def _log_exact(x: float) -> float:
    """Correctly rounded ``ln(x)`` the slow way, for the rare boundary case."""
    with _localcontext() as ctx:
        ctx.prec = 40
        return float(_Decimal(x).ln())


def log(x: float) -> float:
    """Correctly rounded natural logarithm.

    The venue's ``Math.log`` is correctly rounded on every argument measured
    (probe logpow, BINANCE:BTCUSDT 30m: the 8 of 86241 values where the platform
    ``math.log`` differs are all cases where the platform, not the venue, is the
    one off by an ulp), while the platform's is only within an ulp — so a
    recursive script carries that ulp into its output. This computes ln in a
    head/tail double pair and falls back to exact decimal arithmetic on the rare
    argument whose pair lands too close to a rounding boundary to decide.
    """
    if x != x:
        return x
    if x <= 0.0:
        return -_L_INF if x == 0.0 else _L_NAN
    if x == 1.0:
        return 0.0
    if x == _L_INF:
        return x

    if _L_NEAR_LO <= x <= _L_NEAR_HI:
        # ``x - 1`` is exact here (Sterbenz) and the series needs no reduction,
        # which also avoids the cancellation a table term would introduce.
        sh, sl = _log1p_dd(x - 1.0)
    else:
        m, e = _math.frexp(x)
        m += m
        e -= 1
        j = int((m - 1.0) * 128.0)
        if j > 127:
            j = 127
        inv = _L_INV[j]
        # m * inv - 1 as a head/tail pair: rounding it to a single double would
        # cap the whole reduction at 53 bits, which no series precision recovers.
        p = m * inv
        t = _L_SPLIT * m
        mh = t - (t - m)
        ml = m - mh
        t = _L_SPLIT * inv
        ih = t - (t - inv)
        il = inv - ih
        pl = ((mh * ih - p) + mh * il + ml * ih) + ml * il
        rh = p - 1.0  # exact: p is within 2**-8 of 1
        rl = pl
        s = rh + rl
        rl = rl - (s - rh)
        rh = s
        sh, sl = _log1p_dd(rh)
        # the tail of the reduction enters through log1p'(rh) = 1 / (1 + rh)
        corr = rl / (1.0 + rh)
        s = sh + corr
        b = s - sh
        sl = (sh - (s - b)) + (corr - b) + sl
        sh = s + sl
        sl = sl - (sh - s)
        # + the table's -ln(inv)
        hi = _L_HI[j]
        s = sh + hi
        b = s - sh
        sl = (sh - (s - b)) + (hi - b) + sl + _L_LO[j]
        sh = s + sl
        sl = sl - (sh - s)
        if e:
            ef = float(e)
            eh = ef * _L_LN2_HI
            t = _L_SPLIT * ef
            fh = t - (t - ef)
            fl = ef - fh
            t = _L_SPLIT * _L_LN2_HI
            lh = t - (t - _L_LN2_HI)
            ll = _L_LN2_HI - lh
            el = ((fh * lh - eh) + fh * ll + fl * lh) + fl * ll + ef * _L_LN2_LO
            s = sh + eh
            b = s - sh
            sl = (sh - (s - b)) + (eh - b) + sl + el
            sh = s + sl
            sl = sl - (sh - s)

    y = sh + sl
    if y == 0.0:
        return _log_exact(x)
    resid = (sh - y) + sl
    if resid == 0.0:
        return y
    # The pair decides ``y`` unless it sits within its own error of the rounding
    # boundary. That boundary is half the gap on the SIDE the remainder points
    # to, which at a power of two is half the size of the other side's.
    if resid > 0.0:
        d = resid - (_math.nextafter(y, _L_INF) - y) * 0.5
    else:
        d = -resid - (y - _math.nextafter(y, -_L_INF)) * 0.5
    a = y if y > 0.0 else -y
    if (d if d > 0.0 else -d) > a * _L_REL_ERR + 5e-324:
        return y
    return _log_exact(x)
