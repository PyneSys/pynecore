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
from struct import pack as _pack, unpack as _unpack

__all__ = ['cos', 'sin', 'exp']

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

_PI32INV = _b2f(0x40245f306dc9c883)         # 32/pi
_P1 = _b2f(0x3fb921fb54400000)              # pi/32 head
_P2 = _b2f(0x3d90b4611a600000)              # pi/32 middle
_P3 = _b2f(0x3b63198a2e037073)              # pi/32 tail
_SC1_LO, _SC1_HI = _b2f(0xbfc5555555555555), _b2f(0xbfe0000000000000)
_SC2_LO, _SC2_HI = _b2f(0x3f81111111111111), _b2f(0x3fa5555555555555)
_SC3_LO, _SC3_HI = _b2f(0xbf2a01a01a01a01a), _b2f(0xbf56c16c16c16c17)
_SC4_LO, _SC4_HI = _b2f(0x3ec71de3a556c734), _b2f(0x3efa01a01a01a01a)
_PI_4_HEAD = _b2f(0x3fe921fb40000000)       # pi/4 split head
_PI_4_TAIL = _b2f(0x3e64442d18469899)       # pi/4 split tail

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
    if exp16 == 0x7FF0:                     # Inf or NaN
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
    low = prod & _M64                       # bits [0:64)
    mid = (prod >> 64) & _M64               # bits [64:128)
    up32 = (prod >> 128) & _M32             # bits [128:160)
    top = (prod >> 160) & _M64              # bits [160:224)

    sign16 = 32768 if x_bits >> 63 else 0
    expo = ((x_bits >> 52) & 2047) - 1023
    point = off * 8 + 19 - expo             # binary-point offset
    e_ctr = point + 32
    sign_flip = 0

    def _complement(borrow_hi, lo, md, tp):
        """(borrow_hi - fraction): three-limb negate with borrow chain."""
        lo2 = (0 - lo) & _M64
        cf = 1 if lo != 0 else 0
        t = md + cf
        md2 = (0 - t) & _M64
        cf = 1 if t != 0 else 0
        tp2 = (borrow_hi - tp - cf) & _M64
        return lo2, md2, tp2

    if point >= 1:                          # binary point inside the top word
        sh = (29 - point) & 31
        t32 = ((top & _M32) << sh) & _M32
        quad_acc = t32
        frac29 = t32 & 0x1FFFFFFF
        f = (frac29 >> sh) & _M32
        top = ((f << 32) | up32) & _M64
        if frac29 & 0x10000000:             # fraction >= 1/2: complement, bump
            low, mid, top = _complement(((0x20000000 >> sh) & _M32) << 32,
                                        low, mid, top)
            sign_flip = 32768
            quad_acc = (quad_acc + 0x20000000) & _M32
        quad_base = quad_acc >> 29
    else:                                   # point below the top word
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
        if band - 281346048 < 0x80000000:   # |x| >= 90112 (signed positive)
            return _reduce_huge(x_bits, 1865232, True)
        return 1.0 - abs(x)                 # |x| < 2^-252
    n = _round_n(x)
    return _sincos_poly(x, n, (n + 16) & 63, 0.0)


def sin(x: float) -> float:
    """Bit-exact venue-runtime sin."""
    x_bits = _f2b(x)
    band = (((x_bits >> 32) & 2147418112) - 808452096) & _M32
    if band > 281346048:
        if band - 281346048 < 0x80000000:   # |x| >= 90112
            return _reduce_huge(x_bits, 1865216, False)
        if (band >> 20) == 3325:
            return x * _b2f(0x3fefffffffffffff)
        return x
    n = _round_n(x)
    return _sincos_poly(x, n, n & 63, 0.0)


# --- exp ---------------------------------------------------------------------

_E_LOG2_64 = _b2f(0x40571547652b82fe)       # 64/ln2
_E_LN2_64_HEAD = _b2f(0x3f862e42fefa0000)   # ln2/64 head
_E_LN2_64_TAIL = _b2f(0x3d1cf79abc9e3b3a)   # ln2/64 tail
_E_HALF = _b2f(0x3fdffffffffffffe)          # 0.5 - 1/4 ulp
_E_P3_LO, _E_P3_HI = _b2f(0x3f56c15ce3289860), _b2f(0x3fa55555555b9e25)
_E_P5_LO, _E_P5_HI = _b2f(0x3f811115c090cf0f), _b2f(0x3fc5555555548ba1)
_E_SHIFTER = _b2f(0x4338000000000000)       # 1.5 * 2^52
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
        if mag >= 1083179008:               # |x| >= ~709.78 or non-finite
            if mag < 2146435072:
                if hi32 >= 0x80000000:      # underflow to zero
                    return _E_XMIN * _E_XMIN
                return _E_XMAX * _E_XMAX    # overflow to inf
            if mag > 2146435072 or (x_bits & _M32) != 0:
                return x + x                # NaN
            if hi32 == 2146435072:
                return _b2f(_E_INF_BITS)    # exp(+inf)
            return 0.0                      # exp(-inf)
        return x + 1.0                      # tiny |x|

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
        if n >= 1023:                       # overflow side
            return (acc2 + dropped) * scale3
        neg = (_f2b(acc2) >> 48) & 32768
        if (shift | neg) == 0:
            return (acc2 + dropped) * scale3
        saved = acc2
        result = (acc2 + dropped) * scale3
        if (_f2b(result) >> 48) & 32752:
            return result                   # still a normal number
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
