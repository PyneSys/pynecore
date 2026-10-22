//! Port of `core/fdlibm.py`: fdlibm 5.3 sin/cos/tan/exp/log/pow/asin/acos/atan, the
//! StrictMath doubles TradingView's parser folds constant expressions with.
//!
//! Ported from the SunSoft fdlibm sources (netlib.org/fdlibm) by way of the Python
//! port, which carry the following notice that must be preserved:
//!
//! ```text
//! ====================================================
//! Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//!
//! Developed at SunSoft, a Sun Microsystems, Inc. business.
//! Permission to use, copy, modify, and distribute this
//! software is freely granted, provided that this notice
//! is preserved.
//! ====================================================
//! ```
//!
//! The Python module is the reference, so its integer arithmetic is mirrored with
//! `i64` (Python ints, never C's 32-bit wraparound) and every floating-point
//! operation is written in the Python order: Rust never contracts `a * b + c`
//! into a fused multiply-add, so each operation rounds exactly as CPython's does.
//! Not the `libm` crate: musl's exp/log/pow are other algorithms.

// The constants are spelled as the fdlibm sources spell them, digit for digit, and
// ``x - x`` is how they make the NaN of an infinity
#![allow(clippy::excessive_precision, clippy::approx_constant, clippy::eq_op)]

const M32: i64 = 0xFFFF_FFFF;

/// Signed high word of x (C's `__HI`).
fn hi(x: f64) -> i64 {
    ((x.to_bits() >> 32) as u32 as i32) as i64
}

/// Unsigned low word of x (C's `__LO`).
fn lo(x: f64) -> i64 {
    (x.to_bits() & 0xFFFF_FFFF) as i64
}

fn set_hi(x: f64, hi: i64) -> f64 {
    f64::from_bits((((hi & M32) as u64) << 32) | (x.to_bits() & 0xFFFF_FFFF))
}

fn set_lo(x: f64, lo: i64) -> f64 {
    f64::from_bits((x.to_bits() & 0xFFFF_FFFF_0000_0000) | (lo & M32) as u64)
}

/// Python's `float(int(x))`: truncated toward zero, a zero always positive.
fn trunc(x: f64) -> f64 {
    (x as i64) as f64
}

/// `math.ldexp(x, n)`: `x * 2**n` rounded once (musl's `scalbn`, which keeps the
/// final scaling out of a double rounding in the subnormal range).
pub fn ldexp(x: f64, n: i64) -> f64 {
    let mut y = x;
    let mut n = n;
    if n > 1023 {
        y *= f64::from_bits(0x7FE0_0000_0000_0000); // 2**1023
        n -= 1023;
        if n > 1023 {
            y *= f64::from_bits(0x7FE0_0000_0000_0000);
            n -= 1023;
            if n > 1023 {
                n = 1023;
            }
        }
    } else if n < -1022 {
        // 2**-1022 * 2**53, exact
        let down = f64::from_bits(0x0010_0000_0000_0000) * 9007199254740992.0;
        y *= down;
        n += 1022 - 53;
        if n < -1022 {
            y *= down;
            n += 1022 - 53;
            if n < -1022 {
                n = -1022;
            }
        }
    }
    y * f64::from_bits(((0x3ff + n) as u64) << 52)
}

// --- __kernel_sin / __kernel_cos (k_sin.c, k_cos.c) --------------------------

const S1: f64 = -1.66666666666666324348e-01;
const S2: f64 = 8.33333333332248946124e-03;
const S3: f64 = -1.98412698298579493134e-04;
const S4: f64 = 2.75573137070700676789e-06;
const S5: f64 = -2.50507602534068634195e-08;
const S6: f64 = 1.58969099521155010221e-10;

const C1: f64 = 4.16666666666666019037e-02;
const C2: f64 = -1.38888888888741095749e-03;
const C3: f64 = 2.48015872894767294178e-05;
const C4: f64 = -2.75573143513906633035e-07;
const C5: f64 = 2.08757232129817482790e-09;
const C6: f64 = -1.13596475577881948265e-11;

fn kernel_sin(x: f64, y: f64, iy: i64) -> f64 {
    let ix = hi(x) & 0x7fffffff;
    if ix < 0x3e400000 {
        // |x| < 2**-27
        return x;
    }
    let z = x * x;
    let v = z * x;
    let r = S2 + z * (S3 + z * (S4 + z * (S5 + z * S6)));
    if iy == 0 {
        return x + v * (S1 + z * r);
    }
    x - ((z * (0.5 * y - v * r) - y) - v * S1)
}

fn kernel_cos(x: f64, y: f64) -> f64 {
    let ix = hi(x) & 0x7fffffff;
    if ix < 0x3e400000 {
        // |x| < 2**-27
        return 1.0;
    }
    let z = x * x;
    let r = z * (C1 + z * (C2 + z * (C3 + z * (C4 + z * (C5 + z * C6)))));
    if ix < 0x3FD33333 {
        // |x| < 0.3
        return 1.0 - (0.5 * z - (z * r - x * y));
    }
    let qx = if ix > 0x3fe90000 {
        // |x| > 0.78125
        0.28125
    } else {
        // x/4, low word zero
        f64::from_bits((((ix - 0x00200000) & M32) as u64) << 32)
    };
    let hz = 0.5 * z - qx;
    let a = 1.0 - qx;
    a - (hz - (z * r - x * y))
}

// --- __ieee754_rem_pio2 (e_rem_pio2.c) + __kernel_rem_pio2 (k_rem_pio2.c) ----

const TWO_OVER_PI: [i64; 66] = [
    0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62, 0x95993C, 0x439041, 0xFE5163,
    0xABDEBB, 0xC561B7, 0x246E3A, 0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129,
    0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C, 0x7026B4, 0x5F7E41, 0x3991D6, 0x398353, 0x39F49C,
    0x845F8B, 0xBDF928, 0x3B1FF8, 0x97FFDE, 0x05980F, 0xEF2F11, 0x8B5A0A, 0x6D1F6D, 0x367ECF,
    0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5, 0xF17B3D, 0x0739F7, 0x8A5292,
    0xEA6BFB, 0x5FB11F, 0x8D5D08, 0x560330, 0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3,
    0x91615E, 0xE61B08, 0x659985, 0x5F14A0, 0x68408D, 0xFFD880, 0x4D7327, 0x310606, 0x1556CA,
    0x73A8C9, 0x60E27B, 0xC08C6B,
];

const NPIO2_HW: [i64; 32] = [
    0x3FF921FB, 0x400921FB, 0x4012D97C, 0x401921FB, 0x401F6A7A, 0x4022D97C, 0x4025FDBB, 0x402921FB,
    0x402C463A, 0x402F6A7A, 0x4031475C, 0x4032D97C, 0x40346B9C, 0x4035FDBB, 0x40378FDB, 0x403921FB,
    0x403AB41B, 0x403C463A, 0x403DD85A, 0x403F6A7A, 0x40407E4C, 0x4041475C, 0x4042106C, 0x4042D97C,
    0x4043A28C, 0x40446B9C, 0x404534AC, 0x4045FDBB, 0x4046C6CB, 0x40478FDB, 0x404858EB, 0x404921FB,
];

const PIO2: [f64; 8] = [
    1.57079625129699707031e+00,
    7.54978941586159635335e-08,
    5.39030252995776476554e-15,
    3.28200341580791294123e-22,
    1.27065575308067607349e-29,
    1.22933308981111328932e-36,
    2.73370053816464559624e-44,
    2.16741683877804819444e-51,
];

const TWO24: f64 = 1.67772160000000000000e+07;
const TWON24: f64 = 5.96046447753906250000e-08;

const INVPIO2: f64 = 6.36619772367581382433e-01;
const PIO2_1: f64 = 1.57079632673412561417e+00;
const PIO2_1T: f64 = 6.07710050650619224932e-11;
const PIO2_2: f64 = 6.07710050630396597660e-11;
const PIO2_2T: f64 = 2.02226624879595063154e-21;
const PIO2_3: f64 = 2.02226624871116645580e-21;
const PIO2_3T: f64 = 8.47842766036889956997e-32;

const INIT_JK: [i64; 4] = [2, 3, 4, 6];

/// A Python list index: a negative one counts from the end.
fn at(index: i64, len: usize) -> usize {
    if index < 0 { (len as i64 + index) as usize } else { index as usize }
}

/// The fixed-size work arrays of `_kernel_rem_pio2`, indexed the Python way.
struct Work<T>([T; 20]);

impl<T: Copy> Work<T> {
    fn get(&self, index: i64) -> T {
        self.0[at(index, 20)]
    }

    fn set(&mut self, index: i64, value: T) {
        self.0[at(index, 20)] = value;
    }
}

/// Returns `(n & 7, y0, y1)` for prec in (1, 2).
fn kernel_rem_pio2(x: &[f64; 3], e0: i64, nx: i64, prec: usize) -> (i64, f64, f64) {
    let mut iq = Work([0i64; 20]);
    let mut f = Work([0.0f64; 20]);
    let mut fq = Work([0.0f64; 20]);
    let mut q = Work([0.0f64; 20]);
    let xv = |index: i64| x[at(index, 3)];

    let jk = INIT_JK[prec];
    let jp = jk;
    let jx = nx - 1;
    let mut jv = if e0 >= 3 { (e0 - 3).div_euclid(24) } else { 0 };
    if jv < 0 {
        jv = 0;
    }
    let mut q0 = e0 - 24 * (jv + 1);

    let m = jx + jk;
    for i in 0..=m {
        let j = jv - jx + i;
        f.set(i, if j < 0 { 0.0 } else { TWO_OVER_PI[j as usize] as f64 });
    }

    for i in 0..=jk {
        let mut fw = 0.0;
        for j in 0..=jx {
            fw += xv(j) * f.get(jx + i - j);
        }
        q.set(i, fw);
    }

    let mut jz = jk;
    let mut z;
    let mut n;
    let mut ih;
    loop {
        // recompute:
        let mut i = 0;
        let mut j = jz;
        z = q.get(jz);
        while j > 0 {
            let fw = trunc(TWON24 * z);
            iq.set(i, (z - TWO24 * fw) as i64);
            z = q.get(j - 1) + fw;
            i += 1;
            j -= 1;
        }

        z = ldexp(z, q0);
        // ``math.floor`` answers an int, which ``8.0 *`` takes back exactly
        z -= 8.0 * (z * 0.125).floor();
        n = z as i64;
        z -= n as f64;
        ih = 0;
        if q0 > 0 {
            let i = iq.get(jz - 1) >> (24 - q0);
            n += i;
            iq.set(jz - 1, iq.get(jz - 1) - (i << (24 - q0)));
            ih = iq.get(jz - 1) >> (23 - q0);
        } else if q0 == 0 {
            ih = iq.get(jz - 1) >> 23;
        } else if z >= 0.5 {
            ih = 2;
        }

        if ih > 0 {
            n += 1;
            let mut carry = 0;
            for i in 0..jz {
                let j = iq.get(i);
                if carry == 0 {
                    if j != 0 {
                        carry = 1;
                        iq.set(i, 0x1000000 - j);
                    }
                } else {
                    iq.set(i, 0xffffff - j);
                }
            }
            if q0 == 1 {
                iq.set(jz - 1, iq.get(jz - 1) & 0x7fffff);
            } else if q0 == 2 {
                iq.set(jz - 1, iq.get(jz - 1) & 0x3fffff);
            }
            if ih == 2 {
                z = 1.0 - z;
                if carry != 0 {
                    z -= ldexp(1.0, q0);
                }
            }
        }

        if z == 0.0 {
            let mut j = 0;
            let mut i = jz - 1;
            while i > jk - 1 {
                j |= iq.get(i);
                i -= 1;
            }
            if j == 0 {
                // need recomputation
                let mut k = 1;
                while iq.get(jk - k) == 0 {
                    k += 1;
                }
                for i in jz + 1..jz + k + 1 {
                    f.set(jx + i, TWO_OVER_PI[(jv + i) as usize] as f64);
                    let mut fw = 0.0;
                    for j in 0..=jx {
                        fw += xv(j) * f.get(jx + i - j);
                    }
                    q.set(i, fw);
                }
                jz += k;
                continue;
            }
        }
        break;
    }

    if z == 0.0 {
        jz -= 1;
        q0 -= 24;
        while iq.get(jz) == 0 {
            jz -= 1;
            q0 -= 24;
        }
    } else {
        z = ldexp(z, -q0);
        if z >= TWO24 {
            let fw = trunc(TWON24 * z);
            iq.set(jz, (z - TWO24 * fw) as i64);
            jz += 1;
            q0 += 24;
            iq.set(jz, fw as i64);
        } else {
            iq.set(jz, z as i64);
        }
    }

    let mut fw = ldexp(1.0, q0);
    let mut i = jz;
    while i >= 0 {
        q.set(i, fw * iq.get(i) as f64);
        fw *= TWON24;
        i -= 1;
    }

    let mut i = jz;
    while i >= 0 {
        let mut fw = 0.0;
        let mut k = 0;
        while k <= jp && k <= jz - i {
            fw += PIO2[k as usize] * q.get(i + k);
            k += 1;
        }
        fq.set(jz - i, fw);
        i -= 1;
    }

    let mut fw = 0.0;
    let mut i = jz;
    while i >= 0 {
        fw += fq.get(i);
        i -= 1;
    }
    let y0 = if ih == 0 { fw } else { -fw };
    fw = fq.get(0) - fw;
    for i in 1..=jz {
        fw += fq.get(i);
    }
    let y1 = if ih == 0 { fw } else { -fw };
    (n & 7, y0, y1)
}

/// `__ieee754_rem_pio2`: returns `(n, y0, y1)`.
fn rem_pio2(x: f64) -> (i64, f64, f64) {
    let hx = hi(x);
    let ix = hx & 0x7fffffff;
    if ix <= 0x3fe921fb {
        // |x| ~<= pi/4
        return (0, x, 0.0);
    }
    if ix < 0x4002d97c {
        // |x| < 3pi/4
        if hx > 0 {
            let mut z = x - PIO2_1;
            let (y0, y1);
            if ix != 0x3ff921fb {
                y0 = z - PIO2_1T;
                y1 = (z - y0) - PIO2_1T;
            } else {
                z -= PIO2_2;
                y0 = z - PIO2_2T;
                y1 = (z - y0) - PIO2_2T;
            }
            return (1, y0, y1);
        }
        let mut z = x + PIO2_1;
        let (y0, y1);
        if ix != 0x3ff921fb {
            y0 = z + PIO2_1T;
            y1 = (z - y0) + PIO2_1T;
        } else {
            z += PIO2_2;
            y0 = z + PIO2_2T;
            y1 = (z - y0) + PIO2_2T;
        }
        return (-1, y0, y1);
    }
    if ix <= 0x413921fb {
        // |x| ~<= 2^19*(pi/2)
        let t = x.abs();
        let n = (t * INVPIO2 + 0.5) as i64;
        let fn_ = n as f64;
        let mut r = t - fn_ * PIO2_1;
        let mut w = fn_ * PIO2_1T;
        let mut y0;
        if n < 32 && ix != NPIO2_HW[at(n - 1, 32)] {
            y0 = r - w;
        } else {
            let j = ix >> 20;
            y0 = r - w;
            let i = j - ((hi(y0) >> 20) & 0x7ff);
            if i > 16 {
                // 2nd iteration, good to 118 bits
                let t2 = r;
                w = fn_ * PIO2_2;
                r = t2 - w;
                w = fn_ * PIO2_2T - ((t2 - r) - w);
                y0 = r - w;
                let i = j - ((hi(y0) >> 20) & 0x7ff);
                if i > 49 {
                    // 3rd iteration, 151 bits
                    let t3 = r;
                    w = fn_ * PIO2_3;
                    r = t3 - w;
                    w = fn_ * PIO2_3T - ((t3 - r) - w);
                    y0 = r - w;
                }
            }
        }
        let y1 = (r - y0) - w;
        if hx < 0 {
            return (-n, -y0, -y1);
        }
        return (n, y0, y1);
    }
    if ix >= 0x7ff00000 {
        // inf or NaN
        return (0, x - x, x - x);
    }
    // all other (large) arguments
    let low = lo(x);
    let e0 = (ix >> 20) - 1046;
    let mut z = f64::from_bits(((((ix - ((e0 << 20) & M32)) & M32) as u64) << 32) | low as u64);
    let mut tx = [0.0f64; 3];
    for item in tx.iter_mut().take(2) {
        *item = trunc(z);
        z = (z - *item) * TWO24;
    }
    tx[2] = z;
    let mut nx = 3;
    while tx[at(nx - 1, 3)] == 0.0 {
        nx -= 1;
    }
    let (n, y0, y1) = kernel_rem_pio2(&tx, e0, nx, 2);
    if hx < 0 {
        return (-n, -y0, -y1);
    }
    (n, y0, y1)
}

// --- sin, cos (s_sin.c, s_cos.c) ---------------------------------------------

pub fn sin(x: f64) -> f64 {
    let ix = hi(x) & 0x7fffffff;
    if ix <= 0x3fe921fb {
        return kernel_sin(x, 0.0, 0);
    }
    if ix >= 0x7ff00000 {
        return x - x;
    }
    let (n, y0, y1) = rem_pio2(x);
    match n & 3 {
        0 => kernel_sin(y0, y1, 1),
        1 => kernel_cos(y0, y1),
        2 => -kernel_sin(y0, y1, 1),
        _ => -kernel_cos(y0, y1),
    }
}

pub fn cos(x: f64) -> f64 {
    let ix = hi(x) & 0x7fffffff;
    if ix <= 0x3fe921fb {
        return kernel_cos(x, 0.0);
    }
    if ix >= 0x7ff00000 {
        return x - x;
    }
    let (n, y0, y1) = rem_pio2(x);
    match n & 3 {
        0 => kernel_cos(y0, y1),
        1 => -kernel_sin(y0, y1, 1),
        2 => -kernel_cos(y0, y1),
        _ => kernel_sin(y0, y1, 1),
    }
}

// --- exp (e_exp.c) -----------------------------------------------------------

const HALF: [f64; 2] = [0.5, -0.5];
const HUGE: f64 = 1.0e+300;
const TWOM1000: f64 = 9.33263618503218878990e-302;
const O_THRESHOLD: f64 = 7.09782712893383973096e+02;
const U_THRESHOLD: f64 = -7.45133219101941108420e+02;
const LN2HI: [f64; 2] = [6.93147180369123816490e-01, -6.93147180369123816490e-01];
const LN2LO: [f64; 2] = [1.90821492927058770002e-10, -1.90821492927058770002e-10];
const INVLN2: f64 = 1.44269504088896338700e+00;
const EP1: f64 = 1.66666666666666019037e-01;
const EP2: f64 = -2.77777777770155933842e-03;
const EP3: f64 = 6.61375632143793436117e-05;
const EP4: f64 = -1.65339022054652515390e-06;
const EP5: f64 = 4.13813679705723846039e-08;

pub fn exp(x: f64) -> f64 {
    let mut x = x;
    let mut hx = hi(x);
    let xsb = ((hx >> 31) & 1) as usize;
    hx &= 0x7fffffff;

    let mut hi_part = 0.0;
    let mut lo_part = 0.0;
    let k: i64;
    if hx >= 0x40862E42 {
        // |x| >= 709.78...
        if hx >= 0x7ff00000 {
            if ((hx & 0xfffff) | lo(x)) != 0 {
                return x + x; // NaN
            }
            return if xsb == 0 { x } else { 0.0 }; // exp(+-inf)
        }
        if x > O_THRESHOLD {
            return HUGE * HUGE; // overflow
        }
        if x < U_THRESHOLD {
            return TWOM1000 * TWOM1000; // underflow
        }
    }

    if hx > 0x3fd62e42 {
        // |x| > 0.5 ln2
        if hx < 0x3FF0A2B2 {
            // |x| < 1.5 ln2
            hi_part = x - LN2HI[xsb];
            lo_part = LN2LO[xsb];
            k = 1 - xsb as i64 - xsb as i64;
        } else {
            k = (INVLN2 * x + HALF[xsb]) as i64;
            let t = k as f64;
            hi_part = x - t * LN2HI[0];
            lo_part = t * LN2LO[0];
        }
        x = hi_part - lo_part;
    } else if hx < 0x3e300000 {
        // |x| < 2**-28
        return 1.0 + x;
    } else {
        k = 0;
    }

    let t = x * x;
    let c = x - t * (EP1 + t * (EP2 + t * (EP3 + t * (EP4 + t * EP5))));
    if k == 0 {
        return 1.0 - ((x * c) / (c - 2.0) - x);
    }
    let y = 1.0 - ((lo_part - (x * c) / (2.0 - c)) - hi_part);
    if k >= -1021 {
        return set_hi(y, hi(y) + (k << 20));
    }
    let y = set_hi(y, hi(y) + ((k + 1000) << 20));
    y * TWOM1000
}

// --- asin, acos (e_asin.c, e_acos.c) -----------------------------------------

const PIO2_HI: f64 = 1.57079632679489655800e+00;
const PIO2_LO: f64 = 6.12323399573676603587e-17;
const PIO4_HI: f64 = 7.85398163397448278999e-01;
const PI_ACOS: f64 = 3.14159265358979311600e+00;
const PS0: f64 = 1.66666666666666657415e-01;
const PS1: f64 = -3.25565818622400915405e-01;
const PS2: f64 = 2.01212532134862925881e-01;
const PS3: f64 = -4.00555345006794114027e-02;
const PS4: f64 = 7.91534994289814532176e-04;
const PS5: f64 = 3.47933107596021167570e-05;
const QS1: f64 = -2.40339491173441421878e+00;
const QS2: f64 = 2.02094576023350569471e+00;
const QS3: f64 = -6.88283971605453293030e-01;
const QS4: f64 = 7.70381505559019352791e-02;

pub fn asin(x: f64) -> f64 {
    let hx = hi(x);
    let ix = hx & 0x7fffffff;
    if ix >= 0x3ff00000 {
        // |x| >= 1
        if ((ix - 0x3ff00000) | lo(x)) == 0 {
            return x * PIO2_HI + x * PIO2_LO;
        }
        return f64::NAN; // (x-x)/(x-x)
    }
    if ix < 0x3fe00000 {
        // |x| < 0.5
        if ix < 0x3e400000 {
            // |x| < 2**-27
            return x;
        }
        let t = x * x;
        let p = t * (PS0 + t * (PS1 + t * (PS2 + t * (PS3 + t * (PS4 + t * PS5)))));
        let q = 1.0 + t * (QS1 + t * (QS2 + t * (QS3 + t * QS4)));
        let w = p / q;
        return x + x * w;
    }
    // 1 > |x| >= 0.5
    let w = 1.0 - x.abs();
    let mut t = w * 0.5;
    let mut p = t * (PS0 + t * (PS1 + t * (PS2 + t * (PS3 + t * (PS4 + t * PS5)))));
    let mut q = 1.0 + t * (QS1 + t * (QS2 + t * (QS3 + t * QS4)));
    let s = t.sqrt();
    if ix >= 0x3FEF3333 {
        // |x| > 0.975
        let w = p / q;
        t = PIO2_HI - (2.0 * (s + s * w) - PIO2_LO);
    } else {
        let w = set_lo(s, 0);
        let c = (t - w * w) / (s + w);
        let r = p / q;
        p = 2.0 * s * r - (PIO2_LO - 2.0 * c);
        q = PIO4_HI - 2.0 * w;
        t = PIO4_HI - (p - q);
    }
    if hx > 0 { t } else { -t }
}

pub fn acos(x: f64) -> f64 {
    let hx = hi(x);
    let ix = hx & 0x7fffffff;
    if ix >= 0x3ff00000 {
        // |x| >= 1
        if ((ix - 0x3ff00000) | lo(x)) == 0 {
            if hx > 0 {
                return 0.0;
            }
            return PI_ACOS + 2.0 * PIO2_LO;
        }
        return f64::NAN; // (x-x)/(x-x)
    }
    if ix < 0x3fe00000 {
        // |x| < 0.5
        if ix <= 0x3c600000 {
            // |x| < 2**-57
            return PIO2_HI + PIO2_LO;
        }
        let z = x * x;
        let p = z * (PS0 + z * (PS1 + z * (PS2 + z * (PS3 + z * (PS4 + z * PS5)))));
        let q = 1.0 + z * (QS1 + z * (QS2 + z * (QS3 + z * QS4)));
        let r = p / q;
        return PIO2_HI - (x - (PIO2_LO - x * r));
    }
    if hx < 0 {
        // x < -0.5
        let z = (1.0 + x) * 0.5;
        let p = z * (PS0 + z * (PS1 + z * (PS2 + z * (PS3 + z * (PS4 + z * PS5)))));
        let q = 1.0 + z * (QS1 + z * (QS2 + z * (QS3 + z * QS4)));
        let s = z.sqrt();
        let r = p / q;
        let w = r * s - PIO2_LO;
        return PI_ACOS - 2.0 * (s + w);
    }
    // x > 0.5
    let z = (1.0 - x) * 0.5;
    let s = z.sqrt();
    let df = set_lo(s, 0);
    let c = (z - df * df) / (s + df);
    let p = z * (PS0 + z * (PS1 + z * (PS2 + z * (PS3 + z * (PS4 + z * PS5)))));
    let q = 1.0 + z * (QS1 + z * (QS2 + z * (QS3 + z * QS4)));
    let r = p / q;
    let w = r * s + c;
    2.0 * (df + w)
}

// --- atan (s_atan.c) ---------------------------------------------------------

const ATAN_HI: [f64; 4] = [
    4.63647609000806093515e-01, // atan(0.5) head
    7.85398163397448278999e-01, // atan(1.0) head
    9.82793723247329054082e-01, // atan(1.5) head
    1.57079632679489655800e+00, // atan(inf) head
];
const ATAN_LO: [f64; 4] = [
    2.26987774529616870924e-17,
    3.06161699786838301793e-17,
    1.39033110312309984516e-17,
    6.12323399573676603587e-17,
];
const AT0: f64 = 3.33333333333329318027e-01;
const AT1: f64 = -1.99999999998764832476e-01;
const AT2: f64 = 1.42857142725034663711e-01;
const AT3: f64 = -1.11111104054623557880e-01;
const AT4: f64 = 9.09088713343650656196e-02;
const AT5: f64 = -7.69187620504482999495e-02;
const AT6: f64 = 6.66107313738753120669e-02;
const AT7: f64 = -5.83357013379057348645e-02;
const AT8: f64 = 4.97687799461593236017e-02;
const AT9: f64 = -3.65315727442169155270e-02;
const AT10: f64 = 1.62858201153657823623e-02;

pub fn atan(x: f64) -> f64 {
    let mut x = x;
    let hx = hi(x);
    let ix = hx & 0x7fffffff;
    let id: i64;
    if ix >= 0x44100000 {
        // |x| >= 2**66
        if ix > 0x7ff00000 || (ix == 0x7ff00000 && lo(x) != 0) {
            return x + x; // NaN
        }
        if hx > 0 {
            return ATAN_HI[3] + ATAN_LO[3];
        }
        return -ATAN_HI[3] - ATAN_LO[3];
    }
    if ix < 0x3fdc0000 {
        // |x| < 0.4375
        if ix < 0x3e200000 {
            // |x| < 2**-29
            return x;
        }
        id = -1;
    } else {
        x = x.abs();
        if ix < 0x3ff30000 {
            // |x| < 1.1875
            if ix < 0x3fe60000 {
                // 7/16 <= |x| < 11/16
                id = 0;
                x = (2.0 * x - 1.0) / (2.0 + x);
            } else {
                // 11/16 <= |x| < 19/16
                id = 1;
                x = (x - 1.0) / (x + 1.0);
            }
        } else if ix < 0x40038000 {
            // |x| < 2.4375
            id = 2;
            x = (x - 1.5) / (1.0 + 1.5 * x);
        } else {
            // 2.4375 <= |x| < 2**66
            id = 3;
            x = -1.0 / x;
        }
    }

    // the sum of aT[i] * z**(i+1) broken into its odd and even halves
    let z = x * x;
    let w = z * z;
    let s1 = z * (AT0 + w * (AT2 + w * (AT4 + w * (AT6 + w * (AT8 + w * AT10)))));
    let s2 = w * (AT1 + w * (AT3 + w * (AT5 + w * (AT7 + w * AT9))));
    if id < 0 {
        return x - x * (s1 + s2);
    }
    let id = id as usize;
    let z = ATAN_HI[id] - ((x * (s1 + s2) - ATAN_LO[id]) - x);
    if hx < 0 { -z } else { z }
}

// --- log (e_log.c) -----------------------------------------------------------

const LN2_HI: f64 = 6.93147180369123816490e-01;
const LN2_LO: f64 = 1.90821492927058770002e-10;
const TWO54: f64 = 1.80143985094819840000e+16;
const LG1: f64 = 6.666666666666735130e-01;
const LG2: f64 = 3.999999999940941908e-01;
const LG3: f64 = 2.857142874366239149e-01;
const LG4: f64 = 2.222219843214978396e-01;
const LG5: f64 = 1.818357216161805012e-01;
const LG6: f64 = 1.531383769920937332e-01;
const LG7: f64 = 1.479819860511658591e-01;

pub fn log(x: f64) -> f64 {
    let mut x = x;
    let mut hx = hi(x);
    let lx = lo(x);
    let mut k = 0;
    if hx < 0x00100000 {
        // x < 2**-1022
        if ((hx & 0x7fffffff) | lx) == 0 {
            return f64::NEG_INFINITY; // log(+-0) = -inf
        }
        if hx < 0 {
            return f64::NAN; // log(-#) = NaN
        }
        k -= 54;
        x *= TWO54; // subnormal: scale up x
        hx = hi(x);
    }
    if hx >= 0x7ff00000 {
        return x + x;
    }
    k += (hx >> 20) - 1023;
    hx &= 0x000fffff;
    let i = (hx + 0x95f64) & 0x100000;
    x = set_hi(x, hx | (i ^ 0x3ff00000)); // normalize x or x/2
    k += i >> 20;
    let f = x - 1.0;
    if (0x000fffff & (2 + hx)) < 3 {
        // |f| < 2**-20
        if f == 0.0 {
            if k == 0 {
                return 0.0;
            }
            let dk = k as f64;
            return dk * LN2_HI + dk * LN2_LO;
        }
        let r = f * f * (0.5 - 0.33333333333333333 * f);
        if k == 0 {
            return f - r;
        }
        let dk = k as f64;
        return dk * LN2_HI - ((r - dk * LN2_LO) - f);
    }
    let s = f / (2.0 + f);
    let dk = k as f64;
    let z = s * s;
    let mut i = hx - 0x6147a;
    let w = z * z;
    let j = 0x6b851 - hx;
    let t1 = w * (LG2 + w * (LG4 + w * LG6));
    let t2 = z * (LG1 + w * (LG3 + w * (LG5 + w * LG7)));
    i |= j;
    let r = t2 + t1;
    if i > 0 {
        let hfsq = 0.5 * f * f;
        if k == 0 {
            return f - (hfsq - s * (hfsq + r));
        }
        return dk * LN2_HI - ((hfsq - (s * (hfsq + r) + dk * LN2_LO)) - f);
    }
    if k == 0 {
        return f - s * (f - r);
    }
    dk * LN2_HI - ((s * (f - r) - dk * LN2_LO) - f)
}

// --- tan (k_tan.c, s_tan.c) --------------------------------------------------

const T: [f64; 13] = [
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
];
const PIO4: f64 = 7.85398163397448278999e-01;
const PIO4LO: f64 = 3.06161699786838301793e-17;

fn kernel_tan(x: f64, y: f64, iy: i64) -> f64 {
    let (mut x, mut y) = (x, y);
    let hx = hi(x);
    let ix = hx & 0x7fffffff;
    if ix < 0x3e300000 {
        // |x| < 2**-28
        if ((ix | lo(x)) | (iy + 1)) == 0 {
            return f64::INFINITY; // 1 / |x| of a zero
        }
        if iy == 1 {
            return x;
        }
        // compute -1 / (x+y) carefully
        let w = x + y;
        let z = set_lo(w, 0);
        let v = y - (z - x);
        let a = div(-1.0, w);
        let t = set_lo(a, 0);
        let s = 1.0 + t * z;
        return t + a * (s + t * v);
    }
    if ix >= 0x3FE59428 {
        // |x| >= 0.6744
        if hx < 0 {
            x = -x;
            y = -y;
        }
        let z = PIO4 - x;
        let w = PIO4LO - y;
        x = z + w;
        y = 0.0;
    }
    let z = x * x;
    let w = z * z;
    let r = T[1] + w * (T[3] + w * (T[5] + w * (T[7] + w * (T[9] + w * T[11]))));
    let v = z * (T[2] + w * (T[4] + w * (T[6] + w * (T[8] + w * (T[10] + w * T[12])))));
    let s = z * x;
    let mut r = y + z * (s * (r + v) + y);
    r += T[0] * s;
    let w = x + r;
    if ix >= 0x3FE59428 {
        let v = iy as f64;
        return (1 - ((hx >> 30) & 2)) as f64 * (v - 2.0 * (x - (w * w / (w + v) - r)));
    }
    if iy == 1 {
        return w;
    }
    // compute -1.0 / (x+r) accurately
    let z = set_lo(w, 0);
    let v = r - (z - x);
    let a = -1.0 / w;
    let t = set_lo(a, 0);
    let s = 1.0 + t * z;
    t + a * (s + t * v)
}

pub fn tan(x: f64) -> f64 {
    let ix = hi(x) & 0x7fffffff;
    if ix <= 0x3fe921fb {
        return kernel_tan(x, 0.0, 1);
    }
    if ix >= 0x7ff00000 {
        return x - x;
    }
    let (n, y0, y1) = rem_pio2(x);
    kernel_tan(y0, y1, 1 - ((n & 1) << 1))
}

// --- pow (e_pow.c) -----------------------------------------------------------

const BP: [f64; 2] = [1.0, 1.5];
const DP_H: [f64; 2] = [0.0, 5.84962487220764160156e-01];
const DP_L: [f64; 2] = [0.0, 1.35003920212974897128e-08];
const TWO53: f64 = 9007199254740992.0;
const TINY: f64 = 1.0e-300;
const PL1: f64 = 5.99999999999994648725e-01;
const PL2: f64 = 4.28571428578550184252e-01;
const PL3: f64 = 3.33333329818377432918e-01;
const PL4: f64 = 2.72728123808534006489e-01;
const PL5: f64 = 2.30660745775561754067e-01;
const PL6: f64 = 2.06975017800338417784e-01;
const LG2_P: f64 = 6.93147180559945286227e-01;
const LG2_H: f64 = 6.93147182464599609375e-01;
const LG2_L: f64 = -1.90465429995776804525e-09;
const OVT: f64 = 8.0085662595372944372e-17;
const CP: f64 = 9.61796693925975554329e-01;
const CP_H: f64 = 9.61796700954437255859e-01;
const CP_L: f64 = -7.02846165095275826516e-09;
const IVLN2: f64 = 1.44269504088896338700e+00;
const IVLN2_H: f64 = 1.44269502162933349609e+00;
const IVLN2_L: f64 = 1.92596299112661746887e-08;

/// C's `a / b`, where a zero divisor gives an infinity or a nan instead of raising.
fn div(a: f64, b: f64) -> f64 {
    if b != 0.0 {
        return a / b;
    }
    if a.is_nan() || a == 0.0 {
        return f64::NAN;
    }
    f64::INFINITY.copysign(a) * 1.0f64.copysign(b)
}

pub fn pow(x: f64, y: f64) -> f64 {
    let hx = hi(x);
    let lx = lo(x);
    let hy = hi(y);
    let ly = lo(y);
    let mut ix = hx & 0x7fffffff;
    let iy = hy & 0x7fffffff;

    if (iy | ly) == 0 {
        // x**0 = 1
        return 1.0;
    }
    if ix > 0x7ff00000
        || (ix == 0x7ff00000 && lx != 0)
        || iy > 0x7ff00000
        || (iy == 0x7ff00000 && ly != 0)
    {
        return x + y; // +-NaN
    }

    // yisint: 0 -- y is not an integer, 1 -- odd integer, 2 -- even integer
    let mut yisint = 0;
    if hx < 0 {
        if iy >= 0x43400000 {
            yisint = 2;
        } else if iy >= 0x3ff00000 {
            let k = (iy >> 20) - 0x3ff;
            if k > 20 {
                let j = ly >> (52 - k);
                if (j << (52 - k)) == ly {
                    yisint = 2 - (j & 1);
                }
            } else if ly == 0 {
                let j = iy >> (20 - k);
                if (j << (20 - k)) == iy {
                    yisint = 2 - (j & 1);
                }
            }
        }
    }

    if ly == 0 {
        if iy == 0x7ff00000 {
            // y is +-inf
            if ((ix - 0x3ff00000) | lx) == 0 {
                return y - y; // inf**+-1 is NaN
            }
            if ix >= 0x3ff00000 {
                // (|x|>1)**+-inf = inf,0
                return if hy >= 0 { y } else { 0.0 };
            }
            return if hy < 0 { -y } else { 0.0 }; // (|x|<1)**-,+inf = inf,0
        }
        if iy == 0x3ff00000 {
            // y is +-1
            return if hy < 0 { div(1.0, x) } else { x };
        }
        if hy == 0x40000000 {
            // y is 2
            return x * x;
        }
        if hy == 0x3fe00000 && hx >= 0 {
            // y is 0.5, x >= +0
            return x.sqrt();
        }
    }

    let mut ax = x.abs();
    if lx == 0 && (ix == 0x7ff00000 || ix == 0 || ix == 0x3ff00000) {
        let mut z = ax; // x is +-0, +-inf, +-1
        if hy < 0 {
            z = div(1.0, z);
        }
        if hx < 0 {
            if ((ix - 0x3ff00000) | yisint) == 0 {
                z = f64::NAN; // (-1)**non-int is NaN
            } else if yisint == 1 {
                z = -z; // (x<0)**odd = -(|x|**odd)
            }
        }
        return z;
    }

    let mut n = (hx >> 31) + 1;
    if (n | yisint) == 0 {
        return f64::NAN; // (x<0)**(non-int) is NaN
    }

    let mut s = 1.0; // sign of the result
    if (n | (yisint - 1)) == 0 {
        s = -1.0; // (-ve)**(odd int)
    }

    let (t1, t2) = if iy > 0x41e00000 {
        // |y| > 2**31
        if iy > 0x43f00000 {
            // |y| > 2**64, must o/uflow
            if ix <= 0x3fefffff {
                return if hy < 0 { HUGE * HUGE } else { TINY * TINY };
            }
            if ix >= 0x3ff00000 {
                return if hy > 0 { HUGE * HUGE } else { TINY * TINY };
            }
        }
        // over/underflow if x is not close to one
        if ix < 0x3fefffff {
            return if hy < 0 { s * HUGE * HUGE } else { s * TINY * TINY };
        }
        if ix > 0x3ff00000 {
            return if hy > 0 { s * HUGE * HUGE } else { s * TINY * TINY };
        }
        // |1-x| <= 2**-20: log(x) by x-x^2/2+x^3/3-x^4/4
        let t = ax - 1.0;
        let w = (t * t) * (0.5 - t * (0.3333333333333333333333 - t * 0.25));
        let u = IVLN2_H * t;
        let v = t * IVLN2_L - w * IVLN2;
        let t1 = set_lo(u + v, 0);
        (t1, v - (t1 - u))
    } else {
        n = 0;
        if ix < 0x00100000 {
            // subnormal x
            ax *= TWO53;
            n -= 53;
            ix = hi(ax);
        }
        n += (ix >> 20) - 0x3ff;
        let j = ix & 0x000fffff;
        ix = j | 0x3ff00000; // normalize ix
        let k: usize;
        if j <= 0x3988E {
            // |x| < sqrt(3/2)
            k = 0;
        } else if j < 0xBB67A {
            // |x| < sqrt(3)
            k = 1;
        } else {
            k = 0;
            n += 1;
            ix -= 0x00100000;
        }
        ax = set_hi(ax, ix);

        // ss = s_h+s_l = (x-1)/(x+1) or (x-1.5)/(x+1.5)
        let u = ax - BP[k];
        let v = 1.0 / (ax + BP[k]);
        let ss = u * v;
        let s_h = set_lo(ss, 0);
        // t_h = ax+bp[k] High
        let t_h = set_hi(0.0, ((ix >> 1) | 0x20000000) + 0x00080000 + ((k as i64) << 18));
        let t_l = ax - (t_h - BP[k]);
        let s_l = v * ((u - s_h * t_h) - s_h * t_l);
        // log(ax)
        let mut s2 = ss * ss;
        let mut r = s2 * s2 * (PL1 + s2 * (PL2 + s2 * (PL3 + s2 * (PL4 + s2 * (PL5 + s2 * PL6)))));
        r += s_l * (s_h + ss);
        s2 = s_h * s_h;
        let t_h = set_lo(3.0 + s2 + r, 0);
        let t_l = r - ((t_h - 3.0) - s2);
        // u+v = ss*(1+...)
        let u = s_h * t_h;
        let v = s_l * t_h + t_l * ss;
        // 2/(3log2)*(ss+...)
        let p_h = set_lo(u + v, 0);
        let p_l = v - (p_h - u);
        let z_h = CP_H * p_h;
        let z_l = CP_L * p_h + p_l * CP + DP_L[k];
        // log2(ax) = (ss+..)*2/(3*log2) = n + dp_h + z_h + z_l
        let t = n as f64;
        let t1 = set_lo(((z_h + z_l) + DP_H[k]) + t, 0);
        (t1, z_l - (((t1 - t) - DP_H[k]) - z_h))
    };

    // split up y into y1+y2 and compute (y1+y2)*(t1+t2)
    let y1 = set_lo(y, 0);
    let p_l = (y - y1) * t1 + y * t2;
    let mut p_h = y1 * t1;
    let z = p_l + p_h;
    let j = hi(z);
    let i = lo(z);
    if j >= 0x40900000 {
        // z >= 1024
        if ((j - 0x40900000) | i) != 0 {
            return s * HUGE * HUGE; // overflow
        }
        if p_l + OVT > z - p_h {
            return s * HUGE * HUGE;
        }
    } else if (j & 0x7fffffff) >= 0x4090cc00 {
        // z <= -1075
        if (((j & M32) - 0xc090cc00) | i) != 0 {
            return s * TINY * TINY; // underflow
        }
        if p_l <= z - p_h {
            return s * TINY * TINY;
        }
    }

    // 2**(p_h+p_l)
    let i = j & 0x7fffffff;
    let mut k = (i >> 20) - 0x3ff;
    let mut n = 0;
    if i > 0x3fe00000 {
        // |z| > 0.5: n = [z+0.5]
        n = j + (0x00100000 >> (k + 1));
        k = ((n & 0x7fffffff) >> 20) - 0x3ff;
        let t = set_hi(0.0, n & !(0x000fffff >> k));
        n = ((n & 0x000fffff) | 0x00100000) >> (20 - k);
        if j < 0 {
            n = -n;
        }
        p_h -= t;
    }
    let t = set_lo(p_l + p_h, 0);
    let u = t * LG2_H;
    let v = (p_l - (t - p_h)) * LG2_P + t * LG2_L;
    let mut z = u + v;
    let w = v - (z - u);
    let t = z * z;
    let t1 = z - t * (EP1 + t * (EP2 + t * (EP3 + t * (EP4 + t * EP5))));
    let r = (z * t1) / (t1 - 2.0) - (w + z * w);
    z = 1.0 - (r - z);
    let j = hi(z) + (n << 20);
    if (j >> 20) <= 0 {
        z = ldexp(z, n); // subnormal output
    } else {
        z = set_hi(z, hi(z) + (n << 20));
    }
    s * z
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        assert_eq!(sin(0.5).to_bits(), 0.479425538604203_f64.to_bits());
        // StrictMath.exp(1) is one ulp above the correctly rounded e
        assert_eq!(exp(1.0), 2.7182818284590455);
        assert_eq!(log(1.0), 0.0);
        assert_eq!(pow(2.0, 10.0), 1024.0);
        assert_eq!(atan(f64::INFINITY), std::f64::consts::FRAC_PI_2);
        assert!(asin(2.0).is_nan());
        assert_eq!(ldexp(1.0, -1074), f64::from_bits(1));
        assert_eq!(ldexp(1.5, -1074), f64::from_bits(2));
        assert_eq!(ldexp(1.0, 1023), f64::from_bits(0x7FE0_0000_0000_0000));
    }
}
