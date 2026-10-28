//! `10**y` correctly rounded, for a host without a platform `pow`.
//!
//! CPython's `math.pow` is the platform's `pow`, which a native build calls too. A
//! WebAssembly build has none worth following, so it answers the double nearest to
//! the exact power: `exp(y * ln 10)` in double-double arithmetic, kept only when the
//! error bound cannot reach a rounding boundary (Ziv's test). Its rare undecidable
//! case answers `None`.

/// `a + b` as an unevaluated sum, exactly.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    let b_virtual = sum - a;
    let a_virtual = sum - b_virtual;
    (sum, (a - a_virtual) + (b - b_virtual))
}

/// `a * b` as an unevaluated sum, exactly.
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let product = a * b;
    (product, a.mul_add(b, -product))
}

/// A double-double: `hi + lo` with `|lo| <= ulp(hi) / 2`.
#[derive(Clone, Copy)]
struct Dd(f64, f64);

impl Dd {
    fn add(self, other: Dd) -> Dd {
        let (sum, error) = two_sum(self.0, other.0);
        let (hi, lo) = two_sum(sum, error + self.1 + other.1);
        Dd(hi, lo)
    }

    fn mul(self, other: Dd) -> Dd {
        let (product, error) = two_prod(self.0, other.0);
        let (hi, lo) = two_sum(product, error + self.0 * other.1 + self.1 * other.0);
        Dd(hi, lo)
    }

    fn scale(self, factor: f64) -> Dd {
        Dd(self.0 * factor, self.1 * factor)
    }
}

const LN10: Dd = Dd(std::f64::consts::LN_10, -2.1707562233822494e-16);
const LN2: (f64, f64, f64) =
    (std::f64::consts::LN_2, 2.3190468138462996e-17, 5.707708438416212e-34);

/// `exp(x)` for `|x| <= ln(2) / 2`: a Taylor series of `x / 1024`, squared ten times.
fn exp_reduced(x: Dd) -> Dd {
    let small = x.scale(1.0 / 1024.0);
    let mut sum = Dd(1.0, 0.0);
    let mut term = Dd(1.0, 0.0);
    for index in 1..=12 {
        term = term.mul(small).scale(1.0 / f64::from(index));
        sum = sum.add(term);
    }
    for _ in 0..10 {
        sum = sum.mul(sum);
    }
    sum
}

/// The double nearest to `10**y` for `-330 < y < 310`; `None` when the computed
/// power lies too close to a rounding boundary to decide.
pub fn pow10(y: f64) -> Option<f64> {
    if y == y.trunc() && (-22.0..=22.0).contains(&y) {
        // An exact power of ten, or the reciprocal one: the literal's own rounding
        return format!("1e{}", y as i32).parse().ok();
    }
    let x = Dd(y, 0.0).mul(LN10);
    let k = (x.0 / LN2.0).round();
    // r = x - k * ln 2, each product exact
    let (p0, e0) = two_prod(k, LN2.0);
    let r = x.add(Dd(-p0, -e0)).add(Dd(-k * LN2.1, -k * LN2.2));
    let value = exp_reduced(r);
    let (hi, lo) = two_sum(value.0, value.1);
    // The computation keeps at least 90 of its 106 bits
    let error = hi.abs() * f64::powi(2.0, -90);
    let up = next_up(hi) - hi;
    let down = hi - next_down(hi);
    let rounded = if lo >= 0.0 {
        if lo + error < up / 2.0 {
            hi
        } else if lo - error > up / 2.0 {
            next_up(hi)
        } else {
            return None;
        }
    } else if -lo + error < down / 2.0 {
        hi
    } else if -lo - error > down / 2.0 {
        next_down(hi)
    } else {
        return None;
    };
    let scaled = rounded * f64::powi(2.0, k as i32);
    // A result below the normal range would round twice
    (scaled.is_normal() && scaled / f64::powi(2.0, k as i32) == rounded).then_some(scaled)
}

fn next_up(value: f64) -> f64 {
    f64::from_bits(value.to_bits() + 1)
}

fn next_down(value: f64) -> f64 {
    f64::from_bits(value.to_bits() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn powers_of_ten_are_their_literals() {
        assert_eq!(pow10(-16.0), Some(1e-16));
        assert_eq!(pow10(2.0), Some(100.0));
        assert_eq!(pow10(0.0), Some(1.0));
    }

    #[test]
    fn fractional_powers_round_to_nearest() {
        // float(Decimal(10) ** Decimal(y)) at 80 digits
        assert_eq!(pow10(-0.75), Some(0.1778279410038923));
        assert_eq!(pow10(-7.324), Some(4.7424198526024486e-08));
        assert_eq!(pow10(-6.36), Some(4.3651583224016567e-07));
        assert_eq!(pow10(-14.093), Some(8.072350302488383e-15));
    }
}
