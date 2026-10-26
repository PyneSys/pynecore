//! `_native._fdlibm`, `_native._quantize_embed` and `_native._const_eval`: the native
//! fdlibm port and the constant fold's evaluation, for the parity tests against
//! `core/fdlibm.py` and `transformers/const_fold.py`. Not used by the pipeline.
//!
//! The doubles cross as packed little-endian `float64` buffers, so a sweep over
//! millions of arguments costs one call.

use pyne_ast::{Constant, Int};
use pyne_transform::fdlibm;
use pyne_transform::steps::const_fold::{self, Num};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyInt};

use crate::Unsupported;
use crate::convert::FromPython;

fn doubles(buffer: &[u8]) -> PyResult<Vec<f64>> {
    if !buffer.len().is_multiple_of(8) {
        return Err(PyValueError::new_err("a buffer of packed float64 values"));
    }
    Ok(buffer.as_chunks::<8>().0.iter().map(|&chunk| f64::from_le_bytes(chunk)).collect())
}

fn packed<'py>(py: Python<'py>, values: impl Iterator<Item = f64>) -> Bound<'py, PyBytes> {
    let buffer: Vec<u8> = values.flat_map(f64::to_le_bytes).collect();
    PyBytes::new(py, &buffer)
}

/// One public function of the native fdlibm port over packed arguments.
///
/// :param name: The `core.fdlibm` name (`sin`, ..., `pow`).
/// :param x: The arguments, packed float64.
/// :param y: The exponents of `pow`, packed float64, as many as `x`.
/// :return: The results, packed float64.
#[pyfunction]
#[pyo3(name = "_fdlibm", signature = (name, x, y=None))]
pub fn fdlibm_call<'py>(
    py: Python<'py>,
    name: &str,
    x: &[u8],
    y: Option<&[u8]>,
) -> PyResult<Bound<'py, PyBytes>> {
    let x = doubles(x)?;
    if name == "pow" {
        let y = doubles(y.ok_or_else(|| PyValueError::new_err("pow takes exponents"))?)?;
        if y.len() != x.len() {
            return Err(PyValueError::new_err("as many exponents as bases"));
        }
        return Ok(packed(py, x.iter().zip(&y).map(|(&x, &y)| fdlibm::pow(x, y))));
    }
    let function: fn(f64) -> f64 = match name {
        "sin" => fdlibm::sin,
        "cos" => fdlibm::cos,
        "tan" => fdlibm::tan,
        "exp" => fdlibm::exp,
        "log" => fdlibm::log,
        "asin" => fdlibm::asin,
        "acos" => fdlibm::acos,
        "atan" => fdlibm::atan,
        _ => return Err(PyValueError::new_err(format!("no fdlibm function {name}"))),
    };
    Ok(packed(py, x.into_iter().map(function)))
}

/// The correctly rounded `10**y` of every packed float64 exponent, NaN where it is
/// undecided (the power WebAssembly folds `math.round` with).
#[pyfunction]
#[pyo3(name = "_pow10")]
pub fn pow10<'py>(py: Python<'py>, values: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    let values = doubles(values)?;
    Ok(packed(py, values.into_iter().map(|y| pyne_transform::pow10::pow10(y).unwrap_or(f64::NAN))))
}

/// `quantize_embed` of every packed float64 value.
#[pyfunction]
#[pyo3(name = "_quantize_embed")]
pub fn quantize_embed<'py>(py: Python<'py>, values: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    let values = doubles(values)?;
    Ok(packed(
        py,
        values.into_iter().map(|value| match const_fold::quantize_embed(Num::Float(value)) {
            Num::Float(value) => value,
            Num::Int(_) => unreachable!(),
        }),
    ))
}

/// `quantize_embed(_try_eval(expr, {}))` of one expression, `None` where the
/// fold leaves it in place.
#[pyfunction]
#[pyo3(name = "_const_eval")]
pub fn const_eval<'py>(py: Python<'py>, expr: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let mut from = FromPython::new();
    let node = from.node(expr)?;
    match const_fold::eval_embedded(&from.ast, node) {
        Ok(Some(Constant::Int(Int::Small(value)))) => value.into_bound_py_any(py),
        Ok(Some(Constant::Int(Int::Big(digits)))) => {
            py.get_type::<PyInt>().call1((digits.as_str(),))
        }
        Ok(Some(Constant::Float(value))) => value.into_bound_py_any(py),
        Ok(Some(other)) => unreachable!("the fold embeds a number, not {other:?}"),
        Ok(None) => Ok(py.None().into_bound(py)),
        Err(err) => Err(Unsupported::new_err(err.to_string())),
    }
}
