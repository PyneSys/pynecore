//! Port of `transformers/security.py`, in parts.
//!
//! The Security step itself is not ported yet; what is here is what other steps
//! read from the Python module: the predicates that recognize a
//! `lib.request.security[_lower_tf]` call.

pub mod predicates;
