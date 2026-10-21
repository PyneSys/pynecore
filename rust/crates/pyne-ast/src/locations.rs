//! Debugger-safe location filling, the counterpart of `transformers/locations.py`.
//!
//! `ast.fix_missing_locations` copies the parent's full span onto every
//! location-less node, which maps parts of a prologue a pass inserted onto the
//! function's last source line. [`fix_locations`] fills the same holes with
//! single point anchors instead:
//!
//! - a located node is never touched, except that a missing end position is
//!   completed from the node's own start;
//! - a location-less statement anchors to the earliest line located inside it,
//!   falling back to the innermost located ancestor's start;
//! - every other location-less node anchors to its enclosing statement's point.

use crate::walk::children;
use crate::{Ast, Category, NodeId, Unsupported};

/// The smallest line located anywhere inside a subtree, its root included.
pub fn earliest_line(ast: &Ast, root: NodeId) -> Option<i64> {
    let mut best: Option<i64> = None;
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if let Some(lineno) = ast[id].loc.lineno
            && best.is_none_or(|line| lineno < line)
        {
            best = Some(lineno);
        }
        stack.extend(children(ast, id));
    }
    best
}

/// Fill missing locations in a tree without leaking parent spans.
///
/// The walk is pre-order, children taken in field order, so a node shared by two
/// parents keeps what its first visit stamped on it.
///
/// `line` and `col` anchor the location-less nodes no located node encloses. A
/// node carrying a start line but no start column is answered `Unsupported`: the
/// Python helper fails on it.
pub fn fix_locations(ast: &mut Ast, root: NodeId, line: i64, col: i64) -> Result<(), Unsupported> {
    let mut stack = vec![(root, line, col)];
    while let Some((id, mut line, mut col)) = stack.pop() {
        if ast.kind(id).class().located {
            let loc = ast[id].loc;
            if let Some(lineno) = loc.lineno {
                let Some(col_offset) = loc.col_offset else {
                    return Err(Unsupported(format!(
                        "a {} with a line but no column",
                        ast.name(id)
                    )));
                };
                line = lineno;
                col = col_offset;
                let loc = &mut ast[id].loc;
                loc.end_lineno.get_or_insert(line);
                loc.end_col_offset.get_or_insert(col);
            } else {
                // Hoisted payloads keep their source lines: a new statement anchors
                // next to them rather than at the function entry
                if ast.kind(id).class().category == Category::Stmt
                    && let Some(inner) = earliest_line(ast, id)
                {
                    line = inner;
                    col = 0;
                }
                let loc = &mut ast[id].loc;
                loc.lineno = Some(line);
                loc.end_lineno = Some(line);
                loc.col_offset = Some(col);
                loc.end_col_offset = Some(col);
            }
        }
        for child in children(ast, id).into_iter().rev() {
            stack.push((child, line, col));
        }
    }
    Ok(())
}
