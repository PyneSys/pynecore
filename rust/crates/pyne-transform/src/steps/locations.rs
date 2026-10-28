//! The `FixLocations` step, `locations.fix_locations(tree)`: the helper itself is
//! ported as [`pyne_ast::locations`], where the steps that call it reach it too.

use pyne_ast::{Ast, NodeId};

use crate::{Context, Error};

pub fn run(ast: &mut Ast, module: NodeId, _ctx: &Context<'_>) -> Result<NodeId, Error> {
    pyne_ast::locations::fix_locations(ast, module, 1, 0)?;
    Ok(module)
}
