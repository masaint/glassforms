//! Read a following pivot from the selected subtree before this modifier.
//! Children are resolved first so their deformation is present in the group's
//! incoming geometry. The authored stacks are restored after the object builds.
use crate::engine::Engine;
use crate::gpu::R;
use crate::pivot_mass::MassKey;
use gf_core::construction::{Construction, Node, Source};
use gf_core::csg::Op;
use gf_core::layers::LayerShape;
use gf_core::modifier_pivot::{PivotFrame, Point};
use std::collections::HashMap;

impl Engine {
    pub(crate) fn install_construction_pivots(&mut self) -> R<Vec<(usize, Construction)>> {
        let mut plans = Vec::new();
        let mut positions = HashMap::new();
        for layer in 0..self.state.layer_count() {
            let Some(tree) = self.state.layer_csg(layer).construction.as_ref() else {
                continue;
            };
            if !needs_resolution(&tree.roots, self.state.modifier_frame(layer).origin) {
                continue;
            }
            let mut working = tree.clone();
            let base = self.state.layer_shape_setup(layer);
            for node in &mut working.roots {
                self.resolve_construction_node(layer, node, &base, &mut positions)?;
            }
            plans.push((layer, working));
        }
        self.state.construction_pivots = positions;
        for (layer, working) in &mut plans {
            let tree = self
                .state
                .layer_csg_mut(*layer)
                .and_then(|c| c.construction.as_mut())
                .expect("selected construction");
            std::mem::swap(tree, working);
        }
        Ok(plans)
    }

    fn resolve_construction_node(
        &mut self,
        layer: usize,
        node: &mut Node,
        base: &LayerShape,
        positions: &mut HashMap<(usize, u64, usize), Point>,
    ) -> R<()> {
        if let Source::Group(children) = &mut node.source {
            for child in children {
                self.resolve_construction_node(layer, child, base, positions)?;
            }
        }
        let frame = self.state.construction_frame(layer, node);
        if !node
            .mods
            .live()
            .any(|m| !m.pivot.is_local() || m.has_world_cut() || frame.origin != [0.; 3])
        {
            return Ok(());
        }
        let key = node.pivot_key();
        let local = PivotFrame {
            placement: node.placement(self.state.layer_csg(layer)),
            origin: [0.; 3],
        };
        let clock = self.state.clock;
        let working = node.mods.resolve_pivots(frame, |stage, prefix| {
            let mut incoming = base.clone();
            incoming.shape.mods = None;
            let mut subtree = node.clone();
            subtree.op = Op::Union;
            subtree.blend = 0.;
            subtree.mods = prefix.clone();
            incoming.construction = Some(Construction {
                roots: vec![subtree],
            });
            let mass =
                self.measure_incoming(MassKey::Construction(layer, key, stage), incoming, clock)?;
            Ok::<_, String>(mass.map(|(_, p)| local.to_local(p)))
        })?;
        for (stage, m) in working.list.iter().enumerate() {
            if m.live() {
                positions.insert((layer, key, stage), m.centre);
            }
        }
        node.mods = working;
        Ok(())
    }
}

fn needs_resolution(nodes: &[Node], origin: Point) -> bool {
    nodes.iter().any(|n| {
        n.mods
            .live()
            .any(|m| !m.pivot.is_local() || m.has_world_cut() || n.local_origin.unwrap_or(origin) != [0.; 3])
            || matches!(&n.source,Source::Group(children) if needs_resolution(children,origin))
    })
}
