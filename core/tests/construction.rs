use gf_core::construction::{CutMaterial, Source};
use gf_core::csg::{Csg, Op, Part};
use gf_core::state::State;
use std::collections::BTreeMap;

#[test]
fn old_operands_migrate_lazily_without_losing_the_group_operation() {
    let mut c = Csg::default();
    c.parts.push(Part {
        level: 1,
        op: Op::Subtract,
        smooth: 0.08,
        ..Part::default()
    });
    c.parts.push(Part {
        level: 1,
        ..Part::default()
    });
    c.parts.push(Part {
        op: Op::Intersect,
        ..Part::default()
    });
    let seen = c.construction_tree();
    assert!(c.construction.is_none());
    let group = seen.node(&[1]).unwrap();
    assert_eq!(group.op, Op::Subtract);
    assert_eq!(group.blend, 0.08);
    assert_eq!(seen.children(&[1]).unwrap().len(), 2);
    assert_eq!(seen.node(&[1, 0]).unwrap().op, Op::Union);
    assert_eq!(seen.node(&[2]).unwrap().op, Op::Intersect);
    assert_eq!(&seen, c.edit_construction());
    let first = c.parts[0].construction_id;
    let second = c.parts[1].construction_id;
    assert_ne!(first, second);
    c.remove_construction_node(&[1, 0]);
    assert_eq!(c.construction_part(second), Some(0));
    assert!(c.construction_part(first).is_none());
}

#[test]
fn setups_restore_editable_primitives_groups_and_cut_materials_on_every_object() {
    let mut st = State::new();
    let mut parts = vec![];
    for _ in 0..2 {
        let f = st.families.iter().position(|f| f.id == "studio").unwrap();
        parts.push(st.create_object(f, 0, true).unwrap());
    }
    for layer in [0, *parts.last().unwrap()] {
        let c = st.layer_csg_mut(layer).unwrap();
        let a = c.add_construction_part(&[], Part::default()).unwrap();
        let b = c.add_construction_part(&[], Part::default()).unwrap();
        let g = c
            .edit_construction()
            .group(&[], &[a[0], b[0]], "Cutter")
            .unwrap();
        let tree = c.edit_construction();
        let group = tree.node_mut(&g).unwrap();
        group.op = Op::Subtract;
        group.blend = 0.07;
        group.cut_material = CutMaterial::Recipient;
        group.material = Some(BTreeMap::from([("bodyR".into(), 0.21)]));
        let mut child = g.clone();
        child.push(1);
        let Source::Part(_) = tree.node(&child).unwrap().source else {
            panic!("lost primitive")
        };
    }
    let saved = st.setup_json();
    let mut restored = State::new();
    restored.apply_setup_json(&saved).unwrap();
    for layer in [0, *parts.last().unwrap()] {
        assert_eq!(
            restored.layer_csg(layer).construction_tree(),
            st.layer_csg(layer).construction_tree()
        );
        assert_eq!(restored.layer_csg(layer).parts, st.layer_csg(layer).parts);
    }
}

#[test]
fn a_cut_finish_uses_the_frame_of_its_nested_construction_layer() {
    use gf_core::{layers::Placement, modifier_pivot::PivotFrame};

    let mut state = State::new();
    let object = Placement {
        pos: [0.13, 0.05, -0.4],
        rot: [13., -27., 35.],
        size: 1.07,
    };
    let group = Placement {
        pos: [0.21, -0.14, 0.06],
        rot: [-12., 5., 62.],
        size: 1.2,
    };
    let cutter = Placement {
        pos: [0.01, -0.19, 0.42],
        rot: [38., 11., 2.],
        size: 0.48,
    };
    let csg = &mut state.csg;
    csg.place = object;
    let path = csg
        .add_construction_part(
            &[],
            Part {
                place: cutter,
                ..Part::default()
            },
        )
        .unwrap();
    let sibling = csg.add_construction_part(&[], Part::default()).unwrap();
    let tree = csg.edit_construction();
    let node = tree.node_mut(&path).unwrap();
    node.op = Op::Subtract;
    node.material = Some(BTreeMap::from([
        ("metal".into(), 10.),
        ("brush".into(), 1.),
    ]));
    let parent = tree.group(&[], &[path[0], sibling[0]], "Recesses").unwrap();
    tree.node_mut(&parent).unwrap().place = group;

    let program = state
        .construction_program(0, &state.csg_tree_layer(0).0)
        .unwrap();
    assert_eq!(program.faces.len(), 1);
    let rows = program.faces[0].object_to_face;
    let local = [0.21, -0.37, 0.04];
    let mut scene = local;
    for placement in [cutter, group, object] {
        scene = PivotFrame {
            placement,
            ..Default::default()
        }
        .to_world(scene);
    }
    for (row, expected) in rows.chunks_exact(4).zip(local) {
        let actual = row[0] * scene[0] + row[1] * scene[1] + row[2] * scene[2] + row[3];
        assert!(
            (actual - expected).abs() < 1e-10,
            "expected {expected}, got {actual}"
        );
    }
    // Object Mode uses the face's packed optical record. World Mode consumes its face table.
    let uploaded = rows.map(|v| v as f32);
    assert!(program.numbers.windows(12).any(|window| window == uploaded));
}

#[test]
fn a_recipient_cut_keeps_its_recipe_for_isolation_without_shading_the_join() {
    let mut state = State::new();
    let path = state
        .csg
        .add_construction_part(
            &[],
            Part {
                op: Op::Subtract,
                ..Default::default()
            },
        )
        .unwrap();
    let cut = state.csg.edit_construction().node_mut(&path).unwrap();
    cut.material = Some(BTreeMap::from([("rough".into(), 0.61)]));
    cut.cut_material = CutMaterial::Recipient;
    let parts = state.csg_tree_layer(0).0;
    let tree = state.csg.construction.as_ref().unwrap();
    let build = |isolate: Option<&[usize]>| {
        gf_core::construction_eval::Program::build(&state.csg, tree, &state.p, &parts, &[], isolate, true)
    };
    assert!(!build(None).has_materials);
    let alone = build(Some(&path));
    assert!(alone.has_materials);
    assert_eq!(alone.faces.len(), 1);
    assert_eq!(alone.faces[0].settings.rough, 0.61);
    assert_eq!(tree.node(&path).unwrap().material.as_ref().unwrap()["rough"], 0.61);
}
