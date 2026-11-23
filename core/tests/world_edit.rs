use gf_core::{groups::{DropAt, Item}, layers::Placement, state::State, world_transform::Reference};

#[test]
fn grouping_a_world_selection_keeps_order_placement_material_and_editability() {
    let mut s = State::new();
    for _ in 1..4 {
        s.create_world_object(s.fam, s.shape().preset, false, None).unwrap();
    }
    for (i, name) in ["front", "unselected", "rear", "last"].into_iter().enumerate() {
        s.stack.names.insert(i, name.into());
        s.set_world_placement(Item::Object(i), Placement { pos: [i as f64, 0.0, 0.0], ..Default::default() });
    }
    assert!(s.apply_arg("B3.material", "copper"));
    let copper = s.body_materials().unwrap()[2].clone();
    s.select_world_item(Item::Object(2), false);
    s.select_world_item(Item::Object(0), true);
    let id = s.group_world_selection().unwrap();
    assert_eq!(s.world_selection(), vec![Item::Group(id)]);
    let frame = s.world_placement(Item::Group(id)).unwrap();
    assert!((frame.pos[0] - 1.0).abs() < 1e-12);
    assert_eq!(s.outline().0.iter().map(|n| n.item()).collect::<Vec<_>>(),
        [Item::Group(id), Item::Object(2), Item::Object(3)]);
    assert_eq!((0..4).map(|i| s.layer_title(i)).collect::<Vec<_>>(), ["front", "rear", "unselected", "last"]);
    assert_eq!(s.stack.layers[1].place.pos, [2.0,0.0,0.0]);
    assert_eq!(s.body_materials().unwrap()[1], &copper);
    assert_eq!(s.bodies(), vec![vec![0], vec![1], vec![2], vec![3]]);
    let mut loaded = State::new();
    loaded.apply_setup_json(&s.setup_json()).unwrap();
    assert_eq!(loaded.outline(), s.outline());
    assert_eq!(loaded.world_placement(Item::Group(id)), Some(frame));
    assert!(loaded.enter_world_item(Item::Group(id)));
    assert!(loaded.enter_world_item(Item::Object(1)));
    loaded.leave_object();
    assert_eq!(loaded.edit_scope.world, Some(id));
}

#[test]
fn reparenting_the_first_world_object_keeps_the_studio_and_its_saved_lights() {
    for custom in [false, true] {
        let args = [("scene", "solids"), ("preset", "ring"), ("layer", "studio:sphere"), ("J12.op", "apart")]
            .map(|(key, value)| (key.to_string(), value.to_string()));
        let mut s = State::with_args(&args, true);
        let original_family = s.layer_family(0).id;
        assert_ne!(original_family, s.layer_family(1).id);
        let stock = s.rig_default();
        if custom { s.rig_mut().ambient = [0.31, 0.04, 0.22]; }
        let lights = s.rig().clone();
        let group = s.new_world_group(None).unwrap();
        s.move_item(Item::Object(0), DropAt::Inside(group)).unwrap();
        assert_eq!(s.layer_family(0).id, "studio");
        assert_eq!(s.rig(), &lights);
        let saved = s.setup();
        assert_eq!(saved.rig_family.as_deref(), Some(original_family));
        let mut reopened = State::new();
        reopened.apply_setup(&saved).unwrap();
        assert_eq!(reopened.rig(), &lights);
        reopened.reset_rig();
        assert_eq!(reopened.rig(), &stock);
        reopened.move_item(Item::Group(group), DropAt::Before(Item::Object(0))).unwrap();
        assert_eq!(reopened.layer_family(0).id, original_family);
        assert!(reopened.setup().rig_family.is_none());
    }
}

#[test]
fn deleting_and_recreating_world_objects_preserves_the_scene_and_starts_fresh_geometry() {
    let args = [
        ("scene", "solids"), ("preset", "ring"), ("material", "moulded"),
        ("layer", "studio:sphere"), ("J12.op", "apart"), ("studio", "lavender"),
        ("light1.el", "17"), ("exposure", "1.43"), ("dist", "4.7"), ("rx", "23"),
    ].map(|(k, v)| (k.to_owned(), v.to_owned()));
    let mut s = State::with_args(&args, true);
    let lights = s.rig().clone();
    let view = s.r;
    let dist = s.p.dist;
    let material = s.p.mat;
    assert!(s.delete_object(0));
    assert_eq!(s.layer_family(0).id, "studio");
    assert_eq!(s.rig(), &lights);
    s.shape_mut().thick = 1.8;
    assert!(s.delete_object(0));
    assert_eq!(s.object_count(), 0);
    let empty = s.setup();
    let mut loaded = State::new();
    loaded.apply_setup(&empty).unwrap();
    assert_eq!(loaded.rig(), &lights);
    let family = loaded.families.iter().position(|f| f.id == "gems").unwrap();
    let cut = loaded.families[family].presets.iter().position(|p| p.id == "radiant").unwrap();
    let new = loaded.create_world_object(family, cut, true, None).unwrap();
    assert_eq!(new, 0);
    assert_eq!(loaded.preset().id, "radiant");
    assert!(loaded.is_csg_object(new));
    assert_eq!(loaded.shape().thick, 1.0);
    assert_eq!(loaded.rig(), &lights);
    assert_eq!(loaded.p.mat, material);
    assert_eq!(loaded.p.exposure, 1.43);
    assert_eq!(loaded.p.dist, dist);
    for (actual, expected) in loaded.r.iter().zip(view.iter()) {
        assert!((actual - expected).abs() < 1e-10);
    }
}

#[test]
fn grouping_mixed_depths_places_each_selected_branch_once_at_its_common_level() {
    let mut s = State::new();
    let parent = s.new_world_group(None).unwrap();
    let nested = s.new_world_group(Some(parent)).unwrap();
    s.move_item(Item::Object(0), DropAt::Inside(nested)).unwrap();
    let sibling = s.create_world_object(s.fam, s.shape().preset, false, Some(parent)).unwrap();
    s.select_world_item(Item::Group(nested), false);
    s.select_world_item(Item::Object(0), true);
    s.select_world_item(Item::Object(sibling), true);
    let id = s.group_world_selection().unwrap();
    assert_eq!(s.edit_scope.world, Some(parent));
    let tree = s.outline();
    assert_eq!(tree.ancestors(Item::Object(0)), Some(vec![parent, id, nested]));
    assert_eq!(tree.ancestors(Item::Object(sibling)), Some(vec![parent, id]));
    assert_eq!(tree.group(id).unwrap().children.len(), 2);
}

#[test]
fn selecting_construction_layers_and_entering_their_group_are_distinct() {
    let mut s = State::new();
    s.enter_world_item(Item::Object(0));
    let tree = s.csg.edit_construction();
    let outer = tree.group(&[], &[0], "Housing").unwrap();
    let inner = tree.group(&outer, &[0], "Body").unwrap();
    let part = s.csg.add_construction_part(&inner, Default::default()).unwrap();
    s.select_construction_node(&outer, false);
    assert!(s.edit_scope.construction_level().is_empty());
    assert!(s.enter_construction_level(&outer));
    assert!(s.enter_construction_level(&inner));
    s.select_construction_node(&part, false);
    assert_eq!(s.edit_scope.construction_level(), inner);
    assert_eq!(s.csg.sel, Some(0));
    assert!(!s.enter_construction_level(&part));
    s.scope_up();
    assert_eq!(s.edit_scope.construction_level(), outer);
    assert_eq!(s.edit_scope.construction(), inner);
    assert_eq!(s.csg.sel, None);
    s.scope_up();
    assert!(s.edit_scope.construction_level().is_empty());
    assert_eq!(s.edit_scope.construction(), outer);
    s.scope_up();
    assert_eq!(s.edit_scope.object(), None);
}

#[test]
fn construction_group_modifier_bounds_exclude_its_siblings() {
    use gf_core::{csg::Part, mods::{Kind, Modifier}};
    let mut s = State::new();
    let part = |x| Part { place: Placement { pos: [x, 0., 0.], size: 0.25, ..Default::default() }, ..Default::default() };
    let inside = s.csg.add_construction_part(&[], part(0.15)).unwrap();
    let group = s.csg.edit_construction().group(&[], &[inside[0]], "Repeated detail").unwrap();
    s.csg.add_construction_part(&[], part(0.75)).unwrap();
    let node = s.csg.edit_construction().node_mut(&group).unwrap();
    node.mods.list.push(Modifier::of(Kind::Taper));
    let program = |s: &State| s.construction_program(0, &s.csg_tree_layer(0).0).unwrap();
    let first = program(&s);
    s.csg.parts[1].place.pos[0] = 4.5;
    let other = program(&s);
    assert_eq!(first.numbers, other.numbers, "an unrelated sibling changed the modifier parameters");
    assert_eq!(first.boundaries[0].signature, other.boundaries[0].signature);
    s.csg.parts[0].place.pos[0] += 0.8;
    let within = program(&s);
    assert_ne!(first.numbers, within.numbers, "moving a child must update the group bounds");
    assert_ne!(first.boundaries[0].signature, within.boundaries[0].signature);
}

#[test]
fn an_edited_construction_origin_stays_fixed_through_rotation_and_save() {
    use gf_core::mods::{Kind, Modifier};
    let mut s = State::new();
    s.enter_world_item(Item::Object(0));
    let group = s.csg.edit_construction().group(&[], &[0], "Housing").unwrap();
    let node = s.csg.edit_construction().node_mut(&group).unwrap();
    let original = Placement { pos: [0.15, -0.1, 0.3], rot: [0., 37., 90.], size: 0.8 };
    node.place = original;
    node.mods.list.push(Modifier::of(Kind::Mirror));
    s.select_construction_node(&group, false);
    let point = [0.53, 0.24, -0.13];
    assert!(s.set_construction_origin(point));
    let node = s.csg.construction_tree().node(&group).unwrap().clone();
    let frame = s.construction_frame(0, &node);
    let close = |a: [f64; 3], b: [f64; 3]| a.iter().zip(b).all(|(x,y)| (x-y).abs() < 1e-8);
    assert_eq!(node.place, original);
    assert!(close(frame.to_world(frame.origin), point));
    assert!(close(node.mods.list[0].pivot_position(frame, None), [0.; 3]));

    let h = s.selected_node_handle().unwrap();
    assert!(close(h.in_scene().0, original.pos));
    let pose = Placement { rot: [22., -15., 71.], size: 1.4, ..h.pose };
    s.csg.edit_construction().node_mut(&group).unwrap().place = h.local(pose);
    let mut loaded = State::new();
    loaded.apply_setup_json(&s.setup_json()).unwrap();
    let tree = loaded.csg.construction_tree();
    let node = tree.node(&group).unwrap();
    let frame = loaded.construction_frame(0, node);
    assert!(close(frame.to_world(frame.origin), point));
    assert_ne!(node.place.pos, original.pos);
}

#[test]
fn editing_a_volume_keeps_its_sampled_base_in_the_construction() {
    let args = [("scene".into(), "studio".into()), ("preset".into(), "sphere".into()), ("trace".into(), "volume".into())];
    let mut s = State::with_args(&args, true);
    assert!(s.enter_world_item(Item::Object(0)));
    assert!(!s.has_tree(0));
    s.csg.add_construction_part(&[], gf_core::csg::Part::default()).unwrap();
    let (parts, joined, skipped) = s.csg_tree_layer(0);
    assert!(!joined);
    assert_eq!((parts.len(), skipped), (2, 0));
    assert_eq!(parts[0].grid.as_ref().unwrap().from, gf_core::csg::GridFrom::Up);
    assert!(!s.is_csg_object(0));
}

#[test]
fn construction_handles_include_the_world_placement_of_the_edited_object() {
    let mut s = State::new();
    let i = s.create_world_object(s.fam, s.shape().preset, true, None).unwrap();
    let leaf = Item::Object(i);
    let outer = Placement { pos: [0.1, 0.2, 0.0], rot: [0.0, 0.0, 90.0], size: 2.0 };
    s.set_world_placement(leaf, outer);
    s.enter_world_item(leaf);
    s.csg.place = Placement { pos: [0.4, 0.0, 0.0], size: 0.5, ..Default::default() };
    let mut part = gf_core::csg::Part::default();
    part.place.pos = [0.1, 0.3, 0.0];
    s.csg.add_construction_part(&[], part).unwrap();
    let h = s.construction_handle(Some(0)).unwrap();
    let (centre, size) = h.in_scene();
    assert!((centre[0] + 0.2).abs() < 1e-10 && (centre[1] - 1.1).abs() < 1e-10);
    assert!((size - s.csg.parts[0].place.size).abs() < 1e-10);
    let mut pose = h.pose;
    pose.pos[0] += 0.2;
    s.csg.parts[0].place = h.local(pose);
    assert!((s.csg.parts[0].place.pos[0] - 0.3).abs() < 1e-10);
    assert_eq!(s.stack.layers[i].place, outer);
}

#[test]
fn world_transform_respects_animated_components() {
    let mut s = State::new();
    let knob = s.place_knobs(0).remove(0);
    s.toggle_wave(&knob);
    let old = s.stack.layers[0].place;
    s.set_world_placement(Item::Object(0), Placement { pos: [0.6, 0.4, 0.2], ..old });
    assert_eq!(s.stack.layers[0].place.pos, [old.pos[0], 0.4, 0.2]);
}

#[test]
fn undo_in_an_object_keeps_the_editor_and_returns_to_its_group() {
    let mut s = State::new();
    let group = s.new_world_group(None).unwrap();
    let i = s.create_world_object(s.fam, s.shape().preset, false, Some(group)).unwrap();
    s.world_level(Some(group));
    s.enter_world_item(Item::Object(i));
    let before = s.undo_snapshot();
    s.set_world_origin(Item::Object(i), [0.3, 0.1, 0.2]);
    s.restore_snapshot(&before);
    assert_eq!(s.edit_scope.object(), Some(i));
    assert_eq!(s.stack.sel, i);
    assert_eq!(s.world_placement(Item::Object(i)).unwrap().pos, [0.; 3]);
    s.scope_up();
    assert_eq!(s.edit_scope.world, Some(group));
}

#[test]
fn world_origin_preserves_object_and_inherited_construction_pivots() {
    use gf_core::{modifier_pivot::PivotOwner, mods::{Kind, Modifier}};
    let mut s = State::new();
    let leaf = Item::Object(0);
    s.set_world_placement(leaf, Placement { pos: [0.2, -0.1, 0.3], rot: [10., 22., 17.], size: 1.3 });
    s.csg.place = Placement { pos: [-0.2, 0.14, 0.1], rot: [60., 12., 33.], size: 0.6 };
    s.shape_mut().mods.list.push(Modifier::of(Kind::Mirror));
    let tree = s.csg.edit_construction();
    let root = tree.node_mut(&[0]).unwrap();
    root.mods.list.push(Modifier::of(Kind::Twist));
    let inherited = PivotOwner::Construction(0, root.pivot_key());
    let owners = [PivotOwner::Object(0), inherited];
    let before = owners.map(|owner| owner.position(&s, 0).unwrap());
    assert!(s.set_world_origin(leaf, [0.7, -0.3, 0.14]));
    for (owner, expected) in owners.into_iter().zip(before) {
        let actual = owner.position(&s, 0).unwrap();
        assert!(actual.into_iter().zip(expected).all(|(a,b)| (a-b).abs() < 1e-8), "{expected:?} vs {actual:?}");
    }
}

#[test]
fn moving_origins_keeps_deformed_groups_and_all_their_occurrences_in_place() {
    use gf_core::{modifier_pivot::Pivot, mods::{Axes, Kind, Modifier}, world_mods::Scope};
    let mut s = State::new();
    let outer = s.new_world_group(None).unwrap();
    let inner = s.new_world_group(Some(outer)).unwrap();
    let leaf = Item::Object(0);
    s.move_item(leaf, DropAt::Inside(inner)).unwrap();
    s.set_world_placement(Item::Group(inner), Placement { pos: [0.1, 0.2, 0.3], rot: [5., 19., 48.], size: 1.4 });
    s.set_world_placement(leaf, Placement { pos: [0.23, -0.11, 0.37], rot: [27., 8., 13.], size: 0.8 });
    // An existing project may already have a moved origin and no separate
    // arrangement anchor. Its first origin edit must preserve that reference.
    s.stack.layers[0].local_origin = [0.08, -0.11, 0.06];
    let mut radial = Modifier::of(Kind::Repeat);
    radial.mode = 1;
    radial.a = Axes::new([3., 1., 1.], false);
    radial.centre = [0.21, -0.16, 0.04];
    let mut geometry = Modifier::of(Kind::Twist);
    geometry.scope = Scope::SubObjects;
    geometry.a = Axes::new([0., 93., 0.], false);
    geometry.cascade = true;
    let mut pinned = radial.clone();
    pinned.pivot = Pivot::PinnedMass;
    s.stack.world_modifiers_mut(leaf).unwrap().list.push(radial);
    s.stack.world_modifiers_mut(Item::Group(inner)).unwrap().list = vec![geometry, pinned];
    let mut bend = Modifier::of(Kind::Bend);
    bend.a = Axes::new([75., 0., 0.], false);
    s.stack.world_modifiers_mut(Item::Group(outer)).unwrap().list.push(bend);
    let before = s.stack.world_plan();
    let check = |s: &State| {
        let after = s.stack.world_plan();
        let close = |a: [f64;3], b: [f64;3]| assert!(a.into_iter().zip(b).all(|(x,y)| (x-y).abs() < 1e-8), "{a:?} vs {b:?}");
        assert_eq!(before.placements[0].len(), 9);
        assert_eq!(before.placements[0].len(), after.placements[0].len());
        for (a,b) in before.placements[0].iter().zip(&after.placements[0]) {
            close(a.pos, b.pos);
            close(a.rot, b.rot);
            assert!((a.size-b.size).abs() < 1e-10);
        }
        for (a,b) in before.field_ids[0].iter().zip(&after.field_ids[0]) {
            close(before.fields[*a].mods.list[0].centre, after.fields[*b].mods.list[0].centre);
        }
    };
    for (item, point) in [(leaf, [0.5, -0.2, 0.3]), (Item::Group(inner), [-0.4, 0.2, 0.6]), (leaf, [0.1, 0.5, -0.7])] {
        assert!(s.set_world_origin(item, point));
        check(&s);
    }
    let mut loaded = State::new();
    loaded.apply_setup_json(&s.setup_json()).unwrap();
    check(&loaded);
}

#[test]
fn group_frames_and_edited_origins_survive_reparent_and_save() {
    let mut s = State::new();
    let group = s.new_world_group(None).unwrap();
    let child = s.new_world_group(Some(group)).unwrap();
    let leaf = Item::Object(0);
    let initial = Placement { pos: [0.3, -0.1, 0.2], ..Default::default() };
    s.set_world_placement(leaf, initial);
    s.set_world_origin(leaf, [0.45, -0.1, 0.2]);
    assert_eq!(s.stack.layers[0].place, initial);
    s.move_item(leaf, DropAt::Inside(child)).unwrap();
    assert_eq!(s.stack.layers[0].place, initial);
    s.set_world_origin(Item::Group(group), [0.2, 0.0, 0.0]);
    let new = Placement { pos: [0.2, 0.0, 0.0], rot: [0.0,0.0,90.0], size: 2.0 };
    s.transform_world_items(&[Item::Group(group), leaf], Item::Group(group), new);
    let observed = s.stack.layers[0].place;
    assert!((observed.pos[0] - 0.4).abs() < 1e-9 && (observed.pos[1] - 0.2).abs() < 1e-9, "{observed:?}");
    assert_eq!(observed.size, 2.0);
    assert_eq!(s.world_placement(Item::Group(group)), Some(new));
    let mut loaded = State::new(); loaded.apply_setup(&s.setup()).unwrap();
    assert_eq!(s.stack.to_setup(), loaded.stack.to_setup());
    s.enter_world_item(leaf);
    assert_eq!(s.edit_scope.world, Some(child));
    s.scope_up();
    assert_eq!(s.edit_scope.world, None);
    assert_eq!(s.world_selection(), vec![Item::Group(group)]);
}

#[test]
fn world_objects_overlap_independently_and_align_to_selected_bounds() {
    let mut s = State::new();
    let second = s.create_world_object(s.fam, s.shape().preset, false, None).unwrap();
    assert_eq!(s.bodies(), vec![vec![0], vec![1]]);
    let names = [Item::Object(0), Item::Object(second)];
    for (i, pos) in [[0.0,0.0,0.0], [0.4,0.1,0.3]].into_iter().enumerate() {
        s.set_world_placement(names[i], Placement{ pos, ..Default::default() });
    }
    s.world_align(&names, 0, Reference::Max);
    assert_eq!(s.world_bounds(names[0]).unwrap()[3], s.world_bounds(names[1]).unwrap()[3]);
    assert_eq!(s.stack.layers[0].place.pos[1], 0.0);
    s.set_world_origin(names[0], [0.2,0.0,0.0]);
    s.enter_world_item(names[1]);
    assert_eq!(s.edit_scope.object(), Some(1));
    s.scope_up();
    assert_eq!(s.world_selection(), vec![names[1]]);
}

#[test]
fn world_guides_keep_a_rotated_drag_on_its_axis_or_plane() {
    use gf_core::world_transform::DragConstraint::{Axis, Plane};
    let mut s = State::new();
    let moving = Item::Object(0);
    let peer = Item::Group(s.new_world_group(None).unwrap());
    s.select_world_item(moving, false);
    let at = [0.4, 0.4, 0.0];
    s.set_world_origin(peer, [0.42, 0.8, 1.0]);
    let snapped = s.world_snap(moving, at, Axis([1., 1., 0.]), Some(Reference::Origin), 0.04);
    for (a, b) in snapped.into_iter().zip([0.42, 0.42, 0.]) { assert!((a-b).abs() < 1e-12); }
    // A guide can be close in X yet out of reach along the handle.
    assert_eq!(s.world_snap(moving, at, Axis([1., 1., 0.]), Some(Reference::Origin), 0.025), at);
    // An oblique plane has two degrees of freedom. X and Z can both snap while X=Y holds.
    s.set_world_origin(peer, [0.42, 0.8, 0.01]);
    let snapped = s.world_snap(moving, at, Plane { normal: [1., -1., 0.] }, Some(Reference::Origin), 0.04);
    for (a, b) in snapped.into_iter().zip([0.42, 0.42, 0.01]) { assert!((a-b).abs() < 1e-12); }
    // A peer selected for the same group motion cannot attract the drag.
    s.select_world_item(peer, true);
    assert_eq!(s.world_snap(moving, at, Axis([1., 1., 0.]), Some(Reference::Origin), 0.04), at);
}

#[test]
fn deleting_last_object_clears_world_state_and_layer_remap_updates_targets_and_pivots() {
    use gf_core::mods::{Axes, Kind, Modifier};
    let mut s = State::new();
    s.stack.names.insert(0, "hero".into());
    let mut rep = Modifier::of(Kind::Repeat);
    rep.a = Axes::new([3., 1., 1.], false);
    s.stack.world_modifiers_mut(Item::Object(0)).unwrap().list.push(rep.clone());
    s.shape_mut().mods.list.push(Modifier::of(Kind::Mirror));
    s.stack.world_pivots.insert((Item::Object(0), 0), [1.0, 2.0, 3.0]);
    s.modifier_pivots.insert((Item::Object(0), 0), [0.4, 0.5, 0.6]);
    s.construction_pivots.insert((0, 42, 0), [0.7, 0.8, 0.9]);
    s.world_modifier_target = Some(Item::Object(0));
    assert!(s.stack.has_world_modifiers());

    assert!(s.delete_object(0));
    assert!(s.empty);
    assert!(!s.stack.has_world_modifiers());
    assert!(s.stack.world_mods.is_empty());
    assert!(s.stack.names.is_empty());
    assert!(s.stack.world_pivots.is_empty());
    assert!(s.modifier_pivots.is_empty());
    assert!(s.construction_pivots.is_empty());
    assert_eq!(s.world_modifier_target, None);
    assert!(s.selected_modifiers().is_empty());
    // Even if a stale target were poked in while empty, `selected_modifiers_mut` clears it without panicking.
    s.world_modifier_target = Some(Item::Object(0));
    assert!(s.selected_modifiers_mut().is_empty());
    assert_eq!(s.world_modifier_target, None);

    let created = s.create_object(s.fam, 0, false).unwrap();
    assert_eq!(created, 0);
    assert!(!s.stack.has_world_modifiers());
    assert!(!s.stack.names.contains_key(&0));

    // Add a second object, populate targets & cached pivots on layer 1, swap and remove.
    let second = s.create_world_object(s.fam, s.shape().preset, false, None).unwrap();
    assert_eq!(second, 1);
    s.stack.world_modifiers_mut(Item::Object(1)).unwrap().list.push(rep);
    s.world_modifier_target = Some(Item::Object(1));
    s.modifier_pivots.insert((Item::Object(1), 0), [1.0, 0.0, 0.0]);
    s.construction_pivots.insert((1, 99, 0), [0.0, 1.0, 0.0]);

    s.swap_layers(0, 1);
    assert_eq!(s.world_modifier_target, Some(Item::Object(0)));
    assert_eq!(s.modifier_pivots.get(&(Item::Object(0), 0)), Some(&[1.0, 0.0, 0.0]));
    assert_eq!(s.construction_pivots.get(&(0, 99, 0)), Some(&[0.0, 1.0, 0.0]));
    assert_eq!(s.selected_modifiers_mut().list.len(), 1);

    assert!(s.remove_layer(0));
    assert_eq!(s.world_modifier_target, None);
    assert!(s.modifier_pivots.is_empty());
    assert!(s.construction_pivots.is_empty());
    assert!(s.selected_modifiers_mut().is_empty());
}

#[test]
fn deleting_the_object_being_edited_inside_a_construction_level_returns_to_the_world() {
    let mut s = State::new();
    let other = s.create_world_object(s.fam, s.shape().preset, false, None).unwrap();
    let edited = 1 - other;
    assert!(s.enter_world_item(Item::Object(edited)));
    let outer = s.csg.edit_construction().group(&[], &[0], "Housing").unwrap();
    assert!(s.enter_construction_level(&outer));
    assert!(s.delete_object(edited));
    assert_eq!(s.edit_scope.object(), None);
    assert!(s.edit_scope.construction_level().is_empty() && s.edit_scope.construction().is_empty());
    // Esc still leads somewhere: nothing to leave at the world root, and entering works again
    s.scope_up();
    assert!(s.enter_world_item(Item::Object(0)));
    s.scope_up();
    assert_eq!(s.edit_scope.object(), None);
}
