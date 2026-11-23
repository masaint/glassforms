use gf_core::csg::{Op, Part};
use gf_core::layers::Placement;
use gf_core::mods::{Kind, Modifier};
use gf_core::state::{Detail, State};
use glassforms::{engine::Engine, gpu::Gpu};

fn inside(e: &mut Engine) -> usize {
    e.bake(Detail::Full).unwrap();
    let (_, v) = e.read_volume().unwrap();
    v.into_iter()
        .filter(|&b| half::f16::from_bits(b).to_f32() < 0.)
        .count()
}

#[test]
fn a_transformed_group_places_its_base_layer_and_keeps_it_there_when_ungrouped() {
    let Ok(gpu) = Gpu::headless("vulkan") else { return };
    let settings = [("scene", "studio"), ("preset", "sphere"), ("trace", "csg"), ("res", "128")]
        .map(|(k,v)| (k.into(),v.into()));
    let mut e = Engine::new(gpu, State::with_args(&settings, true), true);
    e.resize(80, 60);
    let tree = e.state.csg.edit_construction();
    let group = tree.group(&[], &[0], "Housing").unwrap();
    tree.node_mut(&group).unwrap().place = Placement { pos: [0.3,0.1,0.0], rot: [0.0,0.0,90.0], size: 1.2 };
    tree.node_mut(&[0,0]).unwrap().place = Placement { pos: [0.22,0.0,0.0], size: 0.16, ..Default::default() };
    let occupied = |e: &mut Engine| {
        e.bake(Detail::Full).unwrap();
        let (res, voxels) = e.read_volume().unwrap();
        let r = res as usize;
        let mut centre = [0.0; 3];
        let mut count = 0;
        for (i, bits) in voxels.into_iter().enumerate() {
            if half::f16::from_bits(bits).to_f32() >= 0.0 { continue; }
            count += 1;
            for (axis, value) in [i % r, (i / r) % r, i / (r*r)].into_iter().enumerate() {
                centre[axis] += ((value as f64 + 0.5) / r as f64 * 2.0 - 1.0) * gf_core::HB;
            }
        }
        assert!(count > 100);
        centre.map(|v| v / count as f64)
    };
    let observed = occupied(&mut e);
    for (a,b) in observed.into_iter().zip([0.3,0.364,0.0]) { assert!((a-b).abs() < 0.022, "{observed:?}"); }
    assert!(e.state.csg.edit_construction().ungroup(&group));
    let ungrouped = occupied(&mut e);
    for (a,b) in observed.into_iter().zip(ungrouped) { assert!((a-b).abs() < 0.012, "{observed:?} vs {ungrouped:?}"); }
}

#[test]
fn nested_edits_isolate_the_chosen_geometry_and_keep_parent_modifiers_optional() {
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let settings = [
        ("scene", "studio"),
        ("preset", "block"),
        ("trace", "csg"),
        ("res", "128"),
    ]
    .map(|(k, v)| (k.into(), v.into()));
    let mut e = Engine::new(gpu, State::with_args(&settings, true), true);
    e.resize(80, 60);
    let block = inside(&mut e);
    let make = |x: f64| Part {
        place: Placement {
            pos: [x, 0., 0.],
            size: 0.58,
            ..Default::default()
        },
        ..Part::default()
    };
    let c = &mut e.state.csg;
    let a = c.add_construction_part(&[], make(-0.26)).unwrap();
    let b = c.add_construction_part(&[], make(0.26)).unwrap();
    let group = c
        .edit_construction()
        .group(&[], &[a[0], b[0]], "Both cutters")
        .unwrap();
    c.edit_construction().node_mut(&group).unwrap().op = Op::Subtract;
    let cut = inside(&mut e);
    let cache = gf_core::layers::MAX_LAYERS;
    let key = e.csg.layer_vols[cache].as_ref().unwrap().0.clone();
    let volume = e.csg.layer_vols[cache].as_ref().unwrap().1.tex.clone();
    assert!(e.state.enter_world_item(gf_core::groups::Item::Object(0)));
    assert!(e.state.select_construction_node(&[0], false));
    assert_eq!(inside(&mut e), cut);
    assert_eq!(e.csg.layer_vols[cache].as_ref().unwrap().1.tex, volume);
    e.state.edit_scope.preview_parents = true;
    let mut first_child = group.clone();
    first_child.push(0);
    assert!(e.state.select_construction_node(&first_child, false));
    assert_eq!(inside(&mut e), cut);
    assert_eq!(e.csg.layer_vols[cache].as_ref().unwrap().1.tex, volume);
    e.state.leave_object();
    e.state.csg.edit_construction().node_mut(&[0]).unwrap().name = "Base renamed".into();
    assert_eq!(inside(&mut e), cut);
    assert_eq!(&e.csg.layer_vols[cache].as_ref().unwrap().0, &key);
    e.state.csg.parts[0].place.pos[0] -= 0.08;
    inside(&mut e);
    assert_ne!(&e.csg.layer_vols[cache].as_ref().unwrap().0, &key);
    e.state.csg.parts[0].place.pos[0] += 0.08;
    assert!(cut < block * 9 / 10, "cut {cut} vs original {block}");
    e.state.edit_scope.inside = Some(gf_core::edit_scope::ObjectScope::new(0));
    e.state.edit_scope.inside.as_mut().unwrap().isolated = Some(group.clone());
    let both = inside(&mut e);
    let mut leaf = group.clone();
    leaf.push(0);
    e.state.edit_scope.inside.as_mut().unwrap().isolated = Some(leaf.clone());
    let alone = inside(&mut e);
    assert!(
        both > alone * 12 / 10,
        "group {both} vs first child {alone}"
    );
    let modifier = Modifier::of(Kind::Shell);
    e.state
        .csg
        .edit_construction()
        .node_mut(&group)
        .unwrap()
        .mods
        .list
        .push(modifier);
    e.state.edit_scope.inside.as_mut().unwrap().construction = leaf.clone();
    let shelled = inside(&mut e);
    assert!(
        shelled < alone * 9 / 10,
        "shelled {shelled} vs solid {alone}"
    );
    e.state.edit_scope.preview_parents = false;
    assert_eq!(inside(&mut e), alone);
    e.state.edit_scope.preview_parents = true;
    let mods = &mut e
        .state
        .csg
        .edit_construction()
        .node_mut(&group)
        .unwrap()
        .mods;
    mods.list.clear();
    mods.list.push(Modifier::of(Kind::Noise));
    let rough = inside(&mut e);
    assert!(
        rough.abs_diff(alone) > 100,
        "voxel-only noise: {rough} versus {alone}"
    );
}

#[test]
fn cut_material_switch_changes_the_photo_of_the_cavity() {
    use gf_core::construction::CutMaterial;
    use std::collections::BTreeMap;
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let settings = [
        ("scene", "studio"),
        ("preset", "block"),
        ("trace", "csg"),
        ("res", "128"),
        ("density", "1"),
        ("tintR", "1"),
        ("tintG", "0.32"),
        ("tintB", "0.16"),
        ("disp", "0"),
        ("bloom", "0"),
        ("grain", "0"),
        ("ns", "2"),
    ]
    .map(|(k, v)| (k.into(), v.into()));
    let mut e = Engine::new(gpu, State::with_args(&settings, true), true);
    e.auto = false;
    e.resize(180, 140);
    let cutter = Part {
        place: Placement {
            pos: [0., 0., 0.71],
            size: 0.58,
            ..Default::default()
        },
        ..Part::default()
    };
    let path = e.state.csg.add_construction_part(&[], cutter).unwrap();
    let n = e.state.csg.edit_construction().node_mut(&path).unwrap();
    n.op = Op::Subtract;
    n.material = Some(BTreeMap::from([
        ("tintR".into(), 0.08),
        ("tintG".into(), 0.40),
        ("tintB".into(), 1.),
    ]));
    let read = |e: &mut Engine| {
        e.bake(Detail::Full).unwrap();
        e.render(8);
        e.screenshot().unwrap().2
    };
    let blue = read(&mut e);
    e.state
        .csg
        .edit_construction()
        .node_mut(&path)
        .unwrap()
        .cut_material = CutMaterial::Recipient;
    let red = read(&mut e);
    if let Some(dir) = std::env::var_os("GF_CONSTRUCTION_EVIDENCE") {
        let dir = std::path::Path::new(&dir);
        std::fs::create_dir_all(dir).unwrap();
        for (name, pixels) in [("subtractor", &blue), ("recipient", &red)] {
            let file = std::fs::File::create(dir.join(format!("{name}.png"))).unwrap();
            let mut enc = png::Encoder::new(std::io::BufWriter::new(file), 180, 140);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            enc.write_header()
                .unwrap()
                .write_image_data(pixels)
                .unwrap();
        }
    }
    let n = e.state.csg.edit_construction().node_mut(&path).unwrap();
    n.cut_material = CutMaterial::Subtractor;
    n.material = Some(BTreeMap::from([
        ("metal".into(), 1.),
        ("rough".into(), 0.1),
        ("bodyR".into(), 0.92),
        ("bodyG".into(), 0.55),
        ("bodyB".into(), 0.21),
    ]));
    let metal = read(&mut e);
    let different = metal
        .chunks_exact(4)
        .zip(blue.chunks_exact(4))
        .filter(|(a, b)| a[0].abs_diff(b[0]) > 10)
        .count();
    assert!(different > 150, "metal changed {different} pixels");
    if let Some(dir) = std::env::var_os("GF_CONSTRUCTION_EVIDENCE") {
        let file = std::fs::File::create(std::path::Path::new(&dir).join("metal.png")).unwrap();
        let mut enc = png::Encoder::new(std::io::BufWriter::new(file), 180, 140);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header()
            .unwrap()
            .write_image_data(&metal)
            .unwrap();
    }
    for material_name in ["stoneware", "wax", "oil"] {
        let mut params = gf_core::settings::Settings::default();
        let named = gf_core::material::NAMED
            .iter()
            .find(|mat| mat.id == material_name)
            .unwrap();
        named.apply(&mut params);
        e.state
            .csg
            .edit_construction()
            .node_mut(&path)
            .unwrap()
            .material = Some(
            gf_core::material::body_names()
                .filter_map(|k| Some((k.to_owned(), params.get(k)?)))
                .collect(),
        );
        let pixels = read(&mut e);
        let difference = pixels
            .chunks_exact(4)
            .zip(blue.chunks_exact(4))
            .filter(|(a, b)| a[0].abs_diff(b[0]) > 10)
            .count();
        assert!(
            difference > 70,
            "{material_name} changed {difference} pixels"
        );
        if let Some(dir) = std::env::var_os("GF_CONSTRUCTION_EVIDENCE") {
            let file = std::fs::File::create(
                std::path::Path::new(&dir).join(format!("{material_name}.png")),
            )
            .unwrap();
            let mut enc = png::Encoder::new(std::io::BufWriter::new(file), 180, 140);
            enc.set_color(png::ColorType::Rgba);
            enc.set_depth(png::BitDepth::Eight);
            enc.write_header()
                .unwrap()
                .write_image_data(&pixels)
                .unwrap();
        }
    }
    let mut blue_gain = 0i64;
    let mut red_loss = 0i64;
    let mut changed = 0usize;
    for (a, b) in blue.chunks_exact(4).zip(red.chunks_exact(4)) {
        let r = b[0] as i64 - a[0] as i64;
        let b = a[2] as i64 - b[2] as i64;
        blue_gain += b;
        red_loss += r;
        changed += usize::from(r.abs() + b.abs() > 12);
    }
    assert!(
        changed > 100 && changed < 180 * 140 / 2 && blue_gain > 500 && red_loss > 500,
        "changed={changed} blue gain={blue_gain} red loss={red_loss}"
    );
}

#[test]
fn construction_pivot_measures_the_incoming_subtree_and_restores_authored_frames() {
    use gf_core::modifier_pivot::{Pivot, PivotOwner};
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let settings = [
        ("scene", "studio"),
        ("preset", "block"),
        ("trace", "csg"),
        ("res", "128"),
    ]
    .map(|(k, v)| (k.into(), v.into()));
    let mut e = Engine::new(gpu, State::with_args(&settings, true), true);
    e.resize(80, 60);
    let c = &mut e.state.csg;
    let part = |x: f64| Part {
        place: Placement {
            pos: [x, 0., 0.],
            size: 0.25,
            ..Default::default()
        },
        ..Default::default()
    };
    let a = c.add_construction_part(&[], part(0.48)).unwrap();
    let b = c.add_construction_part(&[], part(0.25)).unwrap();
    let group = c
        .edit_construction()
        .group(&[], &[a[0], b[0]], "Positive pair")
        .unwrap();
    let mut follower = Modifier::of(Kind::Twist);
    follower.pivot = Pivot::CentreOfMass;
    c.edit_construction()
        .node_mut(&group)
        .unwrap()
        .mods
        .list
        .push(follower);
    e.state.edit_scope.inside = Some(gf_core::edit_scope::ObjectScope::new(0));
    e.state.edit_scope.inside.as_mut().unwrap().construction = group.clone();
    let owner = e.state.construction_pivot_owner().unwrap();
    e.bake(Detail::Full).unwrap();
    let first = owner.measured(&e.state, 0).unwrap();
    assert!(first[0] > 0.26 && first[0] < 0.47, "pair centre {first:?}");
    let n = e.state.csg.edit_construction().node_mut(&group).unwrap();
    assert_eq!(n.mods.list[0].pivot, Pivot::CentreOfMass);
    n.mods.list.insert(0, Modifier::of(Kind::Mirror));
    e.bake(Detail::Full).unwrap();
    let mirrored = owner.measured(&e.state, 1).unwrap();
    assert!(
        mirrored[0].abs() < 0.025,
        "after first-stage mirror: {mirrored:?}"
    );
    // Rebasing the group origin must preserve both its mirrored volume and the
    // following stage's measured centre, even with two stages in the stack.
    let (_, before) = e.read_volume().unwrap();
    assert!(e.state.set_construction_origin([0.24, -0.13, 0.21]));
    e.bake(Detail::Full).unwrap();
    let (_, after) = e.read_volume().unwrap();
    assert!(before == after, "setting the group origin moved its geometry");
    assert_eq!(owner.measured(&e.state, 1).unwrap(), mirrored);
    let mut leaf = group.clone();
    leaf.push(0);
    let n = e.state.csg.edit_construction().node_mut(&leaf).unwrap();
    let mut fixed = Modifier::of(Kind::Twist);
    fixed.pivot = Pivot::World;
    fixed.centre = [0.07, 0.08, -0.03];
    n.mods.list.push(fixed);
    e.state.edit_scope.inside.as_mut().unwrap().construction = leaf;
    let child = e.state.construction_pivot_owner().unwrap();
    let target = [0.23, -0.05, 0.08];
    child.set_position(&mut e.state, 0, target);
    e.bake(Detail::Full).unwrap();
    let world = child.position(&e.state, 0).unwrap();
    assert!(
        world.iter().zip(target).all(|(a, b)| (a - b).abs() < 1e-8),
        "{world:?}"
    );
    assert!(matches!(child, PivotOwner::Construction(0, _)));
}

#[test]
fn drawn_cut_is_owned_by_its_stack_and_honours_the_chosen_reference() {
    use gf_core::drawn_cut::{CutSpace, DrawnCut};
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let args = [
        ("scene", "studio"),
        ("preset", "block"),
        ("trace", "csg"),
        ("res", "128"),
    ]
    .map(|(a, b)| (a.into(), b.into()));
    let mut e = Engine::new(gpu, State::with_args(&args, true), true);
    e.resize(80, 60);
    let blank = || Part {
        place: Placement {
            size: 0.62,
            ..Default::default()
        },
        ..Default::default()
    };
    let c = &mut e.state.csg;
    let a = c.add_construction_part(&[], blank()).unwrap();
    let b = c.add_construction_part(&[], blank()).unwrap();
    let group = c
        .edit_construction()
        .group(&[], &[a[0], b[0]], "two copies")
        .unwrap();
    e.state.edit_scope.inside = Some(gf_core::edit_scope::ObjectScope::new(0));
    e.state.edit_scope.inside.as_mut().unwrap().isolated = Some(group.clone());
    let untouched = inside(&mut e);
    let mut cut = Modifier::of(Kind::DrawCut);
    cut.size = 0.19;
    cut.cut = Some(DrawnCut {
        points: vec![[0., -1.2, 0.], [0., 0., 0.], [0., 1.2, 0.]],
        normal: [0., 0., 1.],
        space: CutSpace::Local,
    });
    let mut child = group.clone();
    child.push(0);
    e.state
        .csg
        .edit_construction()
        .node_mut(&child)
        .unwrap()
        .mods
        .list = vec![cut.clone()];
    let overlapping = inside(&mut e);
    assert!(
        overlapping.abs_diff(untouched) < 15,
        "one intact union sibling must remain {overlapping} {untouched}"
    );
    e.state
        .csg
        .edit_construction()
        .node_mut(&group)
        .unwrap()
        .mods
        .list = vec![cut.clone()];
    let opened = inside(&mut e);
    assert!(
        opened < untouched * 9 / 10,
        "group cut {opened} vs {untouched}"
    );
    e.state
        .csg
        .edit_construction()
        .node_mut(&child)
        .unwrap()
        .mods
        .list
        .clear();
    let load = |e: &mut Engine, x: f64| {
        e.bake(Detail::Full).unwrap();
        let (r, data) = e.read_volume().unwrap();
        let at = ((x / (2. * gf_core::HB) + 0.5) * r as f64) as usize;
        let mid = r as usize / 2;
        half::f16::from_bits(data[at + r as usize * (mid + r as usize * mid)]).to_f32()
    };
    e.state.csg.place.pos[0] = 0.22;
    let local = load(&mut e, 0.22);
    assert!(
        local > 0.04,
        "local cut follows the object, sampled {local}"
    );
    e.state
        .csg
        .edit_construction()
        .node_mut(&group)
        .unwrap()
        .mods
        .list[0]
        .cut
        .as_mut()
        .unwrap()
        .space = CutSpace::World;
    let moved = load(&mut e, 0.22);
    let fixed = load(&mut e, 0.);
    assert!(
        moved < -0.04 && fixed > 0.04,
        "fixed plane, translated sample {moved}, fixed sample {fixed}"
    );
    // A long freehand stroke remains one authored operation. This collinear
    // one exercises more than the default eight GPU batches and must leave the
    // same slot as its two-segment equivalent.
    let previous = inside(&mut e);
    let n = e.state.csg.edit_construction().node_mut(&group).unwrap();
    n.mods.list[0].cut.as_mut().unwrap().points = (0..=70)
        .map(|i| [0., -1.2 + 2.4 * i as f64 / 70., 0.])
        .collect();
    let many = inside(&mut e);
    assert!(
        many.abs_diff(previous) < (previous / 40).max(35),
        "single versus long drawn cut: {previous} {many}"
    );
    let saved = e.state.setup();
    let mut restored = State::new();
    restored.apply_setup(&saved).unwrap();
    assert_eq!(restored.csg.construction, e.state.csg.construction);
}

#[test]
fn whole_object_freehand_slot_preserves_the_rest_of_the_block_in_the_tracer() {
    use gf_core::drawn_cut::{CutSpace, DrawnCut};
    let Ok(gpu) = Gpu::headless("vulkan") else {
        return;
    };
    let args = [
        ("scene", "studio"),
        ("preset", "block"),
        ("trace", "csg"),
        ("res", "128"),
    ]
    .map(|(a, b)| (a.into(), b.into()));
    let mut e = Engine::new(gpu, State::with_args(&args, true), true);
    e.resize(160, 120);
    e.state.edit_scope.inside = Some(gf_core::edit_scope::ObjectScope::new(0));
    let pre = inside(&mut e);
    e.render(4);
    let before = e
        .screenshot()
        .unwrap()
        .2
        .chunks_exact(4)
        .filter(|c| c[0] > 25)
        .count();
    let mut cut = Modifier::of(Kind::DrawCut);
    cut.size = 0.1;
    cut.cut = Some(DrawnCut {
        points: vec![[0., -0.9, 0.], [0., 0., 0.], [0.4, 0.8, 0.]],
        normal: [0., 0., 1.],
        space: CutSpace::Local,
    });
    e.state.selected_modifiers_mut().list.push(cut);
    let post = inside(&mut e);
    e.render(4);
    let after = e
        .screenshot()
        .unwrap()
        .2
        .chunks_exact(4)
        .filter(|c| c[0] > 25)
        .count();
    assert!(
        post > pre * 7 / 10 && post < pre * 99 / 100 && after > before / 2,
        "solid {pre} -> {post}, lit {before} -> {after}"
    );
}

#[test]
fn world_bodies_keep_the_cut_surface_material() {
    use gf_core::construction::CutMaterial;
    use std::collections::BTreeMap;
    let Ok(gpu)=Gpu::headless("vulkan") else{return};
    let pairs=[("scene","studio"),("preset","block"),("trace","csg"),("res","128"),("bloom","0"),("grain","0"),("disp","0"),
        ("layer","studio:sphere"),("L2.x","1.2"),("L2.size","0.34"),("J12.op","apart"),("B2.material","chrome")].map(|(k,v)|(k.into(),v.into()));
    let mut state=State::with_args(&pairs,true);
    state.select_layer(0);
    let part=Part{place:Placement{pos:[0.,0.,0.71],size:0.58,..Default::default()},..Part::default()};
    let path=state.csg.add_construction_part(&[],part).unwrap();
    let cut=state.csg.edit_construction().node_mut(&path).unwrap();
    cut.op=Op::Subtract;
    cut.material=Some(BTreeMap::from([("metal".into(),3.0),("rough".into(),0.14)]));
    let mut e=Engine::new(gpu,state,true);e.auto=false;e.resize(240,180);
    let image=|e:&mut Engine|{e.bake(Detail::Full).unwrap();e.render(12);e.screenshot().unwrap().2};
    if let Some(dir)=std::env::var_os("GF_CONSTRUCTION_EVIDENCE"){
        std::fs::write(std::path::PathBuf::from(dir).join("world_cut.json"),e.state.setup_json()).unwrap();
    }
    let own=image(&mut e);
    e.state.csg.edit_construction().node_mut(&path).unwrap().cut_material=CutMaterial::Recipient;
    let recipient=image(&mut e);
    if let Some(dir)=std::env::var_os("GF_CONSTRUCTION_EVIDENCE"){
        let dir=std::path::PathBuf::from(dir);
        for (name,pixels) in [("world_subtractor.png",&own),("world_recipient.png",&recipient)] {
            let output=std::fs::File::create(dir.join(name)).unwrap();
            let mut enc=png::Encoder::new(output,240,180);
            enc.set_color(png::ColorType::Rgba);enc.set_depth(png::BitDepth::Eight);
            enc.write_header().unwrap().write_image_data(pixels).unwrap();
        }
    }
    let changed=own.chunks_exact(4).zip(recipient.chunks_exact(4)).filter(|(a,b)| (0..3).any(|i|a[i].abs_diff(b[i])>12)).count();
    assert!(changed>180,"only {changed} materially changed pixels");
    // A joined world uses the union field as its one physical body.
    assert!(e.state.apply_arg("J12.op","join"));
    let joined_recipient=image(&mut e);
    e.state.csg.edit_construction().node_mut(&path).unwrap().cut_material=CutMaterial::Subtractor;
    let joined_own=image(&mut e);
    let difference=joined_recipient.chunks_exact(4).zip(joined_own.chunks_exact(4)).filter(|(a,b)|(0..3).any(|i|a[i].abs_diff(b[i])>12)).count();
    assert!(difference>180,"joined World lost the face ({difference} changed pixels)");
    // A photographed ray reaches the paint after first crossing a separate front lens.
    for (key,value) in [("J12.op","apart"),("L2.x","0"),("L2.z","1.4"),("L2.size","0.64"),
        ("B2.material","crown"),("rx","0"),("ry","0"),("rz","0")] {assert!(e.state.apply_arg(key,value),"{key}");}
    let mut paint=gf_core::settings::Settings::default();
    let ware=gf_core::material::NAMED.iter().find(|m|m.id=="stoneware").unwrap();
    ware.apply(&mut paint);
    e.state.csg.edit_construction().node_mut(&path).unwrap().material=Some(gf_core::material::RESET.iter()
        .filter_map(|&(key,_)|paint.get(key).map(|v|(key.to_owned(),v))).collect());
    let through_paint=image(&mut e);
    e.state.csg.edit_construction().node_mut(&path).unwrap().cut_material=CutMaterial::Recipient;
    let through_glass=image(&mut e);
    let central=(69..111).flat_map(|y| (93..147).map(move |x|4*(y*240+x)));
    let changed=central.filter(|&j|(0..3).any(|k|through_paint[j+k].abs_diff(through_glass[j+k])>10)).count();
    assert!(changed>45,"front lens showed only {changed} affected central pixels");
    if let Some(dir)=std::env::var_os("GF_CONSTRUCTION_EVIDENCE"){
        let dir=std::path::PathBuf::from(dir);
        for(name,pixels) in [("world_lens_paint.png",&through_paint),("world_lens_recipient.png",&through_glass)]{
            let mut enc=png::Encoder::new(std::fs::File::create(dir.join(name)).unwrap(),240,180);
            enc.set_color(png::ColorType::Rgba);enc.set_depth(png::BitDepth::Eight);
            enc.write_header().unwrap().write_image_data(pixels).unwrap();
        }
    }

}
#[test]
fn baked_whole_object_modifiers_transport_the_cut_face() {
    use gf_core::construction::CutMaterial;
    use std::collections::BTreeMap;
    let Ok(gpu)=Gpu::headless("vulkan") else { return };
    let pairs=[("scene","studio"),("preset","block"),("trace","csg"),("res","128"),
        ("bloom","0"),("grain","0"),("disp","0"),("layer","studio:sphere"),
        ("L2.x","1.18"),("L2.size","0.23"),("J12.op","apart"),("B2.material","chrome")].map(|(a,b)|(a.into(),b.into()));
    let mut state=State::with_args(&pairs,true);
    state.select_layer(0);
    let part=Part{place:Placement{pos:[0.16,0.12,0.7],size:0.55,..Default::default()},..Part::default()};
    let path=state.csg.add_construction_part(&[],part).unwrap();
    let cut=state.csg.edit_construction().node_mut(&path).unwrap();
    cut.op=Op::Subtract;
    cut.material=Some(BTreeMap::from([("metal".into(),3.0),("rough".into(),0.14)]));
    state.shapes[state.fam].mods.list.extend([Modifier::of(Kind::Twist),Modifier::of(Kind::Smooth)]);
    assert!(state.tree_mods(0).unwrap().list.iter().all(Modifier::live));
    let mut e=Engine::new(gpu,state,true);e.auto=false;e.resize(320,240);
    let render=|e:&mut Engine| {e.bake(Detail::Full).unwrap();e.render(12);e.screenshot().unwrap().2};
    let own=render(&mut e);
    e.state.csg.edit_construction().node_mut(&path).unwrap().cut_material=CutMaterial::Recipient;
    let glass=render(&mut e);
    let changed=own.chunks_exact(4).zip(glass.chunks_exact(4)).filter(|(a,b)|(0..3).any(|i|a[i].abs_diff(b[i])>15)).count();
    eprintln!("baked modifier face, {changed} pixels changed");
    assert!(changed>160);
    if let Some(dir)=std::env::var_os("GF_CONSTRUCTION_EVIDENCE") {
        for(name,pixels) in [("baked_world_subtractor.png",&own),("baked_world_recipient.png",&glass)]{
            let mut enc=png::Encoder::new(std::fs::File::create(std::path::PathBuf::from(&dir).join(name)).unwrap(),320,240);
            enc.set_color(png::ColorType::Rgba);enc.set_depth(png::BitDepth::Eight);
            enc.write_header().unwrap().write_image_data(pixels).unwrap();
        }
    }
}


#[test]
fn baked_nested_group_keeps_mixed_child_surface_ids() {
    use gf_core::construction::CutMaterial;
    use std::collections::BTreeMap;
    let Ok(gpu)=Gpu::headless("vulkan") else{return};
    let args=[("scene","studio"),("preset","block"),("trace","csg"),("res","128"),
        ("bloom","0"),("grain","0"),("disp","0"),("layer","studio:sphere"),
        ("L2.x","1.18"),("L2.size","0.23"),("J12.op","apart"),("B2.material","chrome")].map(|(a,b)|(a.into(),b.into()));
    let mut state=State::with_args(&args,true);state.select_layer(0);
    let part=Part{place:Placement{pos:[0.16,0.12,0.7],size:0.55,..Default::default()},..Part::default()};
    let cut_path=state.csg.add_construction_part(&[],part).unwrap();
    let tree=state.csg.edit_construction();
    let node=tree.node_mut(&cut_path).unwrap();node.op=Op::Subtract;
    node.material=Some(BTreeMap::from([("metal".into(),3.0),("rough".into(),0.14)]));
    let group=tree.group(&[],&[0,cut_path[0]],"Housing").unwrap();
    tree.node_mut(&group).unwrap().mods.list.extend([Modifier::of(Kind::Twist),Modifier::of(Kind::Smooth)]);
    let mut child=group.clone();child.push(1);
    let mut e=Engine::new(gpu,state,true);e.auto=false;e.resize(320,240);
    let image=|e:&mut Engine|{e.bake(Detail::Full).unwrap();e.renderer.reseed(0);e.render(12);e.screenshot().unwrap().2};
    let own=image(&mut e);
    e.state.csg.edit_construction().node_mut(&child).unwrap().cut_material=CutMaterial::Recipient;
    let recipient=image(&mut e);
    let mask=|a:&[u8],b:&[u8]|->Vec<bool>{a.chunks_exact(4).zip(b.chunks_exact(4)).map(|(a,b)|(0..3).any(|i|a[i].abs_diff(b[i])>15)).collect()};
    let nested=mask(&own,&recipient);
    // Relocate this complete stack to the whole object: geometry, cut and recipient are unchanged.
    let mods=std::mem::take(&mut e.state.csg.edit_construction().node_mut(&group).unwrap().mods);
    e.state.csg.edit_construction().ungroup(&group);
    e.state.shapes[e.state.fam].mods=mods;
    let whole_recipient=image(&mut e);
    e.state.csg.edit_construction().node_mut(&cut_path).unwrap().cut_material=CutMaterial::Subtractor;
    let whole_own=image(&mut e);
    let whole=mask(&whole_own,&whole_recipient);
    let common=nested.iter().zip(&whole).filter(|(a,b)|**a && **b).count();
    let any=nested.iter().zip(&whole).filter(|(a,b)|**a || **b).count();
    eprintln!("nested vs whole transported cut face pixel overlap {common}/{any}");
    assert!(common>200 && common*100>any*95);
    if let Some(dir)=std::env::var_os("GF_CONSTRUCTION_EVIDENCE") {
        for(name,pixels) in [("baked_group_own.png",&own),("baked_group_recipient.png",&recipient)]{
            let mut enc=png::Encoder::new(std::fs::File::create(std::path::PathBuf::from(&dir).join(name)).unwrap(),320,240);
            enc.set_color(png::ColorType::Rgba);enc.set_depth(png::BitDepth::Eight);
            enc.write_header().unwrap().write_image_data(pixels).unwrap();
        }
    }
}
