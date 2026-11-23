use gf_core::{groups::Item, mods::{Axes, Kind, Modifier}, state::{Detail, State}};
use glassforms::{engine::Engine, gpu::Gpu};

#[test]
fn occurrences_read_the_shared_definition_and_disappear_when_removed() {
    let args=[("scene","studio"),("preset","sphere"),("p.radius","0.15"),("res","96")]
        .into_iter().map(|(k,v)|(k.to_owned(),v.to_owned())).collect::<Vec<_>>();
    let mut st=State::with_args(&args,true);
    let mut array=Modifier::of(Kind::Repeat);array.fit=false;
    array.a=Axes::new([3.,1.,1.],false);array.b=Axes::new([0.72,1.,1.],false);
    st.stack.world_modifiers_mut(Item::Object(0)).unwrap().list.push(array);
    assert_eq!(st.object_count(),1);
    let mut e=Engine::new(Gpu::headless("vulkan").unwrap(),st,true);
    e.bake(Detail::Full).unwrap();
    let read=|e:&Engine,x:f64| {
        let vol=e.scene_volume().unwrap();let hb=e.state.scene_hb();
        let ix=(((x/hb+1.)*0.5*vol.res as f64).floor() as u32).min(vol.res-1);
        let v=e.gpu.read(&vol.tex,ix,vol.res/2,vol.res/2,1,1,2).unwrap();
        half::f16::from_le_bytes([v[0],v[1]]).to_f32()
    };
    assert!(read(&e,-0.72)<0. && read(&e,0.)<0. && read(&e,0.72)<0.);
    assert!(read(&e,0.36)>0.);
    let owner=Item::Object(0);
    e.state.stack.world_modifiers_mut(owner).unwrap().list[0].scope=gf_core::world_mods::Scope::SubObjects;
    e.bake(Detail::Full).unwrap();
    let geometric=e.state.stack.world_plan();
    assert_eq!(geometric.placements[0].len(),1);
    assert_eq!(geometric.geometry[0].list.len(),1);
    assert!(read(&e,-0.72)<0. && read(&e,0.72)<0.);
    e.state.stack.world_modifiers_mut(owner).unwrap().list[0].scope=gf_core::world_mods::Scope::World;
    let radius=e.state.preset().params.iter().position(|p|p.id=="radius").unwrap();
    e.state.shape_mut().vals[radius]=0.28;e.bake(Detail::Full).unwrap();
    assert!(read(&e,0.72+0.21)<0.);
    e.state.edit_scope.inside = Some(gf_core::edit_scope::ObjectScope::new(0));
    e.bake(Detail::Full).unwrap();
    assert!(read(&e,-0.72)>0. && read(&e,0.)<0.);
    assert_eq!(e.state.stack.world_plan().placements[0].len(), 3);
    e.state.edit_scope.inside = None;
    e.bake(Detail::Full).unwrap();
    assert!(read(&e,-0.72)<0. && read(&e,0.72)<0.);
    e.state.stack.world_mods.clear();e.bake(Detail::Full).unwrap();
    assert!(read(&e,-0.72)>0. && read(&e,0.)<0.);
    assert_eq!(e.state.stack.layers[0].place.pos,[0.;3]);
}

#[test]
fn world_mass_weights_source_solids_and_prior_array_instances() {
    use gf_core::{groups::DropAt, modifier_pivot::{Pivot, PivotOwner}};
    let args = [("scene", "studio"), ("preset", "sphere"), ("p.radius", "0.24"), ("res", "96")]
        .into_iter().map(|(k, v)| (k.to_owned(), v.to_owned())).collect::<Vec<_>>();
    let mut st = State::with_args(&args, true);
    let sphere = st.families[st.fam].presets.iter().position(|p| p.id == "sphere").unwrap();
    st.add_layer_of(st.fam, sphere).unwrap();
    let group = st.new_group(None).unwrap();
    st.move_item(Item::Object(1), DropAt::Inside(group)).unwrap();
    st.move_item(Item::Object(0), DropAt::Inside(group)).unwrap();
    st.stack.hierarchy.as_mut().unwrap().group_mut(group).unwrap().world = true;
    // Both sources describe the same sphere. One instance is uniformly scaled
    // by two, so its mass is eight times that of a small occurrence.
    for i in 0..2 {
        st.select_layer(i);
        let radius = st.preset().params.iter().position(|p| p.id == "radius").unwrap();
        st.shape_mut().vals[radius] = 0.24;
        let place = &mut st.stack.layers[i].place;
        place.pos = [if i == 0 { -0.7 } else { 0.7 }, 0., 0.]; place.size = if i == 0 { 1. } else { 0.5 };
    }
    let mut array = Modifier::of(Kind::Repeat);
    array.fit = false; array.a = Axes::new([1., 3., 1.], false);
    array.b = Axes::new([0.35; 3], false);
    st.stack.world_modifiers_mut(Item::Object(1)).unwrap().list.push(array);
    let mut twist = Modifier::of(Kind::Twist);
    twist.a = Axes::new([0., 120., 0.], false); twist.pivot = Pivot::CentreOfMass;
    let owner = Item::Group(group);
    st.stack.world_modifiers_mut(owner).unwrap().list.push(twist);
    let mut e = Engine::new(Gpu::headless("vulkan").unwrap(), st, true);
    e.bake(Detail::Full).unwrap();
    let expected = (-0.7 * 8. + 0.7 * 3.) / 11.;
    let first = e.state.stack.world_pivots[&(owner, 0)];
    assert!((first[0] - expected).abs() < 0.009 && first[1].abs() < 0.009, "{first:?}, expected {expected}");
    let handle = PivotOwner::World(owner);
    assert_eq!(handle.position(&e.state, 0), Some(first));
    e.state.stack.world_modifiers_mut(owner).unwrap().list[0].a = Axes::new([0., 270., 0.], false);
    e.bake(Detail::Full).unwrap();
    assert_eq!(e.state.stack.world_pivots[&(owner, 0)], first);
    e.state.stack.world_modifiers_mut(Item::Object(1)).unwrap().list.clear();
    e.bake(Detail::Full).unwrap();
    let shifted = e.state.stack.world_pivots[&(owner, 0)][0];
    assert!((shifted - (-0.7 * 8. + 0.7) / 9.).abs() < 0.009, "{shifted}");
}

#[test]
fn world_pivot_maps_to_each_occurrences_field_and_preserves_picking() {
    use gf_core::{modifier_pivot::Pivot, world_mods::Scope};
    let args=[("scene","studio"),("preset","sphere"),("p.radius","0.24"),("res","96")]
        .into_iter().map(|(k,v)|(k.to_owned(),v.to_owned())).collect::<Vec<_>>();
    let mut st=State::with_args(&args,true);
    let mut copies=Modifier::of(Kind::Repeat);
    copies.fit=false;
    copies.a=Axes::new([1.,2.,1.],false);
    copies.b=Axes::new([1.,1.1,1.],false);
    let mut taper=Modifier::of(Kind::Taper);
    taper.a=Axes::new([0.,1.8,0.],false);
    taper.scope=Scope::SubObjects;
    taper.pivot=Pivot::World;
    taper.cascade=true;
    let source=Item::Object(0);
    st.stack.world_modifiers_mut(source).unwrap().list=vec![copies,taper];
    let plan=st.stack.world_plan();
    assert!(plan.has_distinct_fields(0));
    assert_eq!(plan.field_ids[0].len(),2);
    let centres=plan.field_ids[0].iter().map(|&id|plan.fields[id].mods.list[0].centre[1]).collect::<Vec<_>>();
    assert_eq!(centres,vec![0.55,-0.55]);
    assert_eq!(st.object_count(),1);

    let mut e=Engine::new(Gpu::headless("vulkan").unwrap(),st,true);
    e.bake(Detail::Full).unwrap();
    let read=|e:&Engine,x:f64,y:f64| {
        let v=e.scene_volume().unwrap(); let h=e.state.scene_hb();
        let index=|p:f64| (((p/h+1.)*0.5*f64::from(v.res)).floor() as u32).min(v.res-1);
        let raw=e.gpu.read(&v.tex,index(x),index(y),v.res/2,1,1,2).unwrap();
        half::f16::from_le_bytes([raw[0],raw[1]]).to_f32()
    };
    assert!(read(&e,0.24,-0.55)>0.);
    assert!(read(&e,0.24,0.55)<0.);
    let plan=e.state.stack.world_plan();
    let small=plan.occurrence_bounds(&e.state.stack.world_field_bounds,0,0).unwrap();
    let large=plan.occurrence_bounds(&e.state.stack.world_field_bounds,0,1).unwrap();
    assert!(large[3]>small[3]+0.12,"{small:?} {large:?}");
    let bounds=e.state.world_bounds(source).unwrap();
    assert!((bounds[3]-large[3]).abs()<1e-8 && bounds[4]>=0.55+large[4]-1e-8);
    e.resize(256,256); e.render(3);
    for (expected,y) in [-0.55,0.55].into_iter().enumerate() {
        let viewed=gf_core::math::apply(&e.state.view_r(),[0.,y,0.]);
        let uv=e.state.camera().project(viewed).unwrap();
        let hit=glassforms::ui::handles::pick_occurrence(&e,0.5+uv[0]*0.5,0.5-uv[1]*0.5).unwrap();
        assert_eq!((hit.source,hit.occurrence),(0,expected));
    }
    // A later World stage measures the differing incoming solids. Treating their
    // equal source sphere volumes as the inputs would leave this at zero.
    let mut next=Modifier::of(Kind::Twist);
    next.a=Axes::new([0.,0.1,0.],false);
    next.pivot=Pivot::CentreOfMass;
    e.state.stack.world_modifiers_mut(source).unwrap().list.push(next);
    e.bake(Detail::Full).unwrap();
    let centre=e.state.stack.world_pivots[&(source,2)];
    assert!(centre[1]>0.15,"{centre:?}");
    e.state.stack.world_modifiers_mut(source).unwrap().list.pop();
    // The local origin travels with either translated copy. Both now share the
    // same field and return to the ordinary one-source BVH.
    e.state.stack.world_modifiers_mut(source).unwrap().list[1].pivot=Pivot::Local;
    e.bake(Detail::Full).unwrap();
    assert!(!e.state.stack.world_plan().has_distinct_fields(0));
    assert!(read(&e,0.17,-0.55)<0. && read(&e,0.17,0.55)<0.);
    let cross_section = |e: &Engine| {
        let v = e.scene_volume().unwrap();
        e.gpu.read(&v.tex, 0, 0, v.res/2, v.res, v.res, 2).unwrap()
    };
    let before_origin_edit = cross_section(&e);
    assert!(e.state.set_world_origin(source, [0.31, -0.13, 0.16]));
    e.bake(Detail::Full).unwrap();
    assert!(before_origin_edit == cross_section(&e), "editing the pivot moved the tapered copies");
    e.state.stack.world_modifiers_mut(source).unwrap().list.clear();
    e.bake(Detail::Full).unwrap();
    assert!(read(&e,0.,0.)<0. && read(&e,0.,0.55)>0.);
}
