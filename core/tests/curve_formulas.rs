use gf_core::presets;
use gf_core::state::{Detail, ShapeJob, State};

#[test]
fn every_generated_tube_builds_from_its_geometry() {
    let mut state = State::new();
    for fi in 0..state.families.len() {
        let shapes = state.families[fi].presets.len();
        for pi in 0..shapes {
            if !state.families[fi].presets[pi].exact_tubes {
                continue;
            }
            state.set_scene(fi);
            state.choose(pi);
            let mut job = state.shape_job(Detail::Full, Some(256)).unwrap();
            let who = format!("{}/{}", state.family().id, state.preset().id);
            assert_eq!(
                gf_core::sdf::support(&state.families[fi], &state.families[fi].presets[pi]),
                gf_core::sdf::Support::Formula { exact: true },
                "{who}: exact sphere-envelope sweep must use direct exact SDF evaluation"
            );
            let t = std::time::Instant::now();
            job.prepare_formula(&state.families)
                .unwrap_or_else(|e| panic!("{who}: {e}"));
            let ShapeJob::Field { table, .. } = job.shape else {
                panic!("{who}: no equation")
            };
            assert_eq!(table[0] as usize * 4, table.len(), "{who}");
            assert!(table.len() <= 4 * gf_core::field::TABLE_MAX, "{who}");
            assert!(table.iter().all(|v| v.is_finite()), "{who}");
            assert!(
                table[11] < table[0],
                "{who}: there must be at least one active piece"
            );
            eprintln!(
                "{who:28} {:7} entries {:8.3} ms",
                table.len() / 4,
                t.elapsed().as_secs_f64() * 1000.0
            );
        }
    }
}

#[test]
fn thickness_animation_changes_the_same_curve_in_both_modes() {
    let mut state = State::new();
    let family = state
        .families
        .iter()
        .position(|f| f.id == presets::links::FAMILY)
        .unwrap();
    state.set_scene(family);
    let mut radii = Vec::new();
    for thick in [0.5, 1.0, 2.0] {
        state.shape_mut().thick = thick;
        let mut job = state.shape_job(Detail::Full, Some(256)).unwrap();
        job.prepare_formula(&state.families).unwrap();
        let ShapeJob::Field { table, .. } = job.shape else {
            panic!("no equation")
        };
        let first_piece = table[11] as usize;
        radii.push(table[(first_piece + 3) * 4 + 3]);
    }
    assert_eq!(radii[1], 2.0 * radii[0]);
    assert_eq!(radii[2], 2.0 * radii[1]);
}

#[test]
fn generated_equations_remain_analytic_inside_a_tree() {
    let families = presets::families();
    let mut kinds = Vec::new();
    for (family_id, id) in [
        ("links", "hopflink"),
        ("crystals", "geode"),
        ("text", "word"),
        ("snowflakes", "dendrite"),
    ] {
        let family_index = families
            .iter()
            .position(|family| family.id == family_id)
            .unwrap();
        let preset_index = families[family_index]
            .presets
            .iter()
            .position(|p| p.id == id)
            .unwrap();
        let part = gf_core::csg::Part {
            family: family_id.into(),
            preset: id.into(),
            ..Default::default()
        };
        assert!(part.resolve(&families).unwrap().grid.is_none());
        kinds.push((family_index, preset_index));
    }
    for walked in [false, true] {
        let source =
            gf_core::csg::wgsl_tree(&families, &kinds, gf_core::sdf::BINDING, walked, false);
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    }
}

#[test]
fn every_cell_of_a_tube_table_holds_the_distance_at_its_middle() {
    let mut state = State::new();
    let fi = state.families.iter().position(|f| f.id == "knots").unwrap();
    state.set_scene(fi);
    let pi = state.family().presets.iter().position(|p| p.id == "trefoil").unwrap();
    state.choose(pi);
    let mut job = state.shape_job(Detail::Full, Some(256)).unwrap();
    job.prepare_formula(&state.families).unwrap();
    let ShapeJob::Field { table, .. } = job.shape else {
        panic!("expected formula field for knots/trefoil");
    };
    assert!(table[3] > 0.0, "knots/trefoil default melt must be > 0");
    let n = table[8] as usize;
    let (reach, clear) = (table[9], table[10]);
    assert!(clear > 2.2 * table[3] && reach >= clear / 0.65 - 1e-6, "reach {reach} clear {clear}");
    let (mut inside, mut far) = (0usize, 0usize);
    for c in 0..(n * n * n) {
        let (count, middle) = (table[(3 + c) * 4 + 1], table[(3 + c) * 4 + 2]);
        assert!(middle.is_finite() && middle < 4.0, "cell {c}: middle {middle}");
        // (a cell with nothing listed has nothing within reach of its middle)
        if count == 0.0 { assert!(middle > reach, "cell {c} lists nothing yet its middle is {middle} from the shape"); }
        if middle < 0.0 { inside += 1; }
        if middle > 4.0 * reach { far += 1; }
    }
    assert!(inside > 0 && far > 0, "expected middles inside the knot and far from it (inside={inside}, far={far})");
}
