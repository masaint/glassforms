//! A path's exact distance as the GPU reads it (`path_formula`): the baker's text and a tree's text with a path
//! among its kinds both compile.

fn compiled(id: &str, src: &str) {
    let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{id}: does not parse\n{}", e.emit_to_string(src)));
    let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty());
    if let Err(e) = v.validate(&module) {
        panic!("{id}: not valid\n{}", e.emit_to_string(src));
    }
}

#[test]
fn the_baker_reads_a_path_exactly() {
    let text = include_str!("../../app/src/shaders/bake_curve.wgsl").replace("//#path", &gf_core::path_formula::wgsl("pts[eb + j]"));
    assert!(text.contains("fn pathPrim("));
    let drawn = text.split("//#compute").next().unwrap();
    compiled("bake", drawn);
}

#[test]
fn a_tree_reads_a_path_exactly() {
    let fams = gf_core::presets::families();
    let f = fams.iter().position(|f| f.id == gf_core::presets::paths::FAMILY).unwrap();
    for (walked, plain) in [(false, true), (true, true), (false, false)] {
        let kinds = vec![(f, 0usize), (f, 3usize)];
        let text = gf_core::csg::wgsl_tree(&fams, &kinds, gf_core::sdf::BINDING, walked, plain);
        assert!(text.contains("fn pathDist("));
        compiled(&format!("tree walked={walked} plain={plain}"), &text);
    }
}
