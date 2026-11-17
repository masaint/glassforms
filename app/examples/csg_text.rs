//! Prints a tree tracer's text (name=value ... as on the command line: scene, preset, trace, csg, csgGrid)
//! to a file, and how often the tracer reads the tree. For reading and measuring shader size.
use glassforms::render::{tracer_source, March, Variant};

fn main() {
    let args: Vec<(String, String)> = std::env::args().skip(1).filter_map(|a| a.split_once('=').map(|(k, v)| (k.to_string(), v.to_string()))).collect();
    let out = args.iter().find(|(k, _)| k == "out").map(|(_, v)| v.clone()).unwrap_or_else(|| "csg_tracer.wgsl".into());
    let st = gf_core::state::State::with_args(&args, true);
    let (parts, _) = st.csg_tree();
    let kinds = gf_core::csg::kinds(&parts);
    let decl = gf_core::csg::wgsl(&st.families, &kinds, gf_core::sdf::BINDING, false);
    let m = March { step_scale: 0.95, steps_out: 220, steps_in: 56, layers: 2 };
    let src = glassforms::sdf::weave(&tracer_source(&m, Variant::default(), false), &decl);
    let count = |s: &str| src.matches(s).count();
    eprintln!("parts {} kinds {} bytes {} lines {} | mapN( {} mapNW( {} csgDist( {} csgDistW( {} csgNormal( {} csgPartAt( {} nrm( {}",
        parts.len(), kinds.len(), src.len(), src.lines().count(), count("mapN("), count("mapNW("), count("csgDist("), count("csgDistW("), count("csgNormal("), count("csgPartAt("), count("nrm("));
    std::fs::write(&out, &src).expect("write");
    // how long the numbers and the cells' lists take to work out (every build, while things move too)
    let t = std::time::Instant::now();
    let mut words = 0;
    for _ in 0..20 { words = gf_core::csg::numbers(&st.families, &parts, &kinds, 768, false, &[], &st.csg.place).1.len(); }
    eprintln!("numbers: {:.3} ms, {} words of lists", t.elapsed().as_secs_f64() * 1e3 / 20.0, words);
}
