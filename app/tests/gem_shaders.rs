//! The tracer built for a cut stone must parse and pass naga's validator. Needs no GPU.

use glassforms::render::{gem_fits, gem_weave, tracer_source, LightFrom, March, Variant};

fn check(name: &str, src: &str) {
    let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name} does not parse:
{}", e.emit_to_string(src)));
    let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
    v.validate(&module).unwrap_or_else(|e| panic!("{name} is not valid:
{}", e.emit_to_string(src)));
}

/// The ways of drawing plain glass that a stone's tracer is built for: lit by the lights' map or by an
/// environment picture, with or without the picture behind the shape.
fn variants() -> Vec<Variant> {
    let mut all = Vec::new();
    for light in [LightFrom::Lights, LightFrom::Picture] {
        for backdrop in [false, true] {
            all.push(Variant { light, backdrop, ..Default::default() });
        }
    }
    all
}

/// The tracer built for a cut stone (render.rs `gem_weave`): its march through the glass is replaced, in
/// every way of drawing it is built for, and the result is a valid shader, also when built to count.
#[test]
fn stone_tracer() {
    let m = March { step_scale: 0.8, steps_out: 160, steps_in: 96, layers: 0 };
    for v in variants().into_iter().filter(gem_fits) {
        for profile in [false, true] {
            let src = gem_weave(&tracer_source(&m, v, profile));
            assert!(src.contains("let s = gemExit(pos, dir, &m);") && src.contains("fn gemExit("), "{v:?}: the stone's way out was not woven in");
            check(&format!("trace.wgsl for a stone {v:?} (counting: {profile})"), &src);
        }
    }
    // (no lamps: they note where each normal was read, and the stone's normals are not read)
    assert!(!gem_fits(&Variant { near: true, ..Default::default() }));
}
