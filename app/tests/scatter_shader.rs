//! The shader that works out the light inside a milky body must parse and pass naga's validator, and the
//! tracer built for a milky body must read the light volume (and the one for wandering light must not).
//! Needs no GPU.

use gf_core::settings::Settings;
use glassforms::render::{tracer_source, March, Variant};

fn check(name: &str, src: &str) {
    let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name} does not parse:\n{}", e.emit_to_string(src)));
    let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
    v.validate(&module).unwrap_or_else(|e| panic!("{name} is not valid:\n{}", e.emit_to_string(src)));
}

#[test]
fn the_light_volume_shader_is_valid() {
    check("scatter.wgsl", include_str!("../src/shaders/scatter.wgsl"));
}

fn tracer(settings: &[(&str, f64)]) -> String {
    let mut p = Settings::default();
    for (k, v) in settings { assert!(p.set(k, *v), "no setting {k}"); }
    let m = March { step_scale: 0.95, steps_out: 220, steps_in: 56, layers: 2 };
    let src = tracer_source(&m, Variant { mat: gf_core::material::Variant::of(&p), ..Variant::default() }, false);
    check(&format!("the tracer for {settings:?}"), &src);
    src
}

#[test]
fn a_milky_body_reads_the_light_volume_and_nothing_random() {
    for extra in [&[][..], &[("rough", 0.3)], &[("coat", 1.0)], &[("film", 300.0)], &[("disp", 0.0)], &[("glow", 1.0)], &[("spectrum", 1.0)], &[("grin", 0.2)]] {
        let mut s = vec![("sss", 1.0)];
        s.extend_from_slice(extra);
        let src = tracer(&s);
        assert!(src.contains("@binding(24) var glowTex"), "{s:?}: the light volume is not declared");
        assert!(src.contains("glowAlong("), "{s:?}: the light volume is not read");
        assert!(!src.contains("fn wander("), "{s:?}: the random walk is in a tracer that should not have it");
        // (a rough surface draws its facets at random; the milky body itself draws nothing)
        if !s.iter().any(|(k, _)| *k == "rough") { assert!(!src.contains("fn pcg("), "{s:?}: random numbers in a tracer that needs none"); }
    }
}

#[test]
fn wandering_light_reads_no_volume() {
    let src = tracer(&[("sss", 1.0), ("sssWalk", 1.0)]);
    assert!(src.contains("fn wander(") && !src.contains("glowTex"));
}

#[test]
fn glass_and_metal_know_nothing_of_the_light_volume() {
    for s in [&[][..], &[("rough", 0.3)], &[("metal", 4.0)], &[("coat", 1.0)]] {
        assert!(!tracer(s).contains("glowTex"), "{s:?}");
    }
}
