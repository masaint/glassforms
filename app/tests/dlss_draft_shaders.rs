//! The Ray Reconstruction draft's shaders (dlss_rr.rs) must parse and pass naga's validator: the plain tracer with every pixel's ray at one
//! offset and the guide pass appended, in the plain and the grown tracer, and the pass that turns the guides into RR's inputs. Needs no GPU.
#![cfg(feature = "dlss")]

use glassforms::render::{tracer_source, March, Variant};

fn check(name: &str, src: &str) {
    let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name} does not parse:\n{}", e.emit_to_string(src)));
    let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
    v.validate(&module).unwrap_or_else(|e| panic!("{name} is not valid:\n{}", e.emit_to_string(src)));
}

const MARCH: March = March { step_scale: 0.9, steps_out: 220, steps_in: 56, layers: 2 };

#[test]
fn the_draft_tracer_with_its_guide_pass() {
    let plain = tracer_source(&MARCH, Variant::default(), false);
    let centred = glassforms::dlss::centre(&plain).expect("the first ray's scatter line is in the tracer");
    let src = glassforms::dlss_rr::with_guides(&centred);
    assert!(src.contains("fn fs_guide("));
    check("trace.wgsl, centred, with guides", &src);
}

#[test]
fn the_variant_pass() {
    check("dlss_draft_variant.wgsl", include_str!("../src/shaders/dlss_draft_variant.wgsl"));
}
