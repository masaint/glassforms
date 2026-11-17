//! The studio's shader text: every tracer built with a backdrop or a floor must parse and pass naga's
//! validator, and a tracer built without a studio must not have a letter of it. Needs no GPU.

use glassforms::render::{tracer_source, LightFrom, March, Variant};
use glassforms::stage::{background, tracer_decl, Seen, BINDING};

fn check(name: &str, src: &str) {
    let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name} does not parse:\n{}", e.emit_to_string(src)));
    let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
    v.validate(&module).unwrap_or_else(|e| panic!("{name} is not valid:\n{}", e.emit_to_string(src)));
}

/// Every way a studio can be seen by the camera.
fn studios() -> Vec<Seen> {
    let mut all = Vec::new();
    for backdrop in [false, true] {
        for (floor, mirror) in [(false, false), (true, false), (true, true)] {
            if backdrop || floor { all.push(Seen { backdrop, floor, mirror, cast: false }); }
        }
    }
    all
}

fn unfilled(src: &str) -> bool {
    src.split("{{").skip(1).any(|rest| rest.starts_with(|c: char| c.is_ascii_uppercase())) || src.contains("//#")
}

/// Every studio with every lighting, glass and clay, for two ways of marching, built for pictures and
/// for counting lookups.
#[test]
fn tracer_with_a_studio() {
    for (step_scale, layers) in [(0.95, 2), (1.0, 0)] {
        let m = March { step_scale, steps_out: 220, steps_in: 56, layers };
        for stage in studios() {
            for light in [LightFrom::Lights, LightFrom::Picture] {
                for backdrop in [false, true] {
                    // (an environment picture is only shown behind the shape when there is one)
                    if backdrop && light == LightFrom::Lights { continue; }
                    for clay in [false, true] {
                        let v = Variant { light, backdrop, clay, stage, ..Default::default() };
                        for profile in [false, true] {
                            let src = tracer_source(&m, v, profile);
                            assert!(!unfilled(&src), "{v:?}: something was left unfilled");
                            check(&format!("trace.wgsl {v:?}"), &src);
                            assert!(src.contains(&format!("@binding({BINDING}) var<uniform> su: SU;")), "{v:?}");
                            assert_eq!(src.contains("fn stageFloor("), stage.floor, "{v:?}");
                            assert_eq!(src.contains("fn stageMirror("), stage.mirror, "{v:?}");
                            assert_eq!(src.contains("var col = stageBackdrop(rd);"), stage.backdrop && !stage.floor, "{v:?}");
                            assert_eq!(src.contains("var col = stageFloor(ro, rd, &t, &g, &depth"), stage.floor, "{v:?}");
                        }
                    }
                }
            }
        }
    }
}

/// Every material with a floor that mirrors the shape (the mirror shades the shape with the material's
/// own text), lit both ways; and what is left of a material under clay.
#[test]
fn materials_with_a_studio() {
    let all = gf_core::material::Variant::samples();
    let m = March { step_scale: 0.95, steps_out: 220, steps_in: 56, layers: 2 };
    let stage = Seen { backdrop: true, floor: true, mirror: true, cast: false };
    for mat in &all {
        for light in [LightFrom::Lights, LightFrom::Picture] {
            let v = Variant { light, mat: *mat, stage, ..Default::default() };
            let src = tracer_source(&m, v, false);
            assert!(!unfilled(&src), "{v:?}: something was left unfilled");
            check(&format!("trace.wgsl {v:?}"), &src);
            let clay = Variant { light, clay: true, mat: mat.for_clay(), stage, ..Default::default() };
            let src = tracer_source(&m, clay, false);
            assert!(!unfilled(&src), "{clay:?}: something was left unfilled");
            check(&format!("trace.wgsl {clay:?}"), &src);
        }
    }
}

/// Without a studio the tracer is built from exactly the text it was built from before there was one:
/// no declaration, and what lies behind the shape is said in the words it always was.
#[test]
fn no_studio_no_text() {
    let none = Seen::default();
    assert_eq!(tracer_decl(none, false, "shade(ro, rd, t, xi, W)"), "");
    assert_eq!(tracer_decl(none, true, "shade(ro, rd, t, xi, W)"), "");
    assert_eq!(background(none, false), "vec3f(0.0)");
    assert_eq!(background(none, true), "backdrop(rd)");
    let m = March { step_scale: 0.95, steps_out: 220, steps_in: 56, layers: 2 };
    for light in [LightFrom::Lights, LightFrom::Picture] {
        for clay in [false, true] {
            let src = tracer_source(&m, Variant { light, clay, ..Default::default() }, false);
            assert!(!src.contains("stage") && !src.contains(" su: SU") && !src.contains("toRoom"), "{light:?} clay {clay}");
        }
    }
    // a floor with no backdrop shows the environment picture behind it when one is shown, else black
    let floor = Seen { floor: true, ..Seen::default() };
    assert!(tracer_decl(floor, true, "x").contains("let behind = backdrop(rd);"));
    assert!(tracer_decl(floor, false, "x").contains("let behind = vec3f(0.0);"));
    assert!(tracer_decl(Seen { backdrop: true, ..floor }, true, "x").contains("let behind = backdropAt(d);"));
    // shadows and caustics on the floor are asked for only when there is someone to answer
    assert!(tracer_decl(floor, false, "x").contains("sweepAt(d.y)) * vec3f(1.0);"));
    // (the light the bodies focus onto the floor is added to its colour, not a multiple of it: cast.wgsl, castFloorLight)
    assert!(tracer_decl(Seen { cast: true, ..floor }, false, "x").contains("sweepAt(d.y)) * castFloor(p) + castFloorLight(p);"));
}

#[test]
fn lowest_point() {
    check("stage_lowest.wgsl", include_str!("../src/shaders/stage_lowest.wgsl"));
}
