//! Every shader of the program must parse and pass naga's validator. Needs no GPU.

use glassforms::render::{tracer_source, LightFrom, March, Variant};

fn check(name: &str, src: &str) {
    let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name} does not parse:\n{}", e.emit_to_string(src)));
    let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
    v.validate(&module).unwrap_or_else(|e| panic!("{name} is not valid:\n{}", e.emit_to_string(src)));
}

/// Every way the tracer is built: lit by the lights' map or by an environment picture; with or without
/// the picture behind the shape; glass or clay.
fn variants() -> Vec<Variant> {
    let mut all = Vec::new();
    for light in [LightFrom::Lights, LightFrom::Picture] {
        for backdrop in [false, true] {
            for clay in [false, true] {
                all.push(Variant { light, backdrop, clay, mat: Default::default(), cast: false, ..Default::default() });
            }
        }
    }
    all
}

#[test]
fn tracer() {
    for (step_scale, layers) in [(0.95, 2), (0.90, 2), (1.00, 0)] {
        let m = March { step_scale, steps_out: 220, steps_in: 56, layers };
        for v in variants() {
            // as it is built for pictures, and as it is built to count its own lookups
            check(&format!("trace.wgsl {v:?}"), &tracer_source(&m, v, false));
            check(&format!("trace.wgsl {v:?} (counting)"), &tracer_source(&m, v, true));
        }
    }
}

/// Nothing is left to fill in, in any variant; and only the default is built without the extra block.
#[test]
fn tracer_variants_are_complete() {
    let m = March { step_scale: 0.95, steps_out: 220, steps_in: 56, layers: 2 };
    for v in variants() {
        let src = tracer_source(&m, v, false);
        // (the only "{{" left is the one in the comment that explains them)
        assert!(!src.split("{{").skip(1).any(|rest| rest.starts_with(|c: char| c.is_ascii_uppercase())), "{v:?}: a placeholder was not filled in");
        assert_eq!(src.contains("pictureTex"), v != Variant::default(), "{v:?}");
        assert_eq!(src.contains("col = clay("), v.clay, "{v:?}");
        assert_eq!(src.contains("var col = backdrop(rd);"), v.backdrop, "{v:?}");
        assert_eq!(src.contains("return picture(rd);"), v.light == LightFrom::Picture, "{v:?}");
    }
}

/// Glass lit by the family's lights is the tracer the program has always had, to the letter: the other
/// ways of drawing are built beside it and add nothing to it. (These are fingerprints of the tracer's
/// text as it was built before those ways existed. A change to the glass tracer itself changes them:
/// check the pictures against the web version, `python tools/parity.py --all`, before putting new ones here.)
#[test]
fn the_default_tracer_is_untouched() {
    let fnv = |text: &str| text.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3));
    for (step_scale, want) in [(0.95, 0xdb48ee470de56b55u64), (0.90, 0x37436ea44d3174be)] {
        let src = tracer_source(&March { step_scale, steps_out: 220, steps_in: 56, layers: 2 }, Variant::default(), false);
        assert_eq!(fnv(&src), want, "the default tracer's text has changed (step {step_scale})");
    }
}

/// Every material variant, lit by the lights' map and by an environment picture, with and without the
/// picture behind, for two ways of marching, built for pictures and for counting; and what is left of a
/// material under clay (a coat, a film).
#[test]
fn tracer_materials() {
    let all = gf_core::material::Variant::samples();
    assert!(all.len() > 40);
    let unfilled = |src: &str| src.split("{{").skip(1).any(|rest| rest.starts_with(|c: char| c.is_ascii_uppercase())) || src.contains("//#");
    for mat in &all {
        for (step_scale, layers) in [(0.95, 2), (1.0, 0)] {
            let m = March { step_scale, steps_out: 220, steps_in: 56, layers };
            for (light, backdrop) in [(LightFrom::Lights, false), (LightFrom::Picture, false), (LightFrom::Picture, true)] {
                let v = Variant { light, backdrop, clay: false, mat: *mat, cast: false, ..Default::default() };
                for profile in [false, true] {
                    let src = tracer_source(&m, v, profile);
                    assert!(!unfilled(&src), "{v:?}: something was left unfilled");
                    check(&format!("trace.wgsl {v:?}{}", if profile { " (counting)" } else { "" }), &src);
                }
                let clay = Variant { light, backdrop, clay: true, mat: mat.for_clay(), cast: false, ..Default::default() };
                let src = tracer_source(&m, clay, false);
                assert!(!unfilled(&src), "{clay:?}: something was left unfilled");
                check(&format!("trace.wgsl {clay:?}"), &src);
            }
        }
    }
}

/// Wear (dust, polishing lines, uneven colour) together with every sample material, those with things in
/// them among them: both bring helpers of their own, and two of the same name stop the tracer building
/// (the owner's crash of 2026-11-17: `fromObj` twice, with dust on a material with inner parts).
#[test]
fn tracer_wear_with_every_material() {
    let m = March { step_scale: 1.0, steps_out: 220, steps_in: 56, layers: 0 };
    for mat in gf_core::material::Variant::samples() {
        let mut worn = mat;
        worn.wear = true;
        for bodies in [0u8, 2] {
            worn.bodies = bodies;
            let v = Variant { light: LightFrom::Lights, backdrop: false, clay: false, mat: worn, cast: false, ..Default::default() };
            check(&format!("trace.wgsl worn {v:?}"), &tracer_source(&m, v, false));
        }
    }
}

/// The tracer for a scene of several bodies with materials of their own: for two, three and four bodies,
/// lit by the lights and by an environment picture with and without it behind, for two ways of marching,
/// built for pictures and for counting.
#[test]
fn tracer_bodies() {
    let unfilled = |src: &str| src.split("{{").skip(1).any(|rest| rest.starts_with(|c: char| c.is_ascii_uppercase())) || src.contains("//#");
    use gf_core::material::Variant as M;
    // (the parts the bodies use between them: none beyond clear glass, each alone, and all at once)
    let mut parts = vec![M::default(), M { film: true, ..M::default() }, M { rough: true, ..M::default() }, M { sss: true, ..M::default() }, M { metal: true, ..M::default() },
        M { hollow: true, film: true, ..M::default() }, M { metal: true, rough: true, film: true, hollow: true, sss: true, ..M::default() }];
    // colour by wavelength, the graded index, double refraction and what lies inside, each alone, beside
    // the others, and everything at once
    let d = M::default();
    parts.extend([M { spectrum: true, ..d }, M { grin: true, ..d }, M { biref: true, ..d }, M { spectrum: true, grin: true, biref: true, ..d }]);
    for i in gf_core::inner::Parts::samples() {
        parts.push(M { inner: i, ..d });
        parts.push(M { inner: i, biref: true, sss: true, ..d });
    }
    let all = gf_core::inner::Parts { grad: true, crack: true, frac: true, crack_depth: true, crack_rough: true, play: true, glitter: true, silk: true, piezo: true };
    parts.push(M { metal: true, rough: true, film: true, hollow: true, sss: true, spectrum: true, grin: true, biref: true, inner: all, ..d });
    for (bodies, with) in (2..=4u8).flat_map(|b| parts.iter().map(move |p| (b, *p))) {
        let mat = M { bodies, ..with };
        for (step_scale, layers) in [(0.95, 2), (1.0, 0)] {
            let m = March { step_scale, steps_out: 220, steps_in: 56, layers };
            for (light, backdrop) in [(LightFrom::Lights, false), (LightFrom::Picture, false), (LightFrom::Picture, true)] {
                let v = Variant { light, backdrop, clay: false, mat, cast: false, ..Default::default() };
                for profile in [false, true] {
                    let src = tracer_source(&m, v, profile);
                    assert!(!unfilled(&src), "{v:?}: something was left unfilled");
                    assert!(src.contains("shadeB(ro, rd, t, xi, W, fc)") && !src.contains("fn shadeM") && !src.contains("mu.") && !src.contains(" iu.") && !src.contains("(iu.") && src.contains(&format!("const COUNT: i32 = {bodies};")));
                    check(&format!("trace.wgsl {v:?}{}", if profile { " (counting)" } else { "" }), &src);
                }
            }
        }
    }
}

/// Bubbles are in the volume, not in the tracer: the plain tracer's text again, but for looking through holes.
#[test]
fn bubbles_leave_the_tracer_alone() {
    let m = March { step_scale: 0.95, steps_out: 220, steps_in: 56, layers: 2 };
    let plain = tracer_source(&m, Variant::default(), false);
    let mat = gf_core::material::Variant { bubbles: true, ..Default::default() };
    assert_eq!(tracer_source(&March { layers: 0, ..m }, Variant { mat, ..Variant::default() }, false), plain);
    assert!(!plain.contains("shadeM") && !plain.contains("binding(9)"));
}

#[test]
fn bubbles() {
    let src = include_str!("../src/shaders/bubbles.wgsl");
    check("bubbles.wgsl", src);
    check("bubbles.wgsl (drawn)", src.split("//#compute").next().unwrap());
}

#[test]
fn accumulate() {
    check("accum.wgsl", include_str!("../src/shaders/accum.wgsl"));
}

#[test]
fn post() {
    check("post.wgsl", include_str!("../src/shaders/post.wgsl"));
}

#[test]
fn bake_curve() {
    check("bake_curve.wgsl", include_str!("../src/shaders/bake_curve.wgsl"));
}

#[test]
fn environment_picture() {
    check("hdri.wgsl", include_str!("../src/shaders/hdri.wgsl"));
}

#[test]
fn focus() {
    check("dof.wgsl", include_str!("../src/shaders/dof.wgsl"));
}

/// A tracer built for a stage larger than ever (a layer moved or sized past the usual box): still valid
/// with the free camera's place in its numbers.
#[test]
fn grown_tracer() {
    let m = March { step_scale: 0.95, steps_out: 220, steps_in: 56, layers: 2 };
    for clay in [false, true] {
        let v = Variant { grown: true, clay, ..Default::default() };
        check(&format!("trace.wgsl grown {v:?}"), &tracer_source(&m, v, false));
    }
}

/// A camera standing off the line through the object's centre gets a tracer of its own (the default
/// tracer is then left as it was): valid for plain glass, clay and a material, and its rays start there.
#[test]
fn eye_tracer() {
    let m = March { step_scale: 0.95, steps_out: 220, steps_in: 56, layers: 2 };
    let mats = gf_core::material::Variant::samples();
    for (clay, mat) in [(false, Default::default()), (true, Default::default()), (false, mats[mats.len() / 2])] {
        let v = Variant { eye: true, clay, mat, ..Default::default() };
        let src = tracer_source(&m, v, false);
        assert!(src.contains("var ro = vec3f(u.eye.xy, u.dist);") && !src.contains("vec3f(0.0, 0.0, u.dist)"), "{v:?}");
        check(&format!("trace.wgsl {v:?}"), &src);
    }
}

/// DLSS's passes (dlss.rs), and the tracer it draws with: every way of building the tracer has the line
/// that scatters each pixel's rays, which DLSS's tracer replaces, and the result is valid.
#[test]
fn dlss() {
    check("dlss_prep.wgsl", include_str!("../src/shaders/dlss_prep.wgsl"));
    check("dlss_resolve.wgsl", include_str!("../src/shaders/dlss_resolve.wgsl"));
    let m = March { step_scale: 0.95, steps_out: 220, steps_in: 56, layers: 2 };
    for v in variants() {
        let src = glassforms::dlss::centre(&tracer_source(&m, v, false)).unwrap_or_else(|| panic!("{v:?}: no line to centre"));
        assert!(src.contains("let shift = vec2f(0.5);"), "{v:?}");
        check(&format!("trace.wgsl centred {v:?}"), &src);
    }
}
