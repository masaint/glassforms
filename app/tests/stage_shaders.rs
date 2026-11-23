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
            if backdrop || floor { all.push(Seen { backdrop, floor, mirror, ..Default::default() }); }
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
    let stage = Seen { backdrop: true, floor: true, mirror: true, ..Default::default() };
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

#[test]
fn a_scene_reflection_reads_the_lit_floor_only_when_the_primary_surface_is_metal() {
    let stage = Seen { floor: true, cast: true, ..Default::default() };
    let m = March { step_scale: 0.95, steps_out: 220, steps_in: 56, layers: 2 };
    for near in [false, true] {
        for light in [LightFrom::Lights, LightFrom::Picture] {
            for rough in [false, true] {
                let mat = gf_core::material::Variant {
                    bodies: 2, metal: true, scene_reflection: true, rough,
                    ..Default::default()
                };
                let v = Variant { stage, cast: true, mat, near, light, ..Default::default() };
                let s = tracer_source(&m, v, false);
                assert!(s.contains("reflectingMetalFloor = metalTop && spread == 0.0"));
                assert_eq!(s.matches("return col + thr * roomEscapeB(ro, rd, sg2);").count(), 2);
                assert_eq!(s.contains("let localLamps = lamps.floor.rgb"), near);
                check("escaped reflected ray from a metal beside clear crown", &s);
                let ambient = Variant { stage: Seen { cast: false, ..stage }, cast: false, ..v };
                assert!(!tracer_source(&m, ambient, false).contains("reflectingMetalFloor"));
            }
        }
    }
}

#[test]
fn a_sampled_metal_reflects_the_cast_floor_beneath_its_coat() {
    let m = March { step_scale: 0.95, steps_out: 220, steps_in: 56, layers: 2 };
    let floor = Seen { floor: true, cast: true, ..Default::default() };
    for near in [false, true] {
        for sampled in [false, true] {
            for coat in [false, true] {
                let mat = gf_core::material::Variant { metal: true, rough: true, rough_ref: sampled, coat, ..Default::default() };
                let v = Variant { stage: floor, cast: true, mat, near, ..Default::default() };
                let src = tracer_source(&m, v, false);
                assert_eq!(src.contains("let reflected = max(lit + metalFloorDelta(p, dir, true)"), sampled);
                assert_eq!(src.contains("fn metalCoatFloor("), coat);
                check("rough conductor floor and coat transport", &src);
            }
        }
    }
}

#[test]
fn polished_metals_read_the_lit_floor_only_on_an_escaping_reflection() {
    let m = March { step_scale: 0.95, steps_out: 220, steps_in: 56, layers: 2 };
    let floor = Seen { floor: true, cast: true, ..Seen::default() };
    for near in [false, true] {
        for wear in [false, true] {
            for light in [LightFrom::Lights, LightFrom::Picture] {
                let v = Variant {
                    stage: floor, cast: true, near, light,
                    mat: gf_core::material::Variant { metal: true, wear, ..Default::default() },
                    ..Default::default()
                };
                let src = tracer_source(&m, v, false);
                assert!(src.contains("let delta = base * (castFloor(point)"));
                assert_eq!(src.contains("let localLamps = lamps.floor.rgb"), near);
                assert_eq!(src.contains("tint * metalFloorDelta(p, reflect(rd, wr.n), false)"), wear);
                check("polished conductor above a lit floor", &src);
            }
        }
    }
    let clear = tracer_source(&m, Variant { stage: floor, cast: true, ..Default::default() }, false);
    assert!(!clear.contains("metalFloorDelta("));
}

#[test]
fn tracer_with_physical_sky() {
    let m = March { step_scale: 0.95, steps_out: 220, steps_in: 56, layers: 2 };
    for stage in [
        Seen { sky: true, ..Default::default() },
        Seen { sky: true, floor: true, mirror: false, ..Default::default() },
        Seen { sky: true, floor: true, mirror: true, cast: true, ..Default::default() },
    ] {
        for clay in [false, true] {
            let v = Variant { light: LightFrom::Lights, clay, stage, cast: stage.cast, ..Default::default() };
            let src = tracer_source(&m, v, false);
            assert!(!unfilled(&src), "{v:?}: unfilled placeholders");
            assert!(src.contains("fn skyBackdrop("), "{v:?}");
            assert_eq!(src.contains("let hor = skyDome(hd);"), stage.floor, "{v:?}");
            assert!(!stage.floor || !src.contains("return mix(fl, behind"), "{v:?}: sky floor fell back to su.sky3");
            check(&format!("sky trace.wgsl {v:?}"), &src);
        }
    }
}

#[test]
fn tracer_with_floor_gi() {
    let m = March { step_scale: 0.95, steps_out: 220, steps_in: 56, layers: 2 };
    for sky in [false, true] {
        for cast in [false, true] {
            for mirror in [false, true] {
                let stage_single = Seen { backdrop: !sky, floor: true, mirror, cast, sky, gi: true, multi: false, ..Default::default() };
                for clay in [false, true] {
                    let mat = if clay {
                        gf_core::material::Variant::default()
                    } else {
                        gf_core::material::Variant { sss: true, ceramic: true, ..Default::default() }
                    };
                    let v = Variant { light: LightFrom::Lights, clay, mat, stage: stage_single, cast, ..Default::default() };
                    let src = tracer_source(&m, v, false);
                    assert!(!unfilled(&src), "{v:?}: unfilled placeholders");
                    assert!(src.contains("fn sgiFloorBounce("), "{v:?}");
                    assert_eq!(src.contains("castFloorDirect(p)"), cast, "{v:?}");
                    check(&format!("single floor GI {v:?}"), &src);
                }
                let stage_multi = Seen { multi: true, ..stage_single };
                let mat_multi = gf_core::material::Variant {
                    bodies: 2,
                    sss: true,
                    ceramic: true,
                    paint_bodies: true,
                    paint_exchange: true,
                    ..Default::default()
                };
                for world_faces in [false, true] {
                    let v = Variant { light: LightFrom::Lights, mat: mat_multi, stage: stage_multi, cast, world_faces, ..Default::default() };
                    let src = tracer_source(&m, v, false);
                    assert!(!unfilled(&src), "{v:?}: unfilled placeholders");
                    assert!(src.contains("fn sgiFloorBounce("), "{v:?}");
                    assert!(src.contains("fn paintExchangeB("), "{v:?}");
                    check(&format!("multi floor GI {v:?}"), &src);
                }
            }
        }
    }
}

#[test]
fn reflected_stage_floor_shares_stage_coverage_and_gloss() {
    let m = March { step_scale: 0.95, steps_out: 220, steps_in: 56, layers: 2 };
    for sky in [false, true] {
        let stage = Seen { backdrop: !sky, floor: true, cast: true, sky, ..Default::default() };
        let decl = tracer_decl(stage, false, "x");
        assert!(decl.contains("fn floorCoverage(") && decl.contains("fn floorGloss("));
        assert!(!decl.contains("su.k.x"), "stage floor should not depend on legacy su.k.x fade");
        for bodies in [0u8, 2] {
            let mat = gf_core::material::Variant {
                bodies,
                metal: true,
                scene_reflection: bodies > 0,
                ..Default::default()
            };
            for near in [false, true] {
                let v = Variant { stage, cast: true, near, mat, ..Default::default() };
                let src = tracer_source(&m, v, false);
                assert!(src.contains("let coverage = floorCoverage(length(point.xz), ray.y);"));
                assert!(src.contains("let gloss = floorGloss(ray.y);"));
                assert!(!src.contains("su.k.x"), "reflected floor should share stage floorCoverage instead of su.k.x");
                check(&format!("reflected stage floor sky={sky} bodies={bodies} near={near}"), &src);
            }
        }
    }
}

#[test]
fn lamps_pool_shares_one_sided_panel_and_tube_area_with_diffuse() {
    let m = March { step_scale: 0.95, steps_out: 220, steps_in: 56, layers: 2 };
    let v = Variant {
        stage: Seen { backdrop: true, floor: true, cast: true, ..Default::default() },
        cast: true,
        near: true,
        ..Default::default()
    };
    let src = tracer_source(&m, v, false);
    assert!(src.contains("fn lampArea(L: LampU, l: vec3f) -> f32"));
    let pool = src.split("fn lampsPool(").nth(1).expect("lampsPool").split("\nfn ").next().unwrap();
    let diff = src.split("fn lampsDiffuse(").nth(1).expect("lampsDiffuse").split("\nfn ").next().unwrap();
    assert!(pool.contains("let area = lampArea(L, l);") && diff.contains("let area = lampArea(L, l);"));
    check("tracer with lampsPool and lampsDiffuse", &src);
}

#[test]
fn tracer_with_volumetric_sky_space_and_atmosphere() {
    use glassforms::render::tracer_source_lens;
    check("clouds.wgsl", include_str!("../src/shaders/clouds.wgsl"));
    let m = March { step_scale: 0.95, steps_out: 220, steps_in: 56, layers: 2 };
    for sky in [false, true] {
        for floor in [false, true] {
            for mirror in [false, true] {
                if !floor && mirror { continue; }
                for cast in [false, true] {
                    for gi in [false, true] {
                        if !floor && gi { continue; }
                        for atmo in [false, true] {
                            let stage = Seen {
                                backdrop: !sky && floor,
                                floor,
                                mirror,
                                cast: floor && cast,
                                sky,
                                gi,
                                multi: false,
                                sky_space: true,
                                atmo,
                            };
                            let mat = gf_core::material::Variant { metal: cast, ..Default::default() };
                            let v = Variant { light: LightFrom::Lights, mat, stage, cast, ..Default::default() };
                            let plain = tracer_source(&m, v, false);
                            let lens = tracer_source_lens(&m, v, false);
                            assert!(!unfilled(&plain) && !unfilled(&lens), "{v:?}: unfilled placeholders");
                            assert!(plain.contains("var skyDomeTex: texture_cube<f32>;"));
                            assert!(plain.contains("var cloudShadowTex: texture_2d<f32>;"));
                            assert_eq!(plain.contains("stageAtmosphere("), atmo, "{v:?}");
                            check(&format!("sky-space plain {v:?}"), &plain);
                            check(&format!("sky-space lens {v:?}"), &lens);
                        }
                    }
                }
            }
        }
    }
}
