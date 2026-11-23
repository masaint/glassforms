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

/// Glass lit by the family's lights keeps an exact source fingerprint. Explicit-medium metadata,
/// grazing-angle `faceIncident` normal projection, analytic cubic B-spline volume normals (`nrmVolume`),
/// and `marchBeyond` grazing silhouette rescue change that text; update only with fixed-frame image evidence.
/// (pbr/hash.wgsl, step 6 of the unification: unify_repeat.py identical on all 14 gate scenes and bubbles, cracks, wear.)
#[test]
fn the_default_tracer_fingerprint() {
    let fnv = |text: &str| text.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3));
    for (step_scale, want) in [(0.95, 1222434318253233382u64), (0.90, 5209576339810201189u64)] {
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
    parts.push(M { sss:true, walk:true, metal:true, rough:true, ..M::default() });
    // colour by wavelength, the graded index, double refraction and what lies inside, each alone, beside
    // the others, and everything at once
    let d = M::default();
    parts.extend([M { spectrum: true, ..d }, M { grin: true, ..d }, M { biref: true, ..d }, M { spectrum: true, grin: true, biref: true, ..d }]);
    for i in gf_core::inner::Parts::samples() {
        parts.push(M { inner: i, ..d });
        parts.push(M { inner: i, biref: true, sss: true, ..d });
    }
    parts.push(M { grating: true, metal: true, rough: true, ..d });
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

/// Different medium priorities specialize ownership only. The equal-priority source is the legacy body
/// tracer, and every relevant transport family still passes Naga with the focused replacement path.
#[test]
fn medium_priority_bodies() {
    let m = March { step_scale: 0.95, steps_out: 220, steps_in: 56, layers: 2 };
    let base = gf_core::material::Variant { bodies: 2, ..Default::default() };
    let legacy = tracer_source(&m, Variant { mat: base, ..Default::default() }, false);
    assert!(legacy.contains("for (var k = m + 1; k < COUNT; k++)"));
    assert!(include_str!("../src/shaders/trace_bodies.wgsl").contains("let interfaceOwner=max(m,b);"));
    assert!(!legacy.contains("fn bodyPrecedesB"));

    let variants = [
        base,
        gf_core::material::Variant { bodies: 2, rough: true, rough_ref: true, ..Default::default() },
        gf_core::material::Variant { bodies: 2, sss: true, walk: true, ..Default::default() },
        gf_core::material::Variant { bodies: 2, grin: true, ..Default::default() },
        gf_core::material::Variant { bodies: 2, film: true, coat: true, ..Default::default() },
    ];
    for mat in variants {
        let priority = gf_core::material::Variant { medium_priority: true, ..mat };
        let src = tracer_source(&m, Variant { mat: priority, ..Default::default() }, false);
        assert!(src.contains("fn bodyPrecedesB") && src.contains("fn containingB"));
        assert!(src.contains("for (var k = 0; k < COUNT; k++)"));
        assert!(!src.contains("let interfaceOwner=max(m,b);"));
        assert!(!src.contains("max(m,b)") && !src.contains("max(incident, destination)"));
        assert!(!src.contains("max(m, nx)") && src.contains("interfaceOwnerB(m, nx)"));
        check(&format!("medium-priority trace.wgsl {priority:?}"), &src);
    }
}

/// Forward caustics use the same compile-time split, with legacy equal-priority text left untouched.
#[test]
fn medium_priority_forward_caustics() {
    let legacy = glassforms::cast_light::light_source(false);
    let priority = glassforms::cast_light::light_source(true);
    assert!(legacy.contains("i < vec4i(m)") && legacy.contains("k < m && dd[k] < 0.0"));
    assert!(!legacy.contains("fn castPrecedes"));
    assert!(priority.contains("fn castPrecedes") && priority.contains("k != m && dd[k] < 0.0"));
    assert!(priority.contains("visible[k] = k == m || castPrecedes(k, m)"));
    assert!(priority.contains("*which = castWhich(a, f);"));
    check("cast light legacy", &legacy);
    check("cast light medium priority", &priority);
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
    let src = concat!(include_str!("../src/shaders/pbr/hash.wgsl"), include_str!("../src/shaders/bubbles.wgsl"));
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
    let src = include_str!("../src/shaders/bake_curve.wgsl").replace("//#path", &gf_core::path_formula::wgsl("pts[eb + j]"));
    check("bake_curve.wgsl", &src);
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

/// One wavelength inside the glass (hero.rs): the plain tracer, with or without a cut stone's planes, is
/// woven for one and for two wavelengths inside, and the result is valid.
#[test]
fn hero_inside() {
    let m = March { step_scale: 0.95, steps_out: 220, steps_in: 56, layers: 2 };
    for v in variants() {
        for src in [tracer_source(&m, v, false), glassforms::render::gem_weave(&tracer_source(&m, v, false))] {
            for k in [1u8, 2] {
                for (clamp, seq) in [(0u8, 0u8), (1, 1), (2, 2), (2, 1), (3, 3), (4, 3)] {
                    let h = glassforms::hero::Hero { mode: 1, k, clamp, seq, every: 6, taa: 0, split: 0 };
                    let woven = glassforms::hero::weave(&src, &h).unwrap_or_else(|| panic!("{v:?}: not woven"));
                    assert!(woven.contains(&format!("const HERO_K: i32 = {k};")), "{v:?}");
                    check(&format!("trace.wgsl hero {h:?} {v:?}"), &woven);
                }
            }
        }
    }
}

/// The antialiasing with colour kept apart from brightness (hero.rs chroma_taa) is woven and valid.
#[test]
fn hero_chroma_taa() {
    let src = glassforms::hero::chroma_taa(include_str!("../src/shaders/accum.wgsl")).expect("woven");
    check("accum.wgsl colour apart", &src);
}

/// One wavelength inside while moving, written apart (hero.rs `moving`, `split`; an experiment, off): the
/// plain tracer, with or without a stone's planes, is woven, valid, and writes three targets; and the pass
/// that puts the frame together is valid.
#[test]
fn hero_split() {
    check("hero_recon.wgsl", include_str!("../src/shaders/hero_recon.wgsl"));
    let m = March { step_scale: 0.95, steps_out: 220, steps_in: 56, layers: 2 };
    for v in variants() {
        for src in [tracer_source(&m, v, false), glassforms::render::gem_weave(&tracer_source(&m, v, false))] {
            let h = glassforms::hero::Hero::moving();
            let woven = glassforms::hero::weave(&src, &h).and_then(|t| glassforms::hero::split(&t)).unwrap_or_else(|| panic!("{v:?}: not split"));
            assert!(woven.contains(glassforms::hero::SPLIT_MARK), "{v:?}");
            check(&format!("trace.wgsl hero split {v:?}"), &woven);
        }
    }
}

/// Keeping the frame rate: the antialiasing for a moving frame traced smaller than the picture is made
/// from accum.wgsl, and the plain one is left exactly as it is.
#[test]
fn accum_for_a_smaller_frame() {
    let src = glassforms::render::ACCUM_SOURCE;
    let dyn_src = glassforms::render::accum_dyn(src).expect("accum.wgsl still has the lines the smaller frame's antialiasing changes");
    check("accum.wgsl (traced smaller)", &dyn_src);
    assert_ne!(dyn_src, src);
    assert!(dyn_src.contains("a.pad1") && !src.contains("a.pad1 + 0.5"));
    // A change to the tracer template must still apply the common ray offset on the
    // rough multi-body path. Otherwise reconstruction would register scattered rays.
    let march = March { step_scale: 1.0, steps_out: 120, steps_in: 50, layers: 2 };
    let mat = gf_core::material::Variant { bodies: 2, rough: true, ..Default::default() };
    let rough = tracer_source(&march, Variant { mat, ..Default::default() }, false);
    assert!(rough.contains("vec2f(0.5), u.pad1 != 0)"));
    check("registered rough assembly", &rough);
    assert!(!tracer_source(&march, Variant::default(), false).contains("u.pad1 != 0"));
    // the region the shader works out is the one the tracer is given
    for (tw, th, s) in [(3840, 2160, 0.71), (1920, 1080, 0.5), (1280, 720, 0.95), (7, 5, 0.5)] {
        let (rw, rh) = glassforms::render::dyn_region(tw, th, s);
        let f = s as f32;
        assert_eq!((rw, rh), (((tw as f32 * f + 0.5).floor() as u32).max(2), ((th as f32 * f + 0.5).floor() as u32).max(2)));
        assert!(rw <= tw && rh <= th);
    }
}

#[test]
fn imported_surfaces_with_rig_and_panorama() {
    let march = March { step_scale: 1.0, steps_out: 120, steps_in: 50, layers: 1 };
    let imported = gf_core::material::Variant::default();
    let assembly = gf_core::material::Variant { bodies: 2, metal: true, ..Default::default() };
    let mixed = gf_core::material::Variant { bodies: 2, metal: true, rough: true, rough_ref: true, ..Default::default() };
    let walk = gf_core::material::Variant { bodies: 2, sss: true, walk: true, ..Default::default() };
    let frost_walk = gf_core::material::Variant { bodies: 2, sss: true, walk: true, rough: true, rough_ref: true, ..Default::default() };
    for light in [LightFrom::Lights, LightFrom::Picture] {
        for near in [false, true] {
            for mat in [imported, assembly, mixed, walk, frost_walk] {
                let variant = Variant { light, near, textured: true, mat, ..Default::default() };
                let src = tracer_source(&march, variant, false);
                let name = format!("imported mesh surface {light:?}, near={near}, bodies={}, rough_ref={}", mat.bodies, mat.rough_ref);
                assert_eq!(src.contains("nine(0,toEnv(n)) + lampsDiffuse(p, n)"), near, "{name}");
                assert_eq!(src.contains("let reflected=fmLampReflection(p,n,rd,r,rough);"), near, "{name}");
                assert!(src.contains("if (dot(-rd,n)<=0.0) { return vec4f(reflected,1.0); }"), "{name}: an accepted opaque hit behind its perturbed normal must keep the legacy grazing gather, not become black");
                assert!(src.contains("return vec4f(pbr_imported_opaque_response(hit,imported,-rd,diffuse,reflected),1.0);"), "{name}: the common closure must partition the finite-lamp and environment gathers exactly once");
                assert_eq!(src.matches("fn pbr_imported_opaque_evaluate(").count(), 1, "{name}");
                assert_eq!(src.matches("fn pbr_ggx_evaluate(").count(), 1, "{name}: imported and measured closures share GGX arithmetic");
                assert_eq!(src.matches("fn roughLampRadiance(").count(), usize::from(near), "{name}: the shared emitter must be included exactly once");
                if near {
                    assert!(src.contains("if (rough <= 0.0) { return env(r); }"), "{name}: polished imported surfaces must keep the environment mirror");
                    assert!(src.contains("return roughLampRadiance(p,n,rd,mid,2.0*alpha*alpha,0,alpha,false,1.0,far);"), "{name}: imported surfaces are uncoated with unit geometric visibility");
                    if src.contains("fn envL0(") {
                        assert!(src.contains("envL0(wo,0.0)"), "{name}: panorama emitters should subtract the existing environment");
                    } else {
                        assert!(!src.contains("envL0(wo,0.0)"), "{name}: rig-only emitters must not reference envL0");
                        assert!(src.contains("envMap(toEnv(wo))"), "{name}: rig-only emitters should use the light map directly");
                    }
                } else {
                    assert!(src.contains("let reflected=mix(env(r),nine(9,toEnv(r)),rough*rough); // imported specular"), "{name}: the no-lamp control must remain unchanged");
                }
                check(&name, &src);
            }
        }
    }
}

#[test]
fn painted_body_reached_from_air_or_glass() {
    use gf_core::material::Variant as Material;
    let march = March { step_scale: 1.0, steps_out: 120, steps_in: 50, layers: 2 };
    for light in [LightFrom::Lights, LightFrom::Picture] {
        for wear in [false,true] {
            for near in [false,true] {
                let mat = Material { bodies: 3, paint_bodies: true, wear, metal:true, rough:true, ..Default::default() };
                let v = Variant { light, mat, near, ..Default::default() };
                let shader = tracer_source(&march,v,false);
                assert!(shader.contains("paintB_seen(nx, q, nq, dir, iorB(m, lam))"));
                assert_eq!(shader.contains("fn paintLampsB("),near);
                if wear {
                    assert!(shader.contains("dot(mb.b[cb].gratingFrame[0],v)"));
                }
                if wear && matches!(light, LightFrom::Picture) {
                    assert_eq!(shader.contains("return lampsSoft(c, rd, sig);"),near);
                }
                check("paint lit at the actual body interface", &shader);
            }
        }
    }
}

#[test]
fn every_paint_site_uses_the_shared_diffuse_response() {
    use gf_core::{material, settings::Settings};
    use gf_core::material::Variant as Material;
    let march = March { step_scale: 1.0, steps_out: 120, steps_in: 50, layers: 2 };

    // Real body settings still derive the coated and bare flags as before.
    let mut finish = Settings::default();
    let index = material::named_index("lacquer").unwrap();
    material::NAMED[index].apply(&mut finish);
    finish.coat = 0.0;
    finish.age = 0.0;
    let glass = Settings::default();
    let derived = Material::of_bodies(&[&finish, &glass]);
    assert!(derived.paint_bodies && !derived.paint_coat_bodies && !derived.coat);
    finish.coat = 1.0;
    let coated = Material::of_bodies(&[&finish, &glass]);
    assert!(coated.paint_coat_bodies && !coated.coat);

    // Bare, coated, filmed and worn paint, and its exchange hits, all go through pbr/layer.wgsl and pbr/diffuse.wgsl (one
    // deterministic diffuse law; the PAINT_OPAQUE_ADAPTER fork is gone).
    let bare = Material { bodies: 3, paint_bodies: true, paint_exchange: true, ..Default::default() };
    let cases = [
        ("bare", bare),
        ("sampled walker beside bare paint", Material { sss: true, walk: true, rough: true, rough_ref: true, ..bare }),
        ("coated", Material { paint_coat_bodies: true, ..bare }),
        ("painted film", Material { film: true, paint_film_bodies: true, ..bare }),
        ("worn", Material { wear: true, ..bare }),
    ];
    for (name, mat) in cases {
        let source = tracer_source(&march, Variant { mat, ..Default::default() }, false);
        assert!(!source.contains("paintOpaqueDiffuseB") && !source.contains("pbr_opaque_paint_response"), "{name}");
        assert_eq!(source.matches("fn pbr_diffuse_albedo(").count(), 1, "{name}");
        assert!(source.contains("let lobes = pbr_opaque_weights(c, incident, bodyIor, coatIor, coated, body, 1.0-0.5*through);"), "{name}");
        let exchange = source.split_once("fn paintExchangeB(").unwrap().1;
        assert!(exchange.contains("throughput *= lobes.diffuse;"), "{name}");
        check(name, &source);
    }

    // The standalone ceramic tracer has no assembly paint.
    let ceramic = Material { ceramic: true, sss: true, ..Default::default() };
    let source = tracer_source(&march, Variant { mat: ceramic, ..Default::default() }, false);
    assert!(!source.contains("paintOpaqueDiffuseB") && !source.contains("pbr_opaque_paint_response"));
    check("standalone ceramic", &source);
}
#[test]
fn opaque_paint_assemblies_stop_at_the_first_surface() {
    use gf_core::material::{named_index, Variant as Material, NAMED};
    use gf_core::settings::Settings;
    let mut paint = Settings::default();
    NAMED[named_index("lacquer").unwrap()].apply(&mut paint);
    let mut metal = Settings::default();
    NAMED[named_index("iron").unwrap()].apply(&mut metal);
    let glass = Settings::default();
    let mut wax = paint.clone();
    wax.sss_walk = 1.0;
    assert!(!Material::of_bodies(&[&paint, &wax]).paint_only);
    let opaque = Material::of_bodies(&[&paint, &paint]);
    assert!(opaque.paint_only);
    assert!(!Material::of_bodies(&[&paint, &glass]).paint_only);
    assert!(!Material::of_bodies(&[&paint, &metal]).paint_only);
    let march = March { step_scale: 1.0, steps_out: 120, steps_in: 50, layers: 2 };
    for world_faces in [false, true] {
        let v = Variant { mat: opaque, world_faces, near: true, light: LightFrom::Picture, ..Default::default() };
        let source = tracer_source(&march, v, false);
        assert_eq!(source.contains("\n  return paintB_seen(k, p, n, rd, 1.0);\n"), !world_faces);
        check("paint first surface and cut-face fallback", &source);
    }
}

#[test]
fn finite_lamps_and_coat_rough_reference() {
    let march = March { step_scale: 1.0, steps_out: 120, steps_in: 70, layers: 0 };
    for coat in [false, true] {
        for near in [false, true] {
            for light in [LightFrom::Picture, LightFrom::Lights] {
                let mat = gf_core::material::Variant { metal: true, rough: true, rough_ref: true, coat, ..Default::default() };
                let source = tracer_source(&march, Variant { mat, near, light, ..Default::default() }, false);
                assert_eq!(source.contains("return glossyLampWeights(p, n);"), near);
                assert_eq!(source.contains("return glossyLampRay(p, lights, r);"), near);
                assert_eq!(source.contains("return glossyLampPdf(p, lights, wo);"), near);
                assert_eq!(source.matches("fn roughLampRadiance(").count(), usize::from(near), "conductor-only emitter inclusion, coat={coat}, near={near}");
                check("finite-lamp proposals for bare and coated metal", &source);
            }
        }
    }
}


#[test]
fn a_sampled_construction_cut_uses_the_finite_lamp_distribution() {
    use gf_core::csg::{Op, Part};
    use gf_core::state::State;
    use std::collections::BTreeMap;

    let settings = [("scene", "studio"), ("preset", "block"), ("trace", "csg")]
        .map(|(key, value)| (key.to_string(), value.to_string()));
    let mut state = State::with_args(&settings, true);
    let path = state.csg.add_construction_part(&[], Part::default()).unwrap();
    let cut = state.csg.edit_construction().node_mut(&path).unwrap();
    cut.op = Op::Subtract;
    cut.material = Some(BTreeMap::from([
        ("metal".into(), 3.0), ("rough".into(), 0.65), ("roughRef".into(), 1.0),
    ]));
    let (parts, _) = state.csg_tree();
    let binding = gf_core::sdf::BINDING;
    let tree = gf_core::csg::wgsl_tree_s(&state.families, &gf_core::csg::kinds(&parts), binding, false, false, false);
    let declaration = state.construction_program(0, &parts).unwrap().shader(tree, binding, false);
    let march = March { step_scale: 1.0, steps_out: 120, steps_in: 70, layers: 0 };
    for near in [false, true] {
        let host = tracer_source(&march, Variant { near, ..Default::default() }, false);
        let cut = glassforms::sdf::weave_tree(&host, &declaration);
        assert!(cut.contains("fn cn1_frGlossySample("));
        assert_eq!(cut.contains("return cn1_glossyLampWeights(p, n);"), near);
        assert_eq!(cut.contains("return cn1_glossyLampPdf(p, lights, wo);"), near);
        check("rough copper construction cut under finite lamps", &cut);
    }
}

#[test]
fn a_selected_rough_conductor_samples_the_light_at_primary_and_secondary_hits() {
    let march = March { step_scale: 1.0, steps_out: 120, steps_in: 70, layers: 2 };
    for near in [false, true] {
        for light in [LightFrom::Picture, LightFrom::Lights] {
            for glass in [false, true] {
                let mat = gf_core::material::Variant {
                    bodies: 2, metal: true, rough: true, rough_ref: glass,
                    sampled_metal: true, scene_reflection: true, ..Default::default()
                };
                let src = tracer_source(&march, Variant { mat, near, light, ..Default::default() }, false);
                assert!(src.contains("return sampledMetalB(p, n, rd, k, xi, fc);"));
                assert!(src.contains("secondaryMetalB(q, nq, rd, b, m, lam)"));
                assert_eq!(src.contains("return glossyLampWeights(p, n);"), near);
                check("selected conductor in an assembly", &src);
            }
        }
    }
}

#[test]
fn an_opaque_construction_layer_uses_the_winning_faces_index() {
    use gf_core::{material, settings::Settings, state::State};

    let args = [("scene", "studio"), ("preset", "sphere"), ("trace", "csg"), ("material", "crown")]
        .map(|(name, value)| (name.into(), value.into()));
    for (id, film_nm) in [("lacquer",0.0), ("modelclay",0.0), ("moulded",0.0), ("satinpaint",0.0), ("satinpaint",385.0)] {
        let mut state = State::with_args(&args, true);
        let mut finish = Settings::default();
        let index = material::named_index(id).unwrap();
        material::NAMED[index].apply(&mut finish);
        finish.mat = index as f64;
        finish.film = film_nm;
        if id == "lacquer" {
            finish.coat = 0.0;
            finish.age = 0.0;
        }
        let node = state.csg.edit_construction().node_mut(&[0]).unwrap();
        node.material = Some(material::body_names().filter_map(|name| Some((name.to_owned(), finish.get(name)?))).collect());
        let (parts, _) = state.csg_tree();
        let binding = gf_core::sdf::BINDING;
        let tree = gf_core::csg::wgsl_tree_s(&state.families, &gf_core::csg::kinds(&parts), binding, false, false, false);
        let tree = state.construction_program(0, &parts).unwrap().shader(tree, binding, false);
        let march = March { step_scale: 1.0, steps_out: 120, steps_in: 70, layers: 0 };
        for near in [false, true] {
            let variant = Variant { near, ..Default::default() };
            let source = glassforms::sdf::weave_tree(&tracer_source(&march, variant, false), &tree);
            assert!(source.contains(&format!("shadePaintSurface(0,p,nrm(p),rd,1.0,cnIor(),cn1_mu.surf.z,{})", finish.coat>0.0)), "{id}");
            assert!(!source.contains("paintOpaqueDiffuseB") && source.contains("fn cn1_pbr_diffuse_albedo("), "{id}: the shared diffuse response");
            assert!(source.contains("fn cn1_samplePaintPanel("), "{id}");
            assert_eq!(source.contains("fn cn1_paintFilmWeights("), film_nm>0.0);
            assert_eq!(source.contains("let lampLight = cn1_paintLampsB(k, p, n);"), near, "{id}");
            check(id, &source);
        }
    }
}

#[test]
fn wavelength_dependent_paint_shares_the_solo_mixed_and_diffuse_sources() {
    use gf_core::material::Variant as Material;
    let march=March { step_scale:1.0,steps_out:120,steps_in:50,layers:2 };
    for material in [
        Material {ceramic:true,sss:true,film:true,coat:true,..Default::default()},
        Material {ceramic:true,sss:true,film:true,coat:false,..Default::default()},
        Material {bodies:3,paint_bodies:true,paint_film_bodies:true,film:true,coat:true,..Default::default()},
    ] {
        let source=tracer_source(&march,Variant { mat:material,..Default::default() },false);
        assert!(source.contains("fn paintFilmWeights("));
        check("painted film surface", &source);
    }
    check("painted film on a diffuse substrate",glassforms::surface::SURFACE_WGSL);
    let mut paint = gf_core::settings::Settings::default();
    let index = gf_core::material::named_index("moulded").unwrap();
    gf_core::material::NAMED[index].apply(&mut paint);
    let mut glass = gf_core::settings::Settings::default();
    glass.film = 360.0;
    for painted in [false,true] {
        paint.film = if painted { 385.0 } else { 0.0 };
        let mat = Material::of_bodies(&[&paint,&glass]);
        assert_eq!(mat.paint_film_bodies,painted);
        let source = tracer_source(&march,Variant {mat,..Default::default()},false);
        assert_eq!(source.contains("fn paintFilmWeights("),painted);
    }
}

#[test]
fn frost_grating_mono_with_direct_light_weights_post_diffraction_bounces() {
    let flags = [
        ("ROUGH", true),
        ("ROUGH_REF", true),
        ("GRATING", true),
        ("MONO", true),
        ("ENVL", true),
        ("RANDOM", true),
        ("GLOSSY_DISTRIBUTIONS", true),
        ("GGX_SAMPLE", true),
        ("FROST_BOUNDARY", true),
        ("FROST_DIELECTRIC", true),
        ("FROST_TRACK_ORDER", true),
        ("FROST_DIRECT", true),
        ("FROST_ROOM_EXIT", true),
    ];
    let decl = gf_core::material::preprocess(glassforms::render::TRACE_MATERIAL_WGSL, &flags)
        .replace("{{WEIGHT}}", "(wOf(i, xi, W) * thr)");
    let host = include_str!("../src/shaders/trace.wgsl")
        .replace(
            "{{DIELECTRIC_FRESNEL}}",
            concat!(
                include_str!("../src/shaders/pbr/hash.wgsl"),
                include_str!("../src/shaders/pbr/types.wgsl"),
                include_str!("../src/shaders/pbr/dielectric.wgsl"),
                include_str!("../src/shaders/dielectric_fresnel.wgsl"),
            ),
        )
        .replace("{{PROFILE_DECL}}", "")
        .replace("{{PROFILE_MAP}}", "")
        .replace("{{PROFILE_MAPN}}", "")
        .replace("{{PROFILE_OUT}}", "")
        .replace("{{STEP_SCALE}}", "0.95")
        .replace("{{STEPS_OUT}}", "220")
        .replace("{{STEPS_IN}}", "56")
        .replace("{{LAYERS}}", "2")
        .replace("{{LIGHT_DECL}}", &decl)
        .replace("{{ENV_LIGHT}}", "textureSampleLevel(envTex, envSmp, uv, 0.0).rgb")
        .replace("{{BACKGROUND}}", "vec3f(0.0)")
        .replace("{{SHADE}}", "frostDielectric(ro, rd, t, xi, W, fc)")
        .replace("{{GRAZE}}", "col");
    assert!(host.contains("var frostOrderWeighted = false;"));
    assert!(host.contains("sum+=weight*directFrost.added;\n      if (frostSampledOrder && !frostOrderWeighted) {\n        weight *= colour;\n        frostOrderWeighted = true;\n      }"));
    check("frost_reference GRATING + MONO + FROST_DIRECT", &host);
}
