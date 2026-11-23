//! The census of weave anchors (app/src/weave.rs). A weave finds a line of a shader's text and puts
//! something else in its place; when that line is reworded the weave silently does nothing. Here every
//! named anchor is counted while a broad, fixed set of tracers is built, and the counts must stay as
//! recorded in `EXPECTED`: for each anchor, how many looks there were, in how many it was found, and the
//! most times it was found in one look. A change fails with the anchor's name.
//!
//! When a change to the variants below or to the shaders is intended, run with WEAVE_CENSUS_PRINT=1 and
//! `--nocapture` to print the table as it now stands, read what moved, and paste it in.
//! Needs no GPU.

use glassforms::render::{tracer_source, tracer_source_lens, LightFrom, March, Variant};
use glassforms::stage::Seen;
use glassforms::surface::Kind;
use glassforms::weave::{census, Census};
use gf_core::material::Variant as M;

/// (name, looks, looks that found it, most found in one look)
type Row = (&'static str, usize, usize, usize);

/// Anchors that must stand at most once wherever they are looked for: a second copy means the weave
/// edits a place it was not written for. (cast.behind is woven only for the checks' test floor, an
/// environment setting, so no tracer built here looks for it.)
const AT_MOST_ONCE: &[&str] = &[
    "cast.clay_light", "cast.clay_sheen", "cast.paint_focus",
    "surface.clay_stretch", "surface.clay_light", "surface.clay_sheen",
    "lamps.env", "lamps.nrm", "lamps.nrm_b", "lamps.env_l", "lamps.mirror_env_picture", "lamps.mirror_env_lights",
    "lamps.worn_panorama", "lamps.clay", "lamps.floor", "lamps.seen", "lamps.finite_ggx", "lamps.imported_specular",
    "lamps.fr_scene_lights", "lamps.fr_scene_ray", "lamps.fr_scene_pdf", "lamps.shadow_stub", "lamps.pixel_at",
    // (not lamps.paint: since perffloor f62530b it stands in paintEmitterOneB too, which the replace-all
    // is meant to reach)
    "stage.metal_coat", "stage.terminal_room", "stage.metal_reflection", "stage.sampled_metal_escape",
    "render.atmo", "render.leave", "render.rough_shift", "render.grown_bound", "render.grown_hb", "render.grown_fs", "render.eye_ro",
    "render.gem_march", "dlss.centre",
];

/// Anchors that no tracer built here contains, and why. One that starts to match fails too, so the list
/// stays true; a BROKEN one is a weave that silently stopped working (fixing it changes the picture).
const NEVER_FOUND: &[(&str, &str)] = &[
    ("model_surface.beyond", "BROKEN since f9c4c2d: beyond() now has `let n = faceIncident(nrm(p), -rd);` and beyondM a `//#if ROUGH` after `let p = ro + rd * t;`, so an imported surface is not looked for where a ray leaves plain glass or a material body"),
    ("construction.class_disp", "dormant: no material text reads u.disp any more (only trace.wgsl's iorAt and `alike`, which the host weave covers)"),
    ("sdf.cone_skip", "BROKEN since 8dd2cd1: the outside march between glass layers moved from beyond()/beyondM into marchBeyond, whose loop is indented 4 spaces, not 6; a cone wrap no longer skips the empty inside of its envelope there (film.wgsl beyondB still has the 6-space text but no host built here includes it)"),
    ("sdf.profile_outside_normal", "BROKEN since f9c4c2d: beyond() and beyondM read `let n = faceIncident(nrm(p), -rd);`, so a ray re-entering a wrapped profile takes the volume normal, not the settled profile normal"),
    ("sdf.profile_shade_normal", "BROKEN since f9c4c2d: shade() reads `let n = faceIncident(nrm(p), -rd);`, so the camera hit on a prepared cone takes the volume normal, not the settled profile normal"),
    ("construction.class_iu", "not driven: only a class with inner parts declares IU, and no cut built here has one"),
];
fn march() -> March {
    March { step_scale: 0.95, steps_out: 220, steps_in: 56, layers: 2 }
}

/// Single-body tracers: every sample material, lit both ways, with and without lamps and casting, on a
/// bare stage and on a floor that takes what is cast; and the same as clay, as clay lit by the surface
/// pass, and with a few of the other ways of drawing.
fn single_body() -> Vec<Variant> {
    let mut all = Vec::new();
    let floor = Seen { floor: true, cast: true, ..Default::default() };
    let samples: Vec<M> = M::samples().into_iter().filter(|m| m.bodies == 0).collect();
    for mat in &samples {
        for light in [LightFrom::Lights, LightFrom::Picture] {
            for (near, cast) in [(false, false), (true, false), (false, true), (true, true)] {
                for stage in [Seen::default(), floor] {
                    all.push(Variant { light, mat: *mat, near, cast, stage, ..Default::default() });
                }
            }
            all.push(Variant { light, clay: true, mat: mat.for_clay(), near: true, cast: true, ..Default::default() });
        }
    }
    for light in [LightFrom::Lights, LightFrom::Picture] {
        for backdrop in [false, true] {
            for (near, cast) in [(false, false), (true, false), (false, true), (true, true)] {
                for surface in [Kind::None, Kind::Clay] {
                    all.push(Variant { light, backdrop, clay: true, near, cast, surface, ..Default::default() });
                    all.push(Variant { light, backdrop, clay: true, near, cast, surface, stage: floor, ..Default::default() });
                }
            }
        }
    }
    for (grown, eye, denoise) in [(true, false, false), (false, true, false), (false, false, true), (true, true, true)] {
        all.push(Variant { grown, eye, denoise, ..Default::default() });
        all.push(Variant { grown, eye, denoise, mat: samples[samples.len() / 2], ..Default::default() });
        all.push(Variant { grown, eye, denoise, clay: true, ..Default::default() });
    }
    for mat in [M { metal: true, ..M::default() }, M { metal: true, coat: true, ..M::default() }, M { metal: true, rough: true, rough_ref: true, ..M::default() }] {
        for near in [false, true] {
            all.push(Variant { mat, near, cast: true, stage: floor, ..Default::default() });
        }
    }
    let atmo = Seen { floor: true, atmo: true, sky: true, ..Default::default() };
    all.push(Variant { stage: atmo, ..Default::default() });
    all.push(Variant { stage: atmo, near: true, ..Default::default() });
    all
}

/// Tracers of several bodies: the parts of tests/shaders.rs `tracer_bodies`, painted assemblies, sampled
/// conductors, cut faces and imported surfaces, with and without lamps and casting.
fn assemblies() -> Vec<Variant> {
    let d = M::default();
    let mut parts = vec![d, M { film: true, ..d }, M { rough: true, ..d }, M { sss: true, ..d }, M { metal: true, ..d },
        M { hollow: true, film: true, ..d }, M { metal: true, rough: true, film: true, hollow: true, sss: true, ..d },
        M { sss: true, walk: true, metal: true, rough: true, ..d }, M { spectrum: true, grin: true, biref: true, ..d },
        M { grating: true, metal: true, rough: true, ..d }, M { rough: true, rough_ref: true, ..d },
        M { metal: true, rough: true, rough_ref: true, sampled_metal: true, scene_reflection: true, ..d },
        M { metal: true, sampled_metal: true, scene_reflection: true, ..d }];
    let paint = M { paint_bodies: true, paint_exchange: true, ..d };
    parts.extend([paint, M { paint_coat_bodies: true, ..paint }, M { film: true, paint_film_bodies: true, ..paint },
        M { wear: true, metal: true, rough: true, ..paint }, M { sss: true, walk: true, rough: true, rough_ref: true, ..paint },
        M { paint_only: true, ..paint }]);
    let floor = Seen { floor: true, cast: true, multi: true, ..Default::default() };
    let mut all = Vec::new();
    for p in parts {
        for bodies in [2u8, 3] {
            let mat = M { bodies, ..p };
            for light in [LightFrom::Lights, LightFrom::Picture] {
                for (near, cast) in [(false, false), (true, false), (true, true)] {
                    all.push(Variant { light, mat, near, cast, ..Default::default() });
                    all.push(Variant { light, mat, near, cast, stage: floor, ..Default::default() });
                }
                all.push(Variant { light, mat, near: true, world_faces: true, ..Default::default() });
                all.push(Variant { light, mat, near: true, textured: true, ..Default::default() });
            }
        }
    }
    for light in [LightFrom::Lights, LightFrom::Picture] {
        for near in [false, true] {
            all.push(Variant { light, near, textured: true, ..Default::default() });
            all.push(Variant { light, near, textured: true, mat: M { metal: true, rough: true, rough_ref: true, ..d }, ..Default::default() });
        }
    }
    all
}

/// The tracers of construction cuts (sdf::weave_tree with material classes), as tests/shaders.rs builds them.
fn construction_cuts() -> Vec<String> {
    use gf_core::{csg::{Op, Part}, material, settings::Settings, state::State};
    use std::collections::BTreeMap;
    let binding = gf_core::sdf::BINDING;
    let m = March { step_scale: 1.0, steps_out: 120, steps_in: 70, layers: 0 };
    let mut out = Vec::new();
    // a sampled rough copper cut
    let settings = [("scene", "studio"), ("preset", "block"), ("trace", "csg")].map(|(k, v)| (k.to_string(), v.to_string()));
    let mut state = State::with_args(&settings, true);
    let path = state.csg.add_construction_part(&[], Part::default()).unwrap();
    let cut = state.csg.edit_construction().node_mut(&path).unwrap();
    cut.op = Op::Subtract;
    cut.material = Some(BTreeMap::from([("metal".into(), 3.0), ("rough".into(), 0.65), ("roughRef".into(), 1.0)]));
    let (parts, _) = state.csg_tree();
    let tree = gf_core::csg::wgsl_tree_s(&state.families, &gf_core::csg::kinds(&parts), binding, false, false, false);
    let decl = state.construction_program(0, &parts).unwrap().shader(tree, binding, false);
    for near in [false, true] {
        for mat in [M::default(), M { film: true, ..M::default() }, M { metal: true, ..M::default() }] {
            out.push(glassforms::sdf::weave_tree(&tracer_source(&m, Variant { near, mat, ..Default::default() }, false), &decl));
        }
    }
    // opaque layers of painted and clay finishes
    let args = [("scene", "studio"), ("preset", "sphere"), ("trace", "csg"), ("material", "crown")].map(|(k, v)| (k.into(), v.into()));
    for (id, film_nm) in [("lacquer", 0.0), ("modelclay", 0.0), ("moulded", 0.0), ("satinpaint", 0.0), ("satinpaint", 385.0)] {
        let mut state = State::with_args(&args, true);
        let mut finish = Settings::default();
        let index = material::named_index(id).unwrap();
        material::NAMED[index].apply(&mut finish);
        finish.mat = index as f64;
        finish.film = film_nm;
        if id == "lacquer" { finish.coat = 0.0; finish.age = 0.0; }
        let node = state.csg.edit_construction().node_mut(&[0]).unwrap();
        node.material = Some(material::body_names().filter_map(|name| Some((name.to_owned(), finish.get(name)?))).collect());
        let (parts, _) = state.csg_tree();
        let tree = gf_core::csg::wgsl_tree_s(&state.families, &gf_core::csg::kinds(&parts), binding, false, false, false);
        let tree = state.construction_program(0, &parts).unwrap().shader(tree, binding, false);
        for near in [false, true] {
            out.push(glassforms::sdf::weave_tree(&tracer_source(&m, Variant { near, ..Default::default() }, false), &tree));
        }
    }
    out
}

/// Tree declarations of a cone wrap (tests/mods.rs `the_prepared_cone_uses_free_uniform_slots`): the
/// generic text, and the prepared one with its uniform profile.
fn cone_decls() -> Vec<String> {
    use gf_core::mods::{Kind, Modifier, Mods};
    let fams = gf_core::presets::families();
    let f = fams.iter().position(|f| f.id == "studio").unwrap();
    let mut out = Vec::new();
    for prepared in [false, true] {
        let mut parts = Vec::new();
        for (name, rot, size) in [("rectframe", [0., 0., 45.], 0.6), ("washer", [90., 0., 0.], 0.2)] {
            let preset = fams[f].presets.iter().find(|p| p.id == name).unwrap();
            let values = preset.params.iter().map(|p| p.value).collect::<Vec<_>>();
            let mut part = gf_core::csg::Part::of(&fams[f], preset, &values, 0., Default::default()).resolve(&fams).unwrap();
            part.place.rot = rot;
            part.place.size = size;
            parts.push(part);
            if !prepared { break; }
        }
        let kinds = gf_core::csg::kinds(&parts);
        let mut cone = Modifier::of(Kind::Wrap);
        cone.mode = 1;
        cone.a.v = [1.15, 1.7, 0.42];
        cone.size = 0.17;
        let mut stack = Mods::default();
        stack.list.push(cone);
        let plan = glassforms::csg::TreeMods::plan(&fams, Some(&stack), &parts, &[], false, &Default::default());
        let base = gf_core::csg::wgsl_tree_s(&fams, &kinds, gf_core::sdf::BINDING, false, true, false);
        out.push(plan.tracer_text(base));
    }
    out
}

fn take() -> Census {
    let m = march();
    census(|| {
        for v in single_body().into_iter().chain(assemblies()) {
            let src = tracer_source(&m, v, false);
            let _ = tracer_source(&m, v, true);
            if v.mat.bodies == 0 && !v.textured && !v.world_faces {
                let _ = tracer_source_lens(&m, v, false);
            }
            let _ = glassforms::dlss::centre(&src);
            let _ = glassforms::render::leave_weave(&src, true);
            let _ = glassforms::render::cut_weave(&src, 0.05);
        }
        // the plain tracer's own weaves: a cut stone, one wavelength inside, the smaller frame's antialiasing
        let plain = tracer_source(&m, Variant::default(), false);
        for src in [plain.clone(), glassforms::render::gem_weave(&plain), glassforms::render::gem_carved_weave(&plain)] {
            for clamp in [0u8, 3] {
                let h = glassforms::hero::Hero { mode: 1, k: 1, clamp, seq: 0, every: 6, taa: 0, split: 0 };
                if let Some(w) = glassforms::hero::weave(&src, &h) { let _ = glassforms::hero::split(&w); }
            }
        }
        let _ = glassforms::render::accum_dyn(glassforms::render::ACCUM_SOURCE);
        let _ = glassforms::hero::chroma_taa(glassforms::render::ACCUM_SOURCE);
        // every formula, into the plain tracer, clay, a material and the lens tracer
        let hosts = [plain.clone(), tracer_source(&m, Variant { clay: true, ..Default::default() }, false),
            tracer_source(&m, Variant { mat: M { metal: true, ..M::default() }, ..Default::default() }, false),
            tracer_source_lens(&m, Variant::default(), false),
            tracer_source(&m, Variant { mat: M { film: true, ..M::default() }, ..Default::default() }, false),
            tracer_source(&m, Variant { mat: M { film: true, mono: true, ..M::default() }, ..Default::default() }, false),
            glassforms::hero::weave(&plain, &glassforms::hero::Hero { mode: 1, k: 1, clamp: 0, seq: 0, every: 6, taa: 0, split: 0 }).unwrap(),
            glassforms::render::gem_weave(&plain)];
        let cone_hosts = [plain.clone(),
            tracer_source(&m, Variant { mat: M { film: true, ..M::default() }, ..Default::default() }, false),
            tracer_source(&m, Variant { mat: M { metal: true, ..M::default() }, ..Default::default() }, false),
            tracer_source(&m, Variant { mat: M { sss: true, ..M::default() }, ..Default::default() }, false),
            tracer_source_lens(&m, Variant::default(), false)];
        for f in gf_core::presets::families() {
            for p in &f.presets {
                let Some(decl) = gf_core::sdf::tracer_decl(&f, p) else { continue };
                for host in &hosts { let _ = glassforms::sdf::weave(host, &decl); }
            }
        }
        let _ = glassforms::sdf::weave(&plain, glassforms::sdf::GUARD_DECL);
        // a cone wrap over two studio parts, as tests/mods.rs prepares it, with and without its uniform profile
        for cone in cone_decls() {
            for host in &cone_hosts { let _ = glassforms::sdf::weave(host, &cone); }
        }
        let _ = construction_cuts();
        // a lens tracer with light bouncing on the floor (the stage reads a pixel back from a ray there too)
        let gi = Seen { floor: true, gi: true, ..Default::default() };
        let _ = tracer_source_lens(&m, Variant { stage: gi, ..Default::default() }, false);
    })
    .1
}

/// Anchors named in core (gf_core::mods, csg, ...): core/tests/weave_census.rs drives and holds them.
const CORE: &[&str] = &["mods.", "csg.", "csg_profile.", "construction_eval.", "construction_pattern."];

fn rows(c: &Census) -> Vec<Row> {
    c.iter().filter(|(name, _)| !CORE.iter().any(|p| name.starts_with(p))).map(|(name, looks)| (*name, looks.len(), looks.iter().filter(|&&n| n > 0).count(), looks.iter().copied().max().unwrap_or(0))).collect()
}

const EXPECTED: &[Row] = &[
    ("camera.pixel_from_ray", 679, 678, 1),
    ("camera.stage_gi_pixel", 679, 1, 1),
    ("cast.clay_light", 5218, 3128, 1),
    ("cast.clay_sheen", 5218, 3128, 1),
    ("cast.paint_focus", 5218, 96, 1),
    ("construction.absorb", 16, 16, 4),
    ("construction.beyond", 16, 16, 1),
    ("construction.beyond_call", 16, 16, 1),
    ("construction.body", 16, 2, 2),
    ("construction.class_absorb", 16, 10, 5),
    ("construction.class_disp", 16, 0, 0),
    ("construction.class_ior", 16, 16, 6),
    ("construction.class_iu", 16, 0, 0),
    ("construction.class_mu", 16, 16, 1),
    ("construction.disp", 16, 16, 2),
    ("construction.entry", 16, 16, 1),
    ("construction.face_visibility", 10, 10, 1),
    ("construction.ior", 16, 16, 3),
    ("construction.paint_cb", 10, 10, 1),
    ("construction.paint_mu", 10, 10, 5),
    ("construction.paint_seen", 10, 10, 1),
    ("construction.pick", 16, 16, 1),
    ("construction.secondary", 16, 16, 1),
    ("construction.shade_call", 16, 16, 1),
    ("dlss.centre", 3580, 3324, 1),
    ("hero.chroma_taa", 1, 1, 1),
    ("hero.clamp", 7, 7, 1),
    ("hero.exit", 7, 7, 1),
    ("hero.loop_head", 7, 7, 1),
    ("hero.loop_tail", 7, 7, 1),
    ("hero.return", 7, 7, 1),
    ("hero.sample", 7, 7, 1),
    ("hero.shade", 7, 7, 1),
    ("hero.split_decl", 6, 6, 1),
    ("hero.split_global", 6, 6, 1),
    ("hero.split_head", 6, 6, 1),
    ("hero.split_out", 6, 6, 1),
    ("hero.split_ret", 6, 3, 1),
    ("hero.split_tail", 6, 6, 1),
    ("lamps.clay", 5836, 3588, 1),
    ("lamps.emitter_env_l0", 4, 4, 2),
    ("lamps.env", 5836, 5836, 1),
    ("lamps.env_l", 5836, 4118, 1),
    ("lamps.finite_ggx", 662, 662, 1),
    ("lamps.floor", 5836, 2281, 1),
    ("lamps.fr_scene_lights", 250, 250, 1),
    ("lamps.fr_scene_pdf", 250, 250, 1),
    ("lamps.fr_scene_ray", 250, 250, 1),
    ("lamps.imported_specular", 160, 160, 1),
    ("lamps.mirror_env_lights", 5836, 2064, 1),
    ("lamps.mirror_env_picture", 5836, 2513, 1),
    ("lamps.nrm", 5836, 5836, 1),
    ("lamps.nrm_b", 5836, 912, 1),
    ("lamps.paint", 293, 293, 2),
    ("lamps.pixel_at", 2697, 2697, 1),
    ("lamps.seen", 5836, 5836, 1),
    ("lamps.shadow_stub", 2697, 2697, 1),
    ("lamps.worn_panorama", 5836, 156, 1),
    ("model_surface.beyond", 320, 0, 0),
    ("model_surface.beyond_paint", 320, 152, 1),
    ("model_surface.shade_b", 152, 144, 1),
    ("model_surface.walk", 24, 24, 1),
    ("render.accum_blend", 1, 1, 1),
    ("render.accum_cur", 1, 1, 1),
    ("render.accum_near", 1, 1, 1),
    ("render.accum_top", 1, 1, 1),
    ("render.atmo", 6, 6, 1),
    ("render.cut_bounce", 3580, 3580, 2),
    ("render.cut_reflected", 3580, 3580, 1),
    ("render.eye_rest", 2979, 2648, 3),
    ("render.eye_ro", 2979, 12, 1),
    ("render.gem_march", 3, 3, 1),
    ("render.grown_bound", 18, 18, 1),
    ("render.grown_fs", 18, 18, 1),
    ("render.grown_hb", 18, 18, 1),
    ("render.leave", 3580, 3580, 1),
    ("render.rough_shift", 256, 256, 1),
    ("sdf.cone_entry", 10, 10, 1),
    ("sdf.cone_k", 10, 10, 1),
    ("sdf.cone_mid", 10, 10, 1),
    ("sdf.cone_skip", 10, 0, 0),
    ("sdf.cone_stretch", 10, 10, 1),
    ("sdf.gem_uneven", 1666, 205, 1),
    ("sdf.guard_map_read", 1640, 1640, 1),
    ("sdf.guard_mapn_head", 1666, 1666, 1),
    ("sdf.guard_nrm_step", 1667, 1667, 1),
    ("sdf.hero_wave_loop", 464, 58, 1),
    ("sdf.hit", 200, 200, 1),
    ("sdf.loop", 464, 464, 1),
    ("sdf.mapn_head", 1666, 1666, 1),
    ("sdf.mat_mono_loop", 464, 58, 1),
    ("sdf.mat_wave_loop", 464, 58, 1),
    ("sdf.nrm_step", 1667, 1667, 1),
    ("sdf.profile_camera_exit", 2, 2, 1),
    ("sdf.profile_camera_loop", 2, 2, 1),
    ("sdf.profile_camera_miss", 2, 2, 1),
    ("sdf.profile_inside_normal", 20, 20, 1),
    ("sdf.profile_outside_normal", 10, 0, 0),
    ("sdf.profile_refine", 2, 2, 1),
    ("sdf.profile_shade_env", 2, 2, 1),
    ("sdf.profile_shade_normal", 2, 0, 0),
    ("sdf.profile_warp_step", 2, 2, 1),
    ("sdf.read_d", 464, 464, 1),
    ("sdf.sample_head", 552, 552, 1),
    ("sdf.wave_loop", 464, 406, 3),
    ("stage.metal_coat", 54, 36, 1),
    ("stage.metal_reflected", 102, 102, 1),
    ("stage.metal_reflection", 56, 56, 1),
    ("stage.sampled_metal_escape", 56, 16, 1),
    ("stage.terminal_room", 56, 56, 1),
    ("surface.clay_light", 96, 96, 1),
    ("surface.clay_sheen", 96, 96, 1),
    ("surface.clay_stretch", 96, 96, 1),
    ("world_faces.alike", 152, 152, 1),
    ("world_faces.comb", 152, 152, 1),
    ("world_faces.exchange", 152, 48, 1),
    ("world_faces.field", 6840, 1152, 15),
    ("world_faces.finish", 152, 152, 1),
    ("world_faces.frost_dispersion", 152, 24, 1),
    ("world_faces.frost_primary", 152, 24, 1),
    ("world_faces.mat", 1368, 984, 113),
    ("world_faces.paint_b", 152, 152, 1),
    ("world_faces.paint_b_walk", 152, 32, 1),
    ("world_faces.paint_k", 152, 152, 2),
    ("world_faces.paint_nx", 152, 152, 1),
];

#[test]
fn every_weave_anchor_is_found_as_recorded() {
    let now = rows(&take());
    if std::env::var("WEAVE_CENSUS_PRINT").is_ok() {
        println!("const EXPECTED: &[Row] = &[");
        for (name, looks, hit, max) in &now { println!("    ({name:?}, {looks}, {hit}, {max}),"); }
        println!("];");
    }
    let mut wrong = Vec::new();
    for &(name, looks, hit, max) in &now {
        match NEVER_FOUND.iter().find(|e| e.0 == name) {
            Some((_, why)) if hit == 0 => eprintln!("weave anchor {name} is never found: {why}"),
            Some(_) => wrong.push(format!("{name}: listed in NEVER_FOUND but now found in {hit} of {looks} looks")),
            None if hit == 0 => wrong.push(format!("{name}: never found in {looks} looks (a weave that does nothing)")),
            None => {}
        }
        if max > 1 && AT_MOST_ONCE.contains(&name) { wrong.push(format!("{name}: found {max} times in one look, expected at most once")); }
        match EXPECTED.iter().find(|e| e.0 == name) {
            None => wrong.push(format!("{name}: not in EXPECTED (now {looks} looks, {hit} found, at most {max})")),
            Some(&(_, l, h, x)) if (l, h, x) != (looks, hit, max) =>
                wrong.push(format!("{name}: was {l} looks, {h} found, at most {x}; now {looks}, {hit}, {max}")),
            _ => {}
        }
    }
    for &(name, ..) in EXPECTED {
        if !now.iter().any(|r| r.0 == name) { wrong.push(format!("{name}: no longer looked for by any tracer built here")); }
    }
    for name in AT_MOST_ONCE {
        if !EXPECTED.iter().any(|e| e.0 == *name) { wrong.push(format!("{name}: listed in AT_MOST_ONCE but not in EXPECTED")); }
    }
    assert!(wrong.is_empty(), "weave anchors changed:\n  {}", wrong.join("\n  "));
}
