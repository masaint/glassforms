//! The mathematical families (fractals, topology, hydrogen orbitals, polyhedra): what every one of their
//! presets must satisfy before a picture is ever drawn, as gens_all.rs asks of the families before them.
//! Shaders are compiled both ways, presets are complete, animated sliders travel, shapes are built at their
//! starting values and with each slider at both ends.
use std::collections::HashSet;

use gf_core::field;
use gf_core::presets::{self, Family, Preset, Shape};
use gf_core::state::{Detail, ShapeJob, State};
use naga::{AddressSpace, ShaderStage, TypeInner};

const NEW: [&str; 4] = ["fractals", "topology", "orbitals", "polyhedra"];

/// One family's name in GENS_ONLY narrows every check here to that family.
fn wanted(id: &str) -> bool {
    NEW.contains(&id) && std::env::var("GENS_ONLY").map_or(true, |only| only == id)
}

fn new_families() -> Vec<Family> {
    let mut all = vec![presets::fractals::family(), presets::topology::family(), presets::orbitals::family(), presets::polyhedra::family()];
    all.retain(|f| wanted(f.id));
    all
}

/// Parse and validate, or stop with the compiler's own account of what is wrong.
fn compiled(id: &str, src: &str) -> naga::Module {
    let module = naga::front::wgsl::parse_str(src)
        .unwrap_or_else(|e| panic!("{id}: the bake shader does not parse\n{}\n{src}", e.emit_to_string(src)));
    let mut validator = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty());
    if let Err(e) = validator.validate(&module) {
        panic!("{id}: the bake shader is not valid\n{}\n{src}", e.emit_to_string(src));
    }
    module
}

/// The uniform block is laid out the way `field::uniforms` fills it: hb, res, puff, a spare, then eight
/// floats of which the first are the preset's sliders, in order and under their own names.
fn check_block(p: &Preset, module: &naga::Module, compute: bool) {
    let id = p.id;
    let global = |name: &str| module.global_variables.iter().map(|(_, g)| g).find(|g| g.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("{id}: no global called {name}"));
    let u = global("u");
    assert_eq!(u.space, AddressSpace::Uniform, "{id}");
    let binding = u.binding.as_ref().unwrap();
    assert_eq!((binding.group, binding.binding), (0, 0), "{id}");
    let members = |ty: naga::Handle<naga::Type>| match &module.types[ty].inner {
        TypeInner::Struct { members, span } => (members.iter().map(|m| (m.name.clone().unwrap(), m.offset, m.ty)).collect::<Vec<_>>(), *span),
        other => panic!("{id}: not a struct: {other:?}"),
    };
    let (block, size) = members(u.ty);
    assert_eq!(size as usize, field::UNIFORM_SIZE, "{id}: size of the uniform block");
    let (pr, pr_size) = members(block[4].2);
    assert_eq!((pr_size, pr.len()), (32, field::MAX_PARAMS), "{id}");
    for (i, (name, offset, ty)) in pr.iter().enumerate() {
        let want = p.params.get(i).map_or(format!("_p{i}"), |d| field::member_name(d.id));
        assert_eq!((name.as_str(), *offset), (want.as_str(), 4 * i as u32), "{id}: parameter {i}");
        assert!(matches!(module.types[*ty].inner, TypeInner::Scalar(naga::Scalar::F32)), "{id}: parameter {i} is not an f32");
    }
    let stages: Vec<(&str, ShaderStage)> = module.entry_points.iter().map(|e| (e.name.as_str(), e.stage)).collect();
    let mut want = vec![("vs", ShaderStage::Vertex), ("fs", ShaderStage::Fragment)];
    if compute { want.push(("cs", ShaderStage::Compute)); }
    assert_eq!(stages, want, "{id}");
}

#[test]
fn the_mathematical_families_are_registered() {
    let ids: Vec<&str> = presets::families().iter().map(|f| f.id).collect();
    for id in NEW { assert!(ids.contains(&id), "{id} is not among {ids:?}"); }
    for f in new_families() {
        assert!(!f.name.is_empty() && f.name != f.id, "{}: a name to show", f.id);
        assert!(!f.presets.is_empty(), "{}: no presets", f.id);
    }
}

#[test]
fn every_new_bake_shader_compiles_both_ways() {
    let mut n = 0;
    for fam in new_families() {
        for p in &fam.presets {
            if !matches!(p.shape, Shape::Field { .. }) { continue; }
            // drawn slice by slice, and written by a compute shader
            check_block(p, &compiled(p.id, &field::bake_shader(p)), false);
            check_block(p, &compiled(p.id, &field::bake_shader_compute(p)), true);
            n += 1;
        }
    }
        println!("{n} bake shaders parsed and validated, each in both forms");
}

#[test]
fn ids_are_unique_across_all_families() {
    let fams = presets::families();
    let mut fam_ids = HashSet::new();
    let mut preset_ids = HashSet::new();
    for f in &fams {
        assert!(fam_ids.insert(f.id), "two families called {}", f.id);
        for p in &f.presets {
            assert!(preset_ids.insert(p.id), "two shapes called {} (the second in {})", p.id, f.id);
            // an id travels on the command line and in file names
            assert!(!p.id.is_empty() && p.id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()), "{}: id \"{}\"", f.id, p.id);
        }
    }
}

#[test]
fn every_new_preset_is_complete() {
    for fam in new_families() {
        let fid = fam.id;
        assert!(!fam.lights.is_empty(), "{fid}: no lights");
        assert!(fam.step_scale > 0.3 && fam.step_scale <= 1.0, "{fid}: step scale {}", fam.step_scale);
        assert!(fam.steps_out >= 60 && fam.steps_in >= 24, "{fid}: marching steps");
        let family_rot = fam.defaults.iter().any(|(k, _)| *k == "rot");
        for p in &fam.presets {
            let id = p.id;
            assert!(!p.name.is_empty(), "{fid}/{id}: no name");
            assert!(!p.eq.is_empty() && p.eq.iter().all(|l| !l.is_empty()), "{fid}/{id}: no mathematics to show");
            assert!(p.rot.is_some() || family_rot, "{fid}/{id}: no starting orientation");
            let mut seen = HashSet::new();
            for d in &p.params {
                assert!(seen.insert(d.id), "{fid}/{id}: two sliders called {}", d.id);
                assert!(!d.label.is_empty(), "{fid}/{id}: slider {} has no label", d.id);
                assert!(d.lo < d.hi && d.step > 0.0, "{fid}/{id}: slider {} runs from {} to {} by {}", d.id, d.lo, d.hi, d.step);
                assert!(d.value >= d.lo && d.value <= d.hi, "{fid}/{id}: slider {} starts at {}, outside {}..{}", d.id, d.value, d.lo, d.hi);
                // the range is a whole number of steps, so the slider can reach both ends
                let steps = (d.hi - d.lo) / d.step;
                assert!((steps - steps.round()).abs() < 1e-6, "{fid}/{id}: slider {} cannot reach its end in steps of {}", d.id, d.step);
            }
            match &p.shape {
                Shape::Field { .. } => {
                    assert!(p.params.len() <= field::MAX_PARAMS, "{fid}/{id}: {} sliders", p.params.len());
                    assert!(p.tube.is_none() && p.radius.is_none() && p.info.is_none(), "{fid}/{id}: an equation with a curve's settings");
                }
                _ => panic!("{fid}/{id}: every shape here is an equation"),
            }
        }
    }
}

#[test]
fn every_animated_slider_exists_and_can_travel() {
    let mut s = State::new();
    for f in 0..s.families.len() {
        if !wanted(s.families[f].id) { continue; }
        s.set_scene(f);
        for p in 0..s.family().presets.len() {
            s.choose(p);
            let who = format!("{}/{}", s.family().id, s.preset().id);
            let animate = s.preset().animate.clone().unwrap_or_else(|| panic!("{who}: nothing travels when the animation plays"));
            assert!(!animate.is_empty(), "{who}: nothing travels when the animation plays");
            let knobs = s.shape_knobs();
            for id in &animate {
                let k = knobs.iter().find(|k| k.id == *id).unwrap_or_else(|| panic!("{who}: animates \"{id}\", which is not one of its sliders"));
                assert!(k.wave, "{who}: \"{id}\" moves in whole steps, so it cannot travel smoothly");
                // choosing the shape gave it a track that is switched on, between two different stops inside the range
                let tr = s.tracks.get(&k.key).unwrap_or_else(|| panic!("{who}: no track for \"{id}\""));
                assert!(tr.on && tr.from < tr.to && tr.from >= k.lo - 1e-9 && tr.to <= k.hi + 1e-9, "{who}: track of \"{id}\" runs {}..{}", tr.from, tr.to);
            }
        }
    }
}

#[test]
fn every_new_preset_builds_at_its_defaults_and_at_each_sliders_two_ends() {
    let mut s = State::new();
    let mut built = 0;
    for f in 0..s.families.len() {
        if !wanted(s.families[f].id) { continue; }
        s.set_scene(f);
        for p in 0..s.family().presets.len() {
            s.choose(p);
            let who = format!("{}/{}", s.family().id, s.preset().id);
            let defaults = s.preset().defaults();
            let ranges: Vec<(f64, f64)> = s.preset().params.iter().map(|d| (d.lo, d.hi)).collect();
            // the starting values, then each slider in turn at its lowest and at its highest
            let mut cases = vec![defaults.clone()];
            for (i, (lo, hi)) in ranges.iter().enumerate() {
                for v in [lo, hi] {
                    let mut c = defaults.clone();
                    c[i] = *v;
                    cases.push(c);
                }
            }
            for vals in cases {
                s.shape_mut().vals = vals.clone();
                let job = s.shape_job(Detail::Full, Some(96)).unwrap_or_else(|e| panic!("{who} at {vals:?}: {e}"));
                match &job.shape {
                    ShapeJob::Field { uniforms } => {
                        assert!(uniforms.iter().all(|v| v.is_finite()), "{who} at {vals:?}");
                        for (i, v) in vals.iter().enumerate() {
                            assert_eq!(uniforms[4 + i], *v as f32, "{who}: slider {i} is not where the shader reads it");
                        }
                    }
                    ShapeJob::Curve { curve, .. } => {
                        assert!(curve.n > 10 && curve.pts.iter().all(|v| v.is_finite()), "{who} at {vals:?}: not finite");
                        // inside the volume, tube and all
                        assert!(curve.pts.chunks(4).all(|q| q[..3].iter().all(|c| c.abs() + q[3] < 1.5)), "{who} at {vals:?}: reaches outside the volume");
                    }
                    ShapeJob::Mesh(_) => panic!("{who}: a model among the equations"),
                }
                built += 1;
            }
        }
    }
    println!("{built} builds");
}
