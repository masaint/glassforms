//! The named materials and the controls: every name means something, every value is one its slider can
//! reach, plain glass is left exactly as it was, and the program's state takes materials by name.

use std::collections::HashSet;

use gf_core::material::*;
use gf_core::settings::Settings;
use gf_core::state::State;

fn args(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

/// The range a setting's slider allows.
fn range(name: &str) -> Option<(f64, f64)> {
    if let Some(c) = CONTROLS.iter().find(|c| c.name == name) { return Some((c.lo, c.hi)); }
    State::engine_knobs().iter().find(|k| k.id == name).map(|k| (k.lo, k.hi))
}

#[test]
fn defaults_are_plain_glass() {
    let p = Settings::default();
    let v = Variant::of(&p);
    assert_eq!(v, Variant::default());
    assert!(v.plain());
    assert_eq!(v.key(), "glass");
    assert!(bubbles(&p).is_none());
    assert_eq!(fit_scale(&p, Some(0.06)), 1.0);
    // and whatever family is up, the state starts on it
    for f in ["attractors", "knots", "surfaces", "algebraic"] {
        let s = State::with_args(&args(&[("scene", f)]), true);
        assert!(Variant::of(&s.p).plain(), "{f} does not start as plain glass");
        assert_eq!(s.p.mat, 0.0);
        assert_eq!(s.p.density, 1.0);
    }
}

#[test]
fn every_named_material_is_within_its_sliders() {
    let mut ids = HashSet::new();
    assert_eq!(NAMED[0].id, "glass");
    assert!(NAMED[0].is_family());
    for n in NAMED {
        assert!(ids.insert(n.id), "two materials called {}", n.id);
        assert!(!n.name.is_empty() && !n.group.is_empty() && !n.note.is_empty());
        let mut p = Settings::default();
        for (k, v) in n.sets {
            assert!(p.set(k, *v), "{}: there is no setting called {k}", n.id);
            let (lo, hi) = range(k).unwrap_or_else(|| panic!("{}: {k} has no slider", n.id));
            assert!((lo..=hi).contains(v), "{}: {k} = {v} is outside {lo}..{hi}", n.id);
            assert!(*k != "fit" && *k != "mat", "{}: {k} is not for a material to set", n.id);
        }
        if let Some(a) = n.absorb { assert!(a.iter().all(|x| (0.0..=30.0).contains(x))); }
        // what it comes to
        let mut p = Settings::default();
        n.apply(&mut p);
        let v = Variant::of(&p);
        assert_eq!(n.costly, v.walk || v.grin || v.biref, "{}: costly is {} but it builds {}", n.id, n.costly, v.key());
        if p.metal >= 0.5 { assert!(p.metal == 1.0 || metal_of(&p).is_some(), "{}: no such metal", n.id); }
        if p.spectrum >= 0.5 { assert!(spectrum_of(&p).is_some(), "{}: no such spectrum", n.id); }
        let u = uniforms(&p, 1.0);
        assert!(u.iter().all(|x| x.is_finite()), "{}: its numbers are not all finite", n.id);
        let sw = swatch(named_index(n.id).unwrap(), [0.1, 0.05, 0.02]);
        assert!(sw.iter().all(|c| (0.0..=1.0).contains(c)));
    }
    // each material is complete in itself: choosing one after another leaves nothing of the first
    let mut a = State::with_args(&args(&[("scene", "knots")]), true);
    assert!(a.apply_arg("material", "porcelain") && a.apply_arg("material", "diamond") && a.apply_arg("material", "gold"));
    let mut b = State::with_args(&args(&[("scene", "knots")]), true);
    assert!(b.apply_arg("material", "gold"));
    assert_eq!(a.p, b.p);
}

#[test]
fn everything_a_material_can_set_is_put_back() {
    let reset: HashSet<&str> = RESET.iter().chain(gf_core::inner::RESET).map(|r| r.0).collect();
    for n in NAMED {
        for (k, _) in n.sets {
            assert!(reset.contains(k) || matches!(*k, "ior" | "disp" | "nb"), "{}: {k} would be left behind by the next material", n.id);
        }
    }
    // and what they are put back to is the setting's own default
    let d = Settings::default();
    for (k, v) in RESET {
        assert_eq!(d.get(k), Some(*v), "{k}");
    }
    // every material control but the tint (which is the family's) is among them; and but how finely a
    // milky body's light is worked out, which is a matter of the picture's quality, not of the material
    for c in CONTROLS {
        // (nor whether the picture is exposed for a milky body: no material sets it, it is how the picture
        // is taken, and choosing another material leaves it as the user put it)
        assert!(reset.contains(c.name) || c.name.starts_with("tint") || c.name == "sssDetail" || c.name == "meter", "{} is never put back", c.name);
    }
}

#[test]
fn controls_are_sound() {
    let mut names = HashSet::new();
    let d = Settings::default();
    for c in CONTROLS {
        assert!(names.insert(c.name), "two controls for {}", c.name);
        let v = d.get(c.name).unwrap_or_else(|| panic!("no setting called {}", c.name));
        assert!(c.lo < c.hi && c.step > 0.0, "{}", c.name);
        // (a derived tint has no default of its own: it shows the family's)
        if !c.name.starts_with("tint") { assert!((c.lo..=c.hi).contains(&v), "{}: its default {v} is outside {}..{}", c.name, c.lo, c.hi); }
        assert!(!c.label.is_empty());
        if matches!(c.kind, Kind::Choice) { assert_eq!(choices(c.name).len() as f64, c.hi + 1.0, "{}", c.name); }
        if matches!(c.kind, Kind::Colour) {
            // the two settings after a colour are its green and blue
            let i = CONTROLS.iter().position(|x| x.name == c.name).unwrap();
            assert!(CONTROLS[i + 1].name.ends_with('G') && CONTROLS[i + 2].name.ends_with('B'), "{}", c.name);
        }
    }
    assert_eq!(choices("mat").len(), NAMED.len());
    // the costly marks are on exactly the switches that build a costly tracer
    let costly: Vec<&str> = CONTROLS.iter().filter(|c| c.costly).map(|c| c.name).collect();
    assert_eq!(costly, ["sssWalk", "biref", "grin"]);
    // every slider is a knob, so that it can be given a track
    let knobs = State::engine_knobs();
    let keys: HashSet<&str> = knobs.iter().map(|k| k.key.as_str()).collect();
    assert_eq!(keys.len(), knobs.len(), "two knobs share a key");
    for c in CONTROLS.iter().filter(|c| matches!(c.kind, Kind::Slider | Kind::Colour | Kind::Whole)) {
        let k = knobs.iter().find(|k| k.id == c.name).unwrap_or_else(|| panic!("{} is not a knob", c.name));
        assert_eq!((k.lo, k.hi, k.step), (c.lo, c.hi, c.step));
    }
}

#[test]
fn what_applies_follows_the_choices() {
    let mut p = Settings::default();
    assert!(applies("ior", &p) && applies("density", &p) && applies("tintR", &p));
    assert!(!applies("sssDepth", &p) && !applies("filmIor", &p) && !applies("coatIor", &p) && !applies("bodyR", &p) && !applies("glowCore", &p));
    p.metal = 4.0;
    assert!(!applies("ior", &p) && !applies("disp", &p) && !applies("sss", &p) && !applies("density", &p) && applies("bodyR", &p) && applies("rough", &p));
    p.metal = 0.0;
    p.sss = 0.5;
    assert!(applies("sssDepth", &p) && applies("sssWalk", &p) && applies("bodyR", &p));
    p.sss_walk = 1.0;
    assert!(!applies("grin", &p) && !applies("biref", &p));
    p.sss = 0.0;
    p.spectrum = 1.0;
    assert!(!applies("tintR", &p) && applies("density", &p));
    p.film = 300.0;
    assert!(applies("filmIor", &p) && !applies("filmFlow", &p));
    p.film_var = 0.3;
    assert!(applies("filmFlow", &p));
}

#[test]
fn the_tint_is_the_absorption_said_as_a_colour() {
    let mut p = Settings::default();
    p.absorb = [0.10, 0.05, 0.02];
    assert!((p.get("tintR").unwrap() - (-0.10f64).exp()).abs() < 1e-15);
    assert!(p.set("tintG", 0.5));
    assert!((p.absorb[1] - std::f64::consts::LN_2).abs() < 1e-12);
    assert_eq!(p.absorb[0], 0.10);
    // white is no absorption; black is as much as there is
    assert!(p.set("tintB", 1.0) && p.absorb[2] == 0.0);
    assert!(p.set("tintB", 0.0) && p.absorb[2] > 9.0 && p.absorb[2].is_finite());
    for n in Settings::DERIVED { assert!(p.get(n).is_some()); }
}

#[test]
fn variants_follow_the_settings() {
    let v = |pairs: &[(&str, f64)]| {
        let mut p = Settings::default();
        for (k, x) in pairs { assert!(p.set(k, *x)); }
        Variant::of(&p)
    };
    assert!(v(&[("rough", 0.3)]).rough);
    assert!(v(&[("film", 200.0)]).film);
    assert!(v(&[("coat", 1.0)]).coat);
    assert!(v(&[("glow", 1.0)]).glow);
    assert!(v(&[("spectrum", 3.0)]).spectrum);
    assert!(!v(&[("spectrum", 99.0)]).spectrum);
    let milky = v(&[("sss", 1.0)]);
    assert!(milky.sss && !milky.walk);
    assert!(v(&[("sss", 1.0), ("sssWalk", 1.0)]).walk);
    assert!(!v(&[("sssWalk", 1.0)]).walk, "wandering light needs a milky body");
    // a metal has no inside
    let m = v(&[("metal", 4.0), ("sss", 1.0), ("spectrum", 2.0), ("grin", 0.2), ("biref", 0.1), ("rough", 0.2)]);
    assert!(m.metal && m.rough && !m.sss && !m.spectrum && !m.grin && !m.biref);
    // the costly ways through the body exclude one another
    let x = v(&[("sss", 1.0), ("sssWalk", 1.0), ("grin", 0.2), ("biref", 0.1)]);
    assert!(x.walk && !x.grin && !x.biref);
    let x = v(&[("grin", 0.2), ("biref", 0.1)]);
    assert!(x.biref && !x.grin);
    // bubbles alone leave the tracer plain
    let b = v(&[("bubbles", 0.3)]);
    assert!(b.bubbles && b.plain());
    // every variant has its own name
    let all = Variant::samples();
    let keys: HashSet<String> = all.iter().map(|v| v.key()).collect();
    assert_eq!(keys.len(), all.len());
}

#[test]
fn lines_are_kept_by_their_conditions() {
    let src = "a\n//#if X\nb\n//#if Y\nc\n//#else\nd\n//#endif\n//#else\ne\n//#endif\n//#if !X\nf\n//#endif\ng\n";
    assert_eq!(preprocess(src, &[("X", true), ("Y", false)]), "a\nb\nd\ng\n");
    assert_eq!(preprocess(src, &[("X", true), ("Y", true)]), "a\nb\nc\ng\n");
    assert_eq!(preprocess(src, &[("X", false), ("Y", true)]), "a\ne\nf\ng\n");
    assert_eq!(preprocess(src, &[]), "a\ne\nf\ng\n");
    // a block for each group of wavelengths: one group, or three with no dispersion
    let src = "x\n//#each\nt[g] = f(GROUP);\n//#if X\nu[GROUP] = t[g];\n//#endif\n//#end\ny\n";
    assert_eq!(preprocess(src, &[("X", true)]), "x\nt0 = f(0);\nu[0] = t0;\ny\n");
    assert_eq!(preprocess(src, &[("MONO", true)]), "x\nt0 = f(0);\nt1 = f(1);\nt2 = f(2);\ny\n");
    // plain glass takes nothing from the material's source at all
    let parts = tracer_parts(&Variant::default(), "struct MU {}\n", false, false);
    assert!(parts.decl.is_empty() && parts.shade == "shade(ro, rd, t, xi, W)" && parts.min_layers == 0);
    let parts = tracer_parts(&Variant { rough: true, bubbles: true, ..Variant::default() }, "//#if ROUGH\nrough\n//#endif\n//#if FILM\nfilm\n//#endif\n{{WEIGHT}}\n", false, false);
    assert_eq!(parts.decl, "rough\n(wOf(i, xi, W) * thr)\n");
    assert_eq!(parts.min_layers, 2);
}

#[test]
fn fitting_scales_the_materials_distances() {
    let mut p = Settings::default();
    NAMED[named_index("jade").unwrap()].apply(&mut p);
    assert_eq!(p.fit, 1.0);
    assert_eq!(fit_scale(&p, None), 1.0);
    assert!((fit_scale(&p, Some(FIT_THICKNESS)) - 1.0).abs() < 1e-12);
    let thin = fit_scale(&p, Some(0.06));
    assert!((thin - 0.2).abs() < 1e-12);
    // a milky body is never made denser to fit a thin shape (light that gets through a thin piece is what
    // shows it to be translucent: material::sss_fit); on a shape twice as thick it scatters half as often
    let (a, b, c) = (uniforms(&p, 1.0), uniforms(&p, thin), uniforms(&p, 2.0));
    assert!((b[4 * 4] / a[4 * 4] - 1.0).abs() < 1e-6);
    assert!((c[4 * 4] / a[4 * 4] - 0.5).abs() < 1e-4);
    // a material that only changes the family's glass is not fitted: the glass stays as the family has it
    for id in ["frosted", "satin", "seeded", "gold", "mirror"] {
        let mut p = Settings::default();
        NAMED[named_index(id).unwrap()].apply(&mut p);
        assert_eq!(p.fit, 0.0, "{id}");
    }
}

#[test]
fn bubbles_follow_their_settings() {
    let mut p = Settings::default();
    p.bubbles = 0.2;
    let a = bubbles(&p).unwrap();
    assert!(a.cell > 2.0 * a.size);
    p.bubbles = 0.8;
    let b = bubbles(&p).unwrap();
    assert!(b.cell < a.cell, "more bubbles means smaller cells");
    p.bubble_shape = 1.0;
    let c = bubbles(&p).unwrap();
    assert!(c.cell > b.cell, "needles reach further, so the cells are larger");
    p.bubble_seed = 7.0;
    assert_ne!(bubbles(&p).unwrap(), c);
}

#[test]
fn the_state_takes_materials_by_name() {
    let mut s = State::with_args(&args(&[("scene", "attractors"), ("preset", "thomas")]), true);
    let (ior, absorb) = (s.p.ior, s.p.absorb);
    assert!(s.apply_arg("material", "diamond"));
    assert_eq!(s.p.mat as usize, named_index("diamond").unwrap());
    assert!((s.p.ior - 2.423).abs() < 1e-9 && s.p.nb == 6.0);
    // another family: a named material keeps its own glass
    s.set_scene(2);
    assert!((s.p.ior - 2.423).abs() < 1e-9);
    // back to the family's own glass: its index, dispersion and tint return
    assert!(s.apply_arg("material", "glass"));
    assert_eq!(s.p.mat, 0.0);
    let own = State::with_args(&args(&[("scene", "surfaces")]), true);
    assert_eq!((s.p.ior, s.p.disp, s.p.nb, s.p.absorb), (own.p.ior, own.p.disp, own.p.nb, own.p.absorb));
    assert!(Variant::of(&s.p).plain());
    s.set_scene(0);
    assert_eq!((s.p.ior, s.p.absorb), (ior, absorb));
    // a material that only changes the surface keeps the family's glass exactly
    assert!(s.apply_arg("material", "frosted"));
    assert_eq!((s.p.ior, s.p.absorb, s.p.density), (ior, absorb, 1.0));
    assert!(s.p.rough > 0.0);
    // by position as well, and nonsense is refused
    assert!(s.apply_arg("mat", "0") && s.p.rough == 0.0);
    assert!(!s.apply_arg("material", "unobtainium"));
    // on the command line a material comes first, and settings beside it are laid over it
    let s = State::with_args(&args(&[("rough", "0.4"), ("material", "gold"), ("scene", "knots")]), true);
    assert!(metal_of(&s.p).is_some_and(|m| m.id == "gold") && s.p.rough == 0.4);
    let s = State::with_args(&args(&[("tintR", "0.5")]), true);
    assert!((s.p.absorb[0] - std::f64::consts::LN_2).abs() < 1e-12);
    // a tube's thickness is known without looking; an equation's is not
    let s = State::with_args(&args(&[("scene", "attractors"), ("preset", "thomas"), ("thick", "2")]), true);
    let one = State::with_args(&args(&[("scene", "attractors"), ("preset", "thomas")]), true);
    assert!((s.thickness_hint().unwrap() / one.thickness_hint().unwrap() - 2.0).abs() < 1e-12);
    assert!(State::with_args(&args(&[("scene", "surfaces")]), true).thickness_hint().is_none());
}

#[test]
fn a_saved_setup_keeps_the_material() {
    let mut s = State::with_args(&args(&[("scene", "knots"), ("material", "pearl"), ("filmVar", "0.3")]), true);
    s.p.glow = 0.5;
    let text = s.setup_json();
    let mut t = State::new();
    t.apply_setup_json(&text).unwrap();
    assert_eq!(t.p, s.p);
    // a setup saved before there were materials opens as plain glass
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let mut old = v.clone();
    let settings = old["settings"].as_object_mut().unwrap();
    for (k, _) in RESET { settings.remove(*k); }
    settings.remove("mat");
    let mut t = State::new();
    t.apply_setup_json(&old.to_string()).unwrap();
    assert!(Variant::of(&t.p).plain() && t.p.mat == 0.0);
}
