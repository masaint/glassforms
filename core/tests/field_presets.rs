//! The two equation families, compared with the web version's: every name, number, note and light.
//! `data/fields.json` is written by `tools/dump_fields.mjs` from forms.js and the scene files.
//! (The shader text itself is compared by running it: see field_translation.rs.)
use gf_core::field;
use gf_core::presets::{self, Family, Kind, Setting, Shape};
use serde_json::Value;

fn dump() -> Value {
    serde_json::from_str(include_str!("data/fields.json")).unwrap()
}
fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}
fn nums(v: &Value) -> Vec<f64> {
    v.as_array().unwrap_or_else(|| panic!("not a list: {v}")).iter().map(num).collect()
}
fn text(v: &Value) -> &str {
    v.as_str().unwrap_or_else(|| panic!("not text: {v}"))
}

fn check_family(fam: &Family, j: &Value) -> usize {
    let fid = fam.id;
    assert_eq!(fam.id, text(&j["id"]));
    assert_eq!(fam.name, text(&j["name"]), "{fid}");
    assert_eq!(text(&j["kind"]), "field", "{fid}");
    assert_eq!(fam.kind, Kind::Field, "{fid}");

    // the settings the family overrides, in the same order
    let defaults = j["defaults"].as_array().unwrap();
    assert_eq!(fam.defaults.len(), defaults.len(), "{fid}: number of settings");
    for ((key, value), d) in fam.defaults.iter().zip(defaults) {
        assert_eq!(*key, text(&d[0]), "{fid}: settings out of order");
        match value {
            Setting::Num(x) => assert_eq!(*x, num(&d[1]), "{fid}: {key}"),
            Setting::Vec3(x) => assert_eq!(x.to_vec(), nums(&d[1]), "{fid}: {key}"),
        }
    }

    let lights = j["lights"].as_array().unwrap();
    assert_eq!(fam.lights.len(), lights.len(), "{fid}: number of lights");
    for (i, (l, d)) in fam.lights.iter().zip(lights).enumerate() {
        let soft = d.get("soft").map_or(0.6, num);
        assert_eq!((l.az, l.el, l.w, l.h, l.soft), (num(&d["az"]), num(&d["el"]), num(&d["w"]), num(&d["h"]), soft), "{fid}: light {i}");
        assert_eq!(l.rgb.to_vec(), nums(&d["rgb"]), "{fid}: light {i}");
    }
    assert_eq!(fam.ambient.to_vec(), nums(&j["ambient"]), "{fid}: ambient");

    // the marching settings: where the family gives one it is that, and all four are what the tracer's
    // shader ends up being told
    for (ours, key) in [(fam.step_scale, "stepScale"), (fam.steps_out as f64, "stepsOut"), (fam.steps_in as f64, "stepsIn"), (fam.layers as f64, "layers")] {
        if !j[key].is_null() { assert_eq!(ours, num(&j[key]), "{fid}: {key}"); }
    }
    assert_eq!((fam.step_scale, fam.layers), (num(&j["stepScale"]), num(&j["layers"]) as u32), "{fid}");
    let t = &j["tracer"];
    assert_eq!(format!("{:.2}", fam.step_scale), text(&t["STEP_SCALE"]), "{fid}");
    assert_eq!(fam.steps_out.to_string(), text(&t["STEPS_OUT"]), "{fid}");
    assert_eq!(fam.steps_in.to_string(), text(&t["STEPS_IN"]), "{fid}");
    assert_eq!(fam.layers.to_string(), text(&t["LAYERS"]), "{fid}");

    let presets = j["presets"].as_array().unwrap();
    let ids: Vec<&str> = fam.presets.iter().map(|p| p.id).collect();
    let want: Vec<&str> = presets.iter().map(|p| text(&p["id"])).collect();
    assert_eq!(ids, want, "{fid}: the presets and their order");
    for (p, d) in fam.presets.iter().zip(presets) {
        let id = p.id;
        assert_eq!(p.name, text(&d["name"]), "{id}");
        let eq: Vec<&str> = d["eq"].as_array().unwrap().iter().map(text).collect();
        assert_eq!(p.eq, eq, "{id}: eq");
        assert_eq!(p.note, d["note"].as_str().unwrap_or(""), "{id}: note");

        // sliders: [value, min, max, step, label?], and the label the page ends up showing
        let params = d["params"].as_array().unwrap();
        assert_eq!(p.params.len(), params.len(), "{id}: number of sliders");
        for (a, b) in p.params.iter().zip(params) {
            let e = b["entry"].as_array().unwrap();
            assert_eq!(a.id, text(&b["id"]), "{id}: sliders out of order");
            assert_eq!((a.value, a.lo, a.hi, a.step), (num(&e[0]), num(&e[1]), num(&e[2]), num(&e[3])), "{id}: {}", a.id);
            assert_eq!(a.label, e.get(4).map_or(a.id, text), "{id}: label of {}", a.id);
            let knob = d["knobs"].as_array().unwrap().iter().find(|k| text(&k["id"]) == a.id).unwrap();
            assert_eq!(a.label, text(&knob["label"]), "{id}: label of {} on the page", a.id);
            assert_eq!((a.value, a.lo, a.hi, a.step), (num(&knob["value"]), num(&knob["lo"]), num(&knob["hi"]), num(&knob["step"])), "{id}: {} on the page", a.id);
        }

        assert_eq!(p.rot.map(|r| r.to_vec()), d["rot"].as_array().map(|_| nums(&d["rot"])), "{id}: rot");
        let animate: Option<Vec<&str>> = d["animate"].as_array().map(|a| a.iter().map(text).collect());
        assert_eq!(p.animate, animate, "{id}: animate");
        // every slider named there exists: the page found each one
        assert_eq!(d["animate"], d["animated"], "{id}: a slider to animate that the page did not find");

        assert!(matches!(p.shape, Shape::Field { .. }), "{id}: not an equation");
        // nothing a curve would use has crept in
        assert!(p.tube.is_none() && p.axes.is_none() && p.align.is_none() && p.fit.is_none() && p.radius.is_none() && p.info.is_none() && p.centre && !p.fixed_length, "{id}");

        // what the bake is handed for the starting values: the same numbers, in the same places
        let b = &d["bake"];
        let res = num(&b["res"]) as u32;
        let handed: Vec<u32> = b["uniformBits"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let ours = field::uniforms(p, &p.defaults(), res, 0.0);
        assert_eq!(ours.map(f32::to_bits).to_vec(), handed, "{id}: uniforms {ours:?}");
    }
    presets.len()
}

#[test]
fn families_match_the_web_version() {
    let j = dump();
    let fams = [presets::surfaces::family(), presets::algebraic::family()];
    let dumped = j["families"].as_array().unwrap();
    assert_eq!(dumped.len(), fams.len());
    let mut n = 0;
    for (fam, d) in fams.iter().zip(dumped) {
        n += check_family(fam, d);
    }
    assert_eq!(n, 18);
    println!("{n} presets in {} families match fields.json", fams.len());
}
