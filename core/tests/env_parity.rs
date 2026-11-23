//! The lights of every family, painted here and by the web version, compared value by value.
//! `data/env_<family>.f32` holds the floats the web version's GLASS.envTexture handed to WebGL, and
//! `data/env.json` how it set the texture up; both are written by `tools/dump_env.mjs`.
use std::path::Path;

use gf_core::env::{self, ENV_H, ENV_W};
use gf_core::presets::{self, Light};
use serde_json::Value;

fn data(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("data").join(name)
}
fn dump() -> Value {
    serde_json::from_str(&std::fs::read_to_string(data("env.json")).unwrap()).unwrap()
}
fn floats(name: &str) -> Vec<f32> {
    let bytes = std::fs::read(data(name)).unwrap();
    bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
}
fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("not a number: {v}"))
}
fn three(v: &Value) -> [f64; 3] {
    [num(&v[0]), num(&v[1]), num(&v[2])]
}
/// A family's lights as the web version passed them (a light that gives no softness gets 0.6 there).
fn lights(family: &Value) -> Vec<Light> {
    family["lights"].as_array().unwrap().iter().map(|l| Light {
        az: num(&l["az"]),
        el: num(&l["el"]),
        w: num(&l["w"]),
        h: num(&l["h"]),
        rgb: three(&l["rgb"]),
        soft: l.get("soft").map_or(0.6, num),
    }).collect()
}

/// How far two pictures are apart: values that are not bit for bit the same, the largest difference,
/// and the largest difference as a fraction of the value.
fn apart(ours: &[f32], theirs: &[f32]) -> (usize, f64, f64) {
    assert_eq!(ours.len(), theirs.len());
    let (mut differ, mut worst, mut worst_rel) = (0, 0.0f64, 0.0f64);
    for (a, b) in ours.iter().zip(theirs) {
        if a.to_bits() == b.to_bits() { continue; }
        differ += 1;
        let d = (*a as f64 - *b as f64).abs();
        assert!(d.is_finite(), "{a} against {b}");
        worst = worst.max(d);
        worst_rel = worst_rel.max(d / (*b as f64).abs().max(1e-30));
    }
    (differ, worst, worst_rel)
}

#[test]
fn every_family_paints_the_same_picture() {
    let j = dump();
    let families = j["families"].as_array().unwrap();
    let ids: Vec<&str> = families.iter().map(|f| f["id"].as_str().unwrap()).collect();
    assert_eq!(ids, ["algebraic", "attractors", "knots", "surfaces"]);
    for f in families {
        let id = f["id"].as_str().unwrap();
        let theirs = floats(f["file"].as_str().unwrap());
        assert_eq!(theirs.len(), ENV_W * ENV_H * 4, "{id}");
        let ours = env::paint(&lights(f), three(&f["ambient"]), ENV_W, ENV_H);
        let (differ, worst, worst_rel) = apart(&ours, &theirs);
        let lit = theirs.as_chunks::<4>().0.iter().filter(|p| p[0] > 0.1).count();
        println!("{id:10} {} values ({lit} texels lit): {differ} not bit-identical, worst difference {worst:e}, relative {worst_rel:e}", ours.len());
        assert!(worst_rel <= 1e-6, "{id}: off by {worst_rel:e} of the value");
        assert!(lit > 500, "{id}: hardly anything is lit");
    }
}

#[test]
fn the_equation_families_have_the_web_versions_lights() {
    let j = dump();
    for fam in [presets::surfaces::family(), presets::algebraic::family()] {
        let f = j["families"].as_array().unwrap().iter().find(|f| f["id"] == fam.id).unwrap();
        assert_eq!(fam.lights, lights(f), "{}", fam.id);
        assert_eq!(fam.ambient, three(&f["ambient"]), "{}", fam.id);
        let (_, _, worst_rel) = apart(&env::paint(&fam.lights, fam.ambient, ENV_W, ENV_H), &floats(f["file"].as_str().unwrap()));
        assert!(worst_rel <= 1e-6, "{}: off by {worst_rel:e} of the value", fam.id);
    }
}

/// What env.rs says about the texture, held against what the web version really asked WebGL for.
#[test]
fn the_texture_is_set_up_as_described() {
    let j = dump();
    let t = &j["texture"];
    assert_eq!((t["width"].as_u64(), t["height"].as_u64()), (Some(ENV_W as u64), Some(ENV_H as u64)));
    assert_eq!((ENV_W, ENV_H), (256, 128));
    assert_eq!(t["target"], "TEXTURE_2D");
    assert_eq!(t["internalFormat"], "RGBA16F");
    assert_eq!((&t["format"], &t["type"]), (&Value::from("RGBA"), &Value::from("FLOAT")));
    assert_eq!(t["level"], 0);
    assert_eq!(t["mipmaps"], false);
    let p = &t["parameters"];
    assert_eq!(p["TEXTURE_MIN_FILTER"], "LINEAR");
    assert_eq!(p["TEXTURE_MAG_FILTER"], "LINEAR");
    assert_eq!(p["TEXTURE_WRAP_S"], "REPEAT");
    assert_eq!(p["TEXTURE_WRAP_T"], "CLAMP_TO_EDGE");
    assert_eq!(p.as_object().unwrap().len(), 4);
}

/// The picture is laid out as described: row 0 is the top of the sky, and a light sits where the
/// tracer's lookup will find it.
#[test]
fn a_light_lands_where_the_lookup_finds_it() {
    use std::f64::consts::PI;
    // the texel the tracer reads for a direction (envMap in shaders.js, with its own constants), nearest texel
    #[allow(clippy::approx_constant)]
    let texel = |d: [f64; 3]| {
        let u = d[2].atan2(d[0]) * 0.15915494 + 0.5;
        let v = d[1].clamp(-1.0, 1.0).acos() * 0.31830989;
        (((u * ENV_W as f64) as usize).min(ENV_W - 1), ((v * ENV_H as f64) as usize).min(ENV_H - 1))
    };
    let dir = |az: f64, el: f64| {
        let (az, el) = (az * PI / 180.0, el * PI / 180.0);
        [az.sin() * el.cos(), el.sin(), -az.cos() * el.cos()]
    };
    let one = |az: f64, el: f64| vec![Light { az, el, w: 6.0, h: 6.0, rgb: [1.0, 2.0, 3.0], soft: 0.5 }];
    // behind the object, to the right, at the camera, to the left, high up, low down
    for (az, el) in [(0.0, 0.0), (90.0, 0.0), (180.0, 0.0), (-90.0, 0.0), (40.0, 60.0), (-130.0, -50.0)] {
        let px = env::paint(&one(az, el), [0.0; 3], ENV_W, ENV_H);
        let at = |d: [f64; 3]| {
            let (i, j) = texel(d);
            let o = (j * ENV_W + i) * 4;
            [px[o], px[o + 1], px[o + 2], px[o + 3]]
        };
        assert_eq!(at(dir(az, el)), [1.0, 2.0, 3.0, 1.0], "the light at az {az}, el {el}");
        assert_eq!(at(dir(az + 180.0, -el)), [0.0, 0.0, 0.0, 1.0], "opposite the light at az {az}, el {el}");
    }
    // straight up is row 0
    let px = env::paint(&one(0.0, 90.0), [0.0; 3], ENV_W, ENV_H);
    assert_eq!(px[0], 1.0);
    assert_eq!(px[(ENV_H - 1) * ENV_W * 4], 0.0);
    // ambient is added everywhere
    let px = env::paint(&[], [0.25, 0.5, 0.75], 8, 4);
    assert!(px.as_chunks::<4>().0.iter().all(|p| *p == [0.25, 0.5, 0.75, 1.0]));
}

/// The value of a 16-bit float, exactly.
fn half_value(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let (e, m) = (((h >> 10) & 0x1f) as i32, (h & 0x3ff) as f32);
    match e {
        0 => sign * m * 2f32.powi(-24),
        31 => if m == 0.0 { sign * f32::INFINITY } else { f32::NAN },
        _ => sign * (1.0 + m / 1024.0) * 2f32.powi(e - 15),
    }
}

#[test]
fn rounding_to_half_floats() {
    // every half float is its own nearest half float
    for h in 0..=u16::MAX {
        let x = half_value(h);
        if x.is_nan() {
            assert!(half_value(env::half_bits(x)).is_nan());
        } else {
            assert_eq!(env::half_bits(x), h, "{x}");
        }
    }
    // between two neighbours: below the middle goes down, above goes up, the middle itself to the even one
    for h in 0..0x7c00u16 {
        let (lo, hi) = (half_value(h), half_value(h + 1));
        let mid = if hi.is_infinite() { 65520.0 } else { (lo + hi) / 2.0 };
        let even = if h & 1 == 0 { h } else { h + 1 };
        assert_eq!(env::half_bits(mid), even, "the middle of {lo} and {hi}");
        assert_eq!(env::half_bits(mid.next_down()), h, "just below the middle of {lo} and {hi}");
        assert_eq!(env::half_bits(mid.next_up()), h + 1, "just above the middle of {lo} and {hi}");
        assert_eq!(env::half_bits(-mid), even | 0x8000);
        assert_eq!(env::half_bits(-mid.next_up()), (h + 1) | 0x8000);
    }
    assert_eq!(env::half_bits(1.0), 0x3c00);
    assert_eq!(env::half_bits(-2.0), 0xc000);
    assert_eq!(env::half_bits(1e9), 0x7c00);
    assert_eq!(env::half_bits(1e-12), 0);
    assert_eq!(env::half_bits(f32::MIN_POSITIVE / 2.0), 0);
    assert_eq!(env::half_bits(-0.0), 0x8000);

    // and the web version's pictures, rounded here, are the ones a browser's own half floats give
    // (a fingerprint of the rounded picture, left by dump_env.mjs where node has half floats)
    let j = dump();
    for f in j["families"].as_array().unwrap() {
        let id = f["id"].as_str().unwrap();
        let Some(want) = f["half"].as_u64() else {
            println!("{id}: no half-float fingerprint in env.json, not compared");
            continue;
        };
        let mut hash = 0x811c9dc5u32;
        for x in floats(f["file"].as_str().unwrap()) {
            for byte in env::half_bits(x).to_le_bytes() {
                hash = (hash ^ byte as u32).wrapping_mul(0x01000193);
            }
        }
        assert_eq!(hash as u64, want, "{id}: the picture rounds to different half floats");
        let fam_lights = lights(f);
        assert_eq!(env::paint_half(&fam_lights, three(&f["ambient"]), 16, 8).len(), 16 * 8 * 4);
    }
}

#[test]
fn extreme_light_sizes_stay_finite_and_within_one_hemisphere() {
    let weird = [
        Light { az: 0.0, el: 0.0, w: 0.0, h: 0.0, rgb: [4.0, 3.0, 2.0], soft: 0.5 },
        Light { az: 0.0, el: 0.0, w: -10.0, h: f64::NAN, rgb: [2.0, 2.0, 2.0], soft: 0.0 },
        Light { az: 0.0, el: 0.0, w: 90.0, h: 135.0, rgb: [3.0, 1.0, 0.5], soft: 0.6 },
    ];
    let mut px = env::paint(&weird, [0.0; 3], 64, 32);
    assert!(px.iter().all(|v| v.is_finite()), "env::paint emitted non-finite values");
    // Opposite direction (az = 180°, i.e. ph = +pi/2 at i = 48, j = 16 -> d = [0, 0, 1], da = -1) must stay unlit.
    let back = (16 * 64 + 48) * 4;
    assert_eq!(&px[back..back + 4], &[0.0, 0.0, 0.0, 1.0]);

    let (tint, lvl) = gf_core::room::level(&weird);
    assert!(tint.iter().all(|v| v.is_finite()) && lvl.is_finite());
    gf_core::room::paint(&mut px, 64, 32, &weird, &gf_core::math::I3, 1.0);
    assert!(px.iter().all(|v| v.is_finite()), "room::paint emitted non-finite values");
}
