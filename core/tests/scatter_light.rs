//! The lighting as the light volume takes it (beams and a rest), and a body as a scattering medium.

use gf_core::env;
use gf_core::presets::Light;
use gf_core::scatter::{beams, beams_at_most, beams_total, detail, diffuse_reflectance, Medium, MAX_BEAMS};
use gf_core::settings::Settings;

const PI: f64 = std::f64::consts::PI;

fn map(lights: &[Light], ambient: [f64; 3]) -> Vec<f32> {
    env::paint(lights, ambient, env::ENV_W, env::ENV_H)
}

/// All the light of a map: every texel's brightness times the patch of sky it covers.
fn light_of(px: &[f32]) -> [f64; 3] {
    let (w, h) = (env::ENV_W, env::ENV_H);
    let mut t = [0.0; 3];
    for j in 0..h {
        let sky = (2.0 * PI / w as f64) * ((j as f64 / h as f64 * PI).cos() - ((j + 1) as f64 / h as f64 * PI).cos());
        for i in 0..w {
            for c in 0..3 { t[c] += px[(j * w + i) * 4 + c] as f64 * sky; }
        }
    }
    t
}

fn lamp(az: f64, el: f64, size: f64, rgb: [f64; 3]) -> Light {
    Light { az, el, w: size, h: size, rgb, soft: 0.6 }
}

#[test]
fn no_light_no_beams() {
    let l = beams(&map(&[], [0.0; 3]), env::ENV_W, env::ENV_H, None);
    assert!(l.beams.is_empty());
    assert_eq!(l.rest.even, [0.0; 3]);
}

#[test]
fn the_beams_and_the_rest_hold_the_light_of_the_map() {
    let rigs: Vec<(Vec<Light>, [f64; 3])> = vec![
        (vec![lamp(140.0, 35.0, 22.0, [7.0, 7.0, 7.0])], [0.02; 3]),
        (vec![lamp(38.0, 18.0, 14.0, [5.0, 4.9, 4.7]), lamp(-55.0, -8.0, 30.0, [0.5, 1.4, 4.0]), lamp(0.0, 85.0, 34.0, [0.8, 0.9, 1.1])], [0.006, 0.011, 0.022]),
        (vec![], [1.0, 0.5, 0.25]),
    ];
    for (lights, ambient) in rigs {
        let px = map(&lights, ambient);
        let (want, l) = (light_of(&px), beams(&px, env::ENV_W, env::ENV_H, None));
        assert!(l.beams.len() <= MAX_BEAMS);
        let got = beams_total(&l);
        for c in 0..3 {
            // (a beam carries a little less than its patch's light: by the patch's mean cosine, never
            // less than that of a patch as wide as they come)
            assert!(got[c] <= want[c] * 1.0001 && got[c] >= want[c] * 0.93, "colour {c}: beams and rest hold {} of {}", got[c], want[c]);
        }
    }
}

#[test]
fn a_small_lamp_is_a_beam_from_the_lamp() {
    for (az, el) in [(140.0, 35.0), (10.0, -20.0), (250.0, 70.0)] {
        let l = beams(&map(&[lamp(az, el, 4.0, [50.0; 3])], [0.0; 3]), env::ENV_W, env::ENV_H, None);
        assert!(!l.beams.is_empty() && l.beams.len() <= 4, "{} beams for one small lamp", l.beams.len());
        // the lamp's direction, as the map has it
        let (a, e) = (az.to_radians(), el.to_radians());
        let d = [a.sin() * e.cos(), e.sin(), -a.cos() * e.cos()];
        // the beams' light, weighed by direction
        let mut m = [0.0; 3];
        for b in &l.beams {
            for i in 0..3 { m[i] += b.dir[i] * b.rgb[1]; }
        }
        let len = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
        let cos = (m[0] * d[0] + m[1] * d[1] + m[2] * d[2]) / len;
        assert!(cos > 0.999, "lamp at {az}, {el}: the beams point {:.2} degrees off", cos.clamp(-1.0, 1.0).acos().to_degrees());
        // and they are narrow: a small lamp casts a sharp shadow
        assert!(l.beams.iter().all(|b| b.tan < 0.12), "a small lamp's beam is {:?} wide", l.beams.iter().map(|b| b.tan).collect::<Vec<_>>());
    }
}

#[test]
fn the_beams_light_a_surface_as_the_map_does() {
    // What arrives on a surface facing any way, from the beams and the rest, is what arrives from the map
    // itself (every texel by how squarely it faces the surface); and stays so while a lamp moves across
    // the patches of the sky, so that nothing jumps where it passes from one patch to the next.
    let (w, h) = (env::ENV_W, env::ENV_H);
    let mut worst: f64 = 0.0;
    for n in [[0.0, 1.0, 0.0], [0.6, 0.0, -0.8], [-0.7, 0.7, 0.1], [0.0, -0.6, 0.8]] {
        let mut az = 100.0;
        while az < 190.0 {
            let px = map(&[lamp(az, 20.0, 12.0, [5.0; 3])], [0.01; 3]);
            let mut truth = 0.0;
            for j in 0..h {
                let sky = (2.0 * PI / w as f64) * ((j as f64 / h as f64 * PI).cos() - ((j + 1) as f64 / h as f64 * PI).cos());
                for i in 0..w {
                    let d = env::paint_dir((i as f64 + 0.5) / w as f64, (j as f64 + 0.5) / h as f64);
                    truth += px[(j * w + i) * 4 + 1] as f64 * sky * (d[0] * n[0] + d[1] * n[1] + d[2] * n[2]).max(0.0);
                }
            }
            let l = beams(&px, w, h, None);
            // (a patch too faint for a beam of its own is in the rest: light from all round, with a lean)
            let rest = PI * l.rest.even[1] + 2.0 * PI / 3.0 * (l.rest.lean[1][0] * n[0] + l.rest.lean[1][1] * n[1] + l.rest.lean[1][2] * n[2]);
            let got = rest.max(0.0) + l.beams.iter().map(|b| b.rgb[1] * (b.dir[0] * n[0] + b.dir[1] * n[1] + b.dir[2] * n[2]).max(0.0)).sum::<f64>();
            let off = (got - truth).abs() / truth.max(0.3);
            worst = worst.max(off);
            assert!(off < 0.08, "facing {n:?}, lamp at {az}: {got} for {truth}");
            az += 1.5;
        }
    }
    assert!(worst > 0.0);
}

#[test]
fn fewer_beams_keep_the_light() {
    let px = map(&[lamp(38.0, 18.0, 30.0, [5.0; 3]), lamp(200.0, -30.0, 40.0, [2.0; 3])], [0.05; 3]);
    let all = beams_total(&beams(&px, env::ENV_W, env::ENV_H, None));
    for most in [0, 4, 12] {
        let l = beams_at_most(&px, env::ENV_W, env::ENV_H, None, most);
        assert!(l.beams.len() <= most);
        let t = beams_total(&l);
        assert!((t[1] - all[1]).abs() < 0.05 * all[1], "{most} beams hold {} of {}", t[1], all[1]);
    }
}

#[test]
fn a_surface_lets_out_what_it_lets_in() {
    // glass in air reflects 9.2% of light from all round; and a body of air's own index holds nothing in
    assert!((diffuse_reflectance(1.0, 1.5) - 0.092).abs() < 0.003, "{}", diffuse_reflectance(1.0, 1.5));
    let mut p = Settings { sss: 1.0, ..Settings::default() };
    for ior in [1.0, 1.3, 1.5, 1.61, 2.4] {
        p.ior = ior;
        let m = Medium::of(&p, 1.0);
        // from inside, the surface lets out 1 / ior^2 of what it lets in from outside: the two shares
        // together make a body that absorbs nothing give back exactly what arrives
        let back = (m.hold - 1.0) / (m.hold + 1.0);
        assert!(((1.0 - back) * ior * ior - m.enters).abs() < 1e-9, "index {ior}");
        assert!(m.hold >= 1.0 && m.enters <= 1.0);
        if ior == 1.0 { assert!((m.hold - 1.0).abs() < 1e-9 && (m.enters - 1.0).abs() < 1e-9); }
    }
}

#[test]
fn the_medium_follows_the_sliders() {
    let p = Settings { sss: 1.0, sss_depth: 0.05, body_r: 0.3, body_g: 0.6, body_b: 0.9, ..Settings::default() };
    let m = Medium::of(&p, 1.0);
    assert!((m.sig_s[0] - 20.0).abs() < 1e-9 && m.sig_s[0] == m.sig_s[2]);
    // the darker a colour of the body, the more of it is absorbed
    assert!(m.sig_a[0] > m.sig_a[1] && m.sig_a[1] > m.sig_a[2] && m.sig_a[2] > 0.0);
    // fitted to a shape twice as thick, light travels twice as far
    let m2 = Medium::of(&p, 2.0);
    assert!((m2.sig_s[0] - 10.0).abs() < 1e-9 && (m2.sig_a[0] - 0.5 * m.sig_a[0]).abs() < 1e-9);
}

#[test]
fn detail_is_a_sane_size() {
    assert_eq!(detail(160.0), 160);
    assert_eq!(detail(10.0), 64);
    assert_eq!(detail(1e9), 256);
    assert_eq!(detail(f64::NAN), 160);
    assert_eq!(detail(150.0) % 16, 0);
}
