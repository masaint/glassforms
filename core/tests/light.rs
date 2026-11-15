//! The parts of the lighting that are worked out without a GPU: lights made brighter together, the glows
//! an environment picture's brightest light comes back as, the lighting as nine smooth terms for clay,
//! and how the small picture of the lighting is laid out.

use std::f64::consts::PI;

use gf_core::env::{self, Glow, ENV_H, ENV_W};
use gf_core::presets::{self, Light};

/// The patch of sky a texel of row j of a `w` x `h` map covers.
fn patch(j: usize, w: usize, h: usize) -> f64 {
    (2.0 * PI / w as f64) * ((j as f64 / h as f64 * PI).cos() - ((j + 1) as f64 / h as f64 * PI).cos())
}

#[test]
fn a_gain_of_one_paints_what_was_always_painted() {
    for f in presets::families() {
        let plain = env::paint(&f.lights, f.ambient, ENV_W, ENV_H);
        let one = env::paint_gain(&f.lights, f.ambient, 1.0, ENV_W, ENV_H);
        assert!(plain.iter().zip(&one).all(|(a, b)| a.to_bits() == b.to_bits()), "{}", f.id);
        // twice as bright is twice the numbers (to the rounding of a float), nothing is nothing
        let two = env::paint_gain(&f.lights, f.ambient, 2.0, ENV_W, ENV_H);
        for (a, b) in plain.chunks(4).zip(two.chunks(4)) {
            for c in 0..3 {
                assert!((b[c] as f64 - 2.0 * a[c] as f64).abs() <= 1e-6 * a[c].abs() as f64 + 1e-12, "{} against twice {}", b[c], a[c]);
            }
        }
        assert!(env::paint_gain(&f.lights, f.ambient, 0.0, ENV_W, ENV_H).chunks(4).all(|p| p[0] == 0.0 && p[1] == 0.0 && p[2] == 0.0));
    }
}

#[test]
fn a_glow_holds_its_light_whatever_its_width() {
    let dir = {
        let d = [0.3f64, 0.8, -0.5];
        let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        [d[0] / l, d[1] / l, d[2] / l]
    };
    for (sigma, extra) in [(0.05, 0.0), (0.1, 0.0), (0.1, 0.3), (0.5, 0.0), (0.05, 1.0)] {
        let mut px = vec![0.0f32; ENV_W * ENV_H * 4];
        env::add_glows(&mut px, &[Glow { dir, energy: [2.0, 1.0, 0.5], sigma }], 1.5, extra, ENV_W, ENV_H);
        let mut sum = [0.0f64; 3];
        let (mut best, mut at) = (0.0f32, (0, 0));
        for j in 0..ENV_H {
            for i in 0..ENV_W {
                let o = (j * ENV_W + i) * 4;
                for c in 0..3 { sum[c] += px[o + c] as f64 * patch(j, ENV_W, ENV_H); }
                if px[o] > best { best = px[o]; at = (i, j); }
            }
        }
        // all of its light is there (times the gain), in its colours
        for (c, want) in [3.0, 1.5, 0.75].iter().enumerate() {
            assert!((sum[c] / want - 1.0).abs() < 0.02, "sigma {sigma} + {extra}: {} of {want}", sum[c]);
        }
        // brightest where it points: the texel the tracer's lookup finds for that direction
        let u = dir[2].atan2(dir[0]) / (2.0 * PI) + 0.5;
        let v = dir[1].acos() / PI;
        assert!((at.0 as f64 - u * ENV_W as f64).abs() < 2.0 && (at.1 as f64 - v * ENV_H as f64).abs() < 2.0, "sigma {sigma}: brightest at {at:?}, the glow is at ({}, {})", u * ENV_W as f64, v * ENV_H as f64);
        // and as bright in the middle as a bell of that width holding that light is
        let s2 = sigma * sigma + extra * extra;
        let peak = 3.0 / (2.0 * PI * s2 * (1.0 - (-2.0 / s2).exp()));
        assert!((best as f64 / peak - 1.0).abs() < 0.05, "sigma {sigma} + {extra}: middle {best}, expected {peak}");
    }
    // no brightness, no glow
    let mut px = vec![0.0f32; ENV_W * ENV_H * 4];
    env::add_glows(&mut px, &[Glow { dir, energy: [2.0, 1.0, 0.5], sigma: 0.1 }], 0.0, 0.0, ENV_W, ENV_H);
    assert!(px.iter().all(|v| *v == 0.0));
}

#[test]
fn nine_terms_hold_the_light_a_matte_surface_gathers() {
    // the same light from everywhere: a matte surface gathers exactly that, whichever way it faces
    let (w, h) = (64, 32);
    let px = env::paint(&[], [0.25, 0.5, 0.75], w, h);
    let t = env::sh_terms(&env::sh_of_map(&px, w, h, 4, env::paint_dir), env::SH_MATTE);
    for d in [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.6, -0.48, 0.64]] {
        let e = env::sh_eval(&t, d);
        assert!((e[0] - 0.25).abs() < 1e-3 && (e[1] - 0.5).abs() < 1e-3 && (e[2] - 0.75).abs() < 1e-3, "{e:?}");
    }
    // a family's lights: the nine terms against the sum they stand for, for surfaces facing many ways
    for f in presets::families() {
        let px = env::paint(&f.lights, f.ambient, ENV_W, ENV_H);
        let t = env::sh_terms(&env::sh_of_map(&px, ENV_W, ENV_H, 4, env::paint_dir), env::SH_MATTE);
        let mut worst = 0.0f64;
        let mut level = 0.0;
        for k in 0..40 {
            // directions spread over the sphere
            let y = 1.0 - (k as f64 + 0.5) / 20.0;
            let a = k as f64 * 2.399963;
            let n = [(1.0 - y * y).sqrt() * a.cos(), y, (1.0 - y * y).sqrt() * a.sin()];
            let mut sum = 0.0;
            for j in 0..ENV_H {
                for i in 0..ENV_W {
                    let d = env::paint_dir((i as f64 + 0.5) / ENV_W as f64, (j as f64 + 0.5) / ENV_H as f64);
                    let c = d[0] * n[0] + d[1] * n[1] + d[2] * n[2];
                    if c > 0.0 { sum += px[(j * ENV_W + i) * 4 + 1] as f64 * c * patch(j, ENV_W, ENV_H); }
                }
            }
            sum /= PI;
            worst = worst.max((env::sh_eval(&t, n)[1] - sum).abs());
            level += sum / 40.0;
        }
        // (nine terms follow one small light to within a tenth of the most it gives; with several lights
        // round the shape those misses can add up to a fifth of the average, which is smooth and of no
        // account in a picture of clay)
        assert!(worst < 0.2 * level, "{}: off by {worst} where the average is {level}", f.id);
    }
    // the terms of a glow are those of its picture
    let g = Glow { dir: [0.0, 0.6, -0.8], energy: [1.0, 2.0, 3.0], sigma: 0.2 };
    let mut px = vec![0.0f32; ENV_W * ENV_H * 4];
    env::add_glows(&mut px, std::slice::from_ref(&g), 1.0, 0.0, ENV_W, ENV_H);
    let (a, b) = (env::sh_of_map(&px, ENV_W, ENV_H, 4, env::paint_dir), env::sh_of_glow(&g));
    for k in 0..9 {
        for c in 0..3 {
            assert!((a[k][c] - b[k][c]).abs() < 0.02 * (c + 1) as f64, "term {k}, colour {c}: {} against {}", a[k][c], b[k][c]);
        }
    }
    // adding lightings adds their terms
    let mut sum = a;
    env::sh_add(&mut sum, &b, 2.0);
    assert!((sum[0][1] - (a[0][1] + 2.0 * b[0][1])).abs() < 1e-12);
}

#[test]
fn the_picture_of_the_lighting_is_laid_out_as_described() {
    // its middle is straight ahead of the camera, right of the middle is to the right, the top is up
    let d = env::preview_dir(0.5, 0.5);
    assert!(d[0].abs() < 1e-12 && d[1].abs() < 1e-12 && (d[2] + 1.0).abs() < 1e-12);
    assert!((env::preview_dir(0.75, 0.5)[0] - 1.0).abs() < 1e-12);
    assert!((env::preview_dir(0.3, 0.0)[1] - 1.0).abs() < 1e-12);
    // its edges are behind the camera
    assert!((env::preview_dir(0.0, 0.5)[2] - 1.0).abs() < 1e-12 && (env::preview_dir(1.0, 0.5)[2] - 1.0).abs() < 1e-12);
    for (u, v) in [(0.1, 0.2), (0.5, 0.5), (0.9, 0.7), (0.33, 0.95)] {
        let (a, b) = env::preview_uv(env::preview_dir(u, v));
        assert!((a - u).abs() < 1e-9 && (b - v).abs() < 1e-9);
    }
    // a light painted into the lights' map is, in the lights' own frame, where its azimuth and elevation say
    let l = Light { az: 30.0, el: 40.0, w: 4.0, h: 4.0, rgb: [1.0, 1.0, 1.0], soft: 0.5 };
    let px = env::paint(std::slice::from_ref(&l), [0.0; 3], ENV_W, ENV_H);
    let d = gf_core::state::State::light_dir(l.az, l.el);
    let u = d[2].atan2(d[0]) / (2.0 * PI) + 0.5;
    let v = d[1].acos() / PI;
    assert_eq!(px[(((v * ENV_H as f64) as usize) * ENV_W + (u * ENV_W as f64) as usize) * 4], 1.0);
}
