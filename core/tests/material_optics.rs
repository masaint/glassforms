//! The optics behind the materials, checked against values that are known: thin films against the
//! textbook cases, metals against the reflectances their optical constants give at normal incidence,
//! and nothing anywhere reflecting more light than arrives or less than none.

use gf_core::material::*;

fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol, "{what}: {a} against {b} (allowed {tol})");
}

#[test]
fn a_film_of_no_thickness_is_the_bare_surface() {
    for &n3 in &[1.33, 1.5, 2.4] {
        for &nf in &[1.0, 1.33, 1.9, 2.4] {
            for i in 0..=20 {
                let c = i as f64 / 20.0;
                for &lam in &[400.0, 550.0, 700.0] {
                    // (exactly edge-on the film's sums are taken a hair off the edge)
                    let tol = if i == 0 { 2e-3 } else { 1e-12 };
                    close(film_reflectance(c, lam, 0.0, 1.0, nf, (n3, 0.0)), fresnel(c, 1.0, n3), tol, "no film, from outside");
                    // and from inside the body, where there may be total reflection
                    close(film_reflectance(c, lam, 0.0, n3, nf, (1.0, 0.0)), fresnel(c, n3, 1.0), tol, "no film, from inside");
                }
            }
        }
    }
}

#[test]
fn a_quarter_wave_film_of_the_right_index_reflects_nothing() {
    // the ideal anti-reflection coating: index the geometric mean of what lies either side of it, a
    // quarter of a wavelength (inside the film) thick
    for &n3 in &[1.5f64, 1.8, 2.4] {
        let nf = n3.sqrt();
        for &lam in &[450.0, 550.0, 650.0] {
            let d = lam / (4.0 * nf);
            assert!(film_reflectance(1.0, lam, d, 1.0, nf, (n3, 0.0)) < 1e-12);
            // off its wavelength it reflects again
            assert!(film_reflectance(1.0, lam * 0.6, d, 1.0, nf, (n3, 0.0)) > 1e-3);
        }
    }
}

#[test]
fn a_quarter_wave_film_matches_the_textbook_formula() {
    // R = ((n1 n3 - n2^2) / (n1 n3 + n2^2))^2 for a quarter-wave film at normal incidence
    for &(nf, n3) in &[(1.38f64, 1.52f64), (1.33, 1.0), (2.35, 1.5), (1.9, 1.5)] {
        let lam = 550.0;
        let want = ((n3 - nf * nf) / (n3 + nf * nf)).powi(2);
        close(film_reflectance(1.0, lam, lam / (4.0 * nf), 1.0, nf, (n3, 0.0)), want, 1e-12, "quarter-wave film");
    }
    // magnesium fluoride on crown glass: the classic single-layer coating leaves about 1.3%
    close(film_reflectance(1.0, 550.0, 550.0 / (4.0 * 1.38), 1.0, 1.38, (1.52, 0.0)), 0.0126, 2e-4, "MgF2 on glass");
    // a soap film in air is at its brightest there: about 7.7%
    close(film_reflectance(1.0, 550.0, 550.0 / (4.0 * 1.33), 1.0, 1.33, (1.0, 0.0)), 0.0771, 2e-4, "soap film");
}

#[test]
fn a_half_wave_film_is_as_if_absent() {
    for &(nf, n3) in &[(1.38, 1.52), (2.0, 1.5), (1.33, 1.0)] {
        let lam = 600.0;
        close(film_reflectance(1.0, lam, lam / (2.0 * nf), 1.0, nf, (n3, 0.0)), fresnel(1.0, 1.0, n3), 1e-12, "half-wave film");
    }
}

#[test]
fn a_soap_film_goes_black_as_it_thins() {
    // air on both sides: the two reflections cancel when the film is far thinner than the light's wavelength
    assert!(film_reflectance(1.0, 550.0, 2.0, 1.0, 1.33, (1.0, 0.0)) < 1e-3);
    assert!(film_reflectance(1.0, 550.0, 100.0, 1.0, 1.33, (1.0, 0.0)) > 0.05);
}

#[test]
fn film_colour_changes_with_thickness() {
    // the reflected colour of a soap film moves through the spectrum as the film thickens
    let hue = |d: f64| {
        let c = colour_of(|l| film_reflectance(1.0, l, d, 1.0, 1.33, (1.0, 0.0)));
        let m = c[0].max(c[1]).max(c[2]);
        [c[0] / m, c[1] / m, c[2] / m]
    };
    let (a, b, c) = (hue(120.0), hue(210.0), hue(300.0));
    let far = |x: [f64; 3], y: [f64; 3]| (0..3).map(|i| (x[i] - y[i]).abs()).fold(0.0, f64::max);
    assert!(far(a, b) > 0.2 && far(b, c) > 0.2 && far(a, c) > 0.2, "{a:?} {b:?} {c:?}");
}

#[test]
fn no_surface_creates_light() {
    // clear bodies under films, at every angle, wavelength and thickness tried
    for &n1 in &[1.0, 1.5, 2.4] {
        for &nf in &[1.0, 1.33, 1.9, 2.6] {
            for &n3 in &[1.0, 1.31, 1.5, 2.7] {
                for i in 0..=40 {
                    let c = i as f64 / 40.0;
                    for j in 0..=12 {
                        let lam = 400.0 + 25.0 * j as f64;
                        for &d in &[0.0, 30.0, 137.0, 420.0, 1500.0] {
                            let r = film_reflectance(c, lam, d, n1, nf, (n3, 0.0));
                            assert!(r.is_finite() && (-1e-9..=1.0 + 1e-9).contains(&r), "film {n1} {nf} {n3} cos {c} {lam} nm {d} nm: {r}");
                        }
                    }
                    let r = fresnel(c, n1, n3);
                    assert!((0.0..=1.0).contains(&r));
                }
            }
        }
    }
    // every metal, bare, under a coat's index, and under films
    for m in METALS {
        for k in 0..NM_N {
            let (lam, n, kk) = (NM0 + NM_STEP * k as f64, m.n[k] as f64, m.k[k] as f64);
            for i in 0..=40 {
                let c = i as f64 / 40.0;
                for &n1 in &[1.0, 1.5, 2.0] {
                    let r = fresnel_metal(c, n1, n, kk);
                    assert!(r.is_finite() && (0.0..=1.0 + 1e-9).contains(&r), "{} {lam} nm cos {c} from {n1}: {r}", m.id);
                    for &d in &[0.0, 50.0, 95.0, 400.0] {
                        let r = film_reflectance(c, lam, d, n1, 2.4, (n, kk));
                        assert!(r.is_finite() && (-1e-9..=1.0 + 1e-9).contains(&r), "{} under a film: {r}", m.id);
                    }
                }
            }
        }
    }
}

#[test]
fn metals_at_normal_incidence_match_their_constants() {
    // straight on, the reflectance of a metal is ((n-1)^2 + k^2) / ((n+1)^2 + k^2)
    for m in METALS {
        for k in 0..NM_N {
            let (n, kk) = (m.n[k] as f64, m.k[k] as f64);
            let want = ((n - 1.0).powi(2) + kk * kk) / ((n + 1.0).powi(2) + kk * kk);
            close(fresnel_metal(1.0, 1.0, n, kk), want, 1e-12, m.id);
            // a film of no thickness on it changes nothing
            close(film_reflectance(1.0, 550.0, 0.0, 1.0, 2.0, (n, kk)), want, 1e-12, m.id);
            // and at grazing everything reflects
            close(fresnel_metal(0.0, 1.0, n, kk), 1.0, 1e-9, m.id);
        }
    }
}

#[test]
fn metals_match_the_published_reflectances() {
    // The values the gathered data gives at 450, 550 and 650 nm (docs/materials_optical_data.json, taken
    // from the same tables): gold is weak in the blue and strong in the red, silver strong throughout,
    // copper turns on between green and red, chromium is flat.
    let want: &[(&str, [f64; 3])] = &[
        ("silver", [0.9802, 0.9831, 0.9898]),
        ("gold", [0.4082, 0.7916, 0.9565]),
        ("copper", [0.5383, 0.6235, 0.9352]),
        ("aluminium", [0.9219, 0.9154, 0.9056]),
        ("chromium", [0.5548, 0.5545, 0.5556]),
        ("titanium", [0.5445, 0.5823, 0.6159]),
        ("iron", [0.4958, 0.5124, 0.5312]),
        ("nickel", [0.5707, 0.6443, 0.7013]),
        ("platinum", [0.5884, 0.6424, 0.6789]),
        ("tungsten", [0.4685, 0.4962, 0.5173]),
    ];
    assert_eq!(want.len(), METALS.len());
    for (id, r) in want {
        let m = METALS.iter().find(|m| m.id == *id).unwrap_or_else(|| panic!("no metal called {id}"));
        for (i, lam) in [450.0, 550.0, 650.0].iter().enumerate() {
            let got = fresnel_metal(1.0, 1.0, table_at(&m.n, *lam), table_at(&m.k, *lam));
            close(got, r[i], 6e-4, &format!("{id} at {lam} nm"));
        }
    }
}

#[test]
fn a_metal_that_does_not_absorb_is_glass() {
    for &n in &[1.33, 1.5, 2.4] {
        for i in 0..=20 {
            let c = i as f64 / 20.0;
            close(fresnel_metal(c, 1.0, n, 0.0), fresnel(c, 1.0, n), 1e-9, "k = 0");
        }
    }
}

#[test]
fn metals_have_their_colours() {
    let rgb = |id: &str| {
        let m = METALS.iter().find(|m| m.id == id).unwrap();
        colour_of(|l| fresnel_metal(1.0, 1.0, table_at(&m.n, l), table_at(&m.k, l)))
    };
    let (au, cu, ag, cr) = (rgb("gold"), rgb("copper"), rgb("silver"), rgb("chromium"));
    assert!(au[0] > au[1] && au[1] > au[2] * 1.5, "gold is yellow: {au:?}");
    assert!(cu[0] > cu[1] * 1.2 && cu[1] > cu[2], "copper is red-orange: {cu:?}");
    assert!(ag.iter().all(|c| *c > 0.95), "silver is white and bright: {ag:?}");
    assert!((cr[0] - cr[2]).abs() < 0.03 && cr[1] > 0.5 && cr[1] < 0.6, "chrome is grey: {cr:?}");
}

#[test]
fn spectra_have_their_colours() {
    // what a piece of each looks like against white, at a strength that gives a clear colour
    let through = |id: &str, depth: f64| {
        let s = SPECTRA.iter().find(|s| s.id == id).unwrap_or_else(|| panic!("no spectrum called {id}"));
        colour_of(|l| (-table_at(&s.shape, l) * depth).exp())
    };
    let ruby = through("ruby", 4.0);
    assert!(ruby[0] > 2.0 * ruby[1] && ruby[0] > ruby[2], "ruby is red: {ruby:?}");
    let emerald = through("emerald", 3.0);
    assert!(emerald[1] > emerald[0] && emerald[1] > emerald[2], "emerald is green: {emerald:?}");
    let cobalt = through("cobalt", 3.0);
    assert!(cobalt[2] > 2.0 * cobalt[1] && cobalt[2] > cobalt[0], "cobalt glass is blue: {cobalt:?}");
    let amber = through("amber", 3.0);
    assert!(amber[0] > amber[1] && amber[1] > amber[2], "amber glass is brown: {amber:?}");
    let signal = through("signal", 3.0);
    assert!(signal[0] > 5.0 * signal[1].max(signal[2]), "selenium ruby is pure red: {signal:?}");
    for s in SPECTRA {
        let top = s.shape.iter().cloned().fold(0.0f32, f32::max);
        assert!((top - 1.0).abs() < 1e-3 && s.shape.iter().all(|v| (0.0..=1.0).contains(v)), "{}: a shape runs from 0 to 1", s.id);
        assert!(!s.source.is_empty());
    }
}

#[test]
fn ruby_deepens_with_thickness() {
    // a thin ruby is pink-purple (some blue gets through); a thick one is pure red
    let s = SPECTRA.iter().find(|s| s.id == "ruby").unwrap();
    let blue_share = |depth: f64| {
        let c = colour_of(|l| (-table_at(&s.shape, l) * depth).exp());
        c[2] / c[0]
    };
    assert!(blue_share(2.0) > 2.0 * blue_share(30.0), "{} against {}", blue_share(2.0), blue_share(30.0));
}

#[test]
fn scattering_follows_the_body_colour() {
    let white = scatter([1.0, 1.0, 1.0], 0.1, 1.0);
    assert!((white.sigma_s - 10.0).abs() < 1e-9);
    // a white body hardly absorbs; a dark one absorbs much; and what absorbs more dies away faster
    assert!(white.sigma_a.iter().all(|a| *a < 0.02 * white.sigma_s), "{white:?}");
    let jade = scatter([0.36, 0.63, 0.41], 0.1, 1.0);
    assert!(jade.sigma_a[0] > jade.sigma_a[2] && jade.sigma_a[2] > jade.sigma_a[1]);
    assert!(jade.kappa[0] > jade.kappa[2] && jade.kappa[2] > jade.kappa[1]);
    // less milkiness, or light that travels further, scatters less
    assert!(scatter([0.5; 3], 0.1, 0.5).sigma_s < scatter([0.5; 3], 0.1, 1.0).sigma_s);
    assert!(scatter([0.5; 3], 0.4, 1.0).sigma_s < scatter([0.5; 3], 0.1, 1.0).sigma_s);
    for c in [0.0, 0.01, 0.5, 0.99, 1.0] {
        let s = scatter([c; 3], 0.2, 1.0);
        assert!(s.sigma_a[0].is_finite() && s.kappa[0].is_finite() && s.sigma_a[0] >= 0.0);
    }
}

#[test]
fn softened_lights_keep_their_light() {
    // a map with a few bright patches on a dim ground
    let (w, h) = (256usize, 128usize);
    let mut map = vec![0.0f32; w * h * 4];
    for j in 0..h {
        for i in 0..w {
            let o = (j * w + i) * 4;
            let lit = (40..52).contains(&j) && (100..130).contains(&i) || (80..90).contains(&j) && (10..20).contains(&i);
            map[o] = if lit { 9.0 } else { 0.02 };
            map[o + 1] = if lit { 6.0 } else { 0.03 };
            map[o + 2] = if lit { 3.0 } else { 0.05 };
            map[o + 3] = 1.0;
        }
    }
    // the light a map holds: each texel weighed by the share of the sphere it covers
    let total = |m: &[f32], w: usize, h: usize| -> [f64; 3] {
        let mut t = [0.0f64; 3];
        let mut ws = 0.0;
        for j in 0..h {
            let s = ((j as f64 + 0.5) / h as f64 * std::f64::consts::PI).sin();
            for i in 0..w {
                for c in 0..3 { t[c] += m[(j * w + i) * 4 + c] as f64 * s; }
                ws += s;
            }
        }
        t.map(|v| v / ws)
    };
    let levels = env_levels(&map, w, h);
    assert_eq!(levels.len(), 7);
    let t0 = total(&map, w, h);
    let (mut lw, mut lh) = (w, h);
    let mut last_peak = 9.0f32;
    for (k, (a, b, data)) in levels.iter().enumerate() {
        lw /= 2;
        lh /= 2;
        assert_eq!((*a, *b), (lw, lh));
        assert_eq!(data.len(), lw * lh * 4);
        let t = total(data, lw, lh);
        for c in 0..3 {
            // (the coarsest copies hold a handful of texels: the sum is rougher there)
            let tol = if lw >= 16 { 0.04 } else { 0.25 };
            assert!((t[c] / t0[c] - 1.0).abs() < tol, "copy {}: holds {} of the light in channel {c}", k + 1, t[c] / t0[c]);
        }
        // each copy is softer than the one before: its brightest point is lower
        let peak = data.chunks(4).map(|p| p[0]).fold(0.0f32, f32::max);
        assert!(peak <= last_peak * 1.0001, "copy {} is not softer", k + 1);
        last_peak = peak;
        assert!(data.iter().all(|v| v.is_finite() && *v >= 0.0));
    }
    // the last is close to the same everywhere
    let (_, _, last) = &levels[6];
    let (lo, hi) = last.chunks(4).fold((f32::MAX, 0.0f32), |(lo, hi), p| (lo.min(p[0]), hi.max(p[0])));
    assert!(hi < 2.5 * lo, "the softest copy still varies from {lo} to {hi}");
    // an even map stays even at every level
    let even = vec![0.5f32; w * h * 4];
    for (_, _, data) in env_levels(&even, w, h) {
        assert!(data.chunks(4).all(|p| (p[0] - 0.5).abs() < 1e-4));
    }
}
