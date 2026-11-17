//! The camera in a photographer's terms: the numbers, the names, and the two directions of every lens.

use gf_core::camera::{self, Camera, Projection};
use gf_core::math::DEG;
use gf_core::settings::Settings;
use gf_core::state::{KnobRef, State};

fn args(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// Settings for one kind of lens, with something of everything switched on.
fn lens(proj: Projection, distort: f64) -> Settings {
    let mut p = Settings::default();
    p.dist = 4.0;
    p.fov = 40.0;
    p.shift = [0.07, -0.04];
    p.distort = distort;
    p.proj = proj.index() as f64;
    p
}

#[test]
fn field_of_view_from_focal_length_and_sensor() {
    // 50 mm on full frame: 27 degrees top to bottom; 24 mm: 53; the same 50 mm on APS-C: 17.8
    assert!(close(camera::fov_of(50.0, 24.0), 26.99, 0.01));
    assert!(close(camera::fov_of(24.0, 24.0), 53.13, 0.01));
    assert!(close(camera::fov_of(50.0, 15.7), 17.85, 0.01));
    for fov in [6.0, 26.0, 28.0, 80.0, 143.0] {
        for sensor in [8.8, 24.0, 102.0] {
            assert!(close(camera::fov_of(camera::focal_of(fov, sensor), sensor), fov, 1e-10));
        }
    }
    // the settings that every family starts with read as a normal lens close to a small subject
    let mut p = Settings::default();
    p.fov = 26.0;
    p.dist = 6.6;
    let r = Camera::of(&p).readout(16.0 / 9.0);
    assert!(close(r.focal, 51.98, 0.01), "{}", r.focal);
    assert!(close(r.distance_mm, 253.8, 0.1), "{}", r.distance_mm);
    assert!(close(r.field_v, 26.0, 1e-9) && close(r.field_h, 44.63, 0.01), "{} {}", r.field_v, r.field_h);
    assert!(r.fstop.is_none() && r.notes.is_empty(), "{:?}", r.notes);
}

#[test]
fn the_old_names_are_drawn_from_untouched() {
    // nothing said the photographer's way: the camera is the stored numbers themselves, to the last bit
    let mut p = Settings::default();
    (p.fov, p.dist, p.aperture, p.fnear, p.ffar) = (27.3456, 5.5, 0.12, -0.4, -0.1);
    let before = p.clone();
    camera::sync(&mut p);
    assert_eq!(p, before);
    let c = Camera::of(&p);
    assert_eq!(c.t.to_bits(), (27.3456f64 * DEG / 2.0).tan().to_bits());
    assert_eq!((c.lens, c.near, c.far, c.dist), (0.12, -0.4, -0.1, 5.5));
    assert!(c.plain());
    // and setting them by name is no conversion either
    let mut q = Settings::default();
    for (k, v) in [("fov", 27.3456), ("aperture", 0.12)] {
        assert!(camera::set(&mut q, k, v));
    }
    assert_eq!((q.fov, q.aperture, q.focal, q.fstop), (27.3456, 0.12, 0.0, 0.0));
}

#[test]
fn whichever_name_was_said_last_is_kept() {
    let mut p = Settings::default();
    camera::set(&mut p, "focal", 85.0);
    assert!(close(p.fov, camera::fov_of(85.0, 24.0), 1e-12) && p.focal == 85.0);
    assert_eq!(camera::get(&p, "focal"), Some(85.0));
    // the old name takes the lens back
    camera::set(&mut p, "fov", 40.0);
    assert_eq!((p.fov, p.focal), (40.0, 0.0));
    assert!(close(camera::get(&p, "focal").unwrap(), camera::focal_of(40.0, 24.0), 1e-12));
    // the same for the lens opening
    camera::set(&mut p, "fstop", 4.0);
    let f = camera::focal_of(40.0, 24.0);
    assert!(close(p.aperture, f / 8.0 / (100.0 / camera::SPAN), 1e-12), "{}", p.aperture);
    assert!(close(camera::get(&p, "fstop").unwrap(), 4.0, 1e-12));
    camera::set(&mut p, "aperture", 0.2);
    assert_eq!((p.aperture, p.fstop), (0.2, 0.0));
    assert!(close(camera::get(&p, "fstop").unwrap(), f / (2.0 * 0.2 * 100.0 / camera::SPAN), 1e-12));
    // no lens opening reads as no f-stop
    camera::set(&mut p, "aperture", 0.0);
    assert_eq!(camera::get(&p, "fstop"), Some(0.0));
    assert!(Camera::of(&p).fstop().is_none());
}

#[test]
fn the_rings_behave_as_a_lens_does() {
    let mut p = Settings::default();
    camera::set(&mut p, "focal", 50.0);
    camera::set(&mut p, "fstop", 2.8);
    let a50 = p.aperture;
    // zooming at the same f-stop opens the lens in proportion
    camera::set(&mut p, "focal", 100.0);
    assert!(close(p.aperture, 2.0 * a50, 1e-12));
    // stopping down two stops halves it
    camera::set(&mut p, "fstop", 5.6);
    assert!(close(p.aperture, a50, 1e-12));
    // a subject ten times the size: the same lens is ten times smaller beside it
    camera::set(&mut p, "subject", 1000.0);
    assert!(close(p.aperture, a50 / 10.0, 1e-12));
    // a smaller sensor behind the same lens: a narrower view, the same opening
    let (fov, a) = (p.fov, p.aperture);
    camera::set(&mut p, "sensor", 15.7);
    assert!(p.fov < fov && close(p.fov, camera::fov_of(100.0, 15.7), 1e-12) && p.aperture == a);
    // ... also when the lens was given the old way: it is the same lens that moves to the new sensor
    let mut q = Settings::default();
    q.fov = 28.0;
    let f = camera::focal_mm(&q);
    camera::set(&mut q, "sensor", 15.7);
    assert!(close(q.focal, f, 1e-12) && close(q.fov, camera::fov_of(f, 15.7), 1e-12));
    // with the opening given the old way it is a fixed size whatever the lens does, as it always was
    let mut q = Settings::default();
    camera::set(&mut q, "aperture", 0.1);
    camera::set(&mut q, "focal", 200.0);
    camera::set(&mut q, "subject", 5.0);
    assert_eq!(q.aperture, 0.1);
}

#[test]
fn the_views_read_and_write_the_stored_numbers() {
    let mut p = Settings::default();
    (p.fnear, p.ffar, p.dist) = (-0.5, 0.1, 6.0);
    assert!(close(camera::get(&p, "focus").unwrap(), -0.2, 1e-12) && close(camera::get(&p, "band").unwrap(), 0.6, 1e-12));
    camera::set(&mut p, "focus", 0.3);
    assert!(close(p.fnear, 0.0, 1e-6) && close(p.ffar, 0.6, 1e-6));
    camera::set(&mut p, "band", 0.0);
    assert!(close(p.fnear, 0.3, 1e-6) && p.fnear == p.ffar);
    // the distance focused at, in millimetres from the lens
    let unit = 100.0 / camera::SPAN;
    assert!(close(camera::get(&p, "focusDist").unwrap(), 6.3 * unit, 1e-3));
    camera::set(&mut p, "focusDist", 200.0);
    assert!(close(camera::get(&p, "focus").unwrap(), 200.0 / unit - 6.0, 1e-6));
    // exposure in stops
    camera::set(&mut p, "ev", 1.0);
    assert_eq!(p.exposure, 2.0);
    camera::set(&mut p, "ev", -0.5);
    assert!(close(p.exposure, 0.5f64.sqrt(), 1e-15) && close(camera::get(&p, "ev").unwrap(), -0.5, 1e-12));
    camera::set(&mut p, "shiftX", 0.25);
    camera::set(&mut p, "shiftY", -0.1);
    assert_eq!(p.shift, [0.25, -0.1]);
    // the scales a slider wants: stops, doublings of the focal length, powers of ten of the subject's size
    camera::set(&mut p, "av", 3.0);
    assert!(close(p.fstop, 2.0f64.sqrt().powi(3), 1e-12) && close(camera::get(&p, "av").unwrap(), 3.0, 1e-12));
    camera::set(&mut p, "aperture", 0.0);
    assert_eq!(camera::get(&p, "av"), Some(12.0));
    camera::set(&mut p, "zoom", 1.0);
    assert!(close(p.focal, 100.0, 1e-12) && close(camera::get(&p, "zoom").unwrap(), 1.0, 1e-12));
    camera::set(&mut p, "subjectLog", 3.0);
    assert!(close(p.subject, 1000.0, 1e-9) && close(camera::get(&p, "subjectLog").unwrap(), 3.0, 1e-12));
    // names that are not the camera's are left to the caller
    assert!(!camera::set(&mut p, "ior", 1.7) && p.ior == 1.5);
    assert!(camera::get(&p, "ior").is_none());
}

#[test]
fn a_point_comes_back_to_the_pixel_whose_ray_passes_through_it() {
    // the round trip the antialiasing relies on: every kind of lens, straight and bowed either way
    for proj in Projection::ALL {
        for distort in [0.0, 0.25, -0.25, 0.4, -0.4] {
            let c = Camera::of(&lens(proj, distort));
            let mut seen = 0;
            for j in -8..=8 {
                for i in -14..=14 {
                    let uv = [i as f64 * 0.125, j as f64 * 0.125];
                    let Some((o, d)) = c.ray(uv) else { continue };
                    assert!(close(d[0] * d[0] + d[1] * d[1] + d[2] * d[2], 1.0, 1e-12));
                    for t in [0.5, 3.0, 9.0] {
                        let p = [o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t];
                        // straight up or down in a panorama, and straight behind a fisheye, are every
                        // point of a row at once: nothing to come back to
                        if proj == Projection::Panorama && (uv[1] + c.shift[1]).abs() * c.t > 1.57 { continue; }
                        let Some(back) = c.project(p) else {
                            // only the very edge of a barrel lens's picture may fail to return
                            assert!(distort > 0.0 && (uv[0] + c.shift[0]).hypot(uv[1] + c.shift[1]) * c.warp >= camera::WARP_MAX - 1e-9, "{proj:?} {distort} {uv:?}");
                            continue;
                        };
                        // (a panorama wraps: the same place may come back one turn away)
                        let mut dx = back[0] - uv[0];
                        if proj == Projection::Panorama || proj == Projection::Cylindrical {
                            let turn = 2.0 * std::f64::consts::PI / c.t;
                            dx -= (dx / turn).round() * turn;
                        }
                        assert!(dx.abs() < 1e-7 && (back[1] - uv[1]).abs() < 1e-7, "{proj:?} distort {distort} {uv:?} t {t}: {back:?}");
                        assert!(close(c.along(p), t, 1e-9));
                        seen += 1;
                    }
                }
            }
            assert!(seen > 300, "{proj:?} {distort}: only {seen} points were visible");
        }
    }
}

#[test]
fn every_lens_looks_the_same_way_at_the_middle_of_the_picture() {
    // the same focal length gives the same scale at the middle whatever the lens: a small step
    // across the picture turns the ray by the same angle
    let mut steps = Vec::new();
    for proj in Projection::ALL {
        if matches!(proj, Projection::Orthographic | Projection::Panorama) { continue; }
        let mut p = Settings::default();
        p.proj = proj.index() as f64;
        p.fov = 40.0;
        let c = Camera::of(&p);
        let (_, d0) = c.ray([0.0, 0.0]).unwrap();
        assert_eq!(d0, [0.0, 0.0, -1.0]);
        let (_, d1) = c.ray([0.001, 0.0]).unwrap();
        let (_, d2) = c.ray([0.0, 0.001]).unwrap();
        assert!(d1[0] > 0.0 && d2[1] > 0.0, "{proj:?}: across is +x, up is +y");
        steps.push(d1[0].atan2(-d1[2]));
        steps.push(d2[1].atan2(-d2[2]));
    }
    for s in &steps {
        assert!(close(*s / steps[0], 1.0, 1e-5), "{steps:?}");
    }
    // an orthographic view shows of the object's centre what the perspective one does
    let mut p = Settings::default();
    (p.fov, p.dist) = (40.0, 5.0);
    let persp = Camera::of(&p).point([0.3, 0.6], 5.0 / Camera::of(&p).ray([0.3, 0.6]).unwrap().1[2].abs()).unwrap();
    p.proj = 1.0;
    let ortho = Camera::of(&p).point([0.3, 0.6], 5.0).unwrap();
    for k in 0..3 {
        assert!(close(persp[k], ortho[k], 1e-12), "{persp:?} {ortho:?}");
    }
    // a panorama twice as wide as high is the whole sphere
    p.proj = 5.0;
    let r = Camera::of(&p).readout(2.0);
    assert!(close(r.field_v, 180.0, 1e-9) && close(r.field_h, 360.0, 1e-9), "{} {}", r.field_v, r.field_h);
}

#[test]
fn the_fisheyes_cover_what_real_ones_do() {
    // 15 mm equisolid on full frame: the classic fisheye that takes in half the world corner to corner;
    // 8 mm equidistant: a circle of half the world, as high as the frame
    let mut p = Settings::default();
    camera::set(&mut p, "focal", 15.0);
    p.proj = Projection::Equisolid.index() as f64;
    let r = Camera::of(&p).readout(1.5);
    assert!(close(r.field_d, 185.0, 1.0) && close(r.field_v, 94.3, 0.5), "{} {}", r.field_d, r.field_v);
    camera::set(&mut p, "focal", 8.0);
    p.proj = Projection::Equidistant.index() as f64;
    let c = Camera::of(&p);
    let r = c.readout(1.5);
    assert!(close(r.field_v, 171.9, 0.2), "{}", r.field_v);
    // beyond its circle a fisheye draws nothing
    assert!(c.ray([2.2, 0.0]).is_none() && c.ray([1.4, 0.0]).is_some());
    // and what lies behind the camera still has a place in the picture
    let behind = c.project([0.3, 0.0, c.dist + 0.2]).unwrap();
    assert!(behind[0] > 1.0);
}

/// Where a point shows through each point of the lens opening: the pixel whose ray, aimed at the surface
/// in focus from that point of the lens, passes through it. The spread of those is the point's blur.
fn spread(c: &Camera, uv: [f64; 2], t: f64, h: f64) -> (f64, f64) {
    let centre = uv;
    let (mut far, mut mean) = (0.0f64, [0.0f64; 2]);
    let n = 24;
    for k in 0..n {
        let a = std::f64::consts::TAU * k as f64 / n as f64;
        let l = [a.cos(), a.sin()];
        // look for the pixel whose lens ray from l passes through the point: start at the point's own
        // pixel and step by the miss (the picture of a small move is nearly the move itself)
        let target = c.point(uv, t).unwrap();
        let mut at = centre;
        for _ in 0..30 {
            let (o, d) = c.lens_ray(at, c.along_focus(at), l).unwrap();
            // the lens ray's nearest approach to the point, seen back through the middle of the lens
            let s = (0..3).map(|i| (target[i] - o[i]) * d[i]).sum::<f64>();
            let q = [o[0] + d[0] * s, o[1] + d[1] * s, o[2] + d[2] * s];
            let miss = [target[0] - q[0], target[1] - q[1], target[2] - q[2]];
            let here = c.project(target).unwrap();
            let there = c.project([target[0] - miss[0], target[1] - miss[1], target[2] - miss[2]]).unwrap();
            at = [at[0] + here[0] - there[0], at[1] + here[1] - there[1]];
        }
        let r = ((at[0] - centre[0]).powi(2) + (at[1] - centre[1]).powi(2)).sqrt() * h / 2.0;
        far = far.max(r);
        mean = [mean[0] + (at[0] - centre[0]) / n as f64, mean[1] + (at[1] - centre[1]) / n as f64];
    }
    (far, (mean[0].powi(2) + mean[1].powi(2)).sqrt() * h / 2.0)
}

trait AlongFocus {
    fn along_focus(&self, uv: [f64; 2]) -> f64;
}
impl AlongFocus for Camera {
    /// How far along a pixel's central ray the surface in focus is (a band of no depth: one surface).
    fn along_focus(&self, uv: [f64; 2]) -> f64 {
        self.reach(uv, self.near).unwrap()
    }
}

#[test]
fn the_traced_lens_and_the_blur_formula_agree() {
    // The two ways of focusing follow the same camera: rays traced through the lens opening spread a
    // point over a disc (worked out here by following the rays), and the blur stage gives every pixel
    // a radius from its depth (`Camera::blur`). They must be the same size, for every kind of lens.
    let h = 1080.0;
    for proj in Projection::ALL {
        // (the lenses that see all round are held to the formula near the middle of the picture, where a
        // step across the picture is the same angle every way: further out their blur is drawn round
        // where a traced one is oval)
        let places: &[([f64; 2], f64)] = if proj.radial() { &[([0.0, 0.0], 0.0), ([0.2, 0.1], 0.0)] } else { &[([0.0, 0.0], 0.0), ([0.35, 0.2], 0.0), ([0.35, 0.2], 20.0), ([-0.6, 0.45], 0.0)] };
        for &(uv, tilt) in places {
            let mut p = lens(proj, 0.0);
            p.shift = [0.0; 2];
            (p.aperture, p.fnear, p.ffar) = (0.05, 0.25, 0.25);
            (p.tilt_x, p.tilt_y) = (tilt, -0.5 * tilt);
            let c = Camera::of(&p);
            let tf = c.along_focus(uv);
            for off in [-1.2, -0.4, 0.5, 1.3] {
                let t = tf + off;
                let want = c.blur(uv, t, h).unwrap();
                assert_eq!(want > 0.0, off > 0.0, "{proj:?}: behind the plane of focus is positive");
                let (got, drift) = spread(&c, uv, t, h);
                // exact for a lens that focuses on a plane square to its axis; within a few percent
                // where the plane leans (the disc is then a little oval) and for the lenses whose
                // opening faces along each ray
                let tol = if proj.radial() { 0.06 } else if tilt != 0.0 { 0.03 } else { 0.004 };
                assert!((got / want.abs() - 1.0).abs() < tol, "{proj:?} uv {uv:?} tilt {tilt} off {off}: traced {got:.3} px, formula {:.3} px", want.abs());
                assert!(drift < 0.02 * got + 0.01, "{proj:?}: the disc is centred on the point ({drift} of {got})");
            }
            // in focus: no blur at all, through every point of the lens
            assert!(c.blur(uv, tf, h).unwrap().abs() < 1e-9);
        }
    }
}

#[test]
fn the_blur_is_the_thin_lens_formula() {
    // A lens of focal length F at f-number N focused at distance s draws a point at distance z as a
    // disc of diameter  F^2 / (N s) x |z - s| / z  on the sensor (F and N the effective ones).
    // 100 mm, f/4, a 50 mm subject, focused at 300 mm, a point at 330 mm: 0.758 mm across on a 24 mm
    // sensor; in a picture 1200 pixels high that is a radius of 18.9 pixels.
    let mut p = Settings::default();
    camera::set(&mut p, "focal", 100.0);
    camera::set(&mut p, "fstop", 4.0);
    camera::set(&mut p, "subject", 50.0);
    let unit = 50.0 / camera::SPAN;
    p.dist = 300.0 / unit;
    camera::set(&mut p, "focus", 0.0);
    camera::set(&mut p, "band", 0.0);
    let c = Camera::of(&p);
    assert!(close(c.focus_mm(), 300.0, 1e-3));
    let on_sensor = 100.0 * 100.0 / (4.0 * 300.0) * 30.0 / 330.0;
    assert!(close(on_sensor, 0.7576, 1e-4));
    let px = c.blur([0.0, 0.0], 330.0 / unit, 1200.0).unwrap();
    assert!(close(px, on_sensor / 24.0 * 1200.0 / 2.0, 0.01), "{px}");
    // the same lens and framing on a subject forty times the size: a fortieth of the blur
    camera::set(&mut p, "subject", 2000.0);
    let px2 = Camera::of(&p).blur([0.0, 0.0], 330.0 / unit, 1200.0).unwrap();
    assert!(close(px2 * 40.0, px, 0.01), "{px2}");
    // what passes as sharp: 1/z between 1/s -+ c N / F^2 with c = 24 mm / 800
    let r = c.readout(1.5);
    let k = 0.03 * 4.0 / 10000.0;
    assert!(close(r.sharp_from_mm, 1.0 / (1.0 / 300.0 + k), 1e-3) && close(r.sharp_to_mm, 1.0 / (1.0 / 300.0 - k), 1e-3), "{} {}", r.sharp_from_mm, r.sharp_to_mm);
    assert!(close(r.hyperfocal_mm.unwrap(), 10000.0 / (4.0 * 0.03), 1e-6));
    assert!(close(r.magnification, 1.0 / 3.0, 1e-6));
    // what a real lens focused this close would be marked: F / (1 + m), N / (1 + m)
    assert!(close(r.engraved_focal, 75.0, 1e-6) && close(r.engraved_fstop.unwrap(), 3.0, 1e-6));
}

#[test]
fn a_click_focuses_on_what_was_clicked() {
    for proj in Projection::ALL {
        for (distort, tilt) in [(0.0, 0.0), (0.3, 0.0), (0.0, 25.0)] {
            let mut st = State::new();
            st.p = lens(proj, distort);
            st.p.aperture = 0.1;
            (st.p.tilt_x, st.p.tilt_y) = (tilt, tilt / 3.0);
            (st.p.fnear, st.p.ffar) = (-0.3, 0.1);
            // a pixel of a picture 640 x 360, counted from the bottom left, and glass 3.7 along its ray
            let (x, y, w, h, t) = (417.5, 251.5, 640.0, 360.0, 3.7);
            let e = st.focus_at(t, x, y, w, h);
            let c = st.camera();
            let uv = [(2.0 * x - w) / h, (2.0 * y - h) / h];
            assert!(close(c.depth_at(uv, t).unwrap(), e, 1e-12));
            // the band kept its depth and has that glass in its middle: it is sharp there, and blurred
            // half the band's depth and more to either side
            assert!(close(st.p.ffar - st.p.fnear, 0.4, 1e-12) && close(0.5 * (st.p.fnear + st.p.ffar), e, 1e-12));
            assert!(c.blur(uv, t, h).unwrap().abs() < 1e-9, "{proj:?}");
            assert!(c.blur(uv, t + 0.6, h).unwrap() > 0.0 && c.blur(uv, t - 0.6, h).unwrap() < 0.0, "{proj:?} {distort} {tilt}");
        }
    }
    // with the camera the program always had, the same numbers as ever
    let mut st = State::new();
    (st.p.fov, st.p.dist, st.p.shift) = (26.0, 6.6, [0.1, 0.0]);
    let e = st.focus_at(6.0, 400.0, 100.0, 640.0, 360.0);
    let tf = (26.0f64 * DEG / 2.0).tan();
    let (u, v) = ((2.0 * 400.0 - 640.0) / 360.0 + 0.1, (2.0 * 100.0 - 360.0) / 360.0);
    assert_eq!(e, 6.0 / ((u * tf).powi(2) + (v * tf).powi(2) + 1.0f64).sqrt() - 6.6);
    assert!(close(st.camera().depth_at([(2.0 * 400.0 - 640.0) / 360.0, v], 6.0).unwrap(), e, 1e-12));
}

#[test]
fn by_name_on_the_command_line() {
    // the old names alone: stored as given, nothing derived
    let st = State::with_args(&args(&[("fov", "33"), ("aperture", "0.12"), ("fnear", "-0.4"), ("ffar", "-0.1")]), true);
    assert_eq!((st.p.fov, st.p.aperture, st.p.fnear, st.p.ffar, st.p.focal, st.p.fstop), (33.0, 0.12, -0.4, -0.1, 0.0, 0.0));
    // the photographer's names
    let st = State::with_args(&args(&[("focal", "85"), ("fstop", "2"), ("subject", "20"), ("focusDist", "45"), ("band", "0")]), true);
    assert!(close(st.p.fov, camera::fov_of(85.0, 24.0), 1e-12));
    let unit = 20.0 / camera::SPAN;
    assert!(close(st.p.aperture, 85.0 / 4.0 / unit, 1e-12));
    assert!(st.p.fnear == st.p.ffar && close(st.p.fnear, 45.0 / unit - st.p.dist, 1e-5));
    // said both ways, the later one is meant
    let st = State::with_args(&args(&[("focal", "85"), ("fov", "40")]), true);
    assert_eq!((st.p.fov, st.p.focal), (40.0, 0.0));
    let st = State::with_args(&args(&[("fov", "40"), ("focal", "85")]), true);
    assert!(st.p.focal == 85.0 && close(st.p.fov, camera::fov_of(85.0, 24.0), 1e-12));
    let st = State::with_args(&args(&[("fstop", "8"), ("aperture", "0.05")]), true);
    assert_eq!((st.p.aperture, st.p.fstop), (0.05, 0.0));
    // a projection by its word; a lens change that keeps the subject's size
    let mut st = State::with_args(&args(&[("proj", "equisolid"), ("focal", "15")]), true);
    assert_eq!(st.camera().proj, Projection::Equisolid);
    assert!(st.apply_arg("proj", "pano") && st.p.proj == 5.0 && st.cut);
    assert!(st.apply_arg("proj", "0") && st.camera().proj == Projection::Perspective);
    assert!(!st.apply_arg("proj", "sideways"));
    let mut st = State::with_args(&args(&[("fov", "26"), ("dist", "6.6")]), true);
    let size = st.p.dist * st.camera().t;
    assert!(st.apply_arg("dollyZoom", "200"));
    assert!(close(st.p.dist * st.camera().t, size, 1e-12) && close(st.p.dist, 6.6 * 200.0 / camera::focal_of(26.0, 24.0), 1e-9), "{}", st.p.dist);
    assert!(st.apply_arg("ev", "-1") && st.p.exposure == 0.5);
    assert!(st.apply_arg("focus", "0.5") && st.apply_arg("shiftX", "0.2") && st.p.shift[0] == 0.2);
}

#[test]
fn sliders_and_tracks_go_through_the_camera() {
    let mut st = State::new();
    // every new slider exists, reads what is in force, and writes through
    for name in ["focal", "zoom", "av", "subjectLog", "fstop", "focus", "band", "subject", "ev", "distort", "fringe", "falloff", "squeeze", "bokeh", "catseye", "tiltX", "tiltY", "shiftX", "shiftY", "vignette", "grain"] {
        let k = State::engine_knob(name);
        let _ = st.get(&k.r);
    }
    assert!(close(st.get(&KnobRef::P("focal")), camera::focal_of(st.p.fov, 24.0), 1e-12));
    st.set(&KnobRef::P("focal"), 35.0);
    assert!(close(st.p.fov, camera::fov_of(35.0, 24.0), 1e-12));
    st.set(&KnobRef::P("fstop"), 5.6);
    assert!(st.p.aperture > 0.0 && close(st.get(&KnobRef::P("fstop")), 5.6, 1e-12));
    // a focal-length pull as a track: the lens opening follows it at every moment, and the state at
    // a given time does not depend on how it was reached
    let k = State::engine_knob("focal");
    st.toggle_wave(&k);
    st.set_playing(true);
    st.clock = 3.21;
    st.apply_time();
    let once = (st.p.focal, st.p.fov, st.p.aperture);
    for c in [0.4, 9.9, 1.0, 3.21] {
        st.clock = c;
        st.apply_time();
    }
    assert_eq!((st.p.focal, st.p.fov, st.p.aperture), once);
    assert!(close(st.p.aperture, st.p.focal / (2.0 * 5.6) / (100.0 / camera::SPAN), 1e-12));
    // a track on the old name still drives the lens, and takes it back to the old way
    let mut st = State::new();
    st.set(&KnobRef::P("focal"), 35.0);
    st.set(&KnobRef::P("fov"), 31.0);
    assert_eq!((st.p.fov, st.p.focal), (31.0, 0.0));
    // changing family puts that family's own lens on
    let mut st = State::new();
    st.set(&KnobRef::P("focal"), 135.0);
    st.set_scene(2);
    assert!(st.p.focal == 0.0 && close(st.p.fov, 26.0, 1e-12), "{} {}", st.p.focal, st.p.fov);
    // the wheel with the lens key: a focal length that was said stays said, and the size is kept
    let mut st = State::new();
    st.set(&KnobRef::P("focal"), 50.0);
    st.set(&KnobRef::P("fstop"), 4.0);
    let size = st.p.dist * st.camera().t;
    st.wheel(300.0, true);
    assert!(st.p.focal > 0.0 && st.p.focal < 50.0 && close(st.p.focal, camera::focal_of(st.p.fov, 24.0), 1e-9));
    assert!(close(st.p.dist * st.camera().t, size, 1e-9));
    assert!(close(st.p.aperture, st.p.focal / 8.0 / (100.0 / camera::SPAN), 1e-12));
    st.reset_view();
    assert_eq!(st.p.focal, 0.0);
}

#[test]
fn a_setup_keeps_the_camera() {
    let mut st = State::new();
    for (k, v) in [("focal", "85"), ("fstop", "2.8"), ("subject", "30"), ("sensor", "15.7"), ("proj", "cylindrical"), ("distort", "0.1"), ("tiltX", "12"), ("band", "0")] {
        assert!(st.apply_arg(k, v), "{k}");
    }
    let text = st.setup_json();
    let mut back = State::new();
    back.apply_setup_json(&text).unwrap();
    assert_eq!(back.p, st.p);
    assert_eq!(back.camera(), st.camera());
    // a setup from before the camera had these names opens as it was drawn: nothing said, nothing derived
    let mut old: serde_json::Value = serde_json::from_str(&State::new().setup_json()).unwrap();
    let s = old["settings"].as_object_mut().unwrap();
    for name in ["focal", "sensor", "fstop", "subject", "proj", "distort", "fringe", "falloff", "squeeze", "bokeh", "catseye", "tiltX", "tiltY"] {
        assert!(s.remove(name).is_some(), "{name} is saved with a setup");
    }
    s.insert("fov".into(), 31.5.into());
    s.insert("aperture".into(), 0.07.into());
    let mut back = State::new();
    back.apply_setup_json(&old.to_string()).unwrap();
    assert_eq!((back.p.fov, back.p.aperture, back.p.focal, back.p.fstop, back.p.sensor, back.p.subject), (31.5, 0.07, 0.0, 0.0, 24.0, 100.0));
    assert!(back.camera().plain());
    // a file edited by hand so that the two ways disagree: the focal length that was said is the lens
    s_set(&mut old, "focal", 50.0);
    let mut back = State::new();
    back.apply_setup_json(&old.to_string()).unwrap();
    assert!(close(back.p.fov, camera::fov_of(50.0, 24.0), 1e-12));
}

fn s_set(v: &mut serde_json::Value, name: &str, x: f64) {
    v["settings"].as_object_mut().unwrap().insert(name.into(), x.into());
}

#[test]
fn numbers_no_real_lens_has_are_said_not_refused() {
    let mut p = Settings::default();
    camera::set(&mut p, "focal", 50.0);
    camera::set(&mut p, "fstop", 0.3);
    let r = Camera::of(&p).readout(1.5);
    assert!(close(r.fstop.unwrap(), 0.3, 1e-12));
    assert!(r.notes.iter().any(|n| n.contains("f/0.30") && n.contains("f/0.5")), "{:?}", r.notes);
    camera::set(&mut p, "fstop", 0.6);
    assert!(Camera::of(&p).readout(1.5).notes.iter().any(|n| n.contains("faster than any camera lens")));
    camera::set(&mut p, "fstop", 45.0);
    assert!(Camera::of(&p).readout(1.5).notes.iter().any(|n| n.contains("diffraction")));
    camera::set(&mut p, "fstop", 8.0);
    assert!(Camera::of(&p).readout(1.5).notes.is_empty(), "{:?}", Camera::of(&p).readout(1.5).notes);
    // 2 mm on full frame keeping lines straight: wider than any such lens; as a fisheye it is fine
    camera::set(&mut p, "focal", 2.0);
    let c = Camera::of(&p);
    assert!(c.t > 5.9 && c.readout(1.5).notes.iter().any(|n| n.contains("2.0 mm") && n.contains("wider than any lens")), "{:?}", c.readout(1.5).notes);
    camera::set(&mut p, "focal", 8.0);
    p.proj = Projection::Equidistant.index() as f64;
    assert!(Camera::of(&p).readout(1.5).notes.is_empty());
    // a telescope, and a microscope
    let mut p = Settings::default();
    camera::set(&mut p, "focal", 2000.0);
    assert!(Camera::of(&p).readout(1.5).notes.iter().any(|n| n.contains("telescope")));
    let mut p = Settings::default();
    camera::set(&mut p, "focal", 100.0);
    camera::set(&mut p, "subject", 2.0);
    assert!(Camera::of(&p).readout(1.5).notes.iter().any(|n| n.contains("microscope")));
}

#[test]
fn a_fisheye_chosen_on_a_long_lens_is_given_a_short_one() {
    let mut s = State::with_args(&[], true);
    let (focal, dist) = (s.camera().focal, s.p.dist);
    assert!(focal > 40.0);
    s.lens_kind(Projection::Equidistant.index());
    let c = s.camera();
    assert_eq!(c.proj, Projection::Equidistant);
    assert!((c.focal - 12.0).abs() < 1e-6 && s.p.dist < dist && s.p.dist >= 2.5, "{} {}", c.focal, s.p.dist);
    // a short lens already: left as it is
    let mut s = State::with_args(&args(&[("focal", "14")]), true);
    s.lens_kind(Projection::Equisolid.index());
    assert!((s.camera().focal - 14.0).abs() < 1e-6);
    // an ordinary lens chosen again keeps whatever focal length there is
    s.lens_kind(Projection::Perspective.index());
    assert!((s.camera().focal - 14.0).abs() < 1e-6);
}
