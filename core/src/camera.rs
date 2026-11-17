//! The camera in a photographer's terms: a focal length on a sensor, an f-stop, a distance focused at,
//! and the size of the subject, which is what ties the scene's own units to millimetres.
//!
//! The renderer itself knows a camera distance, the tangent of half its field of view, the radius of its
//! lens opening and a band of depths that is in focus, all in the scene's units (`dist`, `fov`,
//! `aperture`, `fnear`, `ffar`). Those stay what is stored and what is drawn from. This file is the
//! photographer's names for the same things:
//!
//!   field of view   tan(fov / 2) = sensor / (2 x focal)          (`focal`, `sensor`, millimetres)
//!   lens opening    aperture = focal / (2 x fstop) / unit        (`fstop`; unit = millimetres a scene unit)
//!   the unit        subject / SPAN                               (`subject`: millimetres across the shape)
//!   focus           the middle of fnear..ffar, and its depth     (`focus`, `band`; `focusDist` in millimetres)
//!   exposure        exposure = 2 ^ ev                            (`ev`, stops)
//!
//! and, for sliders that should move by stops and not by millimetres, the same three on the scales a
//! photographer thinks in: `av` (the f-stop counted in stops: f/1 = 0, f/1.4 = 1, f/2 = 2 ...), `zoom`
//! (the focal length in doublings from 50 mm) and `subjectLog` (the subject's size as a power of ten).
//!
//! `focal` and `fstop` are kept when they are said (0 = not said) and `fov`, `aperture` are rewritten
//! from them; saying `fov` or `aperture` takes that quantity back to the old way. So a command line or
//! a saved setup that never used a photographer's name is never converted, and draws exactly as before.
//! The rest are views: they are worked out from, and written into, the stored numbers.
//!
//! The lens is described by its effect. `focal` gives the field of view the picture really has, and
//! `fstop` the opening the blur really comes from: what photographers call the effective focal length
//! and f-number. For a subject further off than ten focal lengths they are the numbers engraved on the
//! lens to within a tenth; at close range a real lens marked F, N and focused to a magnification m
//! behaves as F (1 + m), N (1 + m) (`Readout::engraved_*` gives the numbers the other way round).
//!
//! Besides that, this is where a point of the picture becomes a ray and a point in space becomes a point
//! of the picture, for every kind of lens (`Projection`) and for a lens that bows straight lines
//! (`distort`). The shaders do the same arithmetic (shaders/camera.wgsl); the tests hold the two together.

use std::f64::consts::{FRAC_PI_2, PI};

use crate::anim::js_round;
use crate::math::DEG;
use crate::settings::Settings;
use crate::state::State;

/// Scene units that `subject` millimetres stand for: the width of a shape fitted the usual way.
pub const SPAN: f64 = 2.0 * crate::FIT;
/// The object fits in a sphere of this radius about the origin (trace.wgsl: BOUND).
pub const BOUND: f64 = 1.47;
/// The corner of a 3:2 picture, in half-heights from its middle, squared: where `distort` is measured.
const CORNER2: f64 = 3.25;
/// `distort` beyond this either way is not drawn.
pub const DISTORT_MAX: f64 = 0.4;
/// A barrel lens's picture ends where the angle `a x radius` reaches this (its tangent runs away beyond).
pub const WARP_MAX: f64 = 1.5;
/// A blur this share of the picture's height across still counts as sharp (0.03 mm on full frame).
pub const COC: f64 = 1.0 / 800.0;
/// The focus views keep the band's two planes on a grid this fine, so that moving the middle a thousand
/// times leaves the depth exactly what it was (sums of grid numbers this size are exact).
const GRID: f64 = 1.0 / 1048576.0;

/// What kind of picture the lens draws. The numbers are those of the `proj` setting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Projection {
    /// straight lines stay straight: an ordinary lens
    Perspective,
    /// no perspective at all: rays run parallel (a telecentric lens)
    Orthographic,
    /// fisheye: distance from the middle of the picture goes as the angle off the axis
    Equidistant,
    /// fisheye: equal areas of the picture hold equal shares of the sphere (what most fisheye lenses do)
    Equisolid,
    /// fisheye: small shapes keep their shape everywhere (circles stay circles)
    Stereographic,
    /// the whole sphere unrolled: 360 degrees across, 180 up
    Panorama,
    /// angle across, perspective up and down: a wide view without the stretch at the sides
    Cylindrical,
}

impl Projection {
    pub const ALL: [Projection; 7] = [
        Projection::Perspective, Projection::Orthographic, Projection::Equidistant, Projection::Equisolid,
        Projection::Stereographic, Projection::Panorama, Projection::Cylindrical,
    ];
    /// From the `proj` setting.
    pub fn of(v: f64) -> Self {
        let i = if v.is_finite() { js_round(v).clamp(0.0, 6.0) as usize } else { 0 };
        Self::ALL[i]
    }
    pub fn index(self) -> usize {
        Self::ALL.iter().position(|p| *p == self).unwrap_or(0)
    }
    /// The word for it on the command line.
    pub fn id(self) -> &'static str {
        ["perspective", "orthographic", "fisheye", "equisolid", "stereographic", "panorama", "cylindrical"][self.index()]
    }
    /// What the window calls it.
    pub fn name(self) -> &'static str {
        ["perspective", "orthographic", "fisheye · equidistant", "fisheye · equisolid", "fisheye · stereographic", "panorama 360 × 180", "cylindrical"][self.index()]
    }
    pub fn from_id(s: &str) -> Option<Self> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "perspective" | "rectilinear" => Projection::Perspective,
            "orthographic" | "ortho" | "telecentric" => Projection::Orthographic,
            "fisheye" | "equidistant" => Projection::Equidistant,
            "equisolid" => Projection::Equisolid,
            "stereographic" => Projection::Stereographic,
            "panorama" | "pano" | "equirectangular" | "equirect" | "360" => Projection::Panorama,
            "cylindrical" | "cylinder" => Projection::Cylindrical,
            _ => return None,
        })
    }
    pub fn fisheye(self) -> bool {
        matches!(self, Projection::Equidistant | Projection::Equisolid | Projection::Stereographic)
    }
    /// Focus is measured as distance from the lens (a sphere is sharp), not as depth along the axis
    /// (a plane is sharp): the lenses that see more than half the world have no plane in front of them.
    pub fn radial(self) -> bool {
        self.fisheye() || matches!(self, Projection::Panorama | Projection::Cylindrical)
    }
    /// The radius in the picture, in focal lengths, of a ray this far (radians) off the axis.
    fn radius(self, theta: f64) -> f64 {
        match self {
            Projection::Equisolid => 2.0 * (theta / 2.0).sin(),
            Projection::Stereographic => 2.0 * (theta / 2.0).tan(),
            _ => theta,
        }
    }
    /// The other way. None beyond the edge of the lens's picture.
    fn angle(self, rho: f64) -> Option<f64> {
        match self {
            Projection::Equisolid => (rho <= 2.0).then(|| 2.0 * (rho / 2.0).asin()),
            Projection::Stereographic => Some(2.0 * (rho / 2.0).atan()),
            _ => (rho <= PI).then_some(rho),
        }
    }
}

/// Focal length (mm) that gives this field of view (degrees, top to bottom) on a sensor this high (mm).
pub fn focal_of(fov: f64, sensor: f64) -> f64 {
    sensor / (2.0 * (fov * DEG / 2.0).tan())
}
/// The field of view (degrees, top to bottom) of this focal length on a sensor this high.
pub fn fov_of(focal: f64, sensor: f64) -> f64 {
    2.0 * (sensor / (2.0 * focal)).atan() / DEG
}
/// Millimetres to a scene unit.
pub fn unit_mm(p: &Settings) -> f64 {
    (if p.subject > 0.0 && p.subject.is_finite() { p.subject } else { 100.0 }) / SPAN
}
fn sensor_mm(p: &Settings) -> f64 {
    if p.sensor > 0.0 && p.sensor.is_finite() { p.sensor } else { 24.0 }
}
/// The focal length in force, whichever way the lens was given.
pub fn focal_mm(p: &Settings) -> f64 {
    if p.focal > 0.0 { p.focal } else { focal_of(p.fov, sensor_mm(p)) }
}

/// Bring `fov` and `aperture` up to date with `focal` and `fstop` where those are said. Changes nothing
/// otherwise, and nothing the second time.
pub fn sync(p: &mut Settings) {
    if p.focal > 0.0 { p.fov = fov_of(p.focal, sensor_mm(p)); }
    if p.fstop > 0.0 { p.aperture = focal_mm(p) / (2.0 * p.fstop) / unit_mm(p); }
}

/// The names this file answers to besides the stored settings.
pub const VIEWS: [&str; 9] = ["focus", "band", "focusDist", "ev", "shiftX", "shiftY", "av", "zoom", "subjectLog"];
/// `av` when there is no lens opening at all: the far end of the ring.
const AV_SHUT: f64 = 12.0;

/// A camera value by name, as it is in force: the views, and `focal` and `fstop` whichever way the lens
/// was given (`fstop` 0 = no lens opening: everything sharp). None for any other name.
pub fn get(p: &Settings, name: &str) -> Option<f64> {
    Some(match name {
        "focal" => focal_mm(p),
        "fstop" => {
            if p.fstop > 0.0 { p.fstop } else if p.aperture > 0.0 { focal_mm(p) / (2.0 * p.aperture * unit_mm(p)) } else { 0.0 }
        }
        "focus" => 0.5 * (p.fnear + p.ffar),
        "band" => (p.ffar - p.fnear).abs(),
        "focusDist" => (p.dist + 0.5 * (p.fnear + p.ffar)) * unit_mm(p),
        "ev" => p.exposure.max(1e-9).log2(),
        "shiftX" => p.shift[0],
        "shiftY" => p.shift[1],
        "av" => match get(p, "fstop") {
            Some(n) if n > 0.0 => 2.0 * n.log2(),
            _ => AV_SHUT,
        },
        "zoom" => (focal_mm(p) / 50.0).log2(),
        "subjectLog" => (unit_mm(p) * SPAN).log10(),
        _ => return None,
    })
}

/// The band put at this middle with this depth, both on the grid.
fn set_band(p: &mut Settings, middle: f64, depth: f64) {
    let half = (depth.max(0.0) / GRID).round() * GRID * 0.5;
    let mid = (middle / (GRID * 0.5)).round() * GRID * 0.5;
    p.fnear = mid - half;
    p.ffar = mid + half;
}

/// Set a camera value by name: a view, a photographer's name, or one of the old names it stands
/// beside. Whatever follows from it is brought up to date. False if the name is not the camera's
/// (the caller then sets it as any other setting).
pub fn set(p: &mut Settings, name: &str, v: f64) -> bool {
    if !v.is_finite() { return false; }
    match name {
        "focal" => p.focal = v.max(0.0),
        // the old names take their quantity back to the old way
        "fov" => { p.focal = 0.0; p.fov = v; }
        "fstop" => p.fstop = v.max(0.0),
        "aperture" => { p.fstop = 0.0; p.aperture = v; }
        // a lens moved to another sensor is the same lens: the field of view is what changes
        "sensor" => {
            if v <= 0.0 { return true; }
            if p.focal <= 0.0 { p.focal = focal_mm(p); }
            p.sensor = v;
        }
        "subject" => { if v > 0.0 { p.subject = v; } }
        "proj" => p.proj = Projection::of(v).index() as f64,
        "focus" => { let d = (p.ffar - p.fnear).abs(); set_band(p, v, d); }
        "band" => { let m = 0.5 * (p.fnear + p.ffar); set_band(p, m, v); }
        "focusDist" => { let d = (p.ffar - p.fnear).abs(); let m = v / unit_mm(p) - p.dist; set_band(p, m, d); }
        "ev" => p.exposure = v.exp2(),
        "shiftX" => p.shift[0] = v,
        "shiftY" => p.shift[1] = v,
        "av" => p.fstop = (v / 2.0).exp2(),
        "zoom" => p.focal = 50.0 * v.exp2(),
        "subjectLog" => p.subject = 10f64.powf(v),
        _ => return false,
    }
    sync(p);
    // (the fields are written directly above: hold them all to their hard limits, `limits`)
    p.hold_hard();
    true
}

/// The same from text: a number, or for `proj` also the projection's word.
pub fn set_text(p: &mut Settings, name: &str, text: &str) -> bool {
    if name == "proj" {
        if let Some(pr) = Projection::from_id(text) { return set(p, name, pr.index() as f64); }
    }
    match text.trim().parse::<f64>() {
        Ok(v) if v.is_finite() => set(p, name, v),
        _ => false,
    }
}

/// The camera as the settings come to: everything the renderer draws from, in the scene's units, with
/// the photographer's numbers beside it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub proj: Projection,
    /// the lens sits at (eye[0], eye[1], dist) and looks down -z: at the object's centre while `eye` is 0
    pub dist: f64,
    /// how far the lens stands across and up from the line through the object's centre (a free camera)
    pub eye: [f64; 2],
    /// Half the picture's height on the sensor, in focal lengths: the tangent of half the field of view of
    /// an ordinary lens. For a panorama, where the picture's height is half a turn whatever the lens: pi / 2.
    pub t: f64,
    /// the picture slid across the lens's image, in half-heights (a shift lens)
    pub shift: [f64; 2],
    /// how strongly straight lines are bowed: above 0 outward (barrel), below 0 inward (pincushion)
    pub warp: f64,
    /// radius of the lens opening, scene units (0 = a pinhole: everything sharp)
    pub lens: f64,
    /// the band that is in focus: depth of its two faces from the object's centre
    pub near: f64,
    pub far: f64,
    /// the plane of focus's slope: depth gained per unit across, and per unit up
    pub tilt: [f64; 2],
    pub focal: f64,
    pub sensor: f64,
    /// millimetres to a scene unit
    pub unit: f64,
}

/// Which way the plane of focus leans for a lens that focuses by depth.
fn tilt_of(p: &Settings) -> [f64; 2] {
    let slope = |deg: f64| (deg.clamp(-80.0, 80.0) * DEG).tan();
    [slope(p.tilt_y), slope(p.tilt_x)]
}

fn norm(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / l, v[1] / l, v[2] / l]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

impl Camera {
    pub fn of(p: &Settings) -> Camera {
        let proj = Projection::of(p.proj);
        let d = if p.distort.is_finite() { p.distort.clamp(-DISTORT_MAX, DISTORT_MAX) } else { 0.0 };
        Camera {
            proj,
            dist: p.dist,
            eye: [p.eye_x, p.eye_y],
            t: if proj == Projection::Panorama { FRAC_PI_2 } else { (p.fov * DEG / 2.0).tan() },
            shift: p.shift,
            // (a lens that bows lines by d at the corner: radius -> tan(a r) / a = r (1 + a^2 r^2 / 3 ...))
            warp: if proj == Projection::Panorama { 0.0 } else { (3.0 * d.abs() / CORNER2).sqrt() * d.signum() },
            lens: p.aperture.max(0.0),
            near: p.fnear.min(p.ffar),
            far: p.fnear.max(p.ffar),
            tilt: if proj.radial() { [0.0; 2] } else { tilt_of(p) },
            focal: focal_mm(p),
            sensor: sensor_mm(p),
            unit: unit_mm(p),
        }
    }

    /// An ordinary lens that keeps straight lines straight, with a level plane of focus: the camera the
    /// program always had, which the shaders draw without any of this file's arithmetic.
    pub fn plain(&self) -> bool {
        self.proj == Projection::Perspective && self.warp == 0.0 && self.tilt == [0.0; 2]
    }

    /// Half the height of an orthographic view, in scene units: what a perspective lens at the same
    /// distance shows of the plane through the object's centre.
    pub fn half_height(&self) -> f64 {
        // (a free camera may stand level with the object's centre, or beyond it)
        self.t * self.dist.abs().max(1e-3)
    }

    /// Where the lens stands.
    pub fn at(&self) -> [f64; 3] {
        [self.eye[0], self.eye[1], self.dist]
    }

    /// A radius on the film (half-heights from the lens's axis) to where a lens that kept straight lines
    /// straight would have put it. A barrel lens's picture has an edge: the radius is held inside it.
    fn warp_out(&self, r: f64) -> f64 {
        let a = self.warp;
        if a > 0.0 { (a * r).min(WARP_MAX).tan() / a } else if a < 0.0 { (-a * r).atan() / -a } else { r }
    }
    /// The other way. None for a radius the lens cannot show.
    fn warp_in(&self, r: f64) -> Option<f64> {
        let a = self.warp;
        if a > 0.0 {
            Some((a * r).atan() / a)
        } else if a < 0.0 {
            (-a * r < WARP_MAX).then(|| (-a * r).tan() / -a)
        } else {
            Some(r)
        }
    }

    /// The ray through a point of the picture (across and up from its middle, in half-heights): where it
    /// starts and which way it goes. None where the lens draws nothing (outside a fisheye's circle).
    pub fn ray(&self, uv: [f64; 2]) -> Option<([f64; 3], [f64; 3])> {
        let (fx, fy) = (uv[0] + self.shift[0], uv[1] + self.shift[1]);
        let r = fx.hypot(fy);
        let k = if r > 0.0 { self.warp_out(r) / r } else { 1.0 };
        let (x, y) = (fx * k, fy * k);
        let at = self.at();
        let t = self.t;
        Some(match self.proj {
            Projection::Perspective => (at, norm([x * t, y * t, -1.0])),
            Projection::Orthographic => ([x * self.half_height() + at[0], y * self.half_height() + at[1], self.dist], [0.0, 0.0, -1.0]),
            Projection::Equidistant | Projection::Equisolid | Projection::Stereographic => {
                let rho = x.hypot(y) * t;
                let th = self.proj.angle(rho)?;
                if rho <= 0.0 { return Some((at, [0.0, 0.0, -1.0])); }
                let s = th.sin() * t / rho;
                (at, [x * s, y * s, -th.cos()])
            }
            Projection::Panorama => {
                let (lam, phi) = (x * t, y * t);
                if phi.abs() > FRAC_PI_2 { return None; }
                (at, [phi.cos() * lam.sin(), phi.sin(), -phi.cos() * lam.cos()])
            }
            Projection::Cylindrical => {
                let lam = x * t;
                (at, norm([lam.sin(), y * t, -lam.cos()]))
            }
        })
    }

    /// The point of the picture at which a point in space appears (across and up from the picture's
    /// middle, in half-heights). None if the lens cannot see it.
    pub fn project(&self, p: [f64; 3]) -> Option<[f64; 2]> {
        let v = [p[0] - self.eye[0], p[1] - self.eye[1], p[2] - self.dist];
        let t = self.t;
        let (x, y) = match self.proj {
            Projection::Perspective => {
                if v[2] > -1e-9 { return None; }
                (v[0] / (-v[2] * t), v[1] / (-v[2] * t))
            }
            Projection::Orthographic => {
                if v[2] > 0.0 { return None; }
                (v[0] / self.half_height(), v[1] / self.half_height())
            }
            Projection::Equidistant | Projection::Equisolid | Projection::Stereographic => {
                let r = v[0].hypot(v[1]);
                if r <= 0.0 {
                    if v[2] < 0.0 { (0.0, 0.0) } else { return None; }
                } else {
                    let rho = self.proj.radius(r.atan2(-v[2]));
                    (v[0] / r * rho / t, v[1] / r * rho / t)
                }
            }
            Projection::Panorama => (v[0].atan2(-v[2]) / t, v[1].atan2(v[0].hypot(v[2])) / t),
            Projection::Cylindrical => (v[0].atan2(-v[2]) / t, v[1] / (v[0].hypot(v[2]) * t)),
        };
        let r = x.hypot(y);
        let k = if r > 0.0 { self.warp_in(r)? / r } else { 1.0 };
        Some([x * k - self.shift[0], y * k - self.shift[1]])
    }

    /// How far along its own ray a point is: the distance the tracer stores with a pixel.
    pub fn along(&self, p: [f64; 3]) -> f64 {
        if self.proj == Projection::Orthographic {
            self.dist - p[2]
        } else {
            ((p[0] - self.eye[0]).powi(2) + (p[1] - self.eye[1]).powi(2) + (p[2] - self.dist).powi(2)).sqrt()
        }
    }

    /// The point `t` along the ray through a point of the picture.
    pub fn point(&self, uv: [f64; 2], t: f64) -> Option<[f64; 3]> {
        let (o, d) = self.ray(uv)?;
        Some([o[0] + d[0] * t, o[1] + d[1] * t, o[2] + d[2] * t])
    }

    /// How deep a point is, the way focus measures it (what `fnear`, `ffar` and `focus` are compared
    /// with): for a lens that focuses on a plane, its depth from the object's centre along the axis,
    /// less what the plane of focus has gained by its slope where the point is; for one that focuses
    /// on a sphere, its distance from the lens less the camera's.
    pub fn depth_of(&self, p: [f64; 3]) -> f64 {
        if self.proj.radial() {
            self.along(p) - self.dist
        } else {
            -p[2] - (self.tilt[0] * p[0] + self.tilt[1] * p[1])
        }
    }
    /// The same for glass found `t` along the ray through a point of the picture.
    pub fn depth_at(&self, uv: [f64; 2], t: f64) -> Option<f64> {
        self.point(uv, t).map(|p| self.depth_of(p))
    }

    /// How far along the ray through a point of the picture the surface of depth `e` is met (the plane of
    /// focus, when `e` is where the lens is focused). None if the ray never meets it.
    pub fn reach(&self, uv: [f64; 2], e: f64) -> Option<f64> {
        let (o, d) = self.ray(uv)?;
        if self.proj.radial() { return Some((self.dist + e).max(1e-6)); }
        // -(o.z + d.z s) - tilt . (o.xy + d.xy s) = e
        let slope = -d[2] - (self.tilt[0] * d[0] + self.tilt[1] * d[1]);
        let s = (e + o[2] + self.tilt[0] * o[0] + self.tilt[1] * o[1]) / slope;
        (slope > 1e-9 && s > 0.0).then_some(s)
    }

    /// One pixel of a picture `h` pixels high, as the blur is measured against it: an angle for a lens
    /// with a centre, a length in the scene for an orthographic one.
    pub fn pixel(&self, h: f64) -> f64 {
        2.0 * (if self.proj == Projection::Orthographic { self.half_height() } else { self.t }) / h
    }

    /// The distance at which an orthographic lens's cone of rays is as wide as the perspective lens's
    /// would be: the middle of the band.
    fn cone_at(&self) -> f64 {
        (self.dist + 0.5 * (self.near + self.far)).max(1e-6)
    }

    /// The radius of the blur, in pixels of a picture `h` high, of glass `t` along the ray through a
    /// point of the picture: negative in front of the band in focus, nothing inside it, positive behind.
    /// A lens of radius A focused at distance f spreads a point at distance z over a disc that is
    /// A |1/f - 1/z| across as an angle (an orthographic one, whose rays come in at A / f to the axis
    /// whatever the distance: A |z - f| / f as a length).
    pub fn blur(&self, uv: [f64; 2], t: f64, h: f64) -> Option<f64> {
        if self.lens <= 0.0 { return Some(0.0); }
        let e = self.depth_at(uv, t)?;
        let f = self.reach(uv, e.clamp(self.near, self.far))?;
        Some(if self.proj == Projection::Orthographic {
            self.lens / self.cone_at() * (t - f) / self.pixel(h)
        } else {
            // (for a lens that focuses by depth, distances along the ray and depths along the axis are
            // in the same proportion on any one ray, so the two reciprocals may be taken either way
            // as long as the lens's radius is measured across the same direction: across the axis)
            let (_, d) = self.ray(uv)?;
            let c = if self.proj.radial() { 1.0 } else { -d[2] };
            self.lens * (1.0 / (f * c) - 1.0 / (t * c)) / self.pixel(h)
        })
    }

    /// The ray the tracer follows when it traces through the lens: from the point `l` of the lens
    /// opening (a point of the unit disc) towards where the ray through the picture's point `uv` meets
    /// the surface in focus. `t`: how far along that central ray the glass was found; the lens is
    /// focused at that depth, held inside the band.
    pub fn lens_ray(&self, uv: [f64; 2], t: f64, l: [f64; 2]) -> Option<([f64; 3], [f64; 3])> {
        let (o, d) = self.ray(uv)?;
        let e = self.depth_at(uv, t)?.clamp(self.near, self.far);
        let s = self.reach(uv, e)?;
        let f = [o[0] + d[0] * s, o[1] + d[1] * s, o[2] + d[2] * s];
        let from = match self.proj {
            // the cone is as wide at the surface in focus as a perspective lens's would be at `cone_at`
            Projection::Orthographic => {
                let k = self.lens * s / self.cone_at();
                [o[0] + l[0] * k, o[1] + l[1] * k, o[2]]
            }
            // a lens that sees all round has no flat front: its opening faces along each ray
            _ if self.proj.radial() => {
                let mut b1 = cross(d, [0.0, 1.0, 0.0]);
                if b1[0] * b1[0] + b1[1] * b1[1] + b1[2] * b1[2] < 1e-12 { b1 = [1.0, 0.0, 0.0]; }
                let b1 = norm(b1);
                let b2 = cross(b1, d);
                let a = self.lens;
                [o[0] + (b1[0] * l[0] + b2[0] * l[1]) * a, o[1] + (b1[1] * l[0] + b2[1] * l[1]) * a, o[2] + (b1[2] * l[0] + b2[2] * l[1]) * a]
            }
            _ => [o[0] + l[0] * self.lens, o[1] + l[1] * self.lens, o[2]],
        };
        Some((from, norm([f[0] - from[0], f[1] - from[1], f[2] - from[2]])))
    }

    /// How much light a point of the picture gets against its middle, by the lens's own law: less of the
    /// world is packed into each piece of film away from the axis, and an ordinary lens's opening is seen
    /// aslant from there. An ordinary lens: the fourth power of the cosine of the angle off the axis. A
    /// fisheye, whose opening turns to face the light: only how the world is packed, which for the
    /// equidistant one is sin(angle) / angle, for the equisolid one nothing at all (that is what
    /// "equal areas" means), for the stereographic one cos^4 of half the angle. A cylindrical lens:
    /// the cube of the cosine of the angle above the level. Orthographic and panorama: none.
    pub fn falloff(&self, uv: [f64; 2]) -> f64 {
        let Some((_, d)) = self.ray(uv) else { return 0.0 };
        let cosine = (-d[2]).clamp(-1.0, 1.0);
        match self.proj {
            Projection::Perspective => cosine.powi(4),
            Projection::Equidistant => {
                let th = cosine.acos();
                if th < 1e-3 { 1.0 } else { th.sin() / th }
            }
            Projection::Stereographic => (0.5 + 0.5 * cosine).powi(2),
            Projection::Cylindrical => d[0].hypot(d[2]).powi(3),
            _ => 1.0,
        }
    }

    /// What the shaders are told (shaders/camera.wgsl: Cam). The last three are the shape of the
    /// out-of-focus highlights: anamorphic squeeze, bright rim or core, cat's-eye.
    pub fn uniform(&self, squeeze: f64, bokeh: f64, catseye: f64) -> [[f32; 4]; 4] {
        let ortho = self.proj == Projection::Orthographic;
        [
            [self.proj.index() as f32, self.t as f32, self.dist as f32, self.half_height() as f32],
            // (a pixel's size times the picture's height: twice the tangent, for the plain lens)
            [self.shift[0] as f32, self.shift[1] as f32, self.warp as f32, self.pixel(1.0) as f32],
            // what a pixel covers `t` along its ray: (x t + y) pixels' worth; then the plane of focus's slope
            [if ortho { 0.0 } else { 1.0 }, if ortho { 1.0 } else { 0.0 }, self.tilt[0] as f32, self.tilt[1] as f32],
            [squeeze.max(1.0) as f32, bokeh.clamp(-1.0, 1.0) as f32, catseye.clamp(0.0, 1.0) as f32, self.cone_at() as f32],
        ]
    }

    // ---- the photographer's numbers ------------------------------------------------------------

    /// The f-number in force. None with no lens opening (everything sharp).
    pub fn fstop(&self) -> Option<f64> {
        (self.lens > 0.0).then(|| self.focal / (2.0 * self.lens * self.unit))
    }
    /// Distance from the lens to the middle of the band in focus, millimetres.
    pub fn focus_mm(&self) -> f64 {
        (self.dist + 0.5 * (self.near + self.far)) * self.unit
    }
    /// How large the subject in focus is on the sensor, against its real size.
    pub fn magnification(&self) -> f64 {
        self.focal / self.focus_mm().max(1e-9)
    }
    /// Half the angle (radians) the picture really takes in from its middle to its top edge, to its
    /// side edge and to its corner, for a picture `aspect` times wider than high. Nothing for an
    /// orthographic lens, whose rays all run the same way.
    pub fn half_field(&self, aspect: f64) -> [f64; 3] {
        if self.proj == Projection::Orthographic { return [0.0; 3]; }
        let plain = Camera { shift: [0.0; 2], ..*self };
        // (beyond the edge of the lens's picture: the whole of what it can show)
        let off_axis = |uv: [f64; 2]| plain.ray(uv).map_or(PI, |(_, d)| d[0].hypot(d[1]).atan2(-d[2]));
        // a lens that unrolls the view sideways can go more than once round: one whole turn is all there is
        let side = if matches!(self.proj, Projection::Panorama | Projection::Cylindrical) { (self.t * aspect).min(PI) } else { off_axis([aspect, 0.0]) };
        let corner = if self.proj == Projection::Panorama && self.t * aspect >= PI { PI } else { off_axis([aspect, 1.0]) };
        [off_axis([0.0, 1.0]), side, corner]
    }

    /// Everything a photographer would want to read off the camera, for a picture `aspect` times wider
    /// than high. Lengths in millimetres.
    pub fn readout(&self, aspect: f64) -> Readout {
        let fstop = self.fstop();
        let focus = self.focus_mm();
        let m = self.magnification();
        // A blur COC of the picture's height across on the sensor: F D |1/s - 1/z| = c, so what is sharp
        // lies between 1 / (1/s + c N / F^2) and 1 / (1/s - c N / F^2); with a band, s is its nearer
        // face for the first and its farther face for the second.
        let c = COC * self.sensor;
        let (mut from, mut to, mut hyper) = (0.0, f64::INFINITY, None);
        if let Some(n) = fstop {
            let k = c * n / (self.focal * self.focal);
            let (sn, sf) = (((self.dist + self.near) * self.unit).max(1e-9), ((self.dist + self.far) * self.unit).max(1e-9));
            from = 1.0 / (1.0 / sn + k);
            to = if 1.0 / sf > k { 1.0 / (1.0 / sf - k) } else { f64::INFINITY };
            hyper = Some(1.0 / k);
        }
        let field = self.half_field(aspect);
        let mut notes = Vec::new();
        let rectilinear = matches!(self.proj, Projection::Perspective | Projection::Cylindrical);
        if let Some(n) = fstop {
            if n < 0.5 {
                notes.push(format!("f/{} is past what any lens in air can do (f/0.5); drawn all the same", stop_text(n)));
            } else if n < 0.7 {
                notes.push(format!("f/{} is faster than any camera lens made (about f/0.7)", stop_text(n)));
            }
            // the smallest spot a lens can draw is about 1.34 micrometres times its f-number across
            if 0.00134 * n > c {
                notes.push(format!("at f/{} a real lens is softened all over by diffraction (not drawn)", stop_text(n)));
            }
            if 2.0 * self.lens > self.dist + self.near && self.dist + self.near > 0.0 {
                notes.push("the lens opening is wider than the subject is far: closer to a microscope than a camera".into());
            }
        }
        if rectilinear && self.focal / self.sensor < 0.375 {
            notes.push(format!("{} on this sensor is wider than any lens that keeps lines straight ({} at the widest): a fisheye would be real", focal_text(self.focal), focal_text(0.375 * self.sensor)));
        }
        if self.proj != Projection::Panorama && self.focal / self.sensor > 50.0 {
            notes.push(format!("{} on this sensor is a telescope, not a camera lens", focal_text(self.focal)));
        }
        if self.proj != Projection::Orthographic && m > 5.0 {
            notes.push(format!("{}× life size is beyond a camera's macro lenses: this is a microscope's view", round_to(m, 2)));
        }
        Readout {
            focal: self.focal, sensor: self.sensor, fstop,
            field_v: 2.0 * field[0] / DEG, field_h: 2.0 * field[1] / DEG, field_d: 2.0 * field[2] / DEG,
            view_mm: (self.proj == Projection::Orthographic).then(|| 2.0 * self.half_height() * self.unit),
            distance_mm: self.dist * self.unit,
            focus_mm: focus,
            band_mm: (self.far - self.near) * self.unit,
            magnification: m,
            sharp_from_mm: from, sharp_to_mm: to, hyperfocal_mm: hyper,
            engraved_focal: self.focal / (1.0 + m), engraved_fstop: fstop.map(|n| n / (1.0 + m)),
            notes,
        }
    }
}

/// What a photographer reads off the camera (`Camera::readout`). Lengths in millimetres, angles in degrees.
#[derive(Clone, Debug, PartialEq)]
pub struct Readout {
    pub focal: f64,
    pub sensor: f64,
    /// None = no lens opening: everything is sharp
    pub fstop: Option<f64>,
    /// what the picture really takes in: top to bottom, side to side, corner to corner
    pub field_v: f64,
    pub field_h: f64,
    pub field_d: f64,
    /// an orthographic view has no angle: the height of what it shows instead
    pub view_mm: Option<f64>,
    /// from the lens to the object's centre
    pub distance_mm: f64,
    /// from the lens to the middle of the band in focus, and the band's depth
    pub focus_mm: f64,
    pub band_mm: f64,
    /// the size of the subject in focus on the sensor against its real size
    pub magnification: f64,
    /// distances from the lens between which a real picture would pass as sharp (infinity = to the horizon)
    pub sharp_from_mm: f64,
    pub sharp_to_mm: f64,
    /// focused here, everything from half this distance to the horizon passes as sharp
    pub hyperfocal_mm: Option<f64>,
    /// what a real lens focused this close would have engraved on it to behave like this
    pub engraved_focal: f64,
    pub engraved_fstop: Option<f64>,
    /// where the numbers are ones no real lens has: said, never refused
    pub notes: Vec<String>,
}

fn round_to(v: f64, digits: i32) -> f64 {
    if v == 0.0 || !v.is_finite() { return v; }
    let k = 10f64.powi(digits - 1 - v.abs().log10().floor() as i32);
    (v * k).round() / k
}

/// A length for reading: "8.0 mm", "25.4 cm", "1.20 m", "to the horizon".
pub fn length_text(mm: f64) -> String {
    if !mm.is_finite() { return "∞".into(); }
    let a = mm.abs();
    if a < 10.0 { format!("{mm:.2} mm") } else if a < 100.0 { format!("{mm:.1} mm") } else if a < 1000.0 { format!("{:.1} cm", mm / 10.0) } else if a < 100000.0 { format!("{:.2} m", mm / 1000.0) } else { format!("{:.0} m", mm / 1000.0) }
}
/// A focal length for reading: "50 mm", "7.6 mm".
pub fn focal_text(mm: f64) -> String {
    if mm < 20.0 { format!("{:.1} mm", mm) } else { format!("{:.0} mm", mm) }
}
/// An f-number as it is written after "f/": "2.8", "11", "0.95".
pub fn stop_text(n: f64) -> String {
    if n < 1.0 { format!("{n:.2}") } else if n < 10.0 { format!("{n:.1}") } else { format!("{n:.0}") }
}
/// The full stops of a lens's ring, and the thirds between them, for a slider that moves by stops.
pub const STOPS: [f64; 14] = [0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0, 32.0, 45.0, 64.0];
/// The f-number `thirds` thirds of a stop along the ring from f/1 (negative = faster).
pub fn stop_at(thirds: f64) -> f64 {
    2f64.powf(thirds / 6.0)
}
/// The other way.
pub fn thirds_of(n: f64) -> f64 {
    6.0 * n.max(1e-9).log2()
}

/// A lens a photographer knows by name.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lens {
    pub name: &'static str,
    pub focal: f64,
    pub proj: Projection,
    /// its widest stop
    pub fastest: f64,
}
pub const LENSES: [Lens; 8] = [
    Lens { name: "24 mm wide", focal: 24.0, proj: Projection::Perspective, fastest: 1.4 },
    Lens { name: "35 mm", focal: 35.0, proj: Projection::Perspective, fastest: 1.4 },
    Lens { name: "50 mm normal", focal: 50.0, proj: Projection::Perspective, fastest: 1.2 },
    Lens { name: "85 mm portrait", focal: 85.0, proj: Projection::Perspective, fastest: 1.4 },
    Lens { name: "100 mm macro", focal: 100.0, proj: Projection::Perspective, fastest: 2.8 },
    Lens { name: "200 mm tele", focal: 200.0, proj: Projection::Perspective, fastest: 2.8 },
    Lens { name: "15 mm fisheye", focal: 15.0, proj: Projection::Equisolid, fastest: 2.8 },
    Lens { name: "8 mm circular fisheye", focal: 8.0, proj: Projection::Equidistant, fastest: 3.5 },
];
/// Sensor and film formats: name, width, height in millimetres (the picture is as tall as the sensor).
pub const SENSORS: [(&str, f64, f64); 9] = [
    ("1-inch", 13.2, 8.8),
    ("Micro Four Thirds", 17.3, 13.0),
    ("APS-C", 23.6, 15.7),
    ("Super 35", 24.9, 18.7),
    ("full frame (35 mm)", 36.0, 24.0),
    ("medium format 44 × 33", 43.8, 32.9),
    ("6 × 6 film", 56.0, 56.0),
    ("IMAX 15/70", 70.4, 52.6),
    ("4 × 5 inch sheet", 127.0, 102.0),
];
/// Subject sizes worth a button: name, millimetres across.
pub const SUBJECTS: [(&str, f64); 6] = [
    ("gem · 8 mm", 8.0),
    ("jewel · 30 mm", 30.0),
    ("tabletop · 10 cm", 100.0),
    ("vase · 40 cm", 400.0),
    ("sculpture · 2 m", 2000.0),
    ("monument · 20 m", 20000.0),
];

/// Towards the corners of the picture a real lens's barrel cuts into its opening from both sides, so
/// that an out-of-focus highlight there is not a disc but a cat's eye. The cut for a point of the film
/// (across and up from the lens's axis, in half-heights) with `k` of cat's-eye: the opening, a unit disc,
/// is what lies within one radius of both +c and -c, where c is this. Each side comes in by half the
/// opening's radius at the corner of a 3:2 picture with k = 1.
pub fn cut(film: [f64; 2], k: f64) -> [f64; 2] {
    let s = 0.5 * k / CORNER2.sqrt();
    [film[0] * s, film[1] * s]
}
/// The share of the opening such a cut leaves: the area two unit discs 2|c| apart have in common, over
/// a disc's.
pub fn open_share(c: [f64; 2]) -> f64 {
    let x = c[0].hypot(c[1]).min(1.0);
    (2.0 * x.acos() - 2.0 * x * (1.0 - x * x).sqrt()) / PI
}

/// A video's shutter (the share of each frame's time it is open) as a film camera's shutter angle, and back.
pub fn shutter_angle(share: f64) -> f64 {
    360.0 * share
}
pub fn shutter_of_angle(degrees: f64) -> f64 {
    (degrees / 360.0).clamp(0.0, 1.0)
}
/// The same as an exposure time: "1/x of a second" at this many frames a second (0 = the shutter never opens).
pub fn shutter_speed(share: f64, fps: f64) -> f64 {
    if share > 0.0 { fps / share } else { 0.0 }
}
pub fn shutter_of_speed(one_over: f64, fps: f64) -> f64 {
    if one_over > 0.0 { (fps / one_over).clamp(0.0, 1.0) } else { 0.0 }
}

impl State {
    /// The camera as the settings stand.
    pub fn camera(&self) -> Camera {
        Camera::of(&self.p)
    }

    /// One of the camera's names with its value as text, as the command line gives them: any of the
    /// settings and views above, `proj` by number or by word, `dollyZoom=<mm>` (the lens changed and
    /// the camera moved so that the subject keeps its size), and a video's `shutterAngle`, `shutterSpeed`.
    /// False if the name is not the camera's.
    pub fn camera_arg(&mut self, name: &str, value: &str) -> bool {
        let number = value.trim().parse::<f64>().ok().filter(|v| v.is_finite());
        match (name, number) {
            ("dollyZoom", Some(v)) if v > 0.0 => { self.dolly_zoom(v); return true; }
            // a video's shutter as a film camera's angle, or as an exposure time (120 = 1/120 s)
            ("shutterAngle", Some(v)) => { self.render.shutter = shutter_of_angle(v); return true; }
            ("shutterSpeed", Some(v)) => { self.render.shutter = shutter_of_speed(v, self.render.fps as f64); return true; }
            ("dollyZoom" | "shutterAngle" | "shutterSpeed", _) => return false,
            _ => {}
        }
        let was = self.p.proj;
        let done = set_text(&mut self.p, name, value);
        // (a picture drawn through one kind of lens is no start for a picture through another)
        if self.p.proj != was { self.cut = true; }
        done
    }

    /// Bring what follows from the photographer's settings up to date (after settings were written
    /// some other way than by name: a setup read from a file).
    pub fn camera_sync(&mut self) {
        sync(&mut self.p);
    }

    /// Change the lens and move the camera so that the subject stays the size it is in the picture: what
    /// changes is the perspective. (An ordinary lens only; the others keep the camera where it is.)
    pub fn dolly_zoom(&mut self, focal: f64) {
        if !(focal > 0.0) || !focal.is_finite() { return; }
        let was = self.camera();
        set(&mut self.p, "focal", focal);
        if was.proj != Projection::Panorama && was.t > 0.0 {
            // the plane through the object's centre keeps its size: distance x tan(half the field)
            self.p.dist = (self.p.dist * was.t / self.camera().t).clamp(1.5, 400.0);
        }
        sync(&mut self.p);
    }

    /// Choose the kind of lens (a place in `Projection::ALL`). A fisheye or a cylindrical lens shows what it
    /// is only over a wide field: on a long focal length it draws much what an ordinary lens draws (at 50 mm a
    /// fisheye takes in 27 degrees, and bends nothing that can be seen). Chosen while the lens is long, it
    /// is given a short one of its kind, and the camera comes in part of the way, so that the subject is
    /// smaller but still the subject.
    pub fn lens_kind(&mut self, i: usize) {
        let was = self.camera();
        if !self.camera_arg("proj", &i.to_string()) { return; }
        let short = match self.camera().proj {
            Projection::Equidistant | Projection::Equisolid | Projection::Stereographic => Some(12.0),
            Projection::Cylindrical => Some(18.0),
            _ => None,
        };
        let Some(f) = short else { return };
        if was.focal <= 1.5 * f { return; }
        set(&mut self.p, "focal", f);
        self.p.dist = (self.p.dist * (f / was.focal).sqrt()).clamp(2.5f64.min(self.p.dist), 400.0);
        sync(&mut self.p);
    }

    /// Put one of the named lenses on, keeping the subject the size it is in the picture.
    pub fn lens_preset(&mut self, i: usize) {
        let Some(l) = LENSES.get(i) else { return };
        let was = Projection::of(self.p.proj);
        set(&mut self.p, "proj", l.proj.index() as f64);
        if l.proj == Projection::Perspective && was == Projection::Perspective {
            self.dolly_zoom(l.focal);
        } else {
            set(&mut self.p, "focal", l.focal);
        }
        if l.proj != was { self.cut = true; }
    }

    /// Open or close the lens. Closed, everything is sharp; opened, it is at this f-stop.
    pub fn set_depth_of_field(&mut self, fstop: Option<f64>) {
        match fstop {
            Some(n) if n > 0.0 => { set(&mut self.p, "fstop", n); }
            _ => { set(&mut self.p, "aperture", 0.0); }
        }
    }

    /// Focus on glass found `t` along the ray through a point of the picture (across and up from its
    /// middle, in half-heights): the band keeps its depth and is centred on that glass. Returns the
    /// depth focused at, or None if the lens draws nothing there.
    pub fn focus_on(&mut self, uv: [f64; 2], t: f64) -> Option<f64> {
        let e = self.camera().depth_at(uv, t)?;
        let band = (self.p.ffar - self.p.fnear).abs();
        self.p.fnear = e - band / 2.0;
        self.p.ffar = e + band / 2.0;
        self.flash_planes = true;
        Some(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_band_stays_as_deep_however_often_it_is_moved() {
        let mut p = Settings::default();
        p.fnear = -1.0;
        p.ffar = -0.6;
        set(&mut p, "focus", 0.1234567);
        let depth = p.ffar - p.fnear;
        assert!((depth - 0.4).abs() < 1e-6);
        for i in 0..20000 {
            set(&mut p, "focus", (i as f64 * 0.618).sin() * 1.3);
            assert_eq!(p.ffar - p.fnear, depth);
        }
        set(&mut p, "band", 0.0);
        assert_eq!(p.fnear, p.ffar);
        assert_eq!(get(&p, "band"), Some(0.0));
    }

    #[test]
    fn bowed_lines_go_out_and_come_back() {
        for d in [-0.4, -0.1, 0.05, 0.4] {
            let mut p = Settings::default();
            p.distort = d;
            let c = Camera::of(&p);
            for r in [0.01, 0.5, 1.0, 2.0] {
                let out = c.warp_out(r);
                assert!((c.warp_in(out).unwrap() - r).abs() < 1e-12, "{d} {r}");
                // barrel: the corners take in more than a lens with straight lines would; pincushion: less
                assert_eq!(out > r, d > 0.0);
            }
            // by `d` at the corner of a 3:2 picture, to the first order
            let r = 0.3;
            assert!(((c.warp_out(r) / r - 1.0) / (d * r * r / CORNER2) - 1.0).abs() < 0.05);
        }
    }

    #[test]
    fn lengths_read_naturally() {
        assert_eq!(length_text(8.0), "8.00 mm");
        assert_eq!(length_text(254.0), "25.4 cm");
        assert_eq!(length_text(1200.0), "1.20 m");
        assert_eq!(stop_text(2.8), "2.8");
        assert_eq!(stop_text(11.0), "11");
        assert!((stop_at(thirds_of(5.6)) - 5.6).abs() < 1e-12);
        assert!((stop_at(6.0) - 2.0).abs() < 1e-12);
    }
}
