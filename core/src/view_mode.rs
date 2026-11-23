//! How the viewport draws the scene: the full render, or a matcap for fast modelling.
//!
//! A matcap here is a viewing aid kept with the view, never part of a material: the surface is shaded
//! by a function of its normal as the camera sees it, at the first surface only. The built-in ones are
//! exact formulas of that normal (`shade`); the renderer's WGSL (app/src/shaders/matcap.wgsl) is the
//! same arithmetic in f32, and this f64 copy is what the tests hold it to. A user picture (a lit ball
//! filling a square) can stand in for them.

use serde::{Deserialize, Serialize};

/// What the viewport draws.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ViewMode {
    /// glass, materials, lights: the picture as it renders
    #[default]
    Render,
    /// the first surface shaded by a matcap: no bounces, no lighting work, immediate
    Matcap,
}

/// The built-in matcaps (the starter set), each a formula of the normal: a parameter block (`Look`) for
/// one analytic model (`shade_look`). The order is the number they are known by (`index`); new ones go at
/// the end.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Builtin {
    /// glossy grey: a key light high left, a fill low right, a rim
    #[default]
    Studio,
    /// the same lights larger and softer, lower contrast
    Soft,
    /// warm matte, wrapped light from above
    Clay,
    /// white glaze: diffuse body, two sharp highlights, a bright rim
    Ceramic,
    /// chrome: the reflected ray against a sky above and a ground below
    Metal,
    /// the normal itself as a colour (x red, y green, z blue)
    Normals,
    /// dark blue-grey with a broad sheen and a cool rim
    Satin,
    /// a warm metal ramp
    Gold,
    /// dark body lit from behind: a strong cool rim
    Rim,
    /// red, softly lit through, a glowing edge
    Wax,
    /// green stone with a glassy highlight
    Jade,
    /// deep blue under a clear coat: sharp highlights and a faint reflection
    Lacquer,
}

/// One built-in matcap's numbers (`shade_look`); linear colours, directions in the ray's frame (z towards
/// the camera, y up, x right).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Look {
    pub base: [f64; 3],
    pub ambient: f64,
    /// light from above on the body: 0.5 + 0.5 y
    pub sky: f64,
    pub key: [f64; 3],
    pub key_rgb: [f64; 3],
    /// light wrapped round the body's terminator (0 = Lambert)
    pub wrap: f64,
    pub fill: [f64; 3],
    pub fill_amt: f64,
    /// the key light's highlight: power, strength
    pub spec: [f64; 2],
    /// a second, white highlight: its direction, power, strength
    pub spec2: [f64; 3],
    pub spec2_pow: f64,
    pub spec2_amt: f64,
    pub rim_rgb: [f64; 3],
    pub rim_amt: f64,
    /// how far in from the silhouette the rim reaches, in z of the normal
    pub rim_width: f64,
    /// white reflection growing at grazing angles: (1 - z)^4 times this
    pub fres: f64,
    /// the reflected ray against a ramp of sky and ground: how much of it
    pub mirror: f64,
    pub sky_top: [f64; 3],
    pub sky_hor: [f64; 3],
    pub ground_hor: [f64; 3],
    pub ground_bot: [f64; 3],
    /// half the width of the horizon in the ramp
    pub horizon: f64,
    /// shows the normal as a colour instead
    pub normals: bool,
}

const NONE: Look = Look {
    base: [0.0; 3], ambient: 0.0, sky: 0.0, key: [-0.55, 0.65, 0.55], key_rgb: [1.0; 3], wrap: 0.0,
    fill: [0.7, -0.25, 0.6], fill_amt: 0.0, spec: [1.0, 0.0], spec2: [0.6, 0.3, 0.7], spec2_pow: 1.0, spec2_amt: 0.0,
    rim_rgb: [1.0; 3], rim_amt: 0.0, rim_width: 0.35, fres: 0.0, mirror: 0.0,
    sky_top: [0.95, 0.97, 1.0], sky_hor: [0.55, 0.6, 0.66], ground_hor: [0.18, 0.16, 0.15], ground_bot: [0.05, 0.05, 0.05],
    horizon: 0.02, normals: false,
};

impl Builtin {
    pub const ALL: [Builtin; 12] = [
        Builtin::Studio, Builtin::Soft, Builtin::Clay, Builtin::Ceramic, Builtin::Metal, Builtin::Normals,
        Builtin::Satin, Builtin::Gold, Builtin::Rim, Builtin::Wax, Builtin::Jade, Builtin::Lacquer,
    ];
    pub fn index(self) -> u32 {
        Self::ALL.iter().position(|b| *b == self).expect("every built-in is listed") as u32
    }
    pub fn name(self) -> &'static str {
        match self {
            Builtin::Studio => "Studio",
            Builtin::Soft => "Soft studio",
            Builtin::Clay => "Clay",
            Builtin::Ceramic => "Ceramic",
            Builtin::Metal => "Chrome",
            Builtin::Normals => "Normals",
            Builtin::Satin => "Satin",
            Builtin::Gold => "Gold",
            Builtin::Rim => "Rim light",
            Builtin::Wax => "Red wax",
            Builtin::Jade => "Jade",
            Builtin::Lacquer => "Blue lacquer",
        }
    }
    /// Its numbers.
    pub fn look(self) -> Look {
        match self {
            Builtin::Studio => Look { base: [0.55, 0.56, 0.57], ambient: 0.07, sky: 0.12, wrap: 0.1, fill_amt: 0.22, spec: [80.0, 0.6], rim_amt: 0.28, fres: 0.1, ..NONE },
            Builtin::Soft => Look { base: [0.5, 0.505, 0.51], ambient: 0.2, sky: 0.16, wrap: 0.6, key_rgb: [0.75; 3], fill_amt: 0.2, spec: [12.0, 0.12], rim_amt: 0.12, rim_width: 0.5, fres: 0.04, ..NONE },
            Builtin::Clay => Look { base: [0.8, 0.52, 0.4], ambient: 0.12, sky: 0.16, key: [-0.3, 0.8, 0.5], key_rgb: [0.75; 3], wrap: 0.5, spec: [8.0, 0.05], ..NONE },
            Builtin::Ceramic => Look { base: [0.9, 0.88, 0.84], ambient: 0.08, key: [-0.5, 0.6, 0.6], key_rgb: [0.8; 3], wrap: 0.2, spec: [120.0, 0.9], spec2_pow: 200.0, spec2_amt: 0.4, fres: 0.35, ..NONE },
            Builtin::Metal => Look { key: [-0.4, 0.7, 0.6], spec: [300.0, 1.0], mirror: 1.0, fres: 0.12, ..NONE },
            Builtin::Normals => Look { normals: true, ..NONE },
            Builtin::Satin => Look { base: [0.33, 0.36, 0.44], ambient: 0.05, sky: 0.05, wrap: 0.3, key_rgb: [0.9; 3], fill_amt: 0.1, spec: [6.0, 0.22], rim_rgb: [0.6, 0.66, 0.82], rim_amt: 0.35, rim_width: 0.5, fres: 0.05, ..NONE },
            Builtin::Gold => Look { key: [-0.4, 0.7, 0.6], key_rgb: [1.0, 0.9, 0.7], spec: [200.0, 0.9], mirror: 1.0, fres: 0.1,
                sky_top: [1.0, 0.82, 0.45], sky_hor: [0.75, 0.55, 0.25], ground_hor: [0.2, 0.12, 0.04], ground_bot: [0.06, 0.04, 0.01], horizon: 0.05, ..NONE },
            Builtin::Rim => Look { base: [0.12, 0.12, 0.14], ambient: 0.03, key: [0.2, 0.3, 0.9], key_rgb: [0.3; 3], rim_rgb: [0.3, 0.7, 1.0], rim_amt: 1.2, rim_width: 0.45,
                spec2: [-0.8, 0.5, 0.3], spec2_pow: 40.0, spec2_amt: 0.5, ..NONE },
            Builtin::Wax => Look { base: [0.7, 0.08, 0.06], ambient: 0.1, sky: 0.1, wrap: 0.8, spec: [20.0, 0.18], rim_rgb: [1.0, 0.35, 0.25], rim_amt: 0.35, rim_width: 0.6, fres: 0.05, ..NONE },
            Builtin::Jade => Look { base: [0.08, 0.35, 0.2], ambient: 0.1, wrap: 0.6, key_rgb: [0.8; 3], spec: [150.0, 0.7], rim_rgb: [0.3, 0.9, 0.6], rim_amt: 0.3, rim_width: 0.5, fres: 0.25, ..NONE },
            Builtin::Lacquer => Look { base: [0.05, 0.12, 0.4], ambient: 0.05, wrap: 0.1, key_rgb: [0.9; 3], spec: [300.0, 1.0], spec2_pow: 400.0, spec2_amt: 0.5, mirror: 0.25,
                sky_top: [0.5, 0.5, 0.52], sky_hor: [0.25, 0.25, 0.27], ground_hor: [0.04; 3], ground_bot: [0.01; 3], fres: 0.3, ..NONE },
        }
    }
}

/// Where the matcap comes from.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MatcapSource {
    Builtin(Builtin),
    /// the scene's own material as the studio's lights light it, on a ball (`bake_material`): a live preview
    /// of the material without anything it shows of the scene
    Material,
    /// a picture of a lit ball filling a square (PNG, JPEG, Radiance .hdr or .exr): its path as given
    Image(String),
}

impl Default for MatcapSource {
    fn default() -> Self { MatcapSource::Builtin(Builtin::default()) }
}

/// How surfaces are told apart by colour.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tint {
    /// one colour for everything
    #[default]
    None,
    /// each body (a layer, or layers joined into one) its own colour
    Body,
}

/// The matcap and what is drawn over it. Every strength is 0 (off) to 1.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Matcap {
    pub source: MatcapSource,
    pub tint: Tint,
    /// creases and pits darkened, from the distance field near the surface
    pub cavity: f64,
    /// convex edges lifted and concave ones darkened, from how fast the normal turns across the screen
    pub edge: f64,
    /// lines where the depth jumps and around the silhouette
    pub outline: f64,
    /// a still view is refined over a few jittered frames
    pub aa: bool,
    /// the matcap's light turned round the view's axis, degrees (-180..180)
    pub turn: f64,
    /// and up or down over the top, degrees (-90..90)
    pub lift: f64,
}

impl Default for Matcap {
    fn default() -> Self {
        Self { source: MatcapSource::default(), tint: Tint::None, cavity: 0.5, edge: 0.3, outline: 0.0, aa: true, turn: 0.0, lift: 0.0 }
    }
}

/// Everything about how the viewport draws that is kept with a setup.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ViewSettings {
    pub mode: ViewMode,
    pub matcap: Matcap,
}

impl ViewSettings {
    pub fn is_default(&self) -> bool { *self == Self::default() }
    /// Every strength held to 0..1 (after a setup has been read).
    pub fn hold(&mut self) {
        let m = &mut self.matcap;
        for v in [&mut m.cavity, &mut m.edge, &mut m.outline] {
            *v = if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 };
        }
        m.turn = if m.turn.is_finite() { m.turn.clamp(-180.0, 180.0) } else { 0.0 };
        m.lift = if m.lift.is_finite() { m.lift.clamp(-90.0, 90.0) } else { 0.0 };
    }
    /// A command-line argument: `view=render|matcap`, `matcap=<built-in's name>|material|<picture's path>`,
    /// `matcapTurn` (degrees), `matcapLift` (degrees),
    /// `matcapTint=none|body`, `matcapCavity`, `matcapEdge`, `matcapOutline` (0..1), `matcapAa=0|1`.
    /// False if the name is none of these or the value is not understood.
    pub fn apply_arg(&mut self, k: &str, v: &str) -> bool {
        let num = || v.trim().parse::<f64>().ok().filter(|x| x.is_finite()).map(|x| x.clamp(0.0, 1.0));
        let m = &mut self.matcap;
        match k {
            "view" => match v { "render" => self.mode = ViewMode::Render, "matcap" => self.mode = ViewMode::Matcap, _ => return false },
            "matcap" => {
                m.source = match Builtin::ALL.iter().find(|b| format!("{b:?}").eq_ignore_ascii_case(v)) {
                    Some(b) => MatcapSource::Builtin(*b),
                    None if v == "material" => MatcapSource::Material,
                    None => MatcapSource::Image(v.to_string()),
                }
            }
            "matcapTint" => match v { "none" => m.tint = Tint::None, "body" => m.tint = Tint::Body, _ => return false },
            "matcapCavity" => match num() { Some(x) => m.cavity = x, None => return false },
            "matcapEdge" => match num() { Some(x) => m.edge = x, None => return false },
            "matcapOutline" => match num() { Some(x) => m.outline = x, None => return false },
            "matcapAa" => m.aa = v != "0",
            "matcapTurn" => match v.trim().parse::<f64>().ok().filter(|x| x.is_finite()) { Some(x) => m.turn = x.clamp(-180.0, 180.0), None => return false },
            "matcapLift" => match v.trim().parse::<f64>().ok().filter(|x| x.is_finite()) { Some(x) => m.lift = x.clamp(-90.0, 90.0), None => return false },
            _ => return false,
        }
        true
    }
    pub fn toggled(&self) -> ViewMode {
        match self.mode { ViewMode::Render => ViewMode::Matcap, ViewMode::Matcap => ViewMode::Render }
    }
}

/// Frames a still matcap view is refined over (antialiasing on), or 1.
pub fn frames(m: &Matcap) -> u32 { if m.aa { 8 } else { 1 } }

/// The colours bodies are told apart by (`Tint::Body`), linear RGB: muted, far apart in hue, equally light.
pub const BODY_TINTS: [[f64; 3]; 4] = [[1.0, 1.0, 1.0], [0.55, 0.75, 1.0], [1.0, 0.62, 0.45], [0.6, 0.95, 0.55]];

type V3 = [f64; 3];
fn dot(a: V3, b: V3) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
fn norm(a: V3) -> V3 { let l = dot(a, a).sqrt(); [a[0] / l, a[1] / l, a[2] / l] }
fn sat(x: f64) -> f64 { x.clamp(0.0, 1.0) }
fn mix3(a: V3, b: V3, t: f64) -> V3 { [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t] }
fn add(a: V3, b: V3) -> V3 { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }
fn mul(a: V3, k: f64) -> V3 { [a[0] * k, a[1] * k, a[2] * k] }
fn smoothstep(a: f64, b: f64, x: f64) -> f64 { let t = sat((x - a) / (b - a)); t * t * (3.0 - 2.0 * t) }

fn highlight(n: V3, l: V3, power: f64) -> f64 {
    let l = norm(l);
    let h = norm([l[0], l[1], l[2] + 1.0]);
    sat(dot(n, h)).powf(power)
}

/// The linear colour a matcap of these numbers gives a normal `n` (unit, in the frame of the pixel's ray:
/// z towards the camera, y up, x right).
/// matcap.wgsl `mcShade` is this, line for line.
pub fn shade_look(k: &Look, n: V3) -> V3 {
    // (defined on the whole ball: a light turned over the top shows the back of it as the formula has it)
    let n = norm(n);
    if k.normals {
        // shown as the usual normal-map colour: the colour is the picture, not light (made linear so the
        // screen shows 0.5 + 0.5 n)
        let lin = |c: f64| if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) };
        return [lin(0.5 + 0.5 * n[0]), lin(0.5 + 0.5 * n[1]), lin(0.5 + 0.5 * n[2])];
    }
    let g = 1.0 - n[2].max(0.0);
    let fres = g * g * g * g;
    let key = norm(k.key);
    let kd = sat((dot(n, key) + k.wrap) / (1.0 + k.wrap));
    let fd = sat((dot(n, norm(k.fill)) + 0.5) / 1.5);
    let light = k.ambient + k.sky * (0.5 + 0.5 * n[1]) + k.fill_amt * fd;
    let mut c = [0.0; 3];
    for i in 0..3 { c[i] = k.base[i] * (light + k.key_rgb[i] * kd); }
    let ks = k.spec[1] * highlight(n, key, k.spec[0]);
    let s2 = k.spec2_amt * highlight(n, k.spec2, k.spec2_pow);
    let rim = k.rim_amt * smoothstep(k.rim_width, 0.0, n[2]);
    // the ray reflected about n, its height in the room: a sky, a horizon, a ground
    let ry = 2.0 * n[2] * n[1];
    let sky = mix3(k.sky_hor, k.sky_top, smoothstep(0.0, 0.8, ry));
    let ground = mix3(k.ground_hor, k.ground_bot, smoothstep(0.0, -0.8, ry));
    let ramp = mul(mix3(ground, sky, smoothstep(-k.horizon, k.horizon, ry)), k.mirror * (0.85 + 0.15 * fres));
    for i in 0..3 { c[i] += k.key_rgb[i] * ks + s2 + k.rim_rgb[i] * rim + k.fres * fres + ramp[i]; }
    c
}

/// The linear colour a built-in matcap gives a normal.
pub fn shade(b: Builtin, n: V3) -> V3 { shade_look(&b.look(), n) }

/// The normal turned as the matcap's light is turned: by `turn` degrees about the view's axis (the light
/// goes round the screen with it) and then `lift` degrees about the screen's horizontal. The lookup reads
/// the turned normal, so what the matcap shows moves the other way round: the light moves by `turn`.
pub fn turned(n: V3, turn: f64, lift: f64) -> V3 {
    let (st, ct) = (-turn.to_radians()).sin_cos();
    let (sl, cl) = (-lift.to_radians()).sin_cos();
    let a = [ct * n[0] - st * n[1], st * n[0] + ct * n[1], n[2]];
    [a[0], cl * a[1] - sl * a[2], sl * a[1] + cl * a[2]]
}

/// The normal `n` (in the camera's frame: z towards the camera) as seen along the ray `rd` (unit, from
/// the camera): turned by the smallest rotation that takes `rd` to straight ahead, (0, 0, -1). A ball
/// then reads the same matcap wherever it is on the screen.
pub fn along_ray(n: V3, rd: V3) -> V3 {
    // rotation taking a = -rd to b = (0, 0, 1): Rodrigues with k = a x b, c = a . b
    let a = [-rd[0], -rd[1], -rd[2]];
    let c = a[2];
    let k = [a[1], -a[0], 0.0];
    let f = 1.0 / (1.0 + c);
    let kn = dot(k, n);
    let cr = [k[1] * n[2] - k[2] * n[1], k[2] * n[0] - k[0] * n[2], k[0] * n[1] - k[1] * n[0]];
    add(add(mul(n, c), cr), mul(k, kn * f))
}

/// sRGB encoding of a linear value, as the screen pass does it.
pub fn srgb(c: f64) -> f64 {
    let c = c.clamp(0.0, 1.0);
    if c <= 0.0031308 { 12.92 * c } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

/// Linear light from sRGB-encoded display values.
pub fn linear(c: f64) -> f64 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// What `bake_material` reads of a material, worked out from its settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Surface {
    /// light scattered back by the body (none for clear glass and metals)
    pub diffuse: [f64; 3],
    /// reflectance straight on (a metal's colour; a dielectric's from its index)
    pub f0: [f64; 3],
    /// GGX width (roughness squared)
    pub alpha: f64,
    /// a clear coat over it: how much, and its reflectance straight on
    pub coat: f64,
    pub coat_f0: f64,
    /// what shows through clear glass, as a share of the ambient light
    pub through: [f64; 3],
}

impl Surface {
    pub fn of(p: &crate::settings::Settings) -> Surface {
        let body = [p.body_r, p.body_g, p.body_b].map(|c| c.clamp(0.0, 1.0));
        let f0d = |ior: f64| ((ior - 1.0) / (ior + 1.0)).powi(2);
        let alpha = (p.rough.clamp(0.0, 1.0).powi(2)).max(0.004);
        let coat = if p.coat > 0.0 && p.hollow < 0.5 { p.coat.clamp(0.0, 1.0) } else { 0.0 };
        let coat_f0 = f0d(p.coat_ior.max(1.0));
        if p.metal >= 0.5 {
            // a tabulated metal: its reflectance straight on at a red, a green and a blue wavelength
            let f0 = match crate::material::metal_of(p) {
                Some(m) => [610.0, 550.0, 465.0].map(|nm| {
                    let (n, k) = (crate::material::table_at(&m.n, nm), crate::material::table_at(&m.k, nm));
                    ((n - 1.0).powi(2) + k * k) / ((n + 1.0).powi(2) + k * k)
                }),
                None => [1.0; 3],
            };
            return Surface { diffuse: [0.0; 3], f0: [f0[0] * body[0], f0[1] * body[1], f0[2] * body[2]], alpha, coat, coat_f0, through: [0.0; 3] };
        }
        let f0 = f0d(p.ior.max(1.0));
        let milky = (p.sss > 0.0 || p.ceramic > 0.5) && p.hollow < 0.5;
        if milky {
            return Surface { diffuse: body, f0: [f0; 3], alpha, coat, coat_f0, through: [0.0; 3] };
        }
        // clear glass: what it reflects, and a little of what shows through it, deepened by its tint
        let t = p.absorb.map(|a| (-a * p.density.max(0.0) * 0.6).exp() * 0.35);
        Surface { diffuse: [0.0; 3], f0: [f0; 3], alpha, coat, coat_f0, through: t }
    }
}

fn ggx_spec(n: V3, l: V3, alpha: f64) -> (f64, f64) {
    // D * G / (4 n.v) for v = (0, 0, 1), and v.h for Fresnel
    let h = norm([l[0], l[1], l[2] + 1.0]);
    let (nh, nl, nv, vh) = (dot(n, h).max(0.0), dot(n, l).max(0.0), n[2].max(1e-4), h[2].max(0.0));
    let a2 = alpha * alpha;
    let d = a2 / (std::f64::consts::PI * (nh * nh * (a2 - 1.0) + 1.0).powi(2));
    let k = alpha / 2.0;
    let g = (nl / (nl * (1.0 - k) + k)) * (nv / (nv * (1.0 - k) + k));
    (d * g / (4.0 * nv), vh)
}

fn schlick(f0: f64, c: f64) -> f64 { f0 + (1.0 - f0) * (1.0 - c).powi(5) }

/// Two axes across a unit normal, and the normal.
fn basis(n: V3) -> [V3; 3] {
    let a = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let t = norm([a[1] * n[2] - a[2] * n[1], a[2] * n[0] - a[0] * n[2], a[0] * n[1] - a[1] * n[0]]);
    let b = [n[1] * t[2] - n[2] * t[1], n[2] * t[0] - n[0] * t[2], n[0] * t[1] - n[1] * t[0]];
    [t, b, n]
}
fn local(f: &[V3; 3], v: V3) -> V3 {
    [0, 1, 2].map(|i| f[0][i] * v[0] + f[1][i] * v[1] + f[2][i] * v[2])
}

/// The scene's material on a ball as the studio's lights (`rig`, turned into the camera's frame by
/// `light_m`, its rows) light it, seen straight on: `res` x `res` RGBA, top row first, the ball filling
/// the square as the matcap's lookup reads it (`0.5 + 0.5 * 0.985 * (x, -y)`), linear display values: the
/// light exposed and graded as the full render's screen pass does (`exposure`, `grade`). The studio's
/// backdrop, when there is one (`room`: it, and the room -> camera turn), is the light from everywhere
/// else, reflected and scattered (32 samples a pixel, the same for every pixel). One bounce only: no
/// environment picture, no floor's own shading, no light inside the body.
pub fn bake_material(s: &Surface, rig: &crate::state::Rig, light_m: &crate::math::Mat3, room: Option<(&crate::stage::Backdrop, &crate::math::Mat3)>, exposure: f64, grade: &crate::stage::Grade, res: usize) -> Vec<f32> {
    // the lamps as patches of the lights' map (without the ambient, which is even and done in closed form)
    let (ew, eh) = (256usize, 128usize);
    let map = crate::env::paint(&rig.lights, [0.0; 3], ew, eh);
    let mut lamps: Vec<(V3, V3)> = Vec::new();
    for j in 0..eh {
        let th = (j as f64 + 0.5) / eh as f64 * std::f64::consts::PI;
        let dw = (2.0 * std::f64::consts::PI / ew as f64) * (std::f64::consts::PI / eh as f64) * th.sin();
        for i in 0..ew {
            let o = (j * ew + i) * 4;
            let l = [map[o] as f64, map[o + 1] as f64, map[o + 2] as f64];
            if l[0] + l[1] + l[2] <= 0.0 { continue; }
            let ph = ((i as f64 + 0.5) / ew as f64 - 0.5) * 2.0 * std::f64::consts::PI;
            let d = [th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin()];
            let c = crate::math::apply(light_m, d);
            lamps.push((c, mul(l, dw)));
        }
    }
    let amb = rig.ambient;
    let mut out = vec![0.0f32; res * res * 4];
    for y in 0..res {
        for x in 0..res {
            let mx = ((x as f64 + 0.5) / res as f64 * 2.0 - 1.0) / 0.985;
            let my = -((y as f64 + 0.5) / res as f64 * 2.0 - 1.0) / 0.985;
            let r2 = (mx * mx + my * my).min(1.0);
            let n = norm([mx, my, (1.0 - r2).sqrt()]);
            let fv = (1.0 - n[2]).powi(5);
            let coat_t = 1.0 - s.coat * schlick(s.coat_f0, n[2]);
            let mut c = [0.0; 3];
            // the ambient: diffuse in full, reflected at its average reflectance, and seen through glass
            for i in 0..3 {
                let fr = s.f0[i] + (1.0 - s.f0[i]) * fv;
                c[i] = amb[i] * (coat_t * (s.diffuse[i] * (1.0 - fr) + fr + s.through[i] * (1.0 - fr)) + s.coat * schlick(s.coat_f0, n[2]));
            }
            if let Some((back, m)) = room {
                // the backdrop as seen from direction l (camera frame), through the room's turn
                let env = |l: V3| back.rgb(crate::math::apply(&crate::math::transpose(m), l));
                let frame = basis(n);
                const N: usize = 32;
                for k in 0..N {
                    // (a Hammersley point set)
                    let (u1, u2) = ((k as f64 + 0.5) / N as f64, (k as u32).reverse_bits() as f64 / 4294967296.0);
                    let phi = std::f64::consts::TAU * u2;
                    // the reflection: half-vectors drawn from GGX, each weighted F G (v.h) / (n.h n.v)
                    let a2 = s.alpha * s.alpha;
                    let ct = ((1.0 - u1) / (1.0 + (a2 - 1.0) * u1)).sqrt();
                    let st = (1.0 - ct * ct).max(0.0).sqrt();
                    let h = local(&frame, [st * phi.cos(), st * phi.sin(), ct]);
                    let vh = h[2];
                    let l = [2.0 * vh * h[0], 2.0 * vh * h[1], 2.0 * vh * h[2] - 1.0];
                    let nl = dot(n, l);
                    if nl > 0.0 && vh > 0.0 {
                        let kk = s.alpha / 2.0;
                        let nv = n[2].max(1e-4);
                        let g = (nl / (nl * (1.0 - kk) + kk)) * (nv / (nv * (1.0 - kk) + kk));
                        let w = g * vh / (dot(n, h).max(1e-4) * nv);
                        let e = env(l);
                        for i in 0..3 {
                            let fr = s.f0[i] + (1.0 - s.f0[i]) * (1.0 - vh).powi(5);
                            c[i] += coat_t * e[i] * fr * w / N as f64;
                        }
                    }
                    // the body's own light: directions drawn by cosine
                    if s.diffuse != [0.0; 3] || s.through != [0.0; 3] {
                        let r = u1.sqrt();
                        let l = local(&frame, [r * phi.cos(), r * phi.sin(), (1.0 - u1).sqrt()]);
                        let e = env(l);
                        for i in 0..3 {
                            let fr = s.f0[i] + (1.0 - s.f0[i]) * fv;
                            c[i] += coat_t * e[i] * s.diffuse[i] * (1.0 - fr) / N as f64;
                        }
                    }
                }
                // the coat mirrors it
                if s.coat > 0.0 {
                    let e = env([2.0 * n[2] * n[0], 2.0 * n[2] * n[1], 2.0 * n[2] * n[2] - 1.0]);
                    for i in 0..3 { c[i] += s.coat * schlick(s.coat_f0, n[2]) * e[i]; }
                }
            }
            for (l, e) in &lamps {
                let nl = dot(n, *l);
                if nl <= 0.0 { continue; }
                let (spec, vh) = ggx_spec(n, *l, s.alpha);
                let (cspec, cvh) = if s.coat > 0.0 { ggx_spec(n, *l, 0.0025) } else { (0.0, 1.0) };
                for i in 0..3 {
                    let fr = s.f0[i] + (1.0 - s.f0[i]) * (1.0 - vh).powi(5);
                    let base = s.diffuse[i] * (1.0 - fr) * std::f64::consts::FRAC_1_PI + spec * fr;
                    let coat = s.coat * cspec * schlick(s.coat_f0, cvh);
                    c[i] += e[i] * nl * (coat_t * base + coat);
                }
            }
            let shown = grade.apply([c[0] * exposure, c[1] * exposure, c[2] * exposure]);
            let o = (y * res + x) * 4;
            for i in 0..3 { out[o + i] = shown[i].max(0.0) as f32; }
            out[o + 3] = 1.0;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_settings_roundtrip_and_default_left_out() {
        let mut v = ViewSettings::default();
        assert!(v.is_default());
        v.mode = ViewMode::Matcap;
        v.matcap.source = MatcapSource::Image("caps/red wax.png".into());
        v.matcap.tint = Tint::Body;
        v.matcap.outline = 0.7;
        let text = serde_json::to_string(&v).unwrap();
        assert!(text.contains("\"mode\":\"matcap\""), "{text}");
        let back: ViewSettings = serde_json::from_str(&text).unwrap();
        assert_eq!(back, v);
        for b in Builtin::ALL {
            let v = ViewSettings { mode: ViewMode::Matcap, matcap: Matcap { source: MatcapSource::Builtin(b), ..Default::default() } };
            let back: ViewSettings = serde_json::from_str(&serde_json::to_string(&v).unwrap()).unwrap();
            assert_eq!(back, v);
        }
        // a file that leaves things out reads as the defaults; strengths are held to 0..1
        let mut part: ViewSettings = serde_json::from_str(r#"{"mode":"matcap","matcap":{"cavity":7}}"#).unwrap();
        part.hold();
        assert_eq!(part.mode, ViewMode::Matcap);
        assert_eq!(part.matcap.cavity, 1.0);
        assert_eq!(part.matcap.edge, Matcap::default().edge);
        assert_eq!(v.toggled(), ViewMode::Render);
        let mut a = ViewSettings::default();
        for (k, x) in [("view", "matcap"), ("matcap", "metal"), ("matcapCavity", "0"), ("matcapOutline", "2"), ("matcapTint", "body"), ("matcapAa", "0")] {
            assert!(a.apply_arg(k, x), "{k}={x}");
        }
        assert_eq!(a.mode, ViewMode::Matcap);
        assert_eq!(a.matcap, Matcap { source: MatcapSource::Builtin(Builtin::Metal), tint: Tint::Body, cavity: 0.0, edge: Matcap::default().edge, outline: 1.0, aa: false, turn: 0.0, lift: 0.0 });
        assert!(a.apply_arg("matcap", "caps/wax.png") && a.matcap.source == MatcapSource::Image("caps/wax.png".into()));
        assert!(!a.apply_arg("view", "glass") && !a.apply_arg("ior", "1.5"));
    }

    #[test]
    fn turning_moves_the_light_round_the_screen_and_over() {
        // the key light (as the lookup reads it) turned by `turn` about the view's axis and lifted by `lift`
        // appears where the turned lookup meets the old key: n = Rz(turn) Rx(lift) key
        let key = norm([-0.55, 0.65, 0.55]);
        for (turn, lift) in [(90.0f64, 0.0f64), (-35.0, 20.0), (180.0, -60.0)] {
            let (st, ct) = turn.to_radians().sin_cos();
            let (sl, cl) = lift.to_radians().sin_cos();
            let a = [key[0], cl * key[1] - sl * key[2], sl * key[1] + cl * key[2]];
            let n = [ct * a[0] - st * a[1], st * a[0] + ct * a[1], a[2]];
            let back = turned(n, turn, lift);
            assert!((0..3).all(|i| (back[i] - key[i]).abs() < 1e-12), "{turn} {lift}: {back:?}");
        }
        // a quarter turn takes the studio's key light from the upper left to the lower left
        let lower_left = norm([-0.6, -0.6, 0.5]);
        let upper_right = norm([0.6, 0.6, 0.5]);
        let c = |n| shade(Builtin::Studio, turned(n, 90.0, 0.0))[1];
        assert!(c(lower_left) > c(upper_right));
    }

    #[test]
    fn the_material_ball_is_lit_by_the_rig() {
        use crate::presets::Light;
        let one = |az: f64, el: f64| crate::state::Rig { lights: vec![Light { az, el, w: 20.0, h: 20.0, rgb: [4.0; 3], soft: 0.3 }], ambient: [0.02; 3] };
        let id = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let grade = crate::stage::Grade::of(&crate::settings::Settings::default());
        let res = 64;
        let at = |img: &[f32], x: usize, y: usize| img[(y * res + x) * 4 + 1];
        // matte white lit from the camera's side (az 180): brightest in the middle, darker towards the rim
        let mut p = crate::settings::Settings::default();
        (p.sss, p.body_r, p.body_g, p.body_b, p.rough) = (1.0, 0.8, 0.8, 0.8, 1.0);
        let img = bake_material(&Surface::of(&p), &one(180.0, 0.0), &id, None, 1.0, &grade, res);
        assert!(at(&img, 32, 32) > at(&img, 3, 32) && at(&img, 32, 32) > 0.05);
        // lit from above: the top brighter than the bottom (top row first)
        let img = bake_material(&Surface::of(&p), &one(180.0, 60.0), &id, None, 1.0, &grade, res);
        assert!(at(&img, 32, 12) > at(&img, 32, 52));
        // a polished mirror shows the lamp where the normal is halfway to it: from the right, at x > middle
        let mut m = crate::settings::Settings::default();
        (m.metal, m.rough) = (1.0, 0.05);
        let img = bake_material(&Surface::of(&m), &one(90.0, 0.0), &id, None, 1.0, &grade, res);
        let (bx, _) = (0..res).map(|x| (x, at(&img, x, 32))).fold((0, -1.0), |b, (x, v)| if v > b.1 { (x, v) } else { b });
        // halfway between straight on and the right: 45 degrees, x = sin 45 of the ball's radius
        let want = ((0.5 + 0.5 * 0.985 * (std::f64::consts::FRAC_PI_4).sin()) * res as f64) as usize;
        assert!((bx as i64 - want as i64).abs() <= 2, "{bx} vs {want}");
        assert!(img.iter().all(|v| v.is_finite()));
        // a mirror in a studio whose backdrop is bright above and dark below: its top half (reflecting the
        // sky side of the room) brighter than its bottom
        let mut st = crate::settings::Settings::default();
        st.backdrop = 1.0;
        if let Some(back) = crate::stage::Backdrop::of(&st) {
            let none = crate::state::Rig { lights: vec![], ambient: [0.0; 3] };
            let img = bake_material(&Surface::of(&m), &none, &id, Some((&back, &id)), 1.0, &grade, res);
            assert!(img.iter().all(|v| v.is_finite()));
            let top = at(&img, 32, 10);
            assert!(top > 0.0, "the mirror shows the backdrop");
        }
    }

    #[test]
    fn builtin_indices_are_stable() {
        for (i, b) in Builtin::ALL.iter().enumerate() { assert_eq!(b.index(), i as u32); }
    }

    #[test]
    fn along_ray_is_a_rotation_taking_the_ray_ahead() {
        let rds = [[0.0, 0.0, -1.0], [0.3, -0.2, -0.93], [-0.5, 0.4, -0.77], [0.01, 0.6, -0.8]];
        for rd in rds {
            let rd = norm(rd);
            let v = along_ray([-rd[0], -rd[1], -rd[2]], rd);
            assert!((v[0]).abs() < 1e-12 && (v[1]).abs() < 1e-12 && (v[2] - 1.0).abs() < 1e-12, "{v:?}");
            for n in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], norm([0.3, 0.4, 0.5])] {
                let m = along_ray(n, rd);
                assert!((dot(m, m) - 1.0).abs() < 1e-12);
            }
            // angles kept
            let (a, b) = (norm([0.2, 0.9, 0.1]), norm([-0.7, 0.1, 0.6]));
            assert!((dot(along_ray(a, rd), along_ray(b, rd)) - dot(a, b)).abs() < 1e-12);
        }
    }

    #[test]
    fn builtins_are_finite_bounded_and_lit_from_above() {
        for b in Builtin::ALL {
            for i in 0..=40 {
                for j in 0..=40 {
                    let (x, y) = (i as f64 / 20.0 - 1.0, j as f64 / 20.0 - 1.0);
                    if x * x + y * y > 1.0 { continue; }
                    let c = shade(b, [x, y, (1.0 - x * x - y * y).max(0.0).sqrt()]);
                    assert!(c.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 2.5), "{b:?} {c:?}");
                }
            }
            if b != Builtin::Normals && b != Builtin::Metal {
                let up = shade(b, norm([0.0, 0.7, 0.7]));
                let down = shade(b, norm([0.0, -0.7, 0.7]));
                assert!(up[1] > down[1], "{b:?}: lit from above");
            }
        }
        // the normals matcap shows 0.5 + 0.5 n on the screen
        let c = shade(Builtin::Normals, [0.0, 0.0, 1.0]);
        assert!((srgb(c[0]) - 0.5).abs() < 1e-9 && (srgb(c[2]) - 1.0).abs() < 1e-9);
    }
}
