//! Light scattering inside a body (milky glass, jade, wax, porcelain), the part that needs no GPU: what
//! the body is as a scattering medium, and the lighting said as a few beams. The light inside the body
//! is then worked out on the GPU into a small volume (`app/src/scatter.rs`, `shaders/scatter.wgsl`) which
//! the tracer reads. `docs/materials.md` has the method and its limits.

use std::sync::OnceLock;

use crate::env::{self, Sh9};
use crate::material;
use crate::settings::Settings;

/// A body as a medium that scatters and absorbs light, per colour (red, green, blue), per unit length.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Medium {
    /// scatterings per unit length
    pub sig_s: [f64; 3],
    /// absorption per unit length
    pub sig_a: [f64; 3],
    /// how much the surface holds diffuse light in: (1 + r) / (1 - r), r being the share of diffuse
    /// light arriving at the surface from inside that it sends back (1 for a body of air's own index)
    pub hold: f64,
    /// the share of light arriving from outside, from all round, that gets in
    pub enters: f64,
    /// the body's index of refraction
    pub ior: f64,
}

/// The share of light arriving evenly from every direction on one side of a surface that the surface
/// reflects, going from index `n1` into `n2` (the Fresnel reflectance averaged over the hemisphere,
/// weighted by the cosine).
pub fn diffuse_reflectance(n1: f64, n2: f64) -> f64 {
    const N: usize = 512;
    let mut sum = 0.0;
    for i in 0..N {
        // even steps of the cosine squared: each stands for the same share of the arriving light
        let c = ((i as f64 + 0.5) / N as f64).sqrt();
        sum += material::fresnel(c, n1, n2);
    }
    sum / N as f64
}

impl Medium {
    /// The medium the settings ask for. `scale`: what the material's distances are multiplied by to fit
    /// the shape (`material::fit_scale`).
    pub fn of(p: &Settings, scale: f64) -> Self {
        let sc = material::scatter([p.body_r, p.body_g, p.body_b], p.sss_depth * scale, p.sss);
        // (under a full clear coat the body meets the coat, and the coat meets the air: for diffuse light
        // the two together hold in what a bare surface of the body's index would)
        let ior = p.ior.max(1.0);
        let enters = 1.0 - diffuse_reflectance(1.0, ior);
        // Seen from inside, the same surface lets out 1 / ior^2 of what it lets in (the cone of
        // directions that can leave is that much narrower); so the two shares are tied together, and a
        // body that absorbs nothing gives back exactly what arrives.
        let back = 1.0 - enters / (ior * ior);
        Self { sig_s: [sc.sigma_s; 3], sig_a: sc.sigma_a, hold: (1.0 + back) / (1.0 - back), enters, ior }
    }

    /// What a ray loses per unit length, by colour.
    pub fn sig_t(&self) -> [f64; 3] {
        [0, 1, 2].map(|i| self.sig_s[i] + self.sig_a[i])
    }
    /// The diffusion constant by colour: how readily scattered light spreads.
    pub fn diffusion(&self) -> [f64; 3] {
        self.sig_t().map(|t| 1.0 / (3.0 * t.max(1e-9)))
    }
    /// How far beyond the surface diffuse light would fall to nothing if it went on falling as it does
    /// at the surface, by colour.
    pub fn beyond(&self) -> [f64; 3] {
        self.diffusion().map(|d| 2.0 * self.hold * d)
    }
}

/// One beam of the lighting: the light of a patch of the sky, taken as coming from one direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Beam {
    /// the direction the light comes from (a unit vector, in the lights' own frame)
    pub dir: [f64; 3],
    /// how wide the patch is, as the tangent of its angular radius: what softens the shadows it casts
    pub tan: f64,
    /// the light it carries per colour: brightness times the size of the patch (radiance times solid angle)
    pub rgb: [f64; 3],
}

/// Light that is not in any beam: the faint rest of the sky, as light from all round with a lean to one
/// side. Its brightness around direction d is `even + lean . d` (per colour).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Rest {
    /// the brightness of the rest of the sky, averaged over all directions, by colour
    pub even: [f64; 3],
    /// how it leans: for red, green and blue, the direction (and amount) in which it is brighter
    pub lean: [[f64; 3]; 3],
}

/// The lighting as the light volume takes it.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Lighting {
    pub beams: Vec<Beam>,
    pub rest: Rest,
}

/// Cells along a side of the light volume for a setting of `sssDetail`: a whole even number, 64 to 256.
pub fn detail(setting: f64) -> u32 {
    let v = if setting.is_finite() { setting } else { 160.0 };
    ((v / 16.0).round() as i64 * 16).clamp(64, 256) as u32
}

/// Patches the sky is divided into.
pub const PATCHES: usize = 48;
/// Beams at most (the tracer's light volume takes this many): the brightest patches each get a beam of
/// their own; the light of all the others is the rest.
pub const MAX_BEAMS: usize = 20;

/// The middles of the patches: points spread evenly over the sphere (a golden-angle spiral).
fn patch_dirs() -> &'static [[f64; 3]; PATCHES] {
    static DIRS: OnceLock<[[f64; 3]; PATCHES]> = OnceLock::new();
    DIRS.get_or_init(|| {
        let mut d = [[0.0; 3]; PATCHES];
        for (i, v) in d.iter_mut().enumerate() {
            let y = 1.0 - 2.0 * (i as f64 + 0.5) / PATCHES as f64;
            let r = (1.0 - y * y).sqrt();
            let a = i as f64 * 2.399963229728653;
            *v = [r * a.cos(), y, r * a.sin()];
        }
        d
    })
}

fn nearest_patch(d: [f64; 3]) -> usize {
    let mut best = (0, f64::MIN);
    for (i, p) in patch_dirs().iter().enumerate() {
        let c = d[0] * p[0] + d[1] * p[1] + d[2] * p[2];
        if c > best.1 { best = (i, c); }
    }
    best.0
}

/// Which patch each texel of a lights' map of this size belongs to, and the patch of sky each row's
/// texels cover.
fn map_patches(w: usize, h: usize) -> (Vec<u8>, Vec<f64>) {
    use std::f64::consts::PI;
    let mut which = vec![0u8; w * h];
    let mut sky = vec![0.0; h];
    for j in 0..h {
        sky[j] = (2.0 * PI / w as f64) * ((j as f64 / h as f64 * PI).cos() - ((j + 1) as f64 / h as f64 * PI).cos());
        for i in 0..w {
            which[j * w + i] = nearest_patch(env::paint_dir((i as f64 + 0.5) / w as f64, (j as f64 + 0.5) / h as f64)) as u8;
        }
    }
    (which, sky)
}

/// The lighting as beams, in the lights' own frame.
///
/// `map`: the lights' map as painted (RGBA floats, `w` x `h`, row 0 the top of the sky): the family's
/// lights and the glows of an environment picture. `smooth`: light that is given as nine smooth terms
/// instead (the body of an environment picture), added to the map's.
///
/// The sky is divided into `PATCHES` equal patches. A patch's beam comes from where the patch's light
/// is, on average (so a small lamp is a beam from the lamp, not from the middle of its patch), carries
/// all of the patch's light, and is as wide as that light is spread. Patches fixed in the lights' frame
/// mean the beams turn with the lights and change smoothly as a light is moved or resized.
///
/// A beam is given a little less than its patch's light: what a surface facing it gets from light
/// spread over a patch of the sky is less than from the same light all coming straight at it (by the
/// patch's mean cosine), and the light volume takes every beam as coming straight.
pub fn beams(map: &[f32], w: usize, h: usize, smooth: Option<&Sh9>) -> Lighting {
    beams_at_most(map, w, h, smooth, MAX_BEAMS)
}

/// `beams`, with no more than `most` beams (the light of every other patch joins the rest).
pub fn beams_at_most(map: &[f32], w: usize, h: usize, smooth: Option<&Sh9>, most: usize) -> Lighting {
    use std::f64::consts::PI;
    static TABLE: OnceLock<(usize, usize, Vec<u8>, Vec<f64>)> = OnceLock::new();
    let t = TABLE.get_or_init(|| { let (a, b) = map_patches(w, h); (w, h, a, b) });
    let own;
    let (which, sky) = if t.0 == w && t.1 == h { (&t.2, &t.3) } else { own = map_patches(w, h); (&own.0, &own.1) };
    // per patch: the light, the direction weighed by brightness, and the brightness itself
    let mut rgb = [[0.0f64; 3]; PATCHES];
    let mut dir = [[0.0f64; 3]; PATCHES];
    let mut lum = [0.0f64; PATCHES];
    let grey = |c: [f64; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    fn add(rgb: &mut [f64; 3], dir: &mut [f64; 3], lum: &mut f64, d: [f64; 3], c: [f64; 3], g: f64) {
        for i in 0..3 {
            rgb[i] += c[i];
            dir[i] += d[i] * g;
        }
        *lum += g;
    }
    for j in 0..h {
        for i in 0..w {
            let o = (j * w + i) * 4;
            let c = [map[o] as f64 * sky[j], map[o + 1] as f64 * sky[j], map[o + 2] as f64 * sky[j]];
            if c[0] > 0.0 || c[1] > 0.0 || c[2] > 0.0 {
                let k = which[j * w + i] as usize;
                add(&mut rgb[k], &mut dir[k], &mut lum[k], env::paint_dir((i as f64 + 0.5) / w as f64, (j as f64 + 0.5) / h as f64), c, grey(c));
            }
        }
    }
    // how wide a patch is: the angular radius of a cap of its size
    let patch = (1.0 - 2.0 / PATCHES as f64).acos();
    let mut spread = [0.0f64; PATCHES];
    for k in 0..PATCHES {
        // The light's own spread within the patch: where its directions agree, their brightness-weighted
        // sum is as long as the brightness; where they are spread over the patch it is shorter.
        let len = (dir[k][0] * dir[k][0] + dir[k][1] * dir[k][1] + dir[k][2] * dir[k][2]).sqrt();
        if lum[k] > 0.0 { spread[k] = (2.0 * (1.0 - (len / lum[k]).min(1.0))).sqrt() * 1.4; }
    }
    if let Some(sh) = smooth {
        // smooth light: each patch's share, from the patch's middle, as wide as the patch
        let sky = 4.0 * PI / PATCHES as f64;
        for (k, d) in patch_dirs().iter().enumerate() {
            let b = env::sh_basis(*d);
            let mut c = [0.0; 3];
            for (j, bj) in b.iter().enumerate() {
                for i in 0..3 { c[i] += sh[j][i] * bj; }
            }
            let c = c.map(|v| v.max(0.0) * sky);
            let (g0, g1) = (lum[k], grey(c));
            if g1 > 0.0 {
                spread[k] = (spread[k] * g0 + 0.7 * patch * g1) / (g0 + g1);
                add(&mut rgb[k], &mut dir[k], &mut lum[k], *d, c, g1);
            }
        }
    }
    let total: f64 = lum.iter().sum();
    if total <= 0.0 { return Lighting::default(); }
    // the brightest patches get beams of their own
    let mut order: Vec<usize> = (0..PATCHES).filter(|&k| lum[k] > 0.0).collect();
    order.sort_by(|&a, &b| lum[b].partial_cmp(&lum[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    let mut out = Lighting::default();
    // (A patch too faint to matter joins the rest: one beyond the count, or under a hundredth of the
    // brightest. Between one and two hundredths a patch is part beam and part rest, so that a light
    // growing fainter, or moving off a patch, goes over from the one to the other little by little.)
    let floor = 0.01 * lum[order[0]];
    let sphere = 4.0 * PI;
    for (n, &k) in order.iter().enumerate() {
        let len = (dir[k][0] * dir[k][0] + dir[k][1] * dir[k][1] + dir[k][2] * dir[k][2]).sqrt();
        let d = if len > 1e-12 { [dir[k][0] / len, dir[k][1] / len, dir[k][2] / len] } else { patch_dirs()[k] };
        let own = if n < most.min(MAX_BEAMS) { ((lum[k] - floor) / floor).clamp(0.0, 1.0) } else { 0.0 };
        if own > 0.0 {
            let wide = spread[k].clamp(0.02, 1.2);
            let straight = 0.5 * (1.0 + wide.cos());
            out.beams.push(Beam { dir: d, tan: wide.tan(), rgb: rgb[k].map(|c| c * straight * own) });
        }
        if own < 1.0 {
            // the rest: its light averaged over the sphere, and three times its light weighed by direction
            // (the brightness that light from all round with a lean has is  even + lean . d)
            for i in 0..3 {
                out.rest.even[i] += (1.0 - own) * rgb[k][i] / sphere;
                for a in 0..3 { out.rest.lean[i][a] += 3.0 * (1.0 - own) * rgb[k][i] * d[a] / sphere; }
            }
        }
    }
    out
}

/// The light of all the beams together, per colour.
pub fn beams_total(l: &Lighting) -> [f64; 3] {
    let mut t = l.rest.even.map(|c| c * 4.0 * std::f64::consts::PI);
    for x in &l.beams {
        for i in 0..3 { t[i] += x.rgb[i]; }
    }
    t
}
