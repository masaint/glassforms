//! The camera's side of the renderer. The camera itself (what a focal length, an f-stop or a fisheye
//! means) is gf_core::camera; the lens's arithmetic for the shaders is shaders/camera.wgsl. This file
//! puts that arithmetic into the four shaders that work out where things are (the tracer, the pass that
//! folds a frame into the picture, the focus stage and the final pass), and packs the camera for them.
//!
//! None of it touches the plain texts of those shaders: with an ordinary lens that keeps straight lines
//! straight they are built and run exactly as before. Each has a second text, made here from the plain
//! one by exact replacements, which is built the first time a lens calls for it.

use std::cell::OnceCell;

use gf_core::camera::{Camera, BOUND};
use gf_core::settings::Settings;

use crate::gpu::*;

/// The lens's arithmetic as the shaders have it.
pub const LENS_WGSL: &str = include_str!("shaders/camera.wgsl");
const FIRST_RAY_WGSL: &str = include_str!("shaders/camera_trace.wgsl");
const FINAL_WGSL: &str = include_str!("shaders/camera_post.wgsl");

/// The lens as the shaders have it (camera.wgsl: Cam).
pub type CamU = [[f32; 4]; 4];

/// `old` replaced by `new` in `src`, where it must stand exactly once.
fn swap(src: &str, old: &str, new: &str) -> String {
    assert_eq!(src.matches(old).count(), 1, "the camera's shaders are made from the plain ones, and this place in one of them has changed:\n{old}");
    src.replace(old, new)
}

/// `src` without the stretch that begins at `from` and ends just before `to`, and that stretch.
fn cut<'a>(src: &'a str, from: &str, to: &str) -> (String, &'a str) {
    let a = src.find(from).unwrap_or_else(|| panic!("the camera's shaders are made from the plain ones, and this line of one of them is gone:\n{from}"));
    let b = a + src[a..].find(to).unwrap_or_else(|| panic!("the camera's shaders are made from the plain ones, and this line of one of them is gone:\n{to}"));
    (format!("{}{}", &src[..a], &src[b..]), &src[a..b])
}

/// `members` put at the end of the block that begins with `head` (`struct U {`): just before its closing
/// brace, so after whatever anyone else has added to the block.
fn append(src: &str, head: &str, members: &str) -> String {
    let a = src.find(head).unwrap_or_else(|| panic!("the camera's shaders are made from the plain ones, and this block of one of them is gone:\n{head}"));
    let b = a + src[a..].find("\n}").expect("a block ends with a brace on its own line") + 1;
    format!("{}{members}{}", &src[..b], &src[b..])
}

const PRIMARY_FROM: &str = "// The first glass along a ray: how far away it is";
const PRIMARY_TO: &str = "// a value 0..1 that differs well between neighbouring pixels";
const LENS_POINT_FROM: &str = "// A point of the lens opening from two numbers in 0..1";
const LENS_POINT_TO: &str = "// Each pixel fires `ns` rays at different points within it";

/// The tracer's text (as it stands before it is filled in for a family and a way of drawing) with the
/// first ray going through any lens: its own primary(), sampleAt() and lensPoint() are taken out and the
/// camera's put in their place, and its block of numbers gets the lens at its end.
pub fn tracer(template: &str) -> String {
    let s = append(template, "struct U {", "  cam: Cam,               // the lens (see the end of this text)\n");
    let (s, _) = cut(&s, PRIMARY_FROM, PRIMARY_TO);
    let (s, _) = cut(&s, LENS_POINT_FROM, LENS_POINT_TO);
    format!("{s}{LENS_WGSL}{FIRST_RAY_WGSL}")
}

/// How code elsewhere in the tracer (a material that draws random numbers for each pixel) works out
/// which pixel a ray came through: back from the ray's direction, as an ordinary lens sent it. Through
/// another lens that is the wrong pixel, or (orthographic: every ray has the same direction) no pixel
/// at all.
const PIXEL_FROM_RAY: &str = "  let uv = rd0.xy / (-rd0.z * u.tan_fov) - u.shift;\n";

/// The last touch to a tracer built for any lens, once every part of it is filled in. Where a part
/// works a pixel back from a ray (see above), the first ray leaves a note of where on the picture it
/// started (`lensAt`: across and up from the picture's middle, in half-heights) and that part reads the
/// note instead. A tracer with no such part gets no note: nothing more for it to carry.
pub fn finish(src: String) -> String {
    if src.contains(PIXEL_FROM_RAY) {
        let s = src.replace(PIXEL_FROM_RAY, "  let uv = lensAt;\n").replace("{{LENS_AT}}", "lensAt = (2.0 * fc - u.res) / u.res.y;\n  ");
        format!("{s}\n// where on the picture the ray being followed started (see sampleAt)\nvar<private> lensAt: vec2f;\n")
    } else {
        src.replace("{{LENS_AT}}", "")
    }
}

/// The part of the tracer's text that `tracer` takes out: what the camera's own first-ray code must
/// keep in step with, statement for statement.
pub fn plain_first_ray(template: &str) -> String {
    let (rest, a) = cut(template, PRIMARY_FROM, PRIMARY_TO);
    let (_, b) = cut(&rest, LENS_POINT_FROM, LENS_POINT_TO);
    format!("{a}{b}")
}

/// The pass that folds a frame into the picture, following the surface through any lens: a pixel's glass
/// is found along the lens's ray, and looked for in the old picture where the lens of that frame put it.
pub fn accum(src: &str) -> String {
    let s = append(src, "struct A {", "  lens: Cam,              // the lens now ...\n  lens_p: Cam,            // ... and at the previous frame\n");
    let s = swap(&s,
        "    let p = vec3f(a.eye.xy, a.cam.x) + normalize(vec3f(uv * a.cam.y, -1.0)) * d;
    let o = p.x * a.obj0.xyz + p.y * a.obj1.xyz + p.z * a.obj2.xyz;                    // world -> object now
    let v = vec3f(dot(a.objp0.xyz, o), dot(a.objp1.xyz, o), dot(a.objp2.xyz, o)) - vec3f(a.eye.zw, a.cam_p.x);
    at = ((v.xy / (-v.z * a.cam_p.y) - a.shift_p) * a.res.y + a.res) * 0.5;
    expect = length(v);
",
        "    var ro: vec3f;
    var rd: vec3f;
    _ = camRay(uv, a.lens, &ro, &rd);
    let p = ro + vec3f(a.eye.xy, 0.0) + rd * d;                                        // (a free camera stands off the axis)
    let o = p.x * a.obj0.xyz + p.y * a.obj1.xyz + p.z * a.obj2.xyz;                    // world -> object now
    // where that point was then, beside the axis the lens then stood on
    let v = vec3f(dot(a.objp0.xyz, o), dot(a.objp1.xyz, o), dot(a.objp2.xyz, o)) - vec3f(a.eye.zw, 0.0);
    let f = camFilm(v, a.lens_p);
    // (if the lens could not see it then, there is no old picture of it: a place off the screen says so)
    at = select(vec2f(-1.0), ((f.xy - a.shift_p) * a.res.y + a.res) * 0.5, f.z > 0.5);
    expect = camAlong(v, a.lens_p);
");
    let s = swap(&s,
        "    let dir = normalize(vec3f(uv * a.cam.y, -1.0));
    let e = dir.x * a.env0.xyz + dir.y * a.env1.xyz + dir.z * a.env2.xyz;
    let v = vec3f(dot(a.envp0.xyz, e), dot(a.envp1.xyz, e), dot(a.envp2.xyz, e));
    // (if it was behind the camera then, there is no old picture of it: a place off the screen says so)
    at = select(vec2f(-1.0), ((v.xy / (-v.z * a.cam_p.y) - a.shift_p) * a.res.y + a.res) * 0.5, v.z < -1e-4);
",
        "    var ro: vec3f;
    var dir: vec3f;
    // (through an orthographic lens every pixel looks the same way: the whole backdrop is one colour,
    // and stays where it is)
    if (camRay(uv, a.lens, &ro, &dir) && (a.lens.a.x < 0.5 || a.lens.a.x > 1.5)) {
      let e = dir.x * a.env0.xyz + dir.y * a.env1.xyz + dir.z * a.env2.xyz;
      let v = vec3f(dot(a.envp0.xyz, e), dot(a.envp1.xyz, e), dot(a.envp2.xyz, e));
      // (if the lens could not see that way then, there is no old picture of it: a place off the screen says so)
      let f = camFilmDir(v, a.lens_p);
      at = select(vec2f(-1.0), ((f.xy - a.shift_p) * a.res.y + a.res) * 0.5, f.z > 0.5);
    }
");
    format!("{s}{LENS_WGSL}")
}

/// The focus stage with every pixel's blur worked out through any lens, and with highlights that may
/// have a bright rim or a bright core.
pub fn focus(src: &str) -> String {
    let s = append(src, "struct D {", "  lens: Cam,           // the lens (see the end of this text)\n");
    let (s, _) = cut(&s, "// The radius of the blur, in pixels, of glass t away along the ray", "// A pixel on the outline of glass, with a picture behind the glass");
    let (s, _) = cut(&s, "// How much of a texel at `off` (texels: right, down) the blur of a texel of radius q", "// the range of blurs that can reach a pixel");
    // every place a blurred texel's light is handed to another: less the part of the lens opening that
    // the barrel cuts off for the texel the light comes from (which is where `at` says, in pixels)
    let s = swap(&s, "        let cov = cover_far(q, vec2i(i, j));\n", "        let cov = cover_far(q, vec2i(i, j)) * eye(q, vec2i(i, j), (vec2f(p + vec2i(i, j)) + 0.5) * lv.scale);\n");
    let s = swap(&s, "        g += cover_near(q, vec2i(i, j)) * vec4f(s.rgb, s.a - q);\n", "        g += cover_near(q, vec2i(i, j)) * eye(q, vec2i(i, j), (vec2f(p + vec2i(i, j)) + 0.5) * lv.scale) * vec4f(s.rgb, s.a - q);\n");
    let s = swap(&s, "        far0 += cover_far(q, vec2i(i, j)) * vec4f(z.rgb, a - q);\n", "        far0 += cover_far(q, vec2i(i, j)) * eye(q, vec2i(i, j), vec2f(c + vec2i(i, j)) + 0.5) * vec4f(z.rgb, a - q);\n");
    let s = swap(&s, "        if (z.a < 0.0) { near0 += cover_near(q, vec2i(i, j)) * vec4f(z.rgb, a - q); }\n", "        if (z.a < 0.0) { near0 += cover_near(q, vec2i(i, j)) * eye(q, vec2i(i, j), vec2f(c + vec2i(i, j)) + 0.5) * vec4f(z.rgb, a - q); }\n");
    let s = swap(&s, "        if (own > 0.0) { hid += cover_near(own, vec2i(i, j)); }\n", "        if (own > 0.0) { hid += cover_near(own, vec2i(i, j)) * eye(own, vec2i(i, j), vec2f(c) + 0.5); }\n");
    format!("{s}{LENS_WGSL}{}", "
// The radius of the blur, in pixels, of glass t away along the ray through the middle of a pixel (fc counts
// upward from the bottom, as the tracer does), through whatever lens is on. Negative = in front of the
// band, 0 = inside it.
fn blur_of(t: f32, fc: vec2f) -> f32 {
  let q = (2.0 * fc - d.res) / d.res.y + d.shift;
  var ro: vec3f;
  var rd: vec3f;
  if (!camRay(q, d.lens, &ro, &rd)) { return 0.0; }
  return clamp(d.k * camBlur(ro, rd, t, d.planes.x - d.cam.x, d.planes.y - d.cam.x, d.lens), -d.cmax, d.cmax);
}

// How a blurred point's light is spread over its shape, by how far from the middle to the edge: evenly,
// or with a share of it drawn towards the edge (as 3 r^4) or towards the middle (as 2 (1 - r^2)): the
// same two laws the tracer's lens follows (camera_trace.wgsl: lensRadius).
fn edge_light(r: f32) -> f32 {
  let s = d.lens.e.y;
  if (s == 0.0) { return 1.0; }
  let r2 = min(r * r, 1.0);
  if (s > 0.0) { return 1.0 - s + s * 3.0 * r2 * r2; }
  return 1.0 + s - s * 2.0 * (1.0 - r2);
}

// How much of a texel at `off` (texels: right, down) the blur of a texel of radius q (32nds) covers, and
// how much of the blur's light falls there. The lens opening is seen the right way round in the blur of
// what is behind the band, and turned half way round in the blur of what is in front of it.
fn cover_far(q: f32, off: vec2i) -> f32 {
  let edge = q * d.ka * 0.03125;
  let at = textureLoad(kernTex, vec2i(32 - off.x, 32 + off.y), 0).r;
  return saturate(edge - at + 0.5) * edge_light(at / max(edge, 1e-3));
}
fn cover_near(q: f32, off: vec2i) -> f32 {
  let edge = q * d.ka * 0.03125;
  let at = textureLoad(kernTex, vec2i(32 + off.x, 32 - off.y), 0).r;
  return saturate(edge - at + 0.5) * edge_light(at / max(edge, 1e-3));
}

// Towards the corners the lens's barrel cuts into its opening from both sides (camera.wgsl: camCut), and
// a blurred point there is a cat's eye. How much of the light of a texel of blur radius q (32nds), which
// is at `at` (pixels from the picture's top left), still reaches the texel `off` away: nothing where the
// cut falls; elsewhere a little more than its share, so that the point keeps all its light (what the cut
// costs the corners in brightness is taken off the finished picture, sharp and blurred parts alike).
fn eye(q: f32, off: vec2i, at: vec2f) -> f32 {
  let k = d.lens.e.z;
  if (k <= 0.0) { return 1.0; }
  let cut = camCut(vec2f(2.0 * at.x - d.res.x, d.res.y - 2.0 * at.y) / d.res.y + d.shift, k);
  let r = max(q * 0.03125, 0.5);
  // where in the opening (as a unit disc; an anamorphic lens's widened back to round) the light passes
  let l = vec2f(f32(off.x) * d.lens.e.x, -f32(off.y)) / r;
  return saturate((1.0 - max(length(l - cut), length(l + cut))) * r + 0.5) / camOpen(cut);
}
")
}

/// The share of a blurred point's light at `r` of the way from its middle to its edge (dof.wgsl:
/// edge_light), for a highlight edge of `s` (-1 a bright core .. 0 even .. 1 a bright rim).
pub fn edge_light(r: f32, s: f32) -> f32 {
    let r2 = (r * r).min(1.0);
    if s > 0.0 { 1.0 - s + s * 3.0 * r2 * r2 } else if s < 0.0 { 1.0 + s - s * 2.0 * (1.0 - r2) } else { 1.0 }
}

/// The bloom and the final pass, with the final pass done through any lens and with the lens's own
/// colour fringing and fall-off of light. The plain pass is kept as it stands but for three lines, which
/// are handed to the camera's functions (camera_post.wgsl): the depth for the focus aid, the picture
/// before bloom is added to it (fringing), and the line after the vignette (fall-off).
pub fn post(src: &str) -> String {
    let s = append(src, "struct F {", "  lens: Cam,              // the lens (see the end of this text)\n  fx: vec4f,              // x: colour fringing, as the share by which the two ends of the spectrum are drawn larger and smaller   y: how much of its fall-off of light the lens keeps   z: how much its barrel cuts into its opening (cat's-eye)\n");
    let s = swap(&s, "  let e = t / length(vec3f(q * fu.view.y, 1.0)) - fu.view.x;\n", "  let e = lensDepth(q, t);\n");
    let s = swap(&s, "  var c = s.rgb * fu.glare.rgb + textureSampleLevel(bloomTex, flin, uv, 0.0).rgb * fu.bloom_amt;\n", "  var c = lensFringe(s.rgb, q) * fu.glare.rgb + textureSampleLevel(bloomTex, flin, uv, 0.0).rgb * fu.bloom_amt;\n");
    let s = swap(&s, "  c *= 1.0 - fu.vignette * dot(v, v);\n", "  c *= 1.0 - fu.vignette * dot(v, v);\n  c *= lensLight(q);\n");
    format!("{s}{FINAL_WGSL}{LENS_WGSL}")
}

/// How far apart the two ends of the spectrum are drawn at `fringe` = 1: this share of the distance from
/// the lens's axis, either way. (Four pixels at the corner of a picture 1080 high.)
pub const FRINGE: f64 = 0.004;

/// GLASSFORMS_LENS=1 (read once, when the program starts) has every picture drawn through the camera's
/// arithmetic, the plain lens too: that is how the checks hold the two texts of each shader to the
/// same picture.
pub fn always() -> bool {
    static ALWAYS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ALWAYS.get_or_init(|| std::env::var_os("GLASSFORMS_LENS").is_some())
}

/// Does the tracer have to trace its first ray through the camera's own arithmetic? Not for the plain
/// lens. `traced_lens`: rays are traced through the lens opening (so the shape of its highlights is the
/// tracer's to draw too).
pub fn wanted(p: &Settings, traced_lens: bool) -> bool {
    always()
        || !Camera::of(p).plain()
        || (traced_lens && p.aperture > 0.0 && (p.squeeze > 1.0 || p.bokeh != 0.0 || p.catseye > 0.0))
}

/// The lens for the tracer. With focus done to the finished picture the tracer has a pinhole, and the
/// shape of the highlights is none of its business (so that changing it does not start the picture again).
pub fn tracer_cam(p: &Settings, traced_lens: bool) -> CamU {
    let c = Camera::of(p);
    if traced_lens { c.uniform(p.squeeze, p.bokeh, p.catseye) } else { c.uniform(1.0, 0.0, 0.0) }
}

/// What the passes after the tracer are told about the lens.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lens {
    pub cam: CamU,
    /// colour fringing: the share by which the two ends of the spectrum are drawn larger and smaller
    pub fringe: f32,
    /// how much of its own fall-off of light towards the corners the lens keeps (0 = none)
    pub falloff: f32,
    /// how much the lens's barrel cuts into its opening towards the corners (cat's-eye highlights): the
    /// corners are darkened by the share cut off. Nothing with no lens opening.
    pub eye: f32,
}

impl Lens {
    pub fn of(p: &Settings) -> Lens {
        Lens {
            cam: Camera::of(p).uniform(p.squeeze, p.bokeh, p.catseye),
            fringe: (p.fringe.clamp(0.0, 1.0) * FRINGE) as f32,
            falloff: p.falloff.clamp(0.0, 1.0) as f32,
            eye: if p.aperture > 0.0 { p.catseye.clamp(0.0, 1.0) as f32 } else { 0.0 },
        }
    }
    /// The finished picture has something of the lens's own done to it.
    pub fn character(&self) -> bool {
        self.fringe > 0.0 || self.falloff > 0.0 || self.eye > 0.0
    }
}

/// The numbers the focus stage works from when the lens is not the plain one (`Dof::encode` has the
/// plain lens's written out).
pub struct FocusNumbers {
    /// lens radius over pixel size: what the lens's blur (camera.wgsl: camBlur) is multiplied by
    pub k: f32,
    /// the most blur, in pixels, of anything in front of the band and of anything behind it
    pub near: f32,
    pub far: f32,
    /// the blur of what lies behind all the glass, infinitely far away
    pub bg: f32,
}

/// Those numbers for a picture `w` x `h`. The most blur there can be is looked for where rays enter and
/// leave the sphere the object fits in, over a spread of pixels.
pub fn focus_numbers(c: &Camera, w: u32, h: u32) -> FocusNumbers {
    let (hf, aspect) = (h as f64, w as f64 / h as f64);
    let (mut near, mut far) = (0.0f64, 0.0f64);
    for j in 0..=8 {
        for i in 0..=12 {
            let uv = [(i as f64 / 6.0 - 1.0) * aspect, j as f64 / 4.0 - 1.0];
            let Some((o, d)) = c.ray(uv) else { continue };
            let b = o[0] * d[0] + o[1] * d[1] + o[2] * d[2];
            let disc = b * b - (o[0] * o[0] + o[1] * o[1] + o[2] * o[2] - BOUND * BOUND);
            if disc <= 0.0 { continue; }
            let (t0, t1) = ((-b - disc.sqrt()).max(0.02), -b + disc.sqrt());
            if let Some(r) = c.blur(uv, t0, hf) { near = near.max(-r); }
            if let Some(r) = c.blur(uv, t1, hf) { far = far.max(r); }
        }
    }
    // (rays between the ones tried may find a little more)
    let more = |x: f64| if x > 0.0 { x * 1.1 + 1.0 } else { 0.0 };
    let bg = c.blur([-c.shift[0], -c.shift[1]], 1e9, hf).unwrap_or(0.0).max(0.0);
    FocusNumbers { k: (c.lens / c.pixel(hf)) as f32, near: more(near) as f32, far: more(far) as f32, bg: bg as f32 }
}

/// The lens's own colour fringing and fall-off of light done to a picture in plain arithmetic, as the
/// final pass does them (camera_post.wgsl), for a 16-bit still. `hdr`: RGBA floats, `w` x `h`, top row
/// first. Returns the picture's RGB with its colours drawn apart, and the share of its light each pixel
/// keeps; None if the lens has neither.
pub fn develop(hdr: &[f32], w: u32, h: u32, p: &Settings) -> Option<(Vec<f32>, Vec<f32>)> {
    let lens = Lens::of(p);
    if !lens.character() { return None; }
    let c = Camera::of(p);
    let (wu, hu) = (w as usize, h as usize);
    let (wf, hf) = (w as f64, h as f64);
    // the picture between its pixels, as the GPU reads it: edges held
    let read = |x: f64, y: f64, ch: usize| -> f64 {
        let (fx, fy) = ((x - 0.5).clamp(0.0, wf - 1.0), (y - 0.5).clamp(0.0, hf - 1.0));
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(wu - 1), (y0 + 1).min(hu - 1));
        let (tx, ty) = (fx - x0 as f64, fy - y0 as f64);
        let at = |xx: usize, yy: usize| hdr[(yy * wu + xx) * 4 + ch] as f64;
        (at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx) * (1.0 - ty) + (at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx) * ty
    };
    let fringe = lens.fringe as f64;
    let mut rgb = Vec::with_capacity(wu * hu * 3);
    let mut kept = Vec::with_capacity(wu * hu);
    for y in 0..hu {
        for x in 0..wu {
            // the pixel across and up from the picture's middle, and from the lens's axis
            let uv = [(2.0 * (x as f64 + 0.5) - wf) / hf, (2.0 * (hf - (y as f64 + 0.5)) - hf) / hf];
            let film = [uv[0] + c.shift[0], uv[1] + c.shift[1]];
            if fringe > 0.0 {
                let mut sum = [0.0f64; 3];
                for i in 0..7 {
                    let k = i as f64 / 3.0 - 1.0;
                    let at = [((film[0] * (1.0 + fringe * k) - c.shift[0]) * hf + wf) * 0.5, ((film[1] * (1.0 + fringe * k) - c.shift[1]) * hf + hf) * 0.5];
                    let part = [k.max(0.0) * 0.5, (1.0 - k.abs()) / 3.0, (-k).max(0.0) * 0.5];
                    for ch in 0..3 {
                        if part[ch] > 0.0 { sum[ch] += part[ch] * read(at[0], hf - at[1], ch); }
                    }
                }
                rgb.extend(sum.iter().map(|v| *v as f32));
            } else {
                let i = (y * wu + x) * 4;
                rgb.extend_from_slice(&hdr[i..i + 3]);
            }
            let open = if lens.eye > 0.0 { gf_core::camera::open_share(gf_core::camera::cut(film, lens.eye as f64)) } else { 1.0 };
            kept.push((open * if lens.falloff > 0.0 { c.falloff(uv).max(0.0).powf(lens.falloff as f64) } else { 1.0 }) as f32);
        }
    }
    Some((rgb, kept))
}

/// A video's shutter from whichever way it was given: as the share of the frame's time it is open, as a
/// film camera's shutter angle, or as an exposure time (`speed` = 120 means 1/120 of a second).
pub fn shutter_arg(share: Option<f64>, angle: Option<f64>, speed: Option<f64>, fps: f64) -> f64 {
    use gf_core::camera::{shutter_of_angle, shutter_of_speed};
    speed.map(|s| shutter_of_speed(s, fps)).or(angle.map(shutter_of_angle)).or(share).unwrap_or(0.0)
}

/// The camera as a photographer would read it, as a line of JSON (lengths in millimetres, angles in
/// degrees; null where there is none: no f-stop with no lens opening, no far limit when sharp to the horizon).
pub fn readout_json(st: &gf_core::state::State, aspect: f64) -> String {
    let c = st.camera();
    let r = c.readout(aspect);
    let p = &st.p;
    let num = |v: f64| if v.is_finite() { serde_json::json!(v) } else { serde_json::Value::Null };
    serde_json::json!({
        "proj": c.proj.id(), "focal": r.focal, "sensor": r.sensor, "fstop": r.fstop, "subject": p.subject, "unitMm": c.unit,
        "fieldV": r.field_v, "fieldH": r.field_h, "fieldD": r.field_d, "viewMm": r.view_mm,
        "distanceMm": r.distance_mm, "focusMm": r.focus_mm, "bandMm": r.band_mm, "magnification": r.magnification,
        "sharpFromMm": r.sharp_from_mm, "sharpToMm": num(r.sharp_to_mm), "hyperfocalMm": r.hyperfocal_mm,
        "engravedFocal": r.engraved_focal, "engravedFstop": r.engraved_fstop, "notes": r.notes,
        "shutter": st.render.shutter, "shutterAngle": gf_core::camera::shutter_angle(st.render.shutter),
        "stored": { "fov": p.fov, "dist": p.dist, "aperture": p.aperture, "fnear": p.fnear, "ffar": p.ffar, "focal": p.focal, "fstop": p.fstop, "exposure": p.exposure, "shift": p.shift },
        "lensVariant": wanted(p, p.dof < 0.5),
    }).to_string()
}

/// The camera's own pipelines and the state of its tracer, as the renderer keeps them.
#[derive(Default)]
pub struct Pipes {
    /// the lens calls for the tracer's first ray to go through the camera's arithmetic
    pub wanted: bool,
    /// the tracer in use does that (it follows `wanted` once that tracer is built)
    pub active: bool,
    accum: OnceCell<wgpu::RenderPipeline>,
    /// the final pass, to the screen and to the 16-bit screen
    last: OnceCell<[wgpu::RenderPipeline; 2]>,
}

impl Pipes {
    /// The pass that folds a frame into the picture, through any lens.
    pub fn accum(&self, gpu: &Gpu, plain: &str, layout: &wgpu::BindGroupLayout, format: wgpu::TextureFormat) -> &wgpu::RenderPipeline {
        self.accum.get_or_init(|| {
            let module = gpu.shader("accumulate (lens)", &accum(plain), false);
            gpu.pipeline(&PipeDesc { label: "accumulate (lens)", module: &module, vs: "vs", fs: "fs", layouts: &[layout], format, blend: None, strip: false, buffers: &[] })
        })
    }
    /// The final pass through any lens: `deep` = to the 16-bit screen.
    pub fn last(&self, gpu: &Gpu, plain: &str, layout: &wgpu::BindGroupLayout, formats: [wgpu::TextureFormat; 2], deep: bool) -> &wgpu::RenderPipeline {
        let both = self.last.get_or_init(|| {
            let module = gpu.shader("post (lens)", &post(plain), true);
            formats.map(|format| gpu.pipeline(&PipeDesc { label: "final (lens)", module: &module, vs: "vs", fs: "fs_final", layouts: &[layout], format, blend: None, strip: false, buffers: &[] }))
        });
        &both[deep as usize]
    }
}
