//! Environment pictures (HDRI): a panorama read from a file, tamed, and put on the GPU as a cube whose
//! levels are ever softer copies of it; and the small picture of the lighting that a light room shows.
//!
//! What happens to a picture between the file and the glass, and why:
//!
//!   read      `.exr` (the `exr` crate, unpacking on several threads) or Radiance `.hdr` (read here).
//!             The picture is taken to be a whole panorama: its middle is what lies straight ahead of
//!             the camera when nothing is turned (behind the shape), its top row is straight up.
//!   level     Pictures come at any brightness (a room may average 0.02, a sunny sky 2). The picture is
//!             scaled so that its average over the whole sphere is `MEAN`, about what a family's own
//!             lights average, so that `hdriGain` = 1 looks right for any file.
//!   ceiling   A sun is tens of thousands of times brighter than the sky and a few texels wide. Glass
//!             focuses it into specks that a pixel catches once in thousands of frames, and no amount of
//!             averaging cleans that up; behind the shape it would drown the picture in bloom. So nothing
//!             in the stored picture is brighter than `CEILING` (colours keep their hue).
//!   glow      The light the ceiling took off is not thrown away, or a sunny picture would light the glass
//!             like an overcast one. It is handed back as a wide soft glow around where it came from, wide
//!             enough that its middle is no brighter than `GLOW_PEAK`: the sun as a large soft lamp, which
//!             glass shows cleanly. The glows are painted into the lights' map, not into the picture, so
//!             the picture shown behind the shape stays the picture.
//!   cube      The tracer reads the picture from a cube (no poles, no seam: the GPU blends across the
//!             edges), with a chain of levels each half the size and softened by a bell-shaped blur of
//!             twice the width, measured as an angle on the sphere. `hdriBlur` picks the level the glass
//!             reads; the level blended between two of them is done by the GPU.
//!
//! Reading and taming a 4096 x 2048 picture takes most of a second, so it is done on another thread, GPU
//! work included (the device may be used from any thread); the engine carries on with the lighting it
//! had and switches over when the cube is ready.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, TryRecvError};
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use gf_core::env::{self, Glow, Sh9};

use crate::gpu::*;

const HDRI_WGSL: &str = include_str!("shaders/hdri.wgsl");
const HDR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The average brightness, over the whole sphere, that a picture is brought to.
pub const MEAN: f64 = 0.25;
/// Nothing in the stored picture is brighter than this (once brought to `MEAN`).
pub const CEILING: f64 = 12.0;
/// Light taken off by the ceiling comes back as a glow no brighter than this at its middle, and
/// between these widths (radians: 2 and 6 degrees).
pub const GLOW_PEAK: f64 = 8.0;
pub const GLOW_NARROW: f64 = 0.0349;
pub const GLOW_WIDE: f64 = 0.1047;
/// The largest cube face made (a face this size holds all there is in a picture about 6400 wide).
pub const FACE_MAX: u32 = 2048;
/// The largest picture read: 16384 x 8192.
const TEXELS_MAX: usize = 16384 * 8192;

/// The size of the small picture of the lighting.
pub const SHOW_W: u32 = 512;
pub const SHOW_H: u32 = 256;

// ---- where pictures are ------------------------------------------------------------------------------

/// The program's folder: where the program file is, or the workspace when it is run straight from a
/// build (`target\release`). Pictures given without a full path are looked for here, and the `hdri`
/// folder of this folder is where a file chooser should open.
pub fn home() -> PathBuf {
    let exe = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)).unwrap_or_else(|| PathBuf::from("."));
    match exe.parent().and_then(Path::parent) {
        Some(root) if root.join("Cargo.toml").exists() => root.to_path_buf(),
        _ => exe,
    }
}

/// The file a given path means: as it is if it is a full path, otherwise in the program's folder
/// (or, failing that, wherever the program was started from).
pub fn resolve(given: &str) -> PathBuf {
    let p = Path::new(given);
    if p.is_absolute() { return p.to_path_buf(); }
    let here = home().join(p);
    if here.exists() || !p.exists() { here } else { p.to_path_buf() }
}

/// A path as a setup should keep it: relative to the program's folder (with forward slashes) when the
/// file is in it, so that the folder can be moved or copied to another PC; otherwise as given.
pub fn tidy(given: &str) -> String {
    let full = std::path::absolute(resolve(given)).unwrap_or_else(|_| PathBuf::from(given));
    let root = std::path::absolute(home()).unwrap_or_else(|_| home());
    let (a, b): (Vec<_>, Vec<_>) = (full.components().collect(), root.components().collect());
    // (Windows does not tell D:\Glass from d:\glass)
    let same = |x: &std::path::Component, y: &std::path::Component| x.as_os_str().to_string_lossy().eq_ignore_ascii_case(&y.as_os_str().to_string_lossy());
    if a.len() > b.len() && a.iter().zip(&b).all(|(x, y)| same(x, y)) {
        return a[b.len()..].iter().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect::<Vec<_>>().join("/");
    }
    given.to_string()
}

// ---- reading -----------------------------------------------------------------------------------------

/// A panorama as it came from the file: red, green, blue floats, top row first.
pub struct Panorama {
    pub w: usize,
    pub h: usize,
    pub rgb: Vec<f32>,
}

/// Read an `.exr` or `.hdr` file (told apart by how the file begins, not by its name).
pub fn read(path: &Path) -> R<Panorama> {
    use std::io::Read;
    let mut start = [0u8; 4];
    let got = std::fs::File::open(path).and_then(|mut f| f.read(&mut start)).map_err(|e| format!("{}: {e}", path.display()))?;
    let pano = if got == 4 && start == [0x76, 0x2f, 0x31, 0x01] {
        read_exr(path)?
    } else if got >= 2 && &start[..2] == b"#?" {
        read_hdr(&std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?)?
    } else {
        return Err(format!("{}: not an OpenEXR (.exr) or Radiance (.hdr) picture", path.display()));
    };
    if pano.w < 8 || pano.h < 4 { return Err(format!("{}: a picture of {} x {} is too small to light anything", path.display(), pano.w, pano.h)); }
    if pano.w * pano.h > TEXELS_MAX { return Err(format!("{}: {} x {} is more than this program takes (16384 x 8192)", path.display(), pano.w, pano.h)); }
    Ok(pano)
}

fn read_exr(path: &Path) -> R<Panorama> {
    use exr::prelude::*;
    let image = read()
        .no_deep_data()
        .largest_resolution_level()
        .rgb_channels(
            |size, _| Panorama { w: size.width(), h: size.height(), rgb: vec![0.0f32; size.width() * size.height() * 3] },
            |p: &mut Panorama, at, (r, g, b): (f32, f32, f32)| {
                let o = (at.y() * p.w + at.x()) * 3;
                p.rgb[o] = r;
                p.rgb[o + 1] = g;
                p.rgb[o + 2] = b;
            },
        )
        .first_valid_layer()
        .all_attributes()
        .from_file(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(image.layer_data.channel_data.pixels)
}

/// A Radiance picture: a few lines of text, the size, then every row as four bytes a pixel (three
/// colours and a shared exponent), usually run-length packed a colour at a time.
pub fn read_hdr(bytes: &[u8]) -> R<Panorama> {
    let bad = |why: &str| format!("not a Radiance picture this program can read: {why}");
    let mut at = 0usize;
    fn next_line<'a>(bytes: &'a [u8], at: &mut usize) -> Option<&'a [u8]> {
        let rest = bytes.get(*at..)?;
        let end = rest.iter().position(|&c| c == b'\n')?;
        *at += end + 1;
        Some(&rest[..end])
    }
    let line = |at: &mut usize| next_line(bytes, at);
    let first = line(&mut at).ok_or_else(|| bad("it has no header"))?;
    if !first.starts_with(b"#?") { return Err(bad("it does not begin with #?RADIANCE")); }
    loop {
        let l = line(&mut at).ok_or_else(|| bad("its header does not end"))?;
        if l.is_empty() || l == b"\r" { break; }
        if l.starts_with(b"FORMAT=") && !l.starts_with(b"FORMAT=32-bit_rle_rgbe") { return Err(bad("its colours are not red, green and blue")); }
    }
    let size = String::from_utf8_lossy(line(&mut at).ok_or_else(|| bad("it gives no size"))?).into_owned();
    let parts: Vec<&str> = size.split_whitespace().collect();
    let num = |s: &str| s.parse::<usize>().ok().filter(|n| *n > 0);
    let (h, w, flip) = match parts.as_slice() {
        ["-Y", h, "+X", w] => (num(h), num(w), false),
        ["+Y", h, "+X", w] => (num(h), num(w), true),
        _ => return Err(bad("its rows are laid out in a way that is not supported")),
    };
    let (w, h) = (w.ok_or_else(|| bad("its size makes no sense"))?, h.ok_or_else(|| bad("its size makes no sense"))?);
    if w * h > TEXELS_MAX { return Err(format!("{w} x {h} is more than this program takes (16384 x 8192)")); }
    let mut rgb = vec![0.0f32; w * h * 3];
    let mut row = vec![0u8; w * 4];
    let short = || bad("it ends before its last row");
    for j in 0..h {
        let head = bytes.get(at..at + 4).ok_or_else(short)?;
        if (8..32768).contains(&w) && head[0] == 2 && head[1] == 2 && ((head[2] as usize) << 8 | head[3] as usize) == w {
            // packed: each of the four bytes of a pixel as its own run-length coded row
            at += 4;
            for c in 0..4 {
                let mut x = 0;
                while x < w {
                    let n = *bytes.get(at).ok_or_else(short)? as usize;
                    at += 1;
                    if n > 128 {
                        let (n, v) = (n - 128, *bytes.get(at).ok_or_else(short)?);
                        at += 1;
                        if x + n > w { return Err(bad("a row is longer than the picture is wide")); }
                        for k in 0..n { row[(x + k) * 4 + c] = v; }
                        x += n;
                    } else {
                        if n == 0 || x + n > w { return Err(bad("a row is longer than the picture is wide")); }
                        let vals = bytes.get(at..at + n).ok_or_else(short)?;
                        at += n;
                        for (k, v) in vals.iter().enumerate() { row[(x + k) * 4 + c] = *v; }
                        x += n;
                    }
                }
            }
        } else {
            // plain: four bytes a pixel (a pixel of 1, 1, 1, n repeats the one before n times: the old packing)
            let (mut x, mut shift) = (0usize, 0u32);
            while x < w {
                let p = bytes.get(at..at + 4).ok_or_else(short)?;
                at += 4;
                if p[0] == 1 && p[1] == 1 && p[2] == 1 && x > 0 {
                    let n = (p[3] as usize) << shift;
                    if x + n > w { return Err(bad("a row is longer than the picture is wide")); }
                    for k in 0..n { row.copy_within((x - 1) * 4..x * 4, (x + k) * 4); }
                    x += n;
                    shift += 8;
                } else {
                    row[x * 4..x * 4 + 4].copy_from_slice(p);
                    x += 1;
                    shift = 0;
                }
            }
        }
        let out = if flip { h - 1 - j } else { j } * w * 3;
        for x in 0..w {
            let e = row[x * 4 + 3];
            if e == 0 { continue; }
            let f = 2f32.powi(e as i32 - 136);
            for c in 0..3 { rgb[out + x * 3 + c] = (row[x * 4 + c] as f32 + 0.5) * f; }
        }
    }
    Ok(Panorama { w, h, rgb })
}

// ---- taming ------------------------------------------------------------------------------------------

/// A panorama ready for the GPU, and what else the lighting needs to know of it.
pub struct Tamed {
    pub w: usize,
    pub h: usize,
    /// red, green, blue, one: 16-bit floats, top row first; brought to `MEAN` and held under `CEILING`
    pub half: Vec<u16>,
    /// the light the ceiling took off, to be given back as glows (in the lights' own frame)
    pub glows: Vec<Glow>,
    /// the whole lighting of the picture (glows included) as nine smooth terms, for clay
    pub sh: Sh9,
    /// what the file's numbers were multiplied by
    pub scale: f64,
    /// the brightest number in the file
    pub peak: f64,
    /// the share of the picture's light that the ceiling took off, and the share the glows give back
    pub taken_off: f64,
    pub given_back: f64,
}

/// The direction (in the lights' own frame) of a point of a panorama: u, v are fractions of its width
/// and height from the top left. The same layout as the small picture of the lighting.
pub fn panorama_dir(u: f64, v: f64) -> [f64; 3] {
    env::preview_dir(u, v)
}

/// Run `f` on bands of rows on several threads: f(first row, rows of the picture, rows of the output).
fn bands<T: Send, A: Send, B: Send>(h: usize, a: &mut [A], b: &mut [B], f: impl Fn(usize, &mut [A], &mut [B]) -> T + Sync) -> Vec<T> {
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
    let rows = h.div_ceil(threads).max(1);
    let (pa, pb) = (a.len() / h, b.len() / h);
    std::thread::scope(|s| {
        let f = &f;
        let jobs: Vec<_> = a.chunks_mut(rows * pa).zip(b.chunks_mut(rows * pb)).enumerate().map(|(i, (ca, cb))| s.spawn(move || f(i * rows, ca, cb))).collect();
        jobs.into_iter().map(|j| j.join().expect("a worker stopped")).collect()
    })
}

const LUM: [f64; 3] = [0.2126, 0.7152, 0.0722];
/// The grid the taken-off light is gathered on before it is grouped into glows.
const GW: usize = 64;
const GH: usize = 32;

pub fn tame(mut p: Panorama) -> R<Tamed> {
    use std::f64::consts::PI;
    let (w, h) = (p.w, p.h);
    // the patch of sky a texel of each row covers
    let dw: Vec<f64> = (0..h).map(|j| (2.0 * PI / w as f64) * ((j as f64 / h as f64 * PI).cos() - ((j + 1) as f64 / h as f64 * PI).cos())).collect();

    // 1. How bright is it? Count the light by how bright its brightest colour is, so that the scale which
    //    brings the average to MEAN, with the ceiling in force, can be found without another pass.
    const BINS: usize = 1440;                                // 24 to the doubling, from 2^-30 to 2^30
    let bin_of = |m: f64| (((m.log2() + 30.0) * 24.0) as isize).clamp(0, BINS as isize - 1) as usize;
    let centre = |i: usize| 2f64.powf((i as f64 + 0.5) / 24.0 - 30.0);
    let mut none: Vec<u8> = vec![0; h];
    let parts = bands(h, &mut p.rgb, &mut none, |j0, rows, _| {
        let (mut light, mut flat, mut peak) = (vec![0.0f64; BINS], vec![0.0f64; BINS], 0.0f64);
        for (r, row) in rows.chunks_mut(w * 3).enumerate() {
            let d = dw[j0 + r];
            for px in row.chunks_mut(3) {
                // (a number that is not one, or is below nothing, is taken as no light)
                for c in px.iter_mut() { if !c.is_finite() || *c < 0.0 { *c = 0.0; } }
                let m = px[0].max(px[1]).max(px[2]) as f64;
                if m <= 0.0 { continue; }
                let y = LUM[0] * px[0] as f64 + LUM[1] * px[1] as f64 + LUM[2] * px[2] as f64;
                let i = bin_of(m);
                light[i] += d * y;
                flat[i] += d * y / m;
                peak = peak.max(m);
            }
        }
        (light, flat, peak)
    });
    let (mut light, mut flat, mut peak) = (vec![0.0f64; BINS], vec![0.0f64; BINS], 0.0f64);
    for (l, f, pk) in &parts {
        for i in 0..BINS { light[i] += l[i]; flat[i] += f[i]; }
        peak = peak.max(*pk);
    }
    if peak <= 0.0 { return Err("the picture is black all over".into()); }
    // the average brightness the picture would have at scale s, nothing brighter than the ceiling
    let mean_at = |s: f64| (0..BINS).map(|i| if s * centre(i) <= CEILING { s * light[i] } else { CEILING * flat[i] }).sum::<f64>() / (4.0 * PI);
    // (a picture that is nearly all black, a lamp in a dark room, cannot reach MEAN however far it is
    // scaled: it is brought half way to the most it can reach, which leaves the lamp about half the ceiling)
    let most = CEILING * flat.iter().sum::<f64>() / (4.0 * PI);
    let want = MEAN.min(0.5 * most);
    let (mut lo, mut hi) = (-40.0f64, 40.0f64);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if mean_at(2f64.powf(mid)) < want { lo = mid; } else { hi = mid; }
    }
    let scale = 2f64.powf(0.5 * (lo + hi));

    // 2. Scale, hold under the ceiling, turn into 16-bit floats; gather what the ceiling took off on a
    //    coarse grid (how much, and from where), and a small copy of the picture for the nine terms.
    let k = w.div_ceil(256);
    let (sw, sh_) = (w.div_ceil(k), h.div_ceil(k));
    let mut half = vec![0u16; w * h * 4];
    let one = half::f16::from_f32(1.0).to_bits();
    struct Part { off: Vec<[f64; 7]>, small: Vec<f64>, all: f64 }
    let parts = bands(h, &mut p.rgb, &mut half, |j0, rows, out| {
        let mut part = Part { off: vec![[0.0; 7]; GW * GH], small: vec![0.0; sw * sh_ * 3], all: 0.0 };
        for (r, (row, orow)) in rows.chunks(w * 3).zip(out.chunks_mut(w * 4)).enumerate() {
            let j = j0 + r;
            let d = dw[j];
            let v = (j as f64 + 0.5) / h as f64;
            for (i, (px, o)) in row.chunks(3).zip(orow.chunks_mut(4)).enumerate() {
                let mut c = [px[0] as f64 * scale, px[1] as f64 * scale, px[2] as f64 * scale];
                let m = c[0].max(c[1]).max(c[2]);
                part.all += d * (LUM[0] * c[0] + LUM[1] * c[1] + LUM[2] * c[2]);
                if m > CEILING {
                    let keep = CEILING / m;
                    let cell = &mut part.off[(j * GH / h) * GW + i * GW / w];
                    let dir = panorama_dir((i as f64 + 0.5) / w as f64, v);
                    let mut y = 0.0;
                    for q in 0..3 {
                        let e = c[q] * (1.0 - keep) * d;
                        cell[q] += e;
                        y += LUM[q] * e;
                        c[q] *= keep;
                    }
                    for q in 0..3 { cell[3 + q] += dir[q] * y; }
                    cell[6] += y;
                }
                let s = ((j / k) * sw + i / k) * 3;
                for q in 0..3 {
                    part.small[s + q] += c[q] * d;
                    o[q] = half::f16::from_f32(c[q] as f32).to_bits();
                }
                o[3] = one;
            }
        }
        part
    });
    drop(p);
    let mut off = vec![[0.0f64; 7]; GW * GH];
    let mut small = vec![0.0f64; sw * sh_ * 3];
    let mut all = 0.0;
    for part in &parts {
        for (a, b) in off.iter_mut().zip(&part.off) { for q in 0..7 { a[q] += b[q]; } }
        for (a, b) in small.iter_mut().zip(&part.small) { *a += b; }
        all += part.all;
    }

    // 3. Group what was taken off into glows: the brightest cell and everything within 20 degrees of it
    //    is one glow, then the brightest of what is left, and so on; what little is left after twelve
    //    is let go. A glow is as wide as it must be for its middle to be no brighter than GLOW_PEAK.
    let taken: f64 = off.iter().map(|c| c[6]).sum();
    let mut given = 0.0;
    let mut glows = Vec::new();
    let unit = |c: &[f64; 7]| {
        let l = (c[3] * c[3] + c[4] * c[4] + c[5] * c[5]).sqrt().max(1e-30);
        [c[3] / l, c[4] / l, c[5] / l]
    };
    while glows.len() < 12 {
        let Some(best) = (0..off.len()).filter(|&i| off[i][6] > 0.0).max_by(|&a, &b| off[a][6].total_cmp(&off[b][6])) else { break };
        let mid = unit(&off[best]);
        let mut sum = [0.0f64; 7];
        for c in off.iter_mut() {
            if c[6] <= 0.0 { continue; }
            let u = unit(c);
            if u[0] * mid[0] + u[1] * mid[1] + u[2] * mid[2] > 0.94 {
                for q in 0..7 { sum[q] += c[q]; }
                *c = [0.0; 7];
            }
        }
        if sum[6] < 0.002 * taken { break; }
        let sigma = (sum[6] / (2.0 * PI * GLOW_PEAK)).sqrt().clamp(GLOW_NARROW, GLOW_WIDE);
        // (a sun's light does not fit in a glow that narrow and that dim: what does not fit is let go)
        let fits = (2.0 * PI * sigma * sigma * GLOW_PEAK / sum[6]).min(1.0);
        given += sum[6] * fits;
        glows.push(Glow { dir: unit(&sum), energy: [sum[0] * fits, sum[1] * fits, sum[2] * fits], sigma });
    }

    // 4. The nine terms: of the small copy (each texel holds light, so divide by the patch of sky it
    //    stands for), and of the glows.
    let mut small_f = vec![0.0f32; sw * sh_ * 3];
    for j in 0..sh_ {
        let patch = (2.0 * PI / sw as f64) * ((j as f64 / sh_ as f64 * PI).cos() - ((j + 1) as f64 / sh_ as f64 * PI).cos());
        for i in 0..sw * 3 { small_f[j * sw * 3 + i] = (small[j * sw * 3 + i] / patch) as f32; }
    }
    // (when the picture's size is not a multiple of the small copy's, the last row and column stand for
    // slightly less sky than the rest: the terms are smooth enough not to notice)
    let mut sh = env::sh_of_map(&small_f, sw, sh_, 3, panorama_dir);
    for g in &glows { env::sh_add(&mut sh, &env::sh_of_glow(g), 1.0); }
    let share = |x: f64| if all > 0.0 { x / all } else { 0.0 };
    Ok(Tamed { w, h, half, glows, sh, scale, peak, taken_off: share(taken), given_back: share(given) })
}

// ---- on the GPU --------------------------------------------------------------------------------------

/// What there is to say about a picture in use.
#[derive(Clone, Debug)]
pub struct Info {
    /// the file's name, and where it is
    pub file: String,
    pub path: PathBuf,
    pub w: u32,
    pub h: u32,
    /// texels along a side of the cube's faces, how many levels, and the video memory they take
    pub face: u32,
    pub levels: u32,
    pub bytes: u64,
    /// what the file's numbers were multiplied by; the brightest of them; the share of the light that
    /// the ceiling took off, the share given back, and by how many glows
    pub scale: f64,
    pub peak: f64,
    pub taken_off: f64,
    pub given_back: f64,
    pub glows: usize,
    /// milliseconds: reading the file, taming, handing over to the GPU, the GPU making the cube
    pub read_ms: f64,
    pub tame_ms: f64,
    pub upload_ms: f64,
    pub gpu_ms: f64,
}

/// An environment picture on the GPU.
pub struct Hdri {
    pub tex: wgpu::Texture,
    /// the whole cube, every level
    pub view: wgpu::TextureView,
    pub levels: u32,
    /// the angle one texel of the sharpest level covers at the middle of a face, radians
    pub texel: f64,
    pub glows: Vec<Glow>,
    pub sh: Sh9,
    pub info: Info,
}

impl Hdri {
    /// Texels along a side of a face of one level.
    pub fn size(&self, level: u32) -> u32 {
        (self.tex.width() >> level).max(1)
    }

    /// The level whose softening is `sigma` radians wide (a level's width is its texel's angle;
    /// each level doubles it).
    pub fn level_for(&self, sigma: f64) -> f32 {
        if sigma <= 0.0 { return 0.0; }
        (sigma / self.texel).log2().clamp(0.0, (self.levels - 1) as f64) as f32
    }
}

/// How wide (radians) the softening is at a setting of `hdriBlur`: nothing at 0, 5 degrees at the
/// default 0.5, 60 degrees at 1, each step of the slider multiplying the width.
pub fn blur_sigma(blur: f64) -> f64 {
    if blur <= 0.0 { return 0.0; }
    let (least, most) = (0.1f64.to_radians(), 60f64.to_radians());
    least * (most / least).powf(blur.min(1.0).powf(0.7))
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MakeU {
    size: f32,
    taps: f32,
    sigma: f32,
    pad: f32,
}

/// What the small picture of the lighting is drawn from.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq, Debug)]
pub struct ShowU {
    /// what the camera sees -> the lights' own frame (the tracer's `env`); only the small picture uses it
    pub env: [[f32; 4]; 3],
    /// 1 = the lights' map is lit; brightness of the picture (0 = none); the level of it shown; unused
    pub k: [f32; 4],
    pub size: [f32; 4],
}

/// The pipelines that turn a panorama into a cube. One is made per engine and shared with the threads
/// that load pictures.
pub struct Maker {
    gpu: Gpu,
    l_face: wgpu::BindGroupLayout,
    l_soft: wgpu::BindGroupLayout,
    l_show: wgpu::BindGroupLayout,
    p_face: wgpu::RenderPipeline,
    p_soft: wgpu::RenderPipeline,
    p_show: wgpu::RenderPipeline,
    p_glass: wgpu::RenderPipeline,
    /// repeats round the horizon, holds at the poles
    wrap: wgpu::Sampler,
    /// blends within a level and between levels
    pub smooth: wgpu::Sampler,
    /// a cube with nothing on it, for when there is no picture
    pub black: wgpu::TextureView,
    u_show: wgpu::Buffer,
}

/// Wait until the GPU has done what it has been handed so far, by asking now and then rather than by
/// blocking on it: a thread that blocks keeps hold of the device meanwhile, and a frame being drawn on
/// another thread would wait behind it.
fn settle(gpu: &Gpu) {
    use std::sync::atomic::{AtomicBool, Ordering};
    let done = Arc::new(AtomicBool::new(false));
    let flag = done.clone();
    gpu.queue.on_submitted_work_done(move || flag.store(true, Ordering::SeqCst));
    loop {
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        if done.load(Ordering::SeqCst) { break; }
        std::thread::sleep(std::time::Duration::from_micros(250));
    }
}

fn cube(gpu: &Gpu, label: &str, face: u32, levels: u32) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: face, height: face, depth_or_array_layers: 6 },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HDR,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}
/// One level of a cube, to read from.
fn level_view(tex: &wgpu::Texture, level: u32) -> wgpu::TextureView {
    tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::Cube), base_mip_level: level, mip_level_count: Some(1), ..Default::default() })
}
/// One face of one level, to draw into.
fn face_view(tex: &wgpu::Texture, level: u32, face: u32) -> wgpu::TextureView {
    tex.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2),
        base_mip_level: level,
        mip_level_count: Some(1),
        base_array_layer: face,
        array_layer_count: Some(1),
        ..Default::default()
    })
}

impl Maker {
    pub fn new(gpu: &Gpu) -> Self {
        let frag = wgpu::ShaderStages::FRAGMENT;
        let (d2, dcube) = (wgpu::TextureViewDimension::D2, wgpu::TextureViewDimension::Cube);
        let l_face = gpu.bind_layout("picture: face", &[uniform_entry(0, frag, false), texture_entry(1, d2, true), sampler_entry(2)]);
        let l_soft = gpu.bind_layout("picture: soften", &[uniform_entry(0, frag, false), texture_entry(3, dcube, true), sampler_entry(4)]);
        let l_show = gpu.bind_layout("lighting shown", &[sampler_entry(2), texture_entry(3, dcube, true), sampler_entry(4), uniform_entry(5, frag, false), texture_entry(6, d2, true)]);
        let module = gpu.shader("picture", HDRI_WGSL, false);
        let pipe = |label, vs, fs, layout: &wgpu::BindGroupLayout, format| gpu.pipeline(&PipeDesc { label, module: &module, vs, fs, layouts: &[layout], format, blend: None, strip: false, buffers: &[] });
        let smooth = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("picture"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let black = cube(gpu, "no picture", 1, 1);
        Self {
            gpu: gpu.clone(),
            p_face: pipe("picture: face", "vs", "fs_face", &l_face, HDR),
            p_soft: pipe("picture: soften", "vs", "fs_soft", &l_soft, HDR),
            p_show: pipe("lighting shown", "vs_show", "fs_show", &l_show, wgpu::TextureFormat::Rgba8Unorm),
            p_glass: pipe("lighting for the glass", "vs", "fs_glass", &l_show, HDR),
            l_face, l_soft, l_show,
            wrap: gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::Repeat),
            smooth,
            black: black.create_view(&wgpu::TextureViewDescriptor { dimension: Some(dcube), ..Default::default() }),
            u_show: gpu.uniform_buffer("lighting shown", std::mem::size_of::<ShowU>() as u64),
        }
    }

    /// Put a tamed panorama on the GPU as a cube with its softer levels. Returns the cube, how many
    /// levels, the angle of a texel, the memory taken, and the milliseconds the handing over and the
    /// GPU's own work took (the call waits for the GPU, so call it from a thread that may wait).
    fn make(&self, t: &Tamed) -> (wgpu::Texture, u32, f64, u64, f64, f64) {
        let gpu = &self.gpu;
        let t0 = Instant::now();
        // A face as fine as the panorama is where the cube is coarsest (the middle of a face): the
        // panorama has w texels round the horizon, a cube four faces of n, each wider at its middle.
        let want = (t.w as f64 / std::f64::consts::PI).ceil() as u32;
        let n = (want.div_ceil(4) * 4).clamp(16, FACE_MAX.min(gpu.device.limits().max_texture_dimension_2d));
        let levels = 32 - n.leading_zeros();
        let size = |level: u32| (n >> level).max(1);
        let pano = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("panorama"),
            size: wgpu::Extent3d { width: t.w as u32, height: t.h as u32, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: HDR,
            // (nothing is ever drawn into it, but a texture that could be drawn into is wiped by the GPU
            // in no time before its first part arrives; one that could not is wiped by copying zeros
            // into it, 67 megabytes of them for a 4096 x 2048 picture, which held a frame up for 70 ms)
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        // Handed over a few megabytes at a time, each sent on its way and waited for before the next, so
        // that a frame being drawn meanwhile waits behind one small copy at most. (Handed over whole, the
        // copy into video memory, 67 megabytes for a 4096 x 2048 picture, held a frame up for 70 to 100 ms.)
        let band = ((4usize << 20) / (t.w * 8)).max(1);
        let bytes: &[u8] = bytemuck::cast_slice(&t.half);
        for y in (0..t.h).step_by(band) {
            let rows = band.min(t.h - y);
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &pano, mip_level: 0, origin: wgpu::Origin3d { x: 0, y: y as u32, z: 0 }, aspect: wgpu::TextureAspect::All },
                &bytes[y * t.w * 8..(y + rows) * t.w * 8],
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(t.w as u32 * 8), rows_per_image: Some(rows as u32) },
                wgpu::Extent3d { width: t.w as u32, height: rows as u32, depth_or_array_layers: 1 },
            );
            gpu.queue.submit(std::iter::empty());
            settle(gpu);
        }
        let upload_ms = t0.elapsed().as_secs_f64() * 1e3;
        let pano_view = pano.create_view(&wgpu::TextureViewDescriptor::default());
        let tex = cube(gpu, "picture", n, levels);
        let uniform = |u: MakeU| {
            let b = gpu.uniform_buffer("picture", std::mem::size_of::<MakeU>() as u64);
            gpu.queue.write_buffer(&b, 0, bytemuck::bytes_of(&u));
            b
        };
        let soft_group = |u: &wgpu::Buffer, from: &wgpu::TextureView| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("picture: soften"),
                layout: &self.l_soft,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(from) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.smooth) },
                ],
            })
        };
        // The GPU is handed the work a face at a time and waited for after each, so that a frame being
        // drawn meanwhile waits behind one face at most and not behind the whole cube.
        let faces = |to: &wgpu::Texture, level: u32, pipe: &wgpu::RenderPipeline, group: &wgpu::BindGroup| {
            for face in 0..6 {
                let view = face_view(to, level, face);
                let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("picture") });
                {
                    let mut rp = pass(&mut enc, &view, None, true);
                    rp.set_pipeline(pipe);
                    rp.set_bind_group(0, group, &[]);
                    rp.draw(0..3, face..face + 1);
                }
                gpu.queue.submit([enc.finish()]);
                settle(gpu);
            }
        };
        // the sharpest level, straight from the panorama (several samples a texel only where the cube
        // had to be made coarser than the panorama)
        let per = (want as f64 / n as f64).ceil().clamp(1.0, 4.0) as f32;
        let u0 = uniform(MakeU { size: n as f32, taps: per, sigma: 0.0, pad: 0.0 });
        let g0 = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("picture: face"),
            layout: &self.l_face,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: u0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&pano_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.wrap) },
            ],
        });
        faces(&tex, 0, &self.p_face, &g0);
        // Each further level: the level above, softened so that the whole softening is as wide as one of
        // this level's texels (widths add as squares, so each step adds what is missing). The sharpest
        // level has detail down to a texel, which a bell of sixty-four samples would step over, so the
        // second level is made in two goes: an exact average of four texels into one first, then the bell.
        let angle = |level: u32| 2.0 / size(level) as f64;
        let mut have = (0.5 * angle(0)).powi(2);
        for level in 1..levels {
            let from = if level == 1 {
                let halfway = cube(gpu, "picture, averaged", size(1), 1);
                let u = uniform(MakeU { size: size(1) as f32, taps: 1.0, sigma: 0.0, pad: 0.0 });
                faces(&halfway, 0, &self.p_soft, &soft_group(&u, &level_view(&tex, 0)));
                have += angle(0).powi(2) / 3.0;
                level_view(&halfway, 0)
            } else {
                level_view(&tex, level - 1)
            };
            let sigma = (angle(level).powi(2) - have).max(1e-8).sqrt();
            let u = uniform(MakeU { size: size(level) as f32, taps: 64.0, sigma: sigma as f32, pad: 0.0 });
            faces(&tex, level, &self.p_soft, &soft_group(&u, &from));
            have = angle(level).powi(2);
        }
        let bytes = (0..levels).map(|l| 6 * (size(l) as u64).pow(2) * 8).sum();
        (tex, levels, angle(0), bytes, upload_ms, t0.elapsed().as_secs_f64() * 1e3 - upload_ms)
    }

    /// Read a picture, tame it and put it on the GPU: everything, start to finish, on the calling thread.
    pub fn load(&self, path: &Path) -> R<Hdri> {
        let t0 = Instant::now();
        let pano = read(path)?;
        let read_ms = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = Instant::now();
        let (w, h) = (pano.w as u32, pano.h as u32);
        let tamed = tame(pano).map_err(|e| format!("{}: {e}", path.display()))?;
        let tame_ms = t1.elapsed().as_secs_f64() * 1e3;
        let (tex, levels, texel, bytes, upload_ms, gpu_ms) = self.make(&tamed);
        let view = tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::Cube), ..Default::default() });
        let info = Info {
            file: path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
            path: path.to_path_buf(),
            w, h, face: tex.width(), levels, bytes,
            scale: tamed.scale, peak: tamed.peak, taken_off: tamed.taken_off, given_back: tamed.given_back, glows: tamed.glows.len(),
            read_ms, tame_ms, upload_ms, gpu_ms,
        };
        Ok(Hdri { tex, view, levels, texel, glows: tamed.glows, sh: tamed.sh, info })
    }

    /// A texture for the small picture of the lighting: display-ready bytes, like the screen's.
    pub fn show_target(&self) -> wgpu::Texture {
        self.gpu.target("lighting shown", SHOW_W, SHOW_H, wgpu::TextureFormat::Rgba8Unorm, &[wgpu::TextureFormat::Rgba8UnormSrgb])
    }

    /// A cube of one level, `face` texels a side, for the light the glass reads.
    pub fn glass_target(&self, face: u32) -> wgpu::Texture {
        cube(&self.gpu, "lighting for the glass", face, 1)
    }

    /// Fill the cube the glass reads: the picture at level `u.k[2]` and brightness `u.k[1]`, with the
    /// lights' map added in if `u.k[0]` says it is lit (`u.env` is not used: the cube is in the lights'
    /// own frame, like the picture).
    pub fn glass(&self, to: &wgpu::Texture, lights: &wgpu::TextureView, picture: &wgpu::TextureView, mut u: ShowU) {
        let gpu = &self.gpu;
        u.size = [to.width() as f32, to.height() as f32, 0.0, 0.0];
        // (a buffer of its own: the small picture's may be waiting to be drawn with other numbers)
        let buf = gpu.uniform_buffer("lighting for the glass", std::mem::size_of::<ShowU>() as u64);
        gpu.queue.write_buffer(&buf, 0, bytemuck::bytes_of(&u));
        let group = self.show_group(&buf, lights, picture);
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("lighting for the glass") });
        for face in 0..6 {
            let view = face_view(to, 0, face);
            let mut rp = pass(&mut enc, &view, None, true);
            rp.set_pipeline(&self.p_glass);
            rp.set_bind_group(0, &group, &[]);
            rp.draw(0..3, face..face + 1);
        }
        gpu.queue.submit([enc.finish()]);
    }

    fn show_group(&self, u: &wgpu::Buffer, lights: &wgpu::TextureView, picture: &wgpu::TextureView) -> wgpu::BindGroup {
        self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lighting shown"),
            layout: &self.l_show,
            entries: &[
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.wrap) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(picture) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.smooth) },
                wgpu::BindGroupEntry { binding: 5, resource: u.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(lights) },
            ],
        })
    }

    /// Draw the small picture of the lighting: the lights' map and the picture (None = no picture).
    pub fn show(&self, to: &wgpu::Texture, lights: &wgpu::TextureView, picture: Option<&wgpu::TextureView>, mut u: ShowU) {
        let gpu = &self.gpu;
        u.size = [to.width() as f32, to.height() as f32, 0.0, 0.0];
        gpu.queue.write_buffer(&self.u_show, 0, bytemuck::bytes_of(&u));
        let group = self.show_group(&self.u_show, lights, picture.unwrap_or(&self.black));
        let view = to.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("lighting shown") });
        {
            let mut rp = pass(&mut enc, &view, None, true);
            rp.set_pipeline(&self.p_show);
            rp.set_bind_group(0, &group, &[]);
            rp.draw(0..3, 0..1);
        }
        gpu.queue.submit([enc.finish()]);
    }
}

/// A picture being read on another thread.
pub struct Loading {
    rx: Receiver<R<Hdri>>,
}
impl Loading {
    /// Start reading `path` on a thread of its own. `maker`: where the pipelines that make the cube are
    /// kept; if there are none yet, that thread builds them too.
    pub fn start(maker: &Arc<OnceLock<Maker>>, gpu: &Gpu, path: PathBuf) -> Loading {
        let (tx, rx) = channel();
        let (maker, gpu) = (maker.clone(), gpu.clone());
        let spawned = std::thread::Builder::new().name("environment picture".into()).spawn(move || {
            let _ = tx.send(maker.get_or_init(|| Maker::new(&gpu)).load(&path));
        });
        if let Err(e) = spawned {
            let (tx, rx) = channel();
            let _ = tx.send(Err(format!("could not start reading the picture: {e}")));
            return Loading { rx };
        }
        Loading { rx }
    }
    /// The picture if it has arrived (or why it never will); None while it is still being read.
    pub fn poll(&self) -> Option<R<Hdri>> {
        match self.rx.try_recv() {
            Ok(r) => Some(r),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(Err("reading the picture stopped unexpectedly".into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Radiance file of `w` x `h`: every pixel from `f(x, y)` as four bytes; `packed` writes each row
    /// run-length coded a colour at a time, the way most programs do.
    fn radiance(w: usize, h: usize, packed: bool, f: impl Fn(usize, usize) -> [u8; 4]) -> Vec<u8> {
        let mut out = format!("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {h} +X {w}\n").into_bytes();
        for y in 0..h {
            let row: Vec<[u8; 4]> = (0..w).map(|x| f(x, y)).collect();
            if !packed {
                for p in &row { out.extend_from_slice(p); }
                continue;
            }
            out.extend_from_slice(&[2, 2, (w >> 8) as u8, (w & 255) as u8]);
            for c in 0..4 {
                let mut x = 0;
                while x < w {
                    // a run of equal bytes as a run, anything else as itself
                    let mut n = 1;
                    while x + n < w && n < 127 && row[x + n][c] == row[x][c] { n += 1; }
                    if n >= 3 {
                        out.extend_from_slice(&[128 + n as u8, row[x][c]]);
                    } else {
                        n = 1;
                        while x + n < w && n < 128 && !(x + n + 2 < w && row[x + n][c] == row[x + n + 1][c] && row[x + n][c] == row[x + n + 2][c]) { n += 1; }
                        out.push(n as u8);
                        for k in 0..n { out.push(row[x + k][c]); }
                    }
                    x += n;
                }
            }
        }
        out
    }

    #[test]
    fn radiance_files_are_read_packed_or_plain() {
        // 128 is a half, with exponent 129 that is one: (128 + 0.5) / 256 * 2
        let f = |x: usize, y: usize| if (x / 5 + y) % 3 == 0 { [128, 64, (x * 7 % 256) as u8, 129] } else { [10, 10, 10, 120 + (y % 9) as u8] };
        let (w, h) = (40, 12);
        let plain = read_hdr(&radiance(w, h, false, f)).unwrap();
        let packed = read_hdr(&radiance(w, h, true, f)).unwrap();
        assert_eq!((plain.w, plain.h, packed.w, packed.h), (w, h, w, h));
        assert_eq!(plain.rgb, packed.rgb);
        let at = |x: usize, y: usize, c: usize| plain.rgb[(y * w + x) * 3 + c];
        assert!((at(0, 0, 0) - 1.00390625).abs() < 1e-6 && (at(0, 0, 1) - 0.50390625).abs() < 1e-6, "{} {}", at(0, 0, 0), at(0, 0, 1));
        assert!((at(5, 0, 0) - 10.5 * 2f32.powi(120 - 136)).abs() < 1e-9);
        // an exponent of nothing is black; rows from the bottom up are turned over
        let black = read_hdr(&radiance(8, 4, false, |_, _| [200, 200, 200, 0])).unwrap();
        assert!(black.rgb.iter().all(|v| *v == 0.0));
        let mut up = radiance(8, 4, false, |_, y| [128, 128, 128, 129 + y as u8]);
        let text = String::from_utf8_lossy(&up).replace("-Y 4", "+Y 4").into_bytes();
        up[..text.len().min(40)].copy_from_slice(&text[..text.len().min(40)]);
        let up = read_hdr(&up).unwrap();
        assert!(up.rgb[0] > up.rgb[3 * 8 * 3], "the first row stored is the bottom one");
        // what is not such a file, or is cut short, is refused with words
        assert!(read_hdr(b"#?RADIANCE\nFORMAT=32-bit_rle_xyze\n\n-Y 4 +X 8\n").is_err());
        assert!(read_hdr(b"hello").is_err());
        let cut = radiance(w, h, true, f);
        assert!(read_hdr(&cut[..cut.len() - 20]).err().is_some_and(|e| e.contains("ends before")));
    }

    /// A panorama with a dim sky, a darker ground and a small, enormously bright sun.
    fn sunny(w: usize, h: usize) -> Panorama {
        let mut rgb = vec![0.0f32; w * h * 3];
        let sun = panorama_dir(0.7, 0.3);
        for j in 0..h {
            for i in 0..w {
                let d = panorama_dir((i as f64 + 0.5) / w as f64, (j as f64 + 0.5) / h as f64);
                let c = d[0] * sun[0] + d[1] * sun[1] + d[2] * sun[2];
                let v = if c > 0.9995 { [40000.0, 36000.0, 30000.0] } else if d[1] > 0.0 { [0.3, 0.5, 0.9] } else { [0.08, 0.1, 0.05] };
                rgb[(j * w + i) * 3..][..3].copy_from_slice(&v);
            }
        }
        Panorama { w, h, rgb }
    }

    #[test]
    fn a_picture_is_levelled_and_its_sun_tamed() {
        let (w, h) = (512, 256);
        let t = tame(sunny(w, h)).unwrap();
        assert_eq!((t.w, t.h, t.half.len()), (w, h, w * h * 4));
        // nothing above the ceiling, and the average over the sphere is the one aimed for
        let (mut top, mut mean) = (0.0f32, 0.0f64);
        for j in 0..h {
            let patch = (2.0 * std::f64::consts::PI / w as f64) * ((j as f64 / h as f64 * std::f64::consts::PI).cos() - ((j + 1) as f64 / h as f64 * std::f64::consts::PI).cos());
            for i in 0..w {
                let px: Vec<f32> = (0..4).map(|c| half::f16::from_bits(t.half[(j * w + i) * 4 + c]).to_f32()).collect();
                top = top.max(px[0]).max(px[1]).max(px[2]);
                assert_eq!(px[3], 1.0);
                mean += patch * (LUM[0] * px[0] as f64 + LUM[1] * px[1] as f64 + LUM[2] * px[2] as f64);
            }
        }
        mean /= 4.0 * std::f64::consts::PI;
        assert!(top as f64 <= CEILING * 1.001 && top as f64 > CEILING * 0.99, "brightest {top}");
        assert!((mean / MEAN - 1.0).abs() < 0.02, "average {mean}");
        assert!(t.peak == 40000.0 && t.scale > 0.0);
        // the sun's light: taken off, and some of it given back as one glow where the sun is, no
        // brighter in the middle than a glow may be
        assert!(t.taken_off > 0.5, "{}", t.taken_off);
        assert_eq!(t.glows.len(), 1);
        let (g, sun) = (&t.glows[0], panorama_dir(0.7, 0.3));
        assert!(g.dir[0] * sun[0] + g.dir[1] * sun[1] + g.dir[2] * sun[2] > 0.9999, "{:?} against {sun:?}", g.dir);
        let y = LUM[0] * g.energy[0] + LUM[1] * g.energy[1] + LUM[2] * g.energy[2];
        assert!(y / (2.0 * std::f64::consts::PI * g.sigma * g.sigma) <= GLOW_PEAK * 1.001);
        assert!(g.sigma >= GLOW_NARROW && g.sigma <= GLOW_WIDE);
        assert!(g.energy[0] > g.energy[2], "the glow has the sun's colour");
        assert!(t.given_back > 0.0 && t.given_back <= t.taken_off);
        // the nine terms: more light gathered by a surface facing up than by one facing down
        let terms = env::sh_terms(&t.sh, env::SH_MATTE);
        assert!(env::sh_eval(&terms, [0.0, 1.0, 0.0])[1] > 2.0 * env::sh_eval(&terms, [0.0, -1.0, 0.0])[1]);

        // a picture with no sun loses nothing and gets no glow; twice as bright a file gives the same picture
        let mut dull = sunny(w, h);
        for v in dull.rgb.iter_mut() { *v = v.min(1.0); }
        let twice = Panorama { w, h, rgb: dull.rgb.iter().map(|v| v * 2.0).collect() };
        let (a, b) = (tame(dull).unwrap(), tame(twice).unwrap());
        assert!(a.glows.is_empty() && a.taken_off == 0.0);
        assert!((a.scale / b.scale - 2.0).abs() < 1e-6);
        assert!(a.half.iter().zip(&b.half).all(|(x, y)| (*x as i32 - *y as i32).abs() <= 1));
        // black all over is refused; numbers that are not numbers are taken as no light
        assert!(tame(Panorama { w: 16, h: 8, rgb: vec![0.0; 16 * 8 * 3] }).is_err());
        let mut odd = sunny(64, 32);
        odd.rgb[5] = f32::NAN;
        odd.rgb[100] = f32::INFINITY;
        odd.rgb[200] = -3.0;
        let t = tame(odd).unwrap();
        assert!(t.half.iter().all(|b| half::f16::from_bits(*b).to_f32().is_finite()));
    }

    #[test]
    fn softening_grows_with_the_setting() {
        assert_eq!(blur_sigma(0.0), 0.0);
        let mut last = 0.0;
        for i in 1..=20 {
            let s = blur_sigma(i as f64 / 20.0);
            assert!(s > last);
            last = s;
        }
        assert!((blur_sigma(0.5).to_degrees() - 5.1).abs() < 0.2, "{}", blur_sigma(0.5).to_degrees());
        assert!((blur_sigma(1.0).to_degrees() - 60.0).abs() < 1e-6 && blur_sigma(3.0) == blur_sigma(1.0));
    }

    #[test]
    fn paths_in_the_programs_folder_are_kept_short() {
        let root = home();
        let inside = root.join("hdri").join("garden.exr");
        assert_eq!(tidy(&inside.to_string_lossy()), "hdri/garden.exr");
        assert_eq!(resolve("hdri/garden.exr"), root.join("hdri/garden.exr"));
        assert_eq!(tidy("hdri/garden.exr"), "hdri/garden.exr");
        // a file somewhere else keeps its full path
        let outside = std::env::temp_dir().join("somewhere").join("sky.hdr");
        if !outside.starts_with(&root) {
            assert_eq!(tidy(&outside.to_string_lossy()), outside.to_string_lossy());
            assert_eq!(resolve(&outside.to_string_lossy()), outside);
        }
    }
}
