//! 3D text: a word typed into the shape's panel, made solid.
//!
//! The letters' outlines come from a font installed with the system (C:\Windows\Fonts), read when the
//! shape is built, by the small TrueType reader below (the `glyf` outlines: tables cmap, head, maxp,
//! hhea, hmtx, loca, glyf and the old `kern`). No font is copied into or shipped with the program, so no
//! font's licence is touched: it is used as any program on the machine uses an installed font, as the
//! panels' own lettering is (app/src/ui/theme.rs). Where none of the fonts asked for is installed, the
//! first one of the list that is takes its place; with no font at all the word is drawn as plain blocks.
//!
//! Each letter is solid (`Term::Glyph`): its outline, flattened into straight edges, filled by the
//! even-odd rule (so the holes of o, e, a, B and 8 come out as holes), extruded to a depth, its edges
//! rounded. A letter is cut into bands across it, a piece each holding only the edges near it: the baker
//! looks at every edge of a piece for every voxel of its box, so this cuts the work about threefold.
//!
//! The word is `ShapeState::model` (`CurveOpts::text`): a `|` or a new line starts another line.

use super::geo::*;
use super::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub const FAMILY: &str = "text";
pub const DEFAULT: &str = "glass";

/// The fonts of the `font` slider: what the panel calls each, and its file.
const FONTS: [(&str, &str); 8] = [
    ("Segoe UI Bold", "segoeuib.ttf"),
    ("Segoe UI Black", "seguibl.ttf"),
    ("Georgia Bold", "georgiab.ttf"),
    ("Arial Black", "ariblk.ttf"),
    ("Gabriola", "Gabriola.ttf"),
    ("Bahnschrift", "bahnschrift.ttf"),
    ("Impact", "impact.ttf"),
    ("Consolas Bold", "consolab.ttf"),
];

/// The names of the `font` slider's settings.
pub fn choices(id: &str) -> Option<Vec<String>> {
    (id == "font").then(|| FONTS.iter().map(|f| f.0.to_string()).collect())
}

// ---- reading a TrueType font -------------------------------------------------------------------

/// A font file, read enough to give outlines and advances.
pub struct Font {
    data: Vec<u8>,
    upem: f64,
    long_loca: bool,
    glyphs: usize,
    metrics: usize,
    t_loca: usize,
    t_glyf: usize,
    t_hmtx: usize,
    /// character -> glyph
    cmap: HashMap<u32, u16>,
    /// (left glyph, right glyph) -> kerning, font units
    kern: HashMap<(u16, u16), i16>,
}

fn u16_at(d: &[u8], o: usize) -> Option<u16> { d.get(o..o + 2).map(|b| u16::from_be_bytes([b[0], b[1]])) }
fn i16_at(d: &[u8], o: usize) -> Option<i16> { u16_at(d, o).map(|v| v as i16) }
fn u32_at(d: &[u8], o: usize) -> Option<u32> { d.get(o..o + 4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]])) }

impl Font {
    pub fn parse(data: Vec<u8>) -> Option<Font> {
        let d = &data;
        // (a collection: its first font)
        let base = if d.get(0..4) == Some(b"ttcf") { u32_at(d, 12)? as usize } else { 0 };
        let n = u16_at(d, base + 4)? as usize;
        let mut tables = HashMap::new();
        for i in 0..n {
            let rec = base + 12 + 16 * i;
            tables.insert(d.get(rec..rec + 4)?.to_vec(), u32_at(d, rec + 8)? as usize);
        }
        let t = |tag: &[u8]| tables.get(tag).copied();
        let (head, maxp, hhea) = (t(b"head")?, t(b"maxp")?, t(b"hhea")?);
        let (t_loca, t_glyf, t_hmtx, cmap_t) = (t(b"loca")?, t(b"glyf")?, t(b"hmtx")?, t(b"cmap")?);
        let upem = u16_at(d, head + 18)? as f64;
        let long_loca = i16_at(d, head + 50)? != 0;
        let glyphs = u16_at(d, maxp + 4)? as usize;
        let metrics = u16_at(d, hhea + 34)? as usize;
        // the character map: a Unicode subtable, the fullest kind there is
        let mut best: Option<(u32, usize)> = None;
        for i in 0..u16_at(d, cmap_t + 2)? as usize {
            let rec = cmap_t + 4 + 8 * i;
            let (pid, eid) = (u16_at(d, rec)?, u16_at(d, rec + 2)?);
            let off = cmap_t + u32_at(d, rec + 4)? as usize;
            let fmt = u16_at(d, off)?;
            let rank = match (pid, eid, fmt) {
                (3, 10, 12) | (0, _, 12) => 3,
                (3, 1, 4) | (0, _, 4) => 2,
                _ => 0,
            };
            if rank > 0 && best.is_none_or(|b| rank > b.0) { best = Some((rank, off)); }
        }
        let (_, off) = best?;
        let mut cmap = HashMap::new();
        if u16_at(d, off)? == 4 {
            let segs = u16_at(d, off + 6)? as usize / 2;
            let (ends, starts, deltas, ranges) = (off + 14, off + 16 + 2 * segs, off + 16 + 4 * segs, off + 16 + 6 * segs);
            for s in 0..segs {
                let (end, start) = (u16_at(d, ends + 2 * s)? as u32, u16_at(d, starts + 2 * s)? as u32);
                let (delta, range) = (u16_at(d, deltas + 2 * s)?, u16_at(d, ranges + 2 * s)? as usize);
                if start > end || end == 0xffff && start == 0xffff { continue; }
                for c in start..=end.min(0x2fff) {
                    let g = if range == 0 {
                        (c as u16).wrapping_add(delta)
                    } else {
                        let at = ranges + 2 * s + range + 2 * (c - start) as usize;
                        match u16_at(d, at) { Some(0) | None => 0, Some(g) => g.wrapping_add(delta) }
                    };
                    if g != 0 { cmap.insert(c, g); }
                }
            }
        } else {
            for i in 0..u32_at(d, off + 12)? as usize {
                let g = off + 16 + 12 * i;
                let (a, b, first) = (u32_at(d, g)?, u32_at(d, g + 4)?, u32_at(d, g + 8)?);
                for c in a..=b.min(a + 0x3000) { cmap.insert(c, (first + c - a) as u16); }
            }
        }
        // kerning, from the old table if there is one (format 0 pairs)
        let mut kern = HashMap::new();
        if let Some(k) = t(b"kern") {
            if u16_at(d, k) == Some(0) && u16_at(d, k + 2).unwrap_or(0) > 0 && u16_at(d, k + 8).map(|c| c >> 8) == Some(0) {
                let pairs = u16_at(d, k + 10).unwrap_or(0) as usize;
                for i in 0..pairs {
                    let p = k + 18 + 6 * i;
                    if let (Some(l), Some(r), Some(v)) = (u16_at(d, p), u16_at(d, p + 2), i16_at(d, p + 4)) { kern.insert((l, r), v); }
                }
            }
        }
        Some(Font { upem, long_loca, glyphs, metrics, t_loca, t_glyf, t_hmtx, cmap, kern, data })
    }

    pub fn glyph(&self, c: char) -> u16 { self.cmap.get(&(c as u32)).copied().unwrap_or(0) }
    /// How far the pen moves after a glyph, in ems.
    pub fn advance(&self, g: u16) -> f64 {
        let i = (g as usize).min(self.metrics.saturating_sub(1));
        u16_at(&self.data, self.t_hmtx + 4 * i).unwrap_or(0) as f64 / self.upem
    }
    pub fn kerning(&self, a: u16, b: u16) -> f64 { self.kern.get(&(a, b)).copied().unwrap_or(0) as f64 / self.upem }

    /// A glyph's outlines in ems, as closed polygons (curves flattened finely).
    pub fn outline(&self, g: u16) -> Vec<Vec<[f64; 2]>> {
        let mut out = Vec::new();
        self.contours(g, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 0, &mut out);
        out.into_iter()
            .map(|c| c.into_iter().map(|p| [p[0] / self.upem, p[1] / self.upem]).collect())
            .collect()
    }

    /// The contours of glyph g, through the transform m (a b c d e f: x' = a x + c y + e), in font units.
    fn contours(&self, g: u16, m: [f64; 6], depth: u32, out: &mut Vec<Vec<[f64; 2]>>) -> Option<()> {
        let d = &self.data;
        if g as usize >= self.glyphs || depth > 6 { return None; }
        let (a, b) = if self.long_loca {
            (u32_at(d, self.t_loca + 4 * g as usize)? as usize, u32_at(d, self.t_loca + 4 * g as usize + 4)? as usize)
        } else {
            (2 * u16_at(d, self.t_loca + 2 * g as usize)? as usize, 2 * u16_at(d, self.t_loca + 2 * g as usize + 2)? as usize)
        };
        if b <= a { return Some(()); }
        let at = self.t_glyf + a;
        let nc = i16_at(d, at)?;
        let tf = |x: f64, y: f64| [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]];
        if nc >= 0 {
            let nc = nc as usize;
            let ends: Vec<usize> = (0..nc).map(|i| u16_at(d, at + 10 + 2 * i).map(|v| v as usize)).collect::<Option<_>>()?;
            let np = ends.last().map_or(0, |e| e + 1);
            let instr = u16_at(d, at + 10 + 2 * nc)? as usize;
            let mut o = at + 12 + 2 * nc + instr;
            let mut flags = Vec::with_capacity(np);
            while flags.len() < np {
                let f = *d.get(o)?;
                o += 1;
                flags.push(f);
                if f & 8 != 0 {
                    let r = *d.get(o)?;
                    o += 1;
                    for _ in 0..r { flags.push(f); }
                }
            }
            flags.truncate(np);
            let mut coord = |short: u8, same: u8| -> Option<Vec<f64>> {
                let mut v = 0i32;
                let mut out = Vec::with_capacity(np);
                for &f in &flags {
                    if f & short != 0 {
                        let dv = *d.get(o)? as i32;
                        o += 1;
                        v += if f & same != 0 { dv } else { -dv };
                    } else if f & same == 0 {
                        v += i16_at(d, o)? as i32;
                        o += 2;
                    }
                    out.push(v as f64);
                }
                Some(out)
            };
            let xs = coord(2, 16)?;
            let ys = coord(4, 32)?;
            let mut s = 0;
            for &e in &ends {
                if e < s || e >= np { return None; }
                let pts: Vec<([f64; 2], bool)> = (s..=e).map(|i| (tf(xs[i], ys[i]), flags[i] & 1 != 0)).collect();
                s = e + 1;
                let poly = flatten(&pts, self.upem);
                if poly.len() >= 3 { out.push(poly); }
            }
        } else {
            // a composite: other glyphs, each moved (and perhaps scaled)
            let mut o = at + 10;
            loop {
                let flags = u16_at(d, o)?;
                let gi = u16_at(d, o + 2)?;
                o += 4;
                let (dx, dy) = if flags & 1 != 0 {
                    let v = (i16_at(d, o)? as f64, i16_at(d, o + 2)? as f64);
                    o += 4;
                    v
                } else {
                    let v = (*d.get(o)? as i8 as f64, *d.get(o + 1)? as i8 as f64);
                    o += 2;
                    v
                };
                let f2 = |o: usize| i16_at(d, o).map(|v| v as f64 / 16384.0);
                let mut k = [1.0, 0.0, 0.0, 1.0];
                if flags & 8 != 0 {
                    k[0] = f2(o)?;
                    k[3] = k[0];
                    o += 2;
                } else if flags & 0x40 != 0 {
                    k[0] = f2(o)?;
                    k[3] = f2(o + 2)?;
                    o += 4;
                } else if flags & 0x80 != 0 {
                    k = [f2(o)?, f2(o + 2)?, f2(o + 4)?, f2(o + 6)?];
                    o += 8;
                }
                // (point-matched placement, flag 2 clear, is rare: taken as no offset)
                let (dx, dy) = if flags & 2 != 0 { (dx, dy) } else { (0.0, 0.0) };
                let sub = [
                    m[0] * k[0] + m[2] * k[1], m[1] * k[0] + m[3] * k[1],
                    m[0] * k[2] + m[2] * k[3], m[1] * k[2] + m[3] * k[3],
                    m[0] * dx + m[2] * dy + m[4], m[1] * dx + m[3] * dy + m[5],
                ];
                self.contours(gi, sub, depth + 1, out);
                if flags & 0x20 == 0 { break; }
            }
        }
        Some(())
    }
}

/// A closed TrueType contour (points on and off the curve) as a polygon: each quadratic piece cut into
/// steps fine enough that the polygon strays from the curve by under a 2000th of an em.
fn flatten(pts: &[([f64; 2], bool)], upem: f64) -> Vec<[f64; 2]> {
    let n = pts.len();
    if n < 2 { return Vec::new(); }
    let mid = |a: [f64; 2], b: [f64; 2]| [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])];
    // start on a point on the curve (or the midpoint of two off it)
    let first = pts.iter().position(|p| p.1);
    let (start, rot) = match first {
        Some(i) => (pts[i].0, i),
        None => (mid(pts[0].0, pts[1].0), 0),
    };
    let mut out = vec![start];
    let mut cur = start;
    let mut ctrl: Option<[f64; 2]> = None;
    let tol = upem / 2000.0;
    let quad = |out: &mut Vec<[f64; 2]>, a: [f64; 2], c: [f64; 2], b: [f64; 2]| {
        // the most a quadratic strays from its chord is a quarter of the control point's distance from it
        let dev = 0.25 * ((a[0] - 2.0 * c[0] + b[0]).powi(2) + (a[1] - 2.0 * c[1] + b[1]).powi(2)).sqrt();
        let k = ((dev / tol).sqrt().ceil() as usize).clamp(1, 24);
        for j in 1..=k {
            let t = j as f64 / k as f64;
            let u = 1.0 - t;
            out.push([u * u * a[0] + 2.0 * u * t * c[0] + t * t * b[0], u * u * a[1] + 2.0 * u * t * c[1] + t * t * b[1]]);
        }
    };
    let count = if first.is_some() { n } else { n + 1 };
    for j in 1..=count {
        let (p, on) = pts[(rot + j) % n];
        match (on, ctrl) {
            (true, None) => {
                out.push(p);
                cur = p;
            }
            (true, Some(c)) => {
                quad(&mut out, cur, c, p);
                cur = p;
                ctrl = None;
            }
            (false, None) => ctrl = Some(p),
            (false, Some(c)) => {
                let m = mid(c, p);
                quad(&mut out, cur, c, m);
                cur = m;
                ctrl = Some(p);
            }
        }
    }
    if let Some(c) = ctrl { quad(&mut out, cur, c, start); }
    // the last point repeats the first: drop it, and any point on top of the one before
    out.dedup_by(|a, b| (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9);
    if out.len() > 1 && (out[0][0] - out[out.len() - 1][0]).abs() < 1e-9 && (out[0][1] - out[out.len() - 1][1]).abs() < 1e-9 { out.pop(); }
    out
}

/// The fonts read so far, by file (None: not there, or not a font this reader can read).
static FONTS_READ: Mutex<Vec<(String, Option<Arc<Font>>)>> = Mutex::new(Vec::new());

fn font_dir() -> std::path::PathBuf {
    std::env::var_os("WINDIR").map(std::path::PathBuf::from).unwrap_or_else(|| r"C:\Windows".into()).join("Fonts")
}

fn read_font(file: &str) -> Option<Arc<Font>> {
    if let Some(f) = FONTS_READ.lock().ok().and_then(|c| c.iter().find(|(k, _)| k == file).map(|(_, f)| f.clone())) { return f; }
    let f = std::fs::read(font_dir().join(file)).ok().and_then(Font::parse).map(Arc::new);
    if let Ok(mut c) = FONTS_READ.lock() { c.push((file.to_string(), f.clone())); }
    f
}

/// The font of the slider's setting, or the first of the others that is installed.
pub fn font(choice: usize) -> Option<Arc<Font>> {
    let i = choice.min(FONTS.len() - 1);
    read_font(FONTS[i].1).or_else(|| FONTS.iter().find_map(|f| read_font(f.1)))
}

// ---- the word as solid pieces ------------------------------------------------------------------

/// How tall a band of a letter is, and how far beyond it its edges are kept (ems).
const BAND_HEIGHT: f64 = 0.12;
const BAND_MARGIN: f64 = 0.1;

const P_FONT: usize = 0;
const P_DEPTH: usize = 1;
const P_ROUND: usize = 2;
const P_SPACING: usize = 3;
const P_SLANT: usize = 4;
const P_ARC: usize = 5;
const P_LEADING: usize = 6;

fn params() -> Vec<ParamDef> {
    vec![
        ParamDef::labelled("font", "font", 0.0, 0.0, (FONTS.len() - 1) as f64, 1.0),
        ParamDef::labelled("depth", "depth", 0.22, 0.02, 1.0, 0.005),
        ParamDef::labelled("round", "round edges", 0.025, 0.0, 0.08, 0.001),
        ParamDef::labelled("spacing", "letter spacing", 0.02, -0.15, 0.5, 0.005),
        ParamDef::labelled("slant", "slant", 0.0, -0.4, 0.4, 0.01),
        ParamDef::labelled("arc", "bend round", 0.0, -1.0, 1.0, 0.01),
        ParamDef::labelled("leading", "line spacing", 1.15, 0.8, 2.0, 0.01),
    ]
}

/// The lines of a text: `|` or a new line between them; blank: the default word.
pub fn lines_of(text: &str) -> Vec<String> {
    let t = text.trim();
    let t = if t.is_empty() { DEFAULT } else { t };
    t.split(['|', '\n']).map(|l| l.trim_end_matches('\r').to_string()).collect()
}

/// The word, solid: a few pieces a letter.
pub fn word(v: &[f64], text: &str) -> RawCurve {
    let get = |i: usize, d: f64| v.get(i).copied().filter(|x| x.is_finite()).unwrap_or(d);
    let depth = get(P_DEPTH, 0.22).max(0.005) * 0.5;
    let round = get(P_ROUND, 0.025).clamp(0.0, 0.5 * depth);
    let (spacing, slant, arc, leading) = (get(P_SPACING, 0.02), get(P_SLANT, 0.0), get(P_ARC, 0.0), get(P_LEADING, 1.15));
    let lines = lines_of(text);
    let mut raw = RawCurve::default();
    let font = font(get(P_FONT, 0.0).round().max(0.0) as usize);

    // every letter laid out: (pen x, baseline y, the outline in ems about the pen)
    let mut laid: Vec<(f64, f64, Vec<Vec<[f64; 2]>>)> = Vec::new();
    let count = lines.len();
    let mut widest: f64 = 0.0;
    for (li, line) in lines.iter().enumerate() {
        let y = ((count - 1) as f64 * 0.5 - li as f64) * leading - 0.35;
        let mut x = 0.0;
        let mut row = Vec::new();
        let mut last: Option<u16> = None;
        for ch in line.chars() {
            match &font {
                Some(f) => {
                    let g = f.glyph(ch);
                    if let Some(l) = last { x += f.kerning(l, g); }
                    let shape = f.outline(g);
                    if !shape.is_empty() { row.push((x, y, shape)); }
                    x += f.advance(g) + spacing;
                    last = Some(g);
                }
                None => {
                    // no font at all: a block for every letter
                    if !ch.is_whitespace() { row.push((x, y, vec![vec![[0.05, 0.0], [0.5, 0.0], [0.5, 0.7], [0.05, 0.7]]])); }
                    x += 0.6 + spacing;
                }
            }
        }
        // each line centred
        let w = (x - spacing).max(0.0);
        widest = widest.max(w);
        for (px, py, s) in row { laid.push((px - 0.5 * w, py, s)); }
    }
    // bent round: the line of text runs round a circle about the upright axis, a whole turn at 1
    let bend = arc * 0.92 * std::f64::consts::PI;
    let radius = if bend.abs() > 1e-3 { widest.max(0.5) / bend.abs() } else { 0.0 };
    for (px, py, shape) in laid {
        // the outline about its own middle, slanted
        let (mut lo, mut hi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for c in &shape {
            for p in c {
                lo = [lo[0].min(p[0]), lo[1].min(p[1])];
                hi = [hi[0].max(p[0]), hi[1].max(p[1])];
            }
        }
        if !(lo[0] < hi[0]) { continue; }
        let mx = 0.5 * (lo[0] + hi[0]);
        let mut edges = Vec::new();
        let (mut elo, mut ehi) = ([f64::MAX; 2], [f64::MIN; 2]);
        for c in &shape {
            let q: Vec<[f64; 2]> = c.iter().map(|p| [p[0] - mx + slant * (p[1] + py + 0.35), p[1]]).collect();
            for (i, a) in q.iter().enumerate() {
                let b = q[(i + 1) % q.len()];
                edges.push([a[0], a[1], b[0], b[1]]);
                elo = [elo[0].min(a[0]), elo[1].min(a[1])];
                ehi = [ehi[0].max(a[0]), ehi[1].max(a[1])];
            }
        }
        // where the letter's middle goes, and which way it faces
        let cx = px + mx;
        let uy = [0.0, 1.0, 0.0];
        let (o, ux) = if radius > 0.0 {
            // round a circle about the upright axis: bent one way the text is on its outside, the other
            // way on its inside; the middle letter stays where it was
            let th = cx / radius;
            let (sn, cs) = th.sin_cos();
            if bend > 0.0 { ([radius * sn, py, radius * (cs - 1.0)], [cs, 0.0, -sn]) } else { ([radius * sn, py, radius * (1.0 - cs)], [cs, 0.0, sn]) }
        } else {
            ([cx, py, 0.0], [1.0, 0.0, 0.0])
        };
        let uz = cross(ux, uy);
        // The letter in bands across it, a piece each: every voxel of a piece looks at every one of its
        // edges, and a band has only those near it (`Term::Glyph::band`)
        let margin = BAND_MARGIN.max(2.0 * round);
        let bands = (((ehi[1] - elo[1]) / BAND_HEIGHT).round() as usize).max(1);
        for b in 0..bands {
            let y0 = elo[1] + (ehi[1] - elo[1]) * b as f64 / bands as f64;
            let y1 = elo[1] + (ehi[1] - elo[1]) * (b + 1) as f64 / bands as f64;
            let mine: Vec<[f64; 4]> = edges.iter().filter(|e| e[1].max(e[3]) >= y0 - margin && e[1].min(e[3]) <= y1 + margin).copied().collect();
            // (the first and last bands own everything below and above the letter)
            let band = if bands == 1 { [0.0; 3] } else { [if b == 0 { -1e4 } else { y0 }, if b + 1 == bands { 1e4 } else { y1 }, margin] };
            // the box: the band's part of the outline's, through its frame, and the depth
            let (mut blo, mut bhi) = ([f64::MAX; 3], [f64::MIN; 3]);
            for k in 0..8 {
                let x = if k & 1 == 0 { elo[0] } else { ehi[0] };
                let y = if k & 2 == 0 { y0 } else { y1 };
                let z = if k & 4 == 0 { -depth } else { depth };
                let p = add(o, add(mul(ux, x), add(mul(uy, y), mul(uz, z))));
                for a in 0..3 {
                    blo[a] = blo[a].min(p[a]);
                    bhi[a] = bhi[a].max(p[a]);
                }
            }
            if mine.is_empty() { continue; }
            raw.pieces.push(Piece { lo: blo, hi: bhi, terms: vec![Term::Glyph { o, ux, uy, depth, round, band, edges: mine }] });
        }
    }
    // placed by the boxes of its letters' outlines (not by the empty space the font leaves round them)
    raw.frame_pieces();
    raw
}

pub fn family() -> Family {
    let preset = Preset {
        eq: vec!["a font's outline, filled even-odd", "extruded: |z| ≤ depth/2, edges rounded"],
        note: "Type a word in the box below. A | starts another line. The letters are an installed font's own outlines, made solid.",
        params: params(),
        fixed_length: true,
        tube: Some(0.03),
        rot: Some([-8.0, 18.0, 0.0]),
        animate: Some(vec!["arc"]),
        info: Some(Box::new(|raw, _v| format!("{} pieces, {} edges", raw.pieces.len(), raw.pieces.iter().map(|p| p.terms.iter().map(|t| if let Term::Glyph { edges, .. } = t { edges.len() } else { 0 }).sum::<usize>()).sum::<usize>()))),
        ..Preset::new("word", "Word", Shape::Curve(Box::new(|v, o| word(v, o.text))))
    };
    Family {
        presets: vec![preset],
        // letters meet with sharp edges where they touch: a hard union
        melt: 0.0,
        lights: solid_lights(),
        defaults: vec![
            ("ior", Setting::Num(1.5)), ("absorb", Setting::Vec3([0.16, 0.08, 0.035])),
            ("dist", Setting::Num(6.6)), ("fov", Setting::Num(26.0)), ("rot", Setting::Vec3([-8.0, 18.0, 0.0])),
        ],
        ..Family::new(FAMILY, "3D text", Kind::Curve)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn some_font() -> Option<Arc<Font>> { font(0) }

    #[test]
    fn letters_with_holes_have_two_or_more_outlines() {
        let Some(f) = some_font() else { return };
        for (c, at_least) in [('o', 2), ('B', 3), ('8', 3), ('l', 1), ('e', 2), ('a', 2)] {
            let n = f.outline(f.glyph(c)).len();
            assert!(n >= at_least, "{c}: {n} outlines");
        }
        // every outline within an em or so
        for c in "glass Wg8".chars() {
            for poly in f.outline(f.glyph(c)) {
                for p in poly { assert!(p[0].abs() < 2.0 && p[1].abs() < 2.0, "{c} {p:?}"); }
            }
        }
    }

    #[test]
    fn a_word_is_pieces_of_letters() {
        let v: Vec<f64> = params().iter().map(|p| p.value).collect();
        let raw = word(&v, "a b|cd");
        if some_font().is_some() { assert!(raw.pieces.len() >= 4); }
        assert!(!raw.frame.is_empty());
        let blank = word(&v, "  ");
        assert!(blank.pieces.len() >= DEFAULT.len());
        // bent round, the letters still each make a piece with a box holding it
        let mut b = v.clone();
        b[P_ARC] = 0.7;
        let bent = word(&b, "glassforms");
        for p in &bent.pieces { for a in 0..3 { assert!(p.lo[a] <= p.hi[a]); } }
    }

    #[test]
    fn every_font_of_the_list_reads_or_falls_back() {
        for i in 0..FONTS.len() {
            if let Some(f) = font(i) {
                assert!(f.advance(f.glyph('m')) > 0.2, "{}", FONTS[i].0);
            }
        }
        assert_eq!(choices("font").unwrap().len(), FONTS.len());
    }
}

#[cfg(test)]
mod timing {
    /// How long a word takes to lay out (cargo test --release -p gf-core text3d_time -- --ignored --nocapture).
    #[test]
    #[ignore]
    fn text3d_time() {
        let v: Vec<f64> = super::params().iter().map(|p| p.value).collect();
        for (i, w) in ["glass", "Glass forms", "Mathematics in glass|and light"].iter().enumerate() {
            let t = std::time::Instant::now();
            let first = super::word(&v, w);
            let a = t.elapsed();
            let t = std::time::Instant::now();
            for _ in 0..20 { super::word(&v, w); }
            println!("{w:32} first {:6.2} ms (reads the font)  then {:6.3} ms  pieces {}", a.as_secs_f64() * 1e3, t.elapsed().as_secs_f64() * 1e3 / 20.0, first.pieces.len());
            let _ = i;
        }
    }
}
