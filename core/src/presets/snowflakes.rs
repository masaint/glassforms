//! Snowflakes, grown rather than drawn: Reiter's hexagonal vapour model (C. Reiter, "A local cellular
//! model for snow crystal growth", 2005).
//!
//! Every cell of a hexagonal grid holds an amount of water s. A cell is frozen once s >= 1; it and the
//! cells next to it are *receptive*. Each step:
//!   - receptive cells keep their water and gain gamma (vapour added from outside the plane);
//!   - the water of the other cells (u; 0 in receptive ones) diffuses: u <- u + alpha/12 (sum of the six
//!     neighbours' u - 6 u), receptive cells taking in what reaches them;
//!   - the grid's edge is held at the background vapour beta.
//! Low beta and gamma give fern-like stellar dendrites; more of either fills the arms out into sectored
//! plates and then plain hexagonal plates.
//!
//! The sliders `temperature` and `humidity` choose beta and gamma across the Nakaya diagram (see
//! `nakaya`): plates near -2 and below -22 C; thick hexagonal prisms (the column regime, which this flat
//! model can only hint at) around -6; sectored and stellar plates near -12 and -18; fern-like dendrites
//! near -15 when the air is wet. Dry air gives simple plates whatever the temperature.
//!
//! The flake is six-fold symmetric (and mirror symmetric within each arm), so growth is worked out on a
//! twelfth of the grid only. Each cell's freezing step is kept, so `grow` only cuts the grown flake.
//!
//! In glass every frozen cell is a hexagonal column (`Term::Hex`) about the plate's normal. Its height is
//! the plate's relief: Reiter's frozen cells go on gaining water, so the older a cell the thicker it is.
//! The centre is thickest and ribs run along the arms, as on real stellar crystals.

use super::*;
use geo::*;

pub const FAMILY: &str = "snowflakes";
const S3: f64 = 0.866_025_403_784_438_6;

// the sliders' places
const TEMP: usize = 0;
const HUMID: usize = 1;
const SEED: usize = 2;
const GROW: usize = 3;
const RELIEF: usize = 4;
const THICK: usize = 5;
const DETAIL: usize = 6;
/// passes of smoothing over the relief
const SMOOTH: usize = 6;

fn params(t: f64, h: f64, seed: f64) -> Vec<ParamDef> {
    vec![
        ParamDef::labelled("temp", "temperature °C", t, -25.0, -1.0, 0.1),
        ParamDef::labelled("humid", "humidity", h, 0.0, 1.0, 0.01),
        ParamDef::labelled("seed", "seed", seed, 1.0, 999.0, 1.0),
        ParamDef::labelled("grow", "grown", 1.0, 0.0, 1.0, 0.002),
        ParamDef::labelled("relief", "relief", 1.4, 0.0, 3.0, 0.01),
        ParamDef::labelled("plate", "plate thickness", 1.0, 0.3, 3.0, 0.01),
        ParamDef::labelled("detail", "detail", 120.0, 40.0, 160.0, 1.0),
    ]
}

/// Reiter's (alpha, beta, gamma) and how thick a prism the crystal is (0 a plate, 1 a stubby column),
/// for a temperature (C) and humidity (0..1).
pub fn nakaya(t: f64, h: f64) -> (f64, f64, f64, f64) {
    let bell = |c: f64, w: f64| (-((t - c) / w).powi(2)).exp();
    // how far the temperature favours branching: highest at -15, sectored shoulders at -12 and -18
    let branchy = (bell(-15.0, 2.2) + 0.55 * bell(-12.0, 1.6) + 0.55 * bell(-18.0, 1.8)).min(1.0);
    // the column regime between -4 and -10
    let column = bell(-6.5, 1.8);
    let b = (h.clamp(0.0, 1.0) * (0.1 + 0.9 * branchy) * (1.0 - 0.8 * column)).clamp(0.0, 1.0);
    let beta = 0.95 - 0.63 * b;
    // gamma on a log scale, from 0.05 (plates) to 0.0003 (ferns)
    let gamma = (0.05f64.ln() + (0.0003f64.ln() - 0.05f64.ln()) * b).exp();
    (1.0, beta, gamma, column)
}

/// A grown flake: cells (axial q, r) of the whole grid with the step each froze at.
pub struct Flake {
    pub cells: Vec<(i32, i32, u32)>,
    pub steps: u32,
    pub radius: f64,
    pub column: f64,
}

/// The cell standing for (q, r) in the twelfth of the grid that is worked out (lowest of its images).
fn canon(q: i32, r: i32) -> (i32, i32) {
    // cube coordinates (x, y, z) = (q, -q-r, r); rotations by 60 deg: (x, y, z) -> (-z, -x, -y)
    let (mut x, mut y, mut z) = (q, -q - r, r);
    let mut best = (i32::MAX, i32::MAX);
    for _ in 0..6 {
        (x, y, z) = (-z, -x, -y);
        for (a, c) in [(x, z), (x, y)] {
            // (x, y, z) and its mirror (x, z, y)
            let k = (a, c);
            if k < best { best = k; }
        }
    }
    best
}

pub fn grow_flake(alpha: f64, beta: f64, gamma: f64, radius: i32, column: f64) -> Flake {
    let rr = radius;
    let n = (2 * rr + 1) as usize;
    let at = |q: i32, r: i32| ((q + rr) as usize) * n + (r + rr) as usize;
    let inside = |q: i32, r: i32| q.abs().max(r.abs()).max((q + r).abs()) < rr;
    // the wedge: one index per class of cells alike under the twelve symmetries
    let mut slot = vec![u32::MAX; n * n];
    let mut reps: Vec<(i32, i32)> = Vec::new();
    let mut count: Vec<u32> = Vec::new();
    for q in -rr..=rr {
        for r in -rr..=rr {
            if !inside(q, r) { continue; }
            let (cq, cr) = canon(q, r);
            let k = at(cq, cr);
            if slot[k] == u32::MAX {
                slot[k] = reps.len() as u32;
                reps.push((cq, cr));
                count.push(0);
            }
            count[slot[k] as usize] += 1;
        }
    }
    let m = reps.len();
    const EDGE: u32 = u32::MAX;
    let dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)];
    let nb: Vec<[u32; 6]> = reps
        .iter()
        .map(|&(q, r)| {
            dirs.map(|(dq, dr)| {
                let (a, b) = (q + dq, r + dr);
                if inside(a, b) { let (cq, cr) = canon(a, b); slot[at(cq, cr)] } else { EDGE }
            })
        })
        .collect();
    let dist = |&(q, r): &(i32, i32)| q.abs().max(r.abs()).max((q + r).abs());
    let far: Vec<i32> = reps.iter().map(dist).collect();
    let mut s = vec![beta; m];
    let centre = slot[at(0, 0)] as usize;
    s[centre] = 1.0;
    let mut when = vec![u32::MAX; m];
    when[centre] = 0;
    let mut u = vec![0.0; m];
    let mut rec = vec![false; m];
    let max_steps = 12000u32;
    let mut step = 0;
    let mut reach = 0;
    while step < max_steps && reach < rr - 2 {
        step += 1;
        for i in 0..m {
            rec[i] = s[i] >= 1.0 || nb[i].iter().any(|&j| j != EDGE && s[j as usize] >= 1.0);
        }
        for i in 0..m {
            u[i] = if rec[i] { 0.0 } else { s[i] };
        }
        let mut next = s.clone();
        for i in 0..m {
            let mut sum = 0.0;
            for &j in &nb[i] {
                sum += if j == EDGE { beta } else { u[j as usize] };
            }
            let du = u[i] + alpha / 12.0 * (sum - 6.0 * u[i]);
            // (receptive cells keep their water, gain gamma, and take in what diffuses to them)
            next[i] = if rec[i] { s[i] + gamma + du } else { du };
            if next[i] >= 1.0 && when[i] == u32::MAX {
                when[i] = step;
                reach = reach.max(far[i]);
            }
        }
        s = next;
    }
    // every frozen cell of the whole grid
    let mut cells = Vec::new();
    for q in -rr..=rr {
        for r in -rr..=rr {
            if !inside(q, r) { continue; }
            let (cq, cr) = canon(q, r);
            let w = when[slot[at(cq, cr)] as usize];
            if w != u32::MAX { cells.push((q, r, w)); }
        }
    }
    Flake { cells, steps: step.max(1), radius: rr as f64, column }
}

static GROWN: Memo<Flake> = Memo::new();

fn flake(v: &[f64], o: &CurveOpts) -> RawCurve {
    let g = |i: usize, d: f64| v.get(i).copied().unwrap_or(d);
    let (t, h, seed) = (g(TEMP, -15.0), g(HUMID, 0.9), g(SEED, 1.0));
    let detail = g(DETAIL, 120.0).round().clamp(20.0, 200.0) as i32;
    let fl = GROWN.get(v, &[GROW, RELIEF, THICK], "", || {
        let (alpha, beta, gamma, column) = nakaya(t, h);
        // the seed: a little of every parameter, so that no two seeds branch quite alike
        let mut rng = Rng::new(seed * 7919.0 + 1.0);
        let (a, b, c) = (rng.sym(), rng.sym(), rng.sym());
        let alpha = alpha * (1.0 + 0.25 * a);
        let beta = (beta * (1.0 + 0.04 * b)).clamp(0.2, 0.97);
        let gamma = gamma * (1.0 + 0.35 * c);
        grow_flake(alpha, beta, gamma, detail, column)
    });
    draw(&fl, g(GROW, 1.0), g(RELIEF, 1.0), g(THICK, 1.0), o.quick)
}

/// The flake as `grow` shows it: one hexagonal column per frozen cell, as tall as the cell is old.
pub fn draw(fl: &Flake, grow: f64, relief: f64, plate: f64, quick: bool) -> RawCurve {
    let a1 = 1.0 / fl.radius;
    let grow = grow.clamp(0.0, 1.0);
    let last = fl.cells.iter().map(|c| c.2).max().unwrap_or(0).max(1) as f64;
    let now = (grow * last).max(0.0);
    // half-thickness: a plate's own and the relief that age adds; in the column regime a stubby prism
    let base = 0.014 * plate * (1.0 + 6.0 * fl.column);
    let rise = 0.045 * relief;
    let big = fl.cells.iter().map(|&(q, r, _)| { let x = (q as f64 + 0.5 * r as f64) * a1; let y = r as f64 * S3 * a1; (x * x + y * y).sqrt() }).fold(0.0f64, f64::max) + a1;
    // (the volume's voxel, as placed: the flake is placed to reach FIT; at the moving resolution, 256,
    // or the still one)
    let voxel = 2.0 * crate::HB / if quick { 256.0 } else { 768.0 } * big / crate::FIT;
    // Cells much smaller than a voxel cost a great deal to bake (every piece is drawn out to a band some
    // voxels wide, so they overlap many times over) and add nothing that shows: they are taken k by k
    // on a coarser grid, a coarse cell being ice from when a third of its cells are.
    let k = ((2.5 * voxel / a1).ceil() as i32).max(1);
    let coarse: Vec<(i32, i32, u32)>;
    let cells: &[(i32, i32, u32)] = if k == 1 {
        &fl.cells
    } else {
        let mut by: std::collections::HashMap<(i32, i32), Vec<u32>> = std::collections::HashMap::new();
        for &(q, r, w) in &fl.cells {
            by.entry(((q as f64 / k as f64).round() as i32, (r as f64 / k as f64).round() as i32)).or_default().push(w);
        }
        let need = ((k * k) as usize).div_ceil(3);
        coarse = by
            .into_iter()
            .filter_map(|(c, mut w)| {
                if w.len() < need { return None; }
                w.sort_unstable();
                Some((c.0, c.1, w[need - 1]))
            })
            .collect();
        &coarse
    };
    let a = a1 * k as f64;
    // the grid, with a border of a cell all round: NO = no ice
    const NO: f64 = -1.0;
    let rr = (fl.radius / k as f64).ceil() as i32 + 2;
    let n = (2 * rr + 1) as usize;
    let at = |q: i32, r: i32| (q + rr) as usize * n + (r + rr) as usize;
    let steps: [isize; 6] = [n as isize, -(n as isize), 1, -1, n as isize - 1, 1 - n as isize];
    let mut age = vec![NO; n * n];
    for &(q, r, w) in cells {
        if (w as f64) <= now { age[at(q, r)] = (now - w as f64) / last; }
    }
    let inner = |i: usize| i > n && i < n * n - n - 1 && i % n != 0 && i % n != n - 1;
    let nb = |i: usize, k: usize| (i as isize + steps[k]) as usize;
    // (pinholes: a cell left unfrozen with frozen cells on four or more sides is filled)
    let mut holes = Vec::new();
    for i in 0..n * n {
        if age[i] != NO || !inner(i) { continue; }
        let (mut c, mut sum) = (0, 0.0);
        for k in 0..6 {
            let v = age[nb(i, k)];
            if v != NO { c += 1; sum += v; }
        }
        if c >= 4 { holes.push((i, sum / c as f64)); }
    }
    for (i, v) in holes { age[i] = v; }
    let ice: Vec<usize> = (0..n * n).filter(|&i| age[i] != NO).collect();
    // Smoothed over the frozen cells (each with its frozen neighbours) some cells wide: the step at which
    // a cell froze is ragged from cell to cell along a growing front, and the steps between columns must
    // stay well under a voxel. The edge of the flake falls off a little, so its rim is rounded.
    let mut hgt = age.clone();
    let mut prev = hgt.clone();
    for pass in 0..(SMOOTH / k as usize).max(2) {
        prev.copy_from_slice(&hgt);
        for &i in &ice {
            let mut sum = prev[i];
            for k in 0..6 {
                let v = prev[nb(i, k)];
                // (outside the flake: nothing, for the first passes only, which rounds the rim)
                if v != NO { sum += v; } else if pass >= 2 { sum += prev[i]; }
            }
            hgt[i] = sum / 7.0;
        }
    }
    // the half-thickness of every cell: the plate's own (thinner towards the rim) and its relief
    let ripe = ice.iter().map(|&i| hgt[i]).fold(0.0f64, f64::max);
    let pos = |q: i32, r: i32| [(q as f64 + 0.5 * r as f64) * a, r as f64 * S3 * a];
    // (no thinner than the volume holds: about a voxel either side)
    let floor = 1.2 * voxel;
    let mut half = vec![NO; n * n];
    for &i in &ice {
        let v = hgt[i];
        half[i] = (base * (0.7 + 0.3 * (3.0 * v / (1e-9 + ripe)).min(1.0)) + rise * v.powf(0.75)).max(floor);
    }
    let mut raw = RawCurve::default();
    raw.pieces.reserve(ice.len());
    let ri = 0.5 * a * 1.02;
    // where each of `steps` goes: +q, -q, +r, -r, (+q, -r), (-q, +r)
    let offs: [[f64; 2]; 6] = [[a, 0.0], [-a, 0.0], [0.5 * a, S3 * a], [-0.5 * a, -S3 * a], [0.5 * a, -S3 * a], [-0.5 * a, S3 * a]];
    for &i in &ice {
        let (q, r) = ((i / n) as i32 - rr, (i % n) as i32 - rr);
        let [x, y] = pos(q, r);
        let hh = half[i];
        // The faces slope with the relief: the slope from the neighbours (least squares), so that the
        // columns' tops meet as one smooth surface and not as terraces.
        let (mut m, mut b, mut around) = ([0.0f64; 3], [0.0f64; 2], 0);
        for k in 0..6 {
            let o = half[nb(i, k)];
            if o == NO { continue; }
            around += 1;
            let d = offs[k];
            m[0] += d[0] * d[0];
            m[1] += d[0] * d[1];
            m[2] += d[1] * d[1];
            b[0] += d[0] * (o - hh);
            b[1] += d[1] * (o - hh);
        }
        // A column inside the flake reaches well into its neighbours. Joined by their least (a hard
        // union) the columns' distances are right outside the glass but not inside it: there every seam
        // between two columns would read as a surface, and the glass would show a honeycomb of grooves.
        // Overlapping, each seam lies deep inside a neighbour. (Columns at the rim keep to their cells,
        // so that the outline swells by no more than a fifth of a cell.)
        let rad = match around { 6 => 0.5 * a * 1.9, 3..=5 => 0.5 * a * 1.4, _ => ri };
        let e = rad / S3;
        let det = m[0] * m[2] - m[1] * m[1];
        let g = if det > 1e-6 * a.powi(4) { [(m[2] * b[0] - m[1] * b[1]) / det, (m[0] * b[1] - m[1] * b[0]) / det] } else { [0.0; 2] };
        let tall = hh + (g[0].abs() + g[1].abs()) * e;
        let l = (g[0] * g[0] + g[1] * g[1] + 1.0).sqrt();
        let d = (hh - g[0] * x - g[1] * y) / l;
        raw.pieces.push(Piece {
            lo: [x - e, y - e, -tall],
            hi: [x + e, y + e, tall],
            terms: vec![
                Term::Hex { c: [x, y, 0.0], axis: [0.0, 0.0, 1.0], u: [1.0, 0.0, 0.0], r: rad, h: tall + 0.01 },
                Term::Plane { n: [-g[0] / l, -g[1] / l, 1.0 / l], d },
                Term::Plane { n: [-g[0] / l, -g[1] / l, -1.0 / l], d },
            ],
        });
    }
    // placed by the whole grown flake, so that it grows in place
    for k in 0..6 {
        let th = TAU * k as f64 / 6.0;
        raw.frame.extend_from_slice(&[big * th.cos(), big * th.sin(), 0.0]);
    }
    raw.frame.extend_from_slice(&[0.0, 0.0, base + rise]);
    raw.frame.extend_from_slice(&[0.0, 0.0, -(base + rise)]);
    raw
}

fn preset(id: &'static str, name: &'static str, t: f64, h: f64, seed: f64, note: &'static str) -> Preset {
    Preset {
        eq: vec!["frozen: s ≥ 1;  receptive: frozen or beside it", "receptive s += γ;  others: u += α/12 (Σ₆ u − 6u)"],
        note,
        params: params(t, h, seed),
        tube: Some(0.03),
        fixed_length: true,
        rot: Some([28.0, 0.0, 0.0]),
        animate: Some(vec!["grow"]),
        info: Some(Box::new(|raw, v| {
            let (_, b, g, _) = nakaya(v[TEMP], v[HUMID]);
            format!("{} cells · β {:.2} γ {:.4}", raw.pieces.len(), b, g)
        })),
        ..Preset::new(id, name, Shape::Curve(Box::new(flake)))
    }
}

pub fn family() -> Family {
    Family {
        presets: vec![
            preset("dendrite", "Stellar dendrite", -15.0, 1.0, 1.0, "Grown from vapour on a hexagonal grid. Near −15 °C in wet air the arms branch into ferns."),
            preset("sectored", "Sectored plate", -18.0, 0.75, 7.0, "Near −18 °C, less wet: a plate ribbed like the sectors of a fan."),
            preset("stellar", "Stellar plate", -12.0, 0.55, 3.0, "Near −12 °C: a star with broad, flat, ribbed arms."),
            preset("plate", "Hexagonal plate", -2.0, 0.25, 1.0, "Near −2 °C, or in dry air: a simple hexagon."),
        ],
        // (a slight melt rounds the cells' corners along the outline into ice; 0.005 rather than less:
        // at 0.003 the sectored plate took 11.3 ms a frame at 4K, at 0.005 9.8)
        melt: 0.005,
        defaults: vec![
            ("ior", Setting::Num(1.31)),
            ("absorb", Setting::Vec3([0.05, 0.03, 0.015])),
            ("rot", Setting::Vec3([28.0, 0.0, 0.0])),
        ],
        // A plate facing the camera shows what is behind it: a dim broad light straight behind (it glows
        // softly) and two narrow lights behind the flake, off to its
        // sides, are bent into view by the relief's slopes; the solids' lights give the rims their glints.
        lights: {
            let mut l = solid_lights();
            l.push(light(0.0, 0.0, 30.0, 30.0, [0.12, 0.15, 0.21], 1.0));
            l.push(light(-28.0, 32.0, 10.0, 26.0, [0.9, 1.1, 1.5], 0.8));
            l.push(light(30.0, -24.0, 8.0, 22.0, [1.4, 1.0, 0.7], 0.8));
            l
        },
        // (one layer of glass is traced: a second shows nothing different on a thin plate, and costs
        // 2 ms a frame at 4K)
        layers: 1,
        ..Family::new(FAMILY, "Snowflakes", Kind::Curve)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_twelfth_stands_for_every_cell_alike() {
        // the twelve images of a cell share its representative
        let (q, r) = (5, 2);
        let c = canon(q, r);
        assert_eq!(canon(-q - r, q), c);
        assert_eq!(canon(r, q), c);
        assert_eq!(canon(-q, -r), c);
    }

    #[test]
    fn a_flake_is_sixfold_and_grows_quickly() {
        let (al, b, g, col) = nakaya(-15.0, 1.0);
        let t = std::time::Instant::now();
        let fl = grow_flake(al, b, g, 80, col);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        eprintln!("dendrite: {} cells, {} steps, {ms:.0} ms", fl.cells.len(), fl.steps);
        let set: std::collections::HashSet<(i32, i32)> = fl.cells.iter().map(|c| (c.0, c.1)).collect();
        for &(q, r, _) in &fl.cells {
            // turned by 60 degrees: (q, r) -> (-r, q + r)
            assert!(set.contains(&(-r, q + r)));
        }
        assert!(fl.cells.len() > 500);
    }

    /// How long the presets take to grow and to draw (`cargo test --release -- --nocapture timings`).
    #[test]
    fn timings() {
        for p in family().presets {
            let Shape::Curve(f) = &p.shape else { continue };
            let v = p.defaults();
            let t = std::time::Instant::now();
            let raw = f(&v, &CurveOpts { len: 1.0, quick: false, text: "" });
            let grown = t.elapsed().as_secs_f64() * 1e3;
            let mut w = v.clone();
            w[GROW] = 0.7;
            let t = std::time::Instant::now();
            let cut = f(&w, &CurveOpts { len: 1.0, quick: true, text: "" });
            let again = t.elapsed().as_secs_f64() * 1e3;
            eprintln!("{}: grown and drawn {grown:.0} ms ({} cells); grow moved {again:.1} ms ({} cells)", p.id, raw.pieces.len(), cut.pieces.len());
        }
    }

    #[test]
    fn growing_only_cuts_the_flake() {
        let (al, b, g, col) = nakaya(-12.0, 0.6);
        let fl = grow_flake(al, b, g, 50, col);
        let half = draw(&fl, 0.5, 1.0, 1.0, false);
        let whole = draw(&fl, 1.0, 1.0, 1.0, false);
        assert!(half.pieces.len() < whole.pieces.len());
        assert_eq!(half.frame, whole.frame);
    }

    #[test]
    fn dry_air_gives_plates_and_wet_air_at_minus_fifteen_ferns() {
        let (_, b0, g0, _) = nakaya(-15.0, 0.0);
        let (_, b1, g1, _) = nakaya(-15.0, 1.0);
        let (_, b2, _, _) = nakaya(-2.0, 1.0);
        assert!(b0 > 0.8 && g0 > 0.02);
        assert!(b1 < 0.45 && g1 < 0.002);
        assert!(b2 > 0.75);
    }
}

#[cfg(test)]
mod cost {
    use super::*;
    #[test]
    fn rebuild_cost() {
        let p = &family().presets[0];
        let Shape::Curve(f) = &p.shape else { return };
        let mut v = p.defaults();
        let _ = f(&v, &CurveOpts { len: 1.0, quick: true, text: "" });
        for g in [0.5, 0.7, 1.0] {
            v[GROW] = g;
            let t = std::time::Instant::now();
            let raw = f(&v, &CurveOpts { len: 1.0, quick: true, text: "" });
            let t1 = t.elapsed().as_secs_f64() * 1e3;
            let ds = 0.75 * 2.0 * crate::HB / 256.0;
            let pose = crate::curve::placement(&[&raw.frame[..]], &crate::curve::PlaceOpts { centre: true, ..Default::default() });
            let t = std::time::Instant::now();
            let mut c = crate::curve::prep_strands(&[], &[], &[], &crate::curve::StrandOpts { thick: 1.0, ds, place: Default::default(), pose: Some(pose), ground: None });
            crate::curve::add_pieces(&mut c, &raw.pieces, 1.0, ds);
            let t2 = t.elapsed().as_secs_f64() * 1e3;
            eprintln!("grow {g}: draw {t1:.1} ms, pieces {t2:.1} ms, {} points", c.n);
        }
    }
}
