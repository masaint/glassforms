//! The scene's cells: a grid of cubes over the scene's box, each holding the list of the pieces that can change a value there
//! (the pieces whose ball, grown by the value cap, touches the cell, and for a union piece only where its own distance there is
//! under the cap plus its seam). The lists are kept on the CPU in the order of the pieces, so the join order is the part order;
//! they are flattened into a pool (each cell a slot with room to grow) that the GPU reads by ranges, and edits write only the cells
//! that changed.
//!
//! Nothing here touches the GPU: `flatten` and `edit_spans` say what to write.

use super::Piece;

/// A box of cells: the low and the high corner (cell coordinates, both included).
pub type CellBox = ([usize; 3], [usize; 3]);

/// What an edit asks to be written.
#[derive(Debug, PartialEq)]
pub enum Spans {
    /// The pool or the pieces outgrew their buffers: write everything (`ranges`, `pool`).
    Full,
    /// Only these: the cells' ranges from the first changed cell to the last (inclusive), the pool slots rewritten where they were
    /// (a span of pool words), the pool words added at the end (a span).
    Edit { ranges: Option<(usize, usize)>, in_place: Option<(usize, usize)>, tail: Option<(usize, usize)> },
}

pub struct Cells {
    /// cells each way, a cell's side, the low corner of the grid
    pub gd: [usize; 3],
    pub cs: f64,
    pub lo: [f64; 3],
    /// the value cap: a piece reaches a cell it comes within this of
    pub band: f64,
    /// how far an empty cell's bound can reach (the far bound's cap)
    pub far_max: f64,
    /// lists only by ball (the union piece's own distance is not used to leave a cell out)
    pub ball_test: bool,
    /// each cell's pieces, in piece order
    pub lists: Vec<Vec<u32>>,
    /// the ball each piece was listed by
    pub listed: Vec<[f64; 4]>,
    /// entries in all lists
    pub items: usize,
    dirty: Vec<usize>,
    far_dirty: Option<CellBox>,
    /// the cells the last edit changed (None: not known, take all as changed)
    pub last_dirty: Option<Vec<usize>>,
    /// the pool: each cell's slot (offset, room), its range (offset, length), the words, the words in use and the capacity
    pub off: Vec<u32>,
    pub room: Vec<u32>,
    pub ranges: Vec<[u32; 2]>,
    pub pool: Vec<u32>,
    pub pool_used: usize,
    pub cap_items: usize,
    /// how many times everything was flattened
    pub full_flattens: usize,
}

fn widen(bb: Option<CellBox>, c: [usize; 3]) -> CellBox {
    match bb { None => (c, c), Some((lo, hi)) => ([0, 1, 2].map(|a| lo[a].min(c[a])), [0, 1, 2].map(|a| hi[a].max(c[a]))) }
}

/// The box that holds both.
pub fn merge(a: Option<CellBox>, b: CellBox) -> CellBox {
    match a { None => b, Some((lo, hi)) => ([0, 1, 2].map(|k| lo[k].min(b.0[k])), [0, 1, 2].map(|k| hi[k].max(b.1[k]))) }
}

impl Cells {
    pub fn new(gd: [usize; 3], cs: f64, lo: [f64; 3], band: f64, far_max: f64) -> Cells {
        let n = gd[0] * gd[1] * gd[2];
        Cells { gd, cs, lo, band, far_max, ball_test: false, lists: vec![Vec::new(); n], listed: Vec::new(), items: 0, dirty: Vec::new(), far_dirty: None, last_dirty: None, off: vec![0; n], room: vec![0; n], ranges: vec![[0, 0]; n], pool: Vec::new(), pool_used: 0, cap_items: 0, full_flattens: 0 }
    }

    pub fn count(&self) -> usize { self.gd[0] * self.gd[1] * self.gd[2] }
    pub fn coords(&self, ci: usize) -> [usize; 3] { [ci % self.gd[0], (ci / self.gd[0]) % self.gd[1], ci / (self.gd[0] * self.gd[1])] }
    pub fn index(&self, c: [usize; 3]) -> usize { c[0] + self.gd[0] * (c[1] + self.gd[1] * c[2]) }
    pub fn mid(&self, c: [usize; 3]) -> [f64; 3] { [0, 1, 2].map(|a| self.lo[a] + (c[a] as f64 + 0.5) * self.cs) }

    /// The cells whose box is within a ball's reach plus the band, and for a union piece (`own`) only those where its own distance at the
    /// cell's middle, less the cell's half diagonal, is under the band plus its seam: elsewhere it cannot change a value below the band.
    pub fn cells_in_ball(&self, b: &[f64; 4], own: Option<&Piece>) -> Vec<usize> {
        let (gd, cs, lo) = (self.gd, self.cs, self.lo);
        let cell = |x: f64, a: usize| ((x - lo[a]) / cs).floor();
        let r = b[3] + self.band;
        let c0 = [0, 1, 2].map(|a| cell(b[a] - r, a).clamp(0.0, (gd[a] - 1) as f64) as usize);
        let c1 = [0, 1, 2].map(|a| cell(b[a] + r, a).clamp(0.0, (gd[a] - 1) as f64) as usize);
        let mut out = Vec::new();
        for z in c0[2]..=c1[2] {
            for y in c0[1]..=c1[1] {
                for x in c0[0]..=c1[0] {
                    let c = [x, y, z];
                    let mut d2 = 0.0;
                    for a in 0..3 {
                        let l = lo[a] + c[a] as f64 * cs;
                        let o = (l - b[a]).max(b[a] - (l + cs)).max(0.0);
                        d2 += o * o;
                    }
                    if d2 > r * r { continue; }
                    if let Some(p) = own {
                        if p.lower(self.mid(c)) - 0.8660254 * cs >= self.band + p.seam() { continue; }
                    }
                    out.push(x + gd[0] * (y + gd[1] * z));
                }
            }
        }
        out
    }

    fn own<'a>(&self, p: &'a Piece) -> Option<&'a Piece> { if p.a[2] < 1.5 && !self.ball_test { Some(p) } else { None } }

    /// Every piece binned into the cells (parallel over the pieces); the lists are made anew.
    pub fn bin_all(&mut self, pieces: &[Piece], balls: &[[f64; 4]]) {
        let n = pieces.len();
        for l in &mut self.lists { l.clear(); }
        self.items = 0;
        let threads = std::thread::available_parallelism().map_or(4, |x| x.get()).clamp(1, 16);
        let chunk = n.div_ceil(threads).max(1);
        let found: Vec<Vec<Vec<usize>>> = {
            let me = &*self;
            std::thread::scope(|sc| {
                let hs: Vec<_> = (0..n).step_by(chunk).map(|a| sc.spawn(move || {
                    (a..(a + chunk).min(n)).map(|i| me.cells_in_ball(&balls[i], me.own(&pieces[i]))).collect::<Vec<_>>()
                })).collect();
                hs.into_iter().map(|h| h.join().unwrap()).collect()
            })
        };
        self.listed = balls.to_vec();
        for (i, cells) in found.into_iter().flatten().enumerate() {
            self.items += cells.len();
            for ci in cells { self.lists[ci].push(i as u32); }
        }
        self.dirty.clear();
        self.far_dirty = None;
    }

    /// Piece `i` (the piece `p`, listed by ball `b`) put in its cells, in order. Returns the box of the cells it entered.
    pub fn list_insert(&mut self, i: usize, b: [f64; 4], p: &Piece) -> Option<CellBox> {
        let mut bb: Option<CellBox> = None;
        let cells = self.cells_in_ball(&b, self.own(p));
        for ci in cells {
            let c = self.coords(ci);
            let l = &mut self.lists[ci];
            let at = l.partition_point(|&x| (x as usize) < i);
            l.insert(at, i as u32);
            self.dirty.push(ci);
            self.items += 1;
            bb = Some(widen(bb, c));
        }
        if self.listed.len() <= i { self.listed.resize(i + 1, [0.0; 4]); }
        self.listed[i] = b;
        bb
    }

    /// Piece `i` taken out of the cells its listing ball touched. Returns the box of the cells it left.
    pub fn list_remove(&mut self, i: usize) -> Option<CellBox> {
        let mut bb: Option<CellBox> = None;
        let b = self.listed[i];
        for ci in self.cells_in_ball(&b, None) {
            let c = self.coords(ci);
            let l = &mut self.lists[ci];
            if let Ok(at) = l.binary_search(&(i as u32)) { l.remove(at); self.dirty.push(ci); self.items -= 1; }
            bb = Some(widen(bb, c));
        }
        bb
    }

    /// The pieces `idx` taken out of the lists where they were listed and put in by their balls now; the bounds near the change marked.
    /// Returns the box of the cells touched.
    pub fn relist(&mut self, pieces: &[Piece], balls: &[[f64; 4]], idx: &[usize]) -> Option<CellBox> {
        let mut bb: Option<CellBox> = None;
        for &i in idx {
            if let Some(b) = self.list_remove(i) { bb = Some(merge(bb, b)); }
            if let Some(b) = self.list_insert(i, balls[i], &pieces[i]) { bb = Some(merge(bb, b)); }
        }
        if let Some((lo, hi)) = bb { self.far_update(lo, hi); }
        bb
    }

    /// The bound of each empty cell (the distance to the nearest cell holding pieces, at most `far_max`) and the cap of each held one
    /// is to be made again for the cells within the grow's reach of the box `lo..=hi` of changed cells.
    pub fn far_update(&mut self, lo: [usize; 3], hi: [usize; 3]) {
        let g = self.gd;
        let cap = self.far_cap();
        let rlo = lo.map(|x| x.saturating_sub(cap));
        let rhi = [0, 1, 2].map(|a| (hi[a] + cap).min(g[a] - 1));
        self.far_dirty = Some(match self.far_dirty { None => (rlo, rhi), Some((a, b)) => ([0, 1, 2].map(|k| a[k].min(rlo[k])), [0, 1, 2].map(|k| b[k].max(rhi[k]))) });
    }

    /// How many cells the far bound's grow reaches.
    pub fn far_cap(&self) -> usize { (self.far_max / self.cs).ceil() as usize + 1 }

    /// The region of cells whose bounds are to be made again (once; cleared).
    pub fn take_far(&mut self) -> Option<CellBox> { self.far_dirty.take() }

    /// Everything flattened anew: each cell's slot in the pool (with room to grow), the ranges, the pool; the whole grid's bounds marked.
    pub fn flatten(&mut self) {
        let n = self.count();
        self.off = vec![0; n];
        self.room = vec![0; n];
        self.ranges = vec![[0, 0]; n];
        let mut used = 0usize;
        for (ci, l) in self.lists.iter().enumerate() {
            let room = if l.is_empty() { 0 } else { l.len() + l.len() / 2 + 2 };
            self.off[ci] = used as u32;
            self.room[ci] = room as u32;
            self.ranges[ci] = [used as u32, l.len() as u32];
            used += room;
        }
        self.cap_items = used + used / 2 + 4096;
        self.pool = vec![0; self.cap_items];
        for (ci, l) in self.lists.iter().enumerate() { let o = self.off[ci] as usize; self.pool[o..o + l.len()].copy_from_slice(l); }
        self.pool_used = used;
        self.full_flattens += 1;
        self.dirty.clear();
        self.far_dirty = Some(([0; 3], [self.gd[0] - 1, self.gd[1] - 1, self.gd[2] - 1]));
    }

    /// Only what changed: the changed cells copied into their slots of the pool (a cell that has outgrown its room gets a new slot at the
    /// pool's end), and the spans to write: the cells' ranges, the slots rewritten, the words added. `Full` (and everything flattened
    /// anew) when the pool is full. (Many small writes cost more than one span: a write is a command of its own.)
    pub fn edit_spans(&mut self) -> Spans {
        let mut cells = std::mem::take(&mut self.dirty);
        cells.sort_unstable();
        cells.dedup();
        self.last_dirty = Some(cells.clone());
        let tail_from = self.pool_used;
        let (mut in_lo, mut in_hi) = (usize::MAX, 0usize);
        for &ci in &cells {
            let need = self.lists[ci].len();
            if need > self.room[ci] as usize {
                let room = need + need / 2 + 2;
                if self.pool_used + room > self.cap_items { self.flatten(); self.last_dirty = None; return Spans::Full; }
                self.off[ci] = self.pool_used as u32;
                self.room[ci] = room as u32;
                self.pool_used += room;
            } else if self.room[ci] > 0 {
                in_lo = in_lo.min(self.off[ci] as usize);
                in_hi = in_hi.max((self.off[ci] + self.room[ci]) as usize);
            }
            let o = self.off[ci] as usize;
            self.pool[o..o + need].copy_from_slice(&self.lists[ci]);
            self.ranges[ci] = [self.off[ci], need as u32];
        }
        Spans::Edit {
            ranges: cells.first().zip(cells.last()).map(|(a, b)| (*a, *b)),
            in_place: (in_lo < in_hi).then_some((in_lo, in_hi)),
            tail: (self.pool_used > tail_from).then_some((tail_from, self.pool_used)),
        }
    }

    /// The cells the pieces' lists changed in since the last `edit_spans` (the bricks, the far bounds follow them).
    pub fn pending(&self) -> usize { self.dirty.len() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parts::kind;

    /// A small deterministic generator (no crate for it here).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> f64 { self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((self.0 >> 11) as f64) / ((1u64 << 53) as f64) }
        fn range(&mut self, a: f64, b: f64) -> f64 { a + (b - a) * self.next() }
    }

    /// A random scene: spheres and capsules, some cuts, in a cube of half-side 1.5.
    fn scene(n: usize, seed: u64) -> (Vec<Piece>, Vec<[f64; 4]>) {
        let mut r = Rng(seed);
        let mut pieces = Vec::new();
        let mut balls = Vec::new();
        for _ in 0..n {
            let c = [r.range(-1.4, 1.4), r.range(-1.4, 1.4), r.range(-1.4, 1.4)];
            let rad = r.range(0.01, 0.2);
            let cut = r.next() < 0.1;
            let mut p = Piece::default();
            p.inv = [[1.0, 0.0, 0.0, -c[0] as f32], [0.0, 1.0, 0.0, -c[1] as f32], [0.0, 0.0, 1.0, -c[2] as f32]];
            if r.next() < 0.5 { p.a = [kind::SPHERE, 1.0, if cut { 2.0 } else { 1.0 }, 0.0]; p.b = [rad as f32, 0.0, 0.0, 0.0]; }
            else { p.a = [kind::CAPSULE, 1.0, if cut { 2.0 } else { 1.0 }, 0.0]; p.b = [(rad * 2.0) as f32, rad as f32, 0.0, 0.0]; }
            let reach = if p.a[0] == kind::SPHERE { rad } else { rad * 3.0 };
            pieces.push(p);
            balls.push([c[0], c[1], c[2], reach]);
        }
        (pieces, balls)
    }

    fn grid() -> Cells { Cells::new([16; 3], 3.0 / 16.0, [-1.5; 3], 0.2, 1.0) }

    /// The lists by a brute-force pass over every piece and every cell.
    fn brute(c: &Cells, pieces: &[Piece], balls: &[[f64; 4]], ball_only: bool) -> Vec<Vec<u32>> {
        let mut out = vec![Vec::new(); c.count()];
        for (i, (p, b)) in pieces.iter().zip(balls).enumerate() {
            let r = b[3] + c.band;
            for ci in 0..c.count() {
                let cc = c.coords(ci);
                let mut d2 = 0.0;
                for a in 0..3 {
                    let l = c.lo[a] + cc[a] as f64 * c.cs;
                    let o = (l - b[a]).max(b[a] - (l + c.cs)).max(0.0);
                    d2 += o * o;
                }
                if d2 > r * r { continue; }
                if !ball_only && p.a[2] < 1.5 && p.lower(c.mid(cc)) - 0.8660254 * c.cs >= c.band + p.seam() { continue; }
                out[ci].push(i as u32);
            }
        }
        out
    }

    #[test]
    fn lists_equal_a_brute_force_ball_test() {
        for (n, seed) in [(1usize, 1u64), (7, 2), (64, 3), (500, 4), (4096, 5)] {
            let (pieces, balls) = scene(n, seed);
            for ball_only in [true, false] {
                let mut c = grid();
                c.ball_test = ball_only;
                c.bin_all(&pieces, &balls);
                assert_eq!(c.lists, brute(&c, &pieces, &balls, ball_only), "n {n} ball_only {ball_only}");
                assert_eq!(c.items, c.lists.iter().map(|l| l.len()).sum::<usize>());
            }
        }
    }

    #[test]
    fn insert_then_remove_leaves_the_lists_byte_equal() {
        let (pieces, balls) = scene(300, 9);
        let mut c = grid();
        c.bin_all(&pieces[..299], &balls[..299]);
        let before = c.lists.clone();
        let items = c.items;
        c.list_insert(299, balls[299], &pieces[299]);
        assert_ne!(c.lists, before);
        c.list_remove(299);
        assert_eq!(c.lists, before);
        assert_eq!(c.items, items);
        // a piece in the middle of the list: out and in again, in its place
        let full = { let mut f = grid(); f.bin_all(&pieces, &balls); f.lists };
        let mut d = grid();
        d.bin_all(&pieces, &balls);
        d.list_remove(150);
        d.list_insert(150, balls[150], &pieces[150]);
        assert_eq!(d.lists, full);
    }

    #[test]
    fn relist_moves_a_piece_and_matches_a_fresh_binning() {
        let (mut pieces, mut balls) = scene(400, 11);
        let mut c = grid();
        c.bin_all(&pieces, &balls);
        for i in [3usize, 77, 399] {
            balls[i][0] += 0.3;
            pieces[i].inv[0][3] -= 0.3;
        }
        c.relist(&pieces, &balls, &[3, 77, 399]);
        let mut fresh = grid();
        fresh.bin_all(&pieces, &balls);
        assert_eq!(c.lists, fresh.lists);
        assert!(c.take_far().is_some());
    }

    #[test]
    fn edits_write_only_the_changed_cells_and_the_pool_holds_the_lists() {
        let (pieces, balls) = scene(500, 13);
        let mut c = grid();
        c.bin_all(&pieces[..499], &balls[..499]);
        c.flatten();
        let check = |c: &Cells| {
            for ci in 0..c.count() {
                let [o, n] = c.ranges[ci];
                assert_eq!(&c.pool[o as usize..(o + n) as usize], &c.lists[ci][..], "cell {ci}");
            }
        };
        check(&c);
        c.list_insert(499, balls[499], &pieces[499]);
        match c.edit_spans() {
            Spans::Edit { ranges, .. } => assert!(ranges.is_some()),
            Spans::Full => panic!("a small edit asked for a full write"),
        }
        check(&c);
        assert_eq!(c.last_dirty.as_ref().map(|d| d.len() > 0), Some(true));
        // an edit that fills a cell's room moves it to the pool's end
        let mut d = grid();
        d.bin_all(&pieces, &balls);
        d.flatten();
        let ci = (0..d.count()).find(|&i| d.lists[i].len() > 3).expect("a busy cell");
        for k in 0..40u32 { d.lists[ci].push(1000 + k); d.dirty.push(ci); }
        match d.edit_spans() {
            Spans::Edit { tail, .. } => assert!(tail.is_some()),
            Spans::Full => {}
        }
        check(&d);
    }
}
