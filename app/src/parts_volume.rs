//! The parts engine's GPU side: a scene of many parts held as one volume, baked from the cells' lists (`gf_core::parts`).
//!
//! The scene's parts are pieces (one record each) binned into a 64^3 grid of cells; a voxel reads only the pieces of its cell, joined in
//! order (union, cut, the quadratic smooth min), its value kept within a band (beyond it a lower bound: the band, or in an empty cell the
//! way to the nearest cell that holds pieces). So an edit of a part touches only the voxels within its reach plus the band: a drag
//! starts, draws and ends over a box, whatever the number of parts.
//!
//! Three volumes of the stage's cube: the scene's (what the tracer reads at rest), the rest (the scene without the dragged parts) and
//! the one shown while dragging (the rest joined with the live parts).
//!   build    cells (CPU) + the whole volume
//!   start    rest = the scene's volume with the dragged parts' box built again without them
//!   live     one pass over the shown volume: the rest, joined with the live parts where they reach, elsewhere
//!            min(rest, the way to their bounding ball): a lower bound, so the march stays safe
//!   fold     the cells with the parts where they now stand; the old and new boxes built again
//!   reshape  a slider moved: the pieces made again, those that changed taken out of the lists and put in, their boxes built again
//!   add      a part added: its cells, its box
//!
//! The kinds' distances are the app's own text (`gf_core::csg::parts_kinds_wgsl`), generated for the kinds the scene holds; the point
//! warps are the app's modifier text read from `Mods::exact_numbers`, the exact tubes the app's table; two objects joined read
//! `combine.wgsl`'s `dip` with the pair's radius and profile. Nothing here ever panics for want of memory: it says
//! "refused: not enough graphics memory" and the caller keeps the existing path.

use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use gf_core::parts::{cells::{Cells, Spans}, Derived, PartsScene, GPU_FLOATS, TUBE_KIND};
use gf_core::state::State;

use crate::bake::{Volume, VOL_FORMAT};
use crate::gpu::{is_out_of_memory, storage_entry, uniform_entry, Gpu, R};

/// The words the caller shows when memory is short.
pub const NOT_ENOUGH_MEMORY: &str = "refused: not enough graphics memory";

/// The most live pieces at once (the dragged parts and the cuts that reach them).
const MAX_LIVE: usize = 128;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct BU { h: f32, band: f32, cs: f32, pad: f32, lo: [f32; 4], gd: [u32; 4], org: [u32; 4], ext: [u32; 4], ball: [f32; 4], n: [u32; 4] }

const HEAD: &str = "
struct Piece { inv0: vec4f, inv1: vec4f, inv2: vec4f, a: vec4f, v0: vec4f, v1: vec4f, w: vec4f, x: vec4f }
@group(0) @binding(0) var<storage, read> pcs: array<Piece>;
struct B { h: f32, band: f32, cs: f32, pad: f32, lo: vec4f, gd: vec4u, org: vec4u, ext: vec4u, ball: vec4f, n: vec4u }
@group(0) @binding(4) var<uniform> b: B;
fn smin(a: f32, b: f32, k: f32) -> f32 {
  if (k <= 0.0) { return min(a, b); }
  let h = max(k - abs(a - b), 0.0) / k;
  return min(a, b) - h * h * k * 0.25;
}
";

const PIECE_DIST: &str = "
fn pieceDist0(i: u32, p: vec3f) -> f32 {
  let o = pcs[i];
  let q = vec3f(dot(o.inv0.xyz, p) + o.inv0.w, dot(o.inv1.xyz, p) + o.inv1.w, dot(o.inv2.xyz, p) + o.inv2.w);
  var d: f32;
  if (o.a.x > 999.5) {
//TUBE
  } else {
    d = partKind(i32(o.a.x), i, q);
    if (o.x.z > 0.5) {
      // (never under its block's distance: the block holds the part)
      let e = abs(q) - o.w.xyz;
      d = max(d, length(max(e, vec3f(0.0))) + min(max(e.x, max(e.y, e.z)), 0.0));
    }
  }
  return d * o.a.y;
}
fn pieceDist(i: u32, p: vec3f) -> f32 {
  if (pcs[i].x.y != 0.0) { return warpDist(i, p); }
  return pieceDist0(i, p);
}
fn join(d: f32, o: Piece, e: f32) -> f32 {
  if (o.a.z > 1.5) { return -smin(-d, e, o.a.w); }
  return smin(d, e, o.a.w);
}
fn voxelAt(v: vec3u) -> vec3f { return b.lo.xyz + (vec3f(v) + 0.5) * b.h; }
";

const TUBE_BODY: &str = "    tbase = u32(o.v0.x);
    let hd = T(1u);
    let nq = u32(T(2u).x);
    let qq = q + hd.xyz + vec3f(0.5 * T(2u).x * hd.w);
    // (far from every span the table's own far field, never over the distance, is read: nothing nearer than the band, plus the seam, matters)
    let fr = tube_far(qq, hd, nq).w;
    if (fr * o.a.y >= b.band + o.a.w) { d = fr; } else { d = tube_eval(qq).w; }";

/// The app's tube formula read from one buffer of tables, a table at `tbase`: every `tbl[i]` of its text becomes `T(i)`.
fn tube_text(src: &str) -> String {
    let mut out = String::from("@group(0) @binding(7) var<storage, read> tbl: array<vec4f>;\nvar<private> tbase: u32 = 0u;\nfn T(i: u32) -> vec4f { return tbl[tbase + i]; }\n");
    let mut rest = src;
    while let Some(at) = rest.find("tbl[") {
        out.push_str(&rest[..at]);
        let mut depth = 0;
        let mut end = at + 3;
        for (k, c) in rest[at + 3..].char_indices() {
            if c == '[' { depth += 1; }
            if c == ']' { depth -= 1; if depth == 0 { end = at + 3 + k; break; } }
        }
        out.push_str("T(");
        out.push_str(&rest[at + 4..end]);
        out.push(')');
        rest = &rest[end + 1..];
    }
    out.push_str(rest);
    out
}

/// The point warps (an object's twist, bend or taper) as shader text: the app's own entries (`Mods::exact_numbers`: a header of the
/// object's size and its way to the object's frame and back, then 36 numbers a modifier) in a storage buffer, and the app's own `modBack`.
/// A warped piece is read at the point the warp brings the sample back to; the distance is times the stack's 1 / Lipschitz bound
/// (the app's `modOn`), and beyond a modifier's guard sphere the way to that sphere.
fn warp_text(offs: &[u32]) -> String {
    if offs.is_empty() { return "fn warpDist(i: u32, p: vec3f) -> f32 { return 0.0; }\n".to_string(); }
    let o: Vec<String> = offs.iter().map(|v| format!("{v}u")).collect();
    format!("{lib}
@group(0) @binding(8) var<storage, read> WV: array<vec4f>;
var<private> WO: array<u32, {n}> = array<u32, {n}>({offs});
fn modAtW(i: u32) -> ModE {{ return ModE(WV[i], WV[i + 1u], WV[i + 2u], WV[i + 3u], WV[i + 4u], WV[i + 5u], WV[i + 6u], WV[i + 7u], WV[i + 8u]); }}
fn warpDist(pi: u32, p: vec3f) -> f32 {{
  let at = WO[u32(pcs[pi].x.y + 0.5) - 1u];
  let n = i32(WV[at].x);
  let so = WV[at].y;
  var x = vec3f(dot(WV[at + 1u].xyz, p) + WV[at + 1u].w, dot(WV[at + 2u].xyz, p) + WV[at + 2u].w, dot(WV[at + 3u].xyz, p) + WV[at + 3u].w);
  var stop = -1;
  var d = 0.0;
  var walls = array<f32, 8>();
  for (var j = n - 1; j >= 0; j--) {{
    let m = modAtW(at + 7u + 9u * u32(j));
    let u = modLocal(m, x);
    if (m.q5.w > 0.0 && length(u) > m.q6.w) {{
      stop = j;
      d = length(u) - m.q5.w;
      break;
    }}
    var w = 1e9;
    x = modWorld(m, modBack(m, u, &w));
    walls[j] = w;
  }}
  if (stop < 0) {{
    let y = vec3f(dot(WV[at + 4u], vec4f(x, 1.0)), dot(WV[at + 5u], vec4f(x, 1.0)), dot(WV[at + 6u], vec4f(x, 1.0)));
    d = pieceDist0(pi, y) / so;
  }}
  for (var j = stop + 1; j < n; j++) {{ d = modOn(modAtW(at + 7u + 9u * u32(j)), d, walls[j]); }}
  d = d * so;
  return d;
}}
", lib = gf_core::mods::lib_wgsl(), n = offs.len(), offs = o.join(", "))
}

/// The pair's join: min(a, b) less what a seam of the pair's radius and profile takes off where the two differ by less than it
/// (`combine.wgsl`: `dip`, `u2`; the radius is in the scene's own units, as `combine`'s `K`).
fn pair_text(pair: Option<(f32, f32)>) -> String {
    let (on, k, n) = pair.map_or(("false", 0.0, 2.0), |p| ("true", p.0, p.1));
    format!("const PAIR_ON: bool = {on};\nconst PAIR_K: f32 = {k:?};\nconst PAIR_N: f32 = {n:?};
fn dip(k: f32, n: f32, delta: f32) -> f32 {{
  if (k <= 0.0 || delta >= k) {{ return 0.0; }}
  let t = 1.0 - delta / k;
  if (n < 1.5) {{ return 0.5 * k * t; }}
  if (n < 2.5) {{ return 0.25 * k * t * t; }}
  return k * t * t * t / 6.0;
}}
fn pairJoin(a: f32, b: f32) -> f32 {{
  if (!PAIR_ON) {{ return min(a, b); }}
  return min(a, b) - dip(PAIR_K, PAIR_N, abs(a - b));
}}
")
}

const BAKE: &str = "
@group(0) @binding(1) var<storage, read> cellRange: array<vec2u>;
@group(0) @binding(2) var<storage, read> cellItems: array<u32>;
@group(0) @binding(3) var<storage, read> cellFar: array<f32>;
@group(0) @binding(5) var outTex: texture_storage_3d<r16float, write>;
@compute @workgroup_size(4, 4, 4)
fn cs(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= b.ext.xyz)) { return; }
  let v = id + b.org.xyz;
  let p = voxelAt(v);
  let c = min(vec3u(clamp(floor((p - b.lo.xyz) / b.cs), vec3f(0.0), vec3f(b.gd.xyz) - 1.0)), b.gd.xyz - 1u);
  let ci = c.x + b.gd.x * (c.y + b.gd.y * c.z);
  let s = cellRange[ci].x;
  let e = s + cellRange[ci].y;
  var d = 1e9;
  var d2 = 1e9;
  var tot = 1e9;
  var cur = -1.0;
  for (var j = s; j < e; j++) {
    let ii = cellItems[j];
    let o = pcs[ii];
    // (a piece whose op is 0 is the one being dragged: left out of the rest)
    if (o.a.z < 0.5) { continue; }
    // (objects set apart: each its own running value, a cut cutting its own object's parts only; they are in the list object after object)
    if (!PAIR_ON && o.x.w != cur) { tot = min(tot, d); d = 1e9; cur = o.x.w; }
    let e1 = pieceDist(ii, p);
    // (the second object of a joined pair has its own value; with no pair d2 stays empty)
    if (o.x.x > 0.5) { d2 = join(d2, o, e1); } else { d = join(d, o, e1); }
  }
  if (!PAIR_ON) { d = min(tot, d); }
  d = min(pairJoin(d, d2), cellFar[ci]);
  textureStore(outTex, vec3i(v), vec4f(d, 0.0, 0.0, 0.0));
}
";

/// Each cell's value cap in a region: the band where it holds pieces, else the distance (in cells, Chebyshev) to the nearest cell that
/// does, less one, at least the band, at most the far cap.
const FAR: &str = "
struct F { gd: vec4u, org: vec4u, ext: vec4u, p: vec4f }
@group(0) @binding(0) var<storage, read> ranges: array<vec2u>;
@group(0) @binding(1) var<storage, read_write> far: array<f32>;
@group(0) @binding(2) var<uniform> f: F;
@compute @workgroup_size(4, 4, 4)
fn cs(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= f.ext.xyz)) { return; }
  let c = vec3i(id + f.org.xyz);
  let g = vec3i(f.gd.xyz);
  let cap = i32(f.p.w);
  var best = cap + 1;
  for (var dz = -cap; dz <= cap; dz++) {
    let z = c.z + dz;
    if (z < 0 || z >= g.z) { continue; }
    for (var dy = -cap; dy <= cap; dy++) {
      let y = c.y + dy;
      if (y < 0 || y >= g.y) { continue; }
      let m = max(abs(dy), abs(dz));
      for (var dx = -cap; dx <= cap; dx++) {
        let x = c.x + dx;
        if (x < 0 || x >= g.x) { continue; }
        if (ranges[x + g.x * (y + g.y * z)].y > 0u) { best = min(best, max(abs(dx), m)); }
      }
    }
  }
  var v: f32;
  if (best == 0) { v = f.p.x; } else if (best > cap) { v = f.p.y; } else { v = min(max(f32(best - 1) * f.p.z, f.p.x), f.p.y); }
  far[c.x + g.x * (c.y + g.y * c.z)] = v;
}
";

const LIVE: &str = "
@group(0) @binding(5) var outTex: texture_storage_3d<r16float, write>;
@group(0) @binding(6) var restTex: texture_3d<f32>;
@compute @workgroup_size(4, 4, 4)
fn cs(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= b.ext.xyz)) { return; }
  let v = id + b.org.xyz;
  let p = voxelAt(v);
  let r = textureLoad(restTex, vec3i(v), 0).r;
  let sd = length(p - b.ball.xyz) - b.ball.w;
  var d = r;
  if (sd > b.band) {
    d = min(r, sd);
  } else {
    // the live parts joined among themselves in list order (op 1 union, 2 cut: the live parts and the rest, 3 a hole
    // cut later in the list: the live parts only, the rest was built with it), then joined onto the rest
    var l = 1e9;
    var rc = r;
    var k = 0.0;
    for (var j = 0u; j < b.n.x; j++) {
      let o = pcs[j];
      let e = pieceDist(j, p);
      if (o.a.z < 1.5) {
        l = smin(l, e, o.a.w);
        k = max(k, o.a.w);
      } else if (o.a.z < 2.5) {
        l = -smin(-l, e, o.a.w);
        rc = -smin(-rc, e, o.a.w);
      } else {
        l = -smin(-l, e, o.a.w);
      }
    }
    if (PAIR_ON) { d = pairJoin(rc, l); } else { d = smin(rc, l, k); }
  }
  textureStore(outTex, vec3i(v), vec4f(d, 0.0, 0.0, 0.0));
}
";


/// The fine bricks (refine.rs): the same exact formulas as BAKE, one brick of nb^3 texels (a one-texel apron) for each record (cell, slot).
const REFINE_BAKE: &str = "
@group(0) @binding(1) var<storage, read> cellRange: array<vec2u>;
@group(0) @binding(2) var<storage, read> cellItems: array<u32>;
@group(0) @binding(3) var<storage, read> cellFar: array<f32>;
@group(0) @binding(5) var outTex: texture_storage_3d<r16float, write>;
@group(0) @binding(9) var<storage, read> recs: array<vec4u>;
@compute @workgroup_size(4, 4, 4)
fn cs(@builtin(global_invocation_id) id: vec3u) {
  let nb = b.ext.x;
  if (id.x >= nb || id.y >= nb || id.z >= nb * b.ext.y) { return; }
  let r = recs[id.z / nb];
  let lz = id.z % nb;
  let hf = b.cs / f32(nb - 2u);
  let p = b.lo.xyz + vec3f(r.xyz) * b.cs + (vec3f(f32(id.x), f32(id.y), f32(lz)) - 0.5) * hf;
  let ci = r.x + b.gd.x * (r.y + b.gd.y * r.z);
  let s = cellRange[ci].x;
  let e = s + cellRange[ci].y;
  var d = 1e9;
  var d2 = 1e9;
  var tot = 1e9;
  var cur = -1.0;
  for (var j = s; j < e; j++) {
    let ii = cellItems[j];
    let o = pcs[ii];
    if (o.a.z < 0.5) { continue; }
    if (!PAIR_ON && o.x.w != cur) { tot = min(tot, d); d = 1e9; cur = o.x.w; }
    let e1 = pieceDist(ii, p);
    if (o.x.x > 0.5) { d2 = join(d2, o, e1); } else { d = join(d, o, e1); }
  }
  if (!PAIR_ON) { d = min(tot, d); }
  d = min(pairJoin(d, d2), cellFar[ci]);
  let k = r.w;
  let o3 = vec3u(k % b.ext.z, (k / b.ext.z) % b.ext.w, k / (b.ext.z * b.ext.w)) * nb;
  textureStore(outTex, vec3i(o3 + vec3u(id.x, id.y, lz)), vec4f(d, 0.0, 0.0, 0.0));
}
";

/// The fine bricks of thin parts: a brick of finer voxels for each cell of the grid that holds a thin part near it.
struct Refine {
    on: bool,
    /// fine voxels a coarse voxel, the thin limit and the gate in coarse voxels, the budget in MB
    f: u32,
    thin: f64,
    gate: f64,
    budget_mb: f64,
    /// texels a brick side (with the apron), bricks along x and y and z, the most bricks
    nb: u32,
    per: [u32; 3],
    cap: u32,
    atlas: Option<(wgpu::Texture, wgpu::TextureView)>,
    page_tex: Option<wgpu::Texture>,
    page_view: Option<wgpu::TextureView>,
    /// cell -> slot + 1 (0 coarse); cells whose brick is hidden (a live part is over them)
    page: Vec<u16>,
    mask: Vec<bool>,
    masked: Vec<usize>,
    free: Vec<u32>,
    pipe: Option<Pipe>,
    recs: Option<wgpu::Buffer>,
    fr_ub: Option<wgpu::Buffer>,
    dropped: u32,
    /// the pieces the edit under way moved (None: every cell asked for is baked again)
    moved: Option<Vec<usize>>,
    thin_cache: Vec<bool>,
}
impl Default for Refine {
    fn default() -> Self {
        Self { on: false, f: 2, thin: 2.0, gate: 1.5, budget_mb: 250.0, nb: 0, per: [0; 3], cap: 0, atlas: None, page_tex: None, page_view: None, page: Vec::new(), mask: Vec::new(), masked: Vec::new(), free: Vec::new(), pipe: None, recs: None, fr_ub: None, dropped: 0, moved: None, thin_cache: Vec::new() }
    }
}

type Pipe = (wgpu::ComputePipeline, wgpu::BindGroupLayout);

fn pipe(gpu: &Gpu, label: &str, text: &str, entries: &[wgpu::BindGroupLayoutEntry]) -> Pipe {
    let layout = gpu.bind_layout(label, entries);
    let module = gpu.shader(label, text, true);
    let pl = gpu.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
    let p = gpu.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(label), layout: Some(&pl), module: &module, entry_point: Some("cs"), compilation_options: Default::default(), cache: None });
    (p, layout)
}

fn out_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: VOL_FORMAT, view_dimension: wgpu::TextureViewDimension::D3 }, count: None }
}
fn tex_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false }, count: None }
}
fn rw_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None }
}

struct Bufs {
    pieces: wgpu::Buffer,
    ranges: wgpu::Buffer,
    items: wgpu::Buffer,
    far: wgpu::Buffer,
}

/// The scene as one volume of its parts.
pub struct PartsVolume {
    gpu: Gpu,
    pub scene: PartsScene,
    cells: Cells,
    pub res: u32,
    hb: f64,
    h: f64,
    band: f64,
    far_max: f64,
    bake: Pipe,
    live_pipe: Pipe,
    far_pipe: Pipe,
    bufs: Bufs,
    tube_buf: wgpu::Buffer,
    warp_buf: wgpu::Buffer,
    ubuf: wgpu::Buffer,
    far_ubuf: wgpu::Buffer,
    lbuf: wgpu::Buffer,
    cap_pieces: usize,
    /// the scene's volume (shown at rest), the rest without the live parts, the volume shown while dragging
    vol: Volume,
    rest: Volume,
    shown: Volume,
    /// the pieces being dragged
    live: Vec<usize>,
    old_box: Option<([f64; 3], [f64; 3])>,
    prev_box: Option<([f64; 3], [f64; 3])>,
    /// the cell lists' bytes and the buffers' (for `memory`)
    bytes: u64,
    /// the shaders' common text (the refinement's pipeline is made from it)
    common: String,
    refine: Refine,
    /// where the time goes: (what, ms) of every step, newest last
    pub laps: Vec<(String, f64)>,
}

fn region_dims(res: u32) -> [u32; 3] { [res; 3] }

impl PartsVolume {
    /// Is the whole scene readable by the parts engine, and does this device have what it needs? The sentence for each refusal.
    pub fn accepts(gpu: &Gpu, state: &State) -> R<()> {
        if !gpu.storage16 { return Err("refused: this device cannot write 16-bit float volumes".into()); }
        gf_core::parts::accepts(state).map_err(|e| format!("refused: {e}"))
    }

    /// Half the side of the stage the volume spans.
    pub fn hb(&self) -> f64 { self.hb }

    /// The memory a build of a scene at `res` would hold, about (3 volumes, the cells' buffers for `pieces` pieces).
    pub fn estimate(res: u32, pieces: usize) -> u64 {
        3 * (res as u64).pow(3) * 2 + (64u64.pow(3)) * 12 + pieces as u64 * 128 * 2 + (pieces as u64 * 24 * 4).max(1 << 20)
    }

    /// Build the scene's volume. `limit`: the most bytes it may hold (the settle budget); over it, or when the device runs out, this is
    /// `NOT_ENOUGH_MEMORY` and nothing is kept.
    pub fn build(gpu: &Gpu, state: &mut State, res: u32, limit: Option<u64>) -> R<PartsVolume> {
        Self::build_with(gpu, state, res, limit, false)
    }

    /// `build`, with the fine bricks of thin parts (`refine`: the tracer reads them, `refine_views`) when asked.
    pub fn build_with(gpu: &Gpu, state: &mut State, res: u32, limit: Option<u64>, refine: bool) -> R<PartsVolume> {
        let t0 = Instant::now();
        let mut laps: Vec<(String, f64)> = Vec::new();
        Self::accepts(gpu, state)?;
        let mut scene = PartsScene::plan(state).map_err(|e| format!("refused: {e}"))?;
        scene.load_tubes(state).map_err(|e| format!("refused: {e}"))?;
        if limit.is_some_and(|l| Self::estimate(res, scene.pieces.len()) > l) { return Err(NOT_ENOUGH_MEMORY.into()); }
        let hb = state.scene_hb();
        let h = 2.0 * hb / res as f64;
        let cs = 2.0 * hb / 64.0;
        let band = scene.band(res, hb);
        let far_max = 6.0 * cs;
        let mut cells = Cells::new([64; 3], cs, [-hb; 3], band, far_max);
        let lap = |laps: &mut Vec<(String, f64)>, what: &str, t: &mut Instant| { laps.push((what.to_string(), t.elapsed().as_secs_f64() * 1e3)); *t = Instant::now(); };
        let mut t = Instant::now();
        // the text, once for the kinds the scene holds
        let pair = scene.pair.map(|p| (p.reach() as f32, p.power as f32));
        let offs = warp_offsets(&scene);
        let tube = scene.tube_text.as_deref();
        let common = format!("{HEAD}{ell}{kinds}{tubes}{warps}{pairs}{pd}",
            ell = gf_core::stretch::ELL_WGSL,
            kinds = gf_core::csg::parts_kinds_wgsl(&state.families, &scene.kinds),
            tubes = tube.map_or(String::new(), tube_text),
            warps = warp_text(&offs),
            pairs = pair_text(pair),
            pd = PIECE_DIST.replace("//TUBE", if tube.is_some() { TUBE_BODY } else { "    d = 1.0;" }));
        let c = wgpu::ShaderStages::COMPUTE;
        let bake = pipe(gpu, "parts bake", &format!("{common}{BAKE}"), &[storage_entry(0, c), storage_entry(1, c), storage_entry(2, c), storage_entry(3, c), uniform_entry(4, c, false), out_entry(5), storage_entry(7, c), storage_entry(8, c)]);
        let live_pipe = pipe(gpu, "parts live", &format!("{common}{LIVE}"), &[storage_entry(0, c), uniform_entry(4, c, false), out_entry(5), tex_entry(6), storage_entry(7, c), storage_entry(8, c)]);
        let far_pipe = pipe(gpu, "parts far", FAR, &[storage_entry(0, c), rw_entry(1), uniform_entry(2, c, false)]);
        lap(&mut laps, "compile the kinds' text (CPU)", &mut t);
        cells.bin_all(&scene.pieces, &scene.balls);
        lap(&mut laps, "bin every part into the cells (CPU, all threads)", &mut t);
        cells.flatten();
        let alloc = |label: &str, bytes: u64, usage: wgpu::BufferUsages| -> R<wgpu::Buffer> {
            gpu.allocate(label, bytes, || gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: bytes.max(16), usage, mapped_at_creation: false }))
                .map_err(|_| NOT_ENOUGH_MEMORY.to_string())
        };
        let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let n = cells.count();
        let cap_pieces = scene.pieces.len() + 64;
        let bufs = Bufs {
            pieces: alloc("parts pieces", (cap_pieces * GPU_FLOATS * 4) as u64, st)?,
            ranges: alloc("parts ranges", n as u64 * 8, st)?,
            items: alloc("parts items", cells.cap_items as u64 * 4, st)?,
            far: alloc("parts far", n as u64 * 4, st)?,
        };
        let tube_buf = alloc("parts tubes", (scene.tube_table.len().max(4) * 4) as u64, st)?;
        let warp_floats: usize = scene.warps.iter().map(|w| w.data.len()).sum();
        let warp_buf = alloc("parts warps", (warp_floats.max(4) * 4) as u64, st)?;
        let lbuf = alloc("parts live", (MAX_LIVE * GPU_FLOATS * 4) as u64, st)?;
        let ubuf = gpu.uniform_buffer("parts u", std::mem::size_of::<BU>() as u64);
        let far_ubuf = gpu.uniform_buffer("parts far u", 64);
        let dims = region_dims(res);
        let vols = (|| -> R<[Volume; 3]> { Ok([Volume::try_new(gpu, res, true)?, Volume::try_new(gpu, res, true)?, Volume::try_new(gpu, res, true)?]) })()
            .map_err(|e| if is_out_of_memory(&e) { NOT_ENOUGH_MEMORY.to_string() } else { e })?;
        let [vol, rest, shown] = vols;
        let bytes = 3 * dims.iter().map(|&x| x as u64).product::<u64>() * 2 + (cap_pieces * GPU_FLOATS * 4) as u64 + n as u64 * 12 + cells.cap_items as u64 * 4 + (scene.tube_table.len() * 4) as u64 + (warp_floats * 4) as u64;
        if limit.is_some_and(|l| bytes > l) { return Err(NOT_ENOUGH_MEMORY.into()); }
        lap(&mut laps, "make 3 volumes + buffers", &mut t);
        let mut me = PartsVolume { gpu: gpu.clone(), scene, cells, res, hb, h, band, far_max, bake, live_pipe, far_pipe, bufs, tube_buf, warp_buf, ubuf, far_ubuf, lbuf, cap_pieces, vol, rest, shown, live: Vec::new(), old_box: None, prev_box: None, bytes, common: common.clone(), refine: Refine { on: refine, ..Default::default() }, laps };
        me.write_static();
        me.write_pieces_all();
        me.write_cells_all();
        me.run_far();
        lap(&mut me.laps, "write buffers + bounds pass (GPU)", &mut t);
        let (vol_dims, _) = (dims, ());
        me.bake_box(0, [0; 3], vol_dims);
        me.gpu.wait();
        lap(&mut me.laps, "bake the whole volume + wait (GPU)", &mut t);
        if me.refine.on {
            me.refine_alloc()?;
            let gd = me.cells.gd;
            me.refine_cells([0; 3], [gd[0] - 1, gd[1] - 1, gd[2] - 1], true);
            lap(&mut me.laps, "refine: every cell's brick (flags, bake, page)", &mut t);
        }
        me.laps.push(("build total".into(), t0.elapsed().as_secs_f64() * 1e3));
        Ok(me)
    }

    /// The scene's volume (what the tracer reads at rest; while a drag is live, `shown`).
    pub fn volume(&self) -> &Volume { if self.live.is_empty() { &self.vol } else { &self.shown } }
    /// The scene's own volume (the last folded one).
    pub fn scene_volume(&self) -> &Volume { &self.vol }
    pub fn rest_volume(&self) -> &Volume { &self.rest }
    pub fn dragging(&self) -> bool { !self.live.is_empty() }
    pub fn band(&self) -> f64 { self.band }
    pub fn far_max(&self) -> f64 { self.far_max }

    /// The pieces the cell holding scene point `p` lists, in list order (for checks).
    pub fn cell_list_at(&self, p: [f64; 3]) -> Vec<u32> {
        let c = [0, 1, 2].map(|a| (((p[a] - self.cells.lo[a]) / self.cells.cs).floor().max(0.0) as usize).min(self.cells.gd[a] - 1));
        self.cells.lists[self.cells.index(c)].clone()
    }

    /// The bytes the build holds (3 volumes, the cells' buffers, the tables).
    pub fn memory(&self) -> u64 { self.bytes }

    // ------------------------------------------------------------------ writes

    fn write_static(&self) {
        if !self.scene.tube_table.is_empty() { self.gpu.queue.write_buffer(&self.tube_buf, 0, bytemuck::cast_slice(&self.scene.tube_table)); }
        self.write_warps();
    }

    fn write_warps(&self) {
        let all: Vec<f32> = self.scene.warps.iter().flat_map(|w| w.data.iter().copied()).collect();
        if !all.is_empty() { self.gpu.queue.write_buffer(&self.warp_buf, 0, bytemuck::cast_slice(&all)); }
    }

    fn write_pieces_all(&self) {
        let all: Vec<f32> = self.scene.recs.iter().flat_map(|r| r.iter().copied()).collect();
        if !all.is_empty() { self.gpu.queue.write_buffer(&self.bufs.pieces, 0, bytemuck::cast_slice(&all)); }
    }

    /// Piece `i` written to the GPU's table (`left_out`: op 0, left out of every bake: the dragged piece).
    fn write_piece(&self, i: usize, left_out: bool) {
        let mut r = self.scene.recs[i];
        if left_out { r[14] = 0.0; }
        self.gpu.queue.write_buffer(&self.bufs.pieces, (i * GPU_FLOATS * 4) as u64, bytemuck::cast_slice(&r));
    }

    fn write_cells_all(&self) {
        let c = &self.cells;
        self.gpu.queue.write_buffer(&self.bufs.ranges, 0, bytemuck::cast_slice(&c.ranges));
        self.gpu.queue.write_buffer(&self.bufs.items, 0, bytemuck::cast_slice(&c.pool[..c.pool_used.max(1)]));
    }

    /// What an edit changed in the cells written: the spans, or everything again when the pool is full.
    fn upload_edit(&mut self) -> R<()> {
        if self.scene.pieces.len() > self.cap_pieces { return self.regrow(); }
        let spans = self.cells.edit_spans();
        let c = &self.cells;
        match spans {
            Spans::Full => return self.regrow(),
            Spans::Edit { ranges, in_place, tail } => {
                if let Some((c0, c1)) = ranges { self.gpu.queue.write_buffer(&self.bufs.ranges, c0 as u64 * 8, bytemuck::cast_slice(&c.ranges[c0..=c1])); }
                if let Some((a, b)) = in_place { self.gpu.queue.write_buffer(&self.bufs.items, a as u64 * 4, bytemuck::cast_slice(&c.pool[a..b])); }
                if let Some((a, b)) = tail { self.gpu.queue.write_buffer(&self.bufs.items, a as u64 * 4, bytemuck::cast_slice(&c.pool[a..b])); }
            }
        }
        self.run_far();
        Ok(())
    }

    /// The pool or the piece table is full: the buffers made anew, everything written.
    fn regrow(&mut self) -> R<()> {
        let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let gpu = self.gpu.clone();
        let alloc = |label: &str, bytes: u64| gpu.allocate(label, bytes, || gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: bytes.max(16), usage: st, mapped_at_creation: false })).map_err(|_| NOT_ENOUGH_MEMORY.to_string());
        self.cap_pieces = self.scene.pieces.len() + 64;
        self.cells.flatten();
        self.bufs.pieces = alloc("parts pieces", (self.cap_pieces * GPU_FLOATS * 4) as u64)?;
        self.bufs.items = alloc("parts items", self.cells.cap_items as u64 * 4)?;
        self.write_pieces_all();
        self.write_cells_all();
        self.run_far();
        Ok(())
    }

    /// The bounds of the region the cells' changes reach, made on the GPU from the ranges buffer (written first): one thread a cell.
    fn run_far(&mut self) {
        let Some((lo, hi)) = self.cells.take_far() else { return };
        let cap = self.cells.far_cap();
        let ext = [0, 1, 2].map(|a| (hi[a] - lo[a] + 1) as u32);
        let gd = self.cells.gd;
        let u: [u32; 16] = [
            gd[0] as u32, gd[1] as u32, gd[2] as u32, 0,
            lo[0] as u32, lo[1] as u32, lo[2] as u32, 0,
            ext[0], ext[1], ext[2], 0,
            (self.band as f32).to_bits(), (self.far_max as f32).to_bits(), (self.cells.cs as f32).to_bits(), (cap as f32).to_bits(),
        ];
        self.gpu.queue.write_buffer(&self.far_ubuf, 0, bytemuck::cast_slice(&u));
        let (p, l) = &self.far_pipe;
        let g = self.gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("parts far"), layout: l, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: self.bufs.ranges.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: self.bufs.far.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: self.far_ubuf.as_entire_binding() },
        ] });
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("parts far"), timestamp_writes: None });
            cp.set_pipeline(p);
            cp.set_bind_group(0, &g, &[]);
            cp.dispatch_workgroups(ext[0].div_ceil(4), ext[1].div_ceil(4), ext[2].div_ceil(4));
        }
        self.gpu.queue.submit([enc.finish()]);
    }

    fn uniform(&self, org: [u32; 3], ext: [u32; 3], ball: [f64; 4], n: usize) { self.uniform4(org, [ext[0], ext[1], ext[2], 0], ball, n); }

    fn uniform4(&self, org: [u32; 3], ext: [u32; 4], ball: [f64; 4], n: usize) {
        let gd = self.cells.gd;
        let u = BU { h: self.h as f32, band: self.band as f32, cs: self.cells.cs as f32, pad: 0.0, lo: [-self.hb as f32, -self.hb as f32, -self.hb as f32, 0.0], gd: [gd[0] as u32, gd[1] as u32, gd[2] as u32, 0], org: [org[0], org[1], org[2], 0], ext, ball: [ball[0] as f32, ball[1] as f32, ball[2] as f32, ball[3] as f32], n: [n as u32, 0, 0, 0] };
        self.gpu.queue.write_buffer(&self.ubuf, 0, bytemuck::bytes_of(&u));
    }

    // ------------------------------------------------------------------ passes

    /// Build the box (org, ext) of volume `into` (0 the scene's, 1 the rest, 2 the shown) from the cells.
    fn bake_box(&self, into: usize, org: [u32; 3], ext: [u32; 3]) {
        if ext.contains(&0) { return; }
        let target = [&self.vol, &self.rest, &self.shown][into];
        let (p, l) = &self.bake;
        let gpu = &self.gpu;
        let g = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("parts bake"), layout: l, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: self.bufs.pieces.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: self.bufs.ranges.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: self.bufs.items.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: self.bufs.far.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: self.ubuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&target.view) },
            wgpu::BindGroupEntry { binding: 7, resource: self.tube_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: self.warp_buf.as_entire_binding() },
        ] });
        let slab = self.slab(ext);
        let mut z = 0;
        while z < ext[2] {
            let dz = slab.min(ext[2] - z);
            self.uniform([org[0], org[1], org[2] + z], [ext[0], ext[1], dz], [0.0; 4], 0);
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            {
                let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("parts bake"), timestamp_writes: None });
                cp.set_pipeline(p);
                cp.set_bind_group(0, &g, &[]);
                cp.dispatch_workgroups(ext[0].div_ceil(4), ext[1].div_ceil(4), dz.div_ceil(4));
            }
            gpu.queue.submit([enc.finish()]);
            // (the uniform is written before the submit it belongs to: queue writes happen in order with submits)
            if slab < ext[2] { gpu.wait(); }
            z += dz;
        }
    }

    /// How many z slices one submit covers: all of them, or with tubes (a dear formula per voxel) about 400 000 voxels.
    fn slab(&self, ext: [u32; 3]) -> u32 {
        if self.scene.tube_table.is_empty() { ext[2] } else { (4_000_000 / (ext[0] * ext[1]).max(1)).max(1) }
    }

    fn copy(&self, from: &Volume, to: &Volume) {
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_texture(from.tex.as_image_copy(), to.tex.as_image_copy(), wgpu::Extent3d { width: self.res, height: self.res, depth_or_array_layers: self.res });
        self.gpu.queue.submit([enc.finish()]);
    }

    // ------------------------------------------------------------------ regions

    /// The union of the live pieces' own bounding boxes.
    fn live_box(&self) -> ([f64; 3], [f64; 3]) {
        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
        for &i in &self.live {
            let (a, b) = self.scene.pieces[i].aabb();
            for k in 0..3 { lo[k] = lo[k].min(a[k]); hi[k] = hi[k].max(b[k]); }
        }
        (lo, hi)
    }

    /// The voxel box (origin, extent) a box of the scene grown by the band + `far` + 2 cells covers.
    fn region_box(&self, bx: ([f64; 3], [f64; 3]), far: f64) -> ([u32; 3], [u32; 3]) {
        let g = self.band + far + 2.0 * self.cells.cs;
        let n = self.res as f64;
        let v = |x: f64| ((x + self.hb) / self.h).floor().clamp(0.0, n);
        let o = [0, 1, 2].map(|a| v(bx.0[a] - g) as u32);
        let e = [0, 1, 2].map(|a| (v(bx.1[a] + g) as u32 + 1).min(self.res) - o[a]);
        (o, e)
    }

    /// The voxel box that pieces' balls + the band + the empty cells' bound can change.
    fn region_of(&self, balls: &[[f64; 4]], far: f64) -> ([u32; 3], [u32; 3]) {
        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
        for b in balls {
            let r = b[3] + self.band + far + 2.0 * self.cells.cs;
            for a in 0..3 { lo[a] = lo[a].min(b[a] - r); hi[a] = hi[a].max(b[a] + r); }
        }
        let n = self.res as f64;
        let v = |x: f64| ((x + self.hb) / self.h).floor().clamp(0.0, n);
        let o = [0, 1, 2].map(|a| v(lo[a]) as u32);
        let e = [0, 1, 2].map(|a| (v(hi[a]) as u32 + 1).min(self.res) - o[a]);
        (o, e)
    }

    fn union_box(a: ([u32; 3], [u32; 3]), b: ([u32; 3], [u32; 3])) -> ([u32; 3], [u32; 3]) {
        let o = [0, 1, 2].map(|k| a.0[k].min(b.0[k]));
        let e = [0, 1, 2].map(|k| (a.0[k] + a.1[k]).max(b.0[k] + b.1[k]) - o[k]);
        (o, e)
    }

    fn vox(e: [u32; 3]) -> u64 { e.iter().map(|&x| x as u64).product() }

    // ------------------------------------------------------------------ edits

    fn apply(&mut self, idx: &[usize], d: Vec<Derived>) {
        for (&i, d) in idx.iter().zip(d) {
            self.scene.pieces[i] = d.piece;
            self.scene.balls[i] = d.ball;
            self.scene.recs[i] = d.rec;
        }
    }

    /// The live pieces as they stand now (from the state) and their bounding ball.
    fn refresh_live(&mut self, state: &State) -> R<[f64; 4]> {
        let idx = self.live.clone();
        let d = self.scene.derive(state, &idx).map_err(|e| format!("refused: {e}"))?;
        self.apply(&idx, d);
        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
        for &i in &idx {
            let b = self.scene.balls[i];
            for a in 0..3 { lo[a] = lo[a].min(b[a] - b[3]); hi[a] = hi[a].max(b[a] + b[3]); }
        }
        let c = [0, 1, 2].map(|a| 0.5 * (lo[a] + hi[a]));
        let r = 0.5 * ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt();
        Ok([c[0], c[1], c[2], r])
    }

    /// A drag of these pieces starts: the rest = the volume with the pieces' box built again without them. The lists stay as they are (the
    /// pieces are flagged out of the GPU's table): only the voxels the pieces' cells reach are built again.
    pub fn start(&mut self, state: &State, idx: &[usize]) -> R<()> {
        if idx.is_empty() { return Err("no part to drag".into()); }
        if idx.iter().any(|&i| i >= self.scene.pieces.len()) { return Err("refused: no such part".into()); }
        if !self.live.is_empty() { self.fold(state)?; }
        let t0 = Instant::now();
        self.live = idx.to_vec();
        for &i in idx { self.write_piece(i, true); }
        let ob = self.live_box();
        self.old_box = Some(ob);
        let (o, e) = self.region_box(ob, 0.0);
        self.copy(&self.vol, &self.rest);
        self.bake_box(1, o, e);
        self.refine.moved = Some(idx.to_vec());
        self.refine_region(o, e, true);
        self.refine.moved = None;
        self.copy(&self.rest, &self.shown);
        self.prev_box = None;
        let ball = self.refresh_live(state)?;
        self.run_live_ball(ball);
        self.gpu.wait();
        self.laps.push(("drag start: copy + rest bake + first live pass + wait (GPU)".into(), t0.elapsed().as_secs_f64() * 1e3));
        Ok(())
    }

    /// One drag frame from the state as it stands: `shown` = the rest joined with the live parts over the box the live ball reaches now and
    /// did on the last frame.
    pub fn run_live(&mut self, state: &State) -> R<()> {
        if self.live.is_empty() { return Ok(()); }
        let t = Instant::now();
        let ball = self.refresh_live(state)?;
        self.run_live_ball(ball);
        self.laps.push(("live frame: CPU side (derive, list, uniform, submit)".into(), t.elapsed().as_secs_f64() * 1e3));
        Ok(())
    }

    fn run_live_ball(&mut self, ball: [f64; 4]) {
        let nb = self.live_box();
        let ub = match self.prev_box { Some(pb) => ([0, 1, 2].map(|k| pb.0[k].min(nb.0[k])), [0, 1, 2].map(|k| pb.1[k].max(nb.1[k]))), None => nb };
        let (org, ext) = self.region_box(ub, self.far_max);
        self.prev_box = Some(nb);
        // (the bricks under the live parts are hidden: further out the shown volume is the rest, which the bricks are)
        let (mo, me) = self.region_box(ub, 0.0);
        self.refine_mask(mo, me);
        // the live parts and, in list order among them, the holes cut later in the list that reach them (flagged 3: they cut the live
        // parts only: the rest was built with them, and they must not cut the parts built after them)
        let first = self.live.iter().copied().min().unwrap_or(0);
        let mut order: Vec<(usize, bool)> = self.live.iter().map(|&i| (i, true)).collect();
        for j in first + 1..self.scene.pieces.len() {
            if order.len() >= MAX_LIVE { break; }
            if self.live.contains(&j) { continue; }
            let (p, bj) = (&self.scene.pieces[j], self.scene.balls[j]);
            if !p.is_cut() || !self.live.iter().any(|&l| self.scene.owner[l].0 == self.scene.owner[j].0) { continue; }
            let d = ((bj[0] - ball[0]).powi(2) + (bj[1] - ball[1]).powi(2) + (bj[2] - ball[2]).powi(2)).sqrt();
            if d < bj[3] + ball[3] + self.band { order.push((j, false)); }
        }
        order.sort_by_key(|o| o.0);
        let lp: Vec<f32> = order.iter().flat_map(|&(i, live)| { let mut r = self.scene.recs[i]; if !live { r[14] = 3.0; } r }).collect();
        self.gpu.queue.write_buffer(&self.lbuf, 0, bytemuck::cast_slice(&lp));
        let (p, l) = &self.live_pipe;
        let gpu = &self.gpu;
        let g = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("parts live"), layout: l, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: self.lbuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: self.ubuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.shown.view) },
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.rest.view) },
            wgpu::BindGroupEntry { binding: 7, resource: self.tube_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: self.warp_buf.as_entire_binding() },
        ] });
        let slab = self.slab(ext);
        let mut z = 0;
        while z < ext[2] {
            let dz = slab.min(ext[2] - z);
            self.uniform([org[0], org[1], org[2] + z], [ext[0], ext[1], dz], ball, order.len());
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            {
                let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("parts live"), timestamp_writes: None });
                cp.set_pipeline(p);
                cp.set_bind_group(0, &g, &[]);
                cp.dispatch_workgroups(ext[0].div_ceil(4), ext[1].div_ceil(4), dz.div_ceil(4));
            }
            gpu.queue.submit([enc.finish()]);
            if slab < ext[2] { gpu.wait(); }
            z += dz;
        }
    }

    /// Release: the pieces listed by their new balls (only the cells they leave or enter change), their old and new boxes built again.
    pub fn fold(&mut self, state: &State) -> R<()> {
        if self.live.is_empty() { return Ok(()); }
        let t = Instant::now();
        let idx = self.live.clone();
        let old = self.old_box.unwrap_or_else(|| self.live_box());
        let (o0, e0) = self.region_box(old, self.far_max);
        let (q0, f0) = self.region_box(old, 0.0);
        self.refresh_live(state)?;
        let (o1, e1) = self.region_box(self.live_box(), self.far_max);
        let (q1, f1) = self.region_box(self.live_box(), 0.0);
        self.live.clear();
        self.relist(&idx)?;
        for &i in &idx { self.write_piece(i, false); }
        // (one box over the old and the new place: they overlap)
        let (uo, ue) = Self::union_box((o0, e0), (o1, e1));
        if Self::vox(ue) <= Self::vox(e0) + Self::vox(e1) {
            self.bake_box(0, uo, ue);
        } else {
            self.bake_box(0, o0, e0);
            self.bake_box(0, o1, e1);
        }
        self.refine_unmask();
        self.refine.moved = Some(idx.clone());
        self.refine_region(q0, f0, false);
        self.refine_region(q1, f1, true);
        self.refine_upload(); // (a box with no extent uploads nothing above: the unmasked page still goes up)
        self.refine.moved = None;
        self.gpu.wait();
        self.old_box = None;
        self.prev_box = None;
        self.laps.push(("fold: relist + rebake old and new box + wait".into(), t.elapsed().as_secs_f64() * 1e3));
        Ok(())
    }

    /// The pieces `idx` taken out of the lists where they were listed and put in by their balls now; the bounds near the change; the buffers written.
    fn relist(&mut self, idx: &[usize]) -> R<()> {
        self.cells.relist(&self.scene.pieces, &self.scene.balls, idx);
        self.upload_edit()
    }

    /// What the state asks of the held scene now, as a gesture: `draft` is a held handle or slider (a moving frame), else a settled build.
    /// A few pieces moved while held: the drag (`start`, then `run_live` each frame); the release (a settled build, or something else
    /// changed) folds it; anything else is `reshape`.
    pub fn drag_step(&mut self, state: &State, draft: bool) -> R<()> {
        let all: Vec<usize> = (0..self.scene.pieces.len()).collect();
        let fresh = self.scene.derive(state, &all).map_err(|e| format!("refused: {e}"))?;
        let changed: Vec<usize> = all.iter().copied().filter(|&i| fresh[i].rec != self.scene.recs[i] || fresh[i].ball != self.scene.balls[i]).collect();
        // (the live pass cuts the whole rest, which holds every object; the bake cuts an object's own parts only (its running value per
        // object). So a dragged cut, with several objects standing apart, is not drawn live: the box is baked, exactly, instead)
        let cuts_apart = self.scene.pair.is_none() && self.scene.objs_at.len() > 1 && changed.iter().any(|&i| (1.5..2.5).contains(&self.scene.pieces[i].a[2]));
        if !self.live.is_empty() {
            if draft && !cuts_apart && changed.iter().all(|i| self.live.contains(i)) { return self.run_live(state); }
            self.fold(state)?;
            return self.reshape(state);
        }
        if changed.is_empty() { return Ok(()); }
        if draft && !cuts_apart && changed.len() <= 16 { return self.start(state, &changed); }
        self.reshape(state)
    }

    /// A slider moved a shape's parameter (the state already holds it): the pieces made again, those that changed taken out of the lists
    /// where they were and put in where they are, and their old and new boxes built again. No drag of parts running.
    pub fn reshape(&mut self, state: &State) -> R<()> {
        if !self.live.is_empty() { return Err("a slider during a part drag is not done".into()); }
        let t = Instant::now();
        self.scene.refresh(state).map_err(|e| format!("refused: {e}"))?;
        self.write_warps();
        let all: Vec<usize> = (0..self.scene.pieces.len()).collect();
        let fresh = self.scene.derive(state, &all).map_err(|e| format!("refused: {e}"))?;
        let changed: Vec<usize> = all.iter().copied().filter(|&i| fresh[i].rec != self.scene.recs[i] || fresh[i].ball != self.scene.balls[i]).collect();
        if changed.is_empty() { return Ok(()); }
        let before: Vec<[f64; 4]> = changed.iter().map(|&i| self.scene.balls[i]).collect();
        let (o0, e0) = self.region_of(&before, self.far_max);
        let picks: Vec<Derived> = fresh.into_iter().enumerate().filter(|(i, _)| changed.contains(i)).map(|(_, d)| d).collect();
        self.apply(&changed, picks);
        let after: Vec<[f64; 4]> = changed.iter().map(|&i| self.scene.balls[i]).collect();
        let (o1, e1) = self.region_of(&after, self.far_max);
        self.relist(&changed)?;
        for &i in &changed { self.write_piece(i, false); }
        let (o, e) = Self::union_box((o0, e0), (o1, e1));
        self.bake_box(0, o, e);
        let mut balls = before.clone();
        balls.extend(after.iter().copied());
        let (ro, re) = self.region_of(&balls, 0.0);
        self.refine.moved = Some(changed.clone());
        self.refine_region(ro, re, true);
        self.refine.moved = None;
        self.gpu.wait();
        self.laps.push(("reshape: pieces, lists, old and new boxes + wait".into(), t.elapsed().as_secs_f64() * 1e3));
        Ok(())
    }

    /// A part added (the state's last): its piece and ball, the cells again, and its box built again. A scene of exact tubes, or one that
    /// the added part changes in any other way than by one more piece, is built again by the caller (`Err`).
    pub fn add(&mut self, state: &State) -> R<()> {
        if !self.live.is_empty() { return Err("a part added during a drag is not done".into()); }
        let t = Instant::now();
        let n = self.scene.pieces.len();
        let now = PartsScene::plan(state).map_err(|e| format!("refused: {e}"))?;
        if now.pieces.len() != n + 1 || now.kinds.len() < self.scene.kinds.len() || now.kinds[..self.scene.kinds.len()] != self.scene.kinds[..] || !now.tubes.iter().all(|t| t.is_none()) || now.pair != self.scene.pair || now.warps.len() != self.scene.warps.len() {
            return Err("the added part changes more than one piece: build again".into());
        }
        if now.kinds.len() != self.scene.kinds.len() { return Err("the added part is a new kind of shape: build again".into()); }
        if now.recs[..n] != self.scene.recs[..] { return Err("other parts changed too: build again".into()); }
        self.scene = now;
        let b = self.scene.balls[n];
        let bb = self.cells.list_insert(n, b, &self.scene.pieces[n]);
        if let Some((lo, hi)) = bb { self.cells.far_update(lo, hi); }
        self.write_piece_grown(n)?;
        self.upload_edit()?;
        let (o, e) = self.region_of(&[b], self.far_max);
        self.bake_box(0, o, e);
        let (ro, re) = self.region_of(&[b], 0.0);
        self.refine.moved = Some(vec![n]);
        self.refine_region(ro, re, true);
        self.refine.moved = None;
        self.gpu.wait();
        self.laps.push(("add: piece, cells, box + wait".into(), t.elapsed().as_secs_f64() * 1e3));
        Ok(())
    }

    /// Piece `i`'s record written, the table grown first when it is full.
    fn write_piece_grown(&mut self, i: usize) -> R<()> {
        if i >= self.cap_pieces {
            let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
            self.cap_pieces = self.scene.pieces.len() + 64;
            let gpu = self.gpu.clone();
            let bytes = (self.cap_pieces * GPU_FLOATS * 4) as u64;
            self.bufs.pieces = gpu.allocate("parts pieces", bytes, || gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("parts pieces"), size: bytes, usage: st, mapped_at_creation: false })).map_err(|_| NOT_ENOUGH_MEMORY.to_string())?;
            self.write_pieces_all();
        } else {
            self.write_piece(i, false);
        }
        Ok(())
    }
}

/// Where each warp's entries start in the one buffer, in vec4s.
fn warp_offsets(scene: &PartsScene) -> Vec<u32> {
    let mut at = 0u32;
    scene.warps.iter().map(|w| { let o = at; at += (w.data.len() / 4) as u32; o }).collect()
}

/// The tube kind's number, for a test to tell tubes from formulas.
pub const TUBE: f32 = TUBE_KIND;

// ---------------------------------------------------------------------- the fine bricks

impl PartsVolume {
    /// The page, the atlas and the numbers the tracer reads (None: no refinement).
    pub fn refine_views(&self) -> Option<(wgpu::TextureView, wgpu::TextureView, wgpu::Buffer)> {
        let r = &self.refine;
        if !r.on { return None; }
        Some((r.page_view.clone()?, r.atlas.as_ref()?.1.clone(), r.fr_ub.clone()?))
    }
    /// Bricks given out, and dropped for want of room (the budget).
    pub fn refine_stats(&self) -> (usize, u32) { (self.refine.page.iter().filter(|&&s| s != 0).count(), self.refine.dropped) }

    fn thin_pieces(&self) -> Vec<bool> {
        let limit = self.refine.thin * self.h;
        self.scene.pieces.iter().map(|o| o.thickness() < limit).collect()
    }

    /// Is this cell to be refined: the thinnest thin piece in its list that comes within the gate of the cell (its distance to the cell's
    /// middle, less the cell's half diagonal, under the gate + 1 voxel), or None.
    fn cell_thin(&self, ci: usize, c: [usize; 3], thin: &[bool]) -> Option<f64> {
        let mid = self.cells.mid(c);
        let reach = (self.refine.gate + 1.0) * self.h;
        let mut best: Option<f64> = None;
        for &i in &self.cells.lists[ci] {
            if !thin[i as usize] { continue; }
            let o = &self.scene.pieces[i as usize];
            if o.lower(mid) - 0.8660254 * self.cells.cs >= reach { continue; }
            let t = o.thickness();
            best = Some(best.map_or(t, |b: f64| b.min(t)));
        }
        best
    }

    /// The atlas, the page and the pipeline for the cells as they stand (once, after the lists are made).
    fn refine_alloc(&mut self) -> R<()> {
        if !self.refine.on { return Ok(()); }
        let gd = self.cells.gd;
        let ncells = gd[0] * gd[1] * gd[2];
        let thin = self.thin_pieces();
        let want = (0..ncells).filter(|&ci| !self.cells.lists[ci].is_empty() && self.cell_thin(ci, self.cells.coords(ci), &thin).is_some()).count() as u32;
        let n = ((self.cells.cs / self.h) * self.refine.f as f64).round().max(4.0) as u32;
        self.refine.nb = n + 2;
        let brick_bytes = (self.refine.nb as u64).pow(3) * 2;
        let budget = (self.refine.budget_mb * 1048576.0 / brick_bytes as f64) as u32;
        let cap = (want + want / 2 + 256).min(budget).min(65_534).max(1); // (the page holds slot + 1 as a u16)
        let side = (cap as f64).cbrt().ceil() as u32;
        let z = cap.div_ceil(side * side);
        self.refine.per = [side, side, z];
        self.refine.cap = side * side * z;
        let dims = [side * self.refine.nb, side * self.refine.nb, z * self.refine.nb];
        let gpu = self.gpu.clone();
        let bytes = dims.iter().map(|&d| d as u64).product::<u64>() * 2;
        let atlas = gpu.allocate("parts atlas", bytes, || gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("parts atlas"), size: wgpu::Extent3d { width: dims[0], height: dims[1], depth_or_array_layers: dims[2] }, mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D3, format: VOL_FORMAT, usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING, view_formats: &[],
        })).map_err(|_| NOT_ENOUGH_MEMORY.to_string())?;
        let view = atlas.create_view(&Default::default());
        self.refine.atlas = Some((atlas, view));
        self.bytes += bytes;
        self.refine.free = (0..self.refine.cap).rev().collect();
        self.refine.page = vec![0; ncells];
        self.refine.mask = vec![false; ncells];
        self.refine.masked.clear();
        self.refine.dropped = 0;
        self.refine.thin_cache.clear();
        let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("parts page"), size: wgpu::Extent3d { width: gd[0] as u32, height: gd[1] as u32, depth_or_array_layers: gd[2] as u32 }, mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D3, format: wgpu::TextureFormat::R32Float, usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, view_formats: &[],
        });
        self.refine.page_view = Some(tex.create_view(&Default::default()));
        self.refine.page_tex = Some(tex);
        let st = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        self.refine.recs = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor { label: Some("parts refine recs"), size: 4096 * 16, usage: st, mapped_at_creation: false }));
        let ub = gpu.uniform_buffer("parts refine u", 64);
        gpu.queue.write_buffer(&ub, 0, bytemuck::cast_slice(&self.refine_uniform()));
        self.refine.fr_ub = Some(ub);
        if self.refine.pipe.is_none() {
            let c = wgpu::ShaderStages::COMPUTE;
            self.refine.pipe = Some(pipe(&gpu, "parts refine", &format!("{}{REFINE_BAKE}", self.common), &[storage_entry(0, c), storage_entry(1, c), storage_entry(2, c), storage_entry(3, c), uniform_entry(4, c, false), out_entry(5), storage_entry(7, c), storage_entry(8, c), storage_entry(9, c)]));
        }
        Ok(())
    }

    /// The tracer's numbers for the bricks (cell grid, gate, brick geometry).
    fn refine_uniform(&self) -> [f32; 16] {
        let nb = self.refine.nb as f32;
        let dims = [0, 1, 2].map(|a| (self.refine.per[a] * self.refine.nb) as f32);
        let gd = self.cells.gd;
        [
            self.cells.lo[0] as f32, self.cells.lo[1] as f32, self.cells.lo[2] as f32, self.cells.cs as f32,
            (self.refine.gate * self.h) as f32, nb, self.refine.per[0] as f32, self.refine.per[1] as f32,
            (self.cells.cs / (nb as f64 - 2.0)) as f32, gd[0] as f32, gd[1] as f32, gd[2] as f32,
            dims[0], dims[1], dims[2], 0.0,
        ]
    }

    /// The cells (corner c0 to c1, cell coordinates) made again: bricks given to the cells that now hold a thin piece, taken from those that no
    /// longer do, baked from the exact formulas, the page written.
    fn refine_cells(&mut self, c0: [usize; 3], c1: [usize; 3], up: bool) {
        if !self.refine.on || self.refine.atlas.is_none() { return; }
        let gd = self.cells.gd;
        let thin = match (&self.refine.moved, self.refine.thin_cache.len() == self.scene.pieces.len()) {
            (Some(m), true) => {
                let limit = self.refine.thin * self.h;
                let mut t = std::mem::take(&mut self.refine.thin_cache);
                for &i in m { t[i] = self.scene.pieces[i].thickness() < limit; }
                t
            }
            _ => self.thin_pieces(),
        };
        let moved = self.refine.moved.clone();
        let mut recs: Vec<[u32; 4]> = Vec::new();
        let mut want: Vec<(f64, usize, [usize; 3])> = Vec::new();
        for z in c0[2]..=c1[2] { for y in c0[1]..=c1[1] { for x in c0[0]..=c1[0] {
            let ci = x + gd[0] * (y + gd[1] * z);
            let th = if self.cells.lists[ci].is_empty() { None } else { self.cell_thin(ci, [x, y, z], &thin) };
            let cur = self.refine.page[ci];
            match (th, cur) {
                (Some(_), c) if c != 0 => {
                    // (a brick already held is baked again only where a moved piece is in the list or the list changed)
                    let again = match (&moved, &self.cells.last_dirty) {
                        (Some(m), Some(d)) => d.binary_search(&ci).is_ok() || self.cells.lists[ci].iter().any(|i| m.contains(&(*i as usize))),
                        _ => true,
                    };
                    if again { recs.push([x as u32, y as u32, z as u32, (c - 1) as u32]); }
                }
                (Some(t), _) => want.push((t, ci, [x, y, z])),
                (None, c) if c != 0 => { self.refine.free.push((c - 1) as u32); self.refine.page[ci] = 0; }
                _ => {}
            }
        } } }
        // (the thinnest are served first when the budget is short)
        want.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (_, ci, c) in want {
            match self.refine.free.pop() {
                Some(slot) => { self.refine.page[ci] = (slot + 1) as u16; recs.push([c[0] as u32, c[1] as u32, c[2] as u32, slot]); }
                None => self.refine.dropped += 1,
            }
        }
        self.refine.thin_cache = thin;
        self.refine_bake(&recs);
        if up { self.refine_upload(); }
    }

    /// The cells a voxel box covers, made again.
    fn refine_region(&mut self, org: [u32; 3], ext: [u32; 3], up: bool) {
        if !self.refine.on || ext.contains(&0) { return; }
        let (c0, c1) = self.cells_of_box(org, ext);
        self.refine_cells(c0, c1, up);
    }

    fn cells_of_box(&self, org: [u32; 3], ext: [u32; 3]) -> ([usize; 3], [usize; 3]) {
        let gd = self.cells.gd;
        let cs = self.cells.cs;
        let c0 = [0, 1, 2].map(|a| (((org[a] as f64 * self.h) / cs).floor().max(0.0) as usize).min(gd[a] - 1));
        let c1 = [0, 1, 2].map(|a| ((((org[a] + ext[a]) as f64 * self.h - 1e-9) / cs).floor().max(0.0) as usize).min(gd[a] - 1));
        (c0, c1)
    }

    /// The cells a live pass covers are read from the coarse volume (a live part is over them): their bricks are hidden until the release.
    fn refine_mask(&mut self, org: [u32; 3], ext: [u32; 3]) {
        if !self.refine.on || self.refine.atlas.is_none() { return; }
        for ci in std::mem::take(&mut self.refine.masked) { self.refine.mask[ci] = false; }
        if !ext.contains(&0) {
            let (c0, c1) = self.cells_of_box(org, ext);
            let gd = self.cells.gd;
            for z in c0[2]..=c1[2] { for y in c0[1]..=c1[1] { for x in c0[0]..=c1[0] {
                let ci = x + gd[0] * (y + gd[1] * z);
                if self.refine.page[ci] != 0 { self.refine.mask[ci] = true; self.refine.masked.push(ci); }
            } } }
        }
        self.refine_upload();
    }

    fn refine_unmask(&mut self) {
        if !self.refine.on || self.refine.atlas.is_none() { return; }
        for ci in std::mem::take(&mut self.refine.masked) { self.refine.mask[ci] = false; }
    }

    fn refine_upload(&self) {
        let (Some(tex), true) = (self.refine.page_tex.as_ref(), self.refine.on) else { return };
        let gd = self.cells.gd;
        let data: Vec<f32> = self.refine.page.iter().zip(&self.refine.mask).map(|(&s, &m)| if m { 0.0 } else { s as f32 }).collect();
        self.gpu.queue.write_texture(tex.as_image_copy(), bytemuck::cast_slice(&data), wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * gd[0] as u32), rows_per_image: Some(gd[1] as u32) }, wgpu::Extent3d { width: gd[0] as u32, height: gd[1] as u32, depth_or_array_layers: gd[2] as u32 });
    }

    /// The bricks of these records baked from the cells' lists.
    fn refine_bake(&self, recs: &[[u32; 4]]) {
        if recs.is_empty() { return; }
        let (Some((p, l)), Some(buf), Some((_, atlas))) = (self.refine.pipe.as_ref(), self.refine.recs.as_ref(), self.refine.atlas.as_ref()) else { return };
        let gpu = &self.gpu;
        let g = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("parts refine"), layout: l, entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: self.bufs.pieces.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: self.bufs.ranges.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: self.bufs.items.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: self.bufs.far.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: self.ubuf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(atlas) },
            wgpu::BindGroupEntry { binding: 7, resource: self.tube_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 8, resource: self.warp_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 9, resource: buf.as_entire_binding() },
        ] });
        let nb = self.refine.nb;
        for chunk in recs.chunks(4096) {
            gpu.queue.write_buffer(buf, 0, bytemuck::cast_slice(chunk));
            self.uniform4([0; 3], [nb, chunk.len() as u32, self.refine.per[0], self.refine.per[1]], [0.0; 4], 0);
            let mut enc = gpu.device.create_command_encoder(&Default::default());
            {
                let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("parts refine"), timestamp_writes: None });
                cp.set_pipeline(p);
                cp.set_bind_group(0, &g, &[]);
                cp.dispatch_workgroups(nb.div_ceil(4), nb.div_ceil(4), (nb * chunk.len() as u32).div_ceil(4));
            }
            gpu.queue.submit([enc.finish()]);
            gpu.wait();
        }
    }
}
