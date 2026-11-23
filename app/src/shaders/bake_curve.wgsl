// Turning a curve (points with a tube radius) into a distance field.
// (COARSE_FS, FAR_FS, SPLAT_VS/FS, MELT_FS, UNMELT_FS of the web version's forms.js: the same arithmetic,
// arranged so that the GPU is handed a few large pieces of work instead of thousands of small ones.)
//
//   1. coarse    distance to a thinned-out cloud of the curve's points, at low resolution: the far field
//   2. gather    every segment drawn into just the slices it can reach, a slab of slices at a time, the
//                slices of the slab laid side by side on one sheet
//   3. resolve   sheet + far field -> the volume: a whole slab in one go by a compute shader where the
//                volume can be written that way, otherwise one slice per pass
//
// Texel row j of the volume is y = ((j + 0.5) / res * 2 - 1) * hb.

struct B {
  hb: f32,                // the volume spans [-hb, hb]
  res: f32,               // voxels per side of the volume being written
  band: f32,              // how far from its own surface a segment writes
  rmax: f32,              // the thickest radius on the curve
  k: f32,                 // melt: 1 / how close strands flow together
  amp: f32,               // melt: makes one straight strand sum to 1 on its own surface
  safety: f32,            // far field: how much the coarse value may be wrong by
  floor: f32,             // far field: never below this
  cres: f32,              // voxels per side of the coarse volume
  cloud_n: u32,           // points in the thinned-out cloud
  slab: u32,              // slices gathered at a time
  tiles: u32,             // slices across the sheet the coarse volume is worked out on
  stiles: u32,            // slices across the sheet a slab is gathered on
  melted: u32,            // 1 = strands flow together, 0 = hard union
  vary: u32,              // 1 = every point has a radius of its own (a tree): see below
  ds: f32,                // the step the curve is cut into
  cut: f32,               // nothing below this height is kept (a tree cut flat at the ground); -1e9 = none
  pad2: u32,
  pad3: u32,
  pad4: u32,
}

@group(0) @binding(0) var<uniform> b: B;

struct Full {
  @builtin(position) pos: vec4f,
  @location(0) @interpolate(flat) inst: u32,
}
@vertex
fn vs_full(@builtin(vertex_index) vi: u32, @builtin(instance_index) inst: u32) -> Full {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  var o: Full;
  o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  o.inst = inst;
  return o;
}

// ---- 1. far field: distance to a thinned-out cloud of the curve's points, brute force, at low resolution.
// Every slice is drawn side by side on one sheet in a single pass (and copied into the coarse volume
// afterwards): each voxel's loop is long, and drawn a thin slice at a time the GPU would mostly be waiting.
@group(1) @binding(0) var<storage, read> cloud: array<vec4f>;

@fragment
fn fs_coarse(in: Full) -> @location(0) vec4f {
  let cres = u32(b.cres);
  let px = vec2u(in.pos.xy);
  let tile = px / cres;
  let slice = tile.x + tile.y * b.tiles;
  let zc = (f32(slice) + 0.5) / b.cres;
  let p = (vec3f((in.pos.xy - vec2f(tile * cres)) / b.cres, zc) * 2.0 - 1.0) * b.hb;
  if (b.vary == 1u) {
    // each point of the cloud carries the thickest radius near it
    var d = 1e9;
    for (var i = 0u; i < 4096u; i++) {
      if (i >= b.cloud_n) { break; }
      let c = cloud[i];
      d = min(d, length(p - c.xyz) - c.w);
    }
    return vec4f(d, 0.0, 0.0, 1.0);
  }
  var d2 = 1e18;
  for (var i = 0u; i < 4096u; i++) {
    if (i >= b.cloud_n) { break; }
    let q = p - cloud[i].xyz;
    d2 = min(d2, dot(q, q));
  }
  return vec4f(sqrt(d2) - b.rmax, 0.0, 0.0, 1.0);
}

// ---- 2. near field: each segment is drawn as a small rectangle into just the slices it can reach.
// This is what lets the volume be large and the curve finely cut: the work goes with the curve's length,
// not with (voxels x segments). One instance = one segment in one slice.
@group(1) @binding(4) var<storage, read> pts: array<vec4f>;      // x, y, z, radius

struct Seg {
  @builtin(position) pos: vec4f,
  @location(0) @interpolate(flat) a: vec4f,
  @location(1) @interpolate(flat) c: vec4f,
  @location(2) @interpolate(flat) origin: vec2f,      // top left of this slice's place on the sheet, in pixels
  @location(3) @interpolate(flat) zc: f32,            // the slice's depth, 0..1
  @location(4) @interpolate(flat) s: vec4f,           // a blade's third point (see below)
  @location(5) @interpolate(flat) seg: u32,           // where the segment's first point is (a piece's terms follow it)
}
@vertex
fn vs_splat(@builtin(vertex_index) vi: u32, @location(0) seg: u32, @location(1) slice: u32) -> Seg {
  let a = pts[seg];
  let c = pts[seg + 1u];
  // (a blade's base has its half-thickness, negated, and its tip its half-width: it reaches its width,
  // a fifth more for its cup, and its thickness)
  var reach = max(a.w, c.w) + b.band;
  if (a.w < 0.0) { reach = 1.2 * c.w - a.w + b.band; }
  // the rectangle the segment can reach, kept inside the slice so that it cannot spill onto its neighbours
  var lo = max(min(a.xy, c.xy) - reach, vec2f(-b.hb));
  var hi = min(max(a.xy, c.xy) + reach, vec2f(b.hb));
  // (a solid piece: its box, a the middle and c the half-size)
  if (a.w < PIECE_BELOW) {
    lo = max(a.xy - c.xy - b.band, vec2f(-b.hb));
    hi = min(a.xy + c.xy + b.band, vec2f(b.hb));
  }
  let corner = vec2f(f32(vi & 1u), f32(vi >> 1u));
  let uv = mix(lo, hi, corner) / b.hb * 0.5 + 0.5;          // 0..1 across the slice, rows from the top
  let t = slice % b.slab;
  let tile = vec2f(f32(t % b.stiles), f32(t / b.stiles));
  let at = (tile + uv) / f32(b.stiles);                     // 0..1 across the sheet
  var o: Seg;
  o.pos = vec4f(at.x * 2.0 - 1.0, 1.0 - at.y * 2.0, 0.0, 1.0);
  if (any(lo >= hi)) { o.pos = vec4f(2.0, 2.0, 0.0, 1.0); }    // wholly outside the volume: draw nothing
  o.a = a;
  o.c = c;
  o.s = vec4f(0.0);
  if (a.w < 0.0 && a.w > PIECE_BELOW) { o.s = pts[seg + 2u]; }
  o.seg = seg;
  o.origin = tile * b.res;
  o.zc = (f32(slice) + 0.5) / b.res;
  return o;
}
// A blade (a leaf): a flat outline round a midrib from a (the base) to c (the tip), thickened by its
// half-thickness. In the blade's own frame u runs along the midrib (0..1), v across it, w through it.
// The outline's half-width at u is c.w * u^0.5 (1 - u) / 0.385: rounded at the base, widest a third of the
// way, coming to a point at the tip. It is cupped: w is taken from a surface that rises as v^2.
// Not an exact distance off the outline's edge (the edge's slope is allowed for, to first order), but
// never more than one: the march stays safe.
fn blade_dist(p: vec3f, a: vec4f, c: vec4f, s: vec4f) -> f32 {
  let ax = c.xyz - a.xyz;
  let L = max(length(ax), 1e-6);
  let x = ax / L;
  let y = normalize(s.xyz - a.xyz - x * dot(s.xyz - a.xyz, x));
  let z = cross(x, y);
  let q = p - a.xyz;
  let W = c.w;
  let u = dot(q, x) / L;
  let v = dot(q, y);
  let w = dot(q, z) - s.w * v * v / max(W, 1e-6);
  let uc = clamp(u, 0.02, 1.0);
  let k = W / 0.385;
  let hw = k * sqrt(uc) * (1.0 - uc);
  let slope = k * (0.5 / sqrt(uc) * (1.0 - uc) - sqrt(uc)) / L;
  let side = (abs(v) - hw) / sqrt(1.0 + slope * slope);
  let ends = max(-u, u - 1.0) * L;
  let d2 = max(side, ends);
  let e = vec2f(d2, abs(w));
  return length(max(e, vec2f(0.0))) + min(max(e.x, e.y), 0.0) + a.w;
}

// A solid piece (`curve::add_pieces`): the intersection of its terms, which follow its first two points.
// Each term is a tag (kind, numbers) and points of its own. Not an exact distance everywhere (the
// intersection of exact distances is a lower bound outside edges and corners), but never more than one.
const PIECE_MARK: f32 = -1.0e6;
const PIECE_BELOW: f32 = -5.0e5;
fn hex_dist(q: vec3f, r: f32, h: f32) -> f32 {
  // q: across (x towards a flat side, y), along the axis (z); iq's hexagonal prism
  let k = vec3f(-0.8660254, 0.5, 0.57735);
  var p = abs(vec3f(q.y, q.x, q.z));
  p = vec3f(p.xy - 2.0 * min(dot(k.xy, p.xy), 0.0) * k.xy, p.z);
  let d = vec2f(length(p.xy - vec2f(clamp(p.x, -k.z * r, k.z * r), r)) * sign(p.y - r), p.z - h);
  return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
}
// A stretch of a profile swept along a path (`path::sweep`): a rounded rectangle carried from a to c, its
// sizes and its turn about the stretch changing evenly between them, cut at either end by the plane
// halfway between this stretch's heading and its neighbour's, so that neighbours meet exactly.
fn sweep_dist(p: vec3f, round: f32, i: u32, ne: u32) -> f32 {
  let a = pts[i + 1u];
  let c = pts[i + 2u];
  let sa = pts[i + 3u];
  let sc = pts[i + 4u];
  let n0 = pts[i + 5u].xyz;
  let n1 = pts[i + 6u].xyz;
  let ax = c.xyz - a.xyz;
  let L = max(length(ax), 1e-7);
  let x = ax / L;
  let v = p - a.xyz;
  let u = dot(v, x);
  let h = clamp(u / L, 0.0, 1.0);
  var side = mix(sa.xyz, sc.xyz, h);
  side = side - x * dot(side, x);
  side = side / max(length(side), 1e-7);
  let up = cross(x, side);
  let w = v - x * u;
  let hs = vec2f(mix(a.w, c.w, h), mix(sa.w, sc.w, h));
  let r = min(round, min(hs.x, hs.y));
  let q = abs(vec2f(dot(w, side), dot(w, up))) - (hs - r);
  var flat = length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - r;
  if (ne > 0u) {
    // an outline instead (`path::Profile::outline`): its edges in a square from -1 to 1 that the half-sizes
    // stretch, filled by the even-odd rule; the distance found there scaled back by the smaller half-size
    // (a little under the true one where the two differ: never over it)
    let x = vec2f(dot(w, side), dot(w, up)) / max(hs, vec2f(1e-6));
    var d2 = 1e18;
    var inside = false;
    for (var e = 0u; e < ne; e++) {
      let s = pts[i + 7u + e];
      let a0 = s.xy;
      let ab = s.zw - s.xy;
      let hh = clamp(dot(x - a0, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
      let ww = x - a0 - ab * hh;
      d2 = min(d2, dot(ww, ww));
      if ((a0.y > x.y) != (s.w > x.y)) {
        if (x.x < ab.x * (x.y - a0.y) / ab.y + a0.x) { inside = !inside; }
      }
    }
    flat = sqrt(d2) * min(hs.x, hs.y);
    if (inside) { flat = -flat; }
  }
  // Outside its two cuts, the distance to the stretch's end. Inside them, the depth under its surface; a cut
  // is a face only where it is an open end of the path (between two stretches it only shares the space out:
  // measured to it, the solid would read nearly flush with its surface all through, and the surface's slope
  // would be lost in that).
  let c0 = -dot(p - a.xyz, n0);
  let c1 = dot(p - c.xyz, n1);
  let cut = max(c0, c1);
  let end0 = pts[i + 5u].w;
  let end1 = pts[i + 6u].w;
  if (cut > 0.0) {
    // (a rounded end: the section's core, rounded over by the rounding, which is a ball's half on a tube)
    let end = select(end1, end0, c0 >= c1);
    if (end > 1.5 && ne == 0u) { return length(vec2f(max(flat + r, 0.0), cut)) - r; }
    return length(vec2f(max(flat, 0.0), cut));
  }
  var inner = flat;
  let flat_end = select(1.5, 2.5, ne > 0u);
  if (end0 > 0.5 && end0 < flat_end) { inner = max(inner, c0); }
  if (end1 > 0.5 && end1 < flat_end) { inner = max(inner, c1); }
  return inner;
}
//#path
fn piece_dist(p: vec3f, seg: u32, mark: f32) -> f32 {
  var i = seg + 2u;
  let end = i + u32(PIECE_MARK - mark + 0.5);
  var d = -1e9;
  for (var guard = 0u; guard < 64u; guard++) {
    if (i >= end) { break; }
    let t = pts[i];
    let q = pts[i + 1u];
    let kind = u32(t.x + 0.5);
    if (kind == 0u) { d = max(d, dot(q.xyz, p) - q.w); i += 2u; }
    else if (kind == 1u) { d = max(d, length(p - q.xyz) - q.w); i += 2u; }
    else if (kind == 2u) { d = max(d, q.w - length(p - q.xyz)); i += 2u; }
    else if (kind == 3u) { d = max(d, abs(length(p - q.xyz) - q.w) - t.y); i += 2u; }
    else if (kind == 4u) { d = max(d, abs(dot(q.xyz, p) - q.w) - t.y); i += 2u; }
    else if (kind == 5u) {
      let ax = pts[i + 2u].xyz;
      let u = pts[i + 3u].xyz;
      let v = p - q.xyz;
      d = max(d, hex_dist(vec3f(dot(v, u), dot(v, cross(ax, u)), dot(v, ax)), t.y, t.z));
      i += 4u;
    } else if (kind == 9u) {
      let ne = u32(t.y + 0.5);
      d = max(d, pathPrim(p, pts[i + 1u], pts[i + 2u], pts[i + 3u], pts[i + 4u], pts[i + 5u], pts[i + 6u], vec4f(0.0), i + 7u, ne, 1e9));
      i += 7u + ne;
    } else if (kind == 7u || kind == 8u) {
      d = max(d, sweep_dist(p, t.y, i, u32(t.z + 0.5) * u32(kind == 8u)));
      i += 7u + u32(t.z + 0.5) * u32(kind == 8u);
    } else {
      // an outline, filled by the even-odd rule, extruded and rounded
      let ux = pts[i + 2u].xyz;
      let uy = pts[i + 3u].xyz;
      let v = p - q.xyz;
      let x = vec2f(dot(v, ux), dot(v, uy)) / q.w;
      let ne = u32(t.y + 0.5);
      let band = pts[i + 4u];
      var d2 = 1e18;
      var inside = false;
      for (var e = 0u; e < ne; e++) {
        let s = pts[i + 5u + e];
        let a0 = s.xy;
        let a1 = s.zw;
        let ab = a1 - a0;
        let h = clamp(dot(x - a0, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
        let w = x - a0 - ab * h;
        d2 = min(d2, dot(w, w));
        if ((a0.y > x.y) != (a1.y > x.y)) {
          if (x.x < ab.x * (x.y - a0.y) / ab.y + a0.x) { inside = !inside; }
        }
      }
      // (a band of a larger outline: only its own points, and no distance told past its margin, beyond
      // which edges were left out)
      if (band.z > 0.0) { d2 = min(d2, band.z * band.z); }
      var flat = sqrt(d2) * q.w;
      if (inside) { flat = -flat; }
      if (band.z > 0.0 && (x.y < band.x || x.y > band.y)) { flat = 1e3; }
      let z = abs(dot(v, cross(ux, uy))) - t.z;
      let e2 = vec2f(flat + t.w, z + t.w);
      d = max(d, length(max(e2, vec2f(0.0))) + min(max(e2.x, e2.y), 0.0) - t.w);
      i += 5u + ne;
    }
  }
  return d;
}

// distance outside this segment's own surface, and the segment's radius there (a blade: -1)
fn seg_dist_r(in: Seg) -> vec2f {
  let p = (vec3f((in.pos.xy - in.origin) / b.res, in.zc) * 2.0 - 1.0) * b.hb;
  if (in.a.w < PIECE_BELOW) { return vec2f(piece_dist(p, in.seg, in.a.w), -1.0); }
  if (in.a.w < 0.0) { return vec2f(blade_dist(p, in.a, in.c, in.s), -1.0); }
  let pa = p - in.a.xyz;
  let ba = in.c.xyz - in.a.xyz;
  let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-12), 0.0, 1.0);
  let r = mix(in.a.w, in.c.w, h);
  return vec2f(length(pa - ba * h) - r, r);
}
fn seg_dist(in: Seg) -> f32 {
  return seg_dist_r(in).x;
}
// hard union: the exact distance, keeping the minimum
@fragment
fn fs_splat(in: Seg) -> @location(0) vec4f {
  return vec4f(seg_dist(in), 0.0, 0.0, 1.0);
}
// The melt's weight for a strand of radius r: 1 / what a straight strand of that radius, cut into steps
// of ds, sums to on its own surface. The sum is 1 + I/ds, I the integral of exp(-k (sqrt(r^2 + x^2) - r))
// along the strand, which is 2 r e^(kr) K1(kr); the expression below is that to within a third of a
// percent, which leaves the surface within a twentieth of a voxel of the radius.
fn melt_amp(r: f32) -> f32 {
  let z = b.k * r;
  let i = 2.0 / b.k * sqrt(1.0 + 1.5707964 * z + 0.19 * z / (z + 0.8));
  return 1.0 / (1.0 + i / b.ds);
}
// Melted union. A hard union of strands that run a hair apart leaves razor-shallow creases, which tear
// every reflection. Instead each segment adds exp(k (r - d)), faded to nothing at the edge of its reach,
// and the sum is turned back into a distance: -log(sum)/k. Strands closer than about 1/k flow together.
@fragment
fn fs_melt(in: Seg) -> @location(0) vec4f {
  let xr = seg_dist_r(in);
  let x = xr.x;
  // (radii of their own: each segment is weighted for its own radius)
  var amp = b.amp;
  if (b.vary == 1u) { amp = melt_amp(xr.y); }
  // (a blade is one piece, with no neighbours along it to add to it)
  if (xr.y < 0.0) { amp = 1.0; }
  // amp makes one straight strand sum to exactly 1 on its own surface (its neighbours along the strand
  // all add to it). The fade to zero sits beyond anywhere a merged surface can be, so it shapes nothing.
  let w = clamp((b.band - x) / (0.35 * b.band), 0.0, 1.0);
  return vec4f(amp * exp(min(-b.k * x, 8.0)) * w * w * (3.0 - 2.0 * w), 0.0, 0.0, 1.0);
}

// ---- 3. the volume: the far field (the coarse value, lowered by more than it can be wrong, never below
// the edge of the exact band) joined with what was gathered.
@group(1) @binding(1) var coarseTex: texture_3d<f32>;
@group(1) @binding(2) var lin: sampler;
@group(1) @binding(3) var sheetTex: texture_2d<f32>;

fn resolve(px: vec2f, slice: u32) -> vec4f {
  let zc = (f32(slice) + 0.5) / b.res;
  let far = max(textureSampleLevel(coarseTex, lin, vec3f(px / b.res, zc), 0.0).r - b.safety, b.floor);
  let t = slice % b.slab;
  let origin = vec2i(i32(t % b.stiles), i32(t / b.stiles)) * i32(b.res);
  let s = textureLoad(sheetTex, origin + vec2i(px), 0).r;
  // (the flat cut: the height below which nothing is kept, as a distance)
  let below = b.cut - (px.y / b.res * 2.0 - 1.0) * b.hb;
  if (b.melted == 0u) { return vec4f(max(min(far, s), below), 0.0, 0.0, 1.0); }
  var d = 1e9;
  if (s > 1e-9) { d = -log(s) / b.k; }
  return vec4f(clamp(max(min(far, d), below), -1.0, 4.0), 0.0, 0.0, 1.0);
}
// drawn: one slice per pass, the slice number arriving as the instance number
@fragment
fn fs_resolve(in: Full) -> @location(0) vec4f {
  return resolve(in.pos.xy, in.inst);
}

//#compute  (everything below is left out where volumes cannot be written by compute shaders)
struct Slab {
  z0: u32,                // the first slice of the slab being written
  n: u32,                 // how many slices it has
  pad0: u32,
  pad1: u32,
}
@group(2) @binding(0) var<uniform> sb: Slab;
@group(1) @binding(5) var outVol: texture_storage_3d<r16float, write>;

// computed: a whole slab of slices at once
@compute @workgroup_size(8, 8, 4)
fn cs_resolve(@builtin(global_invocation_id) id: vec3u) {
  if (id.z >= sb.n) { return; }
  let slice = sb.z0 + id.z;
  textureStore(outVol, vec3u(id.xy, slice), resolve(vec2f(id.xy) + 0.5, slice));
}
