// The surface studio, second part: a field whose zero is the surface made a true distance again, and
// pieces left floating free found and removed.
//
// Distance: every voxel beside the surface (a neighbour on the other side of zero) is a seed; jump
// flooding, in a band a few voxels wide either side of the surface, gives every voxel there its nearest
// seed; and the distance is to the point of the surface that seed stands for (found from the field and
// its slope at the seed): to the surface's tangent plane there, or past an edge to the point itself.
// Beyond the band the field over the most it can slope is used: never more than the true distance.
//
// Pieces: the solid is found as connected pieces on a grid of half the size (union-find, after Playne
// and Hawick), each piece's extent measured, and every piece narrower than asked for (and smaller than
// the largest) turned to empty space.

const NONE: u32 = 0xffffffffu;

struct R {
  res: u32,
  step: i32,
  cres: u32,
  thresh: u32,
  h: f32,
  pre: f32,       // added to the field read
  post: f32,      // added to the distance written
  reach: f32,     // how wide the band is either side (scene units)
  slope: f32,     // the most the field read can slope
  hb: f32,
  pad0: f32,
  pad1: f32,
};
@group(0) @binding(0) var<uniform> r: R;
@group(0) @binding(1) var fld: texture_3d<f32>;
@group(0) @binding(2) var seeds: texture_storage_3d<r32uint, read_write>;
// (the flooding reads one set of seeds and writes the other: never one that others are reading)
@group(0) @binding(4) var seedsOut: texture_storage_3d<r32uint, write>;
@group(0) @binding(3) var outv: texture_storage_3d<r16float, write>;

fn F(v: vec3i) -> f32 {
  let n = i32(r.res) - 1;
  return textureLoad(fld, clamp(v, vec3i(0), vec3i(n)), 0).r + r.pre;
}
fn pack(v: vec3i) -> u32 {
  return u32(v.x) + r.res * (u32(v.y) + r.res * u32(v.z));
}
fn unpack(s: u32) -> vec3i {
  let res = max(r.res, 1u);
  let q = s / res;
  return vec3i(i32(s % res), i32(q % res), i32(q / res));
}
fn pos(v: vec3i) -> vec3f {
  return ((vec3f(v) + 0.5) / f32(r.res) * 2.0 - 1.0) * r.hb;
}
// the field's slope at a voxel (scene units)
fn grad(v: vec3i) -> vec3f {
  return vec3f(F(v + vec3i(1, 0, 0)) - F(v - vec3i(1, 0, 0)),
               F(v + vec3i(0, 1, 0)) - F(v - vec3i(0, 1, 0)),
               F(v + vec3i(0, 0, 1)) - F(v - vec3i(0, 0, 1))) / (2.0 * r.h);
}

@compute @workgroup_size(8, 8, 4)
fn cs_seed(@builtin(global_invocation_id) gid: vec3u) {
  let n = i32(r.res);
  let v = vec3i(gid);
  if (any(v >= vec3i(n))) { return; }
  let f = F(v);
  var s = NONE;
  // (a voxel whose value is this far from zero cannot have a neighbour across it)
  if (abs(f) <= 4.0 * abs(r.slope) * r.h) {
    let inside = f < 0.0;
    let ds = array<vec3i, 6>(vec3i(1, 0, 0), vec3i(-1, 0, 0), vec3i(0, 1, 0), vec3i(0, -1, 0), vec3i(0, 0, 1), vec3i(0, 0, -1));
    for (var k = 0; k < 6; k++) {
      let w = v + ds[k];
      if (any(w < vec3i(0)) || any(w >= vec3i(n))) { continue; }
      if ((F(w) < 0.0) != inside) { s = pack(v); break; }
    }
  }
  textureStore(seedsOut, v, vec4u(s, 0u, 0u, 0u));
}

@compute @workgroup_size(8, 8, 4)
fn cs_jfa(@builtin(global_invocation_id) gid: vec3u) {
  let n = i32(r.res);
  let v = vec3i(gid);
  if (any(v >= vec3i(n))) { return; }
  // outside the band nothing is wanted of the seeds
  var best = textureLoad(seeds, v).x;
  if (abs(F(v)) > r.reach * abs(r.slope)) {
    textureStore(seedsOut, v, vec4u(best, 0u, 0u, 0u));
    return;
  }
  var bd = 1e30;
  if (best != NONE) { let d = vec3f(unpack(best) - v); bd = dot(d, d); }
  let k = r.step;
  for (var z = -1; z <= 1; z++) {
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        if (x == 0 && y == 0 && z == 0) { continue; }
        let w = v + k * vec3i(x, y, z);
        if (any(w < vec3i(0)) || any(w >= vec3i(n))) { continue; }
        let s = textureLoad(seeds, w).x;
        if (s == NONE) { continue; }
        let d = vec3f(unpack(s) - v);
        let dd = dot(d, d);
        if (dd < bd) { bd = dd; best = s; }
      }
    }
  }
  textureStore(seedsOut, v, vec4u(best, 0u, 0u, 0u));
}

@compute @workgroup_size(8, 8, 4)
fn cs_resolve(@builtin(global_invocation_id) gid: vec3u) {
  let n = i32(r.res);
  let v = vec3i(gid);
  if (any(v >= vec3i(n))) { return; }
  let f = F(v);
  var d = 1e9;
  let s = textureLoad(seeds, v).x;
  if (abs(f) <= r.reach * abs(r.slope) && s != NONE) {
    // The nearest seed and the seeds about it (every voxel of the surface within one of it): for each,
    // the point of the surface it stands for (one Newton step from it) and the surface's direction
    // there. The distance to each is to its tangent plane while the voxel lies over that point, and
    // grows smoothly as the voxel lies further off to the side (past an edge); the least of them is taken.
    // Exact for a flat piece, round about an edge, and the same for neighbouring voxels.
    let sv = unpack(s);
    let n = i32(r.res);
    let p = pos(v);
    for (var z = -1; z <= 1; z++) {
      for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
          let c = sv + vec3i(x, y, z);
          if (any(c < vec3i(0)) || any(c >= vec3i(n))) { continue; }
          // (only voxels that are seeds themselves: their nearest seed is their own place)
          if (textureLoad(seeds, c).x != pack(c)) { continue; }
          let fs = F(c);
          let gs = grad(c);
          let gg = max(dot(gs, gs), 1e-8);
          let q = pos(c) - fs * gs / gg;
          let nn = gs * inverseSqrt(gg);
          let e = p - q;
          let along = dot(e, nn);
          let side = length(e - nn * along);
          let past = max(side - 0.4 * r.h, 0.0);
          d = min(d, sqrt(along * along + past * past));
        }
      }
    }
  }
  if (d > 1e8) {
    // no seed found: as far as the field's slope allows (never more than the true distance)
    d = abs(f) / abs(r.slope);
  }
  // (a negative slope: the field read is itself never more than the distance, and is kept where it is
  // less than the distance to the surface the voxels found: there a thin piece lies between them)
  if (r.slope < 0.0) { d = min(d, abs(f)); }
  let out = select(d, -d, f < 0.0) + r.post;
  textureStore(outv, v, vec4f(out, 0.0, 0.0, 0.0));
}


// A coarse cell covers eight fine samples per axis. Include its three positive
// faces, so an interface exactly between two otherwise uniform cells is counted.
var<workgroup> cellSigns: atomic<u32>;
fn warpSign(p: vec3i) -> u32 {
  let f = F(p);
  return select(0u, 1u, f >= 0.0) | select(0u, 2u, f <= 0.0);
}
@compute @workgroup_size(8, 8, 4)
fn cs_warp_cells(@builtin(workgroup_id) cell: vec3u, @builtin(local_invocation_id) local: vec3u,
                 @builtin(local_invocation_index) index: u32) {
  if (index == 0u) { atomicStore(&cellSigns, 0u); }
  workgroupBarrier();
  let p = vec3i(cell * 8u + local);
  var signs = warpSign(p) | warpSign(p + vec3i(0, 0, 4));
  if (local.x == 7u) {
    signs |= warpSign(p + vec3i(1, 0, 0)) | warpSign(p + vec3i(1, 0, 4));
  }
  if (local.y == 7u) {
    signs |= warpSign(p + vec3i(0, 1, 0)) | warpSign(p + vec3i(0, 1, 4));
  }
  if (local.z == 0u) { signs |= warpSign(p + vec3i(0, 0, 8)); }
  atomicOr(&cellSigns, signs);
  workgroupBarrier();
  if (index == 0u) {
    textureStore(seedsOut, vec3i(cell), vec4u(select(0x10000000u, 0u, atomicLoad(&cellSigns) == 3u)));
  }
}

@compute @workgroup_size(8, 8, 4)
fn cs_warp_axis(@builtin(global_invocation_id) id: vec3u) {
  let count = (r.res + 7u) / 8u;
  if (any(id >= vec3u(count))) { return; }
  let cell = vec3i(id);
  var q = cell;
  let axis = u32(r.step);
  var best = textureLoad(seeds, cell).x;
  for (var i = 0; i < i32(count); i++) {
    let delta = i - cell[axis];
    let d2 = u32(delta * delta);
    if (d2 < best) {
      q[axis] = i;
      best = min(best, d2 + textureLoad(seeds, q).x);
    }
  }
  textureStore(seedsOut, cell, vec4u(best));
}

@compute @workgroup_size(8, 8, 4)
fn cs_warp_distance(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= vec3u(r.res))) { return; }
  let voxel = vec3i(id);
  let cell = voxel / 8;
  let cellDistance = sqrt(f32(textureLoad(seeds, cell).x)) * 8.0;
  let toCentre = length(vec3f(voxel - cell * 8) - 3.5);
  // From the nearest mixed cell centre subtract the distance to this cell's
  // centre, and the radius enclosing its fine samples and neighbouring face.
  let lower = max(0.0, cellDistance - toCentre - 7.8) * r.h;
  let f = F(voxel);
  let cellRadius = 13.86 * r.h;
  var magnitude = lower;
  if (lower < cellRadius) {
    // The local field divided by its slope follows the zero crossing as closely
    // as the fine grid. Use the coarse-cell upper bound at a stationary point.
    let local = abs(f) / max(length(grad(voxel)), 1e-9);
    magnitude = clamp(local, lower, lower + cellRadius);
  }
  textureStore(outv, voxel, vec4f(select(magnitude, -magnitude, f < 0.0), 0.0, 0.0, 0.0));
}

// Rounding: the distance is smoothed along one axis (step: 0 x, 1 y, 2 z) with a bump-shaped weight
// `cres` voxels to either side. Three of these one after another smooth it in all directions. A flat
// piece's distance is unchanged by it; an edge comes out round, and anything thinner than the smoothing
// melts away. (Out of the band nothing is changed.)
@compute @workgroup_size(8, 8, 4)
fn cs_blur(@builtin(global_invocation_id) gid: vec3u) {
  let n = i32(r.res);
  let v = vec3i(gid);
  if (any(v >= vec3i(n))) { return; }
  let f = F(v);
  if (abs(f) > r.reach * abs(r.slope)) {
    textureStore(outv, v, vec4f(f, 0.0, 0.0, 0.0));
    return;
  }
  var ax = vec3i(1, 0, 0);
  if (r.step == 1) { ax = vec3i(0, 1, 0); }
  if (r.step == 2) { ax = vec3i(0, 0, 1); }
  let R = i32(r.cres);
  var sum = 0.0;
  var wsum = 0.0;
  for (var i = -R; i <= R; i++) {
    let t = f32(i) / f32(R + 1);
    let w = (1.0 - t * t) * (1.0 - t * t);
    sum += w * F(v + ax * i);
    wsum += w;
  }
  textureStore(outv, v, vec4f(sum / wsum, 0.0, 0.0, 0.0));
}

//#pieces

struct C {
  res: u32,
  cres: u32,
  thresh: u32,
  pad: u32,
  h: f32,
  pad1: f32,
  pad2: f32,
  pad3: f32,
};
@group(0) @binding(0) var<uniform> c: C;
@group(0) @binding(1) var cf: texture_3d<f32>;
@group(0) @binding(2) var<storage, read_write> labels: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> sizes: array<atomic<u32>>;
@group(0) @binding(4) var cout: texture_storage_3d<r16float, write>;

fn cidx(v: vec3u) -> u32 {
  return (v.z * c.cres + v.y) * c.cres + v.x;
}
fn find(x0: u32) -> u32 {
  var x = x0;
  loop {
    let p = atomicLoad(&labels[x]);
    if (p == x || p == NONE) { return x; }
    x = p;
  }
  return x;
}
fn unite(a0: u32, b0: u32) {
  var a = a0;
  var b = b0;
  loop {
    a = find(a);
    b = find(b);
    if (a == b) { break; }
    if (a < b) {
      let old = atomicMin(&labels[b], a);
      if (old == b) { break; }
      b = old;
    } else {
      let old = atomicMin(&labels[a], b);
      if (old == a) { break; }
      a = old;
    }
  }
}

@compute @workgroup_size(4, 4, 4)
fn cs_cc_init(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= vec3u(c.cres))) { return; }
  var inside = false;
  let k = c.res / c.cres;
  for (var z = 0u; z < k; z++) {
    for (var y = 0u; y < k; y++) {
      for (var x = 0u; x < k; x++) {
        if (textureLoad(cf, gid * k + vec3u(x, y, z), 0).r < 0.0) { inside = true; }
      }
    }
  }
  let i = cidx(gid);
  atomicStore(&labels[i], select(NONE, i, inside));
  atomicStore(&sizes[i], 0u);
  if (i == 0u) { atomicStore(&sizes[c.cres * c.cres * c.cres], 0u); }
}

@compute @workgroup_size(4, 4, 4)
fn cs_cc_merge(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= vec3u(c.cres))) { return; }
  let i = cidx(gid);
  if (atomicLoad(&labels[i]) == NONE) { return; }
  if (gid.x > 0u) { let j = i - 1u; if (atomicLoad(&labels[j]) != NONE) { unite(i, j); } }
  if (gid.y > 0u) { let j = i - c.cres; if (atomicLoad(&labels[j]) != NONE) { unite(i, j); } }
  if (gid.z > 0u) { let j = i - c.cres * c.cres; if (atomicLoad(&labels[j]) != NONE) { unite(i, j); } }
}

@compute @workgroup_size(4, 4, 4)
fn cs_cc_compress(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= vec3u(c.cres))) { return; }
  let i = cidx(gid);
  if (atomicLoad(&labels[i]) == NONE) { return; }
  let root = find(i);
  atomicStore(&labels[i], root);
  // a piece's size: how far (in cells, the largest of the three axes) its furthest cell lies from its
  // root. Its width across is at least that, and at most twice it; a thin sheet counts by its width, not
  // by how little it holds.
  let rz = root / (c.cres * c.cres);
  let ry = (root / c.cres) % c.cres;
  let rx = root % c.cres;
  let d = max(max(u32(abs(i32(gid.x) - i32(rx))), u32(abs(i32(gid.y) - i32(ry)))), u32(abs(i32(gid.z) - i32(rz))));
  atomicMax(&sizes[root], d + 1u);
}

@compute @workgroup_size(4, 4, 4)
fn cs_cc_largest(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= vec3u(c.cres))) { return; }
  let i = cidx(gid);
  if (atomicLoad(&labels[i]) != i) { return; }
  atomicMax(&sizes[c.cres * c.cres * c.cres], atomicLoad(&sizes[i]));
}

@compute @workgroup_size(8, 8, 4)
fn cs_cc_remove(@builtin(global_invocation_id) gid: vec3u) {
  if (any(gid >= vec3u(c.res))) { return; }
  var f = textureLoad(cf, gid, 0).r;
  if (f < 0.0) {
    let k = c.res / c.cres;
    let lab = atomicLoad(&labels[cidx(gid / k)]);
    if (lab != NONE) {
      let n = atomicLoad(&sizes[lab]);
      let most = atomicLoad(&sizes[c.cres * c.cres * c.cres]);
      // (the largest piece always stays)
      if (n < c.thresh && n < most) { f = 0.5 * c.h; }
    }
  }
  textureStore(cout, gid, vec4f(f, 0.0, 0.0, 0.0));
}
