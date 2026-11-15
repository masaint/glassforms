// Turning a polygon model into a distance field.
//
//   1. gather    every triangle drawn into just the slices it can reach, a slab of slices at a time, the
//                slices of the slab laid side by side on one sheet; each pixel keeps the least distance
//   2. resolve   sheet + inside/outside + the coarse grid -> the volume: within a few voxels of the
//                surface the exact distance from the sheet, beyond that the coarse grid read smoothly
//
// Inside or outside comes from the CPU, exactly: for every line of voxels along z, the heights at which
// the line passes between the two. The coarse grid is the CPU's too: the exact distance at every point
// of a grid a quarter as fine as the volume.
//
// Texel row j of the volume is y = ((j + 0.5) / res * 2 - 1) * hb.

struct M {
  hb: f32,                // the volume spans [-hb, hb]
  res: f32,               // voxels per side of the volume being written
  band: f32,              // how far from itself a triangle writes; beyond this the sheet says nothing
  lo: f32,                // where the sheet's distance begins to give way to the coarse grid's
  mres: f32,              // points per side of the coarse grid
  mcell: f32,             // the spacing of its points
  offset: f32,            // taken off every distance written (a shell's half thickness, an inflation)
  pad0: f32,
  slab: u32,              // slices gathered at a time
  stiles: u32,            // slices across the sheet
  shell: u32,             // 1 = no inside: the distance to the surface itself
  pad1: u32,
}

@group(0) @binding(0) var<uniform> m: M;

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

// ---- 1. gather: each triangle is drawn as the rectangle it can reach, into each slice it can reach.
// One instance = one triangle in one slice.
@group(1) @binding(0) var<storage, read> tris: array<vec4f>;      // three corners to a triangle

struct Tri {
  @builtin(position) pos: vec4f,
  @location(0) @interpolate(flat) a: vec3f,
  @location(1) @interpolate(flat) b: vec3f,
  @location(2) @interpolate(flat) c: vec3f,
  @location(3) @interpolate(flat) origin: vec2f,      // top left of this slice's place on the sheet, in pixels
  @location(4) @interpolate(flat) zc: f32,            // the slice's depth, 0..1
}
@vertex
fn vs_splat(@builtin(vertex_index) vi: u32, @location(0) tri: u32, @location(1) slice: u32) -> Tri {
  let a = tris[3u * tri].xyz;
  let b = tris[3u * tri + 1u].xyz;
  let c = tris[3u * tri + 2u].xyz;
  // the rectangle the triangle can reach, kept inside the slice so that it cannot spill onto its neighbours
  let lo = max(min(min(a.xy, b.xy), c.xy) - m.band, vec2f(-m.hb));
  let hi = min(max(max(a.xy, b.xy), c.xy) + m.band, vec2f(m.hb));
  let corner = vec2f(f32(vi & 1u), f32(vi >> 1u));
  let uv = mix(lo, hi, corner) / m.hb * 0.5 + 0.5;          // 0..1 across the slice, rows from the top
  let t = slice % m.slab;
  let tile = vec2f(f32(t % m.stiles), f32(t / m.stiles));
  let at = (tile + uv) / f32(m.stiles);                     // 0..1 across the sheet
  var o: Tri;
  o.pos = vec4f(at.x * 2.0 - 1.0, 1.0 - at.y * 2.0, 0.0, 1.0);
  if (any(lo >= hi)) { o.pos = vec4f(2.0, 2.0, 0.0, 1.0); }    // wholly outside the volume: draw nothing
  o.a = a;
  o.b = b;
  o.c = c;
  o.origin = tile * m.res;
  o.zc = (f32(slice) + 0.5) / m.res;
  return o;
}

fn dot2(v: vec3f) -> f32 { return dot(v, v); }

// The distance from p to the triangle a b c: to its face where p is over the face, otherwise to the
// nearest of its edges. (A triangle with no area is its edges.)
fn tri_dist(p: vec3f, a: vec3f, b: vec3f, c: vec3f) -> f32 {
  let ba = b - a;
  let pa = p - a;
  let cb = c - b;
  let pb = p - b;
  let ac = a - c;
  let pc = p - c;
  let nor = cross(ba, ac);
  let over = sign(dot(cross(ba, nor), pa)) + sign(dot(cross(cb, nor), pb)) + sign(dot(cross(ac, nor), pc));
  if (over < 2.0) {
    return sqrt(min(min(
      dot2(ba * clamp(dot(ba, pa) / max(dot2(ba), 1e-30), 0.0, 1.0) - pa),
      dot2(cb * clamp(dot(cb, pb) / max(dot2(cb), 1e-30), 0.0, 1.0) - pb)),
      dot2(ac * clamp(dot(ac, pc) / max(dot2(ac), 1e-30), 0.0, 1.0) - pc)));
  }
  return abs(dot(nor, pa)) / sqrt(max(dot2(nor), 1e-30));
}
@fragment
fn fs_splat(in: Tri) -> @location(0) vec4f {
  let p = (vec3f((in.pos.xy - in.origin) / m.res, in.zc) * 2.0 - 1.0) * m.hb;
  return vec4f(tri_dist(p, in.a, in.b, in.c), 0.0, 0.0, 1.0);
}

// ---- 2. the volume
@group(1) @binding(1) var midTex: texture_3d<f32>;
@group(1) @binding(2) var lin: sampler;
@group(1) @binding(3) var sheetTex: texture_2d<f32>;
@group(1) @binding(4) var<storage, read> lines: array<u32>;      // where each line's heights begin
@group(1) @binding(5) var<storage, read> flips: array<u32>;      // the heights, ascending along each line

// The coarse grid read through a cubic B-spline: no kink at its cell boundaries, which would show in
// anything that is later shaped by these distances (eight interpolated lookups: Sigg & Hadwiger).
fn mid(t: vec3f) -> f32 { return textureSampleLevel(midTex, lin, t, 0.0).r; }
fn mid_smooth(t: vec3f) -> f32 {
  let c = t * m.mres - 0.5;
  let i = floor(c);
  let f = c - i;
  let f2 = f * f;
  let f3 = f2 * f;
  let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
  let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
  let w3 = f3 / 6.0;
  let w2 = 1.0 - w0 - w1 - w3;
  let g0 = w0 + w1;
  let g1 = w2 + w3;
  let a = (i - 0.5 + w1 / g0) / m.mres;
  let b = (i + 1.5 + w3 / g1) / m.mres;
  return g0.z * (g0.y * (g0.x * mid(vec3f(a.x, a.y, a.z)) + g1.x * mid(vec3f(b.x, a.y, a.z)))
               + g1.y * (g0.x * mid(vec3f(a.x, b.y, a.z)) + g1.x * mid(vec3f(b.x, b.y, a.z))))
       + g1.z * (g0.y * (g0.x * mid(vec3f(a.x, a.y, b.z)) + g1.x * mid(vec3f(b.x, a.y, b.z)))
               + g1.y * (g0.x * mid(vec3f(a.x, b.y, b.z)) + g1.x * mid(vec3f(b.x, b.y, b.z))));
}

// Is the centre of this voxel inside the solid? The line it is on passes between outside and inside at
// the heights listed for it; above them all is outside.
fn inside(px: vec2u, slice: u32) -> bool {
  let l = px.y * u32(m.res) + px.x;
  let z = (slice << 16u) + 32768u;
  var n = 0u;
  for (var i = lines[l]; i < lines[l + 1u]; i++) {
    if (flips[i] >= z) { n++; }
  }
  return (n & 1u) == 1u;
}

// What the coarse grid says of how far the surface is, made safe: read between its points it can
// overstate a distance where the surface has a sharp edge it is too coarse to follow, by a share of a
// cell that falls off with the distance. That much is taken off, so that what is stored is never more
// than the truth (the tracer strides by what it reads).
fn safe(far: f32) -> f32 {
  return far - 0.3 * m.mcell * m.mcell / max(far, m.mcell);
}

fn resolve(px: vec2f, slice: u32) -> vec4f {
  let t = slice % m.slab;
  let origin = vec2i(i32(t % m.stiles), i32(t / m.stiles)) * i32(m.res);
  // (held to the band: where no triangle reaches, the sheet holds a very large number, and mixing a
  // small distance with that would cost it most of its digits)
  let near = min(textureLoad(sheetTex, origin + vec2i(px), 0).r, m.band);
  let at = vec3f(px / m.res, (f32(slice) + 0.5) / m.res);
  // how much of the answer is the coarse grid's: none close to the surface, all of it beyond the band
  let w = smoothstep(m.lo, m.band, near);
  var d: f32;
  if (m.shell == 1u) {
    // A distance with no sign has a crease at the surface, which the smooth reading rounds off for two
    // cells either side; nearer than that the plain reading (which does not look so far) is used.
    let plain = mid(at);
    let far = mix(plain, mid_smooth(at), smoothstep(1.5 * m.mcell, 2.25 * m.mcell, plain));
    d = mix(near, max(safe(far), m.lo), w);
  } else {
    // (the coarse grid is trusted for how far, never for which side, and never for less than the band's start)
    let s = select(1.0, -1.0, inside(vec2u(px), slice));
    d = s * mix(near, max(safe(abs(mid_smooth(at))), m.lo), w);
  }
  return vec4f(clamp(d - m.offset, -1.0, 4.0), 0.0, 0.0, 1.0);
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
@group(1) @binding(6) var outVol: texture_storage_3d<r16float, write>;

// computed: a whole slab of slices at once
@compute @workgroup_size(8, 8, 4)
fn cs_resolve(@builtin(global_invocation_id) id: vec3u) {
  if (id.z >= sb.n) { return; }
  let slice = sb.z0 + id.z;
  textureStore(outVol, vec3u(id.xy, slice), resolve(vec2f(id.xy) + 0.5, slice));
}
