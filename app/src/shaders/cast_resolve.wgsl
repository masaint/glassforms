// Caustics: from light landed (cast_light.wgsl) to light arriving on a surface.
//
// What was landed is, for every voxel, a sum of light weighed by how near to the voxel each piece landed.
// What a surface wants is light per unit of its area. The two differ by how much of the receiving surface
// lies in the voxel's reach, weighed the same way, and that depends on where the surface happens to pass
// between the voxels: left out, a perfectly even light would come out striped by the grid. So that area is
// worked out for every voxel from the bodies' distances (cs_area: once per shape, not per frame), and the
// landed light is divided by it (cs_resolve).
//
// There are several volumes, each half the size of the one before (light that a large light source blurs,
// or that a body spread wide, was landed in a coarser one): they are folded together from the coarsest to
// the finest, so that a surface reads the whole of the arriving light with one lookup.

const HB: f32 = 1.5;
const BIG: f32 = 1.0e3;
const LEAST: f32 = 0.05;             // a voxel that holds less of the surface than this share of a voxel's face is not trusted

struct RU {
  recv: array<vec4f, 4>,     // for each body that gathers light: the channel of the bodies' volume that holds its distance
  gone: vec4f,               // for each body: BIG unless it gathers light
  k: vec4f,                  // x: how far off a surface light is landed
                             // y: the brightest light's brightness (landed light is counted against it: 65536
                             //    stands for what an even beam of it lands in a voxel)
  ku: vec4u,                 // x: voxels a side of this level   y: where its numbers begin in the buffer
                             // z: which level this is (0 = the finest)   w: 1 = there is a coarser level to fold in
}
@group(0) @binding(0) var<uniform> ru: RU;
@group(0) @binding(1) var fineTex: texture_3d<f32>;       // the distance to each body, at full detail
@group(0) @binding(2) var smp: sampler;

@group(1) @binding(0) var areaOut: texture_storage_3d<{{AREA}}, write>;

@group(1) @binding(1) var<storage, read> acc: array<i32>;         // landed light: whole numbers, three a voxel
@group(1) @binding(2) var areaTex: texture_3d<f32>;
@group(1) @binding(3) var coarser: texture_3d<f32>;
@group(1) @binding(4) var lightOut: texture_storage_3d<rgba16float, write>;

@group(1) @binding(5) var floorCoarser: texture_2d<f32>;
@group(1) @binding(6) var floorOut: texture_storage_2d<rgba16float, write>;

fn tent(x: f32) -> f32 { return max(1.0 - abs(x), 0.0); }

// the distance to the nearest body that gathers light
fn recvD(p: vec3f) -> f32 {
  let s = textureSampleLevel(fineTex, smp, p / (2.0 * HB) + 0.5, 0.0);
  let d = vec4f(dot(s, ru.recv[0]), dot(s, ru.recv[1]), dot(s, ru.recv[2]), dot(s, ru.recv[3])) + ru.gone;
  return min(min(d.x, d.y), min(d.z, d.w));
}

// How much of a flat surface lies in a voxel's reach, weighed as landed light is (by nearness to the
// voxel's centre along each axis), as a share of a voxel's face. s: the surface's distance from the
// centre, in voxels; n: its unit normal. (For a surface square to an axis this is 1 - |s|; turned, it is
// that folded over the other two axes, which is summed here at 6 x 6 points.)
fn planeIn(s: f32, n: vec3f) -> f32 {
  let a = abs(n);
  let c = max(a.x, max(a.y, a.z));
  var ab = a.xy;
  if (c == a.x) { ab = a.yz; } else if (c == a.y) { ab = a.xz; }
  var g = 0.0;
  for (var i = 0; i < 6; i++) {
    let xi = (f32(i) + 0.5) / 3.0 - 1.0;
    for (var j = 0; j < 6; j++) {
      let eta = (f32(j) + 0.5) / 3.0 - 1.0;
      g += tent(xi) * tent(eta) * tent((s - ab.x * xi - ab.y * eta) / c);
    }
  }
  return g / (9.0 * c);
}

@compute @workgroup_size(8, 8, 4)
fn cs_area(@builtin(global_invocation_id) id: vec3u) {
  let res = ru.ku.x;
  if (any(id >= vec3u(res))) { return; }
  let h = 2.0 * HB / f32(res);
  let p = ((vec3f(id) + 0.5) / f32(res) * 2.0 - 1.0) * HB;
  let d = recvD(p) - ru.k.x;                 // from where light is landed: the surface, moved out a little
  var g = 0.0;
  if (abs(d) < 1.8 * h) {
    if (ru.ku.z == 0u) {
      // the finest volume: its voxels are small against any body, and the surface is flat across one
      let e = vec2f(0.5 * h, 0.0);
      let n = vec3f(recvD(p + e.xyy) - recvD(p - e.xyy), recvD(p + e.yxy) - recvD(p - e.yxy), recvD(p + e.yyx) - recvD(p - e.yyx));
      g = planeIn(d / h, normalize(n + vec3f(1e-9, 0.0, 0.0)));
    } else {
      // coarser: the surface may curve within the voxel's reach, so it is looked for at 6 x 6 x 6 points
      // (a point counts by how near the surface it is: a thin slab stands in for the surface)
      let st = h / 3.0;
      let eps = 1.5 * st;
      for (var k = 0; k < 216; k++) {
        let o = (vec3f(f32(k % 6), f32((k / 6) % 6), f32(k / 36)) + 0.5) / 3.0 - 1.0;
        let dd = recvD(p + o * h) - ru.k.x;
        g += tent(o.x) * tent(o.y) * tent(o.z) * max(1.0 - abs(dd) / eps, 0.0) / eps;
      }
      g *= st * st * st / (h * h);
    }
  }
  textureStore(areaOut, id, vec4f(g, 0.0, 0.0, 1.0));
}

@compute @workgroup_size(8, 8, 4)
fn cs_resolve(@builtin(global_invocation_id) id: vec3u) {
  let res = ru.ku.x;
  if (any(id >= vec3u(res))) { return; }
  let i = ru.ku.y + ((id.z * res + id.y) * res + id.x) * 3u;
  let landed = vec3f(vec3i(acc[i], acc[i + 1u], acc[i + 2u])) * (ru.k.y / 65536.0);
  let g = textureLoad(areaTex, vec3i(id), 0).r;
  var e = vec3f(0.0);
  if (g > 1e-3) { e = landed / max(g, LEAST); }
  if (ru.ku.w == 1u) { e += textureSampleLevel(coarser, smp, (vec3f(id) + 0.5) / f32(res), 0.0).rgb; }
  textureStore(lightOut, id, vec4f(e, 1.0));
}

// The floor's maps. A floor is flat and its maps lie in its plane, so each texel holds exactly a texel's
// worth of floor: the landed light is light per area already (nothing to divide by).
@compute @workgroup_size(8, 8, 1)
fn cs_floor(@builtin(global_invocation_id) id: vec3u) {
  let res = ru.ku.x;
  if (any(id.xy >= vec2u(res))) { return; }
  let i = ru.ku.y + (id.y * res + id.x) * 3u;
  var e = vec3f(vec3i(acc[i], acc[i + 1u], acc[i + 2u])) * (ru.k.y / 65536.0);
  if (ru.ku.w == 1u) { e += textureSampleLevel(floorCoarser, smp, (vec2f(id.xy) + 0.5) / f32(res), 0.0).rgb; }
  textureStore(floorOut, vec2i(id.xy), vec4f(e, 1.0));
}
