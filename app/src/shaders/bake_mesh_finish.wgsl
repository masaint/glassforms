// A model's distance volume, finished: the volume the tracer reads is made from the model's base volume
// (the distances as the model was built: see bake_mesh.wgsl) by whatever its sliders ask for. All of them
// are cheap because the shape is a distance field already:
//
//   size      the base read at p / size, times size (a distance scales with the shape)
//   smooth    the base read from a softened copy of itself: fine detail melts
//   shell     distance to the surface less half the wall's thickness
//   inflate   an amount taken off the distance: the surface moves out (or in)
//
// The softened copies are the base volume's own smaller levels, each half the size of the one before,
// read through a cubic B-spline so that nothing shows of their coarser grids.

struct F {
  hb: f32,                // the volume spans [-hb, hb]
  res: f32,               // voxels per side of the volume being written (or of the level, when making levels)
  base_res: f32,          // voxels per side of the base volume's finest level
  size: f32,
  offset: f32,            // taken off every distance: inflation, and half a shell's wall
  lod: f32,               // the level a plain reading takes (above 0 when the volume is coarser than the base)
  soft_lod: f32,          // the level the softened reading takes
  soft: f32,              // how much of the answer is the softened reading, 0..1
  exact: u32,             // 1 = voxel for voxel: nothing resized, nothing softened
  pad0: u32,
  pad1: u32,
  pad2: u32,
}

@group(0) @binding(0) var<uniform> f: F;
@group(0) @binding(1) var baseTex: texture_3d<f32>;
@group(0) @binding(2) var smp: sampler;

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

fn smin(a: f32, b: f32, k: f32) -> f32 { let h = max(k - abs(a - b), 0.0) / k; return min(a, b) - h * h * k * 0.25; }
fn smax(a: f32, b: f32, k: f32) -> f32 { return -smin(-a, -b, k); }

fn level(t: vec3f, l: f32) -> f32 { return textureSampleLevel(baseTex, smp, t, l).r; }
// one level of the base through a cubic B-spline (eight interpolated lookups)
fn level_smooth(t: vec3f, l: f32) -> f32 {
  let res = vec3f(textureDimensions(baseTex, u32(l)));
  let c = t * res - 0.5;
  let i = floor(c);
  let g = c - i;
  let g2 = g * g;
  let g3 = g2 * g;
  let w0 = (1.0 - 3.0 * g + 3.0 * g2 - g3) / 6.0;
  let w1 = (4.0 - 6.0 * g2 + 3.0 * g3) / 6.0;
  let w3 = g3 / 6.0;
  let w2 = 1.0 - w0 - w1 - w3;
  let g0 = w0 + w1;
  let g1 = w2 + w3;
  let a = (i - 0.5 + w1 / g0) / res;
  let b = (i + 1.5 + w3 / g1) / res;
  return g0.z * (g0.y * (g0.x * level(vec3f(a.x, a.y, a.z), l) + g1.x * level(vec3f(b.x, a.y, a.z), l))
               + g1.y * (g0.x * level(vec3f(a.x, b.y, a.z), l) + g1.x * level(vec3f(b.x, b.y, a.z), l)))
       + g1.z * (g0.y * (g0.x * level(vec3f(a.x, a.y, b.z), l) + g1.x * level(vec3f(b.x, a.y, b.z), l))
               + g1.y * (g0.x * level(vec3f(a.x, b.y, b.z), l) + g1.x * level(vec3f(b.x, b.y, b.z), l)));
}

// the distance for the voxel whose place in the volume is `id`
fn finish(id: vec3u) -> vec4f {
  let p = ((vec3f(id) + 0.5) / f.res * 2.0 - 1.0) * f.hb;
  var d: f32;
  if (f.exact == 1u) {
    d = textureLoad(baseTex, id, 0).r;
  } else {
    // Where p / size falls outside the base volume the shape is at least as far off as the volume's
    // face, and as far again as the value at the nearest point of that face.
    let q = p / f.size;
    let lim = f.hb * (1.0 - 1.0 / f.base_res);
    let qc = clamp(q, vec3f(-lim), vec3f(lim));
    let t = qc / (2.0 * f.hb) + 0.5;
    d = level(t, f.lod);
    if (f.soft > 0.0) {
      let l = floor(f.soft_lod);
      d = mix(d, mix(level_smooth(t, l), level_smooth(t, l + 1.0), f.soft_lod - l), f.soft);
    }
    let out = length(q - qc);
    if (out > 0.0) { d = sqrt(out * out + max(d, 0.0) * max(d, 0.0)); }
    d *= f.size;
  }
  d -= f.offset;
  // the stage: nothing reaches the edge of the volume, whatever the sliders say
  d = smax(d, length(p) - 1.38, 0.03);
  return vec4f(clamp(d, -1.0, 4.0), 0.0, 0.0, 1.0);
}

// a smaller level of the base from the one above it: each voxel the mean of the eight it covers
fn smaller(id: vec3u) -> vec4f {
  return vec4f(textureSampleLevel(baseTex, smp, (vec3f(id) + 0.5) / f.res, 0.0).r, 0.0, 0.0, 1.0);
}

// drawn: one slice per pass, the slice number arriving as the instance number
@fragment
fn fs_finish(in: Full) -> @location(0) vec4f {
  return finish(vec3u(vec2u(in.pos.xy), in.inst));
}
@fragment
fn fs_smaller(in: Full) -> @location(0) vec4f {
  return smaller(vec3u(vec2u(in.pos.xy), in.inst));
}

//#compute  (everything below is left out where volumes cannot be written by compute shaders)
@group(0) @binding(3) var outVol: texture_storage_3d<r16float, write>;

@compute @workgroup_size(8, 8, 4)
fn cs_finish(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= vec3u(u32(f.res)))) { return; }
  textureStore(outVol, id, finish(id));
}
@compute @workgroup_size(8, 8, 4)
fn cs_smaller(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= vec3u(u32(f.res)))) { return; }
  textureStore(outVol, id, smaller(id));
}
