// The distance to each body at modest detail, one body a channel: what the shadow marches and the light
// that is followed forward step through. It is read from the scene's own volumes (which can hold
// piecewise lower bounds or tight-grid cutoffs outside the shape) and refined by a 3-pass separable exact
// 3D Euclidean Distance Transform (Felzenszwalb & Huttenlocher) so floor shadows and light volumes see
// smooth, true Euclidean distances everywhere.

struct B {
  count: u32,             // how many bodies there are: channels beyond them are written as far away
  res: u32,               // voxels a side of the volume being written (<= 192)
  ch: u32,                // channel currently being transformed (0..count-1)
  hb: f32,                // half-width of the scene box in world units
}
@group(0) @binding(0) var<uniform> b: B;
@group(0) @binding(1) var fine: texture_3d<f32>;        // the bodies' volume, or with one body the scene's
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var outVol: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var<storage, read> edtIn: array<f32>;
@group(0) @binding(5) var<storage, read_write> edtOut: array<f32>;
@group(0) @binding(6) var prevVol: texture_3d<f32>;     // prior channels when ch > 0

@compute @workgroup_size(8, 8, 4)
fn cs_seed(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= vec3u(b.res))) { return; }
  let s4 = textureSampleLevel(fine, lin, (vec3f(id) + 0.5) / f32(b.res), 0.0);
  let s = s4[min(b.ch, 3u)];
  let h = 2.0 * b.hb / f32(b.res);
  var f = 1e12;
  if (s <= 0.0) {
    f = 0.0;
  } else if (s < 0.5 * h) {
    let u = s / max(h, 1e-6);
    f = u * u;
  }
  let idx = (id.z * b.res + id.y) * b.res + id.x;
  edtOut[idx] = f;
}

fn edt1d(base: u32, stride: u32) {
  if (b.res > 192u) { return; }
  let n = i32(b.res);
  var v: array<i32, 192>;
  var z: array<f32, 193>;
  var k: i32 = -1;
  for (var q: i32 = 0; q < n; q = q + 1) {
    let fq = edtIn[base + u32(q) * stride];
    if (fq >= 1e11) { continue; }
    if (k < 0) {
      k = 0;
      v[0] = q;
      z[0] = -1e20;
      z[1] = 1e20;
      continue;
    }
    let qf = f32(q);
    var s = 0.0;
    loop {
      let vk = v[k];
      let vkf = f32(vk);
      let fvk = edtIn[base + u32(vk) * stride];
      s = ((fq + qf * qf) - (fvk + vkf * vkf)) / (2.0 * (qf - vkf));
      if (s > z[k]) { break; }
      k = k - 1;
      if (k < 0) { break; }
    }
    k = k + 1;
    v[k] = q;
    z[k] = select(-1e20, s, k > 0);
    z[k + 1] = 1e20;
  }
  if (k < 0) {
    for (var q: i32 = 0; q < n; q = q + 1) {
      edtOut[base + u32(q) * stride] = 1e12;
    }
    return;
  }
  var ki: i32 = 0;
  for (var q: i32 = 0; q < n; q = q + 1) {
    let qf = f32(q);
    while (z[ki + 1] < qf) { ki = ki + 1; }
    let vk = v[ki];
    let d = qf - f32(vk);
    edtOut[base + u32(q) * stride] = d * d + edtIn[base + u32(vk) * stride];
  }
}

@compute @workgroup_size(8, 8, 1)
fn cs_edt_x(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= b.res || id.y >= b.res) { return; }
  edt1d((id.y * b.res + id.x) * b.res, 1u);
}

@compute @workgroup_size(8, 8, 1)
fn cs_edt_y(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= b.res || id.y >= b.res) { return; }
  edt1d(id.y * b.res * b.res + id.x, b.res);
}

@compute @workgroup_size(8, 8, 1)
fn cs_edt_z(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= b.res || id.y >= b.res) { return; }
  edt1d(id.y * b.res + id.x, b.res * b.res);
}

@compute @workgroup_size(8, 8, 4)
fn cs_finish(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= vec3u(b.res))) { return; }
  let ch = min(b.ch, 3u);
  let s4 = textureSampleLevel(fine, lin, (vec3f(id) + 0.5) / f32(b.res), 0.0);
  let s = s4[ch];
  let h = 2.0 * b.hb / f32(b.res);
  let idx = (id.z * b.res + id.y) * b.res + id.x;
  let e = edtIn[idx];
  let dEdt = select(sqrt(e) * h, 4.0, e >= 1e11);
  var d = s;
  if (s > h) {
    d = select(mix(s, dEdt, smoothstep(h, 3.0 * h, s)), 4.0, e >= 1e11);
  }
  var out = vec4f(d, 4.0, 4.0, 4.0);
  if (ch > 0u) {
    out = textureLoad(prevVol, vec3i(id), 0);
    out[ch] = d;
  }
  textureStore(outVol, id, out);
}
