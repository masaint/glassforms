// The distance to each body at modest detail, one body a channel: what the shadow marches and the light
// that is followed forward step through. It is read from the scene's own volumes, which are far finer
// than a soft shadow needs and, with several bodies, several gigabytes: a march through those would spend
// its time waiting for memory.

struct B {
  count: u32,             // how many bodies there are: channels beyond them are written as far away
  res: u32,               // voxels a side of the volume being written
  pad0: u32,
  pad1: u32,
}
@group(0) @binding(0) var<uniform> b: B;
@group(0) @binding(1) var fine: texture_3d<f32>;        // the bodies' volume, or with one body the scene's
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var outVol: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(8, 8, 4)
fn cs(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= vec3u(b.res))) { return; }
  // a distance read between the fine volume's voxels is the distance there: nothing needs averaging
  let s = textureSampleLevel(fine, lin, (vec3f(id) + 0.5) / f32(b.res), 0.0);
  let far = vec4f(4.0);
  let have = vec4u(0u, 1u, 2u, 3u) < vec4u(b.count);
  textureStore(outVol, id, select(far, s, have));
}
