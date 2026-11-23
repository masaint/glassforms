// Lobe weights of an opaque layered surface and the one way they meet the gathered light. Prepend
// paint_interface.wgsl (pbr_layer_shares is its paintLayerWeights) and pbr/diffuse.wgsl.
//   coat     the top interface's own reflection
//   spec     reflection from the interface below that comes back out through the coat
//   spec_ms  the multiple-scatter remainder of spec (zero for these smooth dielectric layers)
//   diffuse  the substrate's diffuse response, albedo included
//   enter    what crosses into the body for a medium estimator (walk, light volume, glass transport)
//   sheen    reserved, zero
struct PbrLobeWeights { coat: vec3f, spec: vec3f, spec_ms: vec3f, diffuse: vec3f, enter: vec3f, sheen: vec3f }
// What each lobe gathers: coat and spec along their mirror directions, ambient and directed for the
// diffuse lobe (irradiance over pi, with and without a direction), sheen reserved.
struct LightGather { coat: vec3f, spec: vec3f, spec_ms: vec3f, ambient: vec3f, directed: vec3f, sheen: vec3f }

fn pbr_layer_shares(c: f32, incident: f32, body_ior: f32, coat_ior: f32, coated: bool) -> vec3f {
  return paintLayerWeights(c, incident, body_ior, coat_ior, coated);
}

// An opaque body under an optional clear coat: shares from the two-interface series, the diffuse lobe
// from pbr_diffuse_albedo (albedo, the body share, and `keep`: what of it the surface itself keeps, the
// rest going on into the body as `enter`).
fn pbr_opaque_weights(c: f32, incident: f32, body_ior: f32, coat_ior: f32, coated: bool,
                      albedo: vec3f, keep: f32) -> PbrLobeWeights {
  let s = pbr_layer_shares(c, incident, body_ior, coat_ior, coated);
  var w: PbrLobeWeights;
  w.coat = vec3f(s.x);
  w.spec = vec3f(s.y);
  w.diffuse = pbr_diffuse_albedo(albedo, s.z, c) * keep;
  w.enter = vec3f(s.z * (1.0 - keep));
  return w;
}

fn pbr_respond(w: PbrLobeWeights, g: LightGather) -> vec3f {
  return w.coat * g.coat + w.spec * g.spec + w.spec_ms * g.spec_ms + w.diffuse * (g.ambient + g.directed) + w.sheen * g.sheen;
}
