// DLSS's full-size picture made the picture so far (dlss.rs): its colour, and, as the picture so far keeps
// beside its colour, the distance to the glass, taken from the small traced picture's pixel underneath.

struct R {
  small: vec2f,           // size of the small traced picture
  res: vec2f,             // size of the picture so far
}

@group(0) @binding(0) var<uniform> r: R;
@group(0) @binding(1) var upTex: texture_2d<f32>;
@group(0) @binding(2) var smallTex: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let c = vec2i(pos.xy);
  let col = textureLoad(upTex, c, 0).rgb;
  let s = clamp(vec2i(pos.xy * r.small / r.res), vec2i(0), vec2i(r.small) - 1);
  let d = textureLoad(smallTex, s, 0).a;
  return vec4f(max(col, vec3f(0.0)), d);
}
