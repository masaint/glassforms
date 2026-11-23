// The matcap view's picture so far onto the screen: no exposure, grade, bloom or glare, only the sRGB
// encoding (and an 8-bit dither on the ordinary screen).

struct ShowU {
  res: vec2f,      // the screen's size
  dither: f32,     // one step of the screen's depth, or 0
  pad: f32,
}
@group(0) @binding(0) var<uniform> su: ShowU;
@group(0) @binding(1) var pic: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn srgb(c: vec3f) -> vec3f {
  let x = clamp(c, vec3f(0.0), vec3f(1.0));
  return select(1.055 * pow(x, vec3f(1.0 / 2.4)) - 0.055, 12.92 * x, x <= vec3f(0.0031308));
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let size = vec2f(textureDimensions(pic));
  let at = vec2i(min(floor(pos.xy * size / su.res), size - 1.0));
  let c = srgb(textureLoad(pic, at, 0).rgb);
  let n = fract(52.9829189 * fract(dot(pos.xy, vec2f(0.06711056, 0.00583715)))) - 0.5;
  return vec4f(clamp(c + n * su.dither, vec3f(0.0), vec3f(1.0)), 1.0);
}
