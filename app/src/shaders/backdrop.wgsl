// The frosted copy of the picture that the panels and the floating windows are drawn on (backdrop.rs).
// A dual-Kawase blur: the picture is halved a few times, five taps each time, then doubled back up with
// eight taps each time; the last pass up also sets its tone for the theme. Everything here works on the
// picture's display bytes as they stand (sRGB-encoded), which is the space the eye judges contrast in.

struct U {
  // xy: one texel of the source, in uv; z: the spread of the taps, in texels
  texel: vec4f,
  // rgb: the darkest colour the frost comes to; w: 1 on the last pass (the tone is set there)
  lo: vec4f,
  // rgb: the brightest colour the frost comes to; w: saturation (1 as it is)
  hi: vec4f,
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var lin: sampler;
@group(0) @binding(2) var<uniform> u: U;

struct V {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}

// one triangle over the whole target
@vertex
fn vs(@builtin(vertex_index) i: u32) -> V {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: V;
  o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2f(p.x, 1.0 - p.y);
  return o;
}

fn at(uv: vec2f) -> vec3f {
  return textureSampleLevel(src, lin, uv, 0.0).rgb;
}

@fragment
fn down(v: V) -> @location(0) vec4f {
  let h = u.texel.xy * u.texel.z;
  var s = at(v.uv) * 4.0;
  s += at(v.uv - h) + at(v.uv + h) + at(v.uv + vec2f(h.x, -h.y)) + at(v.uv - vec2f(h.x, -h.y));
  return vec4f(s / 8.0, 1.0);
}

@fragment
fn up(v: V) -> @location(0) vec4f {
  let h = u.texel.xy * u.texel.z;
  var s = at(v.uv + vec2f(-2.0 * h.x, 0.0)) + at(v.uv + vec2f(2.0 * h.x, 0.0)) + at(v.uv + vec2f(0.0, -2.0 * h.y)) + at(v.uv + vec2f(0.0, 2.0 * h.y));
  s += 2.0 * (at(v.uv + vec2f(-h.x, h.y)) + at(v.uv + vec2f(h.x, h.y)) + at(v.uv + vec2f(h.x, -h.y)) + at(v.uv + vec2f(-h.x, -h.y)));
  var c = s / 12.0;
  if (u.lo.w > 0.5) {
    // the theme's tone: saturation, then the whole range of the picture pressed into the band between
    // lo and hi, so that the brightest picture still leaves the words over it readable
    let l = dot(c, vec3f(0.2126, 0.7152, 0.0722));
    c = clamp(mix(vec3f(l), c, u.hi.w), vec3f(0.0), vec3f(1.0));
    c = u.lo.rgb + (u.hi.rgb - u.lo.rgb) * c;
  }
  return vec4f(c, 1.0);
}
