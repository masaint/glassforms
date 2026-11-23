
// What the tracer is given besides the family's lights when it is built for an environment picture or
// for clay. Glass lit by the family's lights alone is built without any of this: `tracer_source` in
// render.rs puts this text into trace.wgsl, after its bindings, only for the other ways of drawing.

struct PaintLight { a: vec4f, u: vec4f, v: vec4f, rgb: vec4f }
struct LU {
  picture: vec4f,           // x: brightness of the environment picture where it is shown behind the shape
                            // z: the level of its softened copies shown there
  clay: vec4f,              // rgb: the colour of the clay; w: how strong its sheen is
  clay2: vec4f,             // x: the light a matte surface gathers on average; y: how far differences from
                            // that average are drawn out (0 = not at all)
  nine: array<vec4f, 18>,   // the lighting as nine smooth terms a colour (env.rs: `sh_terms`), in the lights'
                            // own frame. First nine: the light a matte surface facing a direction gathers.
                            // Last nine: the light arriving from around a direction (for the sheen).
  paintRoom: array<vec4f, 9>,
  paintLights: array<PaintLight, 12>,
}
@group(0) @binding(5) var<uniform> lu: LU;
@group(0) @binding(6) var pictureTex: texture_cube<f32>;      // the environment picture on a cube, each level softer
@group(0) @binding(7) var pictureSmp: sampler;
@group(0) @binding(8) var glassTex: texture_cube<f32>;        // the light the glass sees: the picture as soft and as
                                                              // bright as it is set, with the lights' map added in

// The light arriving from direction d (in the lights' own frame) as the glass sees it. One lookup and
// nothing else, on purpose: this is called from the innermost part of the tracer.
fn picture(d: vec3f) -> vec3f {
  return textureSampleLevel(glassTex, pictureSmp, d, 0.0).rgb;
}
// The picture as it is, for a ray from the camera that misses the shape.
fn backdrop(rd: vec3f) -> vec3f {
  return textureSampleLevel(pictureTex, pictureSmp, toEnv(rd), lu.picture.z).rgb * lu.picture.x;
}

// One of the two sets of nine terms, summed for direction d (in the lights' own frame).
fn nine(o: i32, d: vec3f) -> vec3f {
  let v = pbr_sh9_sum(lu.nine[o].rgb, lu.nine[o + 1].rgb, lu.nine[o + 2].rgb, lu.nine[o + 3].rgb, lu.nine[o + 4].rgb,
                      lu.nine[o + 5].rgb, lu.nine[o + 6].rgb, lu.nine[o + 7].rgb, lu.nine[o + 8].rgb, d);
  return max(v, vec3f(0.0));
}

// Matte clay at p, seen along rd: no rays are followed into or off the shape.
//   light     what a matte surface facing this way gathers from the lights and the environment picture
//   shadow    how hemmed in the surface is: a few steps out along the normal, each asking the distance field
//             how near the shape is. Open surface finds it as far away as the step; in a fold or where two
//             pieces cross it is nearer. Light bouncing between pale surfaces fills such places back in a
//             little and takes on the clay's colour (the fit is Jimenez et al., "Practical Real-Time
//             Strategies for Accurate Indirect Occlusion", 2016).
//   sheen     a soft reflection of the lighting, stronger towards the outline
fn clay(p: vec3f, rd: vec3f) -> vec3f {
  let n = nrm(p);
  let open = lightOcclusion(p, n);   // light/occlusion.wgsl
  let c = lu.clay.rgb;
  let a = 2.0404 * c - 0.3324;
  let b = -4.7951 * c + 0.6417;
  let d = 2.7552 * c + 0.6903;
  let filled = max(vec3f(open), ((open * a + b) * open + d) * open);
  let facing = clamp(-dot(rd, n), 0.0, 1.0);
  let rim = 1.0 - facing;
  let sheen = lu.clay.w * (0.05 + 0.6 * rim * rim * rim * rim);
  // The glass's lights stand all round the shape, so that glass facing any way catches one; a matte
  // surface gathers nearly the same light from them whichever way it faces, and its form goes flat. So
  // the differences are drawn out: light above the average counts for more, light below it for less
  // (lu.clay2.x is the average, lu.clay2.y how strongly).
  var light = nine(0, toEnv(n));
  let level = dot(light, vec3f(0.2126, 0.7152, 0.0722)) / lu.clay2.x;
  light *= pow(max(level, 1e-4), lu.clay2.y);
  return c * light * filled + nine(9, toEnv(reflect(rd, n))) * (sheen * open);
}
