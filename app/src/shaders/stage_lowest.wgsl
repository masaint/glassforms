// How far a shape reaches in one direction, from its distance volume (stage.rs: `Lowest`).
//
// A sample of the volume at place p, with distance s to the surface, says that the surface comes at
// least as far along d as p . d - s: inside the glass a ball of radius |s| about p is all glass, and
// outside it a ball of radius s just touches the glass. The largest of that over the samples near the
// surface is how far the shape reaches. A subset of the columns gives an initial maximum, letting the
// full-grid search end columns early. Blocks of 16 x 16 columns are reduced to one maximum.

struct M {
  down: vec4f,            // xyz: the direction, in the shape's own frame;  w: samples along a side
  k: vec4f,               // x, y: only distances below x and above y count;  z: the width of the volume
}
@group(0) @binding(0) var<uniform> m: M;
@group(0) @binding(1) var volTex: texture_3d<f32>;
@group(0) @binding(2) var smp: sampler;
@group(0) @binding(3) var prior: texture_2d<f32>;       // the pass before: the columns, or the blocks

const BLOCK: i32 = 16;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}

fn column(xy: vec2f, bound: f32) -> f32 {
  let n = m.down.w;
  let axis = u32(m.k.w);
  var best = bound;
  for (var k = 0; k < 256; k++) {
    if (f32(k) >= n) { break; }
    let z = select(f32(k) + 0.5, n - f32(k) - 0.5, m.down[axis] > 0.0);
    var c = vec3f(xy, z);
    if (axis == 0u) { c = c.zxy; }
    if (axis == 1u) { c = c.yzx; }
    c /= n;
    let projection = dot((c - 0.5) * m.k.z, m.down.xyz);
    // Later samples have smaller projections. Every accepted distance is greater than k.y.
    if (projection - m.k.y + 1e-6 * m.k.z < best) { break; }
    let s = textureSampleLevel(volTex, smp, c, 0.0).r;
    if (s < m.k.x && s > m.k.y) { best = max(best, projection - s); }
  }
  return best;
}

// A few columns from the actual grid supply a lower bound for its maximum.
@fragment
fn fs_seed(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let width = ceil(m.down.w / f32(BLOCK));
  let xy = floor(pos.xy * m.down.w / width) + 0.5;
  return vec4f(column(xy, -1e9), 0.0, 0.0, 1.0);
}

@fragment
fn fs_rows(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  return vec4f(column(pos.xy, textureLoad(prior, vec2i(0), 0).r), 0.0, 0.0, 1.0);
}

// the largest of one block of columns (a block that runs past the last column takes the columns there are)
@fragment
fn fs_blocks(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let n = i32(m.down.w);
  let c0 = vec2i(pos.xy) * BLOCK;
  var best = -1e9;
  for (var j = 0; j < BLOCK; j++) {
    for (var i = 0; i < BLOCK; i++) {
      let c = c0 + vec2i(i, j);
      if (c.x < n && c.y < n) { best = max(best, textureLoad(prior, c, 0).r); }
    }
  }
  return vec4f(best, 0.0, 0.0, 1.0);
}

// the largest of the blocks
@fragment
fn fs_all(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let n = (i32(m.down.w) + BLOCK - 1) / BLOCK;
  var best = -1e9;
  for (var j = 0; j < 16; j++) {
    if (j >= n) { break; }
    for (var i = 0; i < 16; i++) {
      if (i >= n) { break; }
      best = max(best, textureLoad(prior, vec2i(i, j), 0).r);
    }
  }
  return vec4f(best, 0.0, 0.0, 1.0);
}
