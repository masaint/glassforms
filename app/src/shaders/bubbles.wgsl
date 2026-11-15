// Bubbles: small hollows cut into a finished distance field, so that the tracer finds them like any
// other surface and pays nothing extra for them.
//
// The shape's box is divided into cells; each cell may hold one bubble, placed, sized and turned by
// numbers drawn from the cell's position (so the same settings always give the same bubbles). A bubble
// is kept only where it fits wholly inside the body with room to spare, and shrinks as it nears the
// surface instead of vanishing at once. The new field is the old one with every bubble taken out:
// max(old, -distance to the nearest bubble). The outside of the shape is not touched.
//
// Every voxel looks at the 27 cells around it. Bubbles further off than that are at least `reach` away,
// so where none is nearer the field is told `reach`: a safe understatement, as a distance field must be.

struct B {
  res: f32,               // voxels per side
  hb: f32,                // the volume spans -hb .. hb
  cell: f32,              // side of a cell
  size: f32,              // radius of the largest bubble
  even: f32,              // 1 = all alike and everywhere, 0 = every size, in clusters
  shape: f32,             // -1 = flat discs, 0 = round, 1 = needles
  seed: f32,
  margin: f32,            // how far a bubble must stay from the surface
}

@group(0) @binding(0) var<uniform> b: B;
@group(0) @binding(1) var src: texture_3d<f32>;
@group(0) @binding(2) var lin: sampler;

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn u01(h: u32) -> f32 { return f32(h >> 8u) / 16777216.0; }

// the field at voxel `id` with the bubbles taken out; d0: the field there as it stands
fn carve(id: vec3u, d0: f32) -> f32 {
  let vox = 2.0 * b.hb / b.res;
  // at the surface or outside it: nothing changes
  if (d0 > -0.5 * vox) { return d0; }
  let p = ((vec3f(id) + 0.5) / b.res - 0.5) * (2.0 * b.hb);
  let far = 1.0 + 3.0 * abs(b.shape);                 // how far a bubble reaches from its centre, in radii
  var best = b.cell - b.size * far;                   // nothing nearer than this is known of the cells not looked at
  let c0 = vec3i(floor(p / b.cell));
  for (var k = 0; k < 27; k++) {
    let c = c0 + vec3i(k % 3, (k / 3) % 3, k / 9) - 1;
    let h = pcg(u32(c.x) + pcg(u32(c.y) + pcg(u32(c.z) + pcg(u32(b.seed) * 7919u + 17u))));
    // is there a bubble in this cell? Always nearly so when even; otherwise only in patches
    let cc = vec3f(c) * 0.55 + b.seed;
    let clump = 0.5 + 0.5 * sin(cc.x * 1.3 + 2.1 * sin(cc.y * 0.9 + 1.7 * sin(cc.z * 1.1)));
    let chance = mix(clamp(2.2 * clump * clump - 0.15, 0.0, 1.0), 0.85, b.even);
    if (u01(h) > chance) { continue; }
    let ctr = (vec3f(c) + vec3f(u01(pcg(h + 1u)), u01(pcg(h + 2u)), u01(pcg(h + 3u)))) * b.cell;
    // its size: all alike when even; otherwise many small ones and a few large
    var r = b.size * mix(0.15 + 0.85 * pow(u01(pcg(h + 4u)), 2.2), 1.0, b.even);
    let v = p - ctr;
    let lv = length(v);
    if (lv - r * far > best) { continue; }
    // does it fit? The field at its centre says how deep inside the body that is.
    let room = -textureSampleLevel(src, lin, ctr / (2.0 * b.hb) + 0.5, 0.0).r - b.margin;
    r = min(r, room / far);
    if (r < 1.2 * vox) { continue; }                  // too small to draw
    // the distance to it: a ball, a needle (a ball swept along an axis) or a disc with a rounded rim
    let z = 1.0 - 2.0 * u01(pcg(h + 5u));
    let an = 6.2831853 * u01(pcg(h + 6u));
    let a = vec3f(sqrt(max(1.0 - z * z, 0.0)) * vec2f(cos(an), sin(an)), z);
    let along = dot(v, a);
    var d = lv - r;
    if (b.shape > 0.0) {
      let hl = 3.0 * b.shape * r;
      d = length(v - clamp(along, -hl, hl) * a) - r;
    } else if (b.shape < 0.0) {
      let t = r * (1.0 + 0.65 * b.shape);
      if (t < 0.8 * vox) { continue; }
      d = length(vec2f(max(length(v - along * a) + 3.0 * b.shape * r, 0.0), along)) - t;
    }
    best = min(best, d);
  }
  return max(d0, -best);
}

struct Full {
  @builtin(position) pos: vec4f,
  @location(0) @interpolate(flat) inst: u32,
}
@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) inst: u32) -> Full {
  let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  var o: Full;
  o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  o.inst = inst;
  return o;
}
// drawn: one slice per pass, the slice number arriving as the instance number
@fragment
fn fs(in: Full) -> @location(0) vec4f {
  let id = vec3u(vec2u(in.pos.xy), in.inst);
  return vec4f(carve(id, textureLoad(src, id, 0).r), 0.0, 0.0, 1.0);
}

//#compute  (everything below is left out where volumes cannot be written by compute shaders)
@group(0) @binding(3) var dst: texture_storage_3d<r16float, write>;

// computed: the whole volume in one go
@compute @workgroup_size(8, 8, 4)
fn cs(@builtin(global_invocation_id) id: vec3u) {
  textureStore(dst, id, vec4f(carve(id, textureLoad(src, id, 0).r), 0.0, 0.0, 1.0));
}
