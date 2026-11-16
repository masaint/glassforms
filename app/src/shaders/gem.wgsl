
// A cut stone (gems.rs `inside`): every facet's plane in the object's frame, so that a ray inside the
// stone finds where it leaves without marching. info: how many planes, then the band of heights within
// which the rim's curved wall stands (empty, low above high, when the stone has none).
struct GemU { info: vec4f, pl: array<vec4f, {{GEM_MAX}}> }
@group(0) @binding({{GEM_BINDING}}) var<uniform> gem: GemU;

fn gemWorld(v: vec3f) -> vec3f { return vec3f(dot(v, u.obj0.xyz), dot(v, u.obj1.xyz), dot(v, u.obj2.xyz)); }

// From pos inside the stone, heading dir: how far to where the ray leaves it, and the outward normal there.
// The stone is convex, so the ray leaves through the first facet plane it reaches. Where the baked field
// is exactly that plane's (no edge, corner or rim within reach of the smooth lookups), the plane's own
// distance and normal are what marching and nrm() would find (see gemUneven), and are used instead. Elsewhere the
// way out is found on the field as marchIn does, starting just short of the plane, so that edges and
// the rim look as they always have.
fn gemExit(pos: vec3f, dir: vec3f, mOut: ptr<function, vec3f>) -> f32 {
  let po = toObj(pos);
  let dob = toObj(dir);
  let n = i32(gem.info.x);
  var t = 1e9;
  var k = 0;
  for (var i = 0; i < n; i++) {
    let pl = gem.pl[i];
    let dn = dot(pl.xyz, dob);
    if (dn > 1e-6) {
      let ti = (pl.w - dot(pl.xyz, po)) / dn;
      if (ti < t) { t = ti; k = i; }
    }
  }
  t = max(t, 0.0);
  let q = po + dob * t;
  let vox = 2.0 * HB / u.vres;
  // how far about q the smooth lookups and nrm() read the volume, each way along each axis: mapN() reads
  // voxels up to two away, and nrm() reads mapN() one voxel away
  let reach = 3.05 * vox;
  if (q.y > gem.info.y - reach && q.y < gem.info.z + reach) {
    // the rim's wall may cut the ray off sooner: march as ever
    let s = marchIn(pos, dir);
    *mOut = nrm(pos + dir * s);
    return s;
  }
  let nk = gem.pl[k].xyz;
  var exact = true;
  for (var i = 0; i < n; i++) {
    let pl = gem.pl[i];
    let dd = abs(pl.xyz - nk);
    // the most another plane can rise against this one anywhere in the box the lookups read
    if (i != k && dot(pl.xyz, q) - pl.w + reach * (dd.x + dd.y + dd.z) >= 0.0) { exact = false; break; }
  }
  if (exact) {
    *mOut = gemUneven(gemWorld(nk), q);
    return t;
  }
  // near an edge or a corner: onto the field's own surface, as marchIn settles onto it
  var s = max(t - 3.0 * vox, 0.0);
  for (var i = 0; i < STEPS_IN; i++) {
    let d = -mapW(pos + dir * s);
    if (d < 0.0008) { break; }
    s += d * STEP_SCALE;
    if (s > 2.5 * BOUND) { break; }
  }
  s -= mapNW(pos + dir * s);
  s -= mapNW(pos + dir * s);
  *mOut = nrm(pos + dir * s);
  return s;
}

// The normals the tracer has always read off the volume are not quite the facet's own: read through 16-bit
// voxels and the texture unit's interpolation they wander, from one point to the next, by about 0.034
// degrees each way (measured against the planes, cushion at 1920x1080: 70% of readings over 0.03 degrees,
// 19% over 0.06, 0.1% over 0.12: a Rayleigh spread of sigma 0.034 degrees). Small as that is, a stone
// shows the lights through it at about the scale of a pixel, and that wander softens the edges of the
// lights seen through it. The exact normal is given the same wander, so that the stones look as they
// always have: without it their highlights come out sharper and about 1% darker on average (round
// brilliant at 1280x720). The wander is drawn afresh for every point a hundred-thousandth of a unit
// apart, finer than a pixel, so that it averages out over the frames as the old reading did.
fn gemHash(v: vec3u) -> vec3u {
  var x = v * 1664525u + 1013904223u;
  x.x += x.y * x.z; x.y += x.z * x.x; x.z += x.x * x.y;
  x ^= x >> vec3u(16u);
  x.x += x.y * x.z; x.y += x.z * x.x; x.z += x.x * x.y;
  return x;
}
fn gemUneven(n: vec3f, q: vec3f) -> vec3f {
  let h = vec3f(gemHash(vec3u(vec3i(floor(q * 1.0e5)) + vec3i(1 << 20)))) / 4294967296.0;
  let r = 0.00059 * sqrt(-2.0 * log(max(h.x, 1e-7)));        // radians: Rayleigh, sigma 0.034 degrees
  let a = 6.2831853 * h.y;
  let t1 = normalize(cross(n, select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(n.x) > 0.7)));
  let t2 = cross(n, t1);
  return normalize(n + r * (cos(a) * t1 + sin(a) * t2));
}
