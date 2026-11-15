// The light inside a body that scatters it (milky glass, jade, wax, porcelain), worked out once into a
// small volume that the tracer then reads. Nothing here is random and nothing depends on where the
// camera is: the volume changes only when the lights, the shape or the material do.
//
// Three steps, each a compute shader over the light volume (a grid much coarser than the shape's own):
//
//   cs_inject   the light that reaches each cell without having been scattered. The lighting is a few
//               dozen beams (scatter.rs makes them from the lights' map and the environment picture);
//               for each, a march from the cell towards the light through the shape's distance field adds
//               up how much body lies in the way. What the beam loses inside the cell is what the cell
//               scatters: summed over the cell's own stretch of the beam, so that light is neither lost
//               nor made where the beam dies away faster than the grid can show.
//   cs_diffuse  light scattered once is the source of light scattered many times, which spreads through
//               the body as heat does (the diffusion approximation: Stam 1995, Jensen et al. 2001), each
//               colour with its own absorption and so its own reach, and leaves through the surface. One
//               relaxation step of that equation; run many times, from a coarse volume up to the fine one.
//   cs_finish   both kinds of light together, and carried a cell or two beyond the surface so that the
//               tracer can read the volume with interpolation right up to it.
//
// The surface is not a staircase of cells: every cell knows what share of it is inside the body, and
// every face between two cells what share of it is open, from the distance field (the smoothed-boundary
// method: Yu, Chen & Thornton 2012). That is what keeps the answer from rippling as a slanted surface
// passes through the grid (tools/scatter_1d.py tries it in one dimension against the exact answer).

struct Beam {
  dir: vec4f,             // xyz: towards the light, in the shape's own frame   w: tangent of its angular radius
  rgb: vec4f,             // the light it carries: brightness times the patch of sky it stands for
}
struct SU {
  res: f32,               // cells along a side of the light volume
  hb: f32,                // the volume spans -hb .. hb
  nb: u32,                // beams in use
  steps: u32,             // steps a march may take
  st: vec4f,              // rgb: what a ray loses per unit length (scattered or absorbed)   w: radius of the shape's bound, squared
  ss: vec4f,              // rgb: of that, what is scattered   w: how far each relaxation step moves a cell towards
                          // what its neighbours say (1 = all the way, more = past it)
  k: vec4f,               // rgb: cell squared over the diffusion constant   w: the body's index of refraction
  zb: vec4f,              // rgb: cell over the distance beyond the surface at which diffuse light would reach nothing
                          // w: the share of light from all round that gets into the body
  ab: vec4f,              // rgb: absorption times cell squared over the diffusion constant
                          // w: what a cell of the furnace's volume should hold (0: no calibration here)
  even: vec4f,            // light that is in no beam: its brightness averaged over all directions, by colour
                          // w: 1 = the body lights itself where one part of it faces another
  leanX: vec4f,           // and how much brighter it is towards x, y and z (brightness around d = even + lean . d)
  leanY: vec4f,
  leanZ: vec4f,
  beams: array<Beam, 20>,
  pick: vec4f,            // which channel of the volume at binding 1 is the body's distance (the solve's own reads)
  pickSrc: vec4f,         // ... and which channel the small copy is made from (cs_shape)
  // What other bodies cast on this one (cast.rs, cast.wgsl): read from the distances to every body at
  // modest detail (binding 14), one body a channel.
  via: vec4f,             // x: 1 = other bodies' shadows are marched for (the body is one of several, each lit by
                          //    its own light volume)   y: 1 = the body is several bodies of one material, and a
                          //    beam is darkened only by the body its cell is in and by those that cast
                          //    z: how far off its start a march takes its first look   w: its shortest stride
  casts: vec4f,           // for each body: 1 = it casts a shadow here (never the body being lit)
  solid: vec4f,           // for each body: 1 = it lets nothing through
  far: vec4f,             // for each body: 0 if it casts, a large distance if not (it does not shorten the strides)
  cbody: array<vec4f, 4>, // each body: rgb = what a unit length of it takes from light passing through, w = what its surfaces let through
  caus: vec4f,            // x: how strong shadows are   y: how strong caustics are   z: cells a side of the volume of
                          // light arriving by way of the bodies (0: none)   w: how far off the surface that light was landed
  mottle: vec4f,          // x: how uneven the body's colour is (0 = even)
}

@group(0) @binding(0) var<uniform> su: SU;
@group(0) @binding(1) var volTex: texture_3d<f32>;          // the shape's distance field
@group(0) @binding(2) var lin: sampler;
@group(0) @binding(3) var srcOut: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var srcTex: texture_3d<f32>;          // rgb: unscattered light in the cell   a: distance to the surface
@group(0) @binding(5) var phiTex: texture_3d<f32>;          // rgb: scattered light   a: distance to the surface
@group(0) @binding(6) var phiOut: texture_storage_3d<rgba32float, write>;
@group(0) @binding(7) var outTex: texture_storage_3d<rgba16float, write>;
// sums over blocks of cells, for the body's balance as a whole (cs_sum, cs_fold, cs_level)
@group(0) @binding(8) var addTex: texture_3d<f32>;
@group(0) @binding(9) var loseTex: texture_3d<f32>;
@group(0) @binding(10) var addOut: texture_storage_3d<rgba32float, write>;
@group(0) @binding(11) var loseOut: texture_storage_3d<rgba32float, write>;
@group(0) @binding(12) var shapeOut: texture_storage_3d<r16float, write>;
@group(0) @binding(13) var belowTex: texture_3d<f32>;       // the light volume of the size below (nothing in it for the smallest)
@group(0) @binding(14) var castTex: texture_3d<f32>;        // the distance to each body, one a channel, at modest detail
@group(0) @binding(15) var castLight: texture_3d<f32>;      // light arriving at the surfaces by way of the bodies (caustics)
@group(0) @binding(16) var calibTex: texture_3d<f32>;       // the same body in a furnace (scatter.rs: Scatter::solve)

fn sdf(p: vec3f) -> f32 { return dot(textureSampleLevel(volTex, lin, p / (2.0 * su.hb) + 0.5, 0.0), su.pick); }
fn castAt(p: vec3f) -> vec4f { return textureSampleLevel(castTex, lin, p / (2.0 * su.hb) + 0.5, 0.0); }

// The distance a beam marches by, for a body that is several bodies of one material while bodies cast
// (su.via.y): only the bodies that keep (keep: one a channel, the cell's own body and those that cast)
// are in the beam's way. Where the nearest body keeps, the body's own fine distance; elsewhere the
// nearest of those that keep, at modest detail (it is further off than the nearest body anyway).
fn sdfKeep(q: vec3f, keep: vec4f) -> f32 {
  let d = sdf(q);
  if (su.via.y < 0.5) { return d; }
  let c = castAt(q);
  var near = 0;
  var dn = c.x;
  for (var k = 1; k < 4; k++) { if (c[k] < dn) { dn = c[k]; near = k; } }
  if (keep[near] > 0.5) { return d; }
  let m = select(vec4f(9.0), c, keep > vec4f(0.5));
  return max(min(min(m.x, m.y), min(m.z, m.w)), d);
}

// How much of a beam (towards w, a light of angular tangent tanA) the other bodies let through to ro, by
// colour: cast.wgsl's march towards a light, through the distances to the bodies, with the body being lit
// left out. A body that lets nothing through hides the light gradually across its outline; one that lets
// some through dims and colours it by the length of the beam inside it.
fn castThrough(ro: vec3f, w: vec3f, tanA: f32) -> vec3f {
  if (su.via.x < 0.5) { return vec3f(1.0); }
  let b = dot(ro, w);
  let h = b * b - (dot(ro, ro) - su.st.w);
  if (h <= 0.0) { return vec3f(1.0); }
  let tFar = -b + sqrt(h);
  if (tFar <= 0.0) { return vec3f(1.0); }
  let wa = max(tanA, 0.01);
  var t = su.via.z / max(wa, 0.2);
  var r = vec4f(1.0);
  var s = vec4f(0.0);
  for (var i = 0; i < 40; i++) {
    let d = castAt(ro + w * t);
    r = min(r, d / (wa * t));
    if (any(r * su.solid <= vec4f(-1.0))) { break; }
    let a = abs(d) + su.far;
    let dt = clamp(min(min(a.x, a.y), min(a.z, a.w)), su.via.w, 0.7);
    s += dt * step(d, vec4f(0.0));
    t += dt;
    if (t > tFar) { break; }
  }
  let x = clamp(0.5 - 0.5 * r, vec4f(0.0), vec4f(1.0));
  let cover = x * x * (3.0 - 2.0 * x) * su.casts;
  var v = mix(vec3f(1.0), exp(-su.cbody[0].rgb * s.x) * su.cbody[0].w, cover.x);
  v *= mix(vec3f(1.0), exp(-su.cbody[1].rgb * s.y) * su.cbody[1].w, cover.y);
  v *= mix(vec3f(1.0), exp(-su.cbody[2].rgb * s.z) * su.cbody[2].w, cover.z);
  v *= mix(vec3f(1.0), exp(-su.cbody[3].rgb * s.w) * su.cbody[3].w, cover.w);
  return mix(vec3f(1.0), v, su.caus.x);
}

// A smaller copy of the shape's distance field, for the beams to march through. The shape's own volume
// is hundreds of megabytes: marching it from every cell towards every light reads all over it, and that
// is what the time goes on. A copy a third of the size a side is a few dozen megabytes, and still finer
// than the cells the light is worked out in.
@compute @workgroup_size(4, 4, 4)
fn cs_shape(@builtin(global_invocation_id) id: vec3u) {
  let n = textureDimensions(shapeOut);
  if (any(id >= n)) { return; }
  textureStore(shapeOut, id, vec4f(dot(textureSampleLevel(volTex, lin, (vec3f(id) + 0.5) / vec3f(n), 0.0), su.pickSrc), 0.0, 0.0, 1.0));
}

// The share of a cell (or of a face between cells) that is inside the body, from the distance to the
// surface at its middle, in cells: all of it half a cell inside, none half a cell outside.
fn share(d: f32) -> f32 { return clamp(0.5 - d, 0.0, 1.0); }

// How much of the body's surface lies in a cell (in cell faces), from the distance to the surface at its
// middle, in cells: a whole face's worth for a cell the surface passes through the middle of, falling to
// none a cell away, so that across any surface the cells' shares add up to the surface. (The face shares
// would say the same of a surface the grid resolves; of a piece of body thinner than a cell they say
// nothing, opposite faces being equally open, and light scattered into such a piece could never leave.)
fn skin(d: f32) -> f32 { return max(1.0 - abs(d), 0.0); }

// Where the six parts of a beam run, across its width (in units of half the width), from the side nearest
// the surface it is passing to the side furthest from it.
const PARTS_A = vec3f(-0.8333, -0.5, -0.1667);
const PARTS_B = vec3f(0.1667, 0.5, 0.8333);
// The most a beam is taken to have widened by (see `way`).
const WIDEST: f32 = 0.05;
// What a part of a beam loses, by colour, between having crossed `before` of body and `more` beyond that.
fn lost(st: vec3f, before: f32, more: f32) -> vec3f { return exp(-st * before) - exp(-st * (before + more)); }

// What a surface between air and the body lets through of light arriving at cosine c (unpolarised).
fn enters(c: f32) -> f32 {
  let n = su.k.w;
  let ct = sqrt(1.0 - (1.0 - c * c) / (n * n));
  let rs = (c - n * ct) / (c + n * ct);
  let rp = (n * c - ct) / (n * c + ct);
  return 1.0 - 0.5 * (rs * rs + rp * rp);
}

// How much body lies along a beam: for each of its six parts, the length of body crossed from `p`
// towards the light (direction w), starting `t0` along; and about where the beam came into the piece of
// body that p is in.
struct Way {
  a: vec3f,
  b: vec3f,
  came: vec3f,
}
// The beam is as wide as a cell where it starts and widens as the light is wide (tanA), and is followed
// as six parts side by side, from the side nearest the surface it is passing to the side furthest from
// it. At each step the distance field says which of the six are inside the body. So the outline of the
// body is soft by the beam's own width, a wide light casts a soft shadow, and a beam that skims past a
// part of the body loses the share of itself that hits it, not all of itself.
// `outside`: the beam starts outside the body; else it starts in the piece of body its cell is in.
// `keep`: see sdfKeep.
fn way(p: vec3f, w: vec3f, tanA: f32, t0: f32, outside: bool, h: f32, least: f32, keep: vec4f) -> Way {
  var o: Way;
  o.a = vec3f(0.0);
  o.b = vec3f(0.0);
  o.came = p;
  var t = t0;
  var known = outside;
  var wasA = vec3f(0.0);
  var wasB = vec3f(0.0);
  var tWas = t;
  var dWas = 0.0;
  var clear = false;                      // the march got out of the shape's bound, or the beam was used up
  for (var i = 0u; i < su.steps; i++) {
    let q = p + w * t;
    if (dot(q, q) > su.st.w) { clear = true; break; }
    let dq = sdfKeep(q, keep);
    if (!known && dq > 0.0) {
      known = true;
      o.came = q;
    }
    // How far apart the beam's parts meet a surface depends on how the beam strikes it: skimming along
    // it, by the beam's whole width; head on, all at once. How it strikes is read off how fast the
    // distance changes along the beam (as fast as the beam travels: head on).
    var across = 1.0;
    if (i > 0u) { across = sqrt(max(1.0 - pow((dq - dWas) / max(t - tWas, 1e-6), 2.0), 0.0)); }
    // Inside the piece of body the cell itself is in, the beam is kept as narrow as the cell: its parts
    // all cross about as much body as its middle does. And it is never taken wider than the distance
    // field can vouch for: far from the surface a shape's volume holds a cautious understatement of the
    // distance, which would read as body in the way of a wide beam.
    let wider = select(0.0, min(t * tanA, WIDEST), known);
    let r = 0.5 * h + wider * across;
    let nowA = clamp(vec3f(0.5) - 3.0 * (dq / r + PARTS_A), vec3f(0.0), vec3f(1.0));
    let nowB = clamp(vec3f(0.5) - 3.0 * (dq / r + PARTS_B), vec3f(0.0), vec3f(1.0));
    if (i == 0u) {
      wasA = nowA;
      wasB = nowB;
    }
    // (The lengths are added up between one step and the next from the shares at both: taking each step
    // at the share it starts with would count half a step too much body at every surface, which in a
    // dense body is a good part of the beam.)
    o.a += 0.5 * (wasA + nowA) * (t - tWas);
    o.b += 0.5 * (wasB + nowB) * (t - tWas);
    wasA = nowA;
    wasB = nowB;
    tWas = t;
    dWas = dq;
    // (well clear of the surface: straight to where the beam's nearest part would first touch it; near
    // it: in steps fine enough to follow each part in and out)
    t += max(abs(dq) - 0.5 * h - wider, 0.35 * r);
    if (least * min(min(o.a.x, min(o.a.y, o.a.z)), min(o.b.x, min(o.b.y, o.b.z))) > 10.0) { clear = true; break; }
  }
  // (out of steps while still in the body, as a beam skimming along just under the surface is: the rest
  // of the way is taken as body too)
  if (!clear && wasA.z > 0.0) {
    o.a += vec3f(20.0 / least);
    o.b += vec3f(20.0 / least);
  }
  return o;
}

// The light the beams bring to each cell, the rest of the lighting, and the body's own light.
//
// A beam's light is put into the cells in one of two ways, by how dense the body is beside the cells.
// Where the cells are fine enough to follow the beam as it dies away inside the body, each cell follows
// it: back from the cell to the light, adding up the body in the way, and takes what the beam loses on
// its own stretch. Where the beam dies within a cell, that cannot be followed, and what a cell would get
// then depends on how the surface happens to cut it; there the beam is put in at the surface instead:
// what arrives on the surface near the cell (marched from just above it, for the shadows of other
// parts), by the angle it strikes at, dying away below the surface along its bent path; the cell takes
// what is lost between its upper and its lower face. In between, some of each.
@compute @workgroup_size(4, 4, 4)
fn cs_inject(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= vec3u(u32(su.res)))) { return; }
  let h = 2.0 * su.hb / su.res;
  let p = ((vec3f(id) + 0.5) / su.res - 0.5) * (2.0 * su.hb);
  let d = sdf(p);
  let st = su.st.rgb;
  let least = min(st.r, min(st.g, st.b));
  var light = vec3f(0.0);
  // only cells that hold some body, and not so deep that nothing unscattered can reach them
  if (d < 0.5 * h && least * (-d - 0.87 * h) < 10.0) {
    let dense = clamp((least * h - 0.4) / 0.8, 0.0, 1.0);
    // the cell's upper and lower face, as depths below the surface
    let top = max(-d - 0.5 * h, 0.0);
    let bottom = max(-d + 0.5 * h, 0.0);
    let shell = least * top < 8.0;
    // which way the surface near the cell faces (the slope of the distance field, from four lookups),
    // and a point just above it
    let e = vec2f(1.0, -1.0) * (0.5 * h);
    let away = normalize(e.xyy * sdf(p + e.xyy) + e.yyx * sdf(p + e.yyx) + e.yxy * sdf(p + e.yxy) + e.xxx * sdf(p + e.xxx) + vec3f(1e-12));
    let above = p + away * (0.75 * h - d);
    // (and a point inside the body, for the march across it to its other face: the cell's middle, or
    // for a cell whose middle lies just outside, a little way below the surface)
    let inside = p - away * max(d + 0.25 * h, 0.0);
    // How far the far face of the piece lies, straight through it: light coming in through that face
    // from direction w has come (as through a slab) this far over the cosine of its slant.
    let farFace = across(inside, -away, h, least);
    // (of a body that is several bodies of one material, which of them the cell is in: the highest-numbered
    // it is inside, or the nearest; it darkens its own beams whatever it casts)
    var keep = su.casts;
    if (su.via.y > 0.5) {
      let c = castAt(p);
      var own = 0;
      var dn = c.x;
      for (var k = 1; k < 4; k++) { if (c[k] < dn) { dn = c[k]; own = k; } }
      for (var k = 0; k < 4; k++) { if (c[k] < 0.0) { own = k; } }
      keep[own] = 1.0;
    }
    for (var b = 0u; b < su.nb; b++) {
      let w = su.beams[b].dir.xyz;
      let tanA = su.beams[b].dir.w;
      // A wide beam arrives at many angles at once: the wider it is, the more it is given what light
      // from all round gets in on average, in place of what gets in at its middle's angle.
      let wide = clamp(1.6 * tanA, 0.0, 1.0);
      let c = dot(away, w);
      if (dense < 1.0) {
        // ---- followed back from the cell
        let m = way(p, w, tanA, 0.5 * h, false, h, least, keep);
        // Inside the cell: its stretch of the beam in four pieces. Each of the beam's parts crosses so
        // much more body there; what a part loses between coming to the cell and leaving it is light
        // that the cell holds. (A part of the beam is inside the body or it is not: a beam passing the
        // body's edge loses the parts that hit it. Taking the edge as thinned-out body across the whole
        // beam instead makes a thin dense piece catch light as if it were wider than it is.)
        var moreA = vec3f(0.0);
        var moreB = vec3f(0.0);
        for (var j = 0; j < 4; j++) {
          let dq = 2.0 * sdf(p + w * ((0.375 - 0.25 * f32(j)) * h)) / h;
          moreA += clamp(vec3f(0.5) - 3.0 * (dq + PARTS_A), vec3f(0.0), vec3f(1.0)) * (0.25 * h);
          moreB += clamp(vec3f(0.5) - 3.0 * (dq + PARTS_B), vec3f(0.0), vec3f(1.0)) * (0.25 * h);
        }
        var got = (lost(st, m.a.x, moreA.x) + lost(st, m.a.y, moreA.y) + lost(st, m.a.z, moreA.z) + lost(st, m.b.x, moreB.x) + lost(st, m.b.y, moreB.y) + lost(st, m.b.z, moreB.z)) / 6.0;
        // How much of the beam the surface let in depends on the angle it arrived at: nearly all of it
        // straight on, less and less towards grazing. The surface it came in by faces along the slope
        // of the distance field there (four lookups).
        if (max(got.x, max(got.y, got.z)) > 0.001) {
          let face = normalize(e.xyy * sdf(m.came + e.xyy) + e.yyx * sdf(m.came + e.yyx) + e.yxy * sdf(m.came + e.yxy) + e.xxx * sdf(m.came + e.xxx) + vec3f(1e-12));
          got *= mix(enters(clamp(dot(face, w), 0.05, 1.0)), su.zb.w, wide);
        }
        light += su.beams[b].rgb.rgb * got * (1.0 - dense) * castThrough(p, w, tanA);
        if (dense > 0.0 && c < -0.03) {
          // ---- put in at the far side: the beam comes into this piece of the body through its other
          // face (in a thin piece the cell lies within reach of both; through a thick one nothing gets
          // across). The march back from the cell has measured the body in the way, this piece and any
          // beyond it; beyond the far face is what the beam crossed before it got here. (Where the body
          // is so dense that no cell is followed back, nothing gets across a piece a cell thick anyway.)
          let fc = -c;
          let s = farFace / fc;
          let ci = sqrt(1.0 - (1.0 - fc * fc) / (su.k.w * su.k.w));
          let half = 0.5 * h / ci;
          if (least * (s - half) < 10.0) {
            let o = max(s - 0.5 * h, 0.0);
            let reaches = (exp(-st * max(m.a.x - o, 0.0)) + exp(-st * max(m.a.y - o, 0.0)) + exp(-st * max(m.a.z - o, 0.0)) + exp(-st * max(m.b.x - o, 0.0)) + exp(-st * max(m.b.y - o, 0.0)) + exp(-st * max(m.b.z - o, 0.0))) / 6.0;
            light += su.beams[b].rgb.rgb * reaches * (dense * fc * mix(enters(fc), su.zb.w, wide)) * (exp(-st * max(s - half, 0.0)) - exp(-st * (s + half))) * castThrough(p, w, tanA);
          }
        }
      }
      if (dense > 0.0 && shell && c > 0.03) {
        // ---- put in at the surface
        let m = way(above, w, tanA, 0.0, true, h, least, keep);
        let reaches = (exp(-st * m.a.x) + exp(-st * m.a.y) + exp(-st * m.a.z) + exp(-st * m.b.x) + exp(-st * m.b.y) + exp(-st * m.b.z)) / 6.0;
        // (inside, the beam is bent towards the surface's facing: it dies away along that steeper path)
        let ci = sqrt(1.0 - (1.0 - c * c) / (su.k.w * su.k.w));
        light += su.beams[b].rgb.rgb * reaches * (dense * c * mix(enters(c), su.zb.w, wide)) * (exp(-st * (top / ci)) - exp(-st * (bottom / ci))) * castThrough(above, w, tanA);
      }
    }
    // ---- light from all round, and from the body itself
    // Two more kinds of light come through the surface near the cell. The rest of the lighting, which
    // is in no beam: faint light from all round, a little stronger from one side. And the body's own
    // light: where one part of the body faces another, each is lit by the light the other gives off
    // (the folds of a white body glow). Both are taken at the surface, looking out in twelve directions.
    // Where a look meets the body again, what arrives from that twelfth of the surroundings is the
    // light the body holds there (as the volume of the size below has it: so every size adds one more
    // bounce); where it gets away, the rest of the lighting from that direction. Each counts by how
    // squarely it strikes the surface, dies away below the surface as light slanting in that way does,
    // and the cell takes what is lost between its upper and its lower face.
    if (shell) {
      // (twelve directions do not weigh a half-space quite evenly whichever way it faces: their
      // weights are brought to what all directions together would come to)
      var squarely = 0.0;
      for (var b = 0; b < 12; b++) { squarely += max(dot(ROUND[b], away), 0.0); }
      for (var b = 0; b < 12; b++) {
        let w = ROUND[b];
        let c = dot(w, away);
        // Light from this direction comes in through the face near the cell (it faces that way) or,
        // in a piece thin enough for it to get across, through the face on the other side.
        var fc = c;
        var start = above;
        var sTop = top / max(c, 0.03);
        var sBottom = bottom / max(c, 0.03);
        if (c < 0.03) {
          if (c > -0.03) { continue; }
          fc = -c;
          let s = farFace / fc;
          if (least * (s - 0.5 * h / fc) > 10.0) { continue; }
          start = inside + w * (s + 0.75 * h);
          sTop = max(s - 0.5 * h / fc, 0.0);
          sBottom = s + 0.5 * h / fc;
        }
        var t = 0.0;
        var met = false;
        // (whether the look out meets the body again is asked only on the near side: light that has
        // crossed the whole piece is faint, and is taken to come from the surroundings)
        let ask = c >= 0.03;
        for (var i = 0; i < select(0, 28, ask); i++) {
          let q = start + w * t;
          if (dot(q, q) > su.st.w) { break; }
          let dq = sdf(q);
          if (dq < 0.0) {
            met = su.even.w > 0.5;
            break;
          }
          t += max(dq, 0.35 * h);
        }
        var comes = max(su.even.rgb + su.leanX.rgb * w.x + su.leanY.rgb * w.y + su.leanZ.rgb * w.z, vec3f(0.0));
        if (met) {
          // what a point of the body gives off: of the light it holds, the share that is scattered, evenly
          // into all directions, thinned by the square of the index on leaving
          comes = textureSampleLevel(belowTex, lin, (start + w * (t + h)) / (2.0 * su.hb) + 0.5, 0.0).rgb * (su.ss.rgb / st) * (0.0795775 / (su.k.w * su.k.w));
        }
        light += comes * (3.14159265 * fc / squarely * su.zb.w) * (exp(-st * sTop) - exp(-st * sBottom));
      }
      // Light that a clear body has bent and focused on to the surface here (caustics: cast_light.wgsl
      // lands it on the surfaces that gather light, a little way off them). It is counted as the light from
      // all round is: what a white matte surface would show of it, from every direction of the outside
      // half at once, getting in and dying away below the surface as light slanting in at 45 degrees does.
      if (su.caus.z > 1.5) {
        let at = p + away * (su.caus.w - d);
        let arriving = textureSampleLevel(castLight, lin, at / (2.0 * su.hb) + 0.5, 0.0).rgb * su.caus.y;
        light += arriving * (3.14159265 * su.zb.w) * (exp(-st * (top / 0.7071)) - exp(-st * (bottom / 0.7071)));
      }
    }
    // (what was lost from the beams, as light per unit volume of the whole cell)
    light /= st * h;
  }
  textureStore(srcOut, id, vec4f(light, d));
}

// How far the body reaches from p (inside it) along w: where a ray from p that way would leave it.
// (Not far is all that matters: past ten distances between scatterings nothing gets across anyway.)
fn across(p: vec3f, w: vec3f, h: f32, least: f32) -> f32 {
  var t = 0.0;
  for (var i = 0; i < 10; i++) {
    let d = sdf(p + w * t);
    if (d > 0.0) { return t; }
    t += max(-d, 0.5 * h);
    if (least * t > 10.0) { break; }
  }
  return t;
}

// ---- uneven colour ------------------------------------------------------------------------------------
// No stone or wax is the same colour all through: jade has paler cloudy patches and fine pale veins,
// alabaster and wax are a little blotchy. The light the body holds is made uneven in the same way, fixed
// to the shape (the light volume lies in the shape's own frame), as paler and deeper clouds at two sizes
// and thin veins that wander through them. (The surface's own wear is the tracer's: trace_material.wgsl.)
fn mHash(i: vec3i) -> f32 {
  var h = (u32(i.x) * 73856093u) ^ (u32(i.y) * 19349663u) ^ (u32(i.z) * 83492791u);
  h = (h ^ (h >> 16u)) * 0x45d9f3bu;
  h = (h ^ (h >> 16u)) * 0x45d9f3bu;
  h = h ^ (h >> 16u);
  return f32(h & 0xffffffu) * (2.0 / 16777215.0) - 1.0;
}
fn mNoise(x: vec3f) -> f32 {
  let i = vec3i(floor(x));
  let f = fract(x);
  let s = f * f * (3.0 - 2.0 * f);
  let a = mix(mix(mHash(i), mHash(i + vec3i(1, 0, 0)), s.x), mix(mHash(i + vec3i(0, 1, 0)), mHash(i + vec3i(1, 1, 0)), s.x), s.y);
  let b = mix(mix(mHash(i + vec3i(0, 0, 1)), mHash(i + vec3i(1, 0, 1)), s.x), mix(mHash(i + vec3i(0, 1, 1)), mHash(i + vec3i(1, 1, 1)), s.x), s.y);
  return mix(a, b, s.z);
}
fn mottled(v: vec3f, q: vec3f, amount: f32) -> vec3f {
  let cloud = 0.65 * mNoise(q * 2.2 + vec3f(3.1, 7.7, 1.9)) + 0.35 * mNoise(q * 5.3 + vec3f(9.2, 2.6, 5.5));
  let warp = vec3f(mNoise(q * 1.3 + vec3f(1.0, 2.0, 3.0)), mNoise(q * 1.3 + vec3f(4.0, 5.0, 6.0)), mNoise(q * 1.3 + vec3f(7.0, 8.0, 9.0)));
  let vein = smoothstep(0.86, 0.97, 1.0 - abs(mNoise(q * 3.4 + 0.8 * warp)));
  let grey = vec3f(dot(v, vec3f(0.2126, 0.7152, 0.0722)));
  // paler, greyer where the clouds are light and along the veins; deeper and richer where they are dark
  let pale = clamp(amount * (0.45 * max(cloud, 0.0) + 0.5 * vein), 0.0, 1.0);
  var o = mix(v, grey * 1.15, pale);
  o *= 1.0 + amount * (0.3 * cloud + 0.25 * vein);
  return max(o, vec3f(0.0));
}

// Twelve directions spread evenly (the corners of an icosahedron): the ways a cell looks for other parts
// of the body that light it.
const ROUND = array<vec3f, 12>(
  vec3f(0.0, 0.525731, 0.850651), vec3f(0.0, -0.525731, 0.850651), vec3f(0.0, 0.525731, -0.850651), vec3f(0.0, -0.525731, -0.850651),
  vec3f(0.525731, 0.850651, 0.0), vec3f(-0.525731, 0.850651, 0.0), vec3f(0.525731, -0.850651, 0.0), vec3f(-0.525731, -0.850651, 0.0),
  vec3f(0.850651, 0.0, 0.525731), vec3f(0.850651, 0.0, -0.525731), vec3f(-0.850651, 0.0, 0.525731), vec3f(-0.850651, 0.0, -0.525731));

// The six cells that share a face with a cell.
const NEAR = array<vec3i, 6>(vec3i(1, 0, 0), vec3i(-1, 0, 0), vec3i(0, 1, 0), vec3i(0, -1, 0), vec3i(0, 0, 1), vec3i(0, 0, -1));

// One step towards the scattered light's equilibrium. In each cell: what flows in through its open
// faces from its neighbours, plus what the beams' light scattered here adds, against what flows out
// to the neighbours, what leaves through the body's surface inside the cell, and what is absorbed.
// The cells are taken as the two colours of a chessboard, turn about: a step moves the cells of one
// colour, using the other colour's latest values (each cell's six neighbours are all of the other
// colour), and may move them past what the neighbours say by a set amount, which on a small volume gets
// to equilibrium in a number of steps like its size instead of its size squared.
fn relax(id: vec3u, colour: u32) {
  if (any(id >= vec3u(u32(su.res)))) { return; }
  let n = i32(su.res);
  let me = vec3i(id);
  let c = textureLoad(phiTex, me, 0);
  if (((id.x + id.y + id.z) & 1u) != colour) {
    textureStore(phiOut, id, c);
    return;
  }
  let src = textureLoad(srcTex, me, 0);
  let h = 2.0 * su.hb / su.res;
  let d = src.a / h;
  var open = 0.0;
  var flow = vec3f(0.0);
  for (var k = 0; k < 6; k++) {
    let q = me + NEAR[k];
    if (any(q < vec3i(0)) || any(q >= vec3i(n))) { continue; }
    let o = textureLoad(phiTex, q, 0);
    let a = share(0.5 * (d + o.a));
    open += a;
    flow += a * o.rgb;
  }
  if (open <= 0.0) {
    textureStore(phiOut, id, vec4f(0.0, 0.0, 0.0, d));
    return;
  }
  let surface = skin(d);
  let f = share(d);
  let next = (flow + su.k.rgb * su.ss.rgb * src.rgb) / (open + surface * su.zb.rgb + su.ab.rgb * f);
  textureStore(phiOut, id, vec4f(max(mix(c.rgb, next, su.ss.w), vec3f(0.0)), d));
}
@compute @workgroup_size(4, 4, 4)
fn cs_even(@builtin(global_invocation_id) id: vec3u) { relax(id, 0u); }
@compute @workgroup_size(4, 4, 4)
fn cs_odd(@builtin(global_invocation_id) id: vec3u) { relax(id, 1u); }

// Begin a finer volume from the coarser one below it (read with interpolation).
@compute @workgroup_size(4, 4, 4)
fn cs_refine(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= vec3u(u32(su.res)))) { return; }
  let h = 2.0 * su.hb / su.res;
  let d = textureLoad(srcTex, vec3i(id), 0).a / h;
  var v = vec3f(0.0);
  if (d < 1.5) { v = textureSampleLevel(phiTex, lin, (vec3f(id) + 0.5) / su.res, 0.0).rgb; }
  textureStore(phiOut, id, vec4f(v, d));
}

// Begin a volume with nothing in it but each cell's distance to the surface.
@compute @workgroup_size(4, 4, 4)
fn cs_begin(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= vec3u(u32(su.res)))) { return; }
  let h = 2.0 * su.hb / su.res;
  textureStore(phiOut, id, vec4f(0.0, 0.0, 0.0, textureLoad(srcTex, vec3i(id), 0).a / h));
}

// ---- the body's balance as a whole ---------------------------------------------------------------------
// Relaxation settles differences between neighbours quickly and the level of the whole body slowly:
// in a body that absorbs little, light leaves only through the surface, and a step moves the level by
// about the share of the cells that lie on the surface. But the level can be had directly. When the
// scattered light is in equilibrium, what the beams add in the whole body equals what leaves through
// the whole surface and what is absorbed; both are sums over the cells, and both are in proportion to
// the level. So: add the two up (cs_sum over blocks of cells, cs_fold over blocks of blocks, until one
// number a colour is left), and multiply the scattered light by their ratio (cs_level).

// one block of 4 x 4 x 4 cells
@compute @workgroup_size(1)
fn cs_sum(@builtin(global_invocation_id) id: vec3u) {
  let n = i32(su.res);
  var add = vec3f(0.0);
  var lose = vec3f(0.0);
  for (var j = 0; j < 64; j++) {
    let me = vec3i(id) * 4 + vec3i(j % 4, (j / 4) % 4, j / 16);
    if (any(me >= vec3i(n))) { continue; }
    let c = textureLoad(phiTex, me, 0);
    var open = 0.0;
    for (var k = 0; k < 6; k++) {
      let q = me + NEAR[k];
      if (any(q < vec3i(0)) || any(q >= vec3i(n))) { continue; }
      open += share(0.5 * (c.a + textureLoad(phiTex, q, 0).a));
    }
    if (open <= 0.0) { continue; }
    add += su.k.rgb * su.ss.rgb * textureLoad(srcTex, me, 0).rgb;
    lose += (skin(c.a) * su.zb.rgb + su.ab.rgb * share(c.a)) * c.rgb;
  }
  textureStore(addOut, id, vec4f(add, 0.0));
  textureStore(loseOut, id, vec4f(lose, 0.0));
}

// one block of 8 x 8 x 8 sums
@compute @workgroup_size(1)
fn cs_fold(@builtin(global_invocation_id) id: vec3u) {
  let n = vec3i(textureDimensions(addTex));
  var add = vec3f(0.0);
  var lose = vec3f(0.0);
  for (var j = 0; j < 512; j++) {
    let q = vec3i(id) * 8 + vec3i(j % 8, (j / 8) % 8, j / 64);
    if (any(q >= n)) { continue; }
    add += textureLoad(addTex, q, 0).rgb;
    lose += textureLoad(loseTex, q, 0).rgb;
  }
  textureStore(addOut, id, vec4f(add, 0.0));
  textureStore(loseOut, id, vec4f(lose, 0.0));
}

// the scattered light, brought to the level at which the whole body is in balance
@compute @workgroup_size(4, 4, 4)
fn cs_level(@builtin(global_invocation_id) id: vec3u) {
  let c = textureLoad(phiTex, vec3i(id), 0);
  let add = textureLoad(addTex, vec3i(0), 0).rgb;
  let lose = textureLoad(loseTex, vec3i(0), 0).rgb;
  // (nothing to go by where there is no light yet)
  let by = select(vec3f(1.0), clamp(add / max(lose, vec3f(1e-30)), vec3f(0.01), vec3f(100.0)), lose > vec3f(0.0));
  textureStore(phiOut, id, vec4f(c.rgb * by, c.a));
}

// What the tracer reads: both kinds of light together, in the part of each cell that is body. The beams'
// light is kept as what a whole cell holds; deep inside the body that is the light in the body. Near the
// surface a cell is part body and part not, by a share read off the distance at its middle, while the
// beams' light was added up along their own lines through the cell: cell by cell the two do not quite
// agree, and the tracer, which reads the volume right at the surface, would show that as a ripple along
// thin or slanted pieces. So near the surface, and in the cells just outside the body which the tracer's
// interpolation reaches into, a cell is given what the body holds in the cells around it: all their
// light over all their body.
@compute @workgroup_size(4, 4, 4)
fn cs_finish(@builtin(global_invocation_id) id: vec3u) {
  if (any(id >= vec3u(u32(su.res)))) { return; }
  let n = i32(su.res);
  let me = vec3i(id);
  let h = 2.0 * su.hb / su.res;
  let mine = textureLoad(srcTex, me, 0);
  let d = mine.a / h;
  var v = vec3f(0.0);
  if (d < -1.5) { v = textureLoad(phiTex, me, 0).rgb + mine.rgb; }
  if (d > -2.5 && d < 3.0) {
    var light = vec3f(0.0);
    var body = 0.0;
    for (var k = 0; k < 27; k++) {
      let o = vec3i(k % 3, (k / 3) % 3, k / 9) - 1;
      let q = me + o;
      if (any(q < vec3i(0)) || any(q >= vec3i(n))) { continue; }
      let s = textureLoad(srcTex, q, 0);
      let f = share(s.a / h);
      if (f <= 0.0) { continue; }
      // (nearer cells count for more)
      let w = 1.0 / f32(1 + abs(o.x) + abs(o.y) + abs(o.z));
      light += w * (f * textureLoad(phiTex, q, 0).rgb + s.rgb);
      body += w * f;
    }
    var near = textureLoad(phiTex, me, 0).rgb;      // (no body next to it: the scattered light as the relaxation carried it out here)
    if (body > 0.0) { near = light / body; }
    // (from a cell and a half below the surface to two and a half, the one way gives over to the other)
    v = mix(near, v, clamp(-1.5 - d, 0.0, 1.0));
  }
  // The calibration: each cell as a share of what it held in the furnace against what it should have
  // held there. (Kept within bounds: where the furnace's cell holds next to nothing, outside the body,
  // its share says nothing.)
  if (su.ab.w > 0.0) {
    let f = textureLoad(calibTex, me, 0).g;
    v *= clamp(su.ab.w / max(f, 1e-6), 0.5, 3.0);
  }
  if (su.mottle.x > 0.0) { v = mottled(v, ((vec3f(id) + 0.5) / su.res - 0.5) * (2.0 * su.hb), su.mottle.x); }
  textureStore(outTex, id, vec4f(max(v, vec3f(0.0)), mine.a));
}
