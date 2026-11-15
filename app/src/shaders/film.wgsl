
//#if FILM
// ---- thin films: where the film is how thick, what it mirrors, and a film around air ------------------
// (Joined to the materials' text when the tracer is built, and cut by the same marked lines: all of it
// is left out unless a film is in use. The film's optics, filmR and its kin, are with the other
// materials; here is everything that makes a film look like one.)
//
// What makes a film read as a film is less its colour than what it shows: a film is a faint mirror, and
// its colours are only seen where it has something bright and sharp to mirror; and its thickness is
// never even, but ordered by gravity. So this part gives the film
//   - a thickness that drains: level bands, thinner upwards, a sink at every summit ringed by tight
//     bands, and (for a free film of liquid) black film where it has thinned through, with a sharp
//     edge and droplets left behind in it;
//   - the surroundings read sharp: the environment picture itself, at the fineness the mirrored pixel
//     covers, not the softened copy made for the glass; and, when there is no picture, a studio of
//     soft boxes that only the film sees;
//   - for a film around air (a soap bubble), a tracer of its own: both walls mirror, nothing bends,
//     and what is seen straight through is what is behind the bubble.

// Three numbers in 0..1 for a cell of space.
fn filmCell(i: vec3i) -> vec3f {
  var h = (u32(i.x) * 0x8da6b343u) ^ (u32(i.y) * 0xd8163841u) ^ (u32(i.z) * 0xcb1ab31fu);
  h = (h ^ (h >> 15u)) * 0x2c1b3c6du;
  h = (h ^ (h >> 12u)) * 0x297a2d39u;
  h = h ^ (h >> 15u);
  return vec3f(f32(h & 1023u), f32((h >> 10u) & 1023u), f32((h >> 20u) & 1023u)) / 1023.0;
}

//#if HOLLOW
// Droplets in black film: one to a cell of space, somewhere well inside it, of its own size. q: the
// point, in cells. Returns 1 at a droplet's middle falling to 0 at its edge.
fn filmDots(q: vec3f) -> f32 {
  let i = floor(q);
  let r = filmCell(vec3i(i));
  let rad = 0.07 + 0.2 * fract(r.x * 5.0 + r.y * 3.0 + r.z * 7.0);
  return clamp(1.0 - length(q - i - (0.3 + 0.4 * r)) / rad, 0.0, 1.0);
}
//#endif

// The pattern of the film's unevenness at q (a point in the shape's own frame, times the pattern's
// fineness), -1 to 1. Liquid in a film is stirred like cream in coffee: sheets of thicker and thinner
// film folded into each other, drawn out flat (gravity keeps them from standing on end). So: a few
// smooth waves, bent twice by other waves, taken across a space squeezed from above. ph: where the
// pattern is in its flow, as an angle; every wave moves by a whole number of turns as it goes round
// once, so the flow comes back to where it began.
fn filmWave(p: vec3f) -> f32 { return dot(sin(p), cos(p.yzx)); }
fn filmSwirl(q0: vec3f, ph: f32) -> f32 {
  var q = q0 * vec3f(1.0, 1.8, 1.0);
  q += 0.5 * vec3f(filmWave(q * 0.9 + vec3f(ph, 0.3, -ph)), filmWave(q.zxy * 1.1 + vec3f(1.7, ph, 0.3)), filmWave(q.yzx * 0.8 + vec3f(-ph, 2.9, ph)));
  q += 0.22 * vec3f(filmWave(q * 2.3 + vec3f(4.1, -ph, 0.0)), filmWave(q.zxy * 2.6 + vec3f(ph, 5.3, 1.1)), filmWave(q.yzx * 2.1 + vec3f(0.7, 0.0, 2.0 * ph)));
  return clamp(0.4 * filmWave(q * 1.3 + vec3f(0.0, ph, 0.0)) + 0.2 * filmWave(q * 2.9 + vec3f(3.3, 0.0, -ph)) + 0.1 * filmWave(q * 6.1 + vec3f(ph, 1.9, 0.0)), -1.0, 1.0);
}

// The film's thickness in nanometres at a point p of the surface whose outward normal is n.
//
// Draining (mu.film3, mu.film4: worked out from the `filmDrain` slider in film.rs, where the reasons
// for each number are). Liquid runs down, so the film is thinner the higher it is: level bands of
// colour. "Up" is the room's up, the lights' own frame, not the shape's: the bands stay level while
// the shape turns, and turn with the lights. Where the surface is a summit (its normal points
// straight up) the film is thinnest of all, and its thickness climbs steeply round about: tight
// rings, one for every 200 nm or so. The whole film thins as it drains, and the sink widens.
fn filmNm(p: vec3f, n: vec3f) -> f32 {
  let v = filmSwirl(toObj(p) * mu.film.w, 6.2831853 * mu.film2.x);
  let up = vec3f(u.env0.y, u.env1.y, u.env2.y) * mu.film4.w;
  let h = clamp(dot(p, up) / 1.2, -1.0, 1.0);
  // how far round from the summit the surface faces (radians, near enough)
  let s = sqrt(max(2.0 - 2.0 * dot(n, up), 0.0));
  var d = mu.film3.x * (1.0 + mu.film.z * v) * (1.0 - mu.film3.y * (0.5 * h + 0.5));
  d *= 1.0 - mu.film4.x * exp(-max(s - mu.film3.z * (1.0 + 0.15 * v), 0.0) * mu.film3.w);
//#if HOLLOW
  // A free film of liquid does not thin to nothing: it ends as "black film", a few molecules thick,
  // which mirrors almost nothing. Where the film round it is still thick its edge is sharp (the
  // thickness climbs steeply there); where the whole film has grown thin it shades off. Black film,
  // and the film that is nearly black, is strewn with droplets of thicker film that were left behind.
  if (d < mu.film4.y && isHollow()) {
    let o = toObj(p);
    let dots = max(max(filmDots(o * 34.0), filmDots(o * 79.0 + 3.7)), 0.8 * filmDots(o * 15.0 + 1.3));
    d = max(d, 6.0) + 190.0 * dots * dots * smoothstep(mu.film4.y, 0.5 * mu.film4.y, d);
  }
//#endif
  return max(d, 0.0);
}

// The thickness again for the further pieces of a solid body, met after a ray has passed through it:
// the pattern on the film's mean thickness, without the draining. Those pieces are seen through the
// body, bent and small, where the draining's bands would not be made out; and what is worked out for
// them is worked out in the course of the marching, where every number that is also read before it
// slows the whole frame (so this reads the thickness from a place of its own: see skyFar).
fn filmFar(p: vec3f) -> f32 {
  return max(mu.film5.x * (1.0 + mu.film.z * filmSwirl(toObj(p) * mu.film.w, 6.2831853 * mu.film2.x)), 0.0);
}

// A soft box of light seen along d: c points at its middle, r and v run across it and up it (all unit),
// half: the tangents of its half-width and half-height, e: how soft its edge is.
fn filmBox(d: vec3f, c: vec3f, r: vec3f, v: vec3f, half: vec2f, e: f32) -> f32 {
  let z = dot(d, c);
  let a = abs(vec2f(dot(d, r), dot(d, v))) / max(z, 1e-3);
  let m = 1.0 - smoothstep(half - e, half + e, a);
  return m.x * m.y * step(0.0, z);
}

// The film's studio, as bright as 1 at its brightest (d: a direction in the room's frame). A film
// mirrors a few percent of what it faces, so it shows colour only where what it faces is bright, and
// reads as a film only where that has edges: an even glow makes a painted ball of it. A soap film is
// photographed in a studio for that reason, and this is one:
//   - a glow from above that fades out towards the horizon (it is what shows the film's colours over
//     the upper part of a shape; below the horizon it is dark, so that the two mirror images in a
//     bubble, the near wall's upright and the far wall's upside down, both lie on black);
//   - a large soft box overhead with a clean edge, a smaller one at the camera's height off to its
//     right (a bubble shows each of them twice), and two tall panels behind to the sides, which
//     light the rim.
// Only light the film reflects sees any of it; with an environment picture the film mirrors the
// picture instead.
fn filmStudio(d: vec3f) -> f32 {
  let g = smoothstep(-0.05, 0.6, d.y);
  let glow = 0.55 * g * sqrt(g);
  let top = filmBox(d, vec3f(0.0, 0.9701, -0.2425), vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.2425, 0.9701), vec2f(0.84, 0.53), 0.012);
  let side = filmBox(d, vec3f(0.5736, 0.0, 0.8192), vec3f(0.8192, 0.0, -0.5736), vec3f(0.0, 1.0, 0.0), vec2f(0.36, 0.2), 0.012);
  let back = filmBox(d, vec3f(-0.6428, 0.0, -0.766), vec3f(0.766, 0.0, -0.6428), vec3f(0.0, 1.0, 0.0), vec2f(0.3, 0.75), 0.02)
           + filmBox(d, vec3f(0.6428, 0.0, -0.766), vec3f(0.766, 0.0, 0.6428), vec3f(0.0, 1.0, 0.0), vec2f(0.3, 0.75), 0.02);
  return max(glow + 0.45 * top, 0.6 * back) + 0.8 * side;
}
// The studio's glow alone, by how far up a direction points.
fn filmGlow(y: f32) -> f32 {
  let g = smoothstep(-0.05, 0.6, y);
  return 0.55 * g * sqrt(g);
}
// What the film's studio adds to light the film reflects along d (as the camera sees directions):
//   skyOf    the whole studio, for the surface the camera meets and for a film around air;
//   skyDim   its glow alone, for light that leaves a solid body after the film has sent it back inside;
//   skyFar   the same, for the further pieces of the body met after that.
// The last two are seen through the body, where the boxes' edges would be lost anyway, and they are
// worked out in the course of the marching, where every further thing to work out costs far more than
// its arithmetic (docs/materials.md). skyFar also reads the studio's brightness from a second place in
// the block, deliberately: a number read both before the marching and in the course of it is kept in
// hand across it by the shader compiler, and one more number in hand there slows the whole frame.
//#if PICTURE
fn skyOf(d: vec3f) -> vec3f { return vec3f(0.0); }
fn skyDim(d: vec3f) -> vec3f { return vec3f(0.0); }
fn skyFar(d: vec3f) -> vec3f { return vec3f(0.0); }
//#else
fn skyOf(d: vec3f) -> vec3f { return vec3f(mu.more.x * filmStudio(toEnv(d))); }
fn skyDim(d: vec3f) -> vec3f { return vec3f(mu.more.x * filmGlow(dot(d, vec3f(u.env0.y, u.env1.y, u.env2.y)))); }
fn skyFar(d: vec3f) -> vec3f { return vec3f(mu.exo.w * filmGlow(dot(d, vec3f(u.env0.y, u.env1.y, u.env2.y)))); }
//#endif

//#if PICTURE
// How soft a copy of the environment picture a mirror reads at p, where the surface faces n and the
// ray met it at cosine c: the level whose softening is as wide as the patch of surroundings that one
// pixel's worth of surface mirrors. A flat mirror shows the picture pixel for pixel; a curved one
// squeezes it, by twice the surface's curvature times the distance (and more the more glancing the
// ray), and read at full sharpness the squeezed picture would sparkle from frame to frame.
// The curvature is measured from the distance field: three lookups a few voxels out across the
// surface, a third of a turn apart, give its greatest bend whichever way that lies.
fn filmLod(p: vec3f, n: vec3f, c: f32) -> f32 {
  let h = 3.0 * (2.0 * HB / u.vres);
  // any two directions across n (Duff et al. 2017)
  let sg = select(-1.0, 1.0, n.z >= 0.0);
  let k = -1.0 / (sg + n.z);
  let m = n.x * n.y * k;
  let t1 = vec3f(1.0 + sg * n.x * n.x * k, sg * m, -sg * n.x);
  let t2 = vec3f(m, sg + n.y * n.y * k, -n.y);
  let d0 = mapW(p);
  let k1 = mapW(p + h * t1) - d0;
  let k2 = mapW(p + h * (-0.5 * t1 + 0.8660254 * t2)) - d0;
  let k3 = mapW(p + h * (-0.5 * t1 - 0.8660254 * t2)) - d0;
  // (each is half the bend that way times h squared; their mean is the mean bend, their scatter the rest)
  let mean = (k1 + k2 + k3) / 3.0;
  let dev = sqrt(0.6666667 * ((k1 - mean) * (k1 - mean) + (k2 - mean) * (k2 - mean) + (k3 - mean) * (k3 - mean)));
  let bend = min((abs(mean) + dev) * 2.0 / (h * h), 1.0 / h);
  let far = length(p - vec3f(0.0, 0.0, u.dist));
  let pixel = 2.0 * u.tan_fov / u.res.y;
  let wide = pixel * (1.0 + 2.0 * bend * far / max(c, 0.25));
  return clamp(log2(0.4 * wide / mu.pic.y), 0.0, mu.pic.w);
}
// The surroundings as a mirror-like film shows them along rd: the picture itself at that level; and,
// when the family's lights are kept lit beside the picture, the lights' map. (Otherwise that map holds
// only the soft glow that stands in for the picture's sun in the glass. A mirror wants the sun itself,
// small and sharp, and the picture has it.)
fn filmEnv(rd: vec3f, lod: f32) -> vec3f {
  let d = toEnv(rd);
  var c = textureSampleLevel(pictureTex, pictureSmp, d, lod).rgb * mu.pic.x;
  if (mu.film5.y > 0.5) {
    let uv = vec2f(atan2(d.z, d.x) * 0.15915494 + 0.5, acos(clamp(d.y, -1.0, 1.0)) * 0.31830989);
    c += textureSampleLevel(envTex, envSmp, uv, 0.0).rgb;
  }
  return c;
}
//#else
fn filmLod(p: vec3f, n: vec3f, c: f32) -> f32 { return 0.0; }
fn filmEnv(rd: vec3f, lod: f32) -> vec3f { return env(rd); }
//#endif

//#if BUBBLE
// ---- a film around air: a soap bubble --------------------------------------------------------------
// The wall is the film itself. Nothing bends on the way through, so each wall does two things only:
// it mirrors (the near wall as a bulging mirror, the far wall, met from inside, as a hollow one, which
// is why a bubble carries its surroundings twice, once upright and once upside down), and it lets the
// rest pass. What passes straight through both walls is what the camera would see if the bubble were
// not there: the backdrop, black unless the environment picture is shown. That is what makes a bubble
// a nearly invisible skin and not a ball: only its mirror images are on it.

// The film's reflectance as a colour (of white light, white = 1), for light meeting it at cosine c
// where it is d nm thick: weighed at twelve wavelengths spread evenly through the spectrum, the whole
// comb shifted by xi from pixel to pixel and frame to frame.
fn filmCol(c: f32, d: f32, xi: f32) -> vec3f {
  var col = vec3f(0.0);
  var ws = vec3f(0.0);
  for (var j = 0; j < 12; j++) {
    let lam = 400.0 + 25.0 * (f32(j) + xi);
    let w = specW(lam);
    col += w * filmR(c, lam, d, 1.0, mu.film.y, 1.0);
    ws += w;
  }
  // (held to what a screen's red, green and blue can say: a purer colour than that would come out as
  // less than none of one of them, or more than all, and a wall that passes more than all of the
  // light makes light when several stand behind one another)
  return clamp(col / ws, vec3f(0.0), vec3f(1.0));
}

// What the camera would see along rd if nothing were there.
// (by direction alone: the backdrop, or black; a floor is not seen straight through a thin wall)
fn filmBehind(rd: vec3f) -> vec3f { return {{BACKGROUND_DIR}}; }

// A ray has left a wall at ro heading rd: on through whatever further pieces of the film lie along it
// (each adds its near wall's mirror image and keeps its share), to what is finally seen. lod: below
// nothing for light that has come unbent from the camera, which ends on the backdrop; otherwise the
// light has been mirrored, ends on the surroundings, and this is the level they are read at.
fn beyondB(ro_in: vec3f, rd: vec3f, lod: f32, xi: f32) -> vec3f {
  var ro = ro_in;
  var col = vec3f(0.0);
  var thr = vec3f(1.0);
  for (var l = 0; l < LAYERS; l++) {
    let b = dot(ro, rd);
    let h = b * b - (dot(ro, ro) - BOUND * BOUND);
    if (h <= 0.0) { break; }
    let tFar = -b + sqrt(h);
    var t = 0.0;
    var hit = false;
    for (var i = 0; i < STEPS_OUT; i++) {
      let d = mapW(ro + rd * t);
      if (d < 0.0008) { hit = true; break; }
      t += d * STEP_SCALE;
      if (t > tFar) { break; }
    }
    if (!hit) { break; }
    t += mapNW(ro + rd * t);
    let p = ro + rd * t;
    let n = nrm(p);
    let ci = clamp(-dot(rd, n), 0.0, 1.0);
    let F = filmCol(ci, filmNm(p, n), xi);
    let mir = reflect(rd, n);
//#if ROUGH
    col += thr * F * (seen(mir, spreadR()) + skyOf(mir));
//#else
    col += thr * F * (filmEnv(mir, filmLod(p, n, ci)) + skyOf(mir));
//#endif
    // across the air inside, and out by the wall beyond
    let q0 = p - n * 0.004;
    let s = marchIn(q0, rd);
    let q = q0 + rd * s;
    var m = nrm(q);
    var co = dot(rd, m);
    if (co < 0.0) { m = -m; co = -co; }
    thr *= (1.0 - F) * (1.0 - filmCol(co, filmNm(q, m), xi)) * mu.axis.xyz * mu.axis.xyz;
//#if GLOW
    col += thr * mu.glow.rgb * s;
//#endif
    ro = q + m * 0.006;
  }
  if (lod < 0.0) { return col + thr * filmBehind(rd); }
//#if ROUGH
  return col + thr * (seen(rd, spreadR()) + skyOf(rd));
//#else
  return col + thr * (filmEnv(rd, lod) + skyOf(rd));
//#endif
}

fn shadeM(ro: vec3f, rd: vec3f, t: f32, xi: f32, W: i32, fc: vec2f) -> vec3f {
  let p = ro + rd * t;
  let n = nrm(p);
  let cosi = clamp(-dot(rd, n), 0.0, 1.0);
  // the near wall: its mirror image, and what it lets in
  var F = filmCol(cosi, filmNm(p, n), xi);
  let mir = reflect(rd, n);
//#if ROUGH
  var acc = F * (seen(mir, spreadR()) + skyOf(mir));
//#else
  var acc = F * (filmEnv(mir, filmLod(p, n, cosi)) + skyOf(mir));
//#endif
  var thr = (1.0 - F) * mu.axis.xyz;
  var pos = p - n * 0.004;
  var dir = rd;
  for (var b = 0; b < MAX_NB; b++) {
    if (b >= u.nb) { break; }
    // across the air inside to the next wall
    let s = marchIn(pos, dir);
    let q = pos + dir * s;
    var m = nrm(q);
    var c = dot(dir, m);
    if (c < 0.0) { m = -m; c = -c; }
//#if GLOW
    acc += thr * mu.glow.rgb * s;
//#endif
    F = filmCol(c, filmNm(q, m), xi);
    // What leaves through this wall. The first time, that is light that has come straight through the
    // bubble; after that it has been mirrored by a wall on the way, and it sees the surroundings.
    var lod = -1.0;
    if (b > 0) { lod = filmLod(q, m, c); }
    acc += thr * (1.0 - F) * mu.axis.xyz * beyondB(q + m * 0.006, dir, lod, xi);
    // and what the wall sends back across the bubble
    thr *= F;
    if (max(thr.x, max(thr.y, thr.z)) < 0.004) { break; }
    dir = reflect(dir, m);
    pos = q - m * 0.004;
  }
  return acc;
}

fn grazeM(p: vec3f, rd: vec3f) -> vec3f {
  let n = nrm(p);
  let mir = reflect(rd, n);
//#if ROUGH
  return seen(mir, spreadR()) + skyOf(mir);
//#else
  return filmEnv(mir, filmLod(p, n, 0.0)) + skyOf(mir);
//#endif
}
//#endif
//#endif
