
//#if ROUGH
// ---- rough surfaces: frosted and satin glass, satin metal -------------------------------------------
//
// A rough surface is facets too small to see, tilted this way and that (their slopes spread by
// alpha = mu.surf.x, as GGX spreads them). It is drawn in one of two ways.
//
// Live (the default): nothing is left to chance. One path is followed, the one the smooth surface would
// give, and with it travels the width of the cone of directions it stands for: every rough surface it
// meets widens the cone, and so does every curved one, by as much as its normal varies across the patch
// the cone lands on. Whatever depends sharply on direction is then used as its average over the cone:
// the lights are read from a copy blurred that much (seen); the share a surface reflects, and whether
// light is trapped inside by total reflection, are averaged over the cone's spread of angles (frWall);
// normals are taken across the patch (frNrm); a further piece of the body covers as much of the cone as
// it does (frCover). Once the cone is so wide that the light has been scattered every way, the path ends
// there and that light is the light from all round (frAll). One frame is the finished picture: no
// grain, nothing to settle.
//
// By chance alone (roughRef=1): every surface met is one facet drawn at random, the lights are read as
// they are, and the picture is the average of many frames. Slow to settle; kept as the yardstick the
// live way is held to, and for stills.
//
// A direction's spread is carried as a variance, in square radians.

// The widest a cone is ever taken to be: a lobe cannot be wider than the half of the sphere it leaves into.
const FR_WIDE: f32 = 0.8;
fn frCap(v: f32) -> f32 { return v / (1.0 + v / (FR_WIDE * FR_WIDE)); }
// A path inside the body whose directions are spread this far (from, to) has been scattered every way:
// it shows nothing of where it has been, and is ended.
const FR_LOST: vec2f = vec2f(0.25, 1.0);
// How far what is seen through the surface is to be blurred after tracing (frost_blur.wgsl): the radius
// of the cone of light the first path to leave the body stands for, where it lands on the first further
// piece of the body (in the shape's own units); next to nothing if it meets none (the lights beyond are
// softened where they are read, and what the surface's own shape does to the light is not to be blurred).
var<private> frRad: f32;
// ... which way the surface faces where the first ray of the pixel met it (the blur keeps to surface
// that faces much the same way: across a crease the picture changes for good reason), and what the
// surface itself reflects, summed over the pixel's rays (that is not seen through it, and is not blurred).
var<private> frFace: vec3f;
var<private> frTopLight: vec3f;
// ... and the share of the light the surface reflects, summed likewise (what is seen through it is the
// light behind it times the rest: it is the light behind that is blurred).
var<private> frTopShare: f32;

//#if ROUGH_REF
// One facet's normal, drawn from those a ray arriving along -wi can see, in proportion to how much of
// each it sees (Heitz 2018, "Sampling the GGX distribution of visible normals"). n: the surface's own
// normal on the side the ray arrives from; wi: back along the ray; r: two numbers in 0..1.
// A facet drawn this way always faces the ray, so nothing has to fall back on the smooth surface.
fn facetV(n: vec3f, wi: vec3f, r: vec2f) -> vec3f {
  let a = mu.surf.x;
  let s = select(-1.0, 1.0, n.z >= 0.0);
  let k = -1.0 / (s + n.z);
  let b = n.x * n.y * k;
  let t1 = vec3f(1.0 + s * n.x * n.x * k, s * b, -s * n.x);
  let t2 = vec3f(b, s + n.y * n.y * k, -n.y);
  // the ray in the surface's frame, squeezed so that the facets become a hemisphere
  let vh = normalize(vec3f(a * dot(wi, t1), a * dot(wi, t2), max(dot(wi, n), 1e-4)));
  let l2 = vh.x * vh.x + vh.y * vh.y;
  var T1 = vec3f(1.0, 0.0, 0.0);
  if (l2 > 1e-8) { T1 = vec3f(-vh.y, vh.x, 0.0) / sqrt(l2); }
  let T2 = cross(vh, T1);
  // a point of the disc the ray sees that hemisphere as
  let rr = sqrt(r.x);
  let ph = 6.2831853 * r.y;
  let p1 = rr * cos(ph);
  let sh = 0.5 * (1.0 + vh.z);
  let p2 = (1.0 - sh) * sqrt(max(1.0 - p1 * p1, 0.0)) + sh * rr * sin(ph);
  let nh = p1 * T1 + p2 * T2 + sqrt(max(1.0 - p1 * p1 - p2 * p2, 0.0)) * vh;
  let ne = normalize(vec3f(a * nh.x, a * nh.y, max(nh.z, 1e-6)));
  return ne.x * t1 + ne.y * t2 + ne.z * n;
}
// A direction that has been sent below the surface it left (the facet was steep) meets another facet
// on its way and comes back up: taken as mirrored in the surface, which keeps its light.
fn frUp(d: vec3f, n: vec3f) -> vec3f {
  let c = dot(d, n);
  if (c < 0.0) { return d - 2.0 * c * n; }
  return d;
}
fn seen(rd: vec3f, sig: f32) -> vec3f { return envL(rd, sig); }

// The facet light gets into the body through.
fn frIn(n: vec3f, rd: vec3f, xi: f32) -> vec3f { return facetV(n, -rd, rnd2(0, rd, xi)); }
fn frCos(c: f32) -> f32 { return clamp(c, 0.0, 1.0); }
// What the body's own surface reflects towards the eye, for each unit of reflectance at cosine c:
// the lights straight off another facet, weighed by that facet's reflectance in place of c's.
fn frTop(rd: vec3f, n: vec3f, xi: f32, c: f32) -> vec3f {
  let f = facetV(n, -rd, rnd2(7, rd, xi));
  return env(frUp(reflect(rd, f), n)) * (fresnel(clamp(-dot(rd, f), 0.0, 1.0), 1.0, u.ior) / max(fresnel(c, 1.0, u.ior), 1e-4));
}
// The path into the body: as the facet bends it (never back out).
fn frDir(d: vec3f, n: vec3f) -> vec3f { return -frUp(-d, n); }
// The facet a path inside the body meets at a wall whose outward normal is m.
fn frFacet(m: vec3f, dir: vec3f, b: i32, rd: vec3f, xi: f32) -> vec3f { return -facetV(-m, -dir, rnd2(b + 1, rd, xi)); }
// At that wall: the share sent back inside (x), nothing else.
fn frWall(c: f32, v: f32, n1: f32, n2: f32, bend: vec2f) -> vec3f { return vec3f(fresnel(c, n1, n2), 0.0, 0.0); }
// Where the share that leaves goes: through the facet.
fn frExit(dir: vec3f, m: vec3f, mf: vec3f, w: vec3f, n1: f32) -> vec3f {
  let od = refract(dir, -mf, n1);
  if (dot(od, od) < 0.5) { return m; }
  return frUp(od, m);
}
// (a single ray has no spread, lands on a point, and whatever it hits covers it)
fn frVin0(ior: f32) -> f32 { return 0.0; }
fn frNext(v: f32, bend: vec2f) -> f32 { return 0.0; }
fn frVout(v: f32, w: vec3f, co: f32, n1: f32, bend: vec2f) -> f32 { return 0.0; }
fn frEnd(v: f32, bend: vec2f) -> f32 { return 0.0; }
fn frAll(d: vec3f) -> vec3f { return vec3f(0.0); }
fn frDeep(s: f32) -> vec3f { return vec3f(1.0); }
fn frLeak() -> f32 { return 0.001; }
fn frWide(sig: f32) -> f32 { return 0.0; }
fn frHit(l: i32, r: f32, sig: f32) { }
fn frMiss(l: i32) { }
fn frSeen(n: vec3f, rd: vec3f, xi: f32, c: f32, W: i32) { }
fn frNrm(p: vec3f, r: f32) -> vec3f { return nrm(p); }
fn frBend(p: vec3f, n: vec3f, r: f32, dir: vec3f) -> vec2f { return vec2f(0.0); }
fn frCover(p: vec3f, rd: vec3f, n: vec3f, r: f32) -> f32 { return 1.0; }
fn spreadR() -> f32 { return 0.0; }
// A metal: the reflection goes off one facet, to the next surface or to the lights as they are.
fn frMetalDir(rd: vec3f, n: vec3f, b: i32, rd0: vec3f, xi: f32) -> vec3f { return frUp(reflect(rd, facetV(n, -rd, rnd2(b, rd0, xi))), n); }
fn frMetalLit(dir: vec3f, mid: vec3f, v: f32, b: i32) -> vec3f { return env(dir); }
fn frMetalV(v: f32, bend: vec2f) -> f32 { return 0.0; }
fn frMetalHid(p: vec3f, n: vec3f, mid: vec3f, v: f32, lit: vec3f, R: vec3f) -> vec3f { return lit; }
fn frFollow(v: f32) -> f32 { return 1.0; }
fn frMid(rd: vec3f, n: vec3f) -> vec3f { return reflect(rd, n); }
// The lights a further piece of the body mirrors, and the facets light enters and leaves it by.
fn frFar(rd: vec3f, n: vec3f, p: vec3f, sig: f32, r: f32) -> vec3f { return env(frUp(reflect(rd, facetV(n, -rd, hash2(p - rd))), n)); }
fn frFarIn(n: vec3f, rd: vec3f, p: vec3f) -> vec3f { return facetV(n, -rd, hash2(p + rd)); }
fn frFarFacet(m: vec3f, dir: vec3f, q: vec3f) -> vec3f { return -facetV(-m, -dir, hash2(q + dir)); }
//#else

// The facets' slopes are spread as GGX spreads them, which is very nearly two bells laid over one
// another: three quarters of the facets in a narrow one (slope variance alpha^2 / 2 each way) and a
// quarter in a skirt FR_SKIRT times as wide (fitted: the two together are within 3% of GGX's share of
// facets at every slope). The narrow bell is what a path's spread is reckoned in; the skirt is put back
// wherever the lights are read (seenH) and where light meets a wall from inside (frWall).
const FR_SKIRT: f32 = 3.46;
// The lights along rd as rough surfaces show them: spread by sig, with the share h of the light (what
// has gone through a skirt on its way) spread further, by the skirt of one surface. facets: how far the
// facets alone have spread it, in units of alpha^2 (the rest of sig is the shape's doing, and has no skirt).
fn seenH(rd: vec3f, sig: f32, h: f32, facets: f32) -> vec3f {
  let a = mu.surf.x;
  let d = toEnv(rd);
  let uv = vec2f(atan2(d.z, d.x) * 0.15915494 + 0.5, acos(clamp(d.y, -1.0, 1.0)) * 0.31830989);
  let s2 = sqrt(min(sig * sig + 11.0 * min(sig * sig, facets * a * a), 4.0));
  let l1 = clamp(select(log2(max(sig, 1e-6) / T0), sig / (2.0 * T0), sig < 2.0 * T0), 0.0, 9.0);
  let l2 = clamp(select(log2(max(s2, 1e-6) / T0), s2 / (2.0 * T0), s2 < 2.0 * T0), 0.0, 9.0);
  var c = mix(textureSampleLevel(softTex, softSmp, uv, l1).rgb, textureSampleLevel(softTex, softSmp, uv, l2).rgb, h);
//#if PICTURE
  let p1 = clamp(log2(sqrt(sig * sig + mu.pic.z * mu.pic.z) / mu.pic.y), 0.0, mu.pic.w);
  let p2 = clamp(log2(sqrt(s2 * s2 + mu.pic.z * mu.pic.z) / mu.pic.y), 0.0, mu.pic.w);
  c += mix(textureSampleLevel(pictureTex, pictureSmp, d, p1).rgb, textureSampleLevel(pictureTex, pictureSmp, d, p2).rgb, h) * mu.pic.x;
//#endif
  return c;
}
// The share of light that has been through a skirt after meeting k rough surfaces: all but (3/4)^k.
fn frSkirt(k: f32) -> f32 { return 1.0 - exp(-0.2877 * k); }
// The lights as light that has come through the body sees them: in by one rough surface and out by
// another, which between them spread it by about a third of alpha^2. (A milky body's own haze has no skirt.)
fn seen(rd: vec3f, sig: f32) -> vec3f {
//#if SSS
  return envL(rd, sig);
//#else
  return seenH(rd, sig, frSkirt(2.0), 0.3);
//#endif
}
// The light from all round: what light that has been scattered every way inside the body, or by a
// further piece of it, comes to, at a wall facing d. Scattered every way inside, it still came in
// through the surface round that wall, from the lights that surface faces: a copy of the lights spread
// about as widely as a matte surface gathers them (about a radian), read towards d. (It was the lights'
// softest copy, the same everywhere: a frosted body then glowed evenly all over, lit from nowhere, and
// read as a coloured filter, not as a body the light comes through from one side.) With an environment
// picture, its softest level towards d.
const FR_ALL_LOD: f32 = 7.3;
fn frAll(d: vec3f) -> vec3f {
  let e = toEnv(d);
  let uv = vec2f(atan2(e.z, e.x) * 0.15915494 + 0.5, acos(clamp(e.y, -1.0, 1.0)) * 0.31830989);
//#if PICTURE
  return textureSampleLevel(softTex, softSmp, uv, FR_ALL_LOD).rgb + textureSampleLevel(pictureTex, pictureSmp, e, mu.pic.w).rgb * mu.pic.x;
//#else
  return textureSampleLevel(softTex, softSmp, uv, FR_ALL_LOD).rgb;
//#endif
}
//#if !BODIES
//#if !CLAY
//#if !METAL
// What the stage after the tracer is told about the surface a ray met: which way it faces, and the
// light it reflects itself. (Worked out again here, after the paths inside have been followed, so that
// it does not have to be carried while they are. With a film on the surface the reflection's colours
// vary by wavelength and are left to be blurred with the rest.)
fn frSeen(n: vec3f, rd: vec3f, xi: f32, c: f32, W: i32) {
  if (frFace.z == 0.0) { frFace = vec3f(n.xy, 1.0); }
//#if !FINE
//#if COAT
  frTopLight += top1(c) * env(reflect(rd, n)) + (1.0 - top1(c)) * (1.0 - top1(c)) * under(c, iorOf(0, xi, W)) * frTop(rd, n, xi, c);
//#else
  frTopLight += top1(c) * frTop(rd, n, xi, c);
//#endif
  frTopShare += top1(c);
//#endif
}
//#endif
//#endif
//#endif
// The first path to leave the body, its light spread by sig, lands on a further piece of the body where
// its cone has radius r (l: 0 for the first piece met): that is the blur's radius. (No piece is taken to
// be further off than the shape is wide; and once the light is spread so far that further pieces
// scatter it every way, there is nothing left to see through the surface and little to blur.)
fn frHit(l: i32, r: f32, sig: f32) {
  if (l == 0 && frRad == 0.0) { frRad = max(min(r, sig) * (1.0 - 0.85 * frWide(sig)), 1e-6); }
}
// ... or it meets nothing more.
fn frMiss(l: i32) {
  if (l == 0 && frRad == 0.0) { frRad = 1e-6; }
}
// A share of light leaving through a wall that is smaller than this is not followed out of the body
// (the march beyond costs as much for a twentieth of the light as for all of it).
fn frLeak() -> f32 { return 0.04; }
// Light that is ended inside the body has not left it yet: it wanders on from wall to wall, about
// FR_ROAM more crossings like the last one (s long) before it gets out, and the body's tint takes its
// share of it all that way. So where the body is thick, what it gives back is dimmer and deeper in colour.
const FR_ROAM: f32 = 2.5;
fn frDeep(s: f32) -> vec3f { return exp(-u.absorb.xyz * (FR_ROAM * max(s, 0.0))); }
// How much of a cone this wide is taken to be scattered every way by a further piece of the body it
// lands on, instead of being followed through it.
fn frWide(sig: f32) -> f32 { return smoothstep(0.2, 0.45, sig); }

// How far the reflection off the surface is spread (standard deviation, radians).
fn spreadR() -> f32 { return sqrt(frCap(2.0 * mu.surf.x * mu.surf.x)); }

fn frIn(n: vec3f, rd: vec3f, xi: f32) -> vec3f { return n; }
// A rough surface never quite turns into a mirror at a glancing angle: the facets that face the eye
// are the ones it sees. The cosine its reflectance is worked out at.
fn frCos(c: f32) -> f32 { return min(sqrt(c * c + 0.25 * mu.surf.x * mu.surf.x), 1.0); }
// The middle of the lobe the surface reflects: the mirror direction for a smooth surface, turning
// towards the normal as it roughens (Lagarde & de Rousiers 2014).
fn frMid(rd: vec3f, n: vec3f) -> vec3f {
  let a = min(mu.surf.x, 1.0);
  return normalize(mix(n, reflect(rd, n), (1.0 - a) * (sqrt(1.0 - a) + a)));
}
// The body's own reflection: one rough surface, a quarter of it through the skirt.
fn frTop(rd: vec3f, n: vec3f, xi: f32, c: f32) -> vec3f { return seenH(frMid(rd, n), spreadR(), 0.25, 2.0); }
// The path into the body: as the smooth surface bends it, turned towards straight in as it roughens.
fn frDir(d: vec3f, n: vec3f) -> vec3f { return normalize(mix(d, -n, 0.5 * min(mu.surf.x * mu.surf.x, 1.0))); }
fn frFacet(m: vec3f, dir: vec3f, b: i32, rd: vec3f, xi: f32) -> vec3f { return m; }

// How far the directions of a path are spread just inside the body: by the facets it came in through.
fn frVin0(ior: f32) -> f32 {
  let a = mu.surf.x;
  return 0.5 * a * a * (1.0 - 1.0 / ior) * (1.0 - 1.0 / ior);
}
// ... and after a wall has sent it back inside: by that wall's facets, and by how much the wall's
// normal varies across the patch the cone landed on (bend), both doubled as a mirror doubles a tilt.
fn frNext(v: f32, bend: vec2f) -> f32 { return min(v + 2.0 * mu.surf.x * mu.surf.x + 4.0 * bend.y, 1.0); }

// A cone of light inside the body (index n1) meets a wall: its middle at cosine c, its directions spread
// by v, the wall's facets by their own slopes and the wall itself by its bend. Returns
//   x: the share sent back inside, averaged over all of that (Fresnel's, and total reflection for the
//      part of the cone beyond the critical angle: a smooth wall traps a ray outright, a rough one lets
//      part of every cone out)
//   y: the sine of the angle at which the part that leaves met the wall (the middle of what is not trapped)
//   z: the share that is beyond the critical angle
// For no spread at all these are Fresnel's reflectance, the angle itself, and 0 or 1.
fn frWall(c: f32, v: f32, n1: f32, n2: f32, bend: vec2f) -> vec3f {
  let sd = sqrt(v + 0.5 * mu.surf.x * mu.surf.x + bend.x);
  let th = acos(clamp(c, 0.0, 1.0));
  let thc = asin(clamp(n2 / n1, 0.0, 1.0));
  let z = (th - thc) / sd;
  // the share of the facets' two bells beyond z (a logistic curve stands in for each), and the middle of the rest
  let P = 0.75 / (1.0 + exp(-1.702 * z)) + 0.25 / (1.0 + exp(-1.702 * z / FR_SKIRT));
  let thm = clamp(th - sd * 0.5 * (z + sqrt(z * z + 2.56)), 0.0, thc);
  return vec3f(P + (1.0 - P) * fresnel(cos(thm), n1, n2), sin(thm), P);
}
// Where the part that leaves goes: bent as Snell's law bends the angle it met the wall at.
// dir: the path's direction; m: the wall's outward normal; w: what frWall returned.
fn frExit(dir: vec3f, m: vec3f, mf: vec3f, w: vec3f, n1: f32) -> vec3f {
  let t = dir - dot(dir, m) * m;
  let l = length(t);
  if (l < 1e-5) { return m; }
  let so = min(n1 * w.y, 0.9995);
  return t * (so / l) + m * sqrt(1.0 - so * so);
}
// How far the light that leaves is spread outside: what it was spread by inside, widened as leaving
// widens any spread, and the facets (and the wall's bend) it leaves through. co: the cosine it leaves at.
// Leaving at a glancing angle stretches the spread in the plane the light turns in far more than
// across it; one width stands for both, the one that keeps the lobe as bright as it is.
fn frVout(v: f32, w: vec3f, co_in: f32, n1: f32, bend: vec2f) -> f32 {
  let co = max(co_in, 0.05);
  let ci = sqrt(1.0 - w.y * w.y);
  let amp = min(n1 * ci / co, 7.0);
  let a2 = 0.5 * mu.surf.x * mu.surf.x;
  let across = n1 * ci - co;
  let va = v * amp * amp + (a2 + bend.x) * (amp - 1.0) * (amp - 1.0);
  let vb = v * n1 * n1 + (a2 + max(2.0 * bend.y - bend.x, 0.0)) * across * across;
  return frCap(sqrt(va * vb));
}
// At a wall met with spread v: the share of what the wall sends back inside that is taken to have been
// scattered every way, and so is ended here as the light from all round instead of being followed
// further. None for a polished surface; all of it once the walls have spread the path so far that
// following it would show nothing more. (So a rough body is never left dark by light that total
// reflection would trap in a polished one, and the rougher it is the sooner its paths end.) The skirt's
// quarter of the facets spreads its share FR_SKIRT times as far, and is ended that much sooner.
fn frEnd(v: f32, bend: vec2f) -> f32 {
  let a = mu.surf.x;
  return 0.75 * smoothstep(FR_LOST.x, FR_LOST.y, frNext(v, bend)) + 0.25 * smoothstep(FR_LOST.x, FR_LOST.y, v + 24.0 * a * a + 4.0 * bend.y);
}

// ---- the patch of surface a cone of light lands on (r: the cone's radius there) ----
// The furthest the distance field is asked about a patch.
const FR_REACH: f32 = 0.12;
// The patch's normal: the slope of the distance field taken across the patch, not at its middle. For a
// patch no wider than the field's own grain it is the surface's normal as always.
fn frNrm(p: vec3f, r: f32) -> vec3f {
  let vox = 2.0 * HB / u.vres;
  let w = smoothstep(1.5 * vox, 3.0 * vox, 0.7 * r);
  var g = vec3f(0.0);
  if (w < 1.0) { g = nrm(p) * (1.0 - w); }
  if (w > 0.0) {
    let e = vec2f(1.0, -1.0) * min(0.7 * r, FR_REACH);
    g += w * normalize(e.xyy * mapW(p + e.xyy) + e.yyx * mapW(p + e.yyx) + e.yxy * mapW(p + e.yxy) + e.xxx * mapW(p + e.xxx));
  }
  return normalize(g);
}
// How much the surface's normal varies across the patch (a variance, square radians): from how far the
// surface falls away from its own flat within the patch, along the way the light is heading (x: this
// is what spreads the angles it meets the surface at, and so decides how much of it total reflection
// lets go) and on average over both ways (y). It spreads what the patch reflects and lets through
// exactly as facets of that slope would. (Light guided along a tube meets no bend along its way,
// however tightly the tube is curved across it.)
fn frBend(p: vec3f, n: vec3f, r: f32, dir: vec3f) -> vec2f {
  let vox = 2.0 * HB / u.vres;
  let rr = clamp(r, 2.5 * vox, FR_REACH);
  // (a patch much smaller than the field's grain bends by next to nothing: not worth asking)
  let q = r / rr;
  if (q < 0.2) { return vec2f(0.0); }
  var t1 = dir - dot(dir, n) * n;
  let l = length(t1);
  if (l < 1e-4) {
    // (straight at the surface: any two directions across it)
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let k = -1.0 / (s + n.z);
    t1 = vec3f(1.0 + s * n.x * n.x * k, s * n.x * n.y * k, -s * n.x);
  } else {
    t1 = t1 / l;
  }
  let t2 = cross(n, t1) * rr;
  t1 = t1 * rr;
  let f0 = mapW(p + t1);
  let f1 = mapW(p - t1);
  let f2 = mapW(p + t2);
  let f3 = mapW(p - t2);
  let k2 = 2.0 * q * q / (rr * rr) * smoothstep(0.2, 0.4, q);
  let along = k2 * (f0 * f0 + f1 * f1);
  return min(vec2f(along, 0.5 * (along + k2 * (f2 * f2 + f3 * f3))), vec2f(1.0));
}
// How much of a cone of light, radius r where it lands at p on a further piece of the body, that piece
// covers: all of it for a narrow cone or a piece that spreads well beyond it, less near the piece's
// outline and for a piece thinner than the cone. Four rays of the cone are asked whether they would
// meet the piece too, each where it would cross the patch's flat (a little way in).
fn frCover(p: vec3f, rd: vec3f, n: vec3f, r_in: f32) -> f32 {
  let vox = 2.0 * HB / u.vres;
  if (r_in < vox) { return 1.0; }
  let r = min(r_in, 0.5);
  let s = select(-1.0, 1.0, rd.z >= 0.0);
  let k = -1.0 / (s + rd.z);
  let b = rd.x * rd.y * k;
  let u1 = vec3f(1.0 + s * rd.x * rd.x * k, s * b, -s * rd.x);
  let u2 = vec3f(b, s + rd.y * rd.y * k, -rd.y);
  let c = min(dot(rd, n), -0.3);            // (a patch seen at a glancing angle is not asked about further off than this)
  let d = 0.25 * r;
  let o = p - n * d;
  var cov = 0.0;
  for (var i = 0; i < 4; i++) {
    let u = select(u1, u2, i >= 2) * select(r, -r, (i & 1) == 1);
    let f = mapW(o + u - rd * (dot(u, n) / c));
    cov += 1.0 - smoothstep(-0.5 * d, 0.5 * d, f);
  }
  // (half covered is the outline itself, where the cover has to reach nothing: a ray just outside meets nothing)
  let fade = smoothstep(vox, 2.0 * vox, r_in);
  return clamp(0.5 * cov - 1.0, 0.0, 1.0) * fade + (1.0 - fade);
}

// A metal. The spread of the light its surface reflects: what the light was spread by already, this
// surface's facets, and the bend of the patch the cone landed on (doubled, as a mirror doubles a tilt).
fn frMetalV(v: f32, bend: vec2f) -> f32 { return frCap(v + 2.0 * mu.surf.x * mu.surf.x + 4.0 * bend.y); }
fn frMetalDir(rd: vec3f, n: vec3f, b: i32, rd0: vec3f, xi: f32) -> vec3f { return reflect(rd, n); }
// The lights the lobe sees (mid: its middle; v: its spread; after b earlier reflections).
fn frMetalLit(dir: vec3f, mid: vec3f, v: f32, b: i32) -> vec3f { return seenH(mid, sqrt(v), frSkirt(f32(b + 1)), 2.0 * f32(b + 1)); }
// The lobe's middle is followed to the next surface while the lobe is narrow (the shape mirrored in
// itself, softly); the wider the lobe, the less of it is followed, and past a certain width none.
fn frFollow(v: f32) -> f32 { return 1.0 - smoothstep(0.02, 0.12, v); }
// How much of a lobe leaving p about the direction d (spread sig) gets clear of the shape: the distance
// field is asked at a few points along d how near the shape comes, against how wide the lobe is there.
fn frOpen(p: vec3f, d: vec3f, sig: f32) -> f32 {
  var open = 1.0;
  var t = 0.045;
  for (var i = 0; i < 6; i++) {
    let r = max(sig * t, 0.01);
    open = min(open, clamp(0.5 + 0.5 * mapW(p + d * t) / r, 0.0, 1.0));
    t *= 1.75;
  }
  return open;
}
// The part of a lobe that is not followed, with the shape itself in its way: what gets clear of the
// shape sees the lights; the rest lands on more of the same metal, which shows it the light from all
// round, tinted once more (so hollows are darker and deeper in colour, as they are in the metal).
fn frMetalHid(p: vec3f, n: vec3f, mid: vec3f, v: f32, lit: vec3f, R: vec3f) -> vec3f {
  if (v <= 0.02) { return lit; }            // (a lobe this narrow is followed whole)
  let open = frOpen(p + n * 0.01, mid, sqrt(v));
  return lit * open + (1.0 - open) * 0.7 * R * frAll(n);
}

// What a further piece of the body mirrors of the lights: spread by what the light was spread by
// already, by this surface's facets, and by the bend of the patch the cone lands on.
fn frFar(rd: vec3f, n: vec3f, p: vec3f, sig: f32, r: f32) -> vec3f {
  return seenH(frMid(rd, n), sqrt(frCap(sig * sig + 2.0 * mu.surf.x * mu.surf.x + 4.0 * frBend(p, n, r, rd).y)), frSkirt(3.0), 2.3);
}
fn frFarIn(n: vec3f, rd: vec3f, p: vec3f) -> vec3f { return n; }
fn frFarFacet(m: vec3f, dir: vec3f, q: vec3f) -> vec3f { return m; }
//#endif
//#endif
