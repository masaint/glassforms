
// ---- the camera's lens -----------------------------------------------------------------------------------
// How a point of the picture becomes a ray, and how a point in space becomes a point of the picture again,
// for every kind of lens. Put into a shader only when the lens is something other than the plain one
// (an ordinary lens that keeps straight lines straight): the tracer, the pass that folds a frame into the
// picture, the focus stage and the final pass each have a plain text that never sees this.
// gf_core::camera does the same arithmetic in Rust, and its comments say why each line is what it is.
//
// The film is measured across and up from the lens's axis, in units of half the picture's height. The lens
// sits at (0, 0, distance) and looks down -z.

struct Cam {
  a: vec4f,     // x: the kind of lens: 0 perspective, 1 orthographic, 2 fisheye (equidistant), 3 fisheye
                //    (equisolid), 4 fisheye (stereographic), 5 panorama, 6 cylindrical
                // y: half the picture's height in focal lengths (a panorama: pi / 2)
                // z: the lens's distance from the object's centre      w: half the height of an orthographic view
  b: vec4f,     // xy: the picture slid across the lens's image       z: how strongly straight lines are bowed
                //    (above 0 outward, below 0 inward)                w: a pixel's size times the picture's height
  c: vec4f,     // xy: what a pixel covers t along its ray is (x t + y) pixel sizes
                // zw: the slope of the plane of focus: depth gained per unit across, per unit up
  e: vec4f,     // x: anamorphic squeeze (highlights this much taller than wide)   y: highlight edge (-1 soft core .. 1 bright rim)
                // z: cat's-eye      w: the distance at which an orthographic lens's cone is as wide as a perspective one's
}

// A barrel lens's picture ends where its angle reaches this.
const CAM_WARP_MAX: f32 = 1.5;

// A point of the film to where a lens that kept straight lines straight would have put it.
fn camOut(q: vec2f, w: f32) -> vec2f {
  if (w == 0.0) { return q; }
  let r = length(q);
  if (r < 1e-6) { return q; }
  if (w > 0.0) { return q * (tan(min(w * r, CAM_WARP_MAX)) / (w * r)); }
  return q * (atan(-w * r) / (-w * r));
}
// The other way. z = 0 where the lens cannot show it.
fn camIn(q: vec2f, w: f32) -> vec3f {
  if (w == 0.0) { return vec3f(q, 1.0); }
  let r = length(q);
  if (r < 1e-6) { return vec3f(q, 1.0); }
  if (w > 0.0) { return vec3f(q * (atan(w * r) / (w * r)), 1.0); }
  let x = -w * r;
  return vec3f(q * (tan(min(x, CAM_WARP_MAX)) / x), select(0.0, 1.0, x < CAM_WARP_MAX));
}

// The ray through a point of the film: where it starts and which way it goes. False where the lens
// draws nothing (outside a fisheye's circle, above a panorama's pole).
fn camRay(film: vec2f, cm: Cam, ro: ptr<function, vec3f>, rd: ptr<function, vec3f>) -> bool {
  let kind = cm.a.x;
  let T = cm.a.y;
  let q = camOut(film, cm.b.z);
  *ro = vec3f(0.0, 0.0, cm.a.z);
  *rd = vec3f(0.0, 0.0, -1.0);
  if (kind < 0.5) {
    *rd = normalize(vec3f(q * T, -1.0));
    return true;
  }
  if (kind < 1.5) {
    *ro = vec3f(q * cm.a.w, cm.a.z);
    return true;
  }
  if (kind < 4.5) {
    // a fisheye: the radius in the picture says how far off the axis the ray is
    let rho = length(q) * T;
    var th = rho;
    if (kind > 3.5) {
      th = 2.0 * atan(rho * 0.5);
    } else if (kind > 2.5) {
      if (rho > 2.0) { return false; }
      th = 2.0 * asin(rho * 0.5);
    } else if (rho > 3.14159265) {
      return false;
    }
    if (rho < 1e-7) { return true; }
    *rd = vec3f(q * (sin(th) * T / rho), -cos(th));
    return true;
  }
  if (kind < 5.5) {
    let lam = q.x * T;
    let phi = q.y * T;
    if (abs(phi) > 1.5707964) { return false; }
    *rd = vec3f(cos(phi) * sin(lam), sin(phi), -cos(phi) * cos(lam));
    return true;
  }
  let lam = q.x * T;
  *rd = normalize(vec3f(sin(lam), q.y * T, -cos(lam)));
  return true;
}

// Where something in the direction v from the lens shows on the film (xy); z = 0 if the lens cannot see
// that way. (Not for an orthographic lens, which sees one direction only.)
fn camFilmDir(v: vec3f, cm: Cam) -> vec3f {
  let kind = cm.a.x;
  let T = cm.a.y;
  var q = vec2f(0.0);
  var ok = 1.0;
  if (kind < 0.5) {
    q = v.xy / (-v.z * T);
    ok = select(0.0, 1.0, v.z < -1e-4 * length(v));
  } else if (kind < 4.5) {
    let r = length(v.xy);
    let th = atan2(r, -v.z);
    var rho = th;
    if (kind > 3.5) { rho = 2.0 * tan(th * 0.5); } else if (kind > 2.5) { rho = 2.0 * sin(th * 0.5); }
    q = v.xy * (rho / (max(r, 1e-9) * T));
  } else if (kind < 5.5) {
    q = vec2f(atan2(v.x, -v.z), atan2(v.y, length(v.xz))) / T;
  } else {
    q = vec2f(atan2(v.x, -v.z), v.y / max(length(v.xz), 1e-9)) / T;
  }
  let f = camIn(q, cm.b.z);
  return vec3f(f.xy, f.z * ok);
}

// Where a point in space shows on the film (xy); z = 0 if the lens cannot see it.
fn camFilm(p: vec3f, cm: Cam) -> vec3f {
  let v = p - vec3f(0.0, 0.0, cm.a.z);
  if (cm.a.x > 0.5 && cm.a.x < 1.5) {
    let f = camIn(v.xy / cm.a.w, cm.b.z);
    return vec3f(f.xy, f.z * select(0.0, 1.0, v.z < 0.0));
  }
  return camFilmDir(v, cm);
}

// How far along its own ray a point is: the distance the tracer stores with a pixel.
fn camAlong(p: vec3f, cm: Cam) -> f32 {
  if (cm.a.x > 0.5 && cm.a.x < 1.5) { return cm.a.z - p.z; }
  return length(p - vec3f(0.0, 0.0, cm.a.z));
}

// How deep a point is, the way focus measures it: for a lens that focuses on a plane, its depth from the
// object's centre along the axis, less what the plane of focus has gained by its slope there; for one
// that focuses on a sphere (the lenses that see all round), its distance from the lens less the camera's.
fn camDepth(p: vec3f, cm: Cam) -> f32 {
  if (cm.a.x > 1.5) { return length(p - vec3f(0.0, 0.0, cm.a.z)) - cm.a.z; }
  return -p.z - (cm.c.z * p.x + cm.c.w * p.y);
}

// How far along a ray the surface of depth e is met.
fn camReach(ro: vec3f, rd: vec3f, e: f32, cm: Cam) -> f32 {
  if (cm.a.x > 1.5) { return max(cm.a.z + e, 1e-6); }
  return (e + ro.z + (cm.c.z * ro.x + cm.c.w * ro.y)) / max(-rd.z - (cm.c.z * rd.x + cm.c.w * rd.y), 1e-6);
}

// The radius of the blur of glass t along a ray, as an angle (or, for an orthographic lens, a length)
// to be multiplied by lens radius / pixel size: negative in front of the band lo..hi, nothing inside it,
// positive behind. See Camera::blur.
fn camBlur(ro: vec3f, rd: vec3f, t: f32, lo: f32, hi: f32, cm: Cam) -> f32 {
  let f = camReach(ro, rd, clamp(camDepth(ro + rd * t, cm), lo, hi), cm);
  if (cm.a.x > 0.5 && cm.a.x < 1.5) { return (t - f) / cm.e.w; }
  let across = select(-rd.z, 1.0, cm.a.x > 1.5);
  return 1.0 / (f * across) - 1.0 / (t * across);
}

// Towards the corners of the picture a real lens's barrel cuts into its opening from both sides, so that
// an out-of-focus highlight there is not a disc but a cat's eye. The cut for a point of the film: the
// opening (a unit disc) is what lies within one radius of both +c and -c, where c is this. `k`: how much
// cat's-eye the lens has; each side comes in by half the opening's radius at the corner of a 3:2 picture
// with k = 1.
fn camCut(film: vec2f, k: f32) -> vec2f {
  return film * (0.5 * k / 1.8027756);
}
// How much of the opening is left by a cut c: the area two unit discs 2|c| apart share, over a disc's
// (2 acos x - 2 x sqrt(1 - x^2), written as the polynomial that is right to a thousandth up to x = 0.6).
fn camOpen(c: vec2f) -> f32 {
  let x = min(length(c), 0.6);
  return 1.0 - 1.2732395 * x + 0.2122066 * x * x * x;
}
