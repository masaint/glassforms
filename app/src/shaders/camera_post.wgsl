
// ---- the final pass through any lens, and the lens's own character ---------------------------------------
// The plain final pass is used as it stands, with three of its lines handed to the functions below: the
// depth of the glass for the focus aid is worked out through whatever lens is on, and two things a real
// lens does to its picture are done to it:
//   colour fringing   the lens draws the red end of the spectrum a little larger than the blue end, so
//                     edges away from the middle of the picture split into colours. The picture is read
//                     seven times along the line from the lens's axis, each reading at the size one part
//                     of the spectrum is drawn at and counted for that part's colour. Nothing moves at the
//                     axis; towards the corners the picture also loses a little sharpness, as it does
//                     through such a lens.
//   fall-off          a lens lights the corners of its picture less than the middle: less of the world is
//                     packed into each piece of film there, and the lens opening is seen aslant. For an
//                     ordinary lens that comes to the fourth power of the cosine of the angle off the axis;
//                     each other kind of lens has its own law. `fx.y` is how much of it the lens keeps:
//                     real lenses are built to lose less than the law says.
//   the barrel's cut  where the lens's barrel cuts into its opening (cat's-eye highlights), the corners
//                     get that much less light, whether sharp or blurred.

// The depth, the way focus measures it, of glass t along the ray through the film point `film`.
fn lensDepth(film: vec2f, t: f32) -> f32 {
  var ro: vec3f;
  var rd: vec3f;
  if (!camRay(film, fu.lens, &ro, &rd)) { return 0.0; }
  return camDepth(ro + rd * t, fu.lens);
}

// The picture at a film point with its colours drawn apart; `sharp` is the picture there as it stands.
fn lensFringe(sharp: vec3f, film: vec2f) -> vec3f {
  if (fu.fx.x <= 0.0) { return sharp; }
  var c = vec3f(0.0);
  for (var i = 0; i < 7; i++) {
    let k = f32(i) / 3.0 - 1.0;                              // from the blue end (-1) to the red end (1)
    let at = ((film * (1.0 + fu.fx.x * k) - fu.view.zw) * fu.res.y + fu.res) * 0.5;
    // (each colour's readings add up to one)
    let part = vec3f(max(k, 0.0) * 0.5, (1.0 - abs(k)) / 3.0, max(-k, 0.0) * 0.5);
    c += part * textureSampleLevel(sceneTex, flin, vec2f(at.x / fu.res.x, 1.0 - at.y / fu.res.y), 0.0).rgb;
  }
  return c;
}

// How much of its light a film point keeps: the lens's fall-off by its own law, and what the barrel's
// cut takes.
fn lensLight(film: vec2f) -> f32 {
  var keep = 1.0;
  if (fu.fx.y > 0.0) {
    var ro: vec3f;
    var rd: vec3f;
    if (camRay(film, fu.lens, &ro, &rd)) {
      let kind = fu.lens.a.x;
      let cosine = clamp(-rd.z, -1.0, 1.0);
      var light = 1.0;
      if (kind < 0.5) {
        light = cosine * cosine * cosine * cosine;
      } else if (kind > 1.5 && kind < 2.5) {
        let th = acos(cosine);
        light = select(sin(th) / th, 1.0, th < 1e-3);
      } else if (kind > 3.5 && kind < 4.5) {
        let hc = 0.5 + 0.5 * cosine;
        light = hc * hc;
      } else if (kind > 5.5) {
        let level = length(rd.xz);
        light = level * level * level;
      }
      keep = pow(max(light, 0.0), fu.fx.y);
    }
  }
  if (fu.fx.z > 0.0) { keep *= camOpen(camCut(film, fu.fx.z)); }
  return keep;
}
