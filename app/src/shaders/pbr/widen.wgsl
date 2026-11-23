// How wide a reflection is: the pure arithmetic shared by the mirror's width (mirrorWide) and the rough
// lobe's variance cap (frCap). The curvature probes that feed them read the distance field and stay with
// their hosts.

// The width (radians) of the patch of surroundings one pixel's worth of surface mirrors: `pixel` the angle
// of a pixel at unit distance, `bend` the surface's greatest bend, `far` the distance to the eye, `c` the
// cosine of the view angle.
fn pbr_reflection_width(pixel: f32, bend: f32, far: f32, c: f32) -> f32 {
  let wide = pixel * (1.0 + 2.0 * bend * far / max(c, 0.25));
  return 0.4 * wide;
}

// A variance (rad^2) held below `wide` squared smoothly: v / (1 + v / wide^2).
fn pbr_var_cap(v: f32, wide: f32) -> f32 { return v / (1.0 + v / (wide * wide)); }
