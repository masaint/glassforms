// Nine-term spherical harmonic sums along a unit direction d (in the lights' own frame), shared by every
// copy that reads a room's nine terms (trace_light nine, paintRoomB, paintDistantB, the surface pass's
// nineAt and surfGiDistant). The terms are passed one by one so each caller keeps reading its own uniform
// block; the sums keep the exact order the copies had. Neither is clamped: callers clamp as before.
// pbr_sh9_sum: the terms as stored (cosine-convolved: irradiance over pi).
fn pbr_sh9_sum(c0: vec3f, c1: vec3f, c2: vec3f, c3: vec3f, c4: vec3f, c5: vec3f, c6: vec3f, c7: vec3f, c8: vec3f, d: vec3f) -> vec3f {
  return c0
       + c1 * d.y + c2 * d.z + c3 * d.x
       + c4 * (d.x * d.y) + c5 * (d.y * d.z) + c6 * (3.0 * d.z * d.z - 1.0)
       + c7 * (d.x * d.z) + c8 * (d.x * d.x - d.y * d.y);
}
// pbr_sh9_distant: the same terms with the cosine convolution's band factors undone (1, 1.5, 4): the
// room's radiance along d rather than the light a surface facing d gathers.
fn pbr_sh9_distant(c0: vec3f, c1: vec3f, c2: vec3f, c3: vec3f, c4: vec3f, c5: vec3f, c6: vec3f, c7: vec3f, c8: vec3f, d: vec3f) -> vec3f {
  return c0
       + 1.5 * (c1 * d.y + c2 * d.z + c3 * d.x)
       + 4.0 * (c4 * (d.x * d.y) + c5 * (d.y * d.z) + c6 * (3.0 * d.z * d.z - 1.0)
       + c7 * (d.x * d.z) + c8 * (d.x * d.x - d.y * d.y));
}
