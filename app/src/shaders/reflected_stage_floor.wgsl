// Direction-only room maps already contain the unshadowed floor. At a sharp reflection,
// add the local change in radiance at the point where the reflected ray actually reaches it.
fn metalFloorDelta(origin: vec3f, direction: vec3f, hasLampPool: bool) -> vec3f {
  let ray = toRoom(direction);
  if (ray.y >= -1e-5) { return vec3f(0.0); }
  let start = toRoom(origin);
  let distance = (su.mid.w - start.y) / ray.y;
  if (distance <= 0.0 || distance > 2000.0) { return vec3f(0.0); }
  let point = start + distance * ray;
  let coverage = floorCoverage(length(point.xz), ray.y);
  if (coverage <= 0.0) { return vec3f(0.0); }
  let base = su.floor.rgb + su.tint.rgb * sweepAt(ray.y);
  let delta = base * (castFloor(point) - vec3f(1.0)) + castFloorLight(point);
  // The glossy part of the floor still reflects the surroundings. Cast light belongs to its matte part.
  let gloss = floorGloss(ray.y);
  var correction = coverage * (1.0 - gloss) * delta;
  {{FINITE_FLOOR}}
  return correction;
}
