// Fine surface marks in the owning object's coordinates.
fn polishingPads(q: vec3f, nq: vec3f, px: f32) -> vec3f {
  // The slowest ring is 140 radians per object unit. The pixel filter removes every ring here.
  if (px * 140.0 >= 1.0) { return vec3f(0.0); }
  let lower = vec3i(floor((q - 0.5) / 0.7));
  let upper = vec3i(floor((q + 0.5) / 0.7));
  var ring = vec3f(0.0);
  // Only cells touched by the radius of this pad can supply a centre. The remaining
  // interval along X follows from the distance to this Y/Z column of cells.
  for (var z = lower.z; z <= upper.z; z++) {
    let dz = max(abs(q.z - (f32(z) + 0.5) * 0.7) - 0.35, 0.0);
    for (var y = lower.y; y <= upper.y; y++) {
      let dy = max(abs(q.y - (f32(y) + 0.5) * 0.7) - 0.35, 0.0);
      let remaining = 0.25 - dz * dz - dy * dy;
      if (remaining <= 0.0) { continue; }
      let reach = sqrt(remaining);
      let firstX = i32(floor((q.x - reach) / 0.7));
      let lastX = i32(floor((q.x + reach) / 0.7));
      for (var x = firstX; x <= lastX; x++) {
        let c = vec3i(x, y, z);
        let o = (vec3f(c) + 0.5 + 0.5 * vec3f(wHash(c + vec3i(1, 2, 3)), wHash(c + vec3i(4, 5, 6)), wHash(c + vec3i(7, 8, 9)))) * 0.7;
        let delta = q - o;
        let radius = length(delta);
        if (radius >= 0.5) { continue; }
        let r = delta - nq * dot(delta, nq);
        let rl = length(r);
        if (rl < 1e-4) { continue; }
        // fine rings, broken into arcs, fading towards the pad's edge
        let f = 220.0 + 80.0 * wHash(c);
        if (px * f >= 1.0) { continue; }
        // A pad keeps its own interrupted arcs across cell edges and around a full turn.
        let seed = (wHash(c + vec3i(4, 8, 2)) + 1.0) * 4.0;
        let arcs = smoothstep(0.2, 0.7, wNoise(3.0 * r / rl + vec3f(6.0 * rl, seed, seed * 1.73)));
        let lineFine = 1.0 - smoothstep(0.3, 1.0, px * f);
        ring += (r / rl) * (sin(rl * f) * arcs * lineFine * (1.0 - smoothstep(0.3, 0.5, radius)));
      }
    }
  }
  return ring;
}

fn fingerMarks(q: vec3f, nq: vec3f, px: f32, smA: f32) -> f32 {
  // A mark reaches less than half a cell. Include the nearest neighbour on either side.
  let cell = vec3i(floor(q / 0.45 - 0.5));
  var sm = 0.0;
  let resolved = 1.0 - smoothstep(0.3, 1.0, px * 260.0);
  for (var j = 0; j < 8; j++) {
    let c = cell + vec3i(j % 2, (j / 2) % 2, j / 4);
    if (0.5 + 0.5 * wHash(c + vec3i(11, 13, 17)) > 0.55 * smA + 0.1) { continue; }
    let o = (vec3f(c) + 0.5 + 0.5 * vec3f(wHash(c + vec3i(2, 3, 1)), wHash(c + vec3i(5, 1, 4)), wHash(c + vec3i(8, 7, 3)))) * 0.45;
    let r = q - o;
    let rl = length(r - nq * dot(r, nq));
    let pad = (1.0 - smoothstep(0.03, 0.09, rl)) * (1.0 - smoothstep(0.0, 0.08, abs(dot(r, nq))));
    if (pad <= 0.0) { continue; }
    // (the ridges of a fingertip: whorls a third of a millimetre apart at life size, averaged away
    // where they are finer than a pixel)
    var ridges = 0.6;
    if (resolved > 0.0) { ridges = mix(0.6, 0.5 + 0.5 * sin(rl * 260.0 + 4.0 * wNoise(q * 30.0)), resolved); }
    sm = max(sm, pad * ridges);
  }
  return sm;
}
