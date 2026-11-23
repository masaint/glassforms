// Select finite sources by their contrast with the background and their projected solid angle here.
// The selection weight may be approximate. Each directional PDF below is exact for its proposal.
fn glossyLampWeights(p: vec3f, n: vec3f) -> FrLampMix {
  var distribution: FrLampMix;
  for (var i = 0; i < i32(lamps.n.x); i++) {
    let light = lamps.l[i];
    let delta = light.p.xyz - p;
    let distance = max(length(delta), 1e-6);
    let axis = select(n, delta / distance, distance > 1e-6);
    var radius = light.u.w;
    var area = 3.141592654 * radius * radius;
    if (light.p.w > 1.5) {
      radius = length(vec2f(light.u.w, light.v.w));
      area = 4.0 * light.u.w * light.v.w * max(0.0, -dot(axis, cross(light.u.xyz, light.v.xyz)));
    } else if (light.p.w > 0.5) {
      radius += light.v.w;
      area += 4.0 * light.u.w * light.v.w * length(cross(axis, light.u.xyz));
    }
    if (area <= 0.0 || dot(delta, n) + radius <= 0.0) { continue; }
    let luminance = vec3f(0.2126, 0.7152, 0.0722);
    let emission = light.rgb.rgb;
    let brightness = dot(emission, luminance);
    if (brightness <= 0.0) { continue; }
    // An emitter with the same radiance as the environment it covers supplies no highlight.
    // Guiding towards it adds variance, especially to the energy-normalized coated lobe.
    let background = envMap(toEnv(axis));
    let difference = dot(max(emission - background, vec3f(0.0)), luminance);
    let contrast = smoothstep(0.01, 0.5, difference / brightness);
    let weight = difference * contrast * min(6.283185307, area / (distance * distance));
    distribution.contrast = max(distribution.contrast, contrast);
    distribution.weights[i] = weight;
    distribution.total += weight;
  }
  return distribution;
}

// The sphere circumscribes the whole capsule for a tube. The cone of directions covers the
// full sphere when the shading point is inside it, and otherwise ends at its apparent horizon.
fn glossyLampCap(p: vec3f, i: i32) -> vec4f {
  let light = lamps.l[i];
  let delta = light.p.xyz - p;
  let distance = max(length(delta), 1e-7);
  let radius = select(light.u.w, light.u.w + light.v.w, light.p.w > 0.5);
  let sin2 = min(1.0, (radius/distance) * (radius/distance));
  let extent = select(sin2 / (1.0 + sqrt(1.0 - sin2)), 2.0, distance <= radius);
  return vec4f(select(vec3f(0.0, 1.0, 0.0), delta/distance, distance > 1e-6), extent);
}

fn glossyLampRay(p: vec3f, distribution: FrLampMix, r: vec2f) -> vec3f {
  let draw = r.x * distribution.total;
  var cumulative = 0.0;
  var selected = 0;
  var previous = 0.0;
  for (var i = 0; i < i32(lamps.n.x); i++) {
    if (distribution.weights[i] <= 0.0) { continue; }
    selected = i;
    previous = cumulative;
    cumulative += distribution.weights[i];
    if (draw < cumulative) { break; }
  }
  let uv = vec2f(clamp((draw - previous) / distribution.weights[selected], 0.0, 1.0), r.y);
  let light = lamps.l[selected];
  if (light.p.w > 1.5) {
    let offset = 2.0 * uv - 1.0;
    return normalize(light.p.xyz - p + light.u.xyz*(light.u.w*offset.x) + light.v.xyz*(light.v.w*offset.y));
  }
  return frCapDirection(glossyLampCap(p, selected), uv);
}

fn glossyLampPdf(p: vec3f, distribution: FrLampMix, wo: vec3f) -> f32 {
  var density = 0.0;
  for (var i = 0; i < i32(lamps.n.x); i++) {
    if (distribution.weights[i] <= 0.0) { continue; }
    let light = lamps.l[i];
    if (light.p.w > 1.5) {
      let normal = cross(light.u.xyz, light.v.xyz);
      let cosine = -dot(normal, wo);
      if (cosine <= 0.0) { continue; }
      let distance = -dot(light.p.xyz - p, normal) / cosine;
      let offset = p + distance*wo - light.p.xyz;
      if (distance > 0.0 && abs(dot(offset, light.u.xyz)) <= light.u.w && abs(dot(offset, light.v.xyz)) <= light.v.w) {
        density += distribution.weights[i] * distance * distance / (4.0 * light.u.w * light.v.w * cosine);
      }
    } else {
      let cap = glossyLampCap(p, i);
      if (dot(wo, cap.xyz) >= 1.0 - cap.w) {
        density += distribution.weights[i] / (6.283185307 * max(cap.w, 1e-12));
      }
    }
  }
  return density / distribution.total;
}
