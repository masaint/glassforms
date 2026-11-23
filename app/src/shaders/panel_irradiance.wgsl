// Irradiance of a one-sided rectangle from its horizon-clipped spherical polygon.
// The vector sum of its great-circle edges integrates the clamped cosine exactly.
fn panelIntegral(center: vec3f, du: vec3f, dv: vec3f, n: vec3f) -> f32 {
  let corners = array<vec3f,4>(center-du-dv, center+du-dv, center+du+dv, center-du+dv);
  var polygon: array<vec3f, 6>;
  var count = 0;
  var a = corners[3];
  var ca = dot(n,a);
  for (var i=0; i<4; i++) {
    let b = corners[i];
    let cb = dot(n,b);
    if ((ca>0.0) != (cb>0.0)) {
      polygon[count] = normalize(mix(a,b,ca/(ca-cb)));
      count++;
    }
    if (cb>0.0) { polygon[count] = normalize(b); count++; }
    a = b;
    ca = cb;
  }
  if (count<3) { return 0.0; }
  var edge = 0.0;
  var prev = polygon[count-1];
  for (var i=0; i<count; i++) {
    let next = polygon[i];
    let axis = cross(prev,next);
    let sine = length(axis);
    edge += dot(n,axis)*atan2(sine,dot(prev,next))/max(sine,1e-8);
    prev = next;
  }
  return 0.5*abs(edge);
}

fn panelRandom(state: ptr<function,u32>) -> f32 {
  *state = *state*747796405u+2891336453u;
  let w = ((*state >> ((*state >> 28u)+4u)) ^ *state)*277803737u;
  return f32(((w >> 22u) ^ w) >> 8u)/16777216.0;
}

// xyz points from receiver to sampled emitter position. w is the irradiance weight.
// Draw proportionally to the projected cosine, leaving just visibility to sample.
// A bounded rejection loop falls back to an independent ordinary area sample.
fn panelSample(center: vec3f, uAxis: vec4f, vAxis: vec4f, n: vec3f, seed: vec2f) -> vec4f {
  let du = uAxis.xyz*uAxis.w;
  let dv = vAxis.xyz*vAxis.w;
  let plane = -dot(center,cross(uAxis.xyz,vAxis.xyz));
  let maxCos = dot(n,center)+abs(dot(n,du))+abs(dot(n,dv));
  if (plane<=1e-6 || maxCos<=0.0) { return vec4f(center,0.0); }
  let nearest = center-uAxis.xyz*clamp(dot(center,uAxis.xyz),-uAxis.w,uAxis.w)
                      -vAxis.xyz*clamp(dot(center,vAxis.xyz),-vAxis.w,vAxis.w);
  let minDist2 = max(dot(nearest,nearest),1e-12);
  let envelope = min(1.0/minDist2, plane*min(maxCos,sqrt(minDist2))/(minDist2*minDist2));
  let bits = vec2u(seed*65536.0);
  var state = (bits.x | (bits.y << 16u)) ^ 0x80b4d701u;
  var d = center;
  var weight = 0.0;
  for (var i=0; i<=8; i++) {
    let x = 2.0*panelRandom(&state)-1.0;
    let y = 2.0*panelRandom(&state)-1.0;
    d = center+x*du+y*dv;
    let d2 = max(dot(d,d),1e-12);
    weight = max(dot(n,d),0.0)*plane/(d2*d2);
    if (i==8) { break; }
    if (panelRandom(&state)*envelope<weight) {
      return vec4f(d,panelIntegral(center,du,dv,n));
    }
  }
  return vec4f(d,weight*4.0*uAxis.w*vAxis.w);
}
