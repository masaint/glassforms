// Evaluate the derivatives of the tricubic B-spline interpolant in object space.
// The two negative derivative weights and the two positive ones each form one linear sample.
fn volumeSlopePlane(a: vec3f, b: vec3f, g: vec3f, axis: u32) -> f32 {
  if (axis == 0u) {
    return mix(mix(vol(vec3f(a.x,a.y,a.z)),vol(vec3f(a.x,b.y,a.z)),g.y),
               mix(vol(vec3f(a.x,a.y,b.z)),vol(vec3f(a.x,b.y,b.z)),g.y),g.z);
  }
  if (axis == 1u) {
    return mix(mix(vol(vec3f(a.x,a.y,a.z)),vol(vec3f(b.x,a.y,a.z)),g.x),
               mix(vol(vec3f(a.x,a.y,b.z)),vol(vec3f(b.x,a.y,b.z)),g.x),g.z);
  }
  return mix(mix(vol(vec3f(a.x,a.y,a.z)),vol(vec3f(b.x,a.y,a.z)),g.x),
             mix(vol(vec3f(a.x,b.y,a.z)),vol(vec3f(b.x,b.y,a.z)),g.x),g.y);
}
fn volumeGradientNormal(p: vec3f) -> vec3f {
  let res = u.vres;
  let t = (toObj(p)/(2.0*HB)+0.5)*res-0.5;
  let i = floor(t);
  let f = t-i;
  let f2 = f*f;
  let f3 = f2*f;
  let w0 = (1.0-3.0*f+3.0*f2-f3)/6.0;
  let w1 = (4.0-6.0*f2+3.0*f3)/6.0;
  let w3 = f3/6.0;
  let g0 = w0+w1;
  let g1 = 1.0-g0;
  let a = (i-0.5+w1/g0)/res;
  let b = (i+1.5+w3/g1)/res;
  let d0 = -0.5*(1.0-f)*(1.0-f);
  let d1 = 1.5*f2-2.0*f;
  let d3 = 0.5*f2;
  let size = -(d0+d1);
  let lo = (i-0.5-d1/size)/res;
  let hi = (i+1.5+d3/size)/res;
  let dx = volumeSlopePlane(vec3f(hi.x,a.yz),b,g1,0u)-volumeSlopePlane(vec3f(lo.x,a.yz),b,g1,0u);
  let dy = volumeSlopePlane(vec3f(a.x,hi.y,a.z),b,g1,1u)-volumeSlopePlane(vec3f(a.x,lo.y,a.z),b,g1,1u);
  let dz = volumeSlopePlane(vec3f(a.xy,hi.z),b,g1,2u)-volumeSlopePlane(vec3f(a.xy,lo.z),b,g1,2u);
  let local = size*vec3f(dx,dy,dz);
  return normalize(local.x*vec3f(u.obj0.x,u.obj1.x,u.obj2.x) + local.y*vec3f(u.obj0.y,u.obj1.y,u.obj2.y) + local.z*vec3f(u.obj0.z,u.obj1.z,u.obj2.z));
}
