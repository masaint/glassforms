// Evaluate the angular density of the visible, masked GGX reflector per unit
// of its integrated single-scattering energy. A coated substrate folds its
// sub-surface directions back into the visible hemisphere.
fn roughLampDensity(n: vec3f, rd: vec3f, wo: vec3f, a2: f32, geometric: f32, coated: bool) -> f32 {
  let nl = dot(n,wo);
  if (nl <= 0.0) { return 0.0; }
  let nv = max(-dot(n,rd),1e-5);
  let nh = max(dot(n,normalize(wo-rd)),0.0);
  let h = max(1.0-(1.0-a2)*nh*nh,1e-7);
  let d = a2/(3.14159265*h*h);
  let gv = sqrt(a2+(1.0-a2)*nv*nv);
  if (coated) {
    var pdf = d/(2.0*(nv+gv));
    let back = reflect(wo,n)-rd;
    if (dot(back,n)>0.0) {
      let nh2 = dot(normalize(back),n);
      let h2 = max(1.0-(1.0-a2)*nh2*nh2,1e-7);
      pdf += a2/(6.2831853*h2*h2*(nv+gv));
    }
    return pdf;
  }
  let gl = sqrt(a2+(1.0-a2)*nl*nl);
  return d*nl/(2.0*(nl*gv+nv*gl)*geometric);
}

fn roughLampAt(p: vec3f, n: vec3f, rd: vec3f, mid: vec3f, var0: f32, bounce: i32,
               alpha: f32, coated: bool, geometric: f32, incident: vec3f, i: i32) -> vec3f {
  let lamp = lamps.l[i];
  let aim = lamp.p.xyz-p;
  let distance = max(length(aim),1e-5);
  let axis = aim/distance;
  let bound = lamp.u.w + select(0.0,lamp.v.w,lamp.p.w>0.5);
  let apparent = bound/distance;
  let useArea = smoothstep(0.20,0.65,alpha/max(apparent,1e-5))*select(1.0,0.0,bounce>0);
  if (dot(n,axis) < -apparent && useArea >= 1.0) { return vec3f(0.0); }
  var blurred = vec3f(0.0);
  if (useArea < 1.0) {
    let count = f32(bounce+1);
    let outer = sqrt(min(var0+11.0*min(var0,2.0*count*alpha*alpha),4.0));
    let coverage = mix(lampHit(i,p,mid,1e9,sqrt(var0)).x,lampHit(i,p,mid,1e9,outer).x,1.0-exp(-.2877*count));
    blurred = coverage*(lamp.rgb.rgb-incident);
    if (useArea <= 0.0) { return blurred; }
  }
  var area = vec3f(0.0);
  let a2 = max(alpha*alpha,1e-8);
  if (lamp.p.w > 1.5) {
    let ln = cross(lamp.u.xyz,lamp.v.xyz);
    if (dot(ln,axis) >= 0.0) {return mix(blurred,vec3f(0.0),useArea);}
    for (var j=0;j<4;j++) {
      let uv = vec2f(select(-.577350269,.577350269,(j&1)!=0),select(-.577350269,.577350269,(j&2)!=0));
      let q = aim + uv.x*lamp.u.w*lamp.u.xyz + uv.y*lamp.v.w*lamp.v.xyz;
      let d2 = max(dot(q,q),1e-8);
      let wo = q*inverseSqrt(d2);
      let solid = lamp.u.w*lamp.v.w*max(-dot(ln,wo),0.0)/d2;
      area += solid*roughLampDensity(n,rd,wo,a2,geometric,coated)*(lamp.rgb.rgb-envL0(wo,0.0));
    }
    let edge = 1.0-.5*max(lamp.rgb.w,.02);
    area *= edge*edge;
  } else {
    let tangent = normalize(cross(axis,select(vec3f(0.0,1.0,0.0),vec3f(1.0,0.0,0.0),abs(axis.y)>.90)));
    let bitangent = cross(axis,tangent);
    let sin2 = min(1.0,apparent*apparent);
    let extent = select(sin2/(1.0+sqrt(1.0-sin2)),2.0,distance<=bound);
    for (var j=0;j<8;j++) {
      let u = (f32(j)+.5)/8.0;
      let azimuth = 2.39996323*f32(j);
      let z = 1.0-u*extent;
      let wo = axis*z + sqrt(max(0.0,1.0-z*z))*(tangent*cos(azimuth)+bitangent*sin(azimuth));
      let coverage = lampHit(i,p,wo,1e9,0.0).x;
      area += coverage*roughLampDensity(n,rd,wo,a2,geometric,coated)*(lamp.rgb.rgb-envL0(wo,0.0));
    }
    area *= 6.2831853*extent/8.0;
  }
  return mix(blurred,area,useArea);
}

fn roughLampRadiance(p: vec3f, n: vec3f, rd: vec3f, mid: vec3f, var0: f32, bounce: i32, alpha: f32,
                     coated: bool, geometric: f32, incident: vec3f) -> vec3f {
  var result = incident;
  for(var i=0;i<i32(lamps.n.x);i++) {
    result += roughLampAt(p,n,rd,mid,var0,bounce,alpha,coated,geometric,incident,i);
  }
  return max(result,vec3f(0.0));
}
