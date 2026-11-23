fn coneRoots(ro:vec3f,rd:vec3f,k:f32,base:f32,lo:f32,hi:f32)->f32 {
  let r=k*ro.y+base; let change=k*rd.y;
  let a=dot(rd.xz,rd.xz)-change*change;let b=dot(rd.xz,ro.xz)-r*change;
  let c=dot(ro.xz,ro.xz)-r*r;let disc=b*b-a*c;
  var best=1e7;
  if (disc>=0.0 && abs(a)>1e-7) {
    let root=sqrt(disc);
    // This expression retains the near root when b is much larger than c.
    let q=-b-select(-root,root,b>=0.0);
    let first=q/a; let second=select(1e7,c/q,abs(q)>1e-9);
    if (first>=lo && first<=hi && r+first*change>=0.0) { best=first; }
    if (second>=lo && second<=hi && r+second*change>=0.0) { best=min(best,second); }
  } else if (abs(a)<=1e-7 && abs(b)>1e-9) {
    let hit=-0.5*c/b;
    if (hit>=lo && hit<=hi && r+hit*change>=0.0) { best=hit; }
  }
  return best;
}
fn coneEntry(origin:vec3f,direction:vec3f,far:f32)->f32 {
  let ro=toObj(origin); let rd=toObj(direction); let m=modAt(1);
  // A flat profile still tiles correctly. Its axial cone equation has no slope.
  if (m.q5.y < 1e-6) { return 0.0; }
  let half=0.5*m.q5.y;
  let extra=m.q6.z+4.0*far*u.tan_fov/u.res.y;
  let side=half+extra;var lo=0.0;var hi=far;
  if(abs(rd.y)>1e-7) {
    let ends=(vec2f(-side,side)-ro.y)/rd.y;
    lo=max(lo,min(ends.x,ends.y)); hi=min(hi,max(ends.x,ends.y));
  } else if(abs(ro.y)>side) { return far+1.0; }
  if(lo>hi) { return far+1.0; }
  let k=(m.q5.z-m.q5.x)/m.q5.y;
  let stretch=sqrt(1.0+k*k);
  let mid=0.5*(m.q5.x+m.q5.z)+m.q4.w*stretch;
  let spread=extra*stretch;
  let at=ro+lo*rd;
  let separation=length(at.xz)-k*at.y-mid;
  if(abs(separation)<=spread){return max(lo-0.0005,0.0);}
  // Before reaching the shell, the ray must cross the boundary facing it.
  let boundary=mid+select(-spread,spread,separation>0.0);
  return max(coneRoots(ro,rd,k,boundary,lo,hi)-0.0005,0.0);
}
