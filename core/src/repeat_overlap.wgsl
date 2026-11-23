struct RepeatedHit { q: vec3f, d: f32, part: i32, angle: f32, fold: f32 }
fn repeatVisit(m: ModE, H: ModE, raw: vec3f, angle: f32, best: ptr<function,RepeatedHit>) {
  var q=raw; var fold=1.0;
  if (i32(m.q0.y)==1 && m.q0.w>0.5) { fold=select(-1.0,1.0,q.z>=0.0); q.z=abs(q.z); }
  if (i32(m.q0.y)==1) { q.x-=m.q3.w; }
  let record=csgP[{stage}];
  let delta=max(record.s0.xyz-q,q-record.s1.xyz);
  if(max(delta.x,max(delta.y,delta.z))>max((*best).d,0.0)) { return; }
  let x=modWorld(m,q);
  let y=vec3f(dot(H.q4,vec4f(x,1.0)),dot(H.q5,vec4f(x,1.0)),dot(H.q6,vec4f(x,1.0)));
  let hit={sample};
  let d=hit.x/H.q0.y;
  if(d<(*best).d) { *best=RepeatedHit(y,d,i32(hit.y),angle,fold); }
}
fn repeatHit(p: vec3f) -> RepeatedHit {
  let H=modAt(0); let m=modAt(1);
  let x=vec3f(dot(H.q1,vec4f(p,1.0)),dot(H.q2,vec4f(p,1.0)),dot(H.q3,vec4f(p,1.0)));
  let v=modLocal(m,x)/m.q0.z;
  var best=RepeatedHit(vec3f(0.0),1e9,-1,0.0,1.0);
  let n=vec3i(m.q5.xyz);
  if(i32(m.q0.y)==1) {
    let step=6.2831853/f32(n.x);
    let seed=i32(round(atan2(v.z,v.x)/step));
    let record=csgP[{stage}];
    var count=n.x;
    for(var j=0;j<count;j++) {
      let k=seed+select(-((j+1)/2),j/2,(j&1)==0);
      let idx=((k%n.x)+n.x)%n.x;
      let cs=csgP[{rows}+idx].inv0.xy;
      let q=vec3f(dot(v.xz,cs),v.y,dot(v.xz,vec2f(-cs.y,cs.x)));
      repeatVisit(m,H,q,f32(idx),&best);
      if(j==0 && count>1) {
        // The motif plus the current closest distance subtends this angular
        // interval. Every sector further away lies outside its expanded box.
        let reach=max(best.d,0.0); let x=record.s0.x+m.q3.w-reach;
        if(x>0.0) {
          let z=max(abs(record.s0.z),abs(record.s1.z))+reach;
          let half=atan2(z,x);
          count=min(count,1+2*i32(ceil(half/step+0.5)));
        }
      }
    }
  } else {
    let step=m.q6.xyz; let mid=0.5*(vec3f(n)-1.0);
    let seed=clamp(vec3i(round(v/step+mid)),vec3i(0),n-1);
    let offset=vec3f(f32(seed.y&1)*m.q7.w*step.x,0.0,0.0);
    let centre=clamp(i32(round((v.x-offset.x)/step.x+mid.x)),0,n.x-1);
    repeatVisit(m,H,v-(vec3f(f32(centre),f32(seed.y),f32(seed.z))-mid)*step-offset,0.0,&best);
    let record=csgP[{stage}]; let reach=max(best.d,0.0);
    let lo=(v-record.s1.xyz-reach)/step+mid;
    let hi=(v-record.s0.xyz+reach)/step+mid;
    let first=clamp(vec3i(ceil(lo)),vec3i(0),n-1);
    let last=clamp(vec3i(floor(hi)),vec3i(0),n-1);
    for(var z=first.z;z<=last.z;z++) { for(var y=first.y;y<=last.y;y++) {
      let shift=f32(y&1)*m.q7.w;
      let a=clamp(i32(ceil(lo.x-shift)),0,n.x-1); let b=clamp(i32(floor(hi.x-shift)),0,n.x-1);
      for(var i=a;i<=b;i++) {
        if (i==centre && y==seed.y && z==seed.z) { continue; }
        let q=v-(vec3f(f32(i)+shift,f32(y),f32(z))-mid)*step;
        repeatVisit(m,H,q,0.0,&best);
      }
    } }
  }
  return best;
}
fn csgDistW(p: vec3f, only: i32) -> vec2f {
  let hit=repeatHit(p);
  return vec2f(hit.d*modAt(1).q0.z*modAt(0).q0.y,0.0);
}
fn csgWarpGradient(p: vec3f) -> vec4f {
  let H=modAt(0);let m=modAt(1);let hit=repeatHit(p);var g=vec3f(0.0);
  for(var j=0u;j<4u+u32(csgU.info2.y);j++){
    let tap=csgTap(j);
    g+=tap*READ_HIT;
  }
  let fw=mat3x3f(H.q4.xyz,H.q5.xyz,H.q6.xyz);
  let frame=mat3x3f(m.q2.xyz,m.q3.xyz,m.q4.xyz);
  var local=(fw*g)*frame;
  local.z*=hit.fold;
  if(i32(m.q0.y)==1) { let cs=csgP[{rows}+i32(hit.angle)].inv0.xy; local=vec3f(dot(local.xz,vec2f(cs.x,-cs.y)),local.y,dot(local.xz,cs.yx)); }
  let n=frame*local;
  return vec4f(H.q1.xyz*n.x+H.q2.xyz*n.y+H.q3.xyz*n.z,4.0*SDF_NE*hit.d*H.q0.y*m.q0.z);
}
