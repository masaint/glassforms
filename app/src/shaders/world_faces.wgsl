@group(0) @binding(47) var<storage,read> wfData: array<u32>;
@group(0) @binding(65) var wfProjected0: texture_3d<f32>;
@group(0) @binding(66) var wfProjected1: texture_3d<f32>;
@group(0) @binding(67) var wfProjected2: texture_3d<f32>;
@group(0) @binding(68) var wfProjected3: texture_3d<f32>;
var<private> wfCurrent: array<u32,4>;
var<private> wfFrames:array<u32,4>;
var<private> wfValues: array<MU,4>;
fn wfF(i:u32)->f32 {return bitcast<f32>(wfData[i]);}
fn wfRow(i:u32)->vec4f{return vec4f(wfF(i),wfF(i+1u),wfF(i+2u),wfF(i+3u));}
fn wfMat(k:i32)->MU {
  if(wfCurrent[k]!=0u){return wfValues[k];}
  return mb.b[k];
}
fn wfKind(k:i32)->f32 {if(wfCurrent[k]!=0u){return wfF(wfCurrent[k]);}return mb.kind[k];}
fn wfIor(k:i32)->f32 {if(wfCurrent[k]!=0u){return wfF(wfCurrent[k]+1u);}return mb.ior[k];}
fn wfDisp(k:i32)->f32 {if(wfCurrent[k]!=0u){return wfF(wfCurrent[k]+2u);}return mb.disp[k];}
fn wfHas(k:i32)->f32 {if(wfCurrent[k]!=0u){return wfF(wfCurrent[k]+3u);}return mb.has[k];}
fn wfAbsorb(k:i32)->vec4f {if(wfCurrent[k]!=0u){return wfRow(wfCurrent[k]+4u);}return mb.absorb[k];}
fn wfAssign(k:i32,row:u32,frame:u32) {
  let changed = wfCurrent[k] != row;
  if (changed && row != 0u) { wfLoad(row,k); }
  if (row != 0u && (changed || wfFrames[k] != frame)) {
    // First move into the selected occurrence, then into its material-owning layer.
    // Always read the local frame from the authored row. The previous occurrence
    // may use the same finish with a different placement.
    let x = wfRow(frame+4u);
    let y = wfRow(frame+8u);
    let z = wfRow(frame+12u);
    for (var i=0u; i<3u; i++) {
      let local = wfRow(row+{{FACE_FRAME}}u+4u*i);
      wfValues[k].gratingFrame[i] = local.x*x + local.y*y + local.z*z + vec4f(0.0,0.0,0.0,local.w);
    }
  }
  wfCurrent[k] = row;
  wfFrames[k] = frame;
}
fn wfReset() {
  for (var i=0; i<4; i++) { wfCurrent[i]=0u; wfFrames[i]=0u; }
}
fn wfRestore(saved:array<u32,4>, frames:array<u32,4>) {
  for (var k=0; k<COUNT; k++) { wfAssign(k,saved[k],frames[k]); }
}
// A later interface can disperse light even if the primary surface is achromatic.
fn wfDispersive()->bool { return wfData[1] != 0u; }
fn wfProjected(source:u32, cell:vec3i)->vec4f {
  switch source {
    case 0u: {return textureLoad(wfProjected0, cell, 0);}
    case 1u: {return textureLoad(wfProjected1, cell, 0);}
    case 2u: {return textureLoad(wfProjected2, cell, 0);}
    default: {return textureLoad(wfProjected3, cell, 0);}
  }
}
fn wfPick(k:i32,p:vec3f) {
  if(k<0){return;}
  let scene=vec4f(toObj(p),1.0);
  var best=1e8;
  var row=0u;
  var frame=0u;
  for(var layer=0u;layer<wfData[0];layer++){
    let h=4u+16u*layer;
    if(wfData[h]!=u32(k)+1u){continue;}
    let format=wfData[h+1u];
    let res=format & 0x7fffffffu;
    if ((format & 0x80000000u) != 0u) {
      let scale=wfF(h+4u);
      let uv=scene.xyz/(3.0*scale)+0.5;
      if(any(uv<vec3f(0.0)) || any(uv>=vec3f(1.0))){continue;}
      let sample=wfProjected(wfData[h+2u],vec3i(uv*f32(res)));
      let distance=abs(sample.x);
      let score=distance*scale;
      if(distance>2.6*3.0/f32(res) || score>=best){continue;}
      best=score;
      let occurrence=u32(sample.y)+1024u*u32(sample.z);
      frame=wfData[h+5u]+16u*occurrence;
      let ordinal=u32(sample.w);
      if(ordinal==0u){row=0u;}else{row=wfData[h+3u]+(ordinal-1u)*{{RECORD}}u;}
      continue;
    }
    let r0=wfRow(h+4u);
    let local=vec3f(dot(r0,scene),dot(wfRow(h+8u),scene),dot(wfRow(h+12u),scene));
    let uv=local/3.0+0.5;
    if(any(uv<vec3f(0.0)) || any(uv>=vec3f(1.0))){continue;}
    let i=vec3u(uv*f32(res));
    let packed=wfData[wfData[h+2u]+i.x+res*(i.y+res*i.z)];
    let distance=abs(unpack2x16float(packed).x);
    let score=distance/max(length(r0.xyz),1e-5);
    if(distance>2.6*3.0/f32(res) || score>=best){continue;}
    best=score;
    frame=h;
    let ordinal=packed>>16u;
    if(ordinal==0u){row=0u;}else{row=wfData[h+3u]+(ordinal-1u)*{{RECORD}}u;}
  }
  wfAssign(k,row,frame);
}
