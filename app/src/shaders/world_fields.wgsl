// Accumulate one derived field's occurrences into the originating object's
// scene-sized volume. Green and blue retain its occurrence ordinal, alpha its cut-surface ordinal.
@group(0) @binding(0) var<uniform> info: vec4f;
@group(0) @binding(1) var fieldSampler: sampler;
@group(0) @binding(2) var field: texture_3d<f32>;
@group(0) @binding(3) var previous: texture_3d<f32>;
@group(0) @binding(4) var result: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var<storage, read> instances: array<mat4x4f>;

@group(0) @binding(6) var<storage,read_write> bounds: array<atomic<u32>>;
@group(0) @binding(7) var<storage,read> faceCells: array<u32>;
var<workgroup> localBounds: array<atomic<u32>,6>;

@compute @workgroup_size(8,8,4)
fn measure_bounds(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) local:u32) {
  let res=textureDimensions(field).x;
  if(local<6u) {atomicStore(&localBounds[local],0u);}
  workgroupBarrier();
  if(all(id<vec3u(res)) && textureLoad(field,vec3i(id),0).r<=1.5/f32(res)) {
    for(var axis=0u;axis<3u;axis++) {
      atomicMax(&localBounds[axis],res-id[axis]);
      atomicMax(&localBounds[3u+axis],id[axis]+1u);
    }
  }
  workgroupBarrier();
  if(local<6u) {
    let value=atomicLoad(&localBounds[local]);
    if(value>0u){atomicMax(&bounds[u32(info.w)*6u+local],value);}
  }
}

@compute @workgroup_size(8,8,4)
fn accumulate(@builtin(global_invocation_id) id: vec3u) {
  let res=textureDimensions(result);
  if(any(id>=res)) {return;}
  let p=vec4f(((vec3f(id)+0.5)/vec3f(res)*2.0-1.0)*1.5*info.x,1.0);
  let lim=1.5-1.5/f32(textureDimensions(field).x);
  var nearest=vec4f(60000.0,0.0,0.0,0.0);
  if(info.z<0.5) {nearest=textureLoad(previous,vec3i(id),0);}
  for(var i=0u;i<u32(info.y);i++) {
    let m=instances[i];
    let q=vec3f(dot(m[0],p),dot(m[1],p),dot(m[2],p));
    let bound=clamp(q,vec3f(-lim),vec3f(lim));
    let outside=length(q-bound);
    let scale=m[3].x/info.x;
    if(outside>0.0 && outside*scale>=nearest.r) {continue;}
    var d=textureSampleLevel(field,fieldSampler,bound/3.0+0.5,0.0).r;
    if(outside>0.0) { d=length(vec2f(max(d,0.0),outside)); }
    if(d*scale<nearest.r) {
      let side = u32(m[3].w);
      var face = 0u;
      if (side > 0u && outside == 0.0 && abs(d * scale) < 9.0 / f32(res.x)) {
        let source = clamp(facePoint(q) / 3.0 + 0.5, vec3f(0.0), vec3f(1.0));
        let cell = min(vec3u(source * f32(side)), vec3u(side - 1u));
        face = faceCells[cell.x + side * (cell.y + side * cell.z)] >> 16u;
      }
      nearest = vec4f(d*scale,m[3].yz,f32(face));
    }
  }
  textureStore(result,id,nearest);
}
