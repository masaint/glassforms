@group(0) @binding(0) var field: texture_3d<f32>;
@group(0) @binding(1) var<storage, read_write> bound: atomic<u32>;
var<workgroup> maxima: array<f32,64>;

// The greatest value of |x|^2.5 + |y|^2.5 + |z|^2.5 on a column comes from
// its farthest occupied depth. One texel of padding encloses the whole support
// of any trilinear sample that can cross the marcher's 0.0008 threshold.
@compute @workgroup_size(8,8,1)
fn measure(@builtin(global_invocation_id) id:vec3u, @builtin(local_invocation_index) lane:u32) {
  let n=textureDimensions(field).x;
  var h=0.0;
  if(all(id.xy<vec2u(n))){
    for(var k=0u;k<(n+1u)/2u;k++){
      let other=n-1u-k;
      let a=textureLoad(field,vec3i(vec3u(id.xy,k)),0).r;
      let b=textureLoad(field,vec3i(vec3u(id.xy,other)),0).r;
      if(min(a,b)<0.000801){
        // A contributing border texel would extend outside the domain under clamp-to-edge.
        if(k==0u || any(id.xy==vec2u(0)) || any(id.xy==vec2u(n-1u))){
          h=1e6;
        }else{
          let voxel=3.0/f32(n);
          let p=abs((vec3f(vec3u(id.xy,k))+.5)*voxel-1.5)+vec3f(voxel);
          h=dot(p*p,sqrt(p))+.00005;
        }
        break;
      }
    }
  }
  maxima[lane]=h;
  workgroupBarrier();
  for(var stride=32u;stride>0u;stride/=2u){
    if(lane<stride){maxima[lane]=max(maxima[lane],maxima[lane+stride]);}
    workgroupBarrier();
  }
  if(lane==0u){atomicMax(&bound,bitcast<u32>(maxima[0]));}
}
