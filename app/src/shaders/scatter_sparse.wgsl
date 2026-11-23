// The exact stencil updates a cell only when at least one face is open. After
// three ordinary red/black steps both ping-pong textures have zero radiance in
// a closed brick. Set that fixed point directly in both textures. Its neighbours
// have no open face to it and therefore see exactly the same flux.
@group(0) @binding(19) var closedA:texture_storage_3d<rgba32float,write>;
@group(0) @binding(20) var closedB:texture_storage_3d<rgba32float,write>;
@group(0) @binding(17) var<storage,read_write> occupiedBricks:array<u32>;
@group(0) @binding(18) var<storage,read_write> occupiedArgs:array<atomic<u32>>;
var<workgroup> openCells:array<atomic<u32>,2>;
@compute @workgroup_size(4,4,4)
fn cs_occupied(@builtin(global_invocation_id) id:vec3u,@builtin(workgroup_id) g:vec3u,@builtin(local_invocation_index) lid:u32) {
  if(lid<2u){atomicStore(&openCells[lid],0u);}
  workgroupBarrier();
  let N=i32(su.res);
  if(all(id<vec3u(u32(N)))){
    let cell=vec3i(id);
    let h=2.0*su.hb/su.res;
    let d=textureLoad(srcTex,cell,0).a/h;
    for(var i=0;i<6;i++){
      let q=cell+NEAR[i];
      if(any(q<vec3i(0)) || any(q>=vec3i(N))){continue;}
      if(share(0.5*(d+textureLoad(srcTex,q,0).a/h))>0.0){atomicOr(&openCells[lid/32u],1u<<(lid%32u));break;}
    }
  }
  workgroupBarrier();
  let groups=(u32(N)+3u)/4u;
  let brick=g.x+groups*(g.y+groups*g.z);
  let first=groups*groups*groups+2u*brick;
  if(lid<2u){occupiedBricks[first+lid]=atomicLoad(&openCells[lid]);}
  if((atomicLoad(&openCells[0])|atomicLoad(&openCells[1]))==0u){
    if(all(id<vec3u(u32(N)))){
      let d=textureLoad(srcTex,vec3i(id),0).a/(2.0*su.hb/su.res);
      let unchanged=vec4f(0.0,0.0,0.0,d);
      textureStore(closedA,id,unchanged);
      textureStore(closedB,id,unchanged);
    }
  }else if(lid==0u){
    occupiedBricks[atomicAdd(&occupiedArgs[0],1u)]=brick;
  }
}
fn occupiedCell(i:u32,cell:vec3u)->vec3u {
  let groups=(u32(su.res)+3u)/4u;let b=occupiedBricks[i];
  return 4u*vec3u(b%groups,(b/groups)%groups,b/(groups*groups))+cell;
}
@compute @workgroup_size(4,4,4)
fn cs_sparse_even(@builtin(workgroup_id) g:vec3u,@builtin(local_invocation_id) i:vec3u){relax(occupiedCell(g.x,i),0u);}
@compute @workgroup_size(4,4,4)
fn cs_sparse_odd(@builtin(workgroup_id) g:vec3u,@builtin(local_invocation_id) i:vec3u){relax(occupiedCell(g.x,i),1u);}

// The dense balance reduction includes only cells with an open face. Walk the
// classifier's two masks in the original 0..63 order to retain every rounding.
@compute @workgroup_size(4,4,4)
fn cs_sparse_sum(@builtin(global_invocation_id) id:vec3u){
    let groups=(u32(su.res)+3u)/4u;
    if(any(id>=vec3u(groups))){return;}
    let brick=id.x+groups*(id.y+groups*id.z);
    let first=groups*groups*groups+2u*brick;
    var add=vec3f(0.0);var lose=vec3f(0.0);
    for(var half=0u;half<2u;half++){
        var live=occupiedBricks[first+half];
        while(live!=0u){
            let j=half*32u+firstTrailingBit(live);
            let me=vec3i(id*4u+vec3u(j%4u,(j/4u)%4u,j/16u));
            let c=textureLoad(phiTex,me,0);
            add+=su.k.rgb*su.ss.rgb*textureLoad(srcTex,me,0).rgb;
            lose+=(skin(c.a)*su.zb.rgb+su.ab.rgb*share(c.a))*c.rgb;
            live&=live-1u;
        }
    }
    textureStore(addOut,id,vec4f(add,0.0));textureStore(loseOut,id,vec4f(lose,0.0));
}
