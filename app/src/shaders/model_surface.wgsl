@group(0) @binding(45) var<storage,read> fmData: array<u32>;
@group(0) @binding(46) var<storage,read> fmInstances: array<u32>;
fn fmFloat(i:u32)->f32 { return bitcast<f32>(fmData[i]); }
fn fmV2(i:u32)->vec2f { return vec2f(fmFloat(i),fmFloat(i+1u)); }
fn fmV3(i:u32)->vec3f { return vec3f(fmFloat(i),fmFloat(i+1u),fmFloat(i+2u)); }
fn fmRow(i:u32)->vec4f {
  return bitcast<vec4f>(vec4u(fmInstances[i],fmInstances[i+1u],fmInstances[i+2u],fmInstances[i+3u]));
}
fn fmWrap(i:i32, n:i32, mode:u32)->i32 {
  if (mode == 33071u) { return clamp(i,0,n-1); }
  let period = select(n,2*n,mode==33648u);
  let k = ((i%period)+period)%period;
  return select(k,2*n-1-k,k>=n);
}
fn fmTexel(start:u32, extent:vec2i, coord:vec2i, wrap:vec2u, colour:bool)->vec3f {
  let x=fmWrap(coord.x,extent.x,wrap.x);
  let y=fmWrap(coord.y,extent.y,wrap.y);
  let rgb=unpack4x8unorm(fmData[start+u32(y*extent.x+x)]).rgb;
  if (!colour) { return rgb; }
  // Decode each texel before filtering the linear-light colour. Normal maps are linear data.
  return select(rgb/12.92,pow((rgb+0.055)/1.055,vec3f(2.4)),rgb>vec3f(0.04045));
}
fn fmRead(asset:u32, info:u32, uv:vec2f, colour:bool)->vec3f {
  let image=fmData[info];
  if(image==0xffffffffu){return select(vec3f(0.5,0.5,1.0),vec3f(1),colour);}
  let at=asset+fmData[asset+3u]+3u*image;
  let start=asset+fmData[at];
  let extent=vec2i(i32(fmData[at+1u]),i32(fmData[at+2u]));
  let pos=uv*vec2f(extent)-0.5;
  let i=vec2i(floor(pos));let f=fract(pos);
  let wrap=vec2u(fmData[info+1u],fmData[info+2u]);
  return mix(mix(fmTexel(start,extent,i,wrap,colour),fmTexel(start,extent,i+vec2i(1,0),wrap,colour),f.x),
             mix(fmTexel(start,extent,i+vec2i(0,1),wrap,colour),fmTexel(start,extent,i+vec2i(1),wrap,colour),f.x),f.y);
}

// Closest point and its barycentric coordinates, including the three edge regions.
fn fmBary(p:vec3f,a:vec3f,b:vec3f,c:vec3f)->vec3f {
  let ab=b-a;let ac=c-a;let ap=p-a;
  let d1=dot(ab,ap);let d2=dot(ac,ap);
  if(d1<=0 && d2<=0){return vec3f(1,0,0);}
  let bp=p-b;let d3=dot(ab,bp);let d4=dot(ac,bp);
  if(d3>=0 && d4<=d3){return vec3f(0,1,0);}
  let vc=d1*d4-d3*d2;
  if(vc<=0 && d1>=0 && d3<=0){let v=d1/max(d1-d3,1e-18);return vec3f(1-v,v,0);}
  let cp=p-c;let d5=dot(ab,cp);let d6=dot(ac,cp);
  if(d6>=0 && d5<=d6){return vec3f(0,0,1);}
  let vb=d5*d2-d1*d6;
  if(vb<=0 && d2>=0 && d6<=0){let w=d2/max(d2-d6,1e-18);return vec3f(1-w,0,w);}
  let va=d3*d6-d5*d4;
  if(va<=0 && d4>=d3 && d5>=d6){let w=(d4-d3)/max(d4-d3+d5-d6,1e-18);return vec3f(0,1-w,w);}
  let den=max(va+vb+vc,1e-25);let v=vb/den;let w=vc/den;return vec3f(1-v-w,v,w);
}

struct FmHit {tri:u32, bary:vec3f, distance:f32}
fn fmClosest(asset:u32, q:vec3f, reach:f32)->FmHit {
  var result=FmHit(0xffffffffu,vec3f(0),reach*reach);
  let nodes=asset+fmData[asset];let tris=asset+fmData[asset+1u];
  var stack:array<u32,48>;var depth=1u;stack[0]=0u;
  while(depth>0u){
    depth--;let node=nodes+8u*stack[depth];
    let dx=max(max(fmV3(node)-q,q-fmV3(node+4u)),vec3f(0));
    if(dot(dx,dx)>result.distance){continue;}
    let first=fmData[node+3u];let count=fmData[node+7u];
    if(count==0u){
      stack[depth]=first;stack[depth+1u]=first+1u;depth+=2u;continue;
    }
    for(var i=0u;i<count;i++){
      let tri=tris+fmData[asset+4u]*(first+i);
      let a=fmV3(tri);let b=fmV3(tri+3u);let c=fmV3(tri+6u);
      let bary=fmBary(q,a,b,c);
      let diff=q-(a*bary.x+b*bary.y+c*bary.z);
      let ds=dot(diff,diff);
      if(ds<result.distance){result=FmHit(tri,bary,ds);}
    }
  }
  return result;
}
fn fmUV(square:u32, bary:vec3f)->vec2f {
  return bary.x*fmV2(square)+bary.y*fmV2(square+2u)+bary.z*fmV2(square+4u);
}

fn fileSurface(p:vec3f, rd:vec3f, incidentIor:f32)->vec4f {
  let scene=vec4f(toObj(p),1);
  var best=1e20;var winner=0u;var asset=0u;var tri=0xffffffffu;var bary=vec3f(0);
  for(var i=0u;i<fmInstances[0];i++){
    let index=1u+i*15u;let src=fmInstances[index];
    let q=vec3f(dot(fmRow(index+1u),scene),dot(fmRow(index+5u),scene),dot(fmRow(index+9u),scene));
    let scale=bitcast<f32>(fmInstances[index+13u]);
    let hit=fmClosest(src,q,min(bitcast<f32>(fmInstances[index+14u]),sqrt(best)/scale));
    let measure=hit.distance*scale*scale;
    if(hit.tri!=0xffffffffu && measure<best){best=measure;asset=src;winner=index;tri=hit.tri;bary=hit.bary;}
  }
  if(tri==0xffffffffu){return vec4f(0);}
  let material=fmData[tri+21u];
  if(material==0xffffffffu){return vec4f(0);}
  let m=asset+fmData[asset+2u]+material*17u;
  let base=fmV3(m)*fmRead(asset,m+7u,fmUV(tri+9u,bary),true);
  var finish=vec2f(fmFloat(m+4u),fmFloat(m+5u));
  if(fmData[m+14u]!=0xffffffffu){
    // glTF stores perceptual roughness in green and metallic coverage in blue, both linear.
    finish *= fmRead(asset,m+14u,fmUV(tri+34u,bary),false).gb;
  }
  let rough=clamp(finish.x,0.0,1.0);
  let metal=clamp(finish.y,0.0,1.0);
  var n=nrm(p);
  if(fmData[m+10u]!=0xffffffffu){
    let uv0=fmV2(tri+15u);let du=fmV2(tri+17u)-uv0;let dv=fmV2(tri+19u)-uv0;
    let det=du.x*dv.y-du.y*dv.x;
    if(abs(det)>1e-9){
      let e1=fmV3(tri+3u)-fmV3(tri);let e2=fmV3(tri+6u)-fmV3(tri);
      let tanL=(e1*dv.y-e2*du.y)/det;let bitL=(e2*du.x-e1*dv.x)/det;
      let dx=fmRow(winner+1u).xyz;let dy=fmRow(winner+5u).xyz;let dz=fmRow(winner+9u).xyz;
      let toScene=mat3x3f(dx,dy,dz);
      let toCamera=transpose(mat3x3f(u.obj0.xyz,u.obj1.xyz,u.obj2.xyz));
      let authoredTan=bary.x*fmV3(tri+22u)+bary.y*fmV3(tri+26u)+bary.z*fmV3(tri+30u);
      let handed=bary.x*fmFloat(tri+25u)+bary.y*fmFloat(tri+29u)+bary.z*fmFloat(tri+33u);
      let authored=abs(handed)>0.01 && dot(authoredTan,authoredTan)>1e-10;
      let t0=toCamera*toScene*select(tanL,authoredTan,authored);let b0=toCamera*toScene*bitL;
      let tang=t0-dot(t0,n)*n;
      if(dot(tang,tang)>1e-12){
        let t=normalize(tang);
        let b=select(sign(dot(cross(n,t),b0)),sign(handed),authored)*cross(n,t);
        let uv=fmUV(tri+15u,bary);
        var detail=2.0*fmRead(asset,m+10u,uv,false)-1.0;
        if(fmData[m+13u]!=0u) {
          let at=asset+fmData[asset+3u]+3u*fmData[m+10u];
          let step=1.0/vec2f(f32(fmData[at+1u]),f32(fmData[at+2u]));
          let sx=fmRead(asset,m+10u,uv+vec2f(step.x,0),false).r-fmRead(asset,m+10u,uv-vec2f(step.x,0),false).r;
          let sy=fmRead(asset,m+10u,uv+vec2f(0,step.y),false).r-fmRead(asset,m+10u,uv-vec2f(0,step.y),false).r;
          detail=vec3f(-0.5*sx,-0.5*sy,1.0);
        }
        n=normalize(n*detail.z+fmFloat(m+6u)*(t*detail.x+b*detail.y));
      }
    }
  }
  // Geometry, image decoding, UV sets, normal handedness and height detail stay in this adapter.
  // The imported lobe is isotropic, so any stable orthonormal tangent completes its canonical frame.
  let seed=select(vec3f(1.0,0.0,0.0),vec3f(0.0,1.0,0.0),abs(n.y)<0.9);
  let tangent=normalize(cross(seed,n));
  let hit=PbrSurfaceHit(PbrFrame(tangent,cross(n,tangent),n),
    PbrMedium(incidentIor,0.0,0u,0),PbrMedium(1.5,0.0,0u,0),550.0);
  let imported=PbrImportedOpaque(base,metal,rough);
  let diffuse=nine(0,toEnv(n)){{FILE_DIFFUSE_LAMPS}};
  let r=reflect(rd,n);
  let reflected=mix(env(r),nine(9,toEnv(r)),rough*rough); // imported specular
  // The pure closure correctly rejects this frame. Preserve the accepted opaque hit's established
  // clamped-grazing response instead: both Fresnel branches are one and the diffuse partition is zero.
  if (dot(-rd,n)<=0.0) { return vec4f(reflected,1.0); }
  return vec4f(pbr_imported_opaque_response(hit,imported,-rd,diffuse,reflected),1.0);
}
