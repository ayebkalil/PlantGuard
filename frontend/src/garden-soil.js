import * as T from 'three';
import { V, ease, mesh, material, vine, clamp } from './garden-geometry';

export function surface(x,z) {
  return -.10 + .05*Math.sin(1.9*x+.7*z) + .044*Math.cos(2.1*z-.6*x)
    + .022*Math.sin(7*x+3*Math.sin(2*z)) + .018*Math.cos(9*z+3*x);
}

// Keep the opening in the turf irregular, with a soft transition into grass.
function turfCover(x,z) {
  const edge=1.03+.22*Math.sin(3*x+z)+.13*Math.sin(4*z-x);
  return ease(edge-.18,edge+.38,Math.hypot(x,z));
}

function grassTufts(scene,rnd,cut) {
  const positions=[],colors=[],indices=[];
  const root=new T.Color(0x29491c),tip=new T.Color(),color=new T.Color();
  for(let tuft=0;tuft<2600;tuft++){
    const angle=rnd()*Math.PI*2;
    const radius=.76+11*Math.pow(rnd(),.72);
    const cx=radius*Math.cos(angle),cz=radius*Math.sin(angle);
    const cover=turfCover(cx,cz);
    if(rnd()>cover)continue;
    const height=(.08+.14*rnd())*(.45+.55*cover);
    const hue=rnd();
    tip.set(hue>.86?0xa3ad58:hue>.42?0x799340:0x64873c);
    for(let b=0;b<7;b++){
      const a=rnd()*Math.PI*2, dx=Math.cos(a),dz=Math.sin(a);
      const x=cx+.12*(rnd()-.5),z=cz+.12*(rnd()-.5),y=surface(x,z)-.022;
      const h=height*(.65+.65*rnd()),w=.009+.014*rnd(),lean=h*(.22+.66*rnd());
      const base=positions.length/3;
      for(const t of [0,.52,1]){
        const halfWidth=w*(1-t),bow=lean*t*t;
        color.copy(root).lerp(tip,t);
        for(const side of [-1,1]){
          positions.push(x+dx*bow+side*dz*halfWidth,y+t*h,z+dz*bow-side*dx*halfWidth);
          colors.push(color.r,color.g,color.b);
        }
      }
      indices.push(base,base+1,base+2,base+1,base+3,base+2,base+2,base+3,base+4);
    }
  }
  const geometry=new T.BufferGeometry();
  geometry.setAttribute('position',new T.Float32BufferAttribute(positions,3));
  geometry.setAttribute('color',new T.Float32BufferAttribute(colors,3));
  geometry.setIndex(indices);geometry.computeVertexNormals();
  const grass=mesh(scene,geometry,new T.MeshLambertMaterial({
    vertexColors:true,side:T.DoubleSide,clippingPlanes:[cut]
  }));
  grass.castShadow=false;
}

export function soilBed(scene,rnd) {
  const texture=new T.TextureLoader().load('/botany/loam.webp');
  texture.wrapS=texture.wrapT=T.RepeatWrapping;
  texture.colorSpace=T.SRGBColorSpace; texture.anisotropy=4;
  const cut=new T.Plane(V(0,0,-1),-.28);
  const earth=material(0xb0a298,{map:texture,bumpMap:texture,bumpScale:.052,
    roughness:.98,clippingPlanes:[cut],clipShadows:true,side:T.DoubleSide});
  const turf=new T.TextureLoader().load('/botany/meadow.webp');
  turf.wrapS=turf.wrapT=T.RepeatWrapping;turf.colorSpace=T.SRGBColorSpace;turf.anisotropy=4;
  earth.userData.turf=turf;
  earth.onBeforeCompile=shader=>{
    shader.uniforms.turfMap={value:turf};
    shader.vertexShader='attribute float cover; varying float turfAmount;\n'+shader.vertexShader
      .replace('#include <begin_vertex>','#include <begin_vertex>\n turfAmount=cover;');
    shader.fragmentShader='uniform sampler2D turfMap; varying float turfAmount;\n'+shader.fragmentShader
      .replace('#include <map_fragment>',`vec4 soilSample=texture2D(map,vMapUv);
        vec4 meadowSample=texture2D(turfMap,vMapUv*1.72);
        diffuseColor*=mix(soilSample,meadowSample,turfAmount);`);
  };
  // Each wedge meets its neighbour exactly until the shoot displaces the soil.
  const wedges=[], sectors=12, steps=9, rings=44;
  const boundaries=Array.from({length:sectors},(_,i)=>(i+rnd()*.32)*Math.PI*2/sectors);
  function boundary(i,r) {
    return boundaries[i%sectors]+(i===sectors?Math.PI*2:0)+.065*Math.sin(5*r+i%sectors)*Math.exp(-r*.48);
  }
  for(let s=0;s<sectors;s++){
    const positions=[],uv=[],indices=[],weights=[],coverage=[];
    for(let j=0;j<=rings;j++){
      const r=j<28?.025+j*.12:3.385+(j-28)**1.95*.21;
      for(let k=0;k<=steps;k++){
        const a=T.MathUtils.lerp(boundary(s,r),boundary(s+1,r),k/steps);
        const x=r*Math.cos(a),z=r*Math.sin(a);
        positions.push(x,surface(x,z),z);uv.push(x*.36,z*.36);
        coverage.push(turfCover(x,z));
        weights.push(Math.exp(-r*r*.84));
        if(j&&k){const n=j*(steps+1)+k; indices.push(n-steps-2,n-steps-1,n,n-steps-2,n,n-1);}
      }
    }
    for(const k of [0,steps])for(let j=0;j<28;j++){
      const a=j*(steps+1)+k,b=a+steps+1,n=positions.length/3;
      for(const [q,depth] of [[a,0],[b,0],[a,.35],[b,.35]]){
        const [x,y,z]=positions.slice(q*3,q*3+3);
        positions.push(x,y-depth,z);uv.push((x+z)*.36,(y-depth)*.36);weights.push(weights[q]);
        coverage.push(0);
      }
      indices.push(n,n+1,n+2,n+1,n+3,n+2);
    }
    const g=new T.BufferGeometry();
    g.setAttribute('cover',new T.Float32BufferAttribute(coverage,1));
    g.setAttribute('position',new T.Float32BufferAttribute(positions,3));g.setAttribute('uv',new T.Float32BufferAttribute(uv,2));g.setIndex(indices);g.computeVertexNormals();
    mesh(scene,g,earth);
    wedges.push({g,rest:positions,weights,a:boundaries[s]+.24,tilt:.13+rnd()*.1});
  }
  // Exposed face of the soil, with the same surface profile as the top.
  const faceMat=material(0xc5af98,{map:texture,bumpMap:texture,bumpScale:.058,vertexColors:true});
  const faceGeo=new T.PlaneGeometry(36,5,150,32),fp=faceGeo.attributes.position;
  const faceUV=faceGeo.attributes.uv,colors=[];
  for(let i=0;i<fp.count;i++){
    const x=fp.getX(i),depth=2.5-fp.getY(i),z=-.28-.035*Math.sin(7*x+depth*11)*Math.min(depth,1);
    const y=surface(x,-.28)-depth;
    fp.setXYZ(i,x,y,z);faceUV.setXY(i,x*.38,y*.38);
    const shade=.82-.35*ease(.08,2.8,depth)+.08*Math.sin(9*depth+Math.sin(x));
    colors.push(shade,shade*.91,shade*.81);
  }
  faceGeo.setAttribute('color',new T.Float32BufferAttribute(colors,3));faceGeo.computeVertexNormals();
  const wall=mesh(scene,faceGeo,faceMat);

  const grit=new T.IcosahedronGeometry(1,0);
  const pebbles=new T.InstancedMesh(grit,material(0x8f8375,{clippingPlanes:[cut]}),420);
  const pose=new T.Object3D(),tint=new T.Color();
  for(let i=0;i<pebbles.count;i++){
    const a=rnd()*Math.PI*2,r=.8+8*rnd(),x=r*Math.cos(a),z=r*Math.sin(a),size=.008+.06*rnd()**3;
    pose.position.set(x,surface(x,z)+size*.2,z);pose.rotation.set(rnd(),rnd()*6,rnd());
    pose.scale.set(size*1.3,size*.65,size);pose.updateMatrix();pebbles.setMatrixAt(i,pose.matrix);
    pebbles.setColorAt(i,tint.setHSL(.06+.04*rnd(),.15,.12+.16*rnd()));
  }
  pebbles.receiveShadow=true;scene.add(pebbles);
  grassTufts(scene,rnd,cut);
  const debris=new T.InstancedMesh(grit,material(0x66513e),55);scene.add(debris);
  const grains=Array.from({length:debris.count},()=>({a:rnd()*6.28,r:.25+rnd()*1.2,height:.08+rnd()*.45,delay:rnd()*.055,size:.012+rnd()**2*.05}));
  let last=-1;
  const update=p=>{
    cut.constant=T.MathUtils.lerp(-.28,40,ease(.39,.60,p));
    wall.position.z=cut.constant+.28;wall.visible=p<.59;
    const swell=ease(.29,.43,p),settle=ease(.41,.58,p);
    if(p===last || (p>.62&&last>.62))return;
    last=p;
    for(const {g,rest,weights,a,tilt} of wedges){
      const pos=g.attributes.position;
      for(let i=0;i<pos.count;i++){
        const w=weights[i]*swell;
        pos.setXYZ(i,rest[i*3]+Math.cos(a)*.20*w,rest[i*3+1]+tilt*w*(1-.53*settle),rest[i*3+2]+Math.sin(a)*.20*w);
      }
      pos.needsUpdate=true;
    }
    debris.visible=swell>0;
    grains.forEach(({a,r,height,delay,size},i)=>{
      const t=clamp((p-.33-delay)/.16,0,1),d=.22+r*t,x=d*Math.cos(a),z=d*Math.sin(a);
      pose.position.set(x,surface(x,z)+.04+4*height*t*(1-t),z);pose.rotation.set(i+t*4,t*6,0);pose.scale.setScalar(t?size:0);
      pose.updateMatrix();debris.setMatrixAt(i,pose.matrix);
    });debris.instanceMatrix.needsUpdate=true;
  };
  return {update, earth};
}

export function germination(scene) {
  const rootMat=material(0xc3a075,{roughness:.84});
  const shootMat=material(0x91a95b,{roughness:.65});
  const seedY=-.89;
  const rootPath=t=>V(-.12+.08*Math.sin(6*t)-.17*t,seedY-.10-.86*t,.12+.05*Math.sin(5*t));
  const root=vine(scene,rootPath,t=>.042*(1-t)+.007,rootMat,64,9);
  const laterals=Array.from({length:8},(_,i)=>{
    const level=.18+.089*i,origin=rootPath(level),side=i%2?1:-1;
    const path=t=>origin.clone().add(V(side*(.16+.09*Math.sin(i))*t,-.14*t-.1*t*t,.05*Math.sin(4*t)));
    return {level,obj:vine(scene,path,t=>.016*(1-t)+.0025,rootMat,22,6)};
  });
  const shoot=vine(scene,t=>V(.14*(1-t)+.105*Math.sin(t*6),seedY+.04+t*.80,.04*Math.sin(4*t)),t=>.08-.03*t,shootMat,68,12);
  return p=>{
    root.userData.reveal(ease(.035,.28,p));
    laterals.forEach(({level,obj})=>obj.userData.reveal(ease(.08+.12*level,.22+.12*level,p)));
    shoot.userData.reveal(ease(.13,.334,p));
  };
}

