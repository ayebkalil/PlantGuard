import * as T from 'three';

export const V = (x,y,z) => new T.Vector3(x,y,z);
export const clamp = T.MathUtils.clamp;
export function random(seed) {
  return () => { seed = (1664525 * seed + 1013904223) >>> 0; return seed / 4294967296; };
}
export function ease(a,b,x) {return T.MathUtils.smoothstep(x,a,b);}
export function material(color, props={}) {return new T.MeshStandardMaterial({color,roughness:.83,...props});}
export function mesh(parent,geometry,mat) {
  const obj=new T.Mesh(geometry,mat);
  obj.castShadow=true; obj.receiveShadow=true; parent.add(obj);
  return obj;
}

export function grainTexture(leaf=false) {
  const rnd=random(leaf?83:107), size=512;
  const canvas=document.createElement('canvas');canvas.width=canvas.height=size;
  const ctx=canvas.getContext('2d'), pixels=ctx.createImageData(size,size);
  for(let y=0;y<size;y++) for(let x=0;x<size;x++){
    const i=(y*size+x)*4;
    const broad=Math.sin(x*.041+1.7*Math.sin(y*.043))*Math.sin(y*.037)+.6*Math.cos(x*.13+y*.1);
    const n=rnd()*(leaf?18:70)+broad*(leaf?9:18);
    const b=leaf? [63,105,37] : [64,55,44];
    for(let c=0;c<3;c++)pixels.data[i+c]=b[c]+n;
    pixels.data[i+3]=255;
  }
  ctx.putImageData(pixels,0,0);
  if(leaf){
    // Coordinates run from the petiole at the left to the blade tip on the right.
    function stroke(x,y,cx,cy,ex,ey,width,alpha){
      ctx.beginPath();ctx.moveTo(x,y);ctx.quadraticCurveTo(cx,cy,ex,ey);
      ctx.strokeStyle='rgba(206,224,135,'+alpha+')';ctx.lineWidth=width;ctx.stroke();
    }
    for(const side of [-1,1])for(let i=0;i<16;i++){
      const x=i*32-21;
      stroke(x,256,x+28,256+side*64,x+75,256+side*259,2.7,.55);
      for(let k=1;k<9;k++){
        const t=k/10; const sx=x+65*t, sy=256+side*234*t*t;
        stroke(sx,sy,sx+18,sy+side*11,sx+33,sy+side*(23+26*t),1.5,.35);
      }
    }
    stroke(0,256,240,254,512,256,5,.82);
  }
  const tex=new T.CanvasTexture(canvas);
  tex.colorSpace=T.SRGBColorSpace;
  tex.wrapS=tex.wrapT=T.RepeatWrapping;
  if(!leaf)tex.repeat.set(22,22);
  return tex;
}

export function vine(parent,center,radius,mat,rings=85,sides=12) {
  const vertices=[],uv=[],index=[],centers=[];
  for(let i=0;i<=rings;i++){
    const t=i/rings, p=center(t);centers.push(p);
    const tangent=center(Math.min(1,t+.003)).sub(center(Math.max(0,t-.003))).normalize();
    const cross=V(0,0,1).cross(tangent).normalize(), other=tangent.clone().cross(cross);
    for(let j=0;j<=sides;j++){
      const a=j/sides*2*Math.PI;
      const r=radius(t)*(1+.055*Math.cos(a*5+t*20));
      const point=p.clone().addScaledVector(cross,Math.cos(a)*r).addScaledVector(other,Math.sin(a)*r);
      vertices.push(point.x,point.y,point.z);uv.push(j/sides,t);
      if(i&&j){const n=i*(sides+1)+j;index.push(n-sides-2,n-sides-1,n,n-sides-2,n,n-1);}
    }
  }
  const geom=new T.BufferGeometry();
  geom.setAttribute('position',new T.Float32BufferAttribute(vertices,3));geom.setAttribute('uv',new T.Float32BufferAttribute(uv,2));geom.setIndex(index);geom.computeVertexNormals();
  // The two UV seam vertices share a surface normal around the stem.
  const normals=geom.attributes.normal;
  for(let ring=0;ring<=rings;ring++){
    const first=ring*(sides+1),last=first+sides;
    const normal=new T.Vector3().fromBufferAttribute(normals,first)
      .add(new T.Vector3().fromBufferAttribute(normals,last)).normalize();
    normals.setXYZ(first,normal.x,normal.y,normal.z);
    normals.setXYZ(last,normal.x,normal.y,normal.z);
  }
  const obj=mesh(parent,geom,mat);
  let previous=-1;
  obj.userData.reveal=t=>{
    if(t===previous)return;
    const pos=geom.attributes.position;
    const first=Math.max(0,Math.floor((Math.min(previous,t)-.05)*rings));
    const last=Math.min(rings,Math.ceil(t*rings));
    const tipIndex=Math.min(rings-1,Math.floor(t*rings));
    const tip=centers[tipIndex].clone().lerp(centers[tipIndex+1],t*rings-tipIndex);
    // Collapse just the growing end around its centreline so it advances continuously.
    for(let i=first;i<=last;i++){
      const radius=ease(0,.045,t-i/rings),base=i/rings>=t?tip:centers[i];
      for(let j=0;j<=sides;j++){
        const n=i*(sides+1)+j;
        pos.setXYZ(n,base.x+(vertices[3*n]-centers[i].x)*radius,
          base.y+(vertices[3*n+1]-centers[i].y)*radius,
          base.z+(vertices[3*n+2]-centers[i].z)*radius);
      }
    }
    pos.needsUpdate=true;geom.setDrawRange(0,last*sides*6);previous=t;
  };
  return obj;
}

export function bladeGeometry(length,width,bend) {
  const rows=38, cols=20,vertices=[],folded=[],uv=[],idx=[];
  for(let i=0;i<=rows;i++){
    const t=i/rows;
    const w=width*Math.pow(Math.sin(Math.PI*Math.pow(t,.77)),.83)*(1-.18*t);
    for(let j=0;j<=cols;j++){
      const v=2*j/cols-1, edge=Math.abs(v);
      const x=length*t;
      const spine=.026*w*Math.exp(-17*edge);
      const y=bend*Math.sin(Math.PI*t)-.17*length*t*t+spine
        -.13*w*edge*edge+.028*w*edge*Math.sin(t*12+v);
      const z=w*v*(1+.055*Math.sin(12*t+v));
      vertices.push(x,y,z);folded.push(x*.86,y+.55*w*edge+.23*t,z*.12);uv.push(t,j/cols);
      if(i&&j){const n=i*(cols+1)+j;idx.push(n-cols-2,n-cols-1,n,n-cols-2,n,n-1);}
    }
  }
  const geo=new T.BufferGeometry();geo.setAttribute('position',new T.Float32BufferAttribute(vertices,3));
  geo.setAttribute('uv',new T.Float32BufferAttribute(uv,2));geo.setIndex(idx);
  geo.computeVertexNormals();
  const bud=new T.BufferGeometry();
  bud.setIndex(geo.index);bud.setAttribute('position',new T.Float32BufferAttribute(folded,3));
  bud.computeVertexNormals();
  geo.morphAttributes.position=[bud.attributes.position];geo.morphAttributes.normal=[bud.attributes.normal];
  return geo;
}


