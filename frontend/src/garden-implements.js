import * as T from 'three';
import { V, ease, material, mesh, random } from './garden-geometry';
import { surface } from './garden-soil';

const up = V(0, 1, 0);

function rod(parent, from, to, lower, upper, paint) {
  const axis = to.clone().sub(from);
  const part = mesh(parent, new T.CylinderGeometry(upper, lower, axis.length(), 24), paint);
  part.position.copy(from).lerp(to, .5);
  part.quaternion.setFromUnitVectors(up, axis.normalize());
  return part;
}

function bentTube(parent, path, radius, paint) {
  return mesh(parent, new T.TubeGeometry(new T.CatmullRomCurve3(path), 36, radius, 10, false), paint);
}

function turned(parent, profile, paint, depth = 1) {
  const shape = new T.LatheGeometry(profile.map(([r, h]) => new T.Vector2(r, h)), 48);
  const part = mesh(parent, shape, paint);
  part.scale.z = depth;
  return part;
}

function ashWood() {
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#ad8050';
  ctx.fillRect(0, 0, 128, 256);
  const rnd = random(42);
  for (let i = 0; i < 230; i++) {
    const x = rnd() * 128, y = rnd() * 256 - 35;
    ctx.strokeStyle = rnd() < .6 ? 'rgba(69,43,26,.21)' : 'rgba(245,202,139,.25)';
    ctx.lineWidth = .4 + rnd();
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.bezierCurveTo(x + 2, y + 18, x - 2, y + 46, x + .5, y + 61);
    ctx.stroke();
  }
  const map = new T.CanvasTexture(canvas);
  map.colorSpace = T.SRGBColorSpace;
  return material(0xffffff, { map, roughness: .73 });
}

function toolFinish() {
  const canvas=document.createElement('canvas');
  canvas.width=canvas.height=128;
  const ctx=canvas.getContext('2d'), pixels=ctx.createImageData(128,128), rnd=random(718);
  for(let i=0;i<pixels.data.length;i+=4) {
    const grain=212+Math.round(39*rnd());
    pixels.data.set([grain,grain,grain,255],i);
  }
  ctx.putImageData(pixels,0,0);
  const map=new T.CanvasTexture(canvas);
  map.colorSpace=T.SRGBColorSpace;
  return map;
}

function wateringCan(parent) {
  const enamel = material(0x59887e, { map:toolFinish(), metalness: .35, roughness: .48, side: T.DoubleSide });
  const brass = material(0xb2955e, { metalness: .55, roughness: .44 });
  const inside = material(0x162e2a, { roughness: .78 });
  const body = turned(parent, [[.01,.04],[.57,.04],[.62,.09],[.58,.94],
    [.54,1.01],[.34,1.075],[.32,1.055],[.33,.97]], enamel, .77);
  turned(parent, [[.335,.977],[.01,.966]], inside, .77);
  for (const [radius, y] of [[.612,.105],[.60,.25],[.592,.85],[.331,1.06]]) {
    turned(parent, [[radius,y-.015],[radius+.011,y],[radius,y+.014]], enamel, .77);
  }
  bentTube(parent, [V(-.5,.30,0),V(-.87,.47,0),V(-.92,.87,0),
    V(-.76,1.1,0),V(-.48,.95,0)], .057, enamel);
  bentTube(parent, [V(-.49,.97,.02),V(-.37,1.44,.02),V(-.12,1.52,.02),
    V(.29,1.49,.02),V(.48,1.01,.02)], .043, enamel);
  rod(parent, V(-.21,1.515,.02), V(.17,1.515,.02), .067,.067,ashWood());
  const start = V(.42,.28,0), neck = V(1.19,.99,0), face = V(1.39,1.12,0);
  rod(parent,start,neck,.18,.075,enamel);
  rod(parent,neck,face,.072,.19,brass);
  const direction = face.clone().sub(neck).normalize();
  const rose = new T.Group();
  parent.add(rose);
  rose.position.copy(face);
  rose.quaternion.setFromUnitVectors(up,direction);
  // A shallow punched brass rose gives the spray a recognizable source.
  const dots = new T.InstancedMesh(new T.CircleGeometry(.014,7), inside, 19);
  const pose = new T.Object3D();
  pose.rotation.x = -Math.PI/2;
  for (let i=0; i<19; i++) {
    const angle = i*2.39996, r=.158*Math.sqrt(i/19);
    pose.position.set(r*Math.cos(angle),.002,r*Math.sin(angle));
    pose.updateMatrix(); dots.setMatrixAt(i,pose.matrix);
  }
  rose.add(dots);
  return { anchor: body, point: V(0,.55,.38), outlet: rose };
}

function gardenSpade(parent) {
  const steel = material(0x8b9894, { map:toolFinish(), metalness: .45, roughness: .51, side:T.DoubleSide });
  const collar = material(0x45524c, { metalness:.45, roughness:.57 });
  const wood = ashWood();
  // Press a hollow along the blade and turn its shoulders forward.
  const vertices=[], uvs=[], indices=[];
  for(let row=0;row<=18;row++) {
    const t=row/18, width=.43*Math.pow(Math.sin(Math.min(1,t*1.8)*Math.PI/2),.72)+.01;
    for(let col=0;col<=18;col++) {
      const u=col/9-1, edge=u*u;
      vertices.push(u*width,1.09*t-.065*edge*t**7,
        .18*edge*Math.sin(t*Math.PI*.8)-.09*(1-edge)*t+.12*(1-t)**3);
      uvs.push(col/18,t);
      if(row && col) { const k=row*19+col; indices.push(k-20,k-19,k,k-20,k,k-1); }
    }
  }
  const blade=new T.BufferGeometry();
  blade.setAttribute('position',new T.Float32BufferAttribute(vertices,3));
  blade.setAttribute('uv',new T.Float32BufferAttribute(uvs,2));
  blade.setIndex(indices);blade.computeVertexNormals();
  const head=mesh(parent,blade,steel);
  for(const side of [-1,1]) {
    bentTube(parent,[V(side*.10,1.075,-.07),V(side*.27,1.065,-.025),V(side*.435,1.025,.08)],.027,steel);
  }
  rod(parent,V(0,.54,-.054),V(0,1.4,-.095),.029,.088,collar);
  rod(parent,V(0,1.38,-.095),V(0,3.05,-.095),.078,.069,wood);
  rod(parent,V(0,1.34,-.095),V(0,1.42,-.095),.092,.092,steel);
  for(const side of [-1,1]) {
    bentTube(parent,[V(0,2.99,-.095),V(side*.065,3.13,-.095),
      V(side*.23,3.33,-.095),V(side*.25,3.52,-.095)],.04,collar);
  }
  rod(parent,V(-.275,3.51,-.095),V(.275,3.51,-.095),.063,.063,wood);
  return { anchor: head, point: V(0,.83,.04) };
}

function leafRake(parent) {
  const steel = material(0x97a29a, { metalness: .42, roughness: .52 });
  const socket = material(0x555f54, { metalness: .38, roughness: .57 });
  rod(parent, V(0,0,0), V(0,2.64,0), .069,.056,ashWood());
  const ferrule = rod(parent, V(0,2.53,0), V(0,2.89,0), .081,.097,socket);
  const tineCount = 19;
  for (let i=0; i<tineCount; i++) {
    const fan = 2*i/(tineCount-1)-1;
    const endY = 4.06-.26*fan*fan;
    bentTube(parent, [V(fan*.053,2.84,0), V(fan*.39,3.36,-.05),
      V(fan*.84,endY,-.13), V(fan*.86,endY+.045,.045)], .016,steel);
  }
  bentTube(parent,[V(-.42,3.39,-.045),V(0,3.40,-.045),V(.42,3.39,-.045)],.032,socket);
  return { anchor: ferrule, point: V(0,0,0) };
}

function wheelbarrow(parent) {
  const enamel=material(0x92513b,{map:toolFinish(),metalness:.26,roughness:.62,side:T.DoubleSide});
  const frame=material(0x48534d,{metalness:.45,roughness:.57});
  const wood=ashWood();
  const axleHeight=.39;
  const chassis=new T.Group();
  chassis.position.y=axleHeight;
  parent.add(chassis);
  const tyre=mesh(parent,new T.TorusGeometry(.30,.095,12,40),material(0x2d322f));
  tyre.position.y=axleHeight;
  const hub=V(0,axleHeight,0);
  rod(parent,V(0,axleHeight,-.17),V(0,axleHeight,.17),.10,.10,frame);
  for(let i=0;i<10;i++) {
    const angle=i*Math.PI/5;
    rod(parent,hub,V(.27*Math.cos(angle),axleHeight+.27*Math.sin(angle),0),.014,.014,frame);
  }

  // Rounded rectangular cross sections form the pressed tub and its rolled lip.
  const sections=[[.72,.03,.025,.33],[.72,.64,.32,.34],[.68,.79,.43,.49],
    [.64,1.10,.57,.96],[.64,1.12,.59,.99],[.64,1.09,.56,1.015],
    [.64,1.06,.54,.98],[.69,.74,.40,.50],[.72,.61,.29,.39],[.72,.01,.01,.39]];
  const positions=[],uv=[],indices=[],segments=48;
  sections.forEach(([center,length,width,level],ring)=>{
    for(let i=0;i<=segments;i++){
      const angle=i*2*Math.PI/segments,c=Math.cos(angle),s=Math.sin(angle);
      const x=Math.sign(c)*Math.abs(c)**.67;
      positions.push(center+length*x,level-.10*x,width*Math.sign(s)*Math.abs(s)**.67);
      uv.push(i/segments,ring/(sections.length-1));
      if(ring && i){
        const n=ring*(segments+1)+i;
        indices.push(n-segments-2,n-1,n,n-segments-2,n,n-segments-1);
      }
    }
  });
  const tub=new T.BufferGeometry();
  tub.setAttribute('position',new T.Float32BufferAttribute(positions,3));
  tub.setAttribute('uv',new T.Float32BufferAttribute(uv,2));
  tub.setIndex(indices);tub.computeVertexNormals();
  mesh(chassis,tub,enamel);
  for(const side of [-1,1]){
    const z=side*.32;
    bentTube(chassis,[V(0,0,z*.55),V(.64,.26,z),V(1.39,.36,z*1.25),V(2.44,.63,z*1.67)],.044,frame);
    rod(chassis,V(.43,.19,z*.91),V(.36,.42,z*1.13),.035,.035,frame);
    rod(chassis,V(2.31,.60,z*1.60),V(2.78,.72,z*1.80),.068,.06,wood);
    bentTube(chassis,[V(.92,.31,z),V(1.12,-.33,z*1.23),V(1.32,-.34,z*1.28),V(1.57,.40,z*1.36)],.037,frame);
  }
  return {anchor:chassis,point:V(.74,.67,.46),chassis};
}

export function gardenImplements(scene) {
  const stage=new T.Group();
  // Arrange the tools in the plane of the final camera, with independent depths.
  stage.rotation.y=.46;
  scene.add(stage);
  const props = [wateringCan, gardenSpade, leafRake, wheelbarrow].map(build => {
    const base=new T.Group(), model=new T.Group();
    stage.add(base);base.add(model);
    const details=build(model);
    const paints=new Set(), hits=[];
    model.traverse(part=>{
      if(part.isMesh) { hits.push(part); paints.add(part.material); }
    });
    const prop={ ...details, base, model, paints, hits, age:Infinity, hover:false };
    hits.forEach(part=>{part.userData.implement=prop;});
    return prop;
  });
  const [can,spade,rake,barrow]=props;
  can.base.rotation.y=.19;
  spade.base.rotation.y=-.42;
  rake.base.rotation.y=.28;
  barrow.base.rotation.y=-.32;
  const droplets=new T.InstancedMesh(new T.SphereGeometry(.024,6,5),material(0x9bcad0,
    {roughness:.2,metalness:.15}),32);
  stage.add(droplets);
  droplets.visible=false;droplets.frustumCulled=false;
  const pose=new T.Object3D(), sprayOrigin=V(), landing=V();
  function layout(aspect) {
    const wide=ease(.62,1.4,aspect);
    const extraWidth=Math.max(0,aspect-1.5);
    can.base.position.set(-2.91-1.53*wide-2.15*extraWidth,0,.65-.9*wide);
    spade.base.position.set(3.06+1.71*wide+1.58*extraWidth,0,-2.6);
    rake.base.position.set(-3.28-2.91*wide-5.20*extraWidth,0,-3.6);
    barrow.base.visible=aspect>=1.74;
    barrow.base.scale.setScalar(.78+.22*ease(1.74,2.12,aspect));
    barrow.base.position.set(3.92*aspect-2.59,0,1.05);
    can.base.scale.setScalar(1.09+.37*wide);
    spade.base.scale.setScalar(1.13+.30*wide);
    rake.base.scale.setScalar(.94+.21*wide);
    for(const tool of props) {
      const world=stage.localToWorld(tool.base.position.clone());
      tool.base.position.y=surface(world.x,world.z)-(tool===spade?.22:0);
    }
  }
  function update(dt,reduced) {
    for(const prop of props) {
      prop.age+=dt;
      const glow=(prop.hover ? .09 : 0)+(prop.age<1.9 ? .12 : 0);
      prop.paints.forEach(p=>{p.emissive.copy(p.color).multiplyScalar(glow);});
    }
    const lift=reduced?0:ease(0,.64,can.age)*(1-ease(2.98,3.64,can.age));
    const pour=reduced?0:ease(.82,1.22,can.age)*(1-ease(2.62,3.02,can.age));
    can.model.rotation.z=-.66*pour;
    can.model.position.y=1.48*lift;
    const dig=reduced?0:ease(0,.32,spade.age)*(1-ease(.53,1.18,spade.age));
    spade.model.rotation.z=-.13+.20*dig;
    spade.model.rotation.x=-.12-.19*dig;
    spade.model.position.y=-.20*dig;
    const rock=reduced || rake.age>=1.45 ? 0 :
      Math.sin(rake.age*8)*(1-ease(0,1.45,rake.age))*.16;
    rake.model.rotation.set(-.22,.09,.105+rock);
    const tip=reduced?0:ease(0,.55,barrow.age)*(1-ease(.72,1.42,barrow.age));
    barrow.chassis.rotation.z=.23*tip;
    const firstDrop=1.22, dropInterval=.023, fallTime=.67;
    droplets.visible=!reduced && can.age>=firstDrop && can.age<firstDrop+(droplets.count-1)*dropInterval+fallTime;
    if(droplets.visible) {
      can.outlet.getWorldPosition(sprayOrigin);
      stage.worldToLocal(sprayOrigin);
      landing.copy(sprayOrigin);landing.x+=.67;
      const world=stage.localToWorld(landing.clone());
      landing.y=surface(world.x,world.z);
      for(let i=0;i<droplets.count;i++) {
        const birth=firstDrop+i*dropInterval, t=(can.age-birth)/fallTime;
        const fan=Math.sin(i*2.4), s=t>=0 && t<=1?1:0;
        pose.position.copy(sprayOrigin).lerp(landing,t);
        pose.position.x+=.28*fan*t;
        pose.position.z+=.18*Math.cos(i*2.4)*t;
        pose.position.y=T.MathUtils.lerp(sprayOrigin.y,landing.y,t*t);
        pose.scale.set(s,1.65*s,s);pose.updateMatrix();droplets.setMatrixAt(i,pose.matrix);
      }
      droplets.instanceMatrix.needsUpdate=true;
    }
  }
  function activate(prop) {
    if(prop!==can || prop.age>=3.64)prop.age=0;
  }
  return { props, layout, update, activate };
}
