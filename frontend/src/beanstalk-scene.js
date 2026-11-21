import * as T from 'three';
import { V, clamp, ease, random, material, mesh, grainTexture, vine, bladeGeometry } from './garden-geometry';

import { soilBed, germination } from './garden-soil';
import { gardenImplements } from './garden-implements';
import { gardenSky } from './garden-sky';

function center(t) {
  // The ascending shoot inclines smoothly above the last mature node.
  const crown=ease(.77,1,t);
  return V(.48*Math.sin(8*t)*Math.sin(Math.PI*t)+.32*t-.37*crown,
    6.5*t-.12, .31*Math.sin(t*10)+.13*crown);
}

function makePlant(scene, featureIds) {
  const group=new T.Group();scene.add(group);
  const stalks=[];
  const bark=new T.TextureLoader().load('/botany/epidermis.webp');
  bark.colorSpace=T.SRGBColorSpace;
  bark.wrapS=bark.wrapT=T.RepeatWrapping;
  bark.repeat.set(2,3); bark.anisotropy=4;
  const stalkPaint=[0xd6dfbd,0xafc79b,0x9faf75].map(color=>material(color,{
    map:bark,bumpMap:bark,bumpScale:.024,roughness:.72,side:T.DoubleSide
  }));
  // A continuous trunk carries the weight. Two smaller vines climb its flanks.
  stalks.push(vine(group,center,t=>.48*(1-t)**.72+.018,stalkPaint[0],150,32));
  for(let i=0;i<2;i++){
    const offset=i*Math.PI+.4;
    stalks.push(vine(group,t=>{
      const r=.39*(1-t)**.72*(1-ease(.69,.96,t)), a=t*11+offset;
      return center(t).add(V(r*Math.cos(a),0,r*Math.sin(a)));
    },t=>.20*(1-t)**.85+.004,stalkPaint[i+1],140,18));
  }
  const texture=new T.TextureLoader().load('/botany/bean-leaf.webp');texture.colorSpace=T.SRGBColorSpace;const leaves=[];
  const layout=[
    [.19,2.95,2.45,.95,-.55],[.27,-.38,2.25,1.02,.62],[.38,3.72,2.60,.95,-.65],
    [.47,.55,2.78,1.04,.88],[.57,2.83,2.35,.94,-.72],[.67,-.55,2.06,.83,.52],
    [.75,3.35,1.73,.72,-.55],[.83,-.18,1.38,.59,.67],[.91,2.84,.91,.39,-.60],
    [.977,.36,.48,.20,.75],[.994,2.92,.34,.13,-.58]
  ];
  layout.forEach(([level,yaw,len,width,roll],i)=>{
    const node=new T.Group();group.add(node);node.position.copy(center(level));node.rotation.y=yaw;
    const youth=ease(.72,1,level);
    const pitch=.07+.99*youth*youth;
    const bend=len*(.095+.019*(i%3));
    const reach=.24*len+.08*(1-youth);
    const rise=.22*len+.035;
    // Curve out of the stem, then meet the blade along its midrib tangent.
    const end=V(reach,rise,0);
    const tangent=V(1,Math.PI*bend/len,0).applyEuler(new T.Euler(roll,0,pitch)).normalize();
    const petiole=new T.CubicBezierCurve3(V(0,-.04*len,0),V(.12*reach,.6*rise,0),
      end.clone().addScaledVector(tangent,-.32*reach),end);
    vine(node,t=>petiole.getPoint(t),t=>(.04*len+.003)*(1-.84*t),stalkPaint[1],32,14);
    const joint=new T.Group();node.add(joint);joint.position.copy(end);
    const mat=material(new T.Color(0xe1e6ce).lerp(new T.Color(0xffedaa),youth),{
      map:texture,bumpMap:texture,bumpScale:.013,roughness:.62,side:T.DoubleSide
    });
    const blade=mesh(joint,bladeGeometry(len,width,bend),mat);
    const arrival=.31+.60*(.5-Math.sin(Math.asin(1-2*level)/3));
    blade.userData.tool=featureIds[i%featureIds.length];leaves.push({node,blade,joint,roll,arrival,pitch,youth});
  });
  return {leaves,update(p,time){
    const h=ease(.31,.91,p);
    const girth=.14+.86*ease(.34,.82,p);
    group.scale.set(girth,1,girth);
    stalks.forEach(s=>s.userData.reveal(h));
    group.rotation.z=.008*Math.sin(time*.62)*ease(.7,1,p);
    leaves.forEach(({node,joint,blade,roll,arrival,pitch,youth},i)=>{
      const g=ease(arrival,Math.min(.996,arrival+.16),p);
      blade.morphTargetInfluences[0]=1-g*(1-.44*youth*youth);
      node.visible=g>.005;node.scale.setScalar(Math.max(.001,g));
      joint.rotation.set(roll+(1-g)*.48,0,pitch+(1-g)*.85+.021*Math.sin(time*.72+i*.36));
    });
  }};
}

export function mountGarden(host,{active=true,skipIntro=false}={}){
  const holder=host.querySelector('.garden-canvas');
  const links=[...host.querySelectorAll('[data-leaf]')];
  const features=new Map(links.map(link=>[link.dataset.leaf,link]));
  const callout=host.querySelector('.leaf-callout');
  const calloutName=callout.querySelector('span');
  let renderer;
  try {renderer=new T.WebGLRenderer({antialias:true,alpha:true,powerPreference:'high-performance'});}
  catch {host.classList.add('poster');return;}
  holder.append(renderer.domElement);
  renderer.localClippingEnabled=true;
  renderer.shadowMap.enabled=true;renderer.shadowMap.type=T.PCFSoftShadowMap;
  renderer.toneMapping=T.ACESFilmicToneMapping;renderer.toneMappingExposure=1.34;
  const scene=new T.Scene();
  scene.background=new T.Color(0x101a16);
  scene.fog=new T.Fog(scene.background,16,38);
  const camera=new T.PerspectiveCamera(35,1,.05,90);
  const fill=new T.HemisphereLight();scene.add(fill);
  const key=new T.DirectionalLight();key.position.set(5,9,3);scene.add(key);
  key.castShadow=true;key.shadow.mapSize.set(1024,1024);Object.assign(key.shadow.camera,{left:-6,right:6,top:9,bottom:-6});key.shadow.normalBias=.045;
  const rim=new T.DirectionalLight(0xd2edbc,2.9);rim.position.set(3,7,-5);scene.add(rim);
  const rnd=random(1918),ground=soilBed(scene,rnd),roots=germination(scene),plant=makePlant(scene,[...features.keys()]);
  let light=Number(host.dataset.sky==='day');
  const sky=gardenSky(scene,renderer,ground.earth,light);
  const colors=[['#a2b9db','#e4f3ff'],['#4a5157','#92805d'],['#c0d8ff','#fff0c6'],
    ['#a2bfd7','#fbf4cf'],['#2e3f4a','#cfe2dd']].map(pair=>pair.map(value=>new T.Color(value)));
  const implementsScene=gardenImplements(scene);
  const seed=new T.Group();scene.add(seed);seed.position.set(0,-.89,0);seed.rotation.z=.83;seed.scale.set(.78,.78,.78);
  const testa=grainTexture();
  testa.repeat.set(.85,1.25);
  const shell=material(0xac785e,{map:testa,bumpMap:testa,bumpScale:.009,roughness:.56,side:T.DoubleSide});
  const halves=[0,1].map(i=>{
    const geometry=new T.SphereGeometry(1,40,32,Math.PI/2+i*Math.PI,Math.PI);
    const positions=geometry.attributes.position;
    for(let j=0;j<positions.count;j++){
      const y=positions.getY(j);
      positions.setX(j,positions.getX(j)+.24*y*y-.08);
    }
    geometry.computeVertexNormals();
    const half=mesh(seed,geometry,shell);
    half.scale.set(.39,.63,.31);
    const flesh=mesh(half,new T.SphereGeometry(1,24,16),material(0xd5b889,{roughness:.8}));
    flesh.scale.set(.16,.96,.91);flesh.position.x=i===0?.03:-.03;
    return half;
  });
  const scar=mesh(halves[0],new T.SphereGeometry(1,20,14),material(0xb48e65));
  scar.position.set(.01,-.1,.98);scar.scale.set(.066,.26,.023);
  const bounce=new T.PointLight(0xffdbb4,1.15,5);
  bounce.position.set(-.7,-.52,1.7);scene.add(bounce);
  const reduced=matchMedia('(prefers-reduced-motion: reduce)');
  const point=new T.Vector2(20,20),ray=new T.Raycaster(),targets=plant.leaves.map(l=>l.blade);
  const controls=[...host.querySelectorAll('[data-implement]')];
  const screen=new T.Vector3();
  let hover=null,touchLeaf=null,pointerType,calloutHalfWidth=0;
  let frame=0,last=performance.now(),p=skipIntro||reduced.matches?1:0,elapsed=0,desired=p,width=0,height=0;
  function resizeScene(){
    if(!active)return;
    const nextWidth=host.clientWidth,nextHeight=host.clientHeight;
    const ratio=Math.min(devicePixelRatio,1.65);
    if(!nextWidth||!nextHeight || (nextWidth===width && nextHeight===height && ratio===renderer.getPixelRatio()))return;
    width=nextWidth;height=nextHeight;
    if(ratio!==renderer.getPixelRatio())renderer.setPixelRatio(ratio);
    renderer.setSize(width,height,false);
    sky.resize();
    camera.aspect=width/height;
    camera.updateProjectionMatrix();
    implementsScene.layout(camera.aspect);
    calloutHalfWidth=callout.offsetWidth/2;
  }
  const resize=new ResizeObserver(resizeScene);
  resize.observe(host);
  function pick(event){
    const rect=holder.getBoundingClientRect();point.set((event.clientX-rect.left)/rect.width*2-1,1-2*(event.clientY-rect.top)/rect.height);
    ray.setFromCamera(point,camera);
    const selectable=targets.filter(o=>o.parent.parent.visible);
    if(p>.59)for(const prop of implementsScene.props)if(prop.base.visible)selectable.push(...prop.hits);
    const hit=ray.intersectObjects(selectable,false)[0]?.object;
    return hit?.userData.implement || hit || null;
  }
  function highlight(next){
    if(hover===next)return;
    if(hover?.paints)hover.hover=false;
    else if(hover)hover.material.emissive.setHex(0);
    hover=next;
    if(next?.paints)next.hover=true;
    else if(next)next.material.emissive.setHex(0x375216);
    holder.style.cursor=next?'pointer':'default';
    const link=features.get(next?.userData?.tool);
    callout.hidden=!link;
    if(link){
      calloutName.textContent=link.textContent;
      callout.href=link.href;
      callout.toggleAttribute('data-tap',next===touchLeaf);
      calloutHalfWidth=callout.offsetWidth/2;
      positionCallout();
    }
  }
  function projectBlade(blade){
    blade.getVertexPosition(Math.floor(blade.geometry.attributes.position.count/2),screen);
    return blade.localToWorld(screen).project(camera);
  }
  function positionCallout(){
    const anchor=projectBlade(hover);
    const x=clamp((anchor.x+1)*width/2,calloutHalfWidth+8,width-calloutHalfWidth-8);
    const y=(1-anchor.y)*height/2;
    callout.style.left=x+'px';
    callout.style.top=y+'px';
    callout.toggleAttribute('data-below',y<64);
  }
  const rememberPointer=e=>{pointerType=e.pointerType;};
  const move=e=>{
    if(e.pointerType==='touch')return;
    touchLeaf=null;
    callout.removeAttribute('data-tap');
    highlight(pick(e));
  };
  const leave=e=>{if(e.pointerType!=='touch')highlight(null);};
  const click=e=>{
    const obj=pick(e);
    const link=features.get(obj?.userData?.tool);
    if(link){
      if((e.pointerType||pointerType)==='touch' && touchLeaf!==obj){
        touchLeaf=obj;
        highlight(obj);
        callout.setAttribute('data-tap','');
      }else link.click();
    }else{
      touchLeaf=null;
      highlight(obj);
      if(obj?.paints)implementsScene.activate(obj);
    }
  };
  const scrub=e=>{if(!reduced.matches)desired=clamp(desired+e.deltaY*.0005,0,1);};
  holder.addEventListener('pointerdown',rememberPointer);
  holder.addEventListener('pointermove',move);holder.addEventListener('pointerleave',leave);holder.addEventListener('click',click);holder.addEventListener('wheel',scrub,{passive:true});
  let touchY;
  const start=e=>{touchY=e.touches[0].clientY;},drag=e=>{const y=e.touches[0].clientY;scrub({deltaY:(touchY-y)*3});touchY=y;};
  holder.addEventListener('touchstart',start,{passive:true});holder.addEventListener('touchmove',drag,{passive:true});
  const controlProp=target=>implementsScene.props[controls.indexOf(target)];
  const useControl=e=>{const prop=controlProp(e.target);if(prop)implementsScene.activate(prop);};
  host.addEventListener('click',useControl);
  const focus=e=>{if(e.target!==callout)highlight(controlProp(e.target) || targets.find(t=>t.userData.tool===e.target.dataset.leaf) || null);};
  const escape=e=>{if(e.key==='Escape'){touchLeaf=null;highlight(null);}};
  host.addEventListener('focusin',focus);host.addEventListener('focusout',leave);
  window.addEventListener('keydown',escape);
  function draw(now){
    if(!active)return;
    frame=requestAnimationFrame(draw);
    if(!width||!height)return;
    const dt=clamp((now-last)/1000,0,.12);last=now;elapsed+=dt;
    desired=reduced.matches?1:Math.min(1,desired+dt/12);
    p=desired===1 && 1-p<.0005?1:T.MathUtils.damp(p,desired,11,dt);
    const pull=ease(.26,.94,p),burst=ease(.06,.21,p);
    seed.visible=p<.56;
    halves[0].position.x=burst*.095;halves[1].position.x=-burst*.095;
    halves[0].rotation.z=-burst*.20;halves[1].rotation.z=burst*.20;
    const targetLight=Number(host.dataset.sky==='day');
    light=reduced.matches || Math.abs(targetLight-light)<.001 ? targetLight : T.MathUtils.damp(light,targetLight,4.8,dt);
    bounce.intensity=(1.15+4.8*(1-light))*(1-ease(.30,.56,p));
    sky.update(light,pull,elapsed,width,height,reduced.matches);
    fill.color.lerpColors(...colors[0],light); fill.groundColor.lerpColors(...colors[1],light);
    key.color.lerpColors(...colors[2],light); rim.color.lerpColors(...colors[3],light);
    scene.fog.color.lerpColors(...colors[4],light);
    key.intensity=1.30+1.9*light; fill.intensity=.94+.6*light; rim.intensity=1.95+.35*light;
    renderer.toneMappingExposure=1.04+.19*light;
    roots(p);plant.update(p,reduced.matches?0:elapsed);ground.update(p);implementsScene.update(dt,reduced.matches);
    const dist=T.MathUtils.lerp(3.95,Math.max(15.2,13.6/camera.aspect),pull);
    scene.fog.near=dist*.85;scene.fog.far=dist+15;
    const orbit=.035+.425*pull;
    const targetY=-.88+.89*ease(.17,.42,p)+3.26*ease(.34,.94,p);
    const eyeRise=T.MathUtils.lerp(.045,.19,ease(.28,.65,p));
    camera.position.set(Math.sin(orbit)*dist,targetY+dist*eyeRise,Math.cos(orbit)*dist);
    camera.lookAt(0,targetY,0);
    renderer.render(scene,camera);
    controls.forEach((button,i)=>{
      const prop=implementsScene.props[i];
      const screen=prop.anchor.localToWorld(prop.point.clone()).project(camera);
      button.hidden=!prop.base.visible || p<=.59 || Math.abs(screen.x)>.98 || Math.abs(screen.y)>.98 || screen.z>1;
      button.style.left=((screen.x+1)*50)+'%';button.style.top=((1-screen.y)*50)+'%';
    });
    links.forEach((link,i)=>{
      const blade=targets[i],mid=projectBlade(blade);
      link.style.left=((mid.x+1)*50)+'%';link.style.top=((1-mid.y)*50)+'%';
      link.tabIndex=blade.parent.parent.visible?0:-1;
    });
    if(!callout.hidden){
      if(hover.parent.parent.visible)positionCallout();
      else {touchLeaf=null;highlight(null);}
    }
  }
  if(active){resizeScene();frame=requestAnimationFrame(draw);}
  function setActive(next){
    if(active===next)return;
    active=next;
    if(!active){
      cancelAnimationFrame(frame);
      touchLeaf=null;
      highlight(null);
      p=desired=1;
      return;
    }
    resizeScene();
    // Composite the retained scene before the revealed canvas can be painted.
    last=performance.now();
    draw(last);
  }
  function dispose(){
    cancelAnimationFrame(frame);resize.disconnect();host.removeEventListener('focusin',focus);host.removeEventListener('focusout',leave);host.removeEventListener('click',useControl);
    window.removeEventListener('keydown',escape);holder.removeEventListener('pointerdown',rememberPointer);
    holder.removeEventListener('pointermove',move);holder.removeEventListener('pointerleave',leave);holder.removeEventListener('click',click);holder.removeEventListener('wheel',scrub);holder.removeEventListener('touchstart',start);holder.removeEventListener('touchmove',drag);
    const resources=new Set();scene.traverse(o=>{if(o.geometry)resources.add(o.geometry);if(o.material){resources.add(o.material);if(o.material.userData.turf)resources.add(o.material.userData.turf);if(o.material.map)resources.add(o.material.map);if(o.material.bumpMap)resources.add(o.material.bumpMap);}});
    resources.forEach(r=>r.dispose());sky.dispose();key.shadow.dispose();renderer.dispose();renderer.domElement.remove();
  }
  return {setActive,dispose};
}







