import * as T from 'three';
import { flightSheets, flightPose } from './banner-flight';

// Both the sky and the distant turf sample the same atmosphere at the horizon.
const atmosphere = `
  uniform sampler2D nightSky, daySky, lunarDisc;
  uniform vec2 viewport;
  uniform float daylight, pullback;

  vec3 gardenSky(vec2 screen) {
    float aspect = viewport.x / viewport.y;
    float height = mix(.70, .62, pullback);
    vec2 plate = vec2(.53 + (screen.x-.53)*clamp(aspect/1.7,.49,1.),
                      1. - (1.-screen.y)/height);
    plate.x += .016*(1.-pullback);
    vec3 night = texture2D(nightSky,plate).rgb;
    vec3 day = texture2D(daySky,plate).rgb;
    vec2 center = vec2(.77, .80 + .038*(1.-pullback));
    vec2 delta = (screen-center)*vec2(aspect,1.);
    float radius = min(.037,.055*aspect);
    float dist = length(delta)/radius;
    night += vec3(.105,.12,.14)*exp(-1.12*dist);
    day += vec3(.30,.22,.12)*exp(-.45*dist);
    if (dist < 1.05) {
      vec4 moon = texture2D(lunarDisc,.5 + delta/(2.06*radius));
      night = mix(night,max(moon.rgb,night),moon.a);
      day = mix(day,vec3(1.,.985,.87),1.-smoothstep(.78,.90,dist));
    }
    return mix(night,day,daylight);
  }
`;

export function gardenSky(scene, renderer, earth, day) {
  const loader=new T.TextureLoader();
  const flightMaps=flightSheets.map(sheet=>{
    const map=loader.load(sheet.url);
    // Filter each pose locally so adjacent atlas cells cannot bleed into its edge.
    map.generateMipmaps=false;
    map.minFilter=T.LinearFilter;
    return map;
  });
  const uniforms={
    nightSky:{value:loader.load('/botany/night-sky.webp')},
    daySky:{value:loader.load('/botany/day-sky.webp')},
    lunarDisc:{value:loader.load('/botany/lunar-disc.webp')},
    viewport:{value:new T.Vector2(1,1)},
    daylight:{value:Number(day)}, pullback:{value:0}
  };
  const paint=new T.ShaderMaterial({
    uniforms:{...uniforms, flightMap:{value:flightMaps[0]}, flightBounds:{value:new T.Vector4()},
      panelSize:{value:new T.Vector3()}, noseBounds:{value:new T.Vector4()}, noseFrame:{value:new T.Vector2()},
      flutter:{value:0}}, depthTest:false, depthWrite:false, toneMapped:false,
    vertexShader:'void main(){gl_Position=vec4(position.xy, .9999, 1.);}',
    fragmentShader:atmosphere+`
      uniform sampler2D flightMap;
      uniform vec4 flightBounds;
      uniform vec3 panelSize;
      uniform vec4 noseBounds;
      uniform vec2 noseFrame;
      uniform float flutter;
      void main() {
        vec2 screen=gl_FragCoord.xy/viewport;
        vec3 color=gardenSky(screen);
        vec2 uv=(screen-flightBounds.xy)/flightBounds.zw;
        if(all(greaterThan(uv,vec2(0.))) && all(lessThan(uv,vec2(1.)))) {
          float cloth=1.-smoothstep(.03,.65,uv.x);
          uv.y+=cloth*.028*sin(uv.x*16.+flutter);
          vec2 pixel=vec2(uv.x,1.-uv.y)*panelSize.xy;
          vec2 nose=pixel-noseBounds.xy;
          vec2 atlasScale=vec2(1./panelSize.x,-1./panelSize.z);
          vec4 flight=texture2D(flightMap,vec2(0.,1.)+pixel*atlasScale);
          if(all(greaterThanEqual(nose,vec2(0.))) && all(lessThan(nose,noseBounds.zw))) {
            flight=texture2D(flightMap,vec2(0.,1.)+(noseFrame+nose)*atlasScale);
          }
          vec3 tint=mix(vec3(.78,.83,.92),vec3(1.),daylight);
          color=mix(color,flight.rgb*tint,flight.a);
        }
        gl_FragColor=vec4(color,1.);
      }`
  });
  const sky=new T.Mesh(new T.PlaneGeometry(2,2),paint);
  sky.frustumCulled=false;
  sky.renderOrder=-100;
  scene.add(sky);
  const compileGround=earth.onBeforeCompile;
  earth.onBeforeCompile=function(shader,engine) {
    compileGround.call(this,shader,engine);
    Object.assign(shader.uniforms,uniforms);
    shader.fragmentShader=atmosphere+shader.fragmentShader.replace('#include <fog_fragment>',
      T.ShaderChunk.fog_fragment.replace('fogColor, fogFactor',
        '(fogFactor>.65 ? mix(fogColor, gardenSky(gl_FragCoord.xy/viewport), smoothstep(.65,1.,fogFactor)) : fogColor), fogFactor'));
  };
  return {
    resize(){renderer.getDrawingBufferSize(uniforms.viewport.value);},
    update(light,pull,time,width,height,still){
      uniforms.daylight.value=light;uniforms.pullback.value=pull;
      const {x,y,span,tall,compact,sheet,sourceX,sourceY,sheetHeight}=flightPose(width,height,time,still);
      paint.uniforms.flightBounds.value.set(x/width,1-(y+tall)/height,span/width,tall/height);
      paint.uniforms.flightMap.value=flightMaps[Number(compact)];
      paint.uniforms.panelSize.value.set(sheet.width,sheet.height,sheetHeight);
      paint.uniforms.noseBounds.value.fromArray(sheet.nose);
      paint.uniforms.noseFrame.value.set(sourceX,sourceY);
      paint.uniforms.flutter.value=still?0:time*3.7;
    },
    dispose(){
      for(const key of ['nightSky','daySky','lunarDisc'])uniforms[key].value.dispose();
      flightMaps.forEach(texture=>texture.dispose());
    }
  };
}
