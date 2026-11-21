import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { toolsList } from './Botany';
import { bannerCredit, flightSheets, flightPose } from './banner-flight';
import './beanstalk.css';

const skyPreference='plantguard-garden-sky';

function FilmBanner() {
  const canvas=useRef(null);
  useEffect(()=>{
    const plane=canvas.current, frame=plane.parentElement;
    const context=plane.getContext('2d');
    const reduced=matchMedia('(prefers-reduced-motion: reduce)');
    let animation=0, elapsed=0, last;
    const images=flightSheets.map(sheet=>{
      const image=new Image();
      image.onload=redraw;
      image.src=sheet.url;
      return image;
    });
    let lastSheet, lastSourceX;
    function draw(now) {
      const width=frame.clientWidth, height=frame.clientHeight;
      if(!width || !height){last=undefined;return;}
      if(last!==undefined)elapsed+=(now-last)/1000;
      last=now;
      const pose=flightPose(width,height,elapsed,reduced.matches);
      plane.style.width=pose.span+'px';
      plane.style.transform='translate('+pose.x+'px,'+pose.y+'px)';
      const {sheet,sourceX,sourceY}=pose;
      const image=images[Number(pose.compact)];
      if(context && image.complete && image.naturalWidth && (sheet!==lastSheet || sourceX!==lastSourceX)) {
        if(sheet!==lastSheet) {
          plane.width=sheet.width;plane.height=sheet.height;
          context.drawImage(image,0,0,sheet.width,sheet.height,0,0,sheet.width,sheet.height);
        }
        const [x,y,w,h]=sheet.nose;
        context.clearRect(x,y,w,h);
        context.drawImage(image,sourceX,sourceY,w,h,x,y,w,h);
        lastSheet=sheet;lastSourceX=sourceX;
      }
      if(!reduced.matches)animation=requestAnimationFrame(draw);
    }
    function redraw(){cancelAnimationFrame(animation);draw(performance.now());}
    const resize=new ResizeObserver(redraw);
    resize.observe(frame);
    reduced.addEventListener('change',redraw);
    return ()=>{resize.disconnect();images.forEach(image=>{image.onload=null;});reduced.removeEventListener('change',redraw);cancelAnimationFrame(animation);};
  },[]);
  return <div className="film-airspace" aria-hidden="true"><canvas ref={canvas}/></div>;
}

function GardenFilm({day}) {
  const player=useRef(null);
  const position=useRef(0);
  useEffect(()=>{
    const video=player.current;
    const resume=()=>{
      video.currentTime=Math.min(position.current,video.duration);
      if(video.currentTime<video.duration)video.play().catch(()=>{});
    };
    video.addEventListener('loadedmetadata',resume);
    video.src=day?'/botany/beanstalk-day.mp4':'/botany/beanstalk.mp4';
    return ()=>{
      if(video.readyState>=1)position.current=video.currentTime;
      video.removeEventListener('loadedmetadata',resume);
    };
  },[day]);
  return <video ref={player} className="garden-film" muted playsInline aria-hidden="true"
    poster={day?'/botany/buried-seed-day.webp':'/botany/buried-seed.webp'} />;
}

export default function Beanstalk({ active, skipIntro }) {
  const host = useRef(null);
  const scene = useRef(null);
  const sceneMode = useRef({ active, skipIntro });
  const [fallback, setFallback] = useState(false);
  const [day,setDay]=useState(()=>{
    try{return localStorage.getItem(skyPreference)==='day';}catch{return false;}
  });
  const toggleSky=()=>{
    setDay(!day);
    try{localStorage.setItem(skyPreference,day?'night':'day');}catch{/* Storage can be disabled. */}
  };
  useLayoutEffect(() => {
    sceneMode.current = { active, skipIntro };
    scene.current?.setActive(active);
  }, [active, skipIntro]);
  useEffect(() => {
    let disposed = false;
    const stage = host.current;
    import('./beanstalk-scene').then(({ mountGarden }) => {
      if (!disposed) {
        scene.current = mountGarden(stage, sceneMode.current);
        setFallback(!scene.current);
      }
    }).catch(() => { if (!disposed) setFallback(true); });
    return () => {
      disposed = true;
      scene.current?.dispose();
      scene.current = null;
    };
  }, []);
  return <main className={fallback ? "bean-film poster" : "bean-film"} data-sky={day?'day':'night'}
    data-grown={skipIntro || undefined} hidden={!active} ref={host}>
    <h1>PlantGuard</h1>
    <span className="flight-credit">{bannerCredit}</span>
    <button className="sky-switch" type="button" aria-label="Daylight" aria-pressed={day}
      title={day?'Switch to night':'Switch to daylight'} onClick={toggleSky}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19.9 14.4A8.1 8.1 0 0 1 9.6 4.1a8.3 8.3 0 1 0 10.3 10.3Z"/></svg>
      <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="3.8"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/></svg>
    </button>
    {fallback && !skipIntro && !matchMedia("(prefers-reduced-motion: reduce)").matches && <GardenFilm day={day} />}
    <div className="garden-canvas" aria-hidden="true"/>
    {fallback && <FilmBanner />}
    <nav className="leaf-access" aria-label="Plant tools">
      {toolsList.map(tool => <a key={tool.id} href={'#/'+tool.id} data-leaf={tool.id}>
        <span className="leaf-name">{tool.title}</span>
      </a>)}
    </nav>
    <a className="leaf-callout" tabIndex={-1} hidden>
      <span />
      <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8h9M8 4l4 4-4 4" /></svg>
    </a>
    {!fallback && <div className="implement-access">
      <button type="button" hidden data-implement="can" aria-label="Pour the watering can" />
      <button type="button" hidden data-implement="spade" aria-label="Dig with the spade" />
      <button type="button" hidden data-implement="rake" aria-label="Rock the garden rake" />
      <button type="button" hidden data-implement="barrow" aria-label="Lift the wheelbarrow handles" />
    </div>}
  </main>;
}


