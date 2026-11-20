import { useEffect, useRef, useState } from 'react';
import { Glyph, toolsList } from './Botany';

const leaves = [
  {x:317,y:343,w:199,h:147,left:true,angle:-9,tag:[188,276]},
  {x:319,y:265,w:199,h:145,left:false,angle:8,tag:[455,200]},
  {x:325,y:200,w:151,h:128,left:true,angle:9,tag:[230,119]},
  {x:330,y:144,w:121,h:107,left:false,angle:-9,tag:[409,65]},
];
const organs = [
  {id:'pests',asset:'beetle',x:418,y:164,w:37,h:45,tag:[447,166]},
  {id:'weeds',asset:'weed',x:105,y:345,w:110,h:100,tag:[135,377]},
  {id:'fruit',asset:'fruit',x:341,y:287,w:73,h:76,tag:[410,327]},
];
const unit = n => Math.min(1,Math.max(0,n));
const soil = Array.from({length:190},(_,i)=>{
  const angle=i*2.39996, r=Math.sqrt((i+.5)/190);
  return {x:320+Math.cos(angle)*r*173,y:440+Math.sin(angle)*r*12,rx:.9+(i%7)*.45};
});

function useGrowth() {
  const [progress,setProgress]=useState(()=>matchMedia('(prefers-reduced-motion: reduce)').matches?1:0);
  const frame=useRef(0);
  const value=useRef(progress);
  function seek(next) {
    cancelAnimationFrame(frame.current);
    value.current=unit(next);
    setProgress(value.current);
  }
  function play() {
    seek(0);
    const start=performance.now();
    function tick(now) {
      value.current=unit((now-start)/3200);
      setProgress(value.current);
      if(value.current<1) frame.current=requestAnimationFrame(tick);
    }
    frame.current=requestAnimationFrame(tick);
  }
  useEffect(()=>{
    if(!matchMedia('(prefers-reduced-motion: reduce)').matches) play();
    let previous=window.scrollY;
    const onScroll=()=>{
      const delta=window.scrollY-previous;
      previous=window.scrollY;
      if(delta) seek(value.current+delta/(window.innerHeight*.65));
    };
    window.addEventListener('scroll',onScroll,{passive:true});
    return ()=>{cancelAnimationFrame(frame.current);window.removeEventListener('scroll',onScroll);};
  },[]);
  return {progress,seek,play};
}

export default function Specimen({scout,home,active,setActive}) {
  const {progress,seek,play}=useGrowth();
  const stage=progress<.15?'Seed':progress<.45?'Sprout':progress<.95?'Unfolding':'Fully grown';
  function target(id,body,amount) {
    const item=toolsList.find(t=>t.id===id);
    return <a key={id} href={home+'/'+id} aria-label={item.title} tabIndex={amount>.7?0:-1}
      aria-hidden={amount<=.7} className={'target '+(active===id?'highlight':'')}
      style={{pointerEvents:amount>.7?'auto':'none'}} onMouseEnter={()=>setActive(id)}
      onMouseLeave={()=>setActive(null)} onFocus={()=>setActive(id)} onBlur={()=>setActive(null)}>{body}</a>;
  }
  function marker(id,[x,y],amount) {
    const idx=toolsList.findIndex(i=>i.id===id);
    return <g className="pin" opacity={unit((amount-.72)/.28)} transform={'translate('+x+' '+y+')'}>
      <circle r="22" fill="transparent"/><circle r="14"/><text y="4" textAnchor="middle">{String(idx+1).padStart(2,'0')}</text></g>;
  }
  return <section className="viewer" aria-label="Interactive growing specimen">
    <div className="plate-header"><span><i/>{scout?'Garden observation':'Botanical specimen'}</span><small>{scout?'Garden Scout':'Leaf Atlas'}</small></div>
    <svg className="scene" viewBox="0 0 640 500" role="group" aria-label={scout?'Select an organism or fruit':'Select a leaf'}>
      <defs><linearGradient id="stemPaint"><stop stopColor="#52672a"/><stop offset=".4" stopColor="#bed37b"/><stop offset=".8" stopColor="#779848"/><stop offset="1" stopColor="#435826"/></linearGradient></defs>
      <g className="registration" fill="none"><circle cx="320" cy="250" r="190"/><circle cx="320" cy="250" r="145" strokeDasharray="1 9"/><path d="M62 250h12m494 0h12M320 44v10m0 404v10"/></g>
      <g aria-hidden="true"><ellipse cx="320" cy="443" rx="174" ry="13" className="bed"/>
      {soil.map(({x,y,rx},i)=><ellipse key={i} cx={x} cy={y} rx={rx} ry={rx*.55} fill={['#a09465','#766849','#534e38','#90845e'][i%4]}/>)}</g>
      <ellipse cx="320" cy="432" rx="11" ry="17" fill="#ae8350" stroke="#644c34" transform="rotate(24 320 432)" opacity={1-unit((progress-.09)/.28)}/>
      <path d="M320 442C326 397 307 353 319 280S320 208 330 144Q333 123 327 112" fill="none" stroke="url(#stemPaint)" strokeWidth="6" strokeLinecap="round" pathLength="1" strokeDasharray="1" strokeDashoffset={1-unit((progress-.04)/.66)} opacity={unit(progress*20)}/>
      {leaves.map((leaf,i)=>{
        const grow=unit((progress-.24-i*.13)/.28);
        const id=toolsList[i].id;
        const shape=<g transform={`translate(${leaf.x} ${leaf.y}) rotate(${leaf.angle}) scale(${(leaf.left?-1:1)*grow} ${grow})`}>
          <path d={`M0 0 Q${leaf.w*.04} -1 ${leaf.w*.12} ${-leaf.h*.073}`} fill="none" stroke="#859c4a" strokeWidth="4" strokeLinecap="round"/><image href="/botany/leaf.webp" x="-2" y={-leaf.h} width={leaf.w} height={leaf.h} preserveAspectRatio="none"/>
          {scout&&i===0&&<g fill="#895021" stroke="#cfaf59" strokeWidth="2">{[[80,-68,5],[95,-73,8],[107,-62,4],[64,-61,3],[113,-89,5],[84,-85,3]].map(([x,y,r])=><ellipse key={x} cx={x} cy={y} rx={r} ry={r*.76}/>)}</g>}
        </g>;
        return !scout||i===0?target(id,<>{shape}{marker(id,leaf.tag,grow)}</>,grow):<g key={id}>{shape}</g>;
      })}
      {scout&&<path d="M318 273Q368 253 377 302" fill="none" stroke="#758640" strokeWidth="4" opacity={unit((progress-.62)/.2)}/>}
      {scout&&organs.map(obj=>{
        const a=unit((progress-.64)/.31);
        return target(obj.id,<g opacity={a}><image href={'/botany/'+obj.asset+'.webp'} x={obj.x} y={obj.y} width={obj.w} height={obj.h}/>{marker(obj.id,obj.tag,a)}</g>,a);
      })}
    </svg>
    <div className="plate-caption" aria-live="polite">{active?toolsList.find(i=>i.id===active)?.title:'Select a numbered specimen part'} <span>↗</span></div>
    <div className="growth-control"><button onClick={play} aria-label="Replay growth"><Glyph type="replay"/></button><div><label htmlFor="growth">Growth <strong>{stage}</strong></label><input id="growth" type="range" min="0" max="100" value={Math.round(progress*100)} onChange={e=>seek(Number(e.target.value)/100)} aria-valuetext={stage}/></div></div>
  </section>;
}

