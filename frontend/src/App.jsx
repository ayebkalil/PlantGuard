import { useEffect, useState } from 'react';
import Beanstalk from './Beanstalk';
import DiseaseTool from './DiseaseTool';
import { Glyph, toolsList } from './Botany';

function selectedTool() {
  return toolsList.find(item => item.id === location.hash.split('/').pop());
}

export default function App() {
  const [tool, setTool] = useState(selectedTool);
  useEffect(() => {
    const navigate = () => { setTool(selectedTool()); window.scrollTo(0, 0); };
    window.addEventListener('hashchange', navigate);
    return () => window.removeEventListener('hashchange', navigate);
  }, []);
  useEffect(() => { document.title = tool ? tool.title + ' · PlantGuard' : 'PlantGuard'; }, [tool]);
  if (!tool) return <Beanstalk />;
  return <div className="world atlas"><main className="detail">
    <a className="back" href="#/">← PlantGuard</a>
    {tool.id === 'disease' ? <DiseaseTool /> : <div className="pending">
      <div><Glyph type={tool.icon}/><h1>{tool.title}</h1>
        <p>{tool.id === 'fruit' ? 'Fresh or rotten fruit classification' : tool.title} will be available once the model is connected.</p>
        <a href="#/disease">Open plant disease detection ↗</a>
      </div>
      <img alt="" src={'/botany/' + ({ fruit:'fruit', pests:'beetle', weeds:'weed' }[tool.id]) + '.webp'}/>
    </div>}
  </main></div>;
}
