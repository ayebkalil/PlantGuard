import { useEffect, useState } from 'react';
import Beanstalk from './Beanstalk';
import DiseaseTool from './DiseaseTool';
import { toolsList } from './Botany';
import VisionTool from './VisionTool';

const routes = [...toolsList, { id: 'banana', title: 'Banana ripeness', icon: 'fruit' }];

function selectedTool() {
  return routes.find(item => item.id === location.hash.split('/').pop());
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
    {tool.id === 'disease' ? <DiseaseTool /> : <VisionTool key={tool.id} tool={tool} />}
  </main></div>;
}
