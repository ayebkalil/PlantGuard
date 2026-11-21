import { useEffect, useState } from 'react';
import Beanstalk from './Beanstalk';
import DiseaseTool from './DiseaseTool';
import { toolsList } from './Botany';
import VisionTool from './VisionTool';
import FeaturePage from './FeaturePage';

function selectedTool() {
  const id = location.hash.split('/').pop();
  return toolsList.find(item => item.id === (id === 'banana' ? 'disease' : id));
}

export default function App() {
  const [{ tool, skipGardenIntro }, setRoute] = useState(() => ({
    tool: selectedTool(),
    skipGardenIntro: false,
  }));
  useEffect(() => {
    const navigate = () => {
      if (location.hash === '#/banana') history.replaceState(history.state, '', '#/disease');
      const nextTool = selectedTool();
      setRoute(current => current.tool === nextTool ? current : {
        tool: nextTool,
        skipGardenIntro: current.skipGardenIntro || !current.tool,
      });
      window.scrollTo(0, 0);
    };
    navigate();
    window.addEventListener('hashchange', navigate);
    return () => window.removeEventListener('hashchange', navigate);
  }, []);
  useEffect(() => { document.title = tool ? tool.title + ' · PlantGuard' : 'PlantGuard'; }, [tool]);
  return <>
    {(!tool || skipGardenIntro) && <Beanstalk active={!tool} skipIntro={skipGardenIntro} />}
    {tool && <FeaturePage tool={tool}>
      {tool.id === 'disease' ? <DiseaseTool /> : <VisionTool key={tool.id} tool={tool} />}
    </FeaturePage>}
  </>;
}
