import { Glyph, toolsList } from './Botany';

const presentations = {
  disease: {
    tab: 'Disease', heading: ['Plant disease', 'detection'],
    guidance: 'Photograph a single leaf in good light. Upload it or use your camera to check for signs of disease.',
  },
  pests: {
    tab: 'Pests', heading: ['Pest', 'identification'],
    guidance: 'Upload a close-up of the insect to see where pests were detected and which species the model recognizes.',
  },
  weeds: {
    tab: 'Weeds', heading: ['Weed', 'identification'],
    guidance: 'Use a photo taken straight down over a maize field to map the areas covered by crops and weeds.',
  },
  fruit: {
    tab: 'Fruit', heading: ['Fruit', 'quality'],
    guidance: 'Choose a photo of an apple, banana, or orange to check whether it looks fresh or rotten.',
  },
};

export function PhotoPicker({ subject, formats, onBrowse, onDrop }) {
  return <button type="button" className="photo-picker" onClick={onBrowse}
    onDragOver={event => { event.preventDefault(); event.currentTarget.dataset.dragging = 'true'; }}
    onDragLeave={event => { if (!event.currentTarget.contains(event.relatedTarget)) delete event.currentTarget.dataset.dragging; }}
    onDrop={event => {
      event.preventDefault();
      delete event.currentTarget.dataset.dragging;
      onDrop(event.dataTransfer.files[0]);
    }}>
    <span className="picker-label">
      <svg aria-hidden="true" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"><path d="M12 16V3m-5 5 5-5 5 5M4 16v4h16v-4" /></svg>
      Choose {subject}
    </span>
    <small>or drop it here · {formats}</small>
  </button>;
}

export default function FeaturePage({ tool, children }) {
  const { heading, guidance } = presentations[tool.id];
  return <div className={'fieldwork fieldwork-' + tool.id}>
    <header className="field-header">
      <a className="field-brand" href="#/">PlantGuard</a>
      <nav className="field-nav" aria-label="Analysis tools">
        {toolsList.map(item => <a key={item.id} href={'#/' + item.id}
          aria-current={item.id === tool.id ? 'page' : undefined}>
          <Glyph type={item.icon} />{presentations[item.id].tab}
        </a>)}
      </nav>
      <a className="back field-back" href="#/"><span aria-hidden="true">←</span> Back to garden</a>
    </header>
    <main className="field-stage">
      <div className="field-workspace">
        <div className="field-intro">
          <h1>{heading[0]} <em>{heading[1]}</em></h1>
          <p>{guidance}</p>
        </div>
        {children}
      </div>
    </main>
  </div>;
}
