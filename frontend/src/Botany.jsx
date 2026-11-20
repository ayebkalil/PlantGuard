export const toolsList = [
  { id: 'disease', title: 'Plant disease detection', part: 'Spotted leaf', icon: 'leaf' },
  { id: 'pests', title: 'Pest identification', part: 'Beetle', icon: 'bug' },
  { id: 'weeds', title: 'Weed identification', part: 'Grass tuft', icon: 'grass' },
  { id: 'fruit', title: 'Fruit quality', part: 'Tomato', icon: 'fruit' },
];
const drawings = {
  leaf: 'M4 20C2 8 12 4 21 3c0 10-5 18-15 16m-2 2L16 9m-6 6v-5',
  bug: 'M8 8l-3-3m11 3 3-3M7 11l-4-1m4 5-4 2m14-6 4-1m-4 5 4 2M8 9c0-5 8-5 8 0m-9 3c0-6 10-6 10 0v3c0 7-10 7-10 0zm5-3v10',
  grass: 'M11 21C9 12 6 9 2 8c6 0 10 4 11 10m-1 3c0-8 3-15 8-18-2 5-3 11-3 15m-3 3c2-6 5-9 8-10M8 21h10',
  fruit: 'M12 8c-9-5-13 9-4 13 2 1 3 0 4 0 10 3 15-14 3-13l-3 1m0-1c0-4 2-6 5-6m-5 6L8 5m4 3 4-2',
  mark: 'M12 22V11M12 16C5 17 2 11 3 5c7 1 10 5 9 11Zm0-5c0-6 4-9 10-10-1 7-4 11-10 10Z',
  replay: 'M4 9a8 8 0 1 1 0 7M4 3v6h6',
};
export function Glyph({ type }) {
  return <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d={drawings[type]} /></svg>;
}


