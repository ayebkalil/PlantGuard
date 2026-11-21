import { useEffect, useRef, useState } from 'react';
import { PhotoPicker } from './FeaturePage';

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const percent = (value) => (value * 100).toFixed(1) + '%';
const TASKS = {
  pests: { subject: 'an insect photo', action: 'Detect pests' },
  weeds: { subject: 'a field photo', action: 'Map vegetation' },
  fruit: { subject: 'a fruit photo', action: 'Check freshness' },
};

function Results({ result }) {
  if (result.kind === 'classification') {
    return <div>
      <h3 className="vision-verdict">{result.top_predictions[0].label}</h3>
      <ul className="vision-values">
        {result.top_predictions.map(({ label, confidence }) => (
          <li key={label}><span>{label}</span><strong>{percent(confidence)}</strong></li>
        ))}
      </ul>
    </div>;
  }
  if (result.kind === 'segmentation') {
    return <ul className="vision-values">
      {result.coverage.map(({ label, fraction }) => (
        <li key={label}><span><i className={'coverage-dot ' + label} />{label}</span><strong>{percent(fraction)}</strong></li>
      ))}
    </ul>;
  }
  return result.detections.length ? <ol className="vision-values">
    {result.detections.map(({ label, confidence }, index) => (
      <li key={index}><span>{index + 1}. {label}</span><strong>{percent(confidence)}</strong></li>
    ))}
  </ol> : <p>No pests were detected in this image.</p>;
}

export default function VisionTool({ tool }) {
  const details = TASKS[tool.id];
  const input = useRef(null);
  const request = useRef(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [availability, setAvailability] = useState('checking');
  const [pestBenchmark, setPestBenchmark] = useState(null);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const endpoint = API + '/vision/' + tool.id;

  useEffect(() => {
    const controller = new AbortController();
    fetch(endpoint, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error('Connection failed');
        return response.json();
      })
      .then((model) => {
        setAvailability(model.status);
        setPestBenchmark(model.test_map50_95 ?? null);
      })
      .catch((err) => { if (err.name !== 'AbortError') setAvailability('offline'); });
    return () => { controller.abort(); request.current?.abort(); };
  }, [endpoint]);

  useEffect(() => {
    if (!file) {
      setPreview('');
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function choose(next) {
    if (!next) return;
    request.current?.abort();
    setBusy(false);
    setResult(null);
    if (!/\.(png|jpe?g|bmp)$/i.test(next.name) || next.size > 8 * 1024 * 1024) {
      setFile(null);
      setError('Choose a JPG, PNG, or BMP image up to 8 MB.');
      return;
    }
    setFile(next);
    setError('');
  }

  async function analyze() {
    request.current?.abort();
    const controller = new AbortController();
    request.current = controller;
    const form = new FormData();
    form.append('file', file);
    setBusy(true);
    setError('');
    setResult(null);
    try {
      const response = await fetch(endpoint + '/predict', { method: 'POST', body: form, signal: controller.signal });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'Analysis unavailable.');
      setResult(payload);
    } catch (err) {
      if (err.name !== 'AbortError') setError(err.message);
    } finally {
      if (request.current === controller) setBusy(false);
    }
  }

  const messages = {
    checking: 'Checking model availability…',
    training_required: 'Training is pending. This tool will be ready when its first checkpoint is activated.',
    offline: 'PlantGuard could not connect to its analysis service.',
    unavailable: 'The model could not load. Try again after the serving checkpoint is restored.',
  };

  return <section className="tool-panel" aria-label={tool.title}>
      <input ref={input} type="file" accept=".jpg,.jpeg,.png,.bmp" hidden onChange={(event) => {
        choose(event.target.files[0]);
        event.target.value = '';
      }} />
      {preview ? <div className="vision-photo" aria-label="Photo and analysis overlay">
        <img src={preview} alt="Submitted photo" />
        {result?.overlay && <img className="vision-mask" src={result.overlay} alt="Maize and weed segmentation mask" />}
        {result?.detections?.map(({ box }, index) => <div key={index} className="pest-box" aria-hidden="true"
          style={{ left: percent(box[0]), top: percent(box[1]), width: percent(box[2] - box[0]), height: percent(box[3] - box[1]) }}>
          <span>{index + 1}</span>
        </div>)}
      </div> : <PhotoPicker subject={details.subject} formats="JPG, PNG, BMP · 8 MB max"
        onBrowse={() => input.current?.click()} onDrop={choose} />}
      {file && <div className="vision-actions">
        <button type="button" className="button" disabled={busy || availability !== 'ready'} onClick={analyze}>{busy ? 'Analyzing…' : details.action}</button>
        <button type="button" className="text-button" onClick={() => input.current?.click()}>Change photo</button>
      </div>}
      {error && <p className="error" role="alert">{error}</p>}
      {availability !== 'ready' && <p className="vision-notice" role="status">{messages[availability]}</p>}
      {pestBenchmark !== null && pestBenchmark < .10 && <p className="vision-notice">
        The current pest model scored {percent(pestBenchmark)} mAP50–95 on held-out images. It can miss pests and mislabel detections.
      </p>}
      <div aria-live="polite">{result && <>
        <Results result={result} />
        <p className="result-meta">{result.inference_time_ms.toFixed(0)} ms · model {result.model_version}</p>
      </>}</div>
    </section>;
}
