import { useEffect, useRef, useState } from 'react';
import { PhotoPicker } from './FeaturePage';
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
function Icon({ name, size = 20 }) {
  const props = { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.7, strokeLinecap: 'round', strokeLinejoin: 'round', 'aria-hidden': true };
  if (name === 'arrow') return <svg {...props}><path d="M4 12h15m-6-6 6 6-6 6" /></svg>;
  if (name === 'back') return <svg {...props}><path d="M20 12H5m6-6-6 6 6 6" /></svg>;
  if (name === 'upload') return <svg {...props}><path d="M12 16V3m-5 5 5-5 5 5M4 16v4h16v-4" /></svg>;
  if (name === 'camera') return <svg {...props}><path d="M3 7h4l2-3h6l2 3h4v13H3z" /><circle cx="12" cy="13" r="3.5" /></svg>;
  return <svg {...props}><path d="m12 2 1.8 6.2L20 10l-6.2 1.8L12 18l-1.8-6.2L4 10l6.2-1.8z" /></svg>;
}


const prettyLabel = (name) => name.replace(/___/g, ': ').replace(/_/g, ' ').replace(/\s+/g, ' ').trim();
function splitLabel(name) { const [plant, condition = ''] = prettyLabel(name).split(': '); return { plant, condition }; }

export default function DiseaseTool() {
  const [mode, setMode] = useState('upload');
  const [result, setResult] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [webcamOn, setWebcamOn] = useState(false);
  const videoRef = useRef(null), streamRef = useRef(null), abortRef = useRef(null), inputRef = useRef(null);

  useEffect(() => () => { streamRef.current?.getTracks().forEach((track) => track.stop()); abortRef.current?.abort(); }, []);
  useEffect(() => () => { if (previewUrl) URL.revokeObjectURL(previewUrl); }, [previewUrl]);

  async function predictFile(file) {
    abortRef.current?.abort();
    const controller = new AbortController(); abortRef.current = controller;
    const formData = new FormData(); formData.append('file', file);
    setLoading(true); setResult(null); setError(''); setStatus('Analyzing your leaf…');
    try {
      const response = await fetch(`${API_URL}/predict`, { method: 'POST', body: formData, signal: controller.signal });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'The analysis could not be completed.');
      setResult(payload); setStatus('');
    } catch (err) { if (err.name !== 'AbortError') { setError(err.message || 'The analysis could not be completed. Please try again.'); setStatus(''); } }
    finally { if (abortRef.current === controller) setLoading(false); }
  }
  function handleFile(file) { if (!file) return; if (!file.type.startsWith('image/')) { setError('Choose a JPG, PNG, or other image file.'); return; } if (streamRef.current) stopWebcam(); setPreviewUrl(URL.createObjectURL(file)); predictFile(file); }
  async function startWebcam() {
    if (!navigator.mediaDevices?.getUserMedia) { setError('Camera access is unavailable in this browser. Try uploading a photo.'); return; }
    try { const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } } }); streamRef.current = stream; if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play(); } setWebcamOn(true); setError(''); }
    catch (err) { setError(err.name === 'NotAllowedError' ? 'Camera permission was denied. Allow access or upload a photo.' : 'The camera could not start. Try uploading a photo.'); }
  }
  function stopWebcam() { streamRef.current?.getTracks().forEach((track) => track.stop()); streamRef.current = null; if (videoRef.current) videoRef.current.srcObject = null; setWebcamOn(false); }
  function switchMode(next) { setMode(next); setError(''); if (next === 'camera') startWebcam(); else stopWebcam(); }
  function capture() { const video = videoRef.current; if (!webcamOn || !video?.videoWidth) return; const side = Math.min(video.videoWidth, video.videoHeight), out = Math.min(side, 1024); const canvas = document.createElement('canvas'); canvas.width = out; canvas.height = out; canvas.getContext('2d').drawImage(video, (video.videoWidth - side) / 2, (video.videoHeight - side) / 2, side, side, 0, 0, out, out); canvas.toBlob((blob) => { if (blob) handleFile(new File([blob], 'leaf-capture.jpg', { type: 'image/jpeg' })); }, 'image/jpeg', 0.95); }
  function reset() { abortRef.current?.abort(); stopWebcam(); setMode('upload'); setResult(null); setPreviewUrl(''); setError(''); setStatus(''); setLoading(false); }
  const uncertain = result && (result.is_uncertain ?? result.confidence < 0.6);
  const top = result ? splitLabel(result.predicted_class) : null;
  const healthy = top && /healthy/i.test(top.condition);
  const tone = uncertain ? 'warn' : healthy ? 'good' : 'bad';

  return <section className="tool-panel" aria-label="Leaf analysis">
      {!previewUrl && <div className="tabs" role="group" aria-label="Input method">
        <button type="button" aria-pressed={mode === 'upload'} className={mode === 'upload' ? 'tab active' : 'tab'} onClick={() => switchMode('upload')}><Icon name="upload" size={18} /> Upload photo</button>
        <button type="button" aria-pressed={mode === 'camera'} className={mode === 'camera' ? 'tab active' : 'tab'} onClick={() => { if (!webcamOn) switchMode('camera'); }}><Icon name="camera" size={18} /> Use camera</button>
      </div>}
      <div className="stage">{!previewUrl && mode === 'upload' && <PhotoPicker subject="a leaf photo" formats="JPG or PNG recommended" onBrowse={() => inputRef.current?.click()} onDrop={handleFile} />}
        <input ref={inputRef} type="file" accept="image/*" hidden onChange={(event) => { handleFile(event.target.files[0]); event.target.value = ''; }} />
        <div className="camera" style={{ display: mode === 'camera' && !previewUrl ? 'block' : 'none' }}><video ref={videoRef} autoPlay muted playsInline /><div className="guide" />{!webcamOn && <div className="camera-hint">Starting camera…</div>}</div>
        {mode === 'camera' && !previewUrl && <button className="button button-primary capture-button" onClick={capture} disabled={!webcamOn || loading}>Capture leaf <Icon name="camera" size={18} /></button>}
        {previewUrl && <div className="review"><div className="photo"><img src={previewUrl} alt="Leaf submitted for analysis" />{loading && <div className="scan"><span /></div>}</div><div className="verdict" aria-live="polite">{loading && <p className="analysis-status">{status}</p>}{result && top && <><span className={`result-badge ${tone}`}>{uncertain ? 'Uncertain result' : healthy ? 'Healthy leaf' : 'Condition detected'}</span><h3>{top.plant}</h3><p className="condition">{top.condition || 'Condition unavailable'}</p><div className="meter"><div className="meter-head"><span>Confidence</span><strong>{(result.confidence * 100).toFixed(1)}%</strong></div><div className="bar"><i className={tone} style={{ width: `${result.confidence * 100}%` }} /></div></div>{uncertain && <p className="note">Try another photo with one centered leaf and even lighting.</p>}<div className="alternatives"><span className="panel-kicker">OTHER POSSIBILITIES</span><ul>{result.top_predictions?.slice(1).map((item) => { const label = splitLabel(item.class_name); return <li key={item.class_name}><span>{label.plant}{label.condition ? ` · ${label.condition}` : ''}</span><strong>{(item.confidence * 100).toFixed(1)}%</strong></li>; })}</ul></div><p className="result-meta">{result.inference_time_ms.toFixed(0)} ms · model v{result.model_version}</p></>}<button className="text-button" onClick={reset}>Analyze another leaf <Icon name="arrow" size={17} /></button></div></div>}
        {error && <p className="error" role="alert">{error}</p>}</div>{result && <div className="tool-panel-foot">Results are guidance only. Confirm serious cases with an agronomist.</div>}</section>;
}


