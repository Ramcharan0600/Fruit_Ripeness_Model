import { useEffect, useMemo, useState } from 'react';

const API_BASE_URL = (import.meta.env.VITE_API_URL || 'http://localhost:5000').replace(/[/]+$/, '');
// Free-tier ngrok shows an HTML interstitial unless this header is sent.
const NGROK_HEADERS = { 'ngrok-skip-browser-warning': 'true' };

function App() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [predictionMode, setPredictionMode] = useState('compare');
  const [image, setImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [error, setError] = useState('');

  useEffect(() => {
    async function loadModels() {
      try {
        const res = await fetch(`${API_BASE_URL}/api/models`, { headers: NGROK_HEADERS });
        const data = await res.json();
        const modelList = data.models || [];
        setModels(modelList);
        if (modelList.length > 0) {
          setSelectedModel(modelList[0].name);
        }
      } catch (err) {
        setError('Could not connect to the prediction server. Start the backend first.');
      }
    }
    loadModels();
  }, []);

  const benchmarkLeaderboard = useMemo(() => {
    return [...models].sort((a, b) => {
      const accA = a.accuracy ?? -1;
      const accB = b.accuracy ?? -1;
      return (accB ?? -1) - (accA ?? -1);
    });
  }, [models]);

  const handleImageChange = (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setImage(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResults([]);
    setError('');
  };

  const handlePredict = async () => {
    if (!image) {
      setError('Please upload an image before making predictions.');
      return;
    }
    if (predictionMode === 'single' && !selectedModel) {
      setError('Please select a model to run.');
      return;
    }

    const formData = new FormData();
    formData.append('file', image);
    const isSingleModel = predictionMode === 'single';
    const endpoint = isSingleModel ? '/api/predict' : '/api/compare';
    if (isSingleModel) formData.append('model', selectedModel);

    setLoading(true);
    setError('');

    try {
      const res = await fetch(`${API_BASE_URL}${endpoint}`, {
        headers: NGROK_HEADERS,
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || 'Prediction failed');
      }
      const predictions = isSingleModel
        ? [{
            model: data.model,
            label: data.result.benchmark?.label || data.model,
            type: data.result.type,
            predicted_class: data.result.predicted_class,
            confidence: data.result.confidence,
            probabilities: data.result.probabilities,
            labels: data.result.labels,
            benchmark: data.result.benchmark,
          }]
        : data.comparison || [];
      setResults(predictions);
    } catch (err) {
      setError(err.message || 'Unable to generate predictions.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">ML Project Demo</p>
          <h1>Banana Ripeness Comparison Dashboard</h1>
        </div>
      </header>

      <main className="page-grid">
        <section className="panel control-panel">
          <h2>Upload & Evaluate</h2>

          <label className="field-label" htmlFor="prediction-mode">Prediction mode</label>
          <select
            id="prediction-mode"
            value={predictionMode}
            onChange={(event) => {
              setPredictionMode(event.target.value);
              setResults([]);
            }}
          >
            <option value="compare">Compare all models</option>
            <option value="single">Use a single model</option>
          </select>

          {predictionMode === 'single' && (
            <>
              <label className="field-label" htmlFor="single-model">Select model</label>
              <select
                id="single-model"
                value={selectedModel}
                onChange={(event) => setSelectedModel(event.target.value)}
                disabled={models.length === 0}
              >
                {models.length === 0 ? <option value="">No models available</option> : models.map((model) => (
                  <option key={model.name} value={model.name}>{model.label}</option>
                ))}
              </select>
              <p className="field-hint">Only the selected model will run for this image.</p>
            </>
          )}

          <label className="field-label upload-label">
            <span>Upload banana image</span>
            <input type="file" accept="image/*" onChange={handleImageChange} />
          </label>

          {previewUrl && (
            <div className="preview-wrap">
              <img src={previewUrl} alt="Selected banana" className="preview" />
            </div>
          )}

          <button className="predict-btn" onClick={handlePredict} disabled={loading || !image || models.length === 0}>
            {loading ? 'Running prediction...' : predictionMode === 'single' ? 'Predict with selected model' : 'Compare all models'}
          </button>

          {error && <div className="error-box">{error}</div>}
        </section>

        <section className="panel benchmark-panel">
          <h2>Model Accuracy Benchmark</h2>
          <div className="benchmark-grid">
            {benchmarkLeaderboard.map((model) => (
              <div key={model.name} className="metric-card">
                <div className="metric-label">{model.label}</div>
                <div className="metric-value">
                  {model.accuracy !== null && model.accuracy !== undefined ? `${(model.accuracy * 100).toFixed(1)}%` : 'N/A'}
                </div>
                <div className="metric-meta">
                  MAE: {model.mae !== null && model.mae !== undefined ? model.mae.toFixed(4) : 'N/A'}
                </div>
              </div>
            ))}
          </div>

        </section>
      </main>

      <section className="panel comparison-panel">
        <div className="section-header">
          <h2>{predictionMode === 'single' ? 'Selected Model Prediction' : 'Cross-Model Prediction Comparison'}</h2>
        </div>

        {results.length === 0 ? (
          <div className="empty-state">
            Upload an image and click <strong>{predictionMode === 'single' ? 'Predict with selected model' : 'Compare all models'}</strong> to see predictions.
          </div>
        ) : (
          <div className="comparison-grid">
            {results.map((item) => (
              <div key={item.model} className="result-card">
                <div className="result-header">
                  <div>
                    <div className="model-name">{item.label}</div>
                    <div className="model-type">{item.type}</div>
                  </div>
                  <span className={`badge ${item.predicted_class.toLowerCase().replace(/\s+/g, '-')}`}>
                    {item.predicted_class}
                  </span>
                </div>

                <div className="confidence-row">
                  <span>Confidence</span>
                  <strong>{(item.confidence * 100).toFixed(1)}%</strong>
                </div>

                <div className="probability-box">
                  {item.labels.map((label, index) => {
                    const value = item.probabilities[index] ?? 0;
                    return (
                      <div key={`${item.model}-${label}`} className="bar-row">
                        <div className="bar-label">{label}</div>
                        <div className="bar-track">
                          <div className="bar-fill" style={{ width: `${Math.min(value * 100, 100)}%` }} />
                        </div>
                        <div className="bar-value">{(value * 100).toFixed(1)}%</div>
                      </div>
                    );
                  })}
                </div>

                {item.benchmark?.accuracy !== null && item.benchmark?.accuracy !== undefined && (
                  <div className="bench-row">
                    <span>Benchmark Accuracy</span>
                    <strong>{(item.benchmark.accuracy * 100).toFixed(1)}%</strong>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

export default App;
