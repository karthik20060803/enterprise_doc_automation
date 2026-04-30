const { useEffect, useMemo, useState } = React;

const bootstrap = window.__APP_BOOTSTRAP__ || {};

function formatPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(2)}%`;
}

function MetricCards({ cards }) {
  if (!cards || cards.length === 0) return null;
  return (
    <div className="metric-grid">
      {cards.map((card) => (
        <div className="metric-item" key={card.label}>
          <span className="label">{card.label}</span>
          <span className="value">{card.value}</span>
        </div>
      ))}
    </div>
  );
}

function App() {
  const [health, setHealth] = useState({
    status: "Checking...",
    color: "#fbbf24",
    modelText: "Checking...",
  });
  const [runnerMessage, setRunnerMessage] = useState({ text: "", type: "normal" });

  const [file, setFile] = useState(null);
  const [threshold, setThreshold] = useState("0.50");
  const [paragraph, setParagraph] = useState(false);
  const [useGpu, setUseGpu] = useState(Boolean(bootstrap.gpuAvailable));
  const [focusUseful, setFocusUseful] = useState(true);
  const [running, setRunning] = useState(false);

  const [profile, setProfile] = useState("core-text");
  const [metricThreshold, setMetricThreshold] = useState("0.50");
  const [metrics, setMetrics] = useState(null);

  const [stats, setStats] = useState(null);
  const [elements, setElements] = useState([]);
  const [fields, setFields] = useState({});
  const [resultJson, setResultJson] = useState(null);
  const [originalImage, setOriginalImage] = useState("");
  const [overlayImage, setOverlayImage] = useState("");
  const [predictionModel, setPredictionModel] = useState(null);

  const messageColor = useMemo(() => {
    if (runnerMessage.type === "error") return "#ef4444";
    if (runnerMessage.type === "success") return "#16a34a";
    return "#94a3b8";
  }, [runnerMessage]);

  async function parseApiResponse(response) {
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) return response.json();
    const text = await response.text();
    return { error: text || `Server returned status ${response.status}` };
  }

  async function checkBackendHealth() {
    try {
      const response = await fetch("/api/health", { method: "GET" });
      const payload = await parseApiResponse(response);
      if (!response.ok) throw new Error(payload.error || "Health check failed.");

      const model = payload.model || {};
      const readerLoaded = Boolean(model.reader_from_pkl);
      setHealth({
        status: payload.status === "ok" ? `Connected (v${payload.app_version})` : "Unknown",
        color: payload.status === "ok" ? "#22c55e" : "#fbbf24",
        modelText: readerLoaded ? "PKL model loaded" : "Runtime reader fallback",
      });
      return true;
    } catch (_err) {
      setHealth({
        status: "Disconnected",
        color: "#ef4444",
        modelText: "Model unavailable",
      });
      return false;
    }
  }

  async function loadProjectMetrics(nextProfile = profile, nextThreshold = metricThreshold) {
    try {
      const response = await fetch(
        `/api/metrics?profile=${encodeURIComponent(nextProfile)}&threshold=${encodeURIComponent(nextThreshold)}`
      );
      const payload = await parseApiResponse(response);
      if (!response.ok) throw new Error(payload.error || "Failed to load metrics.");
      setMetrics(payload.metrics || null);
    } catch (err) {
      setMetrics({
        count: 0,
        mean: 0,
        threshold_pct: 0,
        threshold_hits: 0,
        categories: [],
        error: err.message || "Failed to load metrics.",
      });
    }
  }

  useEffect(() => {
    checkBackendHealth();
    loadProjectMetrics();
  }, []);

  useEffect(() => {
    function handlePaste(event) {
      const items = event.clipboardData?.items || [];
      for (const item of items) {
        if (!item.type || !item.type.startsWith("image/")) continue;
        const blob = item.getAsFile();
        if (!blob) continue;
        const extension = blob.type.includes("png") ? "png" : "jpg";
        const pastedFile = new File([blob], `pasted-marksheet-${Date.now()}.${extension}`, {
          type: blob.type || "image/png",
        });
        setFile(pastedFile);
        setRunnerMessage({
          text: `Pasted image ready (${pastedFile.name}). Click Run Prediction.`,
          type: "success",
        });
        event.preventDefault();
        return;
      }
    }

    window.addEventListener("paste", handlePaste);
    return () => window.removeEventListener("paste", handlePaste);
  }, []);

  async function runOcr(event) {
    event.preventDefault();
    if (!file) {
      setRunnerMessage({ text: "Please select an image file first.", type: "error" });
      return;
    }

    setRunning(true);
    setRunnerMessage({ text: "Running OCR prediction...", type: "normal" });
    setStats(null);
    setElements([]);
    setFields({});
    setResultJson(null);
    setOriginalImage("");
    setOverlayImage("");
    setPredictionModel(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("threshold", threshold || "0.5");
    formData.append("paragraph", paragraph ? "true" : "false");
    formData.append("use_gpu", useGpu ? "true" : "false");
    formData.append("focus_mode", focusUseful ? "useful" : "all");

    try {
      const response = await fetch("/api/process", { method: "POST", body: formData });
      const payload = await parseApiResponse(response);
      if (!response.ok) throw new Error(payload.error || "OCR processing failed.");

      if (payload.message) {
        setOriginalImage(payload.original_image || "");
        setOverlayImage(payload.overlay_image || "");
        setPredictionModel(payload.prediction_model || null);
        setRunnerMessage({ text: payload.message, type: "error" });
        return;
      }

      setStats(payload.stats || null);
      setElements(payload.display_elements || payload.elements || []);
      setFields(payload.extracted_fields || {});
      setResultJson(payload.result_json || null);
      setOriginalImage(payload.original_image || "");
      setOverlayImage(payload.overlay_image || "");
      setPredictionModel(payload.prediction_model || null);
      setRunnerMessage({ text: "Prediction completed successfully.", type: "success" });
    } catch (err) {
      const healthy = await checkBackendHealth();
      const text = healthy
        ? err.message || "Request failed while running prediction."
        : "Backend is not reachable. Start Flask server and refresh.";
      setRunnerMessage({ text, type: "error" });
    } finally {
      setRunning(false);
    }
  }

  function downloadResultJson() {
    if (!resultJson) return;
    const blob = new Blob([JSON.stringify(resultJson, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "idp_prediction_output.json";
    anchor.click();
    URL.revokeObjectURL(url);
  }

  const outputCards = stats
    ? [
        { label: "Detected Regions", value: String(stats.count || 0) },
        { label: "Mean Confidence", value: formatPercent(stats.mean || 0) },
        {
          label: "Threshold Match",
          value: `${Number(stats.threshold_pct || 0).toFixed(2)}% (${stats.threshold_hits || 0}/${stats.count || 0})`,
        },
      ]
    : [];

  const metricCards = metrics
    ? [
        { label: "Regions Evaluated", value: String(metrics.count || 0) },
        { label: "Accuracy", value: formatPercent(metrics.mean || 0) },
        {
          label: "Threshold Match",
          value: `${Number(metrics.threshold_pct || 0).toFixed(2)}% (${metrics.threshold_hits || 0}/${metrics.count || 0})`,
        },
      ]
    : [];

  const fieldEntries = Object.entries(fields || {});

  return (
    <div className="page-shell">
      <header className="hero">
        <div>
          <h1>Enterprise Document Automation</h1>
          <p>React frontend connected to Flask backend with PKL-based OCR prediction.</p>
          <p className="app-meta">
            Version {bootstrap.appVersion || "N/A"} • API Status:{" "}
            <span id="api-health" style={{ color: health.color }}>
              {health.status}
            </span>
          </p>
          <p className="app-meta">Model: {health.modelText}</p>
        </div>
        <div className="status-badge">
          <span>GPU Available</span>
          <strong>{bootstrap.gpuAvailable ? "Yes" : "No"}</strong>
        </div>
      </header>

      <main className="layout-grid">
        <section className="card">
          <h2>OCR Predictor</h2>
          <form className="form-grid" onSubmit={runOcr}>
            <label className="field">
              <span>Document Image</span>
              <input
                type="file"
                accept=".png,.jpg,.jpeg"
                required
                onChange={(e) => setFile(e.target.files?.[0] || null)}
              />
              <span className="app-meta">
                Tip: You can paste a marksheet image directly with Ctrl+V. Selected file:{" "}
                {file ? file.name : "None"}
              </span>
            </label>

            <label className="field">
              <span>Confidence Threshold</span>
              <input
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={threshold}
                onChange={(e) => setThreshold(e.target.value)}
              />
            </label>

            <label className="field checkbox">
              <input
                type="checkbox"
                checked={paragraph}
                onChange={(e) => setParagraph(e.target.checked)}
              />
              <span>Paragraph Mode</span>
            </label>

            <label className="field checkbox">
              <input type="checkbox" checked={useGpu} onChange={(e) => setUseGpu(e.target.checked)} />
              <span>Use GPU if available</span>
            </label>

            <label className="field checkbox">
              <input
                type="checkbox"
                checked={focusUseful}
                onChange={(e) => setFocusUseful(e.target.checked)}
              />
              <span>Highlight only useful fields</span>
            </label>

            <button type="submit" className="btn primary" disabled={running}>
              {running ? "Processing..." : "Run Prediction"}
            </button>
          </form>
          <p className="inline-message" style={{ color: messageColor }}>
            {runnerMessage.text}
          </p>
        </section>

        <section className="card">
          <h2>Project Metrics</h2>
          <div className="metric-controls">
            <label className="field">
              <span>Profile</span>
              <select
                value={profile}
                onChange={(e) => {
                  setProfile(e.target.value);
                  loadProjectMetrics(e.target.value, metricThreshold);
                }}
              >
                <option value="core-text">core-text</option>
                <option value="all-text">all-text</option>
              </select>
            </label>
            <label className="field">
              <span>Threshold</span>
              <input
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={metricThreshold}
                onChange={(e) => setMetricThreshold(e.target.value)}
              />
            </label>
            <button
              className="btn secondary"
              onClick={() => loadProjectMetrics(profile, metricThreshold)}
              type="button"
            >
              Refresh
            </button>
          </div>

          <MetricCards cards={metricCards} />
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Category</th>
                  <th>Count</th>
                  <th>Mean Confidence</th>
                </tr>
              </thead>
              <tbody>
                {(metrics?.categories || []).map((item) => (
                  <tr key={item.category}>
                    <td>{item.category}</td>
                    <td>{item.count}</td>
                    <td>{`${(Number(item.mean_confidence || 0) * 100).toFixed(2)}%`}</td>
                  </tr>
                ))}
                {metrics?.error ? (
                  <tr>
                    <td colSpan="3">{metrics.error}</td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </section>
      </main>

      <section className="card">
        <h2>Prediction Output</h2>
        <MetricCards cards={outputCards} />
        {predictionModel ? (
          <p className="app-meta">Reader source: {predictionModel.reader_source}</p>
        ) : null}

        <div className="images-grid">
          <figure>
            <figcaption>Original</figcaption>
            <img src={originalImage} alt="Original preview" />
          </figure>
          <figure>
            <figcaption>Overlay</figcaption>
            <img src={overlayImage} alt="OCR overlay preview" />
          </figure>
        </div>

        <div className="actions">
          <button className="btn secondary" type="button" disabled={!resultJson} onClick={downloadResultJson}>
            Download JSON
          </button>
        </div>

        <h3>Extracted Key Fields</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Field</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              {fieldEntries.length === 0 ? (
                <tr>
                  <td colSpan="2">No key fields detected from this image.</td>
                </tr>
              ) : (
                fieldEntries.map(([key, value]) => (
                  <tr key={key}>
                    <td>{key}</td>
                    <td>{Array.isArray(value) ? value.join(", ") : String(value)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Order</th>
                <th>Box (x,y,w,h)</th>
                <th>Confidence</th>
                <th>Text</th>
              </tr>
            </thead>
            <tbody>
              {elements.map((row, index) => (
                <tr key={`${row.element_id || "row"}-${index}`}>
                  <td>{row.reading_order || ""}</td>
                  <td>{`${row.x}, ${row.y}, ${row.width}, ${row.height}`}</td>
                  <td>{Number(row.ocr_confidence || 0).toFixed(4)}</td>
                  <td>{row.ocr_text || ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <h3>Result JSON</h3>
        <pre>{resultJson ? JSON.stringify(resultJson, null, 2) : ""}</pre>
      </section>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
