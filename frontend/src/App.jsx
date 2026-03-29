import { useEffect, useState } from "react";

export default function App() {
  const [meta, setMeta] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    fetch("/api/terrain")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setMeta)
      .catch((e) => setErr(String(e)));
  }, []);

  return (
    <div style={{ fontFamily: "system-ui", padding: "2rem", maxWidth: 640 }}>
      <h1>Mars Route Zero</h1>
      <p>Jezero terrain API smoke test (Three.js scene next).</p>
      {err && <p style={{ color: "crimson" }}>{err}</p>}
      {meta && (
        <pre style={{ background: "#111", color: "#e0e0e0", padding: "1rem" }}>
          {JSON.stringify(meta, null, 2)}
        </pre>
      )}
    </div>
  );
}
