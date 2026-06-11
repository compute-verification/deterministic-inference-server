import { useEffect, useState } from "react";
import GraphView from "./GraphView.jsx";
import { SCENES, viewParams } from "./graph-model.js";

// graphs.json is produced by demos/proof-compare/build_all.py and copied into
// public/ so Vite serves it. The app fetches it at runtime — regenerating the
// data does not require rebuilding the JS bundle for local dev.
//
// Embedding: ?scene=<key> picks the initial tab; ?src=<url> loads a different
// graphs document (the 4-node tap demo points this at a protocol run's
// generated graph — possibly containing only that one scene).
const PARAMS = viewParams(typeof window !== "undefined" ? window.location.search : "");

export default function App() {
  const [data, setData] = useState(null);
  const [active, setActive] = useState(PARAMS.scene);
  const [err, setErr] = useState(null);

  useEffect(() => {
    let alive = true;
    fetch(PARAMS.src)
      .then((r) => {
        if (!r.ok) throw new Error(`${PARAMS.src}: ${r.status}`);
        return r.json();
      })
      .then((d) => alive && setData(d))
      .catch((e) => alive && setErr(String(e)));
    return () => {
      alive = false;
    };
  }, []);

  const scene = SCENES.find((s) => s.key === active);

  return (
    <div className="app">
      <header>
        <h1>Task graphs</h1>
        <p className="sub">
          One node = one forward pass. Every scenario is auto-generated from a
          real H100 run through one tracer → builder → renderer pipeline.
        </p>
        <nav className="tabs">
          {SCENES.map((s) => (
            <button
              key={s.key}
              className={"tab" + (s.key === active ? " active" : "")}
              onClick={() => setActive(s.key)}
              disabled={data ? !data[s.key] : false}
              title={data && !data[s.key] ? "not present in this graphs document" : undefined}
            >
              {s.label}
            </button>
          ))}
        </nav>
      </header>

      {err && <div className="error">Failed to load {PARAMS.src} — {err}</div>}
      {!err && !data && <div className="loading">loading…</div>}
      {data && data[scene.key] && (
        <GraphView key={scene.key} graph={data[scene.key]} caption={scene.caption} />
      )}
      {data && !data[scene.key] && (
        <div className="error">No “{scene.label}” graph in this document.</div>
      )}
    </div>
  );
}
