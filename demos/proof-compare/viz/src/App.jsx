import { useEffect, useState } from "react";
import GraphView from "./GraphView.jsx";
import { SCENES } from "./graph-model.js";

// graphs.json is produced by demos/proof-compare/build_all.py and copied into
// public/ so Vite serves it. The app fetches it at runtime — regenerating the
// data does not require rebuilding the JS bundle for local dev.
export default function App() {
  const [data, setData] = useState(null);
  const [active, setActive] = useState(SCENES[0].key);
  const [err, setErr] = useState(null);

  useEffect(() => {
    let alive = true;
    fetch("./graphs.json")
      .then((r) => {
        if (!r.ok) throw new Error(`graphs.json: ${r.status}`);
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
            >
              {s.label}
            </button>
          ))}
        </nav>
      </header>

      {err && <div className="error">Failed to load graphs.json — {err}</div>}
      {!err && !data && <div className="loading">loading…</div>}
      {data && data[scene.key] && (
        <GraphView key={scene.key} graph={data[scene.key]} caption={scene.caption} />
      )}
    </div>
  );
}
