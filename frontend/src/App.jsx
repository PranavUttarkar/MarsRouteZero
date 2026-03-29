import { useCallback, useEffect, useMemo, useState } from "react";
import TerrainScene from "./components/TerrainScene.jsx";
import StoryPanel from "./components/StoryPanel.jsx";
import {
  getTerrainMeta,
  planRoute,
  getRlStatus,
  getPerseveranceWaypoints,
} from "./api/client.js";
import "./App.css";

export default function App() {
  const [mode, setMode] = useState("story");
  const [meta, setMeta] = useState(null);
  const [err, setErr] = useState(null);
  const [rlStatus, setRlStatus] = useState(null);
  const [start, setStart] = useState(null);
  const [goal, setGoal] = useState(null);
  const [paths, setPaths] = useState({ astar: null, straight: null });
  const [loading, setLoading] = useState(false);
  const [rlTrail, setRlTrail] = useState([]);
  const [rlBusy, setRlBusy] = useState(false);
  const [showCostmap, setShowCostmap] = useState(false);
  const [perseverancePoints, setPerseverancePoints] = useState([]);

  useEffect(() => {
    getTerrainMeta()
      .then(setMeta)
      .catch((e) => setErr(String(e)));
    getRlStatus()
      .then(setRlStatus)
      .catch(() => {});
    getPerseveranceWaypoints()
      .then((w) => setPerseverancePoints(w.points || []))
      .catch(() => {});
  }, []);

  const onTerrainPick = useCallback(
    (col, row) => {
      if (!meta || mode !== "explore") return;
      if (!start) {
        setStart({ col, row });
        setGoal(null);
        setPaths({ astar: null, straight: null });
        setRlTrail([]);
        return;
      }
      if (!goal) {
        setGoal({ col, row });
        return;
      }
      setStart({ col, row });
      setGoal(null);
      setPaths({ astar: null, straight: null });
      setRlTrail([]);
    },
    [meta, start, goal, mode]
  );

  const runPlans = async () => {
    if (!start || !goal) return;
    setLoading(true);
    setErr(null);
    try {
      const [astar, straight] = await Promise.all([
        planRoute({
          start_col: start.col,
          start_row: start.row,
          goal_col: goal.col,
          goal_row: goal.row,
          planner: "astar",
        }),
        planRoute({
          start_col: start.col,
          start_row: start.row,
          goal_col: goal.col,
          goal_row: goal.row,
          planner: "straight",
        }),
      ]);
      setPaths({ astar, straight });
    } catch (e) {
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  };

  const streamRl = useCallback(async () => {
    if (!start || !goal || !meta) return;
    setRlBusy(true);
    setRlTrail([]);
    setErr(null);
    const yScale = Math.max(12, meta.elevation_range_m * 0.85);
    const er = meta.elevation_range_m + 1e-9;
    const toWorld = (col, row, elevM) => ({
      x: col - meta.width / 2,
      z: row - meta.height / 2,
      y: ((elevM - meta.elevation_min) / er) * yScale + 1.2,
    });

    const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${wsProto}//${window.location.host}/ws/rl-episode`);

    const trail = [];
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.pos && msg.elevation_m != null) {
        const [col, row] = msg.pos;
        trail.push(toWorld(col, row, msg.elevation_m));
        setRlTrail([...trail]);
      }
    };
    ws.onerror = () => setErr("WebSocket error (is the API running on :8000?)");
    ws.onclose = () => setRlBusy(false);
    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          start: [start.col, start.row],
          goal: [goal.col, goal.row],
        })
      );
    };
  }, [start, goal, meta]);

  const insight = useMemo(() => {
    if (!paths?.astar || !paths?.straight) return null;
    const a = paths.astar;
    const s = paths.straight;
    const longer = a.total_distance_m > s.total_distance_m * 1.01;
    const lowerEnergy = a.energy_score < s.energy_score * 0.98;
    return { longer, lowerEnergy, a, s };
  }, [paths]);

  const hint = useMemo(() => {
    if (mode === "story") return "";
    if (!start) return "Click terrain: start";
    if (!goal) return "Click terrain: goal";
    return "Plan paths or stream RL";
  }, [mode, start, goal]);

  return (
    <div className="app">
      <aside className="panel">
        <h1>Mars Route Zero</h1>
        {mode === "story" ? (
          <StoryPanel onEnterExplore={() => setMode("explore")} />
        ) : (
          <>
            <button
              type="button"
              className="ghost"
              onClick={() => setMode("story")}
            >
              ← Story
            </button>
            <p className="hint">{hint}</p>
            {start && (
              <p className="mono">
                Start: {start.col}, {start.row}
              </p>
            )}
            {goal && (
              <p className="mono">
                Goal: {goal.col}, {goal.row}
              </p>
            )}
            <label className="toggle">
              <input
                type="checkbox"
                checked={showCostmap}
                onChange={(e) => setShowCostmap(e.target.checked)}
              />
              Cost overlay (traversal heat)
            </label>
            <div className="row">
              <button
                type="button"
                disabled={!start || !goal || loading}
                onClick={runPlans}
              >
                {loading ? "Planning…" : "A* + straight"}
              </button>
              <button
                type="button"
                disabled={!start || !goal || rlBusy}
                onClick={streamRl}
              >
                {rlBusy ? "RL…" : "Stream RL"}
              </button>
            </div>
            <button
              type="button"
              className="ghost"
              onClick={() => {
                setStart(null);
                setGoal(null);
                setPaths({ astar: null, straight: null });
                setRlTrail([]);
              }}
            >
              Clear picks
            </button>
            {rlStatus && (
              <p className="small muted">
                Policy:{" "}
                {rlStatus.ppo_zip_exists ? "PPO zip found" : "heuristic only"} · ONNX{" "}
                {rlStatus.onnx_exists ? "ok" : "run export_onnx"}
              </p>
            )}
            {paths.astar && (
              <div className="stats">
                <h3>Comparison</h3>
                <p>
                  A* — distance {paths.astar.total_distance_m.toFixed(1)} m · energy{" "}
                  {paths.astar.energy_score.toFixed(1)}
                </p>
                <p>
                  Straight — distance {paths.straight.total_distance_m.toFixed(1)} m · energy{" "}
                  {paths.straight.energy_score.toFixed(1)}
                </p>
                {insight && (
                  <p className="insight">
                    {insight.longer && insight.lowerEnergy
                      ? "A* is longer in distance but can use less energy on this pair — the tradeoff NASA cares about."
                      : insight.lowerEnergy
                        ? "A* scores lower energy here than a straight chord across the cost field."
                        : "Compare visually: steeper cells raise energy even when the path looks shorter in map view."}
                  </p>
                )}
              </div>
            )}
            {err && <p className="err">{err}</p>}
          </>
        )}
      </aside>
      <main className="scene-wrap">
        {meta && mode === "explore" && (
          <TerrainScene
            meta={meta}
            paths={paths}
            rlTrail={rlTrail}
            perseverancePoints={perseverancePoints}
            showCostmap={showCostmap}
            onPick={onTerrainPick}
            pickEnabled
          />
        )}
        {meta && mode === "story" && (
          <TerrainScene
            meta={meta}
            paths={{ astar: null, straight: null }}
            rlTrail={[]}
            perseverancePoints={perseverancePoints}
            showCostmap={false}
            pickEnabled={false}
          />
        )}
        {!meta && !err && <div className="loading">Loading terrain…</div>}
      </main>
    </div>
  );
}
