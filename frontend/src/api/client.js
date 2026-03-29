/** @typedef {{ width: number, height: number, meters_per_pixel: number, elevation_min: number, elevation_max: number, elevation_range_m: number, mean_slope_deg: number, traversable_fraction: number }} TerrainMeta */

/** @typedef {{ waypoints: number[][], elevations_m: number[], total_cost: number, total_distance_m: number, energy_score: number, planner: string }} PlanResult */

/**
 * @returns {Promise<TerrainMeta>}
 */
export async function getTerrainMeta() {
  const r = await fetch("/api/terrain");
  if (!r.ok) throw new Error(`terrain ${r.status}`);
  return r.json();
}

/**
 * @param {{ start_col: number, start_row: number, goal_col: number, goal_row: number, planner: string }} body
 * @returns {Promise<PlanResult>}
 */
export async function planRoute(body) {
  const r = await fetch("/api/plan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`plan ${r.status}`);
  return r.json();
}

/**
 * @param {number} row
 * @param {number} col
 */
export async function getCell(row, col) {
  const r = await fetch(`/api/cell?row=${row}&col=${col}`);
  if (!r.ok) throw new Error(`cell ${r.status}`);
  return r.json();
}

/** @returns {Promise<{ ppo_zip_exists: boolean, onnx_exists: boolean, stable_baselines3: boolean }>} */
export async function getRlStatus() {
  const r = await fetch("/api/rl-status");
  if (!r.ok) throw new Error(`rl-status ${r.status}`);
  return r.json();
}

/** @returns {Promise<{ label: string, points: { col: number, row: number, elevation_m?: number }[] }>} */
export async function getPerseveranceWaypoints() {
  const r = await fetch("/api/perseverance-waypoints");
  if (!r.ok) throw new Error(`waypoints ${r.status}`);
  return r.json();
}
