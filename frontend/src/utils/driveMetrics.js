/**
 * Drive telemetry from planner paths (meters, slopes) and RL trails.
 * @typedef {{ waypoints: number[][], elevations_m: number[], total_distance_m?: number, energy_score?: number }} PlanLike
 */

export function segmentMetersPlan(plan, meta, i) {
  const [c0, r0] = plan.waypoints[i];
  const [c1, r1] = plan.waypoints[i + 1];
  const mpp = meta.meters_per_pixel;
  const dx = (c1 - c0) * mpp;
  const dz = (r1 - r0) * mpp;
  const dy = plan.elevations_m[i + 1] - plan.elevations_m[i];
  return Math.hypot(dx, dy, dz);
}

/** Cumulative segment lengths (m) and total path length */
export function planSegmentData(plan, meta) {
  const n = plan.waypoints?.length ?? 0;
  if (n < 2) return { segLens: [], total: 0 };
  const segLens = [];
  let total = 0;
  for (let i = 0; i < n - 1; i++) {
    const m = segmentMetersPlan(plan, meta, i);
    segLens.push(m);
    total += m;
  }
  return { segLens, total };
}

/**
 * @param {PlanLike} plan
 * @param {import('../api/client.js').TerrainMeta} meta
 * @param {number} u progress 0..1
 */
export function metricsAtProgressPlan(plan, meta, u) {
  const { segLens, total } = planSegmentData(plan, meta);
  if (total < 1e-9 || !plan.waypoints?.length)
    return {
      distanceM: 0,
      slopeDeg: 0,
      roughnessM: 0,
      segmentIndex: 0,
      hazard: 0,
      fuelScore: 100,
    };

  const target = Math.min(1, Math.max(0, u)) * total;
  let acc = 0;
  let seg = 0;
  for (; seg < segLens.length; seg++) {
    if (acc + segLens[seg] >= target - 1e-9) break;
    acc += segLens[seg];
  }
  seg = Math.min(seg, segLens.length - 1);
  const sl = segLens[seg] || 1e-9;
  const localT = (target - acc) / sl;
  const i = seg;
  const e0 = plan.elevations_m[i];
  const e1 = plan.elevations_m[i + 1];
  const horiz = Math.sqrt(Math.max(0, sl * sl - (e1 - e0) * (e1 - e0)));
  const slopeDeg =
    horiz > 1e-6 ? (Math.atan2(Math.abs(e1 - e0), horiz) * 180) / Math.PI : 0;

  const ePrev = i > 0 ? plan.elevations_m[i - 1] : e0;
  const eNext = i < plan.waypoints.length - 2 ? plan.elevations_m[i + 2] : e1;
  const roughnessM =
    Math.abs(eNext - 2 * e1 + e0) + Math.abs(e1 - 2 * e0 + ePrev);

  const hazard = Math.min(1, slopeDeg / 35 + roughnessM / 8);
  const fuelScore = Math.max(
    0,
    Math.min(100, 100 - slopeDeg * 1.2 - roughnessM * 3 - hazard * 15)
  );

  return {
    distanceM: target,
    slopeDeg,
    roughnessM,
    segmentIndex: i,
    hazard,
    fuelScore,
  };
}

/** Degrees, segment i -> i+1 */
export function segmentSlopeDegPlan(plan, meta, i) {
  const sl = segmentMetersPlan(plan, meta, i);
  const dy = plan.elevations_m[i + 1] - plan.elevations_m[i];
  const horiz = Math.sqrt(Math.max(0, sl * sl - dy * dy));
  return horiz > 1e-6 ? (Math.atan2(Math.abs(dy), horiz) * 180) / Math.PI : 0;
}

function rlSegmentMeters(a, b, meta) {
  const mpp = meta.meters_per_pixel;
  const dx = (b.x - a.x) * mpp;
  const dz = (b.z - a.z) * mpp;
  const eA = a.elevM ?? 0;
  const eB = b.elevM ?? 0;
  const dy = eB - eA;
  return Math.hypot(dx, dy, dz);
}

export function rlTrailTotalMeters(trail, meta) {
  if (!trail || trail.length < 2) return 0;
  let t = 0;
  for (let i = 0; i < trail.length - 1; i++) t += rlSegmentMeters(trail[i], trail[i + 1], meta);
  return t;
}

/** @param {{x:number,y:number,z:number,elevM?:number}[]} trail */
export function metricsAtProgressRl(trail, meta, u) {
  if (!trail || trail.length < 2) {
    return {
      distanceM: 0,
      slopeDeg: 0,
      roughnessM: 0,
      segmentIndex: 0,
      hazard: 0,
      fuelScore: 100,
    };
  }
  const total = rlTrailTotalMeters(trail, meta);
  if (total < 1e-9) {
    return {
      distanceM: 0,
      slopeDeg: 0,
      roughnessM: 0,
      segmentIndex: 0,
      hazard: 0,
      fuelScore: 100,
    };
  }
  const target = Math.min(1, Math.max(0, u)) * total;
  let acc = 0;
  let seg = 0;
  for (; seg < trail.length - 1; seg++) {
    const sl = rlSegmentMeters(trail[seg], trail[seg + 1], meta);
    if (acc + sl >= target - 1e-9) break;
    acc += sl;
  }
  seg = Math.min(seg, trail.length - 2);
  const sl = rlSegmentMeters(trail[seg], trail[seg + 1], meta) || 1e-9;
  const localT = (target - acc) / sl;
  const a = trail[seg];
  const b = trail[seg + 1];
  const e0 = a.elevM ?? 0;
  const e1 = b.elevM ?? 0;
  const mpp = meta.meters_per_pixel;
  const dx = (b.x - a.x) * mpp;
  const dz = (b.z - a.z) * mpp;
  const horiz = Math.hypot(dx, dz);
  const slopeDeg =
    horiz > 1e-6 ? (Math.atan2(Math.abs(e1 - e0), horiz) * 180) / Math.PI : 0;

  const ePrev = seg > 0 ? trail[seg - 1].elevM ?? e0 : e0;
  const eNext = seg < trail.length - 2 ? trail[seg + 2].elevM ?? e1 : e1;
  const roughnessM =
    Math.abs(eNext - 2 * e1 + e0) + Math.abs(e1 - 2 * e0 + ePrev);

  const hazard = Math.min(1, slopeDeg / 35 + roughnessM / 8);
  const fuelScore = Math.max(
    0,
    Math.min(100, 100 - slopeDeg * 1.2 - roughnessM * 3 - hazard * 15)
  );

  return {
    distanceM: target,
    slopeDeg,
    roughnessM,
    segmentIndex: seg,
    hazard,
    fuelScore,
  };
}
