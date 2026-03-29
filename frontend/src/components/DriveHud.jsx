/**
 * Live telemetry overlay while the rover follows a path.
 * @param {{ progress: number, speedMps: number, accelMps2?: number, baselineMps2?: number, gravAlongMps2?: number, rollResistMps2?: number, dragMps2?: number, distanceM: number, slopeDeg: number, roughnessM: number, hazard: number, fuelScore: number, energyPerKm: number | null, potholeAvoidance: number, totalM: number, label: string } | null} props.metrics
 */
export default function DriveHud({ metrics }) {
  if (!metrics) return null;

  const pct = (v) => `${Math.min(100, Math.max(0, v))}%`;

  return (
    <div className="drive-hud" aria-live="polite">
      <div className="drive-hud-title">{metrics.label} telemetry</div>
      <div className="drive-hud-grid">
        <div className="drive-metric">
          <span className="drive-metric-label">Speed</span>
          <span className="drive-metric-val">
            {(metrics.speedMps * 3.6).toFixed(2)} km/h
          </span>
          <div className="drive-bar">
            <div
              className="drive-bar-fill speed"
              style={{
                width: pct(Math.min(100, ((metrics.speedMps * 3.6) / 45) * 100)),
              }}
            />
          </div>
        </div>
        {metrics.accelMps2 != null && Number.isFinite(metrics.accelMps2) && (
          <div className="drive-metric wide">
            <span className="drive-metric-label">Acceleration (along path)</span>
            <span className="drive-metric-val">
              {metrics.accelMps2 >= 0 ? "+" : ""}
              {metrics.accelMps2.toFixed(2)} m/s²
            </span>
          </div>
        )}
        {metrics.gravAlongMps2 != null && Number.isFinite(metrics.gravAlongMps2) && (
          <div className="drive-metric wide">
            <span className="drive-metric-label">Baseline + gravity / roll / drag</span>
            <span className="drive-metric-val drive-metric-val-small">
              base {metrics.baselineMps2 != null ? metrics.baselineMps2.toFixed(2) : "—"} · g·sin θ{" "}
              {metrics.gravAlongMps2.toFixed(2)} · Crr{" "}
              {metrics.rollResistMps2 != null ? metrics.rollResistMps2.toFixed(2) : "—"} · drag{" "}
              {metrics.dragMps2 != null ? metrics.dragMps2.toFixed(2) : "—"} m/s²
            </span>
          </div>
        )}
        <div className="drive-metric">
          <span className="drive-metric-label">Distance</span>
          <span className="drive-metric-val">
            {metrics.distanceM.toFixed(0)} / {metrics.totalM.toFixed(0)} m
          </span>
        </div>
        <div className="drive-metric">
          <span className="drive-metric-label">Slope</span>
          <span className="drive-metric-val">{metrics.slopeDeg.toFixed(1)}°</span>
          <div className="drive-bar">
            <div
              className="drive-bar-fill slope"
              style={{ width: pct((metrics.slopeDeg / 45) * 100) }}
            />
          </div>
        </div>
        <div className="drive-metric">
          <span className="drive-metric-label">Terrain roughness</span>
          <span className="drive-metric-val">{metrics.roughnessM.toFixed(2)} m Δ²</span>
          <div className="drive-bar">
            <div
              className="drive-bar-fill rough"
              style={{ width: pct(Math.min(100, metrics.roughnessM * 8)) }}
            />
          </div>
        </div>
        <div className="drive-metric">
          <span className="drive-metric-label">Fuel efficiency (proxy)</span>
          <span className="drive-metric-val">{metrics.fuelScore.toFixed(0)} / 100</span>
          <div className="drive-bar">
            <div className="drive-bar-fill fuel" style={{ width: pct(metrics.fuelScore) }} />
          </div>
        </div>
        <div className="drive-metric">
          <span className="drive-metric-label">Pothole / hazard avoidance</span>
          <span className="drive-metric-val">
            {metrics.potholeAvoidance.toFixed(0)} / 100
          </span>
          <div className="drive-bar">
            <div
              className="drive-bar-fill avoid"
              style={{ width: pct(metrics.potholeAvoidance) }}
            />
          </div>
        </div>
        {metrics.energyPerKm != null && Number.isFinite(metrics.energyPerKm) && (
          <div className="drive-metric wide">
            <span className="drive-metric-label">Route energy density</span>
            <span className="drive-metric-val">
              {metrics.energyPerKm.toFixed(2)} energy / km (planner total)
            </span>
          </div>
        )}
      </div>
      <p className="drive-hud-note">
        A fixed baseline drive along the route is increased or decreased by Mars gravity along the
        slope (g·sin θ), with light rolling resistance and drag. Roughness uses elevation Laplacian;
        orange pins mark steep segments.
      </p>
    </div>
  );
}
