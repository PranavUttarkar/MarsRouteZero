import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import {
  metricsAtProgressPlan,
  metricsAtProgressRl,
  planSegmentData,
  rlTrailTotalMeters,
  segmentSlopeDegPlan,
} from "../utils/driveMetrics.js";
import { pathTitle } from "../utils/pathLabels.js";

const PATH_COLORS = { astar: 0xffffff, straight: 0x00d4ff, rl: 0xff6b35 };
/** Stronger spherical falloff so the patch reads as sitting on a planet */
const CURVATURE = 0.00011;
const WHEEL_RADIUS = 0.48;

/** Mars surface gravity (m/s²) */
const G_MARS = 3.71;
/** Rolling resistance (Crr) — kept light so baseline drive dominates */
const MU_ROLL = 0.055;
/** Linear damping (1/s) — light drag so speed stays responsive */
const K_DRAG = 0.038;
/**
 * Baseline drive along the path (m/s²). Gravity adds/subtracts on top of this
 * (downhill faster, uphill slower), rather than this being the only forward push.
 */
const A_BASELINE = 1.225;
/** Tiny assist only if physics nets near zero on extreme edge cases */
const A_STUCK = 0.35;
const V_MAX_MPS = 16;
/** Brisk start so the run does not crawl before physics ramps */
const V_START_MPS = 1.0;

/** Tangent in meters per unit u; local only — does not use path length or start–end distance. */
function tangentComponentsMeters(tan, meta, yScale) {
  const mpp = meta.meters_per_pixel;
  const er = meta.elevation_range_m;
  const tx = tan.x * mpp;
  const tz = tan.z * mpp;
  const ty = tan.y * (er / yScale);
  const dPdu_m = Math.hypot(tx, ty, tz);
  return { tx, ty, tz, dPdu_m };
}

function tangentMetersPerU(tan, meta, yScale) {
  return tangentComponentsMeters(tan, meta, yScale).dPdu_m;
}

/** Total arc length of the drive curve in meters (for progress only — never used to scale v or a). */
function computeCurveLengthMeters(curve, meta, yScale, samples = 160) {
  let sum = 0;
  for (let i = 0; i < samples; i++) {
    const u = (i + 0.5) / samples;
    const dPdu_m = tangentMetersPerU(curve.getTangentAt(u), meta, yScale);
    sum += dPdu_m / samples;
  }
  return Math.max(sum, 1e-6);
}

/**
 * a ≈ A_BASELINE + (gravity along path) + light roll/drag.
 * Gravity term: −G·sinθ speeds you up downhill, slows uphill.
 * v and a depend only on local tangent (slope), not path length or start–end separation.
 * @returns {{ a: number, dPdu_m: number, aGrav: number, aRoll: number, aDrag: number, sinTheta: number, aBaseline: number }}
 */
function tangentialAccelMps2(tan, meta, yScale, vMps) {
  const { tx, ty, tz, dPdu_m } = tangentComponentsMeters(tan, meta, yScale);
  if (dPdu_m < 1e-8) {
    return {
      a: A_BASELINE,
      dPdu_m: 1e-6,
      aGrav: 0,
      aRoll: 0,
      aDrag: 0,
      sinTheta: 0,
      aBaseline: A_BASELINE,
    };
  }
  const sinTheta = ty / dPdu_m;
  const horiz_m = Math.hypot(tx, tz);
  const cosTheta = horiz_m / dPdu_m;

  const aGrav = -G_MARS * sinTheta;
  const aRoll = -MU_ROLL * G_MARS * cosTheta * Math.sign(vMps > 1e-4 ? vMps : 1);
  const aDrag = -K_DRAG * vMps;
  let aNet = A_BASELINE + aGrav + aRoll + aDrag;
  if (vMps < 0.35 && aNet < 0.25) aNet += A_STUCK;

  return {
    a: aNet,
    dPdu_m,
    aGrav,
    aRoll,
    aDrag,
    sinTheta,
    aBaseline: A_BASELINE,
  };
}
const FOG_COLOR = 0x4a1a0a;
/** Exponential smoothing: higher = snappier, lower = more cinematic */
const FOLLOW_CAM_SMOOTH = 5.2;
const FOLLOW_LOOK_SMOOTH = 4.5;
const TANGENT_SMOOTH = 6.0;

function marsColor(t) {
  const x = Math.min(1, Math.max(0, t));
  const c = new THREE.Color();
  const stops = [
    [0, 0x1a0c07],
    [0.22, 0x5a2510],
    [0.42, 0x8b3a18],
    [0.62, 0xc1440e],
    [0.8, 0xd4946a],
    [1.0, 0xf0d4a8],
  ];
  for (let i = 0; i < stops.length - 1; i++) {
    if (x <= stops[i + 1][0]) {
      const f = (x - stops[i][0]) / (stops[i + 1][0] - stops[i][0]);
      c.set(stops[i][1]).lerp(new THREE.Color(stops[i + 1][1]), f);
      return c;
    }
  }
  return c.set(stops[stops.length - 1][1]);
}

function worldY(normH, x, z, yScale) {
  return normH * yScale - (x * x + z * z) * CURVATURE;
}

function buildSky(radius) {
  const cnv = document.createElement("canvas");
  cnv.width = 4;
  cnv.height = 768;
  const ctx = cnv.getContext("2d");
  const g = ctx.createLinearGradient(0, 0, 0, 768);
  g.addColorStop(0.0, "#020208");
  g.addColorStop(0.08, "#05051a");
  g.addColorStop(0.14, "#0a0a28");
  g.addColorStop(0.22, "#12122e");
  g.addColorStop(0.32, "#0c1020");
  g.addColorStop(0.42, "#080818");
  g.addColorStop(0.52, "#120c14");
  g.addColorStop(0.62, "#1a1010");
  g.addColorStop(0.72, "#2a1810");
  g.addColorStop(0.82, "#4a2218");
  g.addColorStop(0.9, "#6a3018");
  g.addColorStop(0.96, "#884020");
  g.addColorStop(1.0, "#a05028");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 4, 768);
  const tex = new THREE.CanvasTexture(cnv);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  const geo = new THREE.SphereGeometry(radius, 64, 64);
  return new THREE.Mesh(
    geo,
    new THREE.MeshBasicMaterial({ map: tex, side: THREE.BackSide, fog: false, depthWrite: false })
  );
}

/** Distant regolith beyond the DEM; follows spherical falloff */
function buildPlanetSkirt(innerR, outerR, minY, _yScale) {
  const segs = 96;
  const geo = new THREE.RingGeometry(innerR, outerR, segs, 2);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position;
  const colors = new Float32Array(pos.count * 3);
  const cDeep = new THREE.Color(0x120a06);
  const cHoriz = new THREE.Color(0x2a140c);
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i);
    const z = pos.getZ(i);
    const r2 = x * x + z * z;
    const r = Math.sqrt(r2);
    const y =
      minY -
      2.5 -
      r2 * CURVATURE * 0.85 -
      Math.max(0, r - innerR) * 0.04;
    pos.setY(i, y);
    const t = Math.min(1, (r - innerR) / Math.max(1e-6, outerR - innerR));
    const c = cDeep.clone().lerp(cHoriz, t * t);
    colors[i * 3] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  return new THREE.Mesh(
    geo,
    new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.97,
      metalness: 0.02,
      flatShading: false,
    })
  );
}

function buildGround(radius, minY) {
  const geo = new THREE.CircleGeometry(radius, 64);
  geo.rotateX(-Math.PI / 2);
  const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0x140a06, roughness: 0.97 }));
  if (minY != null) m.position.y = minY - 2.2;
  return m;
}

function buildDust(halfExtent, count) {
  const pos = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    pos[i * 3] = (Math.random() - 0.5) * halfExtent * 2;
    pos[i * 3 + 1] = Math.random() * 55 + 2;
    pos[i * 3 + 2] = (Math.random() - 0.5) * halfExtent * 2;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  return new THREE.Points(
    geo,
    new THREE.PointsMaterial({
      color: 0xd0a878,
      size: 1.0,
      transparent: true,
      opacity: 0.15,
      sizeAttenuation: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    })
  );
}

/**
 * Procedural Mars 2020–style rover: rocker-bogie deck, six independently driven wheels, mast & RTG.
 * Returns root group, chassis (pitch for terrain), and wheel meshes for rolling animation.
 */
function buildRover() {
  const root = new THREE.Group();
  const chassis = new THREE.Group();
  root.add(chassis);

  const bodyMat = new THREE.MeshStandardMaterial({
    color: 0xb8a898,
    roughness: 0.88,
    metalness: 0.06,
  });
  const darkMat = new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.92 });
  const foilMat = new THREE.MeshStandardMaterial({
    color: 0xc9a85c,
    roughness: 0.42,
    metalness: 0.55,
  });
  const mastMat = new THREE.MeshStandardMaterial({ color: 0x9a9a9a, metalness: 0.35, roughness: 0.45 });

  const deck = new THREE.Mesh(new THREE.BoxGeometry(2.75, 0.38, 1.85), bodyMat);
  deck.position.set(0.1, 0.72, 0);
  chassis.add(deck);

  const avionics = new THREE.Mesh(new THREE.BoxGeometry(1.05, 0.28, 1.35), bodyMat);
  avionics.position.set(-0.35, 1.05, 0);
  chassis.add(avionics);

  const rocker = new THREE.Mesh(new THREE.BoxGeometry(3.0, 0.1, 0.28), darkMat);
  rocker.position.set(0, 0.4, 0);
  chassis.add(rocker);

  const wGeo = new THREE.CylinderGeometry(WHEEL_RADIUS, WHEEL_RADIUS, 0.2, 20);
  wGeo.rotateX(Math.PI / 2);

  const wheels = [];
  const sx = [-1.05, 0, 1.05];
  const sz = [1.02, -1.02];
  sx.forEach((x) => {
    sz.forEach((z) => {
      const w = new THREE.Mesh(wGeo, darkMat);
      w.position.set(x, WHEEL_RADIUS - 0.06, z);
      chassis.add(w);
      wheels.push(w);
    });
  });

  const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 0.85, 8), mastMat);
  mast.position.set(-0.95, 1.62, 0);
  chassis.add(mast);
  const mastExt = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.95, 8), mastMat);
  mastExt.position.set(-0.95, 2.52, 0);
  chassis.add(mastExt);

  const head = new THREE.Mesh(new THREE.BoxGeometry(0.38, 0.22, 0.42), foilMat);
  head.position.set(-0.95, 3.05, 0);
  chassis.add(head);

  const hga = new THREE.Mesh(new THREE.CylinderGeometry(0.35, 0.35, 0.04, 20), foilMat);
  hga.position.set(0.85, 1.35, -0.95);
  hga.rotation.x = Math.PI / 2.3;
  chassis.add(hga);

  const rtg = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.2, 1.0, 10), darkMat);
  rtg.position.set(1.15, 1.05, 0);
  rtg.rotation.z = Math.PI / 2;
  chassis.add(rtg);

  root.visible = false;
  root.userData.chassis = chassis;
  root.userData.wheels = wheels;
  return root;
}

function buildBeacon(hex) {
  const g = new THREE.Group();
  const beamH = 30;
  const beam = new THREE.Mesh(
    new THREE.CylinderGeometry(0.1, 0.1, beamH, 6),
    new THREE.MeshBasicMaterial({
      color: hex,
      transparent: true,
      opacity: 0.45,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    })
  );
  beam.position.y = beamH / 2;
  g.add(beam);

  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(2, 0.2, 8, 28),
    new THREE.MeshBasicMaterial({ color: hex, transparent: true, opacity: 0.7 })
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.3;
  ring.userData.pulse = true;
  g.add(ring);

  const dot = new THREE.Mesh(
    new THREE.SphereGeometry(0.7, 10, 10),
    new THREE.MeshStandardMaterial({ color: hex, emissive: hex, emissiveIntensity: 0.6 })
  );
  dot.position.y = 0.7;
  g.add(dot);

  g.visible = false;
  return g;
}

function dispose3D(o) {
  if (!o) return;
  o.traverse((ch) => {
    if (ch.geometry) ch.geometry.dispose();
    if (ch.material) {
      if (Array.isArray(ch.material)) ch.material.forEach((m) => m.dispose());
      else ch.material.dispose();
    }
  });
}

export default function TerrainScene({
  meta,
  paths,
  rlTrail,
  showCostmap,
  onPick,
  pickEnabled,
  startPos,
  goalPos,
  cameraMode = "orbit",
  flyPath,
  onFlyComplete,
  onDriveMetrics,
  activePlan,
  rlTrailSamples,
  autoRotate = false,
}) {
  const mountRef = useRef(null);
  const internals = useRef(null);
  const clockRef = useRef(new THREE.Clock());
  const animFrameRef = useRef(0);
  const roverAnimRef = useRef({
    active: false,
    progress: 0,
    curve: null,
    speedMps: 0,
    lastPhys: null,
    /** Integrated distance along curve (m) — for HUD/metrics only */
    sMeters: 0,
    /** Precomputed curve length (m) — for progress fraction only; does not scale physics */
    pathLengthMeters: 0,
  });
  const pathCurvesRef = useRef({});
  const showCostmapRef = useRef(showCostmap);
  const onPickRef = useRef(onPick);
  const flyPathRef = useRef(flyPath);
  const cameraModeRef = useRef(cameraMode);
  const onFlyCompleteRef = useRef(onFlyComplete);
  const autoRotateRef = useRef(autoRotate);
  const onDriveMetricsRef = useRef(onDriveMetrics);
  const activePlanRef = useRef(activePlan);
  const rlTrailSamplesRef = useRef(rlTrailSamples);

  onPickRef.current = onPick;
  showCostmapRef.current = showCostmap;
  flyPathRef.current = flyPath;
  cameraModeRef.current = cameraMode;
  onFlyCompleteRef.current = onFlyComplete;
  autoRotateRef.current = autoRotate;
  onDriveMetricsRef.current = onDriveMetrics;
  activePlanRef.current = activePlan;
  rlTrailSamplesRef.current = rlTrailSamples;

  useEffect(() => {
    const el = mountRef.current;
    if (!el || !meta) return;

    const w = el.clientWidth || 800;
    const h = el.clientHeight || 600;
    const gridW = meta.width;
    const gridH = meta.height;
    const yScale = Math.min(160, Math.max(14, meta.elevation_range_m * 0.55));
    const halfW = gridW / 2;
    const halfH = gridH / 2;

    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(FOG_COLOR, 420, 3200);

    const followSmooth = {
      active: false,
      camPos: new THREE.Vector3(),
      lookAt: new THREE.Vector3(),
      tangent: new THREE.Vector3(0, 0, 1),
    };
    let planetSkirt = null;

    const camera = new THREE.PerspectiveCamera(50, w / h, 0.3, 6000);
    camera.position.set(halfW * 0.55, yScale * 1.5, halfH * 0.75);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;
    el.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.06;
    controls.maxPolarAngle = Math.PI * 0.48;
    controls.minDistance = 8;
    controls.maxDistance = 1600;
    controls.target.set(0, yScale * 0.2, 0);
    controls.autoRotate = autoRotateRef.current;
    controls.autoRotateSpeed = 0.35;

    const hemi = new THREE.HemisphereLight(0xd08050, 0x1a0a06, 0.5);
    scene.add(hemi);
    scene.add(new THREE.AmbientLight(0x6a5040, 0.25));
    const sun = new THREE.DirectionalLight(0xfff0d8, 1.35);
    sun.position.set(200, 300, 150);
    scene.add(sun);
    const fill = new THREE.DirectionalLight(0x7090b0, 0.18);
    fill.position.set(-100, 80, -150);
    scene.add(fill);

    const sky = buildSky(3200);
    scene.add(sky);
    const ground = buildGround(2200);
    scene.add(ground);
    const dust = buildDust(halfW * 1.2, 900);
    scene.add(dust);

    const starGeo = new THREE.BufferGeometry();
    const nStars = 4500;
    const starPos = new Float32Array(nStars * 3);
    const starSizes = new Float32Array(nStars);
    for (let i = 0; i < nStars; i++) {
      const r = 2600 + Math.random() * 2400;
      const th = Math.random() * Math.PI * 2;
      const ph = Math.acos(2 * Math.random() - 1);
      const upBias = Math.pow(Math.random(), 0.45);
      const ph2 = ph * (0.35 + upBias * 0.65);
      starPos[i * 3] = r * Math.sin(ph2) * Math.cos(th);
      starPos[i * 3 + 1] = r * Math.cos(ph2) + 180 + Math.random() * 400;
      starPos[i * 3 + 2] = r * Math.sin(ph2) * Math.sin(th);
      starSizes[i] = 0.6 + Math.random() * 1.8;
    }
    starGeo.setAttribute("position", new THREE.BufferAttribute(starPos, 3));
    starGeo.setAttribute("size", new THREE.BufferAttribute(starSizes, 1));
    const starMat = new THREE.ShaderMaterial({
      uniforms: {
        uColor: { value: new THREE.Color(0xc8d8ff) },
        uOpacity: { value: 0.55 },
      },
      vertexShader: `
        attribute float size;
        varying float vBright;
        void main() {
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          vBright = 0.35 + 0.65 * smoothstep(-800.0, 2200.0, position.y);
          gl_PointSize = size * (420.0 / -mvPosition.z);
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: `
        uniform vec3 uColor;
        uniform float uOpacity;
        varying float vBright;
        void main() {
          vec2 c = gl_PointCoord - vec2(0.5);
          float a = 1.0 - smoothstep(0.35, 0.5, length(c));
          if (a < 0.01) discard;
          gl_FragColor = vec4(uColor * vBright, a * uOpacity * vBright);
        }
      `,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      fog: false,
    });
    const stars = new THREE.Points(starGeo, starMat);
    scene.add(stars);

    const pathGroup = new THREE.Group();
    scene.add(pathGroup);
    const markersGroup = new THREE.Group();
    scene.add(markersGroup);
    const hazardGroup = new THREE.Group();
    scene.add(hazardGroup);

    const rover = buildRover();
    scene.add(rover);
    const startBeacon = buildBeacon(0x44ff88);
    const goalBeacon = buildBeacon(0xff4466);
    scene.add(startBeacon);
    scene.add(goalBeacon);

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let terrainMesh = null;

    internals.current = {
      scene,
      camera,
      renderer,
      controls,
      pathGroup,
      markersGroup,
      hazardGroup,
      rover,
      startBeacon,
      goalBeacon,
      dust,
      heightMap: null,
      terrainMesh: null,
      costmapMesh: null,
      gridW,
      gridH,
      yScale,
      halfW,
      halfH,
      meta,
    };

    const loader = new THREE.TextureLoader();
    loader.load(
      "/api/heightmap.png",
      (tex) => {
        const img = tex.image;
        const cnv = document.createElement("canvas");
        const ctx = cnv.getContext("2d");
        if (!ctx) return;
        cnv.width = img.width;
        cnv.height = img.height;
        ctx.drawImage(img, 0, 0);
        const data = ctx.getImageData(0, 0, img.width, img.height).data;

        const hm = new Float32Array(gridW * gridH);
        for (let row = 0; row < gridH; row++) {
          for (let col = 0; col < gridW; col++) {
            const xi = Math.min(img.width - 1, Math.floor((col / gridW) * img.width));
            const yi = Math.min(img.height - 1, Math.floor((row / gridH) * img.height));
            hm[row * gridW + col] = data[(yi * img.width + xi) * 4] / 255;
          }
        }
        if (internals.current) internals.current.heightMap = hm;

        const geo = new THREE.PlaneGeometry(gridW, gridH, gridW - 1, gridH - 1);
        geo.rotateX(-Math.PI / 2);
        const pos = geo.attributes.position;
        const uvs = geo.attributes.uv;
        const colors = new Float32Array(pos.count * 3);

        for (let i = 0; i < pos.count; i++) {
          const u = uvs.getX(i);
          const v = uvs.getY(i);
          const xi = Math.min(img.width - 1, Math.floor(u * img.width));
          const yi = Math.min(img.height - 1, Math.floor((1 - v) * img.height));
          const normH = data[(yi * img.width + xi) * 4] / 255;
          const px = pos.getX(i);
          const pz = pos.getZ(i);
          pos.setY(i, worldY(normH, px, pz, yScale));
          const mc = marsColor(normH);
          colors[i * 3] = mc.r;
          colors[i * 3 + 1] = mc.g;
          colors[i * 3 + 2] = mc.b;
        }
        pos.needsUpdate = true;
        geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
        geo.computeVertexNormals();

        const mat = new THREE.MeshStandardMaterial({
          vertexColors: true,
          roughness: 0.92,
          metalness: 0.03,
          flatShading: false,
        });
        terrainMesh = new THREE.Mesh(geo, mat);
        scene.add(terrainMesh);
        if (internals.current) internals.current.terrainMesh = terrainMesh;

        const box = new THREE.Box3().setFromObject(terrainMesh);
        const minY = box.min.y;
        ground.position.y = minY - 2.2;

        const innerR = Math.hypot(halfW, halfH) * 1.03;
        const outerR = Math.max(2600, innerR * 3.2);
        planetSkirt = buildPlanetSkirt(innerR, outerR, minY, yScale);
        scene.add(planetSkirt);

        loader.load(
          "/api/costmap.png",
          (ctex) => {
            ctex.colorSpace = THREE.LinearSRGBColorSpace;
            ctex.minFilter = THREE.LinearFilter;
            const cg = new THREE.PlaneGeometry(gridW, gridH);
            cg.rotateX(-Math.PI / 2);
            cg.translate(0, box.min.y + yScale * 0.06 + 3, 0);
            const cm = new THREE.MeshBasicMaterial({
              map: ctex,
              transparent: true,
              opacity: 0.38,
              depthWrite: false,
              blending: THREE.AdditiveBlending,
            });
            const cmesh = new THREE.Mesh(cg, cm);
            cmesh.visible = showCostmapRef.current;
            scene.add(cmesh);
            if (internals.current) internals.current.costmapMesh = cmesh;
          },
          undefined,
          () => {}
        );
      },
      undefined,
      (err) => console.error("heightmap", err)
    );

    const onPointerDown = (e) => {
      if (!pickEnabled || !terrainMesh) return;
      const cb = onPickRef.current;
      if (!cb) return;
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObject(terrainMesh, false);
      if (!hits.length) return;
      const uv = hits[0].uv;
      if (!uv) return;
      const col = Math.min(gridW - 1, Math.max(0, Math.floor(uv.x * gridW)));
      const row = Math.min(gridH - 1, Math.max(0, Math.floor((1 - uv.y) * gridH)));
      cb(col, row);
    };
    renderer.domElement.addEventListener("pointerdown", onPointerDown);

    const onResize = () => {
      if (!mountRef.current) return;
      const rw = mountRef.current.clientWidth || 800;
      const rh = mountRef.current.clientHeight || 600;
      camera.aspect = rw / rh;
      camera.updateProjectionMatrix();
      renderer.setSize(rw, rh);
    };
    window.addEventListener("resize", onResize);

    const clock = clockRef.current;
    clock.start();

    const metricAcc = { acc: 0, lastD: 0, lastClock: performance.now() };

    const tick = () => {
      animFrameRef.current = requestAnimationFrame(tick);
      const dt = clock.getDelta();
      const elapsed = clock.elapsedTime;

      controls.autoRotate = autoRotateRef.current;

      dust.rotation.y += dt * 0.01;
      dust.position.x = Math.sin(elapsed * 0.07) * 5;
      dust.position.z = Math.cos(elapsed * 0.05) * 4;

      [startBeacon, goalBeacon].forEach((b) => {
        if (!b.visible) return;
        b.traverse((ch) => {
          if (ch.userData.pulse) {
            const s = 1 + 0.2 * Math.sin(elapsed * 3.5);
            ch.scale.set(s, s, 1);
          }
        });
      });

      const ra = roverAnimRef.current;
      const exp = (k) => 1 - Math.exp(-k * Math.min(dt, 0.05));

      if (ra.active && ra.curve) {
        const metaNow = internals.current?.meta;
        const yScaleNow = internals.current?.yScale;
        const uPre = Math.min(Math.max(ra.progress, 0), 0.9995);
        const tanForPhys = ra.curve.getTangentAt(uPre);
        let v = ra.speedMps ?? V_START_MPS;

        if (metaNow != null && yScaleNow != null) {
          const phys = tangentialAccelMps2(tanForPhys, metaNow, yScaleNow, v);
          v += phys.a * dt;
          v = Math.max(0, Math.min(V_MAX_MPS, v));
          ra.speedMps = v;
          ra.lastPhys = phys;
          const du = (v * dt) / phys.dPdu_m;
          ra.progress += du;
          ra.sMeters += v * dt;
          if (ra.pathLengthMeters > 0) {
            ra.sMeters = Math.min(ra.sMeters, ra.pathLengthMeters);
          }
        } else {
          ra.progress += dt * 0.012;
          ra.lastPhys = null;
        }

        const Lm = ra.pathLengthMeters ?? 0;
        const doneByArc = Lm > 1e-6 && ra.sMeters >= Lm - 0.02;
        const doneByU = ra.progress >= 1;
        if (doneByArc || doneByU) {
          ra.progress = 1;
          if (Lm > 0) ra.sMeters = Lm;
          ra.active = false;
          ra.speedMps = 0;
          ra.lastPhys = null;
          rover.visible = false;
          followSmooth.active = false;
          const ch = rover.userData.chassis;
          if (ch) ch.rotation.x = 0;
          onDriveMetricsRef.current?.(null);
          if (onFlyCompleteRef.current) onFlyCompleteRef.current();
        }
        const tParam = Math.min(ra.progress, 1);
        const tMetric =
          Lm > 1e-6 ? Math.min(1, (ra.sMeters ?? 0) / Lm) : tParam;
        const u = Math.min(Math.max(tParam, 0), 0.9995);
        const pt = ra.curve.getPointAt(u);
        const tanRaw = ra.curve.getTangentAt(u).normalize();
        if (tMetric < 0.02) followSmooth.tangent.copy(tanRaw);
        followSmooth.tangent.lerp(tanRaw, exp(TANGENT_SMOOTH));
        followSmooth.tangent.normalize();
        rover.position.copy(pt);
        rover.lookAt(pt.clone().add(followSmooth.tangent));
        /* Body mesh is long along +X; lookAt aligns local -Z with tangent; -90° local Y aligns +X with forward */
        rover.rotateY(-Math.PI / 2);

        const mpp = metaNow?.meters_per_pixel ?? 1;
        const dsM = (ra.speedMps ?? 0) * dt;
        const wheels = rover.userData.wheels;
        if (wheels?.length && dsM > 0) {
          const rM = WHEEL_RADIUS * mpp;
          const roll = rM > 1e-6 ? -dsM / rM : 0;
          for (let wi = 0; wi < wheels.length; wi++) wheels[wi].rotation.z += roll;
        }
        const ch = rover.userData.chassis;
        if (ch) {
          const horiz = Math.hypot(followSmooth.tangent.x, followSmooth.tangent.z) || 1e-6;
          const targetPitch = -Math.atan2(followSmooth.tangent.y, horiz);
          ch.rotation.x = THREE.MathUtils.lerp(
            ch.rotation.x,
            THREE.MathUtils.clamp(targetPitch, -0.42, 0.42),
            exp(10)
          );
        }

        const plan = activePlanRef.current;
        const rlS = rlTrailSamplesRef.current;
        let telem;
        if (plan?.waypoints?.length >= 2) {
          telem = metricsAtProgressPlan(plan, meta, tMetric);
        } else if (rlS?.length >= 2) {
          telem = metricsAtProgressRl(rlS, meta, tMetric);
        }
        if (telem) {
          if (tMetric < 0.025) {
            metricAcc.lastD = telem.distanceM;
            metricAcc.lastClock = performance.now();
            metricAcc.acc = 0;
          }
          metricAcc.acc += dt;
          if (metricAcc.acc >= 0.05) {
            metricAcc.acc = 0;
            const totalM =
              plan?.waypoints?.length >= 2
                ? planSegmentData(plan, meta).total
                : rlS?.length >= 2
                  ? rlTrailTotalMeters(rlS, meta)
                  : 0;
            const key = flyPathRef.current;
            const label =
              key === "astar"
                ? pathTitle.astar
                : key === "straight"
                  ? pathTitle.straight
                  : pathTitle.rl;
            const energyPerKm =
              plan?.energy_score != null && plan?.total_distance_m > 1
                ? plan.energy_score / (plan.total_distance_m / 1000)
                : null;
            const potholeAvoidance = Math.max(0, Math.min(100, 100 * (1 - telem.hazard)));
            const lp = ra.lastPhys;
            const slopeFromPhys =
              lp != null ? (Math.asin(THREE.MathUtils.clamp(lp.sinTheta, -1, 1)) * 180) / Math.PI : telem.slopeDeg;
            onDriveMetricsRef.current?.({
              progress: tMetric,
              speedMps: Math.max(0, ra.speedMps ?? 0),
              accelMps2: lp?.a ?? 0,
              baselineMps2: lp?.aBaseline ?? A_BASELINE,
              gravAlongMps2: lp?.aGrav ?? 0,
              rollResistMps2: lp?.aRoll ?? 0,
              dragMps2: lp?.aDrag ?? 0,
              distanceM: telem.distanceM,
              slopeDeg: slopeFromPhys,
              roughnessM: telem.roughnessM,
              hazard: telem.hazard,
              fuelScore: telem.fuelScore,
              energyPerKm,
              potholeAvoidance,
              totalM,
              label,
            });
          }
        }
      } else {
        followSmooth.active = false;
      }

      if (cameraModeRef.current === "follow" && ra.active && ra.curve) {
        const t = Math.min(ra.progress, 1);
        const u = Math.min(Math.max(t, 0), 0.9995);
        const pt = ra.curve.getPointAt(u);
        const tan = followSmooth.tangent.clone().normalize();
        const behind = tan.clone().multiplyScalar(-26);
        behind.y = 14;
        const camDest = pt.clone().add(behind);
        const lookTarget = pt.clone().add(tan.multiplyScalar(14));
        lookTarget.y = pt.y + 3.5;

        if (!followSmooth.active || t < 0.015) {
          followSmooth.camPos.copy(camDest);
          followSmooth.lookAt.copy(lookTarget);
          followSmooth.active = true;
        } else {
          followSmooth.camPos.lerp(camDest, exp(FOLLOW_CAM_SMOOTH));
          followSmooth.lookAt.lerp(lookTarget, exp(FOLLOW_LOOK_SMOOTH));
        }
        camera.position.copy(followSmooth.camPos);
        camera.lookAt(followSmooth.lookAt);
        controls.target.lerp(pt, exp(6));
        controls.enabled = false;
      } else {
        followSmooth.active = false;
        controls.enabled = true;
      }

      controls.update();
      renderer.render(scene, camera);
    };
    tick();

    return () => {
      window.removeEventListener("resize", onResize);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      cancelAnimationFrame(animFrameRef.current);
      dispose3D(sky);
      dispose3D(ground);
      if (planetSkirt) dispose3D(planetSkirt);
      dispose3D(dust);
      dispose3D(stars);
      dispose3D(rover);
      dispose3D(startBeacon);
      dispose3D(goalBeacon);
      dispose3D(hazardGroup);
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === el) el.removeChild(renderer.domElement);
      internals.current = null;
    };
  }, [meta, pickEnabled]);

  useEffect(() => {
    if (!flyPath) onDriveMetrics?.(null);
  }, [flyPath, onDriveMetrics]);

  useEffect(() => {
    const s = internals.current;
    if (!s?.hazardGroup || !meta) return;
    const hg = s.hazardGroup;
    while (hg.children.length) {
      const o = hg.children[0];
      hg.remove(o);
      dispose3D(o);
    }
    if (!flyPath) return;
    const { gridW, gridH, yScale, heightMap: hm } = s;
    const pl = activePlan;
    if (pl?.waypoints?.length >= 2 && flyPath !== "rl") {
      for (let i = 0; i < pl.waypoints.length - 1; i++) {
        const slope = segmentSlopeDegPlan(pl, meta, i);
        if (slope < 11) continue;
        const c0 = pl.waypoints[i][0];
        const r0 = pl.waypoints[i][1];
        const c1 = pl.waypoints[i + 1][0];
        const r1 = pl.waypoints[i + 1][1];
        const cm = (c0 + c1) * 0.5;
        const rm = (r0 + r1) * 0.5;
        const x = cm - gridW / 2;
        const z = rm - gridH / 2;
        const ei = Math.min(gridW - 1, Math.max(0, Math.floor(cm)));
        const ej = Math.min(gridH - 1, Math.max(0, Math.floor(rm)));
        const normH = hm ? hm[ej * gridW + ei] : 0.5;
        const y = worldY(normH, x, z, yScale) + 1.8;
        const cone = new THREE.Mesh(
          new THREE.ConeGeometry(0.5, 1.0, 5),
          new THREE.MeshStandardMaterial({
            color: 0xff5522,
            emissive: 0x331100,
            emissiveIntensity: 0.35,
          })
        );
        cone.position.set(x, y, z);
        cone.rotation.x = Math.PI;
        hg.add(cone);
      }
    }
    if (flyPath === "rl" && rlTrailSamples?.length >= 2) {
      const rlS = rlTrailSamples;
      for (let i = 0; i < rlS.length - 1; i++) {
        const a = rlS[i];
        const b = rlS[i + 1];
        const mpp = meta.meters_per_pixel;
        const dx = (b.x - a.x) * mpp;
        const dz = (b.z - a.z) * mpp;
        const eA = a.elevM ?? 0;
        const eB = b.elevM ?? 0;
        const horiz = Math.hypot(dx, dz);
        const slope =
          horiz > 1e-6 ? (Math.atan2(Math.abs(eB - eA), horiz) * 180) / Math.PI : 0;
        if (slope < 13) continue;
        const x = (a.x + b.x) * 0.5;
        const z = (a.z + b.z) * 0.5;
        const y = (a.y + b.y) * 0.5 + 1.2;
        const cone = new THREE.Mesh(
          new THREE.ConeGeometry(0.45, 0.9, 5),
          new THREE.MeshStandardMaterial({
            color: 0xff6622,
            emissive: 0x331100,
            emissiveIntensity: 0.35,
          })
        );
        cone.position.set(x, y, z);
        cone.rotation.x = Math.PI;
        hg.add(cone);
      }
    }
  }, [flyPath, activePlan, rlTrailSamples, meta]);

  useEffect(() => {
    if (internals.current?.costmapMesh) internals.current.costmapMesh.visible = showCostmap;
  }, [showCostmap]);

  useEffect(() => {
    const s = internals.current;
    if (!s || !meta) return;
    const pg = s.pathGroup;
    while (pg.children.length) {
      const o = pg.children[0];
      pg.remove(o);
      dispose3D(o);
    }

    const curves = {};
    const { gridW, gridH, yScale } = s;
    const er = meta.elevation_range_m + 1e-9;

    const makeTube = (plan, colorHex, key) => {
      if (!plan?.waypoints?.length || plan.waypoints.length < 2) return;
      const pts = plan.waypoints.map(([col, row], i) => {
        const x = col - gridW / 2;
        const z = row - gridH / 2;
        const normH = ((plan.elevations_m[i] ?? meta.elevation_min) - meta.elevation_min) / er;
        return new THREE.Vector3(x, worldY(normH, x, z, yScale) + 1.5, z);
      });
      const curve = new THREE.CatmullRomCurve3(pts, false, "centripetal");
      curves[key] = curve;
      const segs = Math.min(640, Math.max(96, pts.length * 32));
      pg.add(
        new THREE.Mesh(
          new THREE.TubeGeometry(curve, segs, 0.2, 8, false),
          new THREE.MeshStandardMaterial({
            color: colorHex,
            emissive: colorHex,
            emissiveIntensity: 0.22,
            transparent: true,
            opacity: 0.82,
            roughness: 0.35,
            metalness: 0.15,
          })
        )
      );
    };

    if (paths?.astar) makeTube(paths.astar, PATH_COLORS.astar, "astar");
    if (paths?.straight) makeTube(paths.straight, PATH_COLORS.straight, "straight");

    if (rlTrail?.length >= 2) {
      const pts = rlTrail.map((p) => new THREE.Vector3(p.x, p.y, p.z));
      const curve = new THREE.CatmullRomCurve3(pts, false, "centripetal");
      curves.rl = curve;
      const segs = Math.min(720, Math.max(120, pts.length * 6));
      pg.add(
        new THREE.Mesh(
          new THREE.TubeGeometry(curve, segs, 0.16, 8, false),
          new THREE.MeshStandardMaterial({
            color: PATH_COLORS.rl,
            emissive: PATH_COLORS.rl,
            emissiveIntensity: 0.28,
            transparent: true,
            opacity: 0.8,
            roughness: 0.38,
            metalness: 0.12,
          })
        )
      );
    }

    pathCurvesRef.current = curves;
  }, [meta, paths, rlTrail]);

  useEffect(() => {
    const s = internals.current;
    if (!s || !meta) return;
    const { startBeacon: sb, goalBeacon: gb, gridW, gridH, yScale, heightMap: hm } = s;

    const placeBeacon = (beacon, pos) => {
      if (!pos) {
        beacon.visible = false;
        return;
      }
      const x = pos.col - gridW / 2;
      const z = pos.row - gridH / 2;
      const normH = hm
        ? hm[Math.min(gridH - 1, pos.row) * gridW + Math.min(gridW - 1, pos.col)]
        : 0.5;
      beacon.position.set(x, worldY(normH, x, z, yScale), z);
      beacon.visible = true;
    };

    placeBeacon(sb, startPos);
    placeBeacon(gb, goalPos);
  }, [startPos, goalPos, meta]);

  useEffect(() => {
    const s = internals.current;
    if (!s) return;
    const curve = flyPath ? pathCurvesRef.current[flyPath] : null;
    if (flyPath && curve) {
      const pathLengthMeters =
        s.meta && s.yScale != null
          ? computeCurveLengthMeters(curve, s.meta, s.yScale)
          : 0;
      roverAnimRef.current = {
        active: true,
        progress: 0,
        curve,
        speedMps: V_START_MPS,
        lastPhys: null,
        sMeters: 0,
        pathLengthMeters,
      };
      const rv = s.rover;
      rv.visible = true;
      const ch = rv.userData.chassis;
      if (ch) ch.rotation.x = 0;
      rv.userData.wheels?.forEach((w) => {
        w.rotation.z = 0;
      });
    } else {
      roverAnimRef.current = {
        active: false,
        progress: 0,
        curve: null,
        speedMps: 0,
        lastPhys: null,
        sMeters: 0,
        pathLengthMeters: 0,
      };
      s.rover.visible = false;
    }
  }, [flyPath]);

  return <div ref={mountRef} className="terrain-canvas" />;
}
