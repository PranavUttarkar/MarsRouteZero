import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

const COLORS = { astar: 0xffffff, straight: 0x00d4ff, rl: 0xff6b35 };

/**
 * @param {object} props
 * @param {import('../api/client.js').TerrainMeta | null} props.meta
 * @param {{ astar?: import('../api/client.js').PlanResult, straight?: import('../api/client.js').PlanResult } | null} props.paths
 * @param {{ x: number, y: number, z: number }[]} props.rlTrail
 * @param {{ col: number, row: number, elevation_m?: number }[]} props.perseverancePoints
 * @param {boolean} props.showCostmap
 * @param {(col: number, row: number) => void} [props.onPick]
 * @param {boolean} props.pickEnabled
 */
export default function TerrainScene({
  meta,
  paths,
  rlTrail,
  perseverancePoints,
  showCostmap,
  onPick,
  pickEnabled,
}) {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const rendererRef = useRef(null);
  const cameraRef = useRef(null);
  const controlsRef = useRef(null);
  const terrainRef = useRef(null);
  const pathGroupRef = useRef(null);
  const markersGroupRef = useRef(null);
  const costmapMeshRef = useRef(null);
  const animRef = useRef(0);
  const raycasterRef = useRef(new THREE.Raycaster());
  const pointerRef = useRef(new THREE.Vector2());
  const onPickRef = useRef(onPick);
  const showCostmapRef = useRef(showCostmap);
  onPickRef.current = onPick;
  showCostmapRef.current = showCostmap;

  useEffect(() => {
    const el = mountRef.current;
    if (!el || !meta) return;

    const w = el.clientWidth || 800;
    const h = el.clientHeight || 600;
    const gridW = meta.width;
    const gridH = meta.height;
    const yScale = Math.max(12, meta.elevation_range_m * 0.85);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d1117);
    scene.fog = new THREE.Fog(0x0d1117, 400, 900);

    const camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 5000);
    camera.position.set(0, gridW * 0.85, gridH * 0.85);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0, 0);

    scene.add(new THREE.AmbientLight(0x8899aa, 0.5));
    const sun = new THREE.DirectionalLight(0xffc879, 1.1);
    sun.position.set(120, 200, 80);
    scene.add(sun);

    const pathGroup = new THREE.Group();
    scene.add(pathGroup);
    const markersGroup = new THREE.Group();
    scene.add(markersGroup);

    sceneRef.current = scene;
    cameraRef.current = camera;
    rendererRef.current = renderer;
    controlsRef.current = controls;
    pathGroupRef.current = pathGroup;
    markersGroupRef.current = markersGroup;

    const loader = new THREE.TextureLoader();
    loader.load(
      "/api/heightmap.png",
      (tex) => {
        tex.colorSpace = THREE.SRGBColorSpace;
        const img = tex.image;
        const cnv = document.createElement("canvas");
        const ctx = cnv.getContext("2d");
        if (!ctx) return;
        cnv.width = img.width;
        cnv.height = img.height;
        ctx.drawImage(img, 0, 0);
        const im = ctx.getImageData(0, 0, img.width, img.height);
        const data = im.data;

        const sampleR = (u, v) => {
          const xi = Math.min(img.width - 1, Math.max(0, Math.floor(u * img.width)));
          const yi = Math.min(img.height - 1, Math.max(0, Math.floor((1 - v) * img.height)));
          return data[(yi * img.width + xi) * 4] / 255;
        };

        const geo = new THREE.PlaneGeometry(gridW, gridH, gridW - 1, gridH - 1);
        geo.rotateX(-Math.PI / 2);
        const pos = geo.attributes.position;
        const uvs = geo.attributes.uv;
        const emin = meta.elevation_min;
        const er = meta.elevation_range_m + 1e-9;
        for (let i = 0; i < pos.count; i++) {
          const u = uvs.getX(i);
          const v = uvs.getY(i);
          const r = sampleR(u, v);
          const elev = emin + r * (meta.elevation_max - emin);
          const y = ((elev - emin) / er) * yScale;
          pos.setY(i, y);
        }
        pos.needsUpdate = true;
        geo.computeVertexNormals();

        const mat = new THREE.MeshStandardMaterial({
          map: tex,
          roughness: 0.92,
          metalness: 0.02,
        });
        const mesh = new THREE.Mesh(geo, mat);
        scene.add(mesh);
        terrainRef.current = mesh;

        loader.load(
          "/api/costmap.png",
          (ctex) => {
            ctex.colorSpace = THREE.SRGBColorSpace;
            const cg = new THREE.PlaneGeometry(gridW, gridH);
            cg.rotateX(-Math.PI / 2);
            cg.translate(0, yScale * 0.55 + 8, 0);
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
            costmapMeshRef.current = cmesh;
          },
          undefined,
          () => {}
        );
      },
      undefined,
      (err) => console.error("heightmap", err)
    );

    const onPointerDown = (e) => {
      if (!pickEnabled || !terrainRef.current) return;
      const cb = onPickRef.current;
      if (!cb) return;
      const rect = renderer.domElement.getBoundingClientRect();
      pointerRef.current.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      pointerRef.current.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycasterRef.current.setFromCamera(pointerRef.current, camera);
      const hits = raycasterRef.current.intersectObject(terrainRef.current, false);
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

    const tick = () => {
      animRef.current = requestAnimationFrame(tick);
      controls.update();
      renderer.render(scene, camera);
    };
    tick();

    return () => {
      window.removeEventListener("resize", onResize);
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      cancelAnimationFrame(animRef.current);
      costmapMeshRef.current = null;
      controls.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === el) el.removeChild(renderer.domElement);
    };
  }, [meta, pickEnabled]);

  useEffect(() => {
    const cm = costmapMeshRef.current;
    if (cm) cm.visible = showCostmap;
  }, [showCostmap]);

  useEffect(() => {
    const g = markersGroupRef.current;
    const m = meta;
    if (!g || !m) return;
    while (g.children.length) {
      const o = g.children[0];
      g.remove(o);
      o.geometry?.dispose();
      o.material?.dispose();
    }
    if (!perseverancePoints?.length) return;
    const yScale = Math.max(12, m.elevation_range_m * 0.85);
    const er = m.elevation_range_m + 1e-9;
    const gw = m.width;
    const gh = m.height;
    for (const p of perseverancePoints) {
      const x = p.col - gw / 2;
      const z = p.row - gh / 2;
      const em = p.elevation_m ?? m.elevation_min;
      const y = ((em - m.elevation_min) / er) * yScale + 2.5;
      const geom = new THREE.SphereGeometry(1.6, 10, 10);
      const mat = new THREE.MeshStandardMaterial({
        color: 0xc1440e,
        emissive: 0x441100,
        emissiveIntensity: 0.25,
      });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.position.set(x, y, z);
      g.add(mesh);
    }
  }, [perseverancePoints, meta]);

  useEffect(() => {
    const pathGroup = pathGroupRef.current;
    const m = meta;
    if (!pathGroup || !m) return;

    while (pathGroup.children.length) {
      const o = pathGroup.children[0];
      pathGroup.remove(o);
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose();
    }

    const yScale = Math.max(12, m.elevation_range_m * 0.85);
    const er = m.elevation_range_m + 1e-9;
    const gw = m.width;
    const gh = m.height;

    const tubeFromPlan = (plan, colorHex) => {
      if (!plan?.waypoints?.length) return;
      const pts = plan.waypoints.map(([col, row], i) => {
        const x = col - gw / 2;
        const z = row - gh / 2;
        const em = plan.elevations_m[i] ?? m.elevation_min;
        const y = ((em - m.elevation_min) / er) * yScale + 1.2;
        return new THREE.Vector3(x, y, z);
      });
      if (pts.length < 2) return;
      const curve = new THREE.CatmullRomCurve3(pts);
      const tube = new THREE.TubeGeometry(
        curve,
        Math.min(256, pts.length * 4),
        0.45,
        6,
        false
      );
      const mat = new THREE.MeshStandardMaterial({
        color: colorHex,
        emissive: colorHex,
        emissiveIntensity: 0.25,
      });
      pathGroup.add(new THREE.Mesh(tube, mat));
    };

    if (paths?.astar) tubeFromPlan(paths.astar, COLORS.astar);
    if (paths?.straight) tubeFromPlan(paths.straight, COLORS.straight);

    if (rlTrail?.length >= 2) {
      const pts = rlTrail.map((p) => new THREE.Vector3(p.x, p.y, p.z));
      const curve = new THREE.CatmullRomCurve3(pts);
      const tube = new THREE.TubeGeometry(
        curve,
        Math.min(400, rlTrail.length * 3),
        0.35,
        5,
        false
      );
      const mat = new THREE.MeshStandardMaterial({
        color: COLORS.rl,
        emissive: COLORS.rl,
        emissiveIntensity: 0.35,
      });
      pathGroup.add(new THREE.Mesh(tube, mat));
    }
  }, [meta, paths, rlTrail]);

  return <div ref={mountRef} className="terrain-canvas" />;
}
