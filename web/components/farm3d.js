// 3D wind-farm view (three.js). Turbines on low-poly terrain; rotors spin with rotor rpm, nacelles yaw,
// a ground ring shows state (dashed ring = unknown, pulsing halo for warning/fault/tripped), click selects.
// Falls back to a 2D SVG map when WebGL is unavailable.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/OrbitControls.js';
import { createMap2D } from './map2d.js';
import { icon } from './icons.js';
import { STATE_LABEL } from './badges.js';
import { esc, isNum } from '../util.js';

const SCALE = 0.1;            // layout metres -> scene units
const TOWER = 12, BLADE = 6.8;
const DEG = Math.PI / 180;

export function createFarm(container, opts) {
  try {
    return create3D(container, opts);
  } catch (e) {
    console.warn('WebGL unavailable - using the 2D map fallback.', e);
    container.innerHTML = '';
    return createMap2D(container, { ...opts, note: '3D unavailable (WebGL) - 2D map shown' });
  }
}

function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
function stateColor(st) { return new THREE.Color(cssVar(`--st-${STATE_LABEL[st] ? st : 'unknown'}`) || '#888'); }

/** Smooth rolling terrain height; flattened near turbine pads. */
function terrainHeight(x, z) {
  return 1.6 * Math.sin(x * 0.045 + 0.7) * Math.cos(z * 0.06) + 0.9 * Math.sin(x * 0.11 + z * 0.07) + 0.6 * Math.cos(z * 0.13 - x * 0.02) - 0.5;
}

function bladeGeometry() {
  const s = new THREE.Shape();
  s.moveTo(-0.18, 0.2); s.lineTo(0.42, 0.9); s.lineTo(0.22, BLADE); s.lineTo(-0.02, BLADE); s.lineTo(-0.22, 0.8); s.closePath();
  const g = new THREE.ExtrudeGeometry(s, { depth: 0.09, bevelEnabled: false });
  g.translate(0, 0, -0.045);
  return g;
}

function arrowGeometry() {
  const s = new THREE.Shape(); // points toward local -y in shape space, mapped to -z after rotation
  s.moveTo(0, -5); s.lineTo(2.4, -1.4); s.lineTo(0.8, -1.4); s.lineTo(0.8, 4.5); s.lineTo(-0.8, 4.5); s.lineTo(-0.8, -1.4); s.lineTo(-2.4, -1.4); s.closePath();
  const g = new THREE.ShapeGeometry(s); g.rotateX(Math.PI / 2); // shape y -> world z
  return g;
}

function create3D(container, { assets, onSelect }) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'low-power' });
  if (!renderer.getContext()) throw new Error('no WebGL context');
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 1, 1200);
  camera.position.set(-18, 62, 102);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 6, 0);
  controls.enableDamping = true; controls.dampingFactor = 0.08;
  controls.maxPolarAngle = 1.42; controls.minDistance = 35; controls.maxDistance = 260;
  controls.update();

  const hemi = new THREE.HemisphereLight(0xffffff, 0x334455, 1.1); scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(-60, 90, 40); scene.add(sun);

  // terrain
  const tg = new THREE.PlaneGeometry(260, 170, 90, 60); tg.rotateX(-Math.PI / 2);
  const pos = tg.attributes.position;
  for (let i = 0; i < pos.count; i++) pos.setY(i, terrainHeight(pos.getX(i), pos.getZ(i)));
  tg.computeVertexNormals();
  const terrainMat = new THREE.MeshStandardMaterial({ flatShading: true, roughness: 0.95, metalness: 0 });
  const terrain = new THREE.Mesh(tg, terrainMat); scene.add(terrain);

  // shared geometry/materials
  const towerGeo = new THREE.CylinderGeometry(0.32, 0.62, TOWER, 18); towerGeo.translate(0, TOWER / 2, 0);
  const nacGeo = new THREE.BoxGeometry(1.25, 1.25, 3.3); nacGeo.translate(0, 0, 0.6);
  const hubGeo = new THREE.SphereGeometry(0.5, 16, 12);
  const bladeGeo = bladeGeometry();
  const ringGeo = new THREE.RingGeometry(2.5, 3.1, 56); ringGeo.rotateX(-Math.PI / 2);
  const haloGeo = new THREE.RingGeometry(3.3, 4.4, 56); haloGeo.rotateX(-Math.PI / 2);
  const dashGeos = []; for (let k = 0; k < 14; k++) { const g = new THREE.RingGeometry(2.5, 3.1, 6, 1, (k / 14) * Math.PI * 2, (0.55 / 14) * Math.PI * 2); g.rotateX(-Math.PI / 2); dashGeos.push(g); }
  const bodyMat = new THREE.MeshStandardMaterial({ color: 0xe6ebef, roughness: 0.55, metalness: 0.08 });
  const bodyDim = new THREE.MeshStandardMaterial({ color: 0x8792a0, roughness: 0.8, transparent: true, opacity: 0.45 });
  const hitMat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false });

  const T = new Map();
  for (const a of assets) {
    const x = a.x * SCALE, z = -a.y * SCALE, y = terrainHeight(x, z);
    const g = new THREE.Group(); g.position.set(x, y, z); scene.add(g);
    const tower = new THREE.Mesh(towerGeo, bodyMat); g.add(tower);
    const head = new THREE.Group(); head.position.y = TOWER; g.add(head);
    const nac = new THREE.Mesh(nacGeo, bodyMat); head.add(nac);
    const lampMat = new THREE.MeshBasicMaterial({ color: 0x888888 });
    const lamp = new THREE.Mesh(new THREE.SphereGeometry(0.22, 10, 8), lampMat); lamp.position.set(0, 0.75, 1.6); head.add(lamp);
    const hub = new THREE.Mesh(hubGeo, bodyMat); hub.position.z = -1.25; head.add(hub);
    const rotor = new THREE.Group(); rotor.position.z = -1.35; head.add(rotor);
    const blades = [0, 1, 2].map((k) => { const b = new THREE.Mesh(bladeGeo, bodyMat); b.rotation.z = (k * 2 * Math.PI) / 3; rotor.add(b); return b; });
    rotor.rotation.z = Math.random() * Math.PI;
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x888888, transparent: true, opacity: 0.95, side: THREE.DoubleSide, depthWrite: false });
    const ring = new THREE.Mesh(ringGeo, ringMat); ring.position.y = 0.12; g.add(ring);
    const dash = new THREE.Group(); dashGeos.forEach((dg) => dash.add(new THREE.Mesh(dg, ringMat))); dash.position.y = 0.12; g.add(dash);
    const haloMat = new THREE.MeshBasicMaterial({ color: 0x888888, transparent: true, opacity: 0.3, side: THREE.DoubleSide, depthWrite: false });
    const halo = new THREE.Mesh(haloGeo, haloMat); halo.position.y = 0.1; g.add(halo);
    const selMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9, side: THREE.DoubleSide, depthWrite: false });
    const sel = new THREE.Mesh(new THREE.RingGeometry(4.7, 4.95, 64).rotateX(-Math.PI / 2), selMat); sel.position.y = 0.14; g.add(sel);
    const hit = new THREE.Mesh(new THREE.CylinderGeometry(3.2, 3.2, TOWER + BLADE + 2, 8), hitMat); hit.position.y = (TOWER + BLADE) / 2; hit.userData.id = a.id; g.add(hit);
    const label = document.createElement('div'); label.className = 'farm-label';
    label.addEventListener('click', () => onSelect?.(a.id));
    container.appendChild(label);
    T.set(String(a.id), { a, g, head, rotor, blades, tower, nac, hub, ring, dash, halo, sel, lampMat, ringMat, haloMat, selMat, hit, label,
      rpm: 0, yaw: 0, targetYaw: 0, state: 'normal', y });
  }

  // wind arrow floating above the farm
  const windMat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.55, side: THREE.DoubleSide, depthWrite: false });
  const wind = new THREE.Mesh(arrowGeometry(), windMat); wind.position.set(0, 24, -26); scene.add(wind);

  // HUD overlays
  const hud = document.createElement('div'); hud.className = 'farm-hud tr';
  hud.innerHTML = `<svg class="compass" viewBox="-32 -32 64 64" aria-hidden="true"><g class="rot"><circle r="28" fill="none" stroke="currentColor" stroke-opacity=".35"/>
    <text y="-19" text-anchor="middle" font-size="9" fill="currentColor" font-weight="700">N</text>
    <g class="arr"><path d="M0 -17 L5 -6 L1.6 -6 L1.6 15 L-1.6 15 L-1.6 -6 L-5 -6 Z" fill="var(--accent)"/></g></g></svg><div class="wtxt">—</div>`;
  container.appendChild(hud);
  const help = document.createElement('div'); help.className = 'farm-hud br'; help.textContent = 'drag: orbit · wheel: zoom · click: select'; container.appendChild(help);

  // interaction: click to select (ignore drags)
  const ray = new THREE.Raycaster(); const ndc = new THREE.Vector2(); let down = null;
  renderer.domElement.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener('pointerup', (e) => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
    const r = renderer.domElement.getBoundingClientRect();
    ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    const hits = ray.intersectObjects([...T.values()].map((t) => t.hit), false);
    if (hits.length) onSelect?.(hits[0].object.userData.id);
  });

  // theme-dependent colours
  function applyTheme() {
    const light = document.documentElement.getAttribute('data-theme') === 'light';
    terrainMat.color.set(light ? 0xc9d6c4 : 0x1d2a33);
    bodyMat.color.set(light ? 0xf4f6f8 : 0xdfe5ea);
    windMat.color.set(cssVar('--accent') || '#22c3d6');
    for (const t of T.values()) { t.selMat.color.set(cssVar('--accent') || '#22c3d6'); paint(t); }
  }
  function paint(t) {
    const nodata = t.state === 'nodata';
    const c = nodata ? new THREE.Color(cssVar('--st-nodata')) : stateColor(t.state);
    t.ringMat.color.copy(c); t.haloMat.color.copy(c); t.lampMat.color.copy(c);
    t.ring.visible = t.state !== 'unknown'; t.dash.visible = t.state === 'unknown';
    t.halo.visible = ['warning', 'fault', 'tripped'].includes(t.state);
    const m = nodata ? bodyDim : bodyMat;
    t.tower.material = m; t.nac.material = m; t.hub.material = m; t.blades.forEach((b) => { b.material = m; });
  }

  let selected = null, windInfo = { dir: null, speed: null }, last = performance.now(), raf = 0, disposed = false;
  const v3 = new THREE.Vector3();

  function update(turbines, { selected: sel, windDir, windSpeed, rated } = {}) {
    selected = sel;
    windInfo = { dir: windDir, speed: windSpeed };
    for (const s of turbines || []) {
      const t = T.get(String(s.id)); if (!t) continue;
      const st = s.nodata ? 'nodata' : s.state || 'unknown';
      if (st !== t.state) { t.state = st; paint(t); }
      t.rpm = s.nodata || st === 'tripped' ? 0 : Math.max(0, s.rotor_rpm || 0);
      if (isNum(s.nacelle_dir)) t.targetYaw = s.nacelle_dir; else if (isNum(windDir)) t.targetYaw = windDir;
      t.sel.visible = String(s.id) === String(sel);
      t.label.classList.toggle('selected', t.sel.visible);
      const pw = isNum(s.power) ? `${(s.power * 100).toFixed(0)} %` : '—';
      const lab = STATE_LABEL[st] || st;
      t.label.innerHTML = `<span class="badge st-${st}" style="height:18px;padding:0 6px 0 4px">${icon('st_' + st)}${esc(lab)}</span>${esc(s.label ?? t.a.label)} <span class="pw">${pw}</span>`;
      t.label.title = `${s.label ?? t.a.label}: ${lab}, power ${pw} of rated`;
    }
    const wt = hud.querySelector('.wtxt');
    wt.innerHTML = isNum(windDir) ? `<b class="tnum">${isNum(windSpeed) ? windSpeed.toFixed(1) + ' m/s' : '—'}</b><br><span class="muted">from ${Math.round(windDir)}°</span>`
      : `<b class="tnum">${isNum(windSpeed) ? windSpeed.toFixed(1) + ' m/s' : '—'}</b><br><span class="muted">direction n/a</span>`;
    hud.querySelector('.arr').style.display = isNum(windDir) ? '' : 'none';
    wind.visible = isNum(windDir);
  }

  function resize() {
    const w = container.clientWidth || 600, h = container.clientHeight || 400;
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
  }
  const ro = new ResizeObserver(resize); ro.observe(container); resize();

  function frame(now) {
    if (disposed) return;
    raf = requestAnimationFrame(frame);
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    controls.update();
    const w = container.clientWidth, h = container.clientHeight;
    const pulse = 0.22 + 0.18 * (0.5 + 0.5 * Math.sin(now / 320));
    for (const t of T.values()) {
      t.rotor.rotation.z -= (t.rpm * 2 * Math.PI / 60) * dt;          // real-time rotor speed
      const d = ((t.targetYaw - t.yaw + 540) % 360) - 180;            // ease yaw toward target
      t.yaw = (t.yaw + d * Math.min(1, dt * 1.5) + 360) % 360;
      t.head.rotation.y = -t.yaw * DEG;                               // nacelle front (-z) faces the wind
      if (t.halo.visible) t.haloMat.opacity = pulse;
      v3.set(t.g.position.x, t.y + TOWER + BLADE + 1.5, t.g.position.z).project(camera);
      const vis = v3.z < 1;
      t.label.style.display = vis ? '' : 'none';
      if (vis) t.label.style.transform = `translate(-50%, -100%) translate(${((v3.x + 1) / 2) * w}px, ${((1 - v3.y) / 2) * h}px)`;
    }
    if (isNum(windInfo.dir)) wind.rotation.y = -((windInfo.dir + 180) % 360) * DEG;
    // compass follows the camera azimuth
    const az = Math.atan2(camera.position.x - controls.target.x, camera.position.z - controls.target.z) / DEG;
    hud.querySelector('.rot').setAttribute('transform', `rotate(${az})`);
    if (isNum(windInfo.dir)) hud.querySelector('.arr').setAttribute('transform', `rotate(${(windInfo.dir + 180) % 360})`);
    renderer.render(scene, camera);
  }
  for (const t of T.values()) t.label.style.left = t.label.style.top = '0px';
  applyTheme();
  raf = requestAnimationFrame(frame);

  return {
    kind: '3d', update, setTheme: applyTheme,
    dispose() {
      disposed = true; cancelAnimationFrame(raf); ro.disconnect(); controls.dispose();
      scene.traverse((o) => { o.geometry?.dispose?.(); });
      renderer.dispose(); renderer.forceContextLoss?.();
      container.innerHTML = '';
    },
  };
}
