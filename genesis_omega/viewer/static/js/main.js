// Genesis Ω Observatory — boot, modes (galaxy → system → planet → organism), live stream, panels,
// inspector, timeline, replay of frozen windows, WebXR, sonification and the reality bridge.

import * as THREE from "three";
import { OrbitControls } from "/static/vendor/OrbitControls.js";
import { api, bin, hasToken, post, Stream } from "./api.js";
import { $, $$, DEG, clamp, esc, fmt, idColor, latLonToXYZ, xyzToLatLon } from "./util.js";
import { GalaxyView } from "./galaxy.js";
import { SystemView } from "./system.js";
import { PlanetView } from "./planet.js";
import * as P from "./panels.js";
import { TimelineBar } from "./timeline.js";
import { Sonifier } from "./audio.js";

// ------------------------------------------------------------------ state
const S = {
  hello: null, role: "viewer", cosmos: "", entered: false, enterUtick: 0,
  mode: null, view: null, switching: false, world: 0, star: null, timeline: "PRIME",
  status: null, telemetry: null, galaxy: null, species: new Map(), speciesFetched: 0,
  events: [], panel: null, selected: null, following: null, overlay: 0, lastVisit: 0, bridge: null,
};
window.genesis = S; // inspectable from the console (and by the headless test)

// ------------------------------------------------------------------ renderer
const canvas = $("#gl");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
renderer.xr.enabled = true;
const scene = new THREE.Scene();
const BACKGROUND = new THREE.Color(0x02030a);
scene.background = BACKGROUND;
const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 2000);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
const root = new THREE.Group(); // content container (scaled / placed in XR)
scene.add(root);
const raycaster = new THREE.Raycaster();

function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize);
resize();

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fade = (on) => { $("#fade").style.opacity = on ? 1 : 0; };
const speciesColor = (sid) => S.species.get(sid) || idColor(sid);

function setSpecies(table) {
  for (const s of table) S.species.set(s.id, s.color);
  S.speciesFetched = performance.now();
}

// ------------------------------------------------------------------ toasts & status line
function statusLine(text) {
  $("#start-status").textContent = text;
}

function toast(html, cls = "", onclick = null, ms = 9000) {
  const box = $("#toasts");
  const t = document.createElement("div");
  t.className = `toast ${cls}`;
  t.innerHTML = html;
  if (onclick) t.addEventListener("click", () => { onclick(); t.remove(); });
  else t.addEventListener("click", () => t.remove());
  box.appendChild(t);
  while (box.children.length > 4) box.firstChild.remove();
  setTimeout(() => t.remove(), ms);
}

function eventToast(e) {
  toast(`<div class="k">${esc(e.type.replace(/_/g, " "))} · ${esc(e.epistemic)}</div>${esc(e.title)}`,
    e.significance >= 3 ? "extraordinary" : "", () => inspectEvent(e.id));
}

function errorToast(e) {
  toast(`<div class="k">ERROR</div>${esc(e && e.message ? e.message : e)}`, "", null, 7000);
}

// ------------------------------------------------------------------ modes
function focusStarOfWorld(w) {
  const fs = S.galaxy ? S.galaxy.meta.focus_stars : [];
  return fs && fs.length ? fs[Math.min(w, fs.length - 1)] : 0;
}

function setClip(mode) {
  const [near, far] = mode === "planet" ? [0.002, 60] : mode === "system" ? [0.01, 2000] : [0.05, 600];
  camera.near = near;
  camera.far = far;
  camera.updateProjectionMatrix();
}

async function setMode(mode, opts = {}) {
  if (S.switching) return;
  S.switching = true;
  fade(true);
  await sleep(260);
  let view = null;
  try {
    const q = { timeline: S.timeline };
    if (mode === "galaxy") {
      view = new GalaxyView(S.galaxy);
    } else if (mode === "system") {
      const star = opts.star !== undefined ? opts.star : focusStarOfWorld(S.world);
      const sys = await api(`/api/system/${star}`, q);
      S.star = star;
      view = new SystemView(sys, S.galaxy, S.status ? S.status.galaxy_years : 0);
    } else if (mode === "planet") {
      const w = opts.world !== undefined ? opts.world : S.world;
      const [terrain, fields, structs, species] = await Promise.all([
        bin(`/api/world/${w}/terrain`, q), bin(`/api/world/${w}/fields`, q), bin(`/api/world/${w}/structures`, q),
        api(`/api/world/${w}/species`, q),
      ]);
      S.world = w;
      setSpecies(species);
      view = new PlanetView(terrain, { overlay: S.overlay, speciesColor });
      view.setFields(fields.arrays);
      view.setStructures(structs.arrays);
      const ws = S.status && S.status.worlds ? S.status.worlds[w] : null;
      if (ws && ws.subsolar) view.setSubsolar(ws.subsolar[0], ws.subsolar[1]);
      stream.subscribe({ world: w, timeline: S.timeline });
    }
  } catch (e) {
    errorToast(e);
  }
  if (view) {
    if (S.view) {
      root.remove(S.view.group);
      S.view.dispose();
    }
    S.view = view;
    S.mode = mode;
    root.add(view.group);
    setClip(mode);
    view.cameraHome(camera, controls);
    if (mode === "galaxy" && opts.star !== undefined) view.focusStar(opts.star, camera, controls);
    if (mode === "planet" && opts.lat !== undefined) view.lookAtLatLon(opts.lat, opts.lon, camera, controls);
    if (mode !== "planet") S.following = null;
    if (mode === "planet" && S.selected !== null) view.setSelected(S.selected);
    else $("#inspector").classList.add("hidden");
    if (mode === "system") inspect(P.inspectStar(view.sys));
    $$("#modes [data-mode]").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  }
  S.switching = false;
  fade(false);
}

// ------------------------------------------------------------------ timelines
async function refreshTimelines() {
  try {
    const f = await api("/api/forks");
    const opts = ["PRIME", ...f.active];
    if (!opts.includes(S.timeline)) opts.push(S.timeline);
    const sel = $("#timeline");
    if (Array.from(sel.options).map((o) => o.value).join() !== opts.join()) {
      sel.innerHTML = opts.map((o) => `<option value="${esc(o)}">${o === "PRIME" ? "PRIME timeline" : esc(o)}</option>`).join("");
    }
    sel.value = S.timeline;
  } catch (e) { /* offline: keep the list */ }
}

async function switchTimeline(tl) {
  if (tl === S.timeline) return;
  S.timeline = tl;
  S.events = [];
  try {
    S.events = await api("/api/events", { timeline: tl, limit: 2000 });
  } catch (e) {
    errorToast(e);
  }
  stream.subscribe({ timeline: tl, world: S.world });
  timelineBar.set(S.events, S.status ? S.status.utick : 1);
  toast(`<div class="k">TIMELINE</div>Now observing ${esc(tl)}${tl === "PRIME" ? "" : " — a marked fork: interventions here are EXTERNAL events"}`);
  if (S.mode) await setMode(S.mode, { world: S.world, star: S.star === null ? undefined : S.star });
  if (S.panel) refreshPanel();
}

// ------------------------------------------------------------------ live stream
const stream = new Stream({
  open() {
    $("#offline").classList.add("hidden");
  },
  close() {
    if (S.entered) {
      $("#offline span").textContent = S.status ? fmt(S.status.utick) : "—";
      $("#offline").classList.remove("hidden");
    }
  },
  json(m) {
    if (m.type === "status") onStatus(m);
  },
  binary(msg) {
    if (S.mode !== "planet" || !S.view || S.switching) return;
    const tl = msg.meta.timeline || "PRIME";
    if (tl !== S.timeline || (msg.meta.world !== undefined && msg.meta.world !== S.world)) return;
    const v = S.view;
    if (msg.kind === "agents") {
      v.setAgents(msg.arrays, msg.meta, performance.now());
      maybeRefreshSpecies(msg.arrays.species);
    } else if (msg.kind === "fields") {
      v.setFields(msg.arrays);
    } else if (msg.kind === "structdelta") {
      v.applyStructDelta(msg.arrays);
    } else if (msg.kind === "structures") {
      v.setStructures(msg.arrays);
    }
  },
});

function maybeRefreshSpecies(sp) {
  if (performance.now() - S.speciesFetched < 8000) return;
  for (let k = 0; k < sp.length; k += 7) {
    if (!S.species.has(sp[k])) {
      S.speciesFetched = performance.now();
      api(`/api/world/${S.world}/species`, { timeline: S.timeline }).then(setSpecies).catch(() => {});
      return;
    }
  }
}

let lastSeenSave = 0;
function onStatus(m) {
  if ((m.timeline || "PRIME") !== S.timeline) return;
  if (m.missing && S.timeline !== "PRIME") {
    toast(`<div class="k">TIMELINE</div>${esc(S.timeline)} is no longer running`);
    switchTimeline("PRIME");
    return;
  }
  S.status = m.status;
  S.telemetry = m.telemetry;
  updateHud();
  if (m.events && m.events.length) {
    const known = new Set(S.events.slice(-400).map((e) => e.id));
    for (const e of m.events) {
      if (known.has(e.id)) continue;
      S.events.push(e);
      if (S.entered && e.significance >= 2 && e.utick >= S.enterUtick) eventToast(e);
    }
    if (S.events.length > 5000) S.events = S.events.filter((e, i) => e.significance > 0 || i > S.events.length - 2000);
  }
  timelineBar.set(S.events, S.status.utick);
  const w = S.status.worlds ? S.status.worlds[S.world] : null;
  if (S.mode === "planet" && S.view && w && w.subsolar) S.view.setSubsolar(w.subsolar[0], w.subsolar[1]);
  if (S.mode === "system" && S.view) S.view.setTime(S.status.galaxy_years);
  if (audio.on && w) audio.update(w, meanEmission());
  if (S.entered && performance.now() - lastSeenSave > 20000) {
    lastSeenSave = performance.now();
    try { localStorage.setItem(`genesis.seen.${S.cosmos}`, String(S.status.utick)); } catch (e) { /* ignore */ }
  }
}

function meanEmission() {
  const v = S.view;
  if (S.mode !== "planet" || !v || !v.n) return 0;
  let s = 0;
  for (let k = 0; k < v.n; k++) s += v.glow[k];
  return s / v.n;
}

function updateHud() {
  const s = S.status, t = S.telemetry || {};
  if (!s) return;
  $("#hud-cosmos").textContent = `${s.cosmos} · ${s.timeline}`;
  const w = s.worlds ? s.worlds[S.world] : null;
  $("#hud-clock").textContent = w
    ? `cycle ${fmt(s.utick)} · ${w.planet} · day ${fmt(w.day)} · year ${fmt(w.year)} · ${fmt(w.population)} organisms`
    : `cycle ${fmt(s.utick)}`;
  const pill = $("#hud-backend");
  pill.textContent = `${s.profile} · ${s.device} · ${fmt(t.tps, 1)} cycles/s${t.halted ? " · HALTED" : ""}`;
  pill.className = `pill ${t.halted ? "bad" : (t.errors || []).length ? "warn" : "ok"}`;
}

// ------------------------------------------------------------------ panels & inspector
function ctx() {
  return {
    world: S.world, timeline: S.timeline, lastVisit: S.lastVisit, status: S.status, role: S.role, speciesColor,
    setSpecies, following: S.following, bridgeOn: !!S.bridge, toggleBridge,
    onFork: async (fid) => { await refreshTimelines(); toast(`<div class="k">FORK</div>${esc(fid)} created — choose it in the timeline menu`); },
  };
}

const PANELS = { history: P.renderHistory, science: P.renderScience, life: P.renderLife, tech: P.renderTech,
  status: P.renderStatus, admin: P.renderAdmin };

async function togglePanel(name) {
  const panel = $("#panel");
  if (S.panel === name && !panel.classList.contains("hidden")) {
    panel.classList.add("hidden");
    S.panel = null;
  } else {
    S.panel = name;
    panel.classList.remove("hidden");
    $("#panel-body").innerHTML = '<div class="note">loading…</div>';
    await refreshPanel();
  }
  $$("#modes [data-panel]").forEach((b) => b.classList.toggle("active", b.dataset.panel === S.panel));
}

async function refreshPanel() {
  const name = S.panel;
  if (!name) return;
  const body = $("#panel-body");
  try {
    const html = await PANELS[name](ctx());
    if (S.panel !== name) return;
    const top = $("#panel").scrollTop;
    body.innerHTML = html;
    $("#panel").scrollTop = top;
  } catch (e) {
    body.innerHTML = `<h2>${esc(name)}</h2><div class="note">could not load: ${esc(e.message || e)}</div>`;
  }
}

setInterval(() => {
  if (!S.panel || S.panel === "admin" || $("#panel").classList.contains("hidden")) return;
  if (document.activeElement && $("#panel").contains(document.activeElement)) return;
  refreshPanel();
}, 6000);

function inspect(html) {
  $("#insp-body").innerHTML = html;
  $("#inspector").classList.remove("hidden");
}

async function selectAgent(id, quiet = false) {
  S.selected = id;
  if (S.mode === "planet" && S.view) S.view.setSelected(id);
  try {
    const { html, d } = await P.inspectAgent(ctx(), id);
    if (S.selected !== id) return;
    const top = $("#inspector").scrollTop;
    inspect(html);
    $("#inspector").scrollTop = top;
    if (d && d.alive) {
      drawMind(d);
      mountBody();
      if (!quiet || mini.bodyId !== id) loadBodyVoxels(id);
    }
  } catch (e) {
    if (!quiet) errorToast(e);
  }
}

setInterval(() => {
  if (S.selected !== null && !$("#inspector").classList.contains("hidden") && $("#insp-body [data-follow]")) {
    selectAgent(S.selected, true);
  }
}, 2500);

async function inspectEvent(id) {
  try {
    inspect(await P.inspectEvent(ctx(), id));
  } catch (e) {
    errorToast(e);
  }
}

// --- body (mini 3D): the organism's real grown lattice
const NODE_COLORS = [[0.75, 0.75, 0.78], [0.92, 0.3, 0.3], [1, 0.9, 0.3], [0.4, 0.9, 0.45], [0.72, 0.5, 1],
  [1, 0.6, 0.25], [0.3, 0.85, 1], [1, 0.5, 0.8], [0.6, 0.42, 0.25], [0.95, 0.95, 0.9], [0.3, 0.7, 0.65]];
const mini = { canvas: document.createElement("canvas"), renderer: null, scene: new THREE.Scene(),
  camera: new THREE.PerspectiveCamera(40, 1, 0.1, 100), mesh: null, bodyId: null };
mini.canvas.className = "mini3d";
mini.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const miniLight = new THREE.DirectionalLight(0xffffff, 1.6);
miniLight.position.set(3, 5, 4);
mini.scene.add(miniLight);

function mountBody() {
  const slot = $("#body3d");
  if (slot && slot !== mini.canvas) slot.replaceWith(mini.canvas);
  if (!mini.renderer) {
    mini.renderer = new THREE.WebGLRenderer({ canvas: mini.canvas, antialias: true, alpha: true });
    mini.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
  }
}

async function loadBodyVoxels(id) {
  try {
    const b = await P.loadBody(ctx(), id);
    mini.bodyId = id;
    if (mini.mesh) {
      mini.scene.remove(mini.mesh);
      mini.mesh.geometry.dispose();
      mini.mesh.material.dispose();
    }
    const n = b.voxels.length;
    const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(0.92, 0.92, 0.92), new THREE.MeshStandardMaterial({ roughness: 0.6 }), n);
    const m = new THREE.Matrix4(), c = new THREE.Color();
    b.voxels.forEach((v, k) => {
      // lattice (x, y, z) with z up in the organism's frame → scene y up
      m.makeTranslation(v[0], v[2], -v[1]);
      mesh.setMatrixAt(k, m);
      const props = v.slice(3);
      let best = 0;
      props.forEach((p, i) => { if (p > props[best]) best = i; });
      const col = NODE_COLORS[best] || [1, 1, 1];
      const s = 0.45 + 0.55 * props[best];
      mesh.setColorAt(k, c.setRGB(col[0] * s, col[1] * s, col[2] * s));
    });
    mesh.instanceMatrix.needsUpdate = true;
    mini.mesh = mesh;
    mini.scene.add(mesh);
    const D = b.size;
    mini.camera.position.set(D * 1.6, D * 1.2, D * 1.9);
    mini.camera.lookAt(0, 0, 0);
  } catch (e) { /* the organism may have died meanwhile */ }
}

function renderMini(dt) {
  if (!mini.renderer || !mini.mesh || !mini.canvas.isConnected || $("#inspector").classList.contains("hidden")) return;
  const w = mini.canvas.clientWidth, h = mini.canvas.clientHeight;
  if (!w || !h) return;
  if (mini.canvas.width !== Math.round(w * mini.renderer.getPixelRatio())) {
    mini.renderer.setSize(w, h, false);
    mini.camera.aspect = w / h;
    mini.camera.updateProjectionMatrix();
  }
  mini.mesh.rotation.y += dt * 0.5;
  mini.renderer.render(mini.scene, mini.camera);
}

function drawMind(d) {
  const c = $("#mind");
  if (!c) return;
  const w = (c.width = c.clientWidth * 2), h = (c.height = c.clientHeight * 2);
  const g = c.getContext("2d");
  const x = d.substrate.state, out = Object.values(d.substrate.outputs);
  const units = d.substrate.units;
  const bw = w / Math.max(1, x.length);
  g.clearRect(0, 0, w, h);
  g.fillStyle = "rgba(140,160,255,0.15)";
  g.fillRect(0, h * 0.35, w, 1);
  x.forEach((v, i) => {
    g.fillStyle = i >= units ? "rgba(90,100,130,0.25)" : v >= 0 ? "#7fd8ff" : "#b38cff";
    const hh = Math.abs(v) * h * 0.33;
    g.fillRect(i * bw + 1, v >= 0 ? h * 0.35 - hh : h * 0.35, bw - 2, hh);
  });
  const ow = w / Math.max(1, out.length);
  out.forEach((v, i) => {
    const a = 1 / (1 + Math.exp(-v));
    g.fillStyle = `rgba(125,255,176,${0.25 + 0.75 * a})`;
    g.fillRect(i * ow + 1, h * 0.75, ow - 2, h * 0.22 * a);
  });
  g.fillStyle = "rgba(138,147,184,0.9)";
  g.font = `${Math.round(h * 0.09)}px ui-monospace, monospace`;
  g.fillText("substrate state", 4, h * 0.1);
  g.fillText("effector drives", 4, h * 0.72);
}

// ------------------------------------------------------------------ clicks
function onPanelClick(e) {
  const t = e.target.closest("[data-event],[data-discovery],[data-agent],[data-follow],[data-world],[data-promote],[data-goto],[data-replay],[data-admin],[data-species]");
  if (!t) return;
  const d = t.dataset;
  if (d.event) inspectEvent(Number(d.event));
  else if (d.discovery) P.inspectDiscovery(ctx(), Number(d.discovery)).then(inspect).catch(errorToast);
  else if (d.agent) {
    if (S.mode !== "planet") setMode("planet", { world: S.world }).then(() => selectAgent(Number(d.agent)));
    else selectAgent(Number(d.agent));
  } else if (d.follow) {
    const id = Number(d.follow);
    S.following = S.following === id ? null : id;
    selectAgent(id, true);
  } else if (d.world) setMode("planet", { world: Number(d.world) });
  else if (d.promote) {
    post("/api/admin/promote", { planet_row: Number(d.promote) }).then(async (r) => {
      toast(`<div class="k">FIDELITY PROMOTION · EXTERNAL</div>${esc(r.world.planet)} now simulated with individuals`);
      S.galaxy = await bin("/api/galaxy");
    }).catch(errorToast);
  } else if (d.goto) {
    const [w, lat, lon] = d.goto.split(":").map(Number);
    setMode("planet", { world: w, lat, lon });
  } else if (d.replay) replayEvent(Number(d.replay));
  else if (d.admin) P.adminAction(ctx(), d.admin).then(() => { if (d.admin !== "bridge") refreshTimelines(); });
  else if (d.species) inspectSpecies(Number(d.species));
}
$("#panel-body").addEventListener("click", onPanelClick);
$("#insp-body").addEventListener("click", onPanelClick);

async function inspectSpecies(sid) {
  try {
    const table = await api(`/api/world/${S.world}/species`, { timeline: S.timeline });
    const s = table.find((x) => x.id === sid);
    if (!s) return;
    const traits = Object.entries(s.traits || {}).map(([k, v]) => `<div class="row"><span>${esc(k)}</span><span>${fmt(v, 3)}</span></div>`).join("");
    inspect(`<h2>SP-${S.world}-${sid}</h2>
      <div class="row"><span>parent</span><span>${s.parent ? `<a data-species="${s.parent}">SP-${S.world}-${s.parent}</a>` : "founders"}</span></div>
      <div class="row"><span>population / peak</span><span>${fmt(s.pop)} / ${fmt(s.peak)}</span></div>
      <div class="row"><span>born / extinct</span><span>${fmt(s.birth)} / ${s.extinct === null ? "—" : fmt(s.extinct)}</span></div>
      <h3>Mean body traits</h3>${traits || '<div class="note">not measured yet</div>'}
      <p class="note">A species is a cluster of genomes found by the observer (INFERRED), not a declared type.</p>`);
  } catch (e) {
    errorToast(e);
  }
}

let down = null;
canvas.addEventListener("pointerdown", (e) => { down = { x: e.clientX, y: e.clientY, t: performance.now() }; });
canvas.addEventListener("pointerup", (e) => {
  if (!down || !S.entered || S.switching) return;
  const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
  const quick = performance.now() - down.t < 450;
  down = null;
  if (moved > 6 || !quick || !S.view) return;
  const ndc = new THREE.Vector2((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  raycaster.setFromCamera(ndc, camera);
  if (S.mode === "galaxy") {
    const hit = S.view.pick(e.clientX, e.clientY, camera, innerWidth, innerHeight);
    if (hit) setMode("system", { star: hit.star });
  } else if (S.mode === "system") {
    const hit = S.view.pick(raycaster);
    if (hit && hit.kind === "planet") inspect(P.inspectPlanet(hit.planet, S.role === "admin"));
    else if (hit) inspect(P.inspectStar(S.view.sys));
  } else if (S.mode === "planet") {
    const hit = S.view.pick(raycaster);
    if (hit && hit.kind === "agent") selectAgent(hit.id);
    else if (hit) inspect(P.inspectCell(hit.info));
  }
});

// ------------------------------------------------------------------ labels
const labelPool = [];
const _v = new THREE.Vector3();
function updateLabels() {
  const targets = S.entered && S.view && S.mode !== "planet" && !renderer.xr.isPresenting ? S.view.labelTargets() : [];
  const box = $("#labels");
  while (labelPool.length < targets.length) {
    const d = document.createElement("div");
    d.className = "label";
    d.addEventListener("click", () => {
      const t = d._target;
      if (!t) return;
      if (t.star !== undefined) setMode("system", { star: t.star });
      else if (t.planet) {
        if (t.planet.f4_world !== null && t.planet.f4_world !== undefined) setMode("planet", { world: t.planet.f4_world });
        else inspect(P.inspectPlanet(t.planet, S.role === "admin"));
      }
    });
    box.appendChild(d);
    labelPool.push(d);
  }
  labelPool.forEach((d, i) => {
    const t = targets[i];
    if (!t) { d.style.display = "none"; return; }
    _v.copy(t.pos).applyMatrix4(root.matrixWorld).project(camera);
    if (_v.z > 1 || Math.abs(_v.x) > 1.1 || Math.abs(_v.y) > 1.1) { d.style.display = "none"; return; }
    d.style.display = "";
    d._target = t;
    if (d.textContent !== t.text) d.textContent = t.text;
    d.className = `label ${t.cls || ""}`;
    d.style.left = `${(_v.x * 0.5 + 0.5) * innerWidth}px`;
    d.style.top = `${(-_v.y * 0.5 + 0.5) * innerHeight}px`;
  });
}

// ------------------------------------------------------------------ replay of a frozen window
async function replayEvent(id) {
  try {
    const msg = await P.loadFrames(ctx(), id);
    const world = msg.meta.world;
    if (S.mode !== "planet" || S.world !== world) await setMode("planet", { world });
    const frames = msg.meta.counts.map((n, k) => ({
      id: msg.arrays[`id_${k}`], lat: msg.arrays[`lat_${k}`], lon: msg.arrays[`lon_${k}`],
      species: msg.arrays[`species_${k}`], bits: msg.arrays[`bits_${k}`], emit: msg.arrays[`emit_${k}`],
      tick: msg.meta.ticks[k],
    }));
    if (!frames.length) throw new Error("the recorded window is empty");
    $("#replay-badge").classList.remove("hidden");
    S.view.playFrames(frames, (k, n, tick) => {
      $("#replay-badge span").textContent = `E-${id} · ${k + 1}/${n} · tick ${tick}`;
    });
  } catch (e) {
    errorToast(e);
  }
}
$("#replay-stop").addEventListener("click", () => {
  if (S.view && S.view.stopReplay) S.view.stopReplay();
  $("#replay-badge").classList.add("hidden");
});

// ------------------------------------------------------------------ reality bridge (admin, forks only)
async function toggleBridge(say) {
  if (S.bridge) {
    clearInterval(S.bridge.timer);
    S.bridge.media.getTracks().forEach((t) => t.stop());
    S.bridge.ac.close();
    S.bridge = null;
    say("bridge stopped");
    return;
  }
  if (S.timeline === "PRIME") { say("the reality bridge only exists in a fork"); return; }
  let lat = 0, lon = 0;
  if (S.mode === "planet") [lat, lon] = xyzToLatLon(camera.position.x, camera.position.y, camera.position.z).map((v) => v / DEG);
  const r = await post("/api/admin/bridge", { timeline: S.timeline, world: S.world, field: "light", lat, lon, gain: 1.0 });
  const media = await navigator.mediaDevices.getUserMedia({ audio: true });
  const ac = new (window.AudioContext || window.webkitAudioContext)();
  const an = ac.createAnalyser();
  an.fftSize = 1024;
  ac.createMediaStreamSource(media).connect(an);
  const buf = new Float32Array(an.fftSize);
  const timer = setInterval(() => {
    an.getFloatTimeDomainData(buf);
    let s = 0;
    for (const v of buf) s += v * v;
    const rms = Math.sqrt(s / buf.length);
    stream.send({ type: "bridge", timeline: S.timeline, world: S.world, source: r.source, value: Math.min(5, rms * 20) });
  }, 100);
  S.bridge = { media, ac, timer };
  say(`bridge source ${r.source} at ${fmt(lat, 1)}°, ${fmt(lon, 1)}° — streaming loudness (a single number)`);
}

// ------------------------------------------------------------------ WebXR
async function setupXR() {
  if (!navigator.xr) return;
  for (const [mode, id] of [["immersive-vr", "#xr-vr"], ["immersive-ar", "#xr-ar"]]) {
    try {
      if (await navigator.xr.isSessionSupported(mode)) $(id).classList.remove("hidden");
    } catch (e) { /* unsupported */ }
  }
}

async function startXR(mode) {
  try {
    const session = await navigator.xr.requestSession(mode, { optionalFeatures: ["local-floor", "bounded-floor", "hand-tracking"] });
    renderer.xr.setReferenceSpaceType("local-floor");
    await renderer.xr.setSession(session);
    const s = S.mode === "planet" ? 0.35 : S.mode === "system" ? 0.05 : 0.035;
    root.scale.setScalar(s);
    root.position.set(0, 1.3, -1.2);
    if (mode === "immersive-ar") scene.background = null;     // passthrough portal
    session.addEventListener("end", () => {
      root.scale.setScalar(1);
      root.position.set(0, 0, 0);
      scene.background = BACKGROUND;
    });
  } catch (e) {
    errorToast(e);
  }
}
$("#xr-vr").addEventListener("click", () => startXR("immersive-vr"));
$("#xr-ar").addEventListener("click", () => startXR("immersive-ar"));

// ------------------------------------------------------------------ audio
const audio = new Sonifier();
$("#audio").addEventListener("click", () => {
  if (audio.on) audio.stop();
  else audio.start();
  $("#audio").classList.toggle("ok", audio.on);
});

// ------------------------------------------------------------------ HUD wiring
$$("#modes [data-mode]").forEach((b) => b.addEventListener("click", () => {
  const m = b.dataset.mode;
  if (m === "system") setMode("system", { star: S.mode === "system" && S.star !== null ? S.star : focusStarOfWorld(S.world) });
  else setMode(m, { world: S.world });
}));
$$("#modes [data-panel]").forEach((b) => b.addEventListener("click", () => togglePanel(b.dataset.panel)));
$$("[data-close]").forEach((b) => b.addEventListener("click", () => {
  $(`#${b.dataset.close}`).classList.add("hidden");
  if (b.dataset.close === "panel") {
    S.panel = null;
    $$("#modes [data-panel]").forEach((x) => x.classList.remove("active"));
  } else {
    S.selected = null;
    S.following = null;
    if (S.view && S.view.setSelected) S.view.setSelected(null);
  }
}));
$("#timeline").addEventListener("change", (e) => switchTimeline(e.target.value));
$("#overlay").addEventListener("change", (e) => {
  S.overlay = Number(e.target.value);
  if (S.mode === "planet" && S.view) S.view.setOverlay(S.overlay);
});
addEventListener("keydown", (e) => {
  if (e.target.closest && e.target.closest("input, textarea, select")) return;
  if (e.key === "Escape") $$("aside").forEach((a) => a.classList.add("hidden"));
  else if (e.key === "g") setMode("galaxy");
  else if (e.key === "s") setMode("system", { star: focusStarOfWorld(S.world) });
  else if (e.key === "p") setMode("planet", { world: S.world });
  else if (e.key === " ") controls.autoRotate = !controls.autoRotate;
});
addEventListener("beforeunload", () => {
  try { if (S.status) localStorage.setItem(`genesis.seen.${S.cosmos}`, String(S.status.utick)); } catch (e) { /* ignore */ }
});

const timelineBar = new TimelineBar($("#tl-canvas"), $("#tl-tip"), (ev) => inspectEvent(ev.id));

// ------------------------------------------------------------------ frame loop
let last = performance.now();
const _t = new THREE.Vector3();
renderer.setAnimationLoop(() => {
  const now = performance.now();
  const dt = Math.min(0.1, (now - last) / 1000);
  last = now;
  if (!renderer.xr.isPresenting) controls.update();
  else root.rotation.y += dt * 0.05;
  if (S.view) S.view.update(dt, camera, renderer, now);
  if (S.following !== null && S.mode === "planet" && S.view) {
    const p = S.view.agentPosition(S.following);
    if (p) {
      controls.target.lerp(p, 0.12);
      _t.copy(p).multiplyScalar(1 + 0.4 / p.length());
      camera.position.lerp(_t, 0.04);
    } else if (!S.view.replay) {
      S.following = null;                     // the organism is gone (died, or left the streamed set)
      controls.target.set(0, 0, 0);
    }
  }
  renderer.render(scene, camera);
  updateLabels();
  renderMini(dt);
});

// ------------------------------------------------------------------ boot
async function boot() {
  if (!hasToken()) {
    statusLine("This observatory needs its access link (…/?token=…). Open the link printed by the launcher.");
    return;
  }
  for (;;) {
    try {
      S.hello = await api("/api/hello");
      break;
    } catch (e) {
      if (e.status === 401) {
        statusLine("The access token was not accepted. Ask the operator for a new link.");
        return;
      }
      statusLine("waiting for the cosmos to answer…");
      await sleep(2000);
    }
  }
  S.role = S.hello.role;
  S.cosmos = S.hello.cosmos;
  statusLine(`${S.hello.cosmos} · cycle ${fmt(S.hello.utick)} · ${S.hello.worlds} world(s) with individuals · ${S.role === "admin" ? "admin access" : "observer access"}`);
  try {
    S.galaxy = await bin("/api/galaxy");
    S.view = new GalaxyView(S.galaxy);
    S.mode = "galaxy";
    root.add(S.view.group);
    setClip("galaxy");
    S.view.cameraHome(camera, controls);
  } catch (e) {
    statusLine(`could not load the galaxy: ${e.message}`);
    return;
  }
  const btn = $("#enter");
  btn.disabled = false;
  btn.addEventListener("click", enter, { once: true });
  document.body.dataset.ready = "1";
}

async function enter() {
  S.entered = true;
  S.enterUtick = S.hello.utick;
  fade(true);
  await sleep(300);
  $("#start").classList.add("hidden");
  $("#hud").classList.remove("hidden");
  $("#timeline-bar").classList.remove("hidden");
  if (S.role === "admin") $("#admin-btn").classList.remove("hidden");
  $$("#modes [data-mode]").forEach((b) => b.classList.toggle("active", b.dataset.mode === S.mode));
  fade(false);
  try { S.lastVisit = Number(localStorage.getItem(`genesis.seen.${S.cosmos}`) || 0); } catch (e) { S.lastVisit = 0; }
  stream.connect();
  refreshTimelines();
  setInterval(refreshTimelines, 10000);
  setupXR();
  try {
    S.events = await api("/api/events", { limit: 2000 });
    timelineBar.set(S.events, S.hello.utick);
  } catch (e) { /* the stream will fill it */ }
  if (S.lastVisit > 0 && S.lastVisit < S.hello.utick) {
    try {
      const n = await api("/api/news", { since: S.lastVisit });
      const total = Object.values(n.counts).reduce((a, b) => a + b, 0);
      toast(`<div class="k">WHILE YOU WERE AWAY</div>${fmt(n.elapsed)} cycles passed · ${fmt(total)} events · ${n.highlights.length} notable — open History`,
        "", () => togglePanel("history"), 15000);
    } catch (e) { /* ignore */ }
  } else {
    toast('<div class="k">WELCOME</div>◉ marks worlds simulated with individual organisms. Click a star, or press <b>Planet</b> to descend.', "", null, 12000);
  }
  document.body.dataset.entered = "1";
}

// console / headless-test hooks (same page, same token: no extra capability)
window.genesisDebug = { camera, controls, setMode, selectAgent, inspectEvent, replayEvent, togglePanel, switchTimeline };

boot();
