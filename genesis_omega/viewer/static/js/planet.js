// Planet view (F4 world). Every layer is simulation state:
//   relief + ocean mask (terrain), surface colour from temperature / primary production / moisture /
//   trail compaction (fields), sea ice where water is below freezing, clouds = humidity field,
//   day/night from the subsolar point, night-side glow = measured organism occupancy,
//   organisms (positions, species, energy, emission, actions) and structures built by them.

import * as THREE from "three";
import { DEG, angularDistance, clamp, idColor, latLonToXYZ, ramp, xyzToLatLon } from "./util.js";

const EXAG = 0.012;           // scene units per km of relief (planet radius = 1)
const AGENT_LIFT = 0.006;
export const OVERLAYS = ["natural", "temperature", "primary production", "moisture", "occupancy", "structures",
  "trails", "shelter", "marks"];

const AGENT_VERT = /* glsl */ `
  attribute vec3 color;
  attribute float size;
  attribute float glow;
  uniform float uScale;
  varying vec3 vColor;
  varying float vGlow;
  void main() {
    vColor = color;
    vGlow = glow;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = clamp(size * uScale / max(-mv.z, 1e-3), 3.5, 30.0);
    gl_Position = projectionMatrix * mv;
  }`;
const AGENT_FRAG = /* glsl */ `
  varying vec3 vColor;
  varying float vGlow;
  void main() {
    vec2 d = gl_PointCoord - 0.5;
    float r = length(d) * 2.0;
    float core = 1.0 - smoothstep(0.42, 0.58, r);
    float halo = vGlow * exp(-r * r * 2.5) * 0.85;
    float a = max(core, halo);
    if (a < 0.04) discard;
    vec3 c = mix(vColor, vec3(1.0), clamp(vGlow * 0.5 * (1.0 - core), 0.0, 1.0));
    gl_FragColor = vec4(c, a);
  }`;
const ATMO_VERT = /* glsl */ `
  varying float vRim;
  varying vec3 vN;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vec3 n = normalize(normalMatrix * normal);
    vRim = 1.0 - max(dot(n, normalize(-mv.xyz)), 0.0);
    vN = normalize((modelMatrix * vec4(normal, 0.0)).xyz);
    gl_Position = projectionMatrix * mv;
  }`;
const ATMO_FRAG = /* glsl */ `
  uniform vec3 uColor;
  uniform float uStrength;
  uniform vec3 uSun;
  varying float vRim;
  varying vec3 vN;
  void main() {
    float day = smoothstep(-0.3, 0.4, dot(vN, uSun));
    float a = pow(vRim, 2.6) * uStrength * (0.18 + 0.82 * day);
    gl_FragColor = vec4(uColor * a, a);
  }`;

function dataTexture(W, H, data) {
  const t = new THREE.DataTexture(data, W, H, THREE.RGBAFormat, THREE.UnsignedByteType);
  t.wrapS = THREE.RepeatWrapping;
  t.wrapT = THREE.ClampToEdgeWrapping;
  t.magFilter = THREE.LinearFilter;
  t.minFilter = THREE.LinearFilter;
  t.generateMipmaps = false;
  t.colorSpace = THREE.SRGBColorSpace;
  t.needsUpdate = true;
  return t;
}

/** Lat/lon grid mesh whose vertices sit on cell corners; uv (j/W, i/H) maps rows north → south. */
function gridGeometry(W, H, radiusAt) {
  const nv = (W + 1) * (H + 1);
  const pos = new Float32Array(nv * 3), uv = new Float32Array(nv * 2), p = [0, 0, 0];
  for (let i = 0; i <= H; i++) {
    const lat = Math.PI / 2 - (i * Math.PI) / H;
    for (let j = 0; j <= W; j++) {
      const lon = -Math.PI + (j * 2 * Math.PI) / W;
      const v = i * (W + 1) + j;
      latLonToXYZ(lat, lon, radiusAt(i, j), p);
      pos.set(p, 3 * v);
      uv[2 * v] = j / W;
      uv[2 * v + 1] = i / H;
    }
  }
  const idx = new Uint32Array(W * H * 6);
  let k = 0;
  for (let i = 0; i < H; i++) {
    for (let j = 0; j < W; j++) {
      const a = i * (W + 1) + j, b = a + 1, c = a + W + 1, d = c + 1;
      idx[k++] = a; idx[k++] = c; idx[k++] = b;
      idx[k++] = b; idx[k++] = c; idx[k++] = d;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  g.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  g.computeVertexNormals();
  return g;
}

export class PlanetView {
  constructor(terrain, opts = {}) {
    const { meta, arrays } = terrain;
    this.meta = meta;
    this.world = meta.world;
    this.W = meta.w;
    this.H = meta.h;
    this.xenos = meta.xenos || {};
    this.group = new THREE.Group();
    this.overlay = opts.overlay || 0;
    this.speciesColor = opts.speciesColor || ((s) => idColor(s));
    const W = this.W, H = this.H;
    this.elevKm = new Float32Array(W * H);
    for (let k = 0; k < W * H; k++) this.elevKm[k] = arrays.elev[k] / 1000;
    this.ocean = Uint8Array.from(arrays.ocean);
    this.cellR = new Float32Array(W * H);
    for (let k = 0; k < W * H; k++) this.cellR[k] = this._radiusFromElev(this.elevKm[k]);

    // ---- surface
    const radiusAt = (i, j) => {
      let s = 0, n = 0;
      for (const di of [-1, 0]) {
        const ii = Math.min(H - 1, Math.max(0, i + di));
        for (const dj of [-1, 0]) {
          const jj = (j + dj + W) % W;
          s += this.elevKm[ii * W + jj];
          n++;
        }
      }
      return this._radiusFromElev(s / n);
    };
    this.colorData = new Uint8Array(W * H * 4);
    this.glowData = new Uint8Array(W * H * 4);
    this.oceanData = new Uint8Array(W * H * 4);
    this.cloudData = new Uint8Array(W * H * 4);
    this.colorTex = dataTexture(W, H, this.colorData);
    this.glowTex = dataTexture(W, H, this.glowData);
    this.oceanTex = dataTexture(W, H, this.oceanData);
    this.cloudTex = dataTexture(W, H, this.cloudData);
    this.surface = new THREE.Mesh(gridGeometry(W, H, radiusAt), new THREE.MeshStandardMaterial({
      map: this.colorTex, emissiveMap: this.glowTex, emissive: new THREE.Color(1.0, 0.72, 0.42),
      emissiveIntensity: 0.9, roughness: 0.93, metalness: 0.0,
    }));
    this.oceanMesh = new THREE.Mesh(gridGeometry(W, H, () => 1.0), new THREE.MeshStandardMaterial({
      map: this.oceanTex, transparent: true, roughness: 0.42, metalness: 0.02, depthWrite: false,
    }));
    this.oceanMesh.renderOrder = 1;
    this.clouds = new THREE.Mesh(gridGeometry(W, H, () => 1.032), new THREE.MeshStandardMaterial({
      map: this.cloudTex, transparent: true, depthWrite: false, roughness: 1.0, metalness: 0.0,
    }));
    this.clouds.renderOrder = 3;
    const sc = this.xenos.star_color || [1, 1, 1];
    const sky = new THREE.Color(0.35, 0.6, 1.0).lerp(new THREE.Color(sc[0], sc[1], sc[2]), 0.35);
    const pressure = Math.max(this.xenos.pressure || 1, 0.001);
    this.atmo = new THREE.Mesh(new THREE.SphereGeometry(1.075, 96, 48), new THREE.ShaderMaterial({
      uniforms: { uColor: { value: sky }, uStrength: { value: clamp(0.55 + 0.35 * Math.log10(pressure), 0.05, 1.6) },
        uSun: { value: new THREE.Vector3(1, 0, 0) } },
      vertexShader: ATMO_VERT, fragmentShader: ATMO_FRAG, transparent: true, depthWrite: false,
      blending: THREE.AdditiveBlending,
    }));
    this.atmo.renderOrder = 4;
    this.sun = new THREE.DirectionalLight(new THREE.Color(sc[0], sc[1], sc[2]), 2.4);
    this.ambient = new THREE.AmbientLight(0x8090b0, 0.07);
    this.group.add(this.surface, this.oceanMesh, this.clouds, this.atmo, this.sun, this.sun.target, this.ambient);
    this.sunDir = new THREE.Vector3(1, 0, 0);

    // ---- organisms
    this.cap = 0;
    this.n = 0;
    this.agentGeom = new THREE.BufferGeometry();
    this.agentMat = new THREE.ShaderMaterial({
      uniforms: { uScale: { value: 600 } }, vertexShader: AGENT_VERT, fragmentShader: AGENT_FRAG,
      transparent: true, depthWrite: false,
    });
    this.agents = new THREE.Points(this.agentGeom, this.agentMat);
    this.agents.frustumCulled = false;
    this.agents.renderOrder = 2;
    this.group.add(this.agents);
    this._ensure(512);
    this.frame = null;            // last live frame (decoded arrays + meta)
    this.prevPos = new Map();     // id → displayed xyz at frame switch
    this.tFrame = 0;
    this.interval = 100;
    this.selected = null;
    this.selMarker = new THREE.Mesh(new THREE.RingGeometry(0.011, 0.016, 32),
      new THREE.MeshBasicMaterial({ color: 0x7fd8ff, side: THREE.DoubleSide, transparent: true, depthTest: false }));
    this.selMarker.visible = false;
    this.selMarker.renderOrder = 5;
    this.group.add(this.selMarker);
    this.replay = null;

    // ---- structures
    const box = new THREE.BoxGeometry(1, 1, 1);
    box.translate(0, 0.5, 0);
    this.structCap = 0;
    this.structMat = new THREE.MeshStandardMaterial({ roughness: 0.8, metalness: 0.05 });
    this.structGeom = box;
    this.structMesh = null;
    this.structures = new Map();  // cell → {h, mat}
    this.structDirty = false;
    this.lastStructBuild = 0;

    this.fields = null;
    this._recolor();
  }

  _radiusFromElev(e) {
    return 1 + (e > 0 ? e * EXAG : e * EXAG * 0.25);
  }

  // ------------------------------------------------------------- camera
  cameraHome(camera, controls) {
    camera.position.set(0.6, 0.9, 3.1);
    controls.target.set(0, 0, 0);
    controls.minDistance = 1.08;
    controls.maxDistance = 14;
    controls.autoRotate = false;
  }

  // ------------------------------------------------------------- fields
  setFields(arrays) {
    this.fields = arrays;
    this._recolor();
  }

  setOverlay(k) {
    this.overlay = k | 0;
    this._recolor();
  }

  setSubsolar(latDeg, lonDeg) {
    const p = [0, 0, 0];
    latLonToXYZ(latDeg * DEG, lonDeg * DEG, 1, p);
    this.sunDir.set(p[0], p[1], p[2]);
    this.sun.position.set(p[0] * 10, p[1] * 10, p[2] * 10);
    this.atmo.material.uniforms.uSun.value.copy(this.sunDir);
  }

  _recolor() {
    const W = this.W, H = this.H, f = this.fields;
    const C = this.colorData, G = this.glowData, O = this.oceanData, L = this.cloudData;
    const mode = this.overlay;
    for (let k = 0; k < W * H; k++) {
      const e = this.elevKm[k], sea = this.ocean[k] === 1;
      const tC = f ? (f.temp[k] / 255) * 90 - 40 : 10;
      const res = f ? (f.resource[k] / 255) * 1.2 : 0.4;
      const moist = f ? f.moisture[k] / 255 : 0.4;
      const comp = f ? f.compaction[k] / 255 : 0;
      let c;
      if (mode === 0) {
        if (sea) {
          const depth = clamp(-e / 4.5, 0, 1);
          c = [0.05 + 0.08 * (1 - depth), 0.12 + 0.2 * (1 - depth), 0.2 + 0.18 * (1 - depth)];
        } else {
          const rock = [0.42, 0.39, 0.35], desert = [0.66, 0.56, 0.38], soil = [0.4, 0.33, 0.22];
          const veg = [0.13, 0.33 + 0.1 * moist, 0.1];
          let base = e > 3.2 ? rock : moist < 0.18 ? desert : soil;
          const v = clamp(res, 0, 1) * (e > 3.6 ? 0.3 : 1);
          c = base.map((x, i) => x * (1 - v) + veg[i] * v);
          if (comp > 0.05) c = c.map((x) => x * (1 - 0.5 * comp) + 0.62 * 0.5 * comp);
          const snow = clamp((-2 - tC) / 6, 0, 1);
          if (snow > 0) c = c.map((x) => x * (1 - snow) + 0.94 * snow);
        }
      } else {
        c = this._overlayColor(mode, k, sea);
      }
      C[4 * k] = c[0] * 255; C[4 * k + 1] = c[1] * 255; C[4 * k + 2] = c[2] * 255; C[4 * k + 3] = 255;
      // ocean layer (transparent over land); sea ice when water is below freezing
      if (sea) {
        if (mode === 0 && tC < -1.5) {
          O[4 * k] = 232; O[4 * k + 1] = 240; O[4 * k + 2] = 248; O[4 * k + 3] = 245;
        } else if (mode === 0) {
          const prod = clamp(res, 0, 1);
          O[4 * k] = 10 + 14 * prod; O[4 * k + 1] = 52 + 40 * prod; O[4 * k + 2] = 96 + 10 * prod; O[4 * k + 3] = 205;
        } else {
          O[4 * k] = c[0] * 255; O[4 * k + 1] = c[1] * 255; O[4 * k + 2] = c[2] * 255; O[4 * k + 3] = 235;
        }
      } else {
        O[4 * k + 3] = 0;
      }
      // night-side glow: organism occupancy (measured), only in the natural view
      const occ = f && mode === 0 ? clamp(((f.occupancy[k] / 255) * 2) * 1.6, 0, 1) : 0;
      G[4 * k] = G[4 * k + 1] = G[4 * k + 2] = occ * 255;
      G[4 * k + 3] = 255;
      // clouds: the humidity field
      const cl = f ? f.cloud[k] / 255 : 0;
      const a = mode === 0 ? clamp((cl - 0.32) / 0.6, 0, 1) * 0.88 : 0;
      L[4 * k] = L[4 * k + 1] = L[4 * k + 2] = 250;
      L[4 * k + 3] = a * 255;
    }
    for (const t of [this.colorTex, this.glowTex, this.oceanTex, this.cloudTex]) t.needsUpdate = true;
  }

  _overlayColor(mode, k, sea) {
    const f = this.fields;
    if (!f) return [0.1, 0.1, 0.12];
    const shade = sea ? 0.8 : 1;
    let c;
    switch (mode) {
      case 1: c = ramp("thermal", f.temp[k] / 255); break;
      case 2: c = ramp("life", f.resource[k] / 255); break;
      case 3: c = ramp("water", f.moisture[k] / 255); break;
      case 4: c = ramp("magma", Math.sqrt(f.occupancy[k] / 255)); break;
      case 5: {
        const h = f.structure[k] / 255;
        c = h > 0.01 ? ramp("magma", 0.35 + 0.65 * Math.sqrt(h)) : [0.08, 0.08, 0.1];
        break;
      }
      case 6: c = ramp("ice", Math.sqrt(f.compaction[k] / 255)); break;
      case 7: c = ramp("ice", f.shelter[k] / 255); break;
      case 8: c = ramp("magma", Math.sqrt(f.marks[k] / 255)); break;
      default: c = [0.5, 0.5, 0.5];
    }
    return c.map((x) => x * shade);
  }

  cellInfo(cell) {
    const f = this.fields, W = this.W;
    const i = Math.floor(cell / W), j = cell % W;
    const out = {
      cell, lat: 90 - ((i + 0.5) * 180) / this.H, lon: -180 + ((j + 0.5) * 360) / W,
      elevation_km: this.elevKm[cell], ocean: this.ocean[cell] === 1,
    };
    if (f) {
      Object.assign(out, {
        temperature_c: (f.temp[cell] / 255) * 90 - 40, primary_production: (f.resource[cell] / 255) * 1.2,
        moisture: f.moisture[cell] / 255, cloud: f.cloud[cell] / 255, compaction: f.compaction[cell] / 255,
        structure_height: (f.structure[cell] / 255) * 4, occupancy: (f.occupancy[cell] / 255) * 2,
        shelter: f.shelter[cell] / 255, marks: (f.marks[cell] / 255) * 2, light: (f.light[cell] / 255) * 1.3,
      });
    }
    const s = this.structures.get(cell);
    if (s) out.structure = s;
    return out;
  }

  cellOf(lat, lon) {
    const i = clamp(Math.floor(((Math.PI / 2 - lat) / Math.PI) * this.H), 0, this.H - 1);
    const j = ((Math.floor(((lon + Math.PI) / (2 * Math.PI)) * this.W) % this.W) + this.W) % this.W;
    return i * this.W + j;
  }

  // ------------------------------------------------------------- organisms
  _ensure(n) {
    if (n <= this.cap) return;
    const cap = Math.max(n, Math.ceil(this.cap * 1.5), 512);
    this.pos = new Float32Array(cap * 3);
    this.col = new Float32Array(cap * 3);
    this.size = new Float32Array(cap);
    this.glow = new Float32Array(cap);
    this.from = new Float32Array(cap * 3);
    this.to = new Float32Array(cap * 3);
    this.agentGeom.setAttribute("position", new THREE.BufferAttribute(this.pos, 3).setUsage(THREE.DynamicDrawUsage));
    this.agentGeom.setAttribute("color", new THREE.BufferAttribute(this.col, 3).setUsage(THREE.DynamicDrawUsage));
    this.agentGeom.setAttribute("size", new THREE.BufferAttribute(this.size, 1).setUsage(THREE.DynamicDrawUsage));
    this.agentGeom.setAttribute("glow", new THREE.BufferAttribute(this.glow, 1).setUsage(THREE.DynamicDrawUsage));
    this.cap = cap;
  }

  /** Live frame from the stream: decoded arrays (quantised lat/lon, species, bits, energy, mass, emit). */
  setAgents(arrays, meta, now) {
    if (this.replay) {
      this.pendingLive = { arrays, meta };
      return;
    }
    this._applyFrame(arrays, meta, now, true);
  }

  _applyFrame(a, meta, now, interpolate) {
    const n = a.id.length;
    // remember where each organism is currently drawn so motion is continuous
    const shown = new Map();
    if (interpolate && this.ids) {
      for (let k = 0; k < this.n; k++) shown.set(this.ids[k], k);
    }
    const prevPos = this.pos ? this.pos.slice(0, 3 * this.n) : new Float32Array(0);
    this._ensure(n);
    const p = [0, 0, 0];
    const latQ = Math.PI / 2 / 32767, lonQ = Math.PI / 32767;
    for (let k = 0; k < n; k++) {
      const lat = a.lat[k] * latQ, lon = a.lon[k] * lonQ;
      const cell = this.cellOf(lat, lon);
      const r = (this.ocean[cell] ? 1.0 : this.cellR[cell]) + AGENT_LIFT;
      latLonToXYZ(lat, lon, r, p);
      this.to.set(p, 3 * k);
      const old = shown.get(a.id[k]);
      if (old !== undefined) this.from.set(prevPos.subarray(3 * old, 3 * old + 3), 3 * k);
      else this.from.set(p, 3 * k);
      const sc = this.speciesColor(a.species[k]);
      const energy = a.energy ? a.energy[k] / 255 : 1;
      const br = 0.5 + 0.5 * energy;
      const bits = a.bits ? a.bits[k] : 0;
      const striking = (bits >> 4) & 1;
      this.col[3 * k] = striking ? 1.0 : sc[0] * br;
      this.col[3 * k + 1] = striking ? 0.35 : sc[1] * br;
      this.col[3 * k + 2] = striking ? 0.3 : sc[2] * br;
      this.size[k] = 0.012 + 0.016 * (a.mass ? a.mass[k] / 255 : 0.3);
      this.glow[k] = Math.max(a.emit ? a.emit[k] / 255 : 0, striking ? 0.6 : 0);
    }
    // copy from→pos for k (the interpolation start)
    this.pos.set(this.from.subarray(0, 3 * n));
    this.ids = Uint32Array.from(a.id);
    this.speciesOf = Uint32Array.from(a.species);
    this.bits = a.bits ? Uint16Array.from(a.bits) : null;
    this.n = n;
    this.frame = { meta, t: now };
    if (this.tFrame) this.interval = clamp(0.8 * this.interval + 0.2 * (now - this.tFrame), 40, 1500);
    this.tFrame = now;
    this.agentGeom.setDrawRange(0, n);
    for (const name of ["color", "size", "glow"]) this.agentGeom.attributes[name].needsUpdate = true;
  }

  agentIndex(id) {
    if (!this.ids) return -1;
    for (let k = 0; k < this.n; k++) if (this.ids[k] === id) return k;
    return -1;
  }

  agentPosition(id) {
    const k = this.agentIndex(id);
    return k < 0 ? null : new THREE.Vector3(this.pos[3 * k], this.pos[3 * k + 1], this.pos[3 * k + 2]);
  }

  setSelected(id) {
    this.selected = id;
  }

  // ------------------------------------------------------------- structures
  setStructures(arrays) {
    this.structures.clear();
    this.applyStructDelta(arrays);
  }

  applyStructDelta(a) {
    for (let k = 0; k < a.cell.length; k++) {
      const h = a.h[k];
      if (h > 0.05) this.structures.set(a.cell[k], { h, mat: a.mat[k] });
      else this.structures.delete(a.cell[k]);
    }
    this.structDirty = true;
  }

  _buildStructures() {
    const n = this.structures.size;
    if (!this.structMesh || n > this.structCap) {
      if (this.structMesh) {
        this.group.remove(this.structMesh);
        this.structMesh.dispose();
      }
      this.structCap = Math.max(256, Math.ceil(n * 1.5));
      this.structMesh = new THREE.InstancedMesh(this.structGeom, this.structMat, this.structCap);
      this.structMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      this.group.add(this.structMesh);
    }
    const m = new THREE.Matrix4(), up = new THREE.Vector3(), east = new THREE.Vector3(), south = new THREE.Vector3();
    const color = new THREE.Color();
    const W = this.W, H = this.H, p = [0, 0, 0];
    let k = 0;
    for (const [cell, s] of this.structures) {
      const i = Math.floor(cell / W), j = cell % W;
      const lat = Math.PI / 2 - ((i + 0.5) * Math.PI) / H, lon = -Math.PI + ((j + 0.5) * 2 * Math.PI) / W;
      latLonToXYZ(lat, lon, 1, p);
      up.set(p[0], p[1], p[2]);
      east.set(-Math.sin(lon), 0, -Math.cos(lon));
      south.crossVectors(east, up);
      const r = this.ocean[cell] ? 1.0 : this.cellR[cell];
      const wE = ((2 * Math.PI) / W) * Math.cos(lat) * 0.9, wN = (Math.PI / H) * 0.9;
      const h = Math.min(s.h, 6) * 0.006;
      m.makeBasis(east.clone().multiplyScalar(wE), up.clone().multiplyScalar(h), south.clone().multiplyScalar(wN));
      m.setPosition(up.x * r, up.y * r, up.z * r);
      this.structMesh.setMatrixAt(k, m);
      const base = s.mat >= 0 ? idColor(s.mat) : [0.5, 0.5, 0.5];
      const wall = s.h >= 1 ? 0.12 : 0;            // walls (block movement) read slightly lighter
      color.setRGB(base[0] * 0.35 + 0.22 + wall, base[1] * 0.3 + 0.18 + wall, base[2] * 0.25 + 0.14 + wall);
      this.structMesh.setColorAt(k, color);
      k++;
    }
    this.structMesh.count = k;
    this.structMesh.instanceMatrix.needsUpdate = true;
    if (this.structMesh.instanceColor) this.structMesh.instanceColor.needsUpdate = true;
    this.structDirty = false;
  }

  // ------------------------------------------------------------- replay of frozen windows
  playFrames(frames, onStep) {
    this.replay = { frames, k: 0, t: 0, onStep };
  }

  stopReplay() {
    this.replay = null;
    if (this.pendingLive) {
      this._applyFrame(this.pendingLive.arrays, this.pendingLive.meta, performance.now(), false);
      this.pendingLive = null;
    }
  }

  // ------------------------------------------------------------- per frame
  update(dt, camera, renderer, now) {
    this.agentMat.uniforms.uScale.value = renderer.domElement.clientHeight / (2 * Math.tan((camera.fov * Math.PI) / 360));
    if (this.replay) {
      const r = this.replay;
      r.t += dt;
      if (r.t >= 0.16 || r.k === 0) {
        r.t = 0;
        const f = r.frames[r.k % r.frames.length];
        this._applyFrame(f, { tick: f.tick }, now, false);
        r.onStep && r.onStep(r.k % r.frames.length, r.frames.length, f.tick);
        r.k++;
      }
    }
    const alpha = this.replay ? 1 : clamp((now - this.tFrame) / this.interval, 0, 1);
    for (let k = 0; k < this.n; k++) {
      const i = 3 * k;
      const x = this.from[i] + (this.to[i] - this.from[i]) * alpha;
      const y = this.from[i + 1] + (this.to[i + 1] - this.from[i + 1]) * alpha;
      const z = this.from[i + 2] + (this.to[i + 2] - this.from[i + 2]) * alpha;
      const rT = Math.hypot(this.to[i], this.to[i + 1], this.to[i + 2]);
      const s = rT / (Math.hypot(x, y, z) || 1);
      this.pos[i] = x * s; this.pos[i + 1] = y * s; this.pos[i + 2] = z * s;
    }
    this.agentGeom.attributes.position.needsUpdate = true;
    if (this.selected !== null) {
      const p = this.agentPosition(this.selected);
      this.selMarker.visible = !!p;
      if (p) {
        this.selMarker.position.copy(p);
        this.selMarker.lookAt(p.clone().multiplyScalar(2));
        const d = camera.position.distanceTo(p);
        this.selMarker.scale.setScalar(clamp(d * 0.55, 0.25, 3));
      }
    } else {
      this.selMarker.visible = false;
    }
    if (this.structDirty && now - this.lastStructBuild > 400) {
      this._buildStructures();
      this.lastStructBuild = now;
    }
  }

  /** Click → nearest organism (within a few cells) or the cell's measured state. */
  pick(raycaster) {
    const hit = raycaster.intersectObject(this.surface, false)[0];
    if (!hit) return null;
    const [lat, lon] = xyzToLatLon(hit.point.x, hit.point.y, hit.point.z);
    let best = -1, bestD = (3.0 * 2 * Math.PI) / this.W;
    for (let k = 0; k < this.n; k++) {
      const [la, lo] = xyzToLatLon(this.pos[3 * k], this.pos[3 * k + 1], this.pos[3 * k + 2]);
      const d = angularDistance(lat, lon, la, lo);
      if (d < bestD) { bestD = d; best = k; }
    }
    if (best >= 0) return { kind: "agent", id: this.ids[best], species: this.speciesOf[best] };
    return { kind: "cell", info: this.cellInfo(this.cellOf(lat, lon)), lat, lon };
  }

  lookAtLatLon(latDeg, lonDeg, camera, controls, dist = 1.9) {
    const p = [0, 0, 0];
    latLonToXYZ(latDeg * DEG, lonDeg * DEG, dist, p);
    camera.position.set(p[0], p[1], p[2]);
    controls.target.set(0, 0, 0);
  }

  dispose() {
    this.group.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose();
    });
    for (const t of [this.colorTex, this.glowTex, this.oceanTex, this.cloudTex]) t.dispose();
    if (this.structMesh) this.structMesh.dispose();
  }
}
