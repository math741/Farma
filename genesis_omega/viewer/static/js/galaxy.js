// Galaxy view: every point is a star of the simulated galaxy (position, temperature, luminosity, stage).
// Green halos: F0 aggregate biospheres. Cyan rings: worlds simulated with individuals (F4).

import * as THREE from "three";
import { blackbody } from "./util.js";

const STAR_VERT = /* glsl */ `
  attribute float size;
  attribute vec3 color;
  uniform float uScale;
  varying vec3 vColor;
  void main() {
    vColor = color;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = clamp(size * uScale / max(-mv.z, 1e-3), 2.0, 56.0);
    gl_Position = projectionMatrix * mv;
  }`;
const STAR_FRAG = /* glsl */ `
  varying vec3 vColor;
  void main() {
    vec2 d = gl_PointCoord - 0.5;
    float r2 = dot(d, d) * 4.0;
    float a = exp(-r2 * 3.5) + 0.35 * exp(-r2 * 18.0);
    if (a < 0.015) discard;
    gl_FragColor = vec4(vColor * a, a);
  }`;
const RING_FRAG = /* glsl */ `
  varying vec3 vColor;
  uniform float uTime;
  void main() {
    vec2 d = gl_PointCoord - 0.5;
    float r = length(d) * 2.0;
    float ring = smoothstep(0.62, 0.78, r) * (1.0 - smoothstep(0.82, 0.98, r));
    float pulse = 0.7 + 0.3 * sin(uTime * 2.2);
    if (ring < 0.02) discard;
    gl_FragColor = vec4(vColor * ring * pulse, ring * pulse);
  }`;

export function starMaterial(scale, frag = STAR_FRAG) {
  return new THREE.ShaderMaterial({
    uniforms: { uScale: { value: scale }, uTime: { value: 0 } },
    vertexShader: STAR_VERT, fragmentShader: frag,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  });
}

/** Map simulation (x, y, z) in kpc to the scene (disk in the XZ plane, y up). */
export function galaxyToScene(pos, i, out) {
  out[0] = pos[3 * i];
  out[1] = pos[3 * i + 2];
  out[2] = -pos[3 * i + 1];
  return out;
}

export class GalaxyView {
  constructor(data) {
    this.data = data;
    this.group = new THREE.Group();
    this.labels = [];
    this.time = 0;
    const { pos, teff, lum, stage, life } = data.arrays;
    const n = data.meta.n;
    const xyz = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    const size = new Float32Array(n);
    const tmp = [0, 0, 0];
    for (let i = 0; i < n; i++) {
      galaxyToScene(pos, i, tmp);
      xyz.set(tmp, 3 * i);
      let c;
      if (stage[i] === 4) c = [0.35, 0.2, 0.55];                     // black hole remnant
      else c = blackbody(teff[i]);
      const l = Math.log10(Math.max(lum[i], 1e-5));
      const bright = 0.55 + 0.12 * Math.max(-3, Math.min(4, l));
      col.set([c[0] * bright, c[1] * bright, c[2] * bright], 3 * i);
      size[i] = stage[i] === 4 ? 0.12 : 0.14 + 0.07 * Math.max(0, Math.min(8, l + 4));
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(xyz, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    g.setAttribute("size", new THREE.BufferAttribute(size, 1));
    this.stars = new THREE.Points(g, starMaterial(600));
    this.group.add(this.stars);
    this.xyz = xyz;
    // frame the camera on where the stars actually are (90th percentile of disk radius)
    const radii = [];
    for (let i = 0; i < n; i++) radii.push(Math.hypot(xyz[3 * i], xyz[3 * i + 2]));
    radii.sort((a, b) => a - b);
    this.extent = Math.max(3, radii[Math.floor(0.9 * (n - 1))] || 8);

    // life markers (real flags from the backend)
    const lifeIdx = [], f4Idx = [];
    for (let i = 0; i < n; i++) {
      if (life[i] === 1) lifeIdx.push(i);
      else if (life[i] === 2) f4Idx.push(i);
    }
    this.lifeMarkers = this._markers(lifeIdx, [0.25, 0.9, 0.5], 0.55, STAR_FRAG);
    this.f4Markers = this._markers(f4Idx, [0.5, 0.85, 1.0], 1.4, RING_FRAG);
    this.group.add(this.lifeMarkers, this.f4Markers);
    this.f4Idx = f4Idx;

    // central supermassive black hole + accretion glow
    const bh = new THREE.Mesh(new THREE.SphereGeometry(0.12, 24, 16), new THREE.MeshBasicMaterial({ color: 0x000000 }));
    const disk = new THREE.Mesh(new THREE.RingGeometry(0.16, 0.7, 64, 1), new THREE.ShaderMaterial({
      uniforms: {}, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
      vertexShader: "varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }",
      fragmentShader: "varying vec2 vUv; void main(){ float r = length(vUv - 0.5) * 2.0; float a = smoothstep(1.0, 0.25, r) * smoothstep(0.18, 0.3, r); gl_FragColor = vec4(vec3(1.0, 0.62, 0.3) * a, a * 0.8); }",
    }));
    disk.rotation.x = -Math.PI / 2;
    this.group.add(bh, disk);
    this.smbh = bh;
  }

  _markers(idx, color, size, frag) {
    const g = new THREE.BufferGeometry();
    const p = new Float32Array(idx.length * 3);
    const c = new Float32Array(idx.length * 3);
    const s = new Float32Array(idx.length);
    idx.forEach((i, k) => {
      p.set(this.xyz.subarray(3 * i, 3 * i + 3), 3 * k);
      c.set(color, 3 * k);
      s[k] = size;
    });
    g.setAttribute("position", new THREE.BufferAttribute(p, 3));
    g.setAttribute("color", new THREE.BufferAttribute(c, 3));
    g.setAttribute("size", new THREE.BufferAttribute(s, 1));
    return new THREE.Points(g, starMaterial(600, frag));
  }

  cameraHome(camera, controls) {
    const e = this.extent;
    camera.position.set(0, e * 1.25, e * 1.85);
    controls.target.set(0, 0, 0);
    controls.minDistance = 1.5;
    controls.maxDistance = 80;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.25;
  }

  focusStar(i, camera, controls) {
    const t = this.xyz.subarray(3 * i, 3 * i + 3);
    controls.target.set(t[0], t[1], t[2]);
    camera.position.set(t[0] + 2.5, t[1] + 1.8, t[2] + 3.2);
  }

  starPosition(i) {
    return new THREE.Vector3(this.xyz[3 * i], this.xyz[3 * i + 1], this.xyz[3 * i + 2]);
  }

  update(dt, camera, renderer) {
    this.time += dt;
    const scale = renderer.domElement.clientHeight / (2 * Math.tan((camera.fov * Math.PI) / 360));
    for (const m of [this.stars.material, this.lifeMarkers.material, this.f4Markers.material]) {
      m.uniforms.uScale.value = scale;
      m.uniforms.uTime.value = this.time;
    }
  }

  /** Nearest star to a screen point (within a few pixels), preferring F4 / life-bearing systems. */
  pick(px, py, camera, width, height) {
    const v = new THREE.Vector3();
    let best = -1, bestD = 14 * 14;
    const { life } = this.data.arrays;
    for (let i = 0; i < this.data.meta.n; i++) {
      v.set(this.xyz[3 * i], this.xyz[3 * i + 1], this.xyz[3 * i + 2]).project(camera);
      if (v.z > 1) continue;
      const sx = (v.x * 0.5 + 0.5) * width, sy = (-v.y * 0.5 + 0.5) * height;
      const d = (sx - px) ** 2 + (sy - py) ** 2 - (life[i] ? 60 : 0);
      if (d < bestD) { bestD = d; best = i; }
    }
    return best >= 0 ? { kind: "star", star: best } : null;
  }

  labelTargets() {
    return this.f4Idx.map((i) => ({ pos: this.starPosition(i), text: `S-${String(i).padStart(5, "0")}`, cls: "f4", star: i }));
  }

  dispose() {
    this.group.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose();
    });
  }
}
