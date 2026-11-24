// Star-system view: the star, its planets on their Keplerian orbits (elements from the simulation,
// positions at the galaxy's current time), the habitable zone, the belt radius, and the real sky:
// every other star of the galaxy as seen from this system.

import * as THREE from "three";
import { blackbody, orbitPosition } from "./util.js";
import { galaxyToScene, starMaterial } from "./galaxy.js";

/** Visual radial scale: √AU keeps inner and outer orbits readable (stated in the HUD). */
export const auToScene = (a) => 4.0 * Math.sqrt(Math.max(a, 0));

function planetColor(p) {
  if (p.kind === 1) return new THREE.Color(0.78, 0.62, 0.42);   // gas giant
  if (p.kind === 2) return new THREE.Color(0.55, 0.78, 0.88);   // ice giant
  const t = p.t_mean, w = p.water;
  if (t > 100) return new THREE.Color(0.62, 0.32, 0.18);
  if (t < -40) return new THREE.Color(0.85, 0.88, 0.92);
  const ocean = new THREE.Color(0.12, 0.3, 0.55), land = new THREE.Color(0.45, 0.4, 0.3);
  if (p.life) land.lerp(new THREE.Color(0.2, 0.45, 0.2), 0.6);
  return land.lerp(ocean, Math.min(1, w));
}

export class SystemView {
  constructor(sys, galaxy, galaxyYears) {
    this.sys = sys;
    this.group = new THREE.Group();
    this.time = galaxyYears || 0;
    this.pickables = [];
    this.planets = [];
    const sc = blackbody(sys.teff > 0 ? sys.teff : 3000);
    const starColor = new THREE.Color(sc[0], sc[1], sc[2]);
    const inner = sys.planets.length ? auToScene(sys.planets[0].a * (1 - sys.planets[0].e)) : 3;
    const starR = Math.min(0.15 + 0.12 * Math.log10(1 + Math.max(sys.radius, 0.01) * 10), Math.max(0.12, inner * 0.35));
    const isBH = sys.stage === "black hole";
    const star = new THREE.Mesh(new THREE.SphereGeometry(starR, 48, 32),
      new THREE.MeshBasicMaterial({ color: isBH ? 0x000000 : starColor }));
    star.userData = { kind: "sun" };
    this.group.add(star);
    this.pickables.push(star);
    if (!isBH) {
      const glow = new THREE.Sprite(new THREE.SpriteMaterial({
        map: glowTexture(), color: starColor, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      }));
      glow.scale.setScalar(starR * 9);
      this.group.add(glow);
    }
    this.light = new THREE.PointLight(starColor, isBH ? 0 : 3.0, 0, 0);
    this.group.add(this.light, new THREE.AmbientLight(0xffffff, 0.07));

    // habitable zone (conservative, from luminosity)
    if (sys.hz_au && sys.hz_au[1] > 0 && !isBH) {
      const hz = new THREE.Mesh(new THREE.RingGeometry(auToScene(sys.hz_au[0]), auToScene(sys.hz_au[1]), 128, 1),
        new THREE.MeshBasicMaterial({ color: 0x3dff8a, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false }));
      hz.rotation.x = -Math.PI / 2;
      this.group.add(hz);
    }
    if (sys.belt_au > 0) this.group.add(circle(auToScene(sys.belt_au), 0x9a8f80, 0.45, true));

    for (const p of sys.planets) {
      const orbit = this._orbitLine(p);
      this.group.add(orbit);
      const r = p.kind === 0 ? 0.07 + 0.05 * Math.log2(1 + p.radius) : 0.12 + 0.07 * Math.log2(1 + p.radius / 4);
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(r, 32, 24),
        new THREE.MeshStandardMaterial({ color: planetColor(p), roughness: 0.85, metalness: 0.0 }));
      mesh.userData = { kind: "planet", planet: p };
      if (p.rings) {
        const ring = new THREE.Mesh(new THREE.RingGeometry(r * 1.4, r * 2.2, 48, 1),
          new THREE.MeshBasicMaterial({ color: 0xc8b89a, transparent: true, opacity: 0.45, side: THREE.DoubleSide }));
        ring.rotation.x = -Math.PI / 2 + (p.tilt * Math.PI) / 180;
        mesh.add(ring);
      }
      if (p.f4_world !== null && p.f4_world !== undefined) {
        const halo = new THREE.Mesh(new THREE.RingGeometry(r * 1.7, r * 1.95, 48, 1),
          new THREE.MeshBasicMaterial({ color: 0x7fd8ff, transparent: true, opacity: 0.9, side: THREE.DoubleSide }));
        halo.userData.billboard = true;
        mesh.add(halo);
      }
      this.group.add(mesh);
      this.pickables.push(mesh);
      this.planets.push({ p, mesh });
    }
    if (galaxy) this.group.add(this._sky(galaxy));
    this.update(0);
  }

  _orbitLine(p) {
    const pts = [];
    for (let k = 0; k <= 160; k++) {
      const t = (k / 160) * p.period;
      const [x, y, z] = orbitPosition(Object.assign({}, p, { m0: 0 }), t);
      pts.push(this._toScene(x, y, z));
    }
    const g = new THREE.BufferGeometry().setFromPoints(pts);
    const color = p.f4_world !== null && p.f4_world !== undefined ? 0x7fd8ff : p.life ? 0x5fd08a : 0x4a5378;
    return new THREE.Line(g, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.55 }));
  }

  _toScene(x, y, z) {
    const r = Math.hypot(x, y);
    const s = r > 0 ? auToScene(r) / r : 0;
    return new THREE.Vector3(x * s, z * s, -y * s);
  }

  _sky(galaxy) {
    const { pos, teff, lum, stage } = galaxy.arrays;
    const n = galaxy.meta.n;
    const me = [0, 0, 0], o = [0, 0, 0];
    galaxyToScene(pos, this.sys.index, me);
    const xyz = [], col = [], size = [];
    for (let i = 0; i < n; i++) {
      if (i === this.sys.index || stage[i] === 4) continue;
      galaxyToScene(pos, i, o);
      const dx = o[0] - me[0], dy = o[1] - me[1], dz = o[2] - me[2];
      const d = Math.hypot(dx, dy, dz) || 1e-3;
      const flux = lum[i] / (d * d);                    // apparent brightness from here
      const b = Math.max(0, Math.min(1, 0.25 + 0.12 * Math.log10(flux + 1e-6) + 0.5));
      if (b <= 0.05) continue;
      xyz.push((dx / d) * 400, (dy / d) * 400, (dz / d) * 400);
      const c = blackbody(teff[i]);
      col.push(c[0] * b, c[1] * b, c[2] * b);
      size.push(1.0 + 2.5 * b);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(xyz, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    g.setAttribute("size", new THREE.Float32BufferAttribute(size, 1));
    const sky = new THREE.Points(g, starMaterial(400));
    sky.material.uniforms.uScale.value = 400;
    this.sky = sky;
    return sky;
  }

  cameraHome(camera, controls) {
    const outer = this.sys.planets.length ? auToScene(this.sys.planets[this.sys.planets.length - 1].a * 1.3) : 6;
    camera.position.set(0, outer * 0.55 + 1.5, outer * 0.95 + 2);
    controls.target.set(0, 0, 0);
    controls.minDistance = 0.4;
    controls.maxDistance = Math.max(40, outer * 4);
    controls.autoRotate = false;
  }

  setTime(years) {
    if (Number.isFinite(years)) this.time = years;
  }

  update(dt, camera) {
    for (const { p, mesh } of this.planets) {
      const [x, y, z] = orbitPosition(p, this.time);
      mesh.position.copy(this._toScene(x, y, z));
      mesh.rotation.y += dt * 0.25 / Math.max(0.3, p.day_factor);
      if (camera) mesh.children.forEach((c) => { if (c.userData.billboard) c.quaternion.copy(camera.quaternion); });
    }
    if (camera && this.sky) this.sky.position.copy(camera.position);
  }

  pick(raycaster) {
    const hit = raycaster.intersectObjects(this.pickables, false)[0];
    if (!hit) return null;
    const u = hit.object.userData;
    return u.kind === "planet" ? { kind: "planet", planet: u.planet } : { kind: "sun", star: this.sys };
  }

  planetPosition(row) {
    const e = this.planets.find((x) => x.p.row === row);
    return e ? e.mesh.position.clone() : null;
  }

  labelTargets() {
    return this.planets.map(({ p, mesh }) => ({
      pos: mesh.position, text: `${p.id}${p.f4_world !== null && p.f4_world !== undefined ? " · F4" : ""}`,
      cls: p.f4_world !== null && p.f4_world !== undefined ? "f4" : p.life ? "life" : "dim", planet: p,
    }));
  }

  dispose() {
    this.group.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose();
    });
  }
}

function circle(radius, color, opacity, dashed) {
  const pts = [];
  for (let k = 0; k <= 180; k++) {
    const a = (k / 180) * Math.PI * 2;
    pts.push(new THREE.Vector3(Math.cos(a) * radius, 0, Math.sin(a) * radius));
  }
  const g = new THREE.BufferGeometry().setFromPoints(pts);
  const m = dashed
    ? new THREE.LineDashedMaterial({ color, transparent: true, opacity, dashSize: 0.08, gapSize: 0.08 })
    : new THREE.LineBasicMaterial({ color, transparent: true, opacity });
  const line = new THREE.Line(g, m);
  if (dashed) line.computeLineDistances();
  return line;
}

let _glow = null;
export function glowTexture() {
  if (_glow) return _glow;
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  const grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grd.addColorStop(0, "rgba(255,255,255,1)");
  grd.addColorStop(0.15, "rgba(255,255,255,0.6)");
  grd.addColorStop(0.4, "rgba(255,255,255,0.12)");
  grd.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grd;
  g.fillRect(0, 0, 128, 128);
  _glow = new THREE.CanvasTexture(c);
  return _glow;
}
