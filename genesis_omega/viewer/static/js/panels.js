// Side panels and the inspector. Every number shown comes from the backend; every claim carries its
// epistemic tag (what kind of statement) and, for science, its validation status (how much evidence).

import { api, post, bin } from "./api.js";
import { $, bar, css, esc, fmt, idColor, pct, row, tagEpi, tagStatus, tagVal } from "./util.js";

const SIG = ["routine", "notable", "historic", "extraordinary"];

export function eventItem(e) {
  return `<div class="list-item" data-event="${e.id}">
    <div>${esc(e.title)}</div>
    <div class="meta">${esc(e.label || "E-" + e.id)} · cycle ${fmt(e.utick)}${e.tick !== null && e.tick !== undefined ? " · tick " + fmt(e.tick) : ""}
      · ${SIG[e.significance] || ""} ${tagEpi(e.epistemic)}${e.validation && e.validation !== "NOT VALIDATED" ? tagVal(e.validation) : ""}</div>
  </div>`;
}

// ================================================================= history
export async function renderHistory(ctx) {
  const q = { timeline: ctx.timeline };
  const [events, pantheon, news] = await Promise.all([
    api("/api/events", { ...q, min_sig: 1, limit: 200 }), api("/api/pantheon", q),
    api("/api/news", { ...q, since: ctx.lastVisit || 0 }),
  ]);
  const firsts = pantheon.filter((p) => p.event);
  return `<h2>History · ${esc(ctx.timeline)}</h2>
    <p class="note">No narrative is generated: events are derived from simulation state by detectors, with
      candidate causes (temporal association only).</p>
    <h3>Since your last visit</h3>
    ${row("cycles elapsed", fmt(news.elapsed))}
    ${Object.entries(news.counts).sort((a, b) => b[1] - a[1]).slice(0, 8).map(([k, v]) => row(k, fmt(v))).join("") || '<div class="note">nothing yet</div>'}
    ${news.extraordinary.map(eventItem).join("")}
    <h3>Pantheon (firsts, never deleted)</h3>
    ${firsts.length ? firsts.map((p) => `<div class="list-item" data-event="${p.event}"><div>${esc(p.key.replace(/_/g, " "))}</div>
        <div class="meta">${esc(p.label)} · ${esc(p.title)} ${tagEpi(p.epistemic)}</div></div>`).join("")
      : '<div class="note">no firsts recorded yet</div>'}
    <h3>Notable events (latest first)</h3>
    ${events.slice().reverse().map(eventItem).join("") || '<div class="note">none yet</div>'}`;
}

// ================================================================= science
export async function renderScience(ctx) {
  const q = { timeline: ctx.timeline };
  const [d, levels] = await Promise.all([api("/api/discoveries", q), api(`/api/world/${ctx.world}/levels`, q)]);
  const ctrl = d.controls || {};
  const pipes = Object.entries(ctrl).filter(([, v]) => v && typeof v === "object" && "valid" in v);
  const lvl = (rows) => (rows || []).map((r) => `<div class="row"><span>${esc(r.level)}</span><span>${statusTag(r.status)}</span></div>`).join("");
  return `<h2>Science</h2>
    <p class="note">The Scientist tries to break every finding first. A claim climbs CANDIDATE → TESTED →
      REPLICATED → CAUSALLY SUPPORTED only by surviving controls, new seeds and ablation. Fitness is never evidence.</p>
    <h3>Pipeline controls (synthetic positive / negative organisms)</h3>
    ${pipes.length ? pipes.map(([k, v]) => row(k, v.valid ? '<span class="tag s-TESTED">VALID</span>' : '<span class="tag s-NOT">INVALID</span>')).join("")
      : '<div class="note">controls not run in this timeline yet</div>'}
    <h3>Discoveries (${d.discoveries.length}) · research queue ${fmt(d.queue)}</h3>
    ${d.discoveries.slice().reverse().map((x) => `<div class="list-item" data-discovery="${x.id}">
        <div>${esc(x.title)}</div>
        <div class="meta">${esc(x.label)} · world ${x.world} · tick ${fmt(x.tick)} ${tagVal(x.status)} ${tagEpi(x.epistemic)}</div>
        <div class="meta">${esc(x.mechanism)}</div></div>`).join("") || '<div class="note">no discovery has survived a probe yet</div>'}
    <h3>Levels (evidence assessments, world ${ctx.world})</h3>
    ${lvl(levels.organism)}
    <h3>Functional affect</h3>${lvl(levels.affect)}
    <h3>Civilization</h3>${lvl(levels.civilization)}
    <h3>Science of the organisms</h3>${lvl(levels.science)}
    <h3>Observer detection</h3>${lvl(levels.observer)}
    <p class="note">${esc(levels.note || "")}</p>
    <h3>Negative results (archived, never discarded)</h3>
    ${(d.negative || []).slice(-12).reverse().map((r) => row(`${r.kind} · tick ${fmt(r.tick)}`, "negative")).join("") || '<div class="note">none</div>'}
    ${(d.failures || []).length ? `<h3>Research failures</h3>${d.failures.slice(-5).map((f) => `<div class="mono">${esc(f.task)}: ${esc(f.error)}</div>`).join("")}` : ""}`;
}

function statusTag(s) {
  if (s === "NOT OBSERVED") return '<span class="tag v-NOT">NOT OBSERVED</span>';
  if (s === "NOT IMPLEMENTED") return '<span class="tag s-NOT">NOT IMPLEMENTED</span>';
  return tagVal(s);
}

// ================================================================= life
export async function renderLife(ctx) {
  const q = { timeline: ctx.timeline };
  const [species, qd, social] = await Promise.all([
    api(`/api/world/${ctx.world}/species`, q), api(`/api/world/${ctx.world}/qd`, q), api(`/api/world/${ctx.world}/social`, q),
  ]);
  ctx.setSpecies && ctx.setSpecies(species);
  const byParent = new Map();
  for (const s of species) {
    const p = s.parent || 0;
    if (!byParent.has(p)) byParent.set(p, []);
    byParent.get(p).push(s);
  }
  const lines = [];
  const walk = (pid, depth) => {
    for (const s of (byParent.get(pid) || []).sort((a, b) => a.birth - b.birth)) {
      if (lines.length > 400) return;
      lines.push(`<div class="node ${s.extinct !== null ? "extinct" : ""}" data-species="${s.id}" style="padding-left:${depth * 14}px">
        <span class="dot" style="background:${css(s.color)}"></span>SP-${ctx.world}-${s.id}
        <span class="meta">pop ${fmt(s.pop)} · peak ${fmt(s.peak)} · born ${fmt(s.birth)}${s.extinct !== null ? " · extinct " + fmt(s.extinct) : ""}</span></div>`);
      walk(s.id, depth + 1);
    }
  };
  walk(0, 0);
  const alive = species.filter((s) => s.extinct === null).length;
  return `<h2>Life · world ${ctx.world}</h2>
    <p class="note">Species are discovered by genetic clustering (observer-side, ${tagEpi("INFERRED")}); none is declared.</p>
    ${row("species alive / ever", `${fmt(alive)} / ${fmt(species.length)}`)}
    <h3>Phylogeny</h3><div class="tree">${lines.join("") || '<div class="note">no species yet</div>'}</div>
    <h3>Ways of life (MAP-Elites niches, coverage ${pct(qd.coverage)})</h3>
    ${qd.elites.slice(0, 24).map((e) => row(`${e.dominant} · aquatic ${e.aquatic_bin} · mass ${e.mass_bin}`,
      `A-${e.agent_id} · fit ${fmt(e.fitness)}`)).join("") || '<div class="note">archive empty</div>'}
    ${row("novelty archive", fmt(qd.novelty_archive))}
    <h3>Social graph (from interactions only)</h3>
    ${row("groups (≥4)", fmt(social.n_groups))}${row("largest group", fmt(social.largest))}
    ${row("group persistence", fmt(social.persistence))}${row("edges", fmt(social.edges))}
    ${Object.entries(social.totals || {}).map(([k, v]) => row(k, fmt(v))).join("")}`;
}

// ================================================================= technology
export async function renderTech(ctx) {
  const q = { timeline: ctx.timeline };
  const [t, cons] = await Promise.all([api(`/api/world/${ctx.world}/technology`, q),
    api(`/api/world/${ctx.world}/construction`, q)]);
  const g = t.graph;
  const nodes = g.nodes;
  const depthOf = new Map(nodes.map((n) => [n.id, n.depth]));
  const cols = new Map();
  for (const n of nodes) {
    if (!cols.has(n.depth)) cols.set(n.depth, []);
    cols.get(n.depth).push(n);
  }
  const Wd = 420, Hd = 360, maxD = Math.max(1, g.max_depth);
  const xy = new Map();
  for (const [d, list] of cols) {
    list.forEach((n, i) => xy.set(n.id, [20 + (d / maxD) * (Wd - 40), 18 + ((i + 0.5) / list.length) * (Hd - 36)]));
  }
  const edges = g.edges.filter((e) => xy.has(e.from) && xy.has(e.to)).map((e) => {
    const [x1, y1] = xy.get(e.from), [x2, y2] = xy.get(e.to);
    return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${e.role === "carried" ? "#7fd8ff" : "#b38cff"}" stroke-opacity=".35"/>`;
  }).join("");
  const circles = nodes.map((n) => {
    const [x, y] = xy.get(n.id);
    const c = css(idColor(n.id));
    const r = n.base ? 6 : 3 + Math.min(6, Math.log2(1 + n.produced));
    return `<circle cx="${x}" cy="${y}" r="${r}" fill="${c}" stroke="${g.used_candidates.includes(n.id) ? "#fff" : "none"}"><title>M-${ctx.world}-${n.id} depth ${n.depth} · made ${n.produced}× by ${n.producers} · built ${n.built_units}</title></circle>`;
  }).join("");
  const c = t.culture.totals;
  return `<h2>Technology · world ${ctx.world}</h2>
    <p class="note">Nodes are materials organisms actually produced through the hidden chemistry; edges are the
      recipes they used (blue: carried, violet: substrate). Depth = length of the production chain.</p>
    ${row("materials", fmt(nodes.length))}${row("products", fmt(g.n_products))}${row("deepest chain", fmt(g.max_depth))}
    ${row("registry saturated", fmt(g.saturated))}
    <svg class="tech" viewBox="0 0 ${Wd} ${Hd}">${edges}${circles}</svg>
    <h3>Technology candidates ${tagEpi("HYPOTHESIS")}</h3>
    ${g.candidates.length ? g.candidates.map((m) => row(`M-${ctx.world}-${m}`, g.used_candidates.includes(m) ? "made by many · built with" : "made by many")).join("") : '<div class="note">none</div>'}
    <h3>Cultural lineage of first productions</h3>
    ${row("after social contact (candidate transmission)", fmt(c.social))}${row("same matriline", fmt(c.kin))}${row("independent invention", fmt(c.independent))}
    <h3>Construction & settlement ${tagEpi("INFERRED")}</h3>
    ${row("structure cells", fmt(cons.structure_cells))}${row("persistent", fmt(cons.persistent_cells))}${row("walls", fmt(cons.wall_cells))}
    ${row("house-like", fmt((cons.house_like || []).length))}${row("storage-like cells", fmt(cons.storage_like_cells))}
    ${row("village-like", fmt((cons.village_like || []).length))}${row("city-like", fmt((cons.city_like || []).length))}
    ${row("trails", fmt((cons.trails || []).length))}${row("ruins", fmt((cons.ruins || []).length))}
    ${(cons.city_criteria || []).map((cr, i) => `<div class="note">village ${i + 1} city criteria: ${Object.entries(cr).map(([k, v]) => `${esc(k)} ${v ? "✓" : "✗"}`).join(" · ")}</div>`).join("")}`;
}

// ================================================================= status
export async function renderStatus(ctx) {
  const [st, features] = await Promise.all([api("/api/status", { timeline: ctx.timeline }), api("/api/features")]);
  const s = st.status, t = st.telemetry || {};
  const mem = t.memory || {};
  const worlds = (s.worlds || []).map((w) => `<h3>World ${w.index} · ${esc(w.planet)} (${esc(w.fidelity)})</h3>
    ${row("tick / day / year", `${fmt(w.tick)} / ${fmt(w.day)} / ${fmt(w.year)}`)}
    ${row("population / slots", `${fmt(w.population)} / ${fmt(w.slots)}`)}${bar(w.population / Math.max(1, w.slots))}
    ${row("species alive", fmt(w.species_alive))}${row("max generation", fmt(w.max_generation))}
    ${row("global temperature °C", fmt(w.global_temp))}${row("materials", fmt(w.materials))}
    ${row("structure cells", fmt(w.structure_cells))}${row("seed bank", fmt(w.seed_bank))}
    ${row("interventions", fmt(w.interventions))}${w.ablations && w.ablations.length ? row("ablations", esc(w.ablations.join(", "))) : ""}
    ${w.observed ? Object.entries(w.observed).map(([k, v]) => row(`observed: ${k}`, fmt(v))).join("") : ""}`).join("");
  const byPhase = new Map();
  for (const f of features) {
    if (!byPhase.has(f.phase)) byPhase.set(f.phase, []);
    byPhase.get(f.phase).push(f);
  }
  return `<h2>System status</h2>
    ${row("cosmos", esc(s.cosmos))}${row("timeline", `${esc(s.timeline)}${s.parent ? " ← " + esc(s.parent) : ""}`)}
    ${row("cycle", fmt(s.utick))}${row("galaxy time (years)", fmt(s.galaxy_years))}
    ${row("F0 life-bearing planets", fmt(s.life_bearing_f0))}${row("events / discoveries / forks", `${fmt(s.events)} / ${fmt(s.discoveries)} / ${fmt(s.forks)}`)}
    ${row("prime experiment lock", s.lock ? "ON" : "OFF")}${row("pipeline controls valid", fmt(s.controls_valid))}
    <h3>Engine room</h3>
    ${row("profile / device", `${esc(s.profile)} / ${esc(s.device)}`)}${row("cycles per second", fmt(t.tps))}
    ${row("target", t.target_tps ? fmt(t.target_tps) : "max")}${row("uptime", `${fmt((t.uptime_s || 0) / 3600, 2)} h`)}
    ${row("science share / queue", `${pct(t.science_share)} / ${fmt(t.science_queue)}`)}${row("lab batch", fmt(t.lab_batch))}
    ${row(`${mem.kind || "memory"} used / free (GB)`, `${fmt(mem.current_gb)} / ${fmt(mem.free_gb)}`)}
    ${row("checkpoints / last age", `${fmt(t.checkpoints)} / ${fmt(t.last_checkpoint_age_s)} s`)}
    ${t.halted ? `<div class="note" style="color:var(--bad)">HALTED: ${esc(t.halted)}</div>` : ""}
    ${(t.errors || []).map((e) => `<div class="mono" style="color:var(--warn)">${esc(e.kind || Object.keys(e)[0])}: ${esc(e.error || Object.values(e)[0])}</div>`).join("")}
    ${worlds}
    <h3>Versions</h3>${Object.entries(s.versions || {}).map(([k, v]) => row(k, esc(v))).join("")}
    <h3>What exists (honest feature registry)</h3>
    ${[...byPhase].map(([ph, list]) => `<div class="note" style="margin-top:10px">${esc(ph)}</div>${list.map((f) => `<div class="row"><span title="${esc(f.notes)}">${esc(f.name)}</span><span>${tagStatus(f.status)}</span></div>`).join("")}`).join("")}`;
}

// ================================================================= admin
export async function renderAdmin(ctx) {
  const forks = await api("/api/forks");
  const st = ctx.status || {};
  return `<h2>Admin</h2>
    <p class="note">Experiments are JSON documents from a closed vocabulary (no code). They always run in a fork;
      every action is an ${tagEpi("EXTERNAL")} event in the timeline where it happens.</p>
    <h3>New fork</h3>
    <input id="fk-label" class="btn" style="width:100%" placeholder="label" value="experiment">
    <textarea id="fk-steps" class="btn mono" style="width:100%;height:110px">[{"op": "perturb", "param": "solar_flux", "value": 1.05},
 {"op": "observe", "metrics": ["population", "mean_energy", "births"]}]</textarea>
    <button class="btn" data-admin="fork">create fork (world ${ctx.world})</button>
    <h3>Intervene in the current timeline (${esc(ctx.timeline)})</h3>
    <textarea id="iv-steps" class="btn mono" style="width:100%;height:70px">[{"op": "ablate", "mechanism": "vibration"}]</textarea>
    <button class="btn danger" data-admin="intervene">apply</button>
    <h3>Timelines</h3>
    ${(forks.forks || []).map((f) => row(`${f.id} · ${f.label}`, `${forks.active.includes(f.id) ? "running" : "stopped"} · from ${f.parent}@${fmt(f.fork_utick)}`)).join("") || '<div class="note">no forks</div>'}
    <h3>Engine</h3>
    <button class="btn" data-admin="lock">${st.lock === false ? "lock" : "unlock"} prime experiments</button>
    <button class="btn" data-admin="checkpoint">checkpoint now</button>
    <button class="btn" data-admin="replay">verify deterministic replay</button>
    <div style="margin-top:6px">speed (cycles/s, 0 = max) <input id="ad-tps" class="btn" style="width:80px" value="0">
      <button class="btn" data-admin="speed">set</button></div>
    <h3>Reality bridge (forks only)</h3>
    <p class="note">Streams one scalar — your microphone's loudness, computed locally — into a marked light
      source of the current fork. No audio leaves this page.</p>
    <button class="btn" data-admin="bridge" ${ctx.timeline === "PRIME" ? "disabled" : ""}>${ctx.bridgeOn ? "stop bridge" : "start bridge"}</button>
    <div id="ad-out" class="mono note"></div>`;
}

export async function adminAction(ctx, action) {
  const out = $("#ad-out");
  const say = (x) => { if (out) out.textContent = typeof x === "string" ? x : JSON.stringify(x, null, 1); };
  try {
    if (action === "fork") {
      const steps = JSON.parse($("#fk-steps").value || "[]");
      const r = await post("/api/admin/fork", { label: $("#fk-label").value, reason: "observatory", steps, world: ctx.world });
      say(`fork ${r.fork} created`);
      ctx.onFork && ctx.onFork(r.fork);
    } else if (action === "intervene") {
      const steps = JSON.parse($("#iv-steps").value || "[]");
      const r = await post("/api/admin/intervene", { timeline: ctx.timeline, experiment: { world: ctx.world, steps } });
      say(`${r.events.length} external event(s) recorded`);
    } else if (action === "lock") {
      const lockNow = !!(ctx.status && ctx.status.lock === false);   // currently unlocked → lock it
      const r = await post("/api/admin/lock", { locked: lockNow });
      say(`prime lock ${r.locked ? "ON" : "OFF"}`);
    } else if (action === "checkpoint") {
      const r = await post("/api/admin/checkpoint", {});
      say(`checkpoint written at cycle ${r.utick}`);
    } else if (action === "replay") {
      say("re-simulating from a keyframe…");
      const r = await post("/api/admin/replay_verify", {});
      say(r.error ? r.error : `${r.identical ? "IDENTICAL" : "DIFFERENT"} · keyframe ${r.keyframe} → ${r.target}`);
    } else if (action === "speed") {
      const r = await post("/api/admin/speed", { tps: Number($("#ad-tps").value) || 0 });
      say(`target ${r.target_tps || "max"}`);
    } else if (action === "bridge") {
      ctx.toggleBridge && (await ctx.toggleBridge(say));
    }
  } catch (e) {
    say(String(e.message || e));
  }
}

// ================================================================= inspector
export async function inspectAgent(ctx, id) {
  const q = { timeline: ctx.timeline };
  const d = await api(`/api/world/${ctx.world}/agent/${id}`, q);
  if (!d.alive) {
    const r = d.record || {};
    return { d, html: `<h2>A-${id}</h2><p class="note">no longer alive</p>
      ${row("species", `SP-${ctx.world}-${r.species}`)}${row("generation", fmt(r.generation))}
      ${row("born / died", `${fmt(r.birth)} / ${fmt(r.death)}`)}${row("cause", esc(r.cause))}${row("offspring", fmt(r.offspring))}` };
  }
  const caps = d.caps, st = d.stats;
  const io = (obj, n) => Object.entries(obj).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, n)
    .map(([k, v]) => row(k, fmt(v, 3))).join("");
  return { d, html: `<h2>A-${d.id} ${tagEpi("SIMULATED")}</h2>
    <div class="row"><span>species</span><span><span class="dot" style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${css(ctx.speciesColor(d.species))}"></span> SP-${ctx.world}-${d.species}</span></div>
    ${row("lineage · generation", `L-${d.lineage} · ${fmt(d.generation)}`)}${row("parent", d.parent >= 0 ? `<a data-agent="${d.parent}">A-${d.parent}</a>` : "founder")}
    ${row("age / lifespan (ticks)", `${fmt(d.age, 0)} / ${fmt(d.lifespan, 0)}`)}
    ${row("energy", `${fmt(d.energy)} / ${fmt(d.energy_cap)}`)}${bar(d.energy / Math.max(d.energy_cap, 1e-6))}
    ${row("integrity", fmt(d.integrity))}${bar(d.integrity)}
    ${row("body °C (optimum ± tolerance)", `${fmt(d.body_temp, 1)} (${fmt(d.t_opt, 1)} ± ${fmt(d.t_tol, 1)})`)}
    ${row("position", `${fmt(d.lat, 1)}°, ${fmt(d.lon, 1)}°`)}
    ${d.carried_material >= 0 ? row("carrying", `M-${ctx.world}-${d.carried_material} × ${fmt(d.carried_amount)}`) : ""}
    <div style="margin:8px 0"><button class="btn" data-follow="${d.id}">${ctx.following === d.id ? "stop following" : "follow"}</button></div>
    <h3>Body (grown from its genome)</h3>
    <canvas class="mini3d" id="body3d"></canvas>
    <div class="note">voxels coloured by dominant node property: rigidity (grey), contractility (red), senses
      (photo yellow · chemo green · mechano violet · thermo orange), conductivity (cyan), storage (brown)</div>
    ${Object.entries(caps).map(([k, v]) => row(k, fmt(v, 3))).join("")}
    <h3>Signal substrate (${fmt(d.substrate.units)} units) — not a "brain"; a measured network</h3>
    <canvas class="mind" id="mind"></canvas>
    ${row("modulatory broadcast", fmt(d.substrate.modulation, 3))}${row("plasticity |ΔW|", fmt(d.substrate.plasticity, 4))}
    <h3>Strongest inputs (raw signals)</h3>${io(d.substrate.inputs, 8)}
    <h3>Outputs (effector drives)</h3>${io(d.substrate.outputs, 8)}
    <h3>Lifetime</h3>${Object.entries(st).filter(([, v]) => v).map(([k, v]) => row(k, fmt(v))).join("")}
    <h3>Ancestry</h3>${d.ancestors.map((a) => a.known === false ? row("…", `A-${a.id} (record rotated out)`) : row(`gen ${a.generation}`, `<a data-agent="${a.id}">A-${a.id}</a> ${a.death >= 0 ? "† " + esc(a.cause) : "alive"}`)).join("") || '<div class="note">founder</div>'}
    ${d.children.length ? row("children", d.children.slice(0, 12).map((c) => `<a data-agent="${c}">A-${c}</a>`).join(" ")) : ""}
    <h3>Social ties (interaction counts)</h3>
    ${d.social.slice(0, 8).map((s) => row(`<a data-agent="${s.id}">A-${s.id}</a>`, `w ${fmt(s.weight)} · give ${s.give} · strike ${s.strike} · mate ${s.mate}`)).join("") || '<div class="note">none recorded</div>'}
    ${d.focus ? '<div class="note">F5 scientific focus individual: full internal state recorded every tick.</div>' : ""}` };
}

export function inspectCell(info) {
  return `<h2>Surface cell ${fmt(info.cell)}</h2>
    ${Object.entries(info).filter(([k]) => k !== "cell" && k !== "structure").map(([k, v]) => row(k.replace(/_/g, " "), fmt(v, 2))).join("")}
    ${info.structure ? row("structure (height · material)", `${fmt(info.structure.h)} · M-${info.structure.mat}`) : ""}
    <p class="note">${tagEpi("SIMULATED")} field values at the last published frame.</p>`;
}

export function inspectStar(sys) {
  return `<h2>${esc(sys.id)} · ${esc(sys.class)}-type ${esc(sys.stage)}</h2>
    ${row("mass (M☉)", fmt(sys.mass, 3))}${row("luminosity (L☉)", fmt(sys.luminosity, 4))}${row("radius (R☉)", fmt(sys.radius, 3))}
    ${row("effective temperature (K)", fmt(sys.teff, 0))}${row("age (Gyr)", fmt(sys.age_gyr, 2))}
    ${row("habitable zone (AU)", `${fmt(sys.hz_au[0], 2)} – ${fmt(sys.hz_au[1], 2)}`)}${row("planets", fmt(sys.n_planets))}
    ${sys.belt_au > 0 ? row("belt (AU)", fmt(sys.belt_au, 2)) : ""}
    <p class="note">${tagEpi("PROGRAMMED")} population relations (no nuclear physics is simulated).</p>`;
}

export function inspectPlanet(p, isAdmin) {
  const f4 = p.f4_world !== null && p.f4_world !== undefined;
  return `<h2>${esc(p.id)} · ${esc(p.kind_name)}</h2>
    ${row("orbit a (AU) · e", `${fmt(p.a, 3)} · ${fmt(p.e, 3)}`)}${row("period (yr)", fmt(p.period, 3))}
    ${row("mass / radius (⊕)", `${fmt(p.mass, 2)} / ${fmt(p.radius, 2)}`)}${row("gravity (g)", fmt(p.gravity, 2))}
    ${row("pressure (bar)", fmt(p.pressure, 3))}${row("mean temperature °C", fmt(p.t_mean, 1))}
    ${row("water fraction", fmt(p.water, 2))}${row("axial tilt °", fmt(p.tilt, 1))}${row("moons · rings", `${p.moons} · ${p.rings ? "yes" : "no"}`)}
    ${row("habitability index", fmt(p.habitability, 3))}${bar(p.habitability)}
    ${p.f0_biomass !== null && p.f0_biomass !== undefined ? row("F0 aggregate biomass", fmt(p.f0_biomass, 3)) : row("life", "none detected")}
    ${f4 ? `<button class="btn" data-world="${p.f4_world}">descend to the surface (F4 world ${p.f4_world})</button>`
      : p.f0_biomass !== null && p.f0_biomass !== undefined && isAdmin
        ? `<button class="btn danger" data-promote="${p.row}">promote F0 → F4 (external act)</button>` : ""}
    <p class="note">${f4 ? "Simulated with individual organisms." : "Only an aggregate population field exists here (F0): no individuals."}</p>`;
}

export async function inspectEvent(ctx, id) {
  const res = await api("/api/search", { q: `E-${id}`, timeline: ctx.timeline });
  const e = res.length ? res[0].event : null;
  if (!e) return `<h2>E-${id}</h2><p class="note">not in the in-memory index</p>`;
  const det = Object.entries(e.detail || {}).filter(([, v]) => typeof v !== "object" || v === null)
    .map(([k, v]) => row(k, esc(fmt(v)))).join("");
  return `<h2>${esc(e.label)}</h2><div>${esc(e.title)}</div>
    <div style="margin:8px 0">${tagEpi(e.epistemic)} ${tagVal(e.validation)}</div>
    ${row("type", esc(e.type))}${row("cycle", fmt(e.utick))}${e.world !== null ? row("world · tick", `${e.world} · ${fmt(e.tick)}`) : ""}
    ${e.lat !== null && e.lat !== undefined ? row("location", `${fmt(e.lat, 1)}°, ${fmt(e.lon, 1)}° <a data-goto="${e.world}:${e.lat}:${e.lon}">go</a>`) : ""}
    ${e.fork ? row("timeline", esc(e.fork)) : ""}
    ${det}
    ${(e.causes || []).length ? `<h3>Candidate causes (association, not proof)</h3>${e.causes.map((c) => `<div class="list-item" data-event="${c.event}"><div>${esc(c.type || "event")} E-${c.event}</div><div class="meta">${esc(c.relation)}</div></div>`).join("")}` : ""}
    ${e.significance >= 2 && e.world !== null ? `<button class="btn" data-replay="${e.id}">replay the recorded window</button>` : ""}`;
}

export async function inspectDiscovery(ctx, id) {
  const res = await api("/api/search", { q: `D-${id}`, timeline: ctx.timeline });
  const d = res.length ? res[0].discovery : null;
  if (!d) return `<h2>D-${id}</h2>`;
  const j = (o) => `<pre class="mono" style="white-space:pre-wrap">${esc(JSON.stringify(o, null, 1))}</pre>`;
  return `<h2>${esc(d.label)}</h2><div>${esc(d.title)}</div>
    <div style="margin:8px 0">${tagVal(d.status)} ${tagEpi(d.epistemic)}</div>
    ${row("kind", esc(d.kind))}${row("world · tick", `${d.world} · ${fmt(d.tick)}`)}${row("seed", fmt(d.seed))}
    ${row("validator", esc(d.validator))}
    <h3>Mechanism</h3><div class="note">${esc(d.mechanism)}</div>
    <h3>Metrics</h3>${j(d.metrics)}
    <h3>Controls</h3>${j(d.controls)}
    <h3>Replications (${d.replication.length})</h3>${d.replication.map((r) => row(`seed ${r.seed ?? "-"} · tick ${fmt(r.tick)}`, r.passed ? "passed" : "FAILED")).join("") || '<div class="note">none yet</div>'}
    <h3>Ablation</h3>${j(d.ablation)}
    ${Object.keys(d.causal || {}).length ? `<h3>Causal forks</h3>${j(d.causal)}` : ""}
    <h3>Replay reference</h3>${j(d.replay)}`;
}

export async function loadBody(ctx, id) {
  return api(`/api/world/${ctx.world}/agent/${id}/body`, { timeline: ctx.timeline });
}

export async function loadFrames(ctx, eventId) {
  return bin(`/api/frames/${eventId}`, { timeline: ctx.timeline });
}
