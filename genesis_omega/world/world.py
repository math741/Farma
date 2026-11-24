"""PlanetEngine at F4 fidelity: one living world (spec §8-13, §22-25, §37, §64-69, §94-96).

A World owns the planet's fields (climate, light, primary production, ground
deposits, structures, stored resources, trail compaction, persistent marks,
vibration and scent fields) and the organisms that live in it. ``step()``
advances one tick:

  sense (raw signals through each body's sensors) → substrate → effectors →
  physics of movement, feeding, digging, building, processing, striking,
  transferring, storing, marking, emitting → homeostasis → death →
  reproduction → field dynamics → observation buffers.

Every rule here is a law of the universe (PROGRAMMED) — none encodes a
behaviour. Shelter, storage, trails, ruins and materials appear only if
organisms' actions produce them.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import torch

from ..core.config import WorldConfig, UniverseConfig
from ..core.rng import StreamRNG, derive_seed
from ..cosmos.galaxy import S_EARTH
from ..life.agents import Agents
from ..life.development import Development, maturity
from ..life.evolution import SpeciesTracker, QDArchive, LineageStore, SeedBank
from ..life.genome import GenomeLayout, proto_genome, mutate, crossover, genome_distance
from ..life.interface import (N_INPUTS, N_DIR_SIGNALS, DIR_SIGNALS, DIR_MODALITY, DIRECTION_ANGLES, O, I, CAP,
                              ST, ACTION_BITS)
from ..life.substrate import substrate_step
from ..society.social import SocialGraph, T_PROX, T_GIVE, T_STRIKE, T_MATE, T_KIN
from .climate import Climate
from .geometry import SphereGrid
from .materials import Chemistry, MaterialRegistry, N_BASE, N_PROPS
from .physics import WorldPhysics, XenosProfile
from .terrain import generate_terrain, Terrain

CAUSE_STARVE, CAUSE_INJURY, CAUSE_AGE, CAUSE_PREDATION = 1, 2, 3, 4
DIR_MOD_IDX = [DIR_MODALITY[s] for s in DIR_SIGNALS]


class World:
    """One F4 world (a planet with individual organisms)."""

    def __init__(self, index: int, planet: dict, star: dict, ucfg: UniverseConfig, wcfg: WorldConfig,
                 phys: WorldPhysics, chem: Chemistry, cosmos_seed: int, device, salt: int = 0):
        self.index = index
        self.planet = planet
        self.star = star
        self.ucfg = ucfg
        self.cfg = wcfg
        self.phys = phys
        self.device = torch.device(device)
        self.seed = derive_seed(cosmos_seed, "world", planet["id"])
        self.rng = StreamRNG(self.seed, self.device, salt)
        self.xenos = XenosProfile.from_planet(planet, star, wcfg.day_ticks)
        self.grid = SphereGrid(wcfg.grid_w, wcfg.grid_h, self.device)
        self.tick = 0
        self.fidelity = "F4"
        self.interventions: list[dict] = []
        self.ablations: set[str] = set()
        self.blocked_materials: set[int] = set()
        self.external_sources: list[dict] = []
        self.param_scale: dict[str, float] = {}
        self.max_births = max(4, wcfg.n_max // 32)
        # genome / development
        self.layout = GenomeLayout(wcfg.hidden, wcfg.dev_channels, wcfg.dev_hidden)
        self.dev = Development(self.layout, wcfg.dev_size, wcfg.dev_steps, self.device, wcfg.emit_bands)
        self.m_max = 512 if wcfg.n_max >= 2048 else 256
        self.agents = Agents(wcfg.n_max, self.layout, self.dev, self.m_max, wcfg.emit_bands, phys.lifespan_days,
                             self.xenos.day_ticks, self.device)
        self.materials = MaterialRegistry(chem, self.m_max, self.device)
        self.chem = chem
        self.species = SpeciesTracker(self.layout.size, self.seed)
        self.qd = QDArchive(self.layout.size, self.device)
        self.lineage = LineageStore(500_000 if wcfg.n_max >= 2048 else 60_000)
        self.bank = SeedBank(wcfg.seed_bank, self.layout.size, self.device)
        self.social = SocialGraph()
        self.next_agent_id = 1
        self.counters = {"births": 0, "deaths": 0, "kills": 0, "germinations": 0, "builds": 0, "processes": 0,
                         "reactions": 0, "gifts": 0, "strikes": 0, "stores": 0, "marks": 0}
        self.new_material_events: list[dict] = []
        self.qd_new_cells: list[int] = []
        self.production_log: deque = deque(maxlen=200_000)
        self.first_kill_tick = None
        self.acc = torch.zeros(4, device=self.device)   # cumulative: energy eaten, emitted, built, processed
        self._soc_a: list[torch.Tensor] = []
        self._soc_b: list[torch.Tensor] = []
        self._soc_t: list[torch.Tensor] = []
        self._build_fields()
        self.focus = FocusRecorder(self, wcfg.focus_agents, wcfg.focus_ticks)
        self.ring = RingBuffer(self, wcfg.ring_frames)

    # ===================================================== construction
    def _build_fields(self) -> None:
        g, p, d = self.grid, self.phys, self.device
        H, W = g.H, g.W
        terr = generate_terrain(g, derive_seed(self.seed, "terrain"), float(self.planet["water"]), N_BASE)
        self.terrain = terr
        self.ocean = terr.ocean
        self.oceanf = terr.ocean.float().flatten()
        self.climate = Climate(g, self.cfg.climate_div, terr, self.xenos, p)
        self.calibration = self.climate.calibrate(float(self.planet["t_mean"]), 0, self.ucfg.climate_every)
        self.temp = self.climate.fine_temperature()
        self.tday = self.temp.clone()
        self.precip = self.climate.fine_precip()
        self.cloud = self.climate.fine_cloud()
        self.light = torch.zeros(H, W, device=d)
        self.K = torch.where(self.ocean, p.k_ocean * terr.fertility, p.k_land * terr.fertility)
        self.R = self.K * 0.6
        self.store = torch.zeros(H, W, device=d)
        self.compaction = torch.zeros(H, W, device=d)
        self.marks = torch.zeros(2, H, W, device=d)
        self.sound = torch.zeros(self.cfg.emit_bands, H, W, device=d)
        self.scent = torch.zeros(2, H, W, device=d)
        self.struct_h = torch.zeros(H, W, device=d)
        self.struct_mp = torch.zeros(N_PROPS, H, W, device=d)   # height-weighted props (extensive)
        self.struct_mat = torch.full((H, W), -1, dtype=torch.long, device=d)
        self.struct_age = torch.zeros(H, W, device=d)           # ticks the cell has been a structure
        self.struct_builder = torch.full((H, W), -1, dtype=torch.long, device=d)  # lineage of last builder
        self.shelter = terr.nat_shelter.clone()
        self.occupancy = torch.zeros(H, W, device=d)            # slow EMA of agent presence
        self.cold_occupancy = torch.zeros(H, W, device=d)
        self.warm_occupancy = torch.zeros(H, W, device=d)
        self.store_in = torch.zeros(H, W, device=d)             # cumulative deposits
        self.store_out = torch.zeros(H, W, device=d)            # cumulative withdrawals
        self.last_visit = torch.zeros(H, W, device=d)
        self.light_ref = S_EARTH * max(self.xenos.flux_rel, 0.05)

    # ======================================================== seeding
    def seed_life(self, n_sites: int = 3, biomass: float = 0.5) -> list[dict]:
        """Introduce founder populations (a PROGRAMMED initial condition: no abiogenesis
        chemistry is simulated). Each site gets its own random proto-genome."""
        g = self.grid
        gen = self.rng.gen("seed-life", 0)
        A = self.agents
        n_total = int(self.cfg.n_max * self.cfg.founders_fraction * (0.5 + biomass))
        n_total = max(8, min(n_total, self.cfg.n_max))
        land = (~self.ocean) & (self.temp > 2) & (self.temp < 32) & (self.K > 0.25)
        cand = torch.nonzero(land.flatten()).flatten()
        if cand.numel() == 0:
            cand = torch.nonzero((self.K > 0.1).flatten()).flatten()
        sites = []
        per = n_total // n_sites
        for s in range(n_sites):
            c = int(cand[int(torch.randint(cand.numel(), (1,), generator=gen, device=self.device))])
            center = g.cell_center(torch.tensor([c], device=self.device))[0]
            proto = proto_genome(self.layout, gen, self.device)
            genomes = proto.repeat(per, 1) + 0.02 * torch.randn(per, self.layout.size, generator=gen,
                                                                device=self.device)
            pos = g.random_points_near(center, per, 9.0 * g.dang, gen)
            head = g.random_tangent(pos, gen)
            slots = A.free_slots(per)
            sid = self.species._new(self.species.project(genomes).mean(0), 0, self.tick)
            ids = A.spawn(slots, genomes, pos, head, torch.full((per,), 8.0, device=self.device),
                          torch.full((per,), -1, dtype=torch.long, device=self.device),
                          torch.full((per,), -1, dtype=torch.long, device=self.device),
                          torch.zeros(per, dtype=torch.long, device=self.device),
                          torch.full((per,), sid, dtype=torch.long, device=self.device), self.tick,
                          self.next_agent_id)
            self.next_agent_id += per
            self._record_births(ids, slots)
            lat, lon = g.latlon(center)
            sites.append({"site": s, "species": sid, "n": per, "lat": math.degrees(float(lat)),
                          "lon": math.degrees(float(lon))})
        self.focus.refill()
        return sites

    # ======================================================== the tick
    @torch.no_grad()
    def step(self) -> dict:
        t = self.tick
        cfg, p, g, A, x = self.cfg, self.phys, self.grid, self.agents, self.xenos
        dev = self.device
        N = A.N
        HW = g.H * g.W
        events: dict = {}
        if t % self.ucfg.climate_every == 0:
            dt = self.ucfg.climate_every / x.day_ticks
            self.climate.step(t, dt)
            self.temp = self.climate.fine_temperature()
            self.precip = self.climate.fine_precip()
            self.cloud = self.climate.fine_cloud()
            a = self.ucfg.climate_every / x.day_ticks
            self.tday = (1 - a) * self.tday + a * self.temp
        ins = self.climate.insolation(g, t)
        self.light = (ins / self.light_ref).clamp(0, 1.5) * x.light_scale * (1 - 0.45 * self.cloud)
        for src in self.external_sources:
            self._apply_external_source(src)

        act = A.active
        actf = act.float()
        ph = A.phys
        caps = A.caps
        mass = caps[:, CAP["mass"]]
        ecap = p.energy_cap_base + p.energy_cap_storage * caps[:, CAP["storage"]]
        mat = maturity(A.age, ph["lifespan"], p.maturity_frac)
        _, _, cell = g.cell_of(A.pos)
        cell = torch.where(act, cell, torch.zeros_like(cell))
        in_water = self.oceanf[cell] > 0.5

        # ---------------- crowding & pairing (spatial hashing by cell)
        key = torch.where(act, cell, torch.full_like(cell, HW))
        order = torch.argsort(key * N + torch.arange(N, device=dev))
        skey = key[order]
        nxt_same = torch.zeros(N, dtype=torch.bool, device=dev)
        prv_same = torch.zeros(N, dtype=torch.bool, device=dev)
        nxt_same[:-1] = (skey[:-1] == skey[1:]) & (skey[:-1] < HW)
        prv_same[1:] = (skey[1:] == skey[:-1]) & (skey[1:] < HW)
        part_sorted = torch.where(nxt_same, torch.roll(order, -1), torch.where(prv_same, torch.roll(order, 1),
                                                                                torch.full_like(order, -1)))
        partner = torch.empty_like(part_sorted)
        partner[order] = part_sorted
        has_p = (partner >= 0) & act
        psafe = partner.clamp(min=0)
        counts = torch.bincount(cell[act], minlength=HW).float()
        crowd = (counts[cell] - 1).clamp(min=0)

        # ---------------- sensing
        pts = g.offset_points(A.pos, A.head, DIRECTION_ANGLES, p.sense_dist * g.dang)
        _, _, pc = g.cell_of(pts.reshape(-1, 3))
        pc = pc.view(N, 4)
        fields = torch.stack([
            self.light.flatten(), self.R.flatten(), ((self.temp - 15.0) / 20.0).flatten(),
            self.scent[0].flatten(), self.scent[1].flatten(),
            *[self.sound[k].flatten() for k in range(cfg.emit_bands)],
            (self.struct_h / p.wall_height).clamp(0, 2).flatten(), self.marks[0].flatten(),
            self.marks[1].flatten(), self.oceanf], 0)                                 # (13, HW)
        if "vibration" in self.ablations:
            fields[5:5 + cfg.emit_bands] = 0
        vals = fields[:, pc].permute(1, 2, 0)                                        # (N,4,13)
        gains = A.sens[:, :, DIR_MOD_IDX]                                            # (N,4,13)
        gen = self.rng.gen("sense", t)
        u_dir = vals * gains + p.sensor_noise * torch.randn(vals.shape, generator=gen, device=dev) * gains
        props = self.materials.props
        cm = A.carried_mat.clamp(min=0)
        carrying = (A.carried_amt > 0.01) & (A.carried_mat >= 0)
        cprops = props[cm] * carrying.unsqueeze(1).float()
        chemo_here = A.sens[:, :, 1].mean(1)
        local = torch.stack([
            self.R.flatten()[cell] * chemo_here, self.store.flatten()[cell] * chemo_here,
            A.carried_amt.clamp(0, 2), cprops[:, 1], cprops[:, 2], cprops[:, 4],
            (crowd / 4).clamp(0, 2), A.last_damage * 4, A.last_gift, A.energy / ecap, A.integrity,
            ((A.tbody - ph["t_opt"]) / 10.0).clamp(-3, 3), A.speed / p.max_speed, A.bump,
            torch.ones(N, device=dev)], 1)
        u = torch.cat([u_dir.reshape(N, 4 * N_DIR_SIGNALS), local], 1)
        if u.shape[1] != N_INPUTS:
            raise RuntimeError("world interface size mismatch")

        # ---------------- substrate
        noise = torch.randn(N, self.layout.hidden, generator=self.rng.gen("substrate", t), device=dev)
        W_new, x_new, y, m, dWn = substrate_step(A.params, A.W, A.x, A.m, u, noise,
                                                 plastic="plasticity" not in self.ablations,
                                                 consolidate=(t % 8 == 0))
        A.W, A.x, A.m, A.dW = W_new, x_new, m, dWn
        A.last_u, A.last_y = u, y

        # ---------------- effector decoding
        turn = torch.tanh(y[:, O["turn"]]) * p.max_turn * caps[:, CAP["turn"]]
        thrust = torch.sigmoid(y[:, O["thrust"]])
        emit = torch.sigmoid(y[:, O["emit0"]:O["emit0"] + cfg.emit_bands])
        emit = emit * A.emit_prof * (caps[:, CAP["emit_power"]] * mat).unsqueeze(1)
        emit = torch.where(emit > 0.05, emit, torch.zeros_like(emit)) * actf.unsqueeze(1)
        do = {k: (y[:, O[k]] > 0) & act for k in ("eat", "dig", "build", "process", "strike", "give", "store",
                                                   "reproduce")}
        mark_out = torch.relu(torch.tanh(y[:, O["mark0"]:O["mark1"] + 1])) * actf.unsqueeze(1)
        tool = 1.0 + p.tool_gain * cprops[:, 1] * (1 - 0.5 * cprops[:, 0]) * A.carried_amt.clamp(0, 1)

        # ---------------- movement
        slope = self.terrain.slope.flatten()[cell]
        comp = self.compaction.flatten()[cell].clamp(0, 1)
        medium = torch.where(in_water, caps[:, CAP["water_speed"]], caps[:, CAP["land_speed"]])
        terrain_f = torch.where(in_water, torch.ones_like(slope),
                                (1 + p.compaction_speed * comp) / (1 + 0.8 * slope))
        load = A.carried_amt * (0.3 + cprops[:, 0])
        speed = thrust * p.max_speed * medium * mat * terrain_f / (1 + load / mass.clamp_min(0.1)) / x.move_scale
        speed = speed * actf
        head = g.turn(A.pos, A.head, turn)
        new_pos, new_head = g.move(A.pos, head, speed * g.dang)
        _, _, ncell = g.cell_of(new_pos)
        blocked = (self.struct_h.flatten()[ncell] >= p.wall_height) & (ncell != cell) & act
        A.pos = torch.where(blocked.unsqueeze(1), A.pos, new_pos)
        A.head = torch.where(blocked.unsqueeze(1), head, new_head)
        A.speed = torch.where(blocked, torch.zeros_like(speed), speed)
        A.bump = blocked.float()
        moving_land = act & (~in_water) & (A.speed > 0.05)
        self.compaction.view(-1).index_add_(0, cell[moving_land],
                                            (p.compaction_gain * mass[moving_land].sqrt()))

        # ---------------- feeding (stores first, then primary production)
        room = ((ecap - A.energy) / p.eat_eff).clamp(min=0)          # ingestion capacity (satiation)
        demand = torch.minimum(do["eat"].float() * caps[:, CAP["bite"]] * p.bite * mat * tool, room)
        store_f, R_f = self.store.view(-1), self.R.view(-1)
        tot = torch.zeros(HW, device=dev).index_add_(0, cell, demand)
        fs = (store_f / tot.clamp_min(1e-9)).clamp(max=1.0)
        from_store = demand * fs[cell]
        store_f.sub_(torch.zeros(HW, device=dev).index_add_(0, cell, from_store)).clamp_(min=0)
        rem = demand - from_store
        tot2 = torch.zeros(HW, device=dev).index_add_(0, cell, rem)
        edible = (R_f - p.refuge * self.K.view(-1)).clamp(min=0)    # grazing refuge (roots survive)
        fr = (edible / tot2.clamp_min(1e-9)).clamp(max=1.0)
        from_R = rem * fr[cell]
        R_f.sub_(torch.zeros(HW, device=dev).index_add_(0, cell, from_R)).clamp_(min=0)
        self.store_out.view(-1).index_add_(0, cell, from_store)
        gained = (from_store + from_R) * p.eat_eff
        A.energy = A.energy + gained
        self.acc[0] += gained.sum()

        # ---------------- digging (ground deposits, or dismantling structures)
        dig = do["dig"] & (~carrying) & (caps[:, CAP["grasp"]] > 0.05) & (~in_water)
        sh_f = self.struct_h.view(-1)
        from_struct = sh_f[cell] > 0.05
        dep = self.terrain.deposits.view(N_BASE, -1)
        dep_here = dep[:, cell]                                       # (M,N)
        dom = dep_here.argmax(0)
        dom_amt = dep_here.gather(0, dom.unsqueeze(0)).squeeze(0)
        src_mat = torch.where(from_struct, self.struct_mat.view(-1)[cell], dom)
        src_ok = from_struct | (dom_amt > 0.1)
        dig = dig & src_ok & (src_mat >= 0)
        want = dig.float() * p.dig_amount * caps[:, CAP["grasp"]] * mat * tool
        # proportional allocation per (cell, source)
        w_s = want * from_struct.float()
        w_d = want * (~from_struct).float()
        tot_s = torch.zeros(HW, device=dev).index_add_(0, cell, w_s)
        frac_s = (sh_f / tot_s.clamp_min(1e-9)).clamp(max=1.0)
        got_s = w_s * frac_s[cell]
        dkey = cell * N_BASE + dom
        tot_d = torch.zeros(HW * N_BASE, device=dev).index_add_(0, dkey, w_d)
        dep_flat = dep.t().reshape(-1)                                # (HW*M) index cell*M+m
        frac_d = (dep_flat / tot_d.clamp_min(1e-9)).clamp(max=1.0)
        got_d = w_d * frac_d[dkey]
        taken_s = torch.zeros(HW, device=dev).index_add_(0, cell, got_s)
        self._remove_structure(taken_s)
        taken_d = torch.zeros(HW * N_BASE, device=dev).index_add_(0, dkey, got_d)
        self.terrain.deposits.copy_((dep_flat - taken_d).clamp(min=0).view(HW, N_BASE).t().reshape(
            N_BASE, g.H, g.W))
        got = got_s + got_d
        picked = dig & (got > 0.01)
        A.carried_mat = torch.where(picked, src_mat, A.carried_mat)
        A.carried_amt = torch.where(picked, got, A.carried_amt)

        # ---------------- processing (hidden chemistry)
        carrying = (A.carried_amt > 0.01) & (A.carried_mat >= 0)
        proc = do["process"] & carrying
        n_proc = int(proc.sum())
        if n_proc:
            self._process(proc, cell, from_struct, dom, dom_amt, t)
        carrying = (A.carried_amt > 0.01) & (A.carried_mat >= 0)

        # ---------------- building (into the cell ahead)
        build = do["build"] & carrying & (~in_water)
        n_build = int(build.sum())
        if n_build:
            self._build(build, t)

        # ---------------- strike / give / store / mark / emit
        strike = do["strike"] & has_p
        dmg = strike.float() * p.strike_damage * caps[:, CAP["strike"]] * mat * (
            1 - 0.8 * caps[psafe, CAP["armor"]])
        recv_dmg = torch.zeros(N, device=dev).index_add_(0, psafe, dmg)
        A.integrity = A.integrity - recv_dmg
        A.last_damage = recv_dmg
        give = do["give"] & has_p & (A.energy > 0.2 * ecap)
        amt = give.float() * p.give_frac * A.energy
        A.energy = A.energy - amt
        recv = torch.zeros(N, device=dev).index_add_(0, psafe, amt * 0.9)
        A.energy = A.energy + recv
        A.last_gift = recv
        store = do["store"] & (~in_water) & (A.energy > 0.25 * ecap)
        samt = store.float() * p.store_frac * A.energy
        A.energy = A.energy - samt
        store_f.index_add_(0, cell, samt / p.eat_eff)
        self.store_in.view(-1).index_add_(0, cell, samt / p.eat_eff)
        mk = mark_out * p.mark_gain
        for c in range(2):
            self.marks[c].view(-1).index_add_(0, cell, mk[:, c])
        n_store, n_mark = torch.stack([store.sum(), (mark_out.sum(1) > 0).sum()]).tolist()
        self.counters["stores"] += int(n_store)
        self.counters["marks"] += int(n_mark)
        self.acc[1] += emit.sum()
        sound_src = torch.zeros(cfg.emit_bands, HW, device=dev)
        sound_src.index_add_(1, cell, (emit * p.emit_gain).t())
        scent_src = torch.zeros(2, HW, device=dev)
        scent_src.index_add_(1, cell, (ph["scent"] * (mass.sqrt() * actf).unsqueeze(1)).t())

        # ---------------- energetics & homeostasis
        costs = (p.metab_mass * mass.pow(0.75) + p.metab_unit * caps[:, CAP["units"]] + p.tolerance_cost * ph["t_tol"]
                 + p.move_cost * mass * (A.speed / p.max_speed) ** 2 * x.move_scale
                 + p.emit_cost * emit.sum(1) + p.dig_cost * dig.float() + p.process_cost * proc.float()
                 + p.build_cost * build.float() + p.strike_cost * strike.float() + p.mark_cost * mark_out.sum(1))
        A.energy = torch.minimum(A.energy - costs * actf, ecap)
        shelter = self.shelter.flatten()[cell]
        heat = (counts[cell] * 0.6).clamp(max=4.0)
        t_env = self.temp.flatten()[cell]
        t_eff = t_env * (1 - shelter) + (self.tday.flatten()[cell] + 2.0 + heat) * shelter
        rate = p.thermal_rate / (1 + 3 * caps[:, CAP["insulation"]])
        A.tbody = A.tbody + (t_eff - A.tbody) * rate
        deviation = ((A.tbody - ph["t_opt"]).abs() - ph["t_tol"]).clamp(min=0)
        A.integrity = A.integrity - p.thermal_damage * deviation
        rep = act & (A.energy > 0.4 * ecap) & (A.integrity < 1)
        A.integrity = torch.where(rep, (A.integrity + p.repair_rate).clamp(max=1.0), A.integrity)
        A.energy = A.energy - rep.float() * p.repair_cost
        old = A.age > ph["lifespan"]
        A.integrity = A.integrity - old.float() * p.senescence
        A.age = A.age + actf

        # ---------------- lifetime statistics
        S = A.stats
        S[:, ST["ticks"]] += actf
        S[:, ST["water_ticks"]] += (in_water & act).float()
        S[:, ST["distance"]] += A.speed
        S[:, ST["emitted"]] += emit.sum(1)
        S[:, ST["builds"]] += build.float()
        S[:, ST["processes"]] += proc.float()
        S[:, ST["strikes"]] += strike.float()
        S[:, ST["gives"]] += give.float()
        S[:, ST["eaten"]] += gained
        S[:, ST["stored"]] += samt
        S[:, ST["dug"]] += picked.float()
        S[:, ST["marks"]] += (mark_out.sum(1) > 0).float()
        S[:, ST["sheltered_ticks"]] += (shelter > 0.3).float() * actf

        # ---------------- deaths
        killed = act & (A.integrity <= 0) & (recv_dmg > 0)
        if bool(killed.any()):
            share = dmg / recv_dmg[psafe].clamp_min(1e-9)
            victim_e = (A.energy[psafe].clamp(min=0) + 0.5 * mass[psafe] * p.eat_eff * 0.2)
            loot = share * victim_e * p.kill_transfer * killed[psafe].float() * strike.float()
            A.energy = torch.minimum(A.energy + loot, ecap)
            S[:, ST["kills"]] += ((loot > 0) & strike).float()
            if self.first_kill_tick is None:
                self.first_kill_tick = t
                events["first_kill"] = True
        dead = act & ((A.energy <= 0) | (A.integrity <= 0) | (A.age > 3 * ph["lifespan"]))
        n_dead = int(dead.sum())
        if n_dead:
            cause = torch.full((N,), CAUSE_INJURY, dtype=torch.long, device=dev)
            cause = torch.where(A.energy <= 0, torch.full_like(cause, CAUSE_STARVE), cause)
            cause = torch.where(old & (A.integrity <= 0), torch.full_like(cause, CAUSE_AGE), cause)
            cause = torch.where(killed, torch.full_like(cause, CAUSE_PREDATION), cause)
            self._die(dead, cause, cell, mass, t)

        # ---------------- reproduction & germination
        births = self._reproduce(do["reproduce"], partner, has_p, ecap, t)
        if births:
            events["births"] = births
        germ = self._germinate(t)
        if germ:
            events["germinated"] = germ

        # ---------------- social interaction records
        self._record_social(has_p, partner, strike, give, t)

        # ---------------- field dynamics
        self._fields(sound_src, scent_src, counts, t)

        # ---------------- observation
        bits = (do["eat"].long() << ACTION_BITS["eat"]) | (dig.long() << ACTION_BITS["dig"]) | \
            (build.long() << ACTION_BITS["build"]) | (proc.long() << ACTION_BITS["process"]) | \
            (strike.long() << ACTION_BITS["strike"]) | (give.long() << ACTION_BITS["give"]) | \
            (store.long() << ACTION_BITS["store"]) | ((mark_out.sum(1) > 0).long() << ACTION_BITS["mark"]) | \
            (do["reproduce"].long() << ACTION_BITS["reproduce"]) | \
            ((emit.sum(1) > 0).long() << ACTION_BITS["emit"]) | (in_water.long() << ACTION_BITS["water"]) | \
            (((A.carried_amt > 0.01) & (A.carried_mat >= 0)).long() << ACTION_BITS["carrying"]) | \
            ((shelter > 0.3).long() << ACTION_BITS["sheltered"]) | ((A.speed > 0.05).long() << ACTION_BITS["moving"])
        self._last_bits = bits
        self._last_emit = emit
        self._last_cell = cell
        self._last_context = (self.R.flatten()[cell], self.light.flatten()[cell], recv_dmg > 0, crowd)
        self.focus.record(u, x_new, y, m, dWn, ecap, shelter, t_env, cell, recv_dmg)
        if t % cfg.ring_every == 0:
            self.ring.capture(t)
        self.tick += 1
        return events

    # ================================================== action helpers
    def _remove_structure(self, taken: torch.Tensor) -> None:
        sh = self.struct_h.view(-1)
        mp = self.struct_mp.view(N_PROPS, -1)
        frac = (taken / sh.clamp_min(1e-9)).clamp(0, 1)
        mp.mul_(1 - frac.unsqueeze(0))
        sh.sub_(taken).clamp_(min=0)
        gone = sh < 0.02
        sh[gone] = 0
        mp[:, gone] = 0
        self.struct_mat.view(-1)[gone] = -1

    def _process(self, proc, cell, from_struct, dom, dom_amt, t) -> None:
        A = self.agents
        idx = torch.nonzero(proc).flatten()
        c = cell[idx]
        b_struct = from_struct[idx]
        b_mat = torch.where(b_struct, self.struct_mat.view(-1)[c], dom[idx])
        b_amt = torch.where(b_struct, self.struct_h.view(-1)[c], dom_amt[idx])
        ok = (b_mat >= 0) & (b_amt > 0.05)
        idx, c, b_mat, b_amt, b_struct = idx[ok], c[ok], b_mat[ok], b_amt[ok], b_struct[ok]
        if idx.numel() == 0:
            return
        a_mat = A.carried_mat[idx]
        band = self.chem.band_of(self.temp.view(-1)[c])
        props = self.materials.props
        prod, reacted = self.chem.react(props[a_mat], props[b_mat], band)
        if self.blocked_materials or "chemistry" in self.ablations:
            pid, _ = self.materials.identify(prod)
            blocked = torch.tensor(sorted(self.blocked_materials), device=self.device, dtype=torch.long)
            hit = (pid.unsqueeze(1) == blocked.unsqueeze(0)).any(1) if blocked.numel() else torch.zeros_like(reacted)
            if "chemistry" in self.ablations:
                hit = torch.ones_like(reacted)
            reacted = reacted & ~hit
        self.counters["processes"] += int(idx.numel())
        if not bool(reacted.any()):
            return
        r = torch.nonzero(reacted).flatten()
        idx, c, b_mat, b_amt, b_struct, a_mat, band, prod = (v[r] for v in (idx, c, b_mat, b_amt, b_struct,
                                                                            a_mat, band, prod))
        use = torch.minimum(A.carried_amt[idx], b_amt)
        # consume the partner material from the cell (structure or ground deposit)
        HW = self.grid.H * self.grid.W
        s_take = torch.zeros(HW, device=self.device).index_add_(0, c[b_struct], use[b_struct])
        self._remove_structure(s_take.clamp(max=self.struct_h.view(-1)))
        if bool((~b_struct).any()):
            dflat = self.terrain.deposits.view(N_BASE, -1)
            nb = ~b_struct
            take = torch.zeros(N_BASE * HW, device=self.device).index_add_(
                0, b_mat[nb] * HW + c[nb], use[nb])
            dflat.copy_((dflat.reshape(-1) - take).clamp(min=0).view(N_BASE, HW))
        recipes = torch.stack([a_mat, b_mat, band], 1)
        ids, new_ids = self.materials.register_products(prod, recipes, t, A.id[idx], c, A.lineage[idx],
                                                             A.species[idx])
        A.carried_mat[idx] = ids
        A.carried_amt[idx] = use
        self.acc[3] += use.sum()
        self.materials.produced.index_add_(0, ids, torch.ones_like(ids))
        first = ~A.made[idx, ids]
        A.made[idx, ids] = True
        self.materials.producers.index_add_(0, ids, first.long())
        self.counters["reactions"] += int(idx.numel())
        # cultural record of who produced what, where, when (CPU)
        rows = torch.stack([A.id[idx], ids, c, A.lineage[idx], A.species[idx], first.long()], 1).cpu().numpy()
        for row in rows:
            self.production_log.append((t, *(int(v) for v in row)))
        for mid in new_ids:
            self.new_material_events.append({"tick": t, "material": mid,
                                             "record": self.materials.records[mid].to_dict()})

    def _build(self, build, t) -> None:
        A, g, p = self.agents, self.grid, self.phys
        idx = torch.nonzero(build).flatten()
        front = g.offset_points(A.pos[idx], A.head[idx], (0.0,), 1.0 * g.dang)[:, 0]
        _, _, tc = g.cell_of(front)
        ok = self.oceanf[tc] < 0.5
        idx, tc = idx[ok], tc[ok]
        if idx.numel() == 0:
            return
        HW = g.H * g.W
        mid = A.carried_mat[idx]
        amt = A.carried_amt[idx]
        coh = self.materials.props[mid, 3]
        hmax = p.max_struct_height * (0.3 + coh) * self.xenos.struct_scale
        sh = self.struct_h.view(-1)
        room = (hmax - sh[tc]).clamp(min=0)
        tot = torch.zeros(HW, device=self.device).index_add_(0, tc, amt)
        frac = (room / tot[tc].clamp_min(1e-9)).clamp(max=1.0)
        placed = amt * frac
        add = torch.zeros(HW, device=self.device).index_add_(0, tc, placed)
        mp = self.struct_mp.view(N_PROPS, -1)
        mp.index_add_(1, tc, (self.materials.props[mid] * placed.unsqueeze(1)).t())
        sh.add_(add)
        top = torch.full((HW,), -1, dtype=torch.long, device=self.device)
        top.scatter_reduce_(0, tc, torch.where(placed > 0, mid, torch.full_like(mid, -1)), reduce="amax")
        self.struct_mat.view(-1).copy_(torch.where(top >= 0, top, self.struct_mat.view(-1)))
        lin = torch.full((HW,), -1, dtype=torch.long, device=self.device)
        lin.scatter_reduce_(0, tc, A.lineage[idx], reduce="amax")
        self.struct_builder.view(-1).copy_(torch.where(lin >= 0, lin, self.struct_builder.view(-1)))
        A.carried_amt[idx] = amt - placed
        self.materials.built.index_add_(0, mid, placed)
        self.acc[2] += placed.sum()
        self.counters["builds"] += int(idx.numel())

    def _die(self, dead, cause, cell, mass, t) -> None:
        A, p = self.agents, self.phys
        idx = torch.nonzero(dead).flatten()
        c = cell[idx]
        corpse = (A.energy[idx].clamp(min=0) + 0.3 * mass[idx] * p.eat_eff) / p.eat_eff
        self.R.view(-1).index_add_(0, c, corpse)
        # carried material is dropped where the organism died (an artefact on the ground);
        # at sea it sinks out of reach — water never holds structures (same rule as building)
        drop = (A.carried_amt[idx] > 0.01) & (A.carried_mat[idx] >= 0) & (self.oceanf[c] < 0.5)
        if bool(drop.any()):
            dc, dm, da = c[drop], A.carried_mat[idx][drop], A.carried_amt[idx][drop]
            self.struct_h.view(-1).index_add_(0, dc, da)
            self.struct_mp.view(N_PROPS, -1).index_add_(1, dc, (self.materials.props[dm] * da.unsqueeze(1)).t())
            top = torch.full_like(self.struct_mat.view(-1), -1)
            top.scatter_reduce_(0, dc, dm, reduce="amax")
            sm = self.struct_mat.view(-1)
            sm.copy_(torch.where((sm < 0) & (top >= 0), top, sm))
        # observer-side archives
        desc = QDArchive.descriptors(A.stats[idx], mass[idx], ST, p.max_speed)
        fit = A.stats[idx, ST["offspring"]] + 1e-4 * A.stats[idx, ST["ticks"]]
        meta = torch.stack([A.id[idx], A.species[idx], torch.full_like(A.id[idx], t), A.lineage[idx]], 1)
        self.qd_new_cells.extend(self.qd.ingest(A.genome[idx], desc, fit, meta, t))
        # seed bank (dormant propagules from adults) — declared law
        adult = A.age[idx] > A.phys["lifespan"][idx] * p.maturity_frac
        coin = torch.rand(idx.numel(), generator=self.rng.gen("bank", t), device=self.device) < p.bank_deposit_prob
        dep = adult & coin
        if bool(dep.any()):
            j = idx[dep]
            self.bank.deposit(A.genome[j], A.pos[j], torch.stack([A.lineage[j], A.species[j], A.generation[j],
                                                                   torch.full_like(A.id[j], t)], 1))
        ids = A.id[idx].cpu().numpy()
        self.lineage.add_deaths(ids, cause[idx].cpu().numpy(), A.stats[idx, ST["offspring"]].long().cpu().numpy(), t)
        self.counters["deaths"] += int(idx.numel())
        self.counters["kills"] += int((cause[idx] == CAUSE_PREDATION).sum())
        A.active[idx] = False
        A.id[idx] = -1
        self.focus.on_deaths(idx)

    def _reproduce(self, want, partner, has_p, ecap, t) -> int:
        A, p = self.agents, self.phys
        ph = A.phys
        mature = A.active & (A.age > ph["lifespan"] * p.maturity_frac)
        ready = want & mature & (A.energy > p.repro_threshold * ecap)
        n_ready = int(ready.sum())
        if n_ready == 0:
            return 0
        free = torch.nonzero(~A.active).flatten()
        n = min(n_ready, int(free.numel()), self.max_births)
        if n <= 0:
            return 0
        gen = self.rng.gen("repro", t)
        prio = torch.rand(A.N, generator=gen, device=self.device)
        prio = torch.where(ready, prio, torch.full_like(prio, -1.0))
        parents = torch.topk(prio, n).indices
        parents = parents[prio[parents] >= 0]
        n = parents.numel()
        if n == 0:
            return 0
        slots = free[:n]
        pg = A.genome[parents]
        mate = partner[parents]
        msafe = mate.clamp(min=0)
        sexual = has_p[parents] & want[msafe] & mature[msafe] & (
            torch.rand(n, generator=gen, device=self.device) < ph["sex"][parents]) & (
            genome_distance(pg, A.genome[msafe]) < 0.08)
        child = pg
        if bool(sexual.any()):
            mixed = crossover(self.layout, pg, A.genome[msafe], gen)
            child = torch.where(sexual.unsqueeze(1), mixed, pg)
        child = mutate(self.layout, child, gen)
        invest = ph["invest"][parents]
        e_child = invest * A.energy[parents]
        A.energy[parents] = A.energy[parents] - e_child * (1 + p.repro_overhead)
        if bool(sexual.any()):
            # the mate contributes a share of the investment
            share = 0.3 * e_child * sexual.float()
            A.energy.index_add_(0, msafe, -share)
            e_child = e_child + share
        off = self.grid.random_tangent(A.pos[parents], gen)
        pos, _ = self.grid.move(A.pos[parents], off, torch.full((n,), 0.3 * self.grid.dang, device=self.device))
        head = self.grid.random_tangent(pos, gen)
        ids = A.spawn(slots, child, pos, head, e_child, A.id[parents], A.lineage[parents],
                      A.generation[parents] + 1, A.species[parents], t, self.next_agent_id)
        self.next_agent_id += n
        # growing a body costs energy in proportion to its mass (construction cost)
        A.energy[slots] = A.energy[slots] - p.body_cost * A.caps[slots, CAP["mass"]]
        A.stats[parents, ST["offspring"]] += 1
        if bool(sexual.any()):
            A.stats[msafe[sexual], ST["offspring"]] += 1
            self._soc_a.append(A.id[parents][sexual])
            self._soc_b.append(A.id[msafe][sexual])
            self._soc_t.append(torch.full((int(sexual.sum()),), T_MATE, dtype=torch.long, device=self.device))
        self._soc_a.append(A.id[parents])
        self._soc_b.append(ids)
        self._soc_t.append(torch.full((n,), T_KIN, dtype=torch.long, device=self.device))
        self._record_births(ids, slots)
        self.counters["births"] += n
        return n

    def _germinate(self, t) -> int:
        A, p, bank = self.agents, self.phys, self.bank
        frac = A.count() / A.N
        if frac >= 0.15 or bank.count() == 0:
            return 0
        gen = self.rng.gen("germ", t)
        prob = p.bank_germination * (1 - frac / 0.15)
        hit = bank.valid & (torch.rand(bank.cap, generator=gen, device=self.device) < prob)
        idx = torch.nonzero(hit).flatten()
        free = torch.nonzero(~A.active).flatten()
        n = min(int(idx.numel()), int(free.numel()))
        if n == 0:
            return 0
        idx = idx[:n]
        genomes = mutate(self.layout, bank.genomes[idx], gen)
        pos = bank.pos[idx]
        head = self.grid.random_tangent(pos, gen)
        meta = bank.meta[idx]
        ids = A.spawn(free[:n], genomes, pos, head, torch.full((n,), 4.0, device=self.device),
                      torch.full((n,), -1, dtype=torch.long, device=self.device), meta[:, 0], meta[:, 2] + 1,
                      meta[:, 1], t, self.next_agent_id)
        self.next_agent_id += n
        bank.valid[idx] = False
        self._record_births(ids, free[:n])
        self.counters["germinations"] += n
        return n

    def _record_births(self, ids, slots) -> None:
        A = self.agents
        self.lineage.add_births(ids.cpu().numpy(), A.parent[slots].cpu().numpy(), A.lineage[slots].cpu().numpy(),
                                A.species[slots].cpu().numpy(), A.generation[slots].cpu().numpy(), self.tick,
                                slots.cpu().numpy())

    def _record_social(self, has_p, partner, strike, give, t) -> None:
        A = self.agents
        psafe = partner.clamp(min=0)
        if t % 5 == 0:
            m = has_p & (torch.arange(A.N, device=self.device) < psafe)
            self._soc_a.append(A.id[m])
            self._soc_b.append(A.id[psafe[m]])
            self._soc_t.append(torch.full((int(m.sum()),), T_PROX, dtype=torch.long, device=self.device))
        for mask, typ in ((strike, T_STRIKE), (give & has_p, T_GIVE)):
            if bool(mask.any()):
                self._soc_a.append(A.id[mask])
                self._soc_b.append(A.id[psafe[mask]])
                self._soc_t.append(torch.full((int(mask.sum()),), typ, dtype=torch.long, device=self.device))
                self.counters["strikes" if typ == T_STRIKE else "gifts"] += int(mask.sum())
        if t % 25 == 0:
            self.flush_social()

    def flush_social(self) -> None:
        if not self._soc_a:
            return
        a = torch.cat(self._soc_a).cpu().numpy()
        b = torch.cat(self._soc_b).cpu().numpy()
        ty = torch.cat(self._soc_t).cpu().numpy()
        self._soc_a, self._soc_b, self._soc_t = [], [], []
        ok = (a >= 0) & (b >= 0)
        self.social.ingest(a[ok], b[ok], ty[ok], self.tick)

    def _fields(self, sound_src, scent_src, counts, t) -> None:
        p, g = self.phys, self.grid
        K = self.cfg.emit_bands
        H, W = g.H, g.W
        # primary production: light, warmth and water limited logistic growth
        tg = torch.exp(-((self.temp - p.plant_t_opt) / p.plant_t_width) ** 2) * (self.temp > 0).float()
        wat = torch.where(self.ocean, torch.ones_like(self.precip), (self.precip / 0.25).clamp(0, 1))
        k_eff = self.K * (1 - p.trample * self.compaction.clamp(0, 1)) * (1 - (self.struct_h > 0.05).float() * 0.8)
        growth = p.growth_rate * self.param_scale.get("growth_rate", 1.0) * tg * wat * self.light.clamp(0, 1) * 2.0
        self.R = (self.R + growth * self.R * (1 - self.R / k_eff.clamp_min(1e-4)) + p.reseed * k_eff).clamp(
            min=0, max=None)
        self.R = torch.minimum(self.R, 2.0 * k_eff + 0.05)
        sheltered = self.shelter
        self.store.mul_(1 - p.store_decay * (1 - 0.9 * sheltered))
        self.compaction.mul_(1 - p.compaction_decay)
        self.marks.mul_(1 - p.mark_decay)
        self.sound = self.xenos.sound_decay * g.blur(self.sound) + sound_src.view(K, H, W)
        self.scent = p.scent_decay * g.blur(self.scent) + scent_src.view(2, H, W)
        occ = counts.view(H, W)
        self.occupancy.mul_(0.999).add_(occ * 0.001)
        cold = (self.temp < self.tday - 1.0).float()
        self.cold_occupancy.mul_(0.999).add_(occ * cold * 0.001)
        self.warm_occupancy.mul_(0.999).add_(occ * (1 - cold) * 0.001)
        self.last_visit = torch.where(occ > 0, torch.full_like(self.last_visit, float(t)), self.last_visit)
        if t % 10 == 0:
            sh = self.struct_h
            has = sh > 0.02
            self.struct_age = torch.where(has, self.struct_age + 10, torch.zeros_like(self.struct_age))
            props = self.struct_mp / sh.clamp_min(1e-6).unsqueeze(0)
            hard, react, insul = props[1], props[4], props[2]
            rain = (self.precip / 0.5).clamp(0, 2)
            loss = sh * p.struct_decay * 10 * (1 - 0.9 * hard) * (1 + 2 * rain * react)
            self._remove_structure(loss.flatten() * has.flatten().float())
            wall = (self.struct_h / p.wall_height).clamp(0, 1) * insul.clamp(0, 1)
            enclosure = torch.stack(g.neighbors8(wall)).mean(0)
            inside = (self.struct_h < p.wall_height).float()
            self.shelter = ((enclosure * 1.25 + self.terrain.nat_shelter) * inside).clamp(0, 0.95)

    def _apply_external_source(self, src: dict) -> None:
        """A marked external field source (reality bridge / experiments). Only exists in forks."""
        c = int(src["cell"])
        amp = float(src.get("value", 0.0)) * float(src.get("gain", 1.0))
        i, j = divmod(c, self.grid.W)
        if src.get("field") == "light":
            self.light[i, j] = self.light[i, j] + amp
        elif src.get("field") == "vibration":
            self.sound[int(src.get("band", 0)), i, j] += amp

    # ============================================================ forks
    def fork(self, salt: int) -> "World":
        """An independent copy of this world (a fork). Observer buffers start empty;
        ``salt`` re-seeds the world's random streams (same salt ⇒ same noise)."""
        import copy

        focus, ring, plog = self.focus, self.ring, self.production_log
        soc = (self._soc_a, self._soc_b, self._soc_t)
        self.flush_social()
        self.focus, self.ring, self.production_log = None, None, deque(maxlen=200_000)
        try:
            w = copy.deepcopy(self)
        finally:
            self.focus, self.ring, self.production_log = focus, ring, plog
            self._soc_a, self._soc_b, self._soc_t = soc
        w.rng = StreamRNG(self.rng.seed, self.device, salt)
        w.focus = FocusRecorder(w, self.cfg.focus_agents, min(self.cfg.focus_ticks, 512))
        w.focus.slots = self.focus.slots.clone()
        w.focus.ids = self.focus.ids.copy()
        w.ring = RingBuffer(w, self.cfg.ring_frames)
        w.production_log = deque(list(plog)[-5000:], maxlen=200_000)
        w.agents.invalidate()
        return w

    # =================================================== observation API
    def summary(self) -> dict:
        A = self.agents
        act = A.active
        n = int(act.sum())
        x = self.xenos
        d = {"index": self.index, "planet": self.planet["id"], "fidelity": self.fidelity, "tick": self.tick,
             "day": self.tick // x.day_ticks, "year": self.tick // max(x.year_ticks, 1),
             "day_fraction": (self.tick % x.day_ticks) / x.day_ticks,
             "season": (self.tick % max(x.year_ticks, 1)) / max(x.year_ticks, 1),
             "population": n, "slots": A.N, "species_alive": sum(1 for s in self.species.species.values()
                                                                 if s["extinct"] is None),
             "materials": self.materials.n, "structure_cells": int((self.struct_h > 0.05).sum()),
             "wall_cells": int((self.struct_h >= self.phys.wall_height).sum()),
             "global_temp": round(self.climate.global_mean_temperature(), 2),
             "resource_total": round(float(self.R.sum()), 2), "counters": dict(self.counters),
             "seed_bank": self.bank.count(), "qd_coverage": round(self.qd.coverage(), 3),
             "subsolar": self.climate.subsolar_point(self.tick), "interventions": len(self.interventions),
             "ablations": sorted(self.ablations)}
        if n:
            d["mean_energy"] = round(float(A.energy[act].mean()), 3)
            d["mean_generation"] = round(float(A.generation[act].float().mean()), 2)
            d["max_generation"] = int(A.generation[act].max())
            d["mean_mass"] = round(float(A.caps[act, CAP["mass"]].mean()), 3)
            d["mean_units"] = round(float(A.caps[act, CAP["units"]].mean()), 2)
        observed = [m for iv in self.interventions if iv.get("op") == "observe" for m in iv.get("metrics", [])]
        if observed:
            d["observed"] = {k: v for k, v in self.metrics().items() if k in observed}
        return d

    def metrics(self) -> dict:
        """The closed metric vocabulary of the Scientist DSL (``observe`` steps), measured now."""
        A = self.agents
        n = A.count()
        c = self.counters
        acc = self.acc.tolist()
        return {"population": n, "mean_energy": round(float(A.energy[A.active].mean()), 4) if n else 0.0,
                "births": c["births"], "deaths": c["deaths"], "kills": c["kills"],
                "structure_cells": int((self.struct_h > 0.05).sum()), "materials": self.materials.n,
                "eaten": round(acc[0], 3), "emission_rate": round(acc[1] / max(self.tick, 1), 5),
                "species_alive": sum(1 for s in self.species.species.values() if s["extinct"] is None),
                "reactions": c["reactions"], "builds": c["builds"]}

    def agent_detail(self, agent_id: int) -> dict | None:
        A = self.agents
        s = A.slot_of(agent_id)
        rec = self.lineage.record(agent_id)
        if s < 0:
            return {"alive": False, "record": rec} if rec else None
        lat, lon = self.grid.latlon(A.pos[s])
        ph = A.phys
        caps = {k: round(float(A.caps[s, i]), 4) for k, i in CAP.items()}
        p = self.phys
        ecap = p.energy_cap_base + p.energy_cap_storage * float(A.caps[s, CAP["storage"]])
        from ..life.interface import LIFE_STATS, DIRECTIONS, MODALITIES, INPUT_NAMES, OUTPUT_NAMES
        stats = {k: round(float(A.stats[s, i]), 3) for i, k in enumerate(LIFE_STATS)}
        return {
            "alive": True, "id": int(A.id[s]), "slot": s, "species": int(A.species[s]),
            "lineage": int(A.lineage[s]), "parent": int(A.parent[s]), "generation": int(A.generation[s]),
            "birth_tick": int(A.birth[s]), "age": float(A.age[s]), "lifespan": float(ph["lifespan"][s]),
            "lat": math.degrees(float(lat)), "lon": math.degrees(float(lon)), "speed": float(A.speed[s]),
            "energy": float(A.energy[s]), "energy_cap": ecap, "integrity": float(A.integrity[s]),
            "body_temp": float(A.tbody[s]), "t_opt": float(ph["t_opt"][s]), "t_tol": float(ph["t_tol"][s]),
            "carried_material": int(A.carried_mat[s]), "carried_amount": float(A.carried_amt[s]),
            "caps": caps, "stats": stats,
            "sensors": {DIRECTIONS[dd]: {MODALITIES[mm]: round(float(A.sens[s, dd, mm]), 3) for mm in range(4)}
                        for dd in range(4)},
            "emit_profile": [round(v, 3) for v in A.emit_prof[s].tolist()],
            "substrate": {"units": int(A.caps[s, CAP["units"]]), "state": [round(v, 3) for v in A.x[s].tolist()],
                          "modulation": float(A.m[s]), "plasticity": float(A.dW[s]),
                          "inputs": dict(zip(INPUT_NAMES, [round(v, 3) for v in A.last_u[s].tolist()])),
                          "outputs": dict(zip(OUTPUT_NAMES, [round(v, 3) for v in A.last_y[s].tolist()]))},
            "record": rec, "ancestors": self.lineage.ancestors(agent_id, 8),
            "children": self.lineage.children(agent_id, 32),
            "social": self.social.neighbors(int(agent_id), 12),
            "focus": self.focus.is_focus(s),
        }

    def agent_body(self, agent_id: int) -> dict | None:
        s = self.agents.slot_of(agent_id)
        if s < 0:
            return None
        return self.dev.export_body(self.agents.genome[s], self.layout.hidden)

    def terrain_export(self) -> dict:
        """Static relief for the viewer (float16 elevation, ocean mask)."""
        return {"w": self.grid.W, "h": self.grid.H,
                "elev": self.terrain.elev_km.cpu().numpy().astype(np.float16),
                "ocean": self.ocean.cpu().numpy().astype(np.uint8)}

    def fields_export(self) -> dict:
        """Dynamic surface state, quantised to bytes (all real simulation fields)."""
        def q(t, lo, hi):
            return ((t.clamp(lo, hi) - lo) / (hi - lo) * 255).round().to(torch.uint8).cpu().numpy()

        sm = self.struct_mat.clamp(min=0)
        hard = self.materials.props[sm, 1] * (self.struct_mat >= 0).float()
        return {"w": self.grid.W, "h": self.grid.H,
                "temp": q(self.temp, -40, 50), "resource": q(self.R / (self.K + 0.05), 0, 1.2),
                "cloud": q(self.cloud, 0, 1), "moisture": q(self.precip, 0, 1.0),
                "compaction": q(self.compaction, 0, 1), "structure": q(self.struct_h, 0, 4),
                "material_hardness": q(hard, 0, 1), "occupancy": q(self.occupancy, 0, 2),
                "marks": q(self.marks.sum(0), 0, 2), "store": q(self.store, 0, 2),
                "light": q(self.light, 0, 1.3), "shelter": q(self.shelter, 0, 1)}

    def agents_frame(self) -> dict:
        A = self.agents
        act = torch.nonzero(A.active).flatten()
        lat, lon = self.grid.latlon(A.pos[act])
        hd = A.head[act]
        bits = getattr(self, "_last_bits", torch.zeros(A.N, dtype=torch.long, device=self.device))[act]
        emit = getattr(self, "_last_emit", torch.zeros(A.N, 4, device=self.device))[act].sum(1)
        p = self.phys
        ecap = p.energy_cap_base + p.energy_cap_storage * A.caps[act, CAP["storage"]]
        return {"id": A.id[act].cpu().numpy(), "lat": lat.cpu().numpy(), "lon": lon.cpu().numpy(),
                "head": hd.cpu().numpy(), "species": A.species[act].cpu().numpy(),
                "bits": bits.cpu().numpy(), "energy": (A.energy[act] / ecap).clamp(0, 1).cpu().numpy(),
                "mass": A.caps[act, CAP["mass"]].cpu().numpy(), "emit": emit.clamp(0, 1).cpu().numpy(),
                "aquatic": A.caps[act, CAP["aquatic"]].cpu().numpy()}

    def structure_cells(self, min_h: float = 0.05) -> dict:
        sh = self.struct_h.flatten()
        idx = torch.nonzero(sh > min_h).flatten()
        return {"cell": idx.cpu().numpy().astype(np.int32), "h": sh[idx].cpu().numpy().astype(np.float32),
                "mat": self.struct_mat.flatten()[idx].cpu().numpy().astype(np.int32),
                "age": self.struct_age.flatten()[idx].cpu().numpy().astype(np.float32)}

    # ======================================================== persistence
    def state_dict(self) -> dict:
        self.flush_social()
        return {
            "tick": self.tick, "next_agent_id": self.next_agent_id, "counters": dict(self.counters),
            "first_kill_tick": self.first_kill_tick, "interventions": list(self.interventions),
            "ablations": sorted(self.ablations), "external_sources": list(self.external_sources),
            "blocked_materials": sorted(self.blocked_materials), "param_scale": dict(self.param_scale),
            "max_births": self.max_births,
            "acc": self.acc,
            "fields": {k: getattr(self, k) for k in ("temp", "tday", "precip", "cloud", "light", "K", "R", "store",
                                                     "compaction", "marks", "sound", "scent", "struct_h",
                                                     "struct_mp", "struct_mat", "struct_age", "struct_builder",
                                                     "shelter", "occupancy", "cold_occupancy", "warm_occupancy",
                                                     "store_in", "store_out", "last_visit")},
            "terrain": self.terrain.state_dict(), "climate": self.climate.state_dict(),
            "agents": self.agents.state_dict(), "materials": self.materials.state_dict(),
            "species": self.species.state_dict(), "qd": self.qd.state_dict(), "lineage": self.lineage.state_dict(),
            "bank": self.bank.state_dict(), "social": self.social.state_dict(),
            "focus": self.focus.state_dict(),
            "production_log": list(self.production_log),
            "planet": self.planet, "xenos": self.xenos.to_dict(), "rng_salt": self.rng.salt,
        }

    def load_state_dict(self, d: dict) -> None:
        self.tick = int(d["tick"])
        self.next_agent_id = int(d["next_agent_id"])
        self.counters = dict(d["counters"])
        self.first_kill_tick = d["first_kill_tick"]
        self.interventions = list(d["interventions"])
        self.ablations = set(d["ablations"])
        self.external_sources = list(d.get("external_sources", []))
        self.blocked_materials = set(d.get("blocked_materials", []))
        self.param_scale = dict(d.get("param_scale", {}))
        self.max_births = int(d.get("max_births", self.max_births))
        if "acc" in d:
            self.acc = d["acc"].to(self.device).clone()
        for k, v in d["fields"].items():
            setattr(self, k, v.to(self.device).clone())
        self.terrain = Terrain.from_state({k: v.to(self.device).clone() for k, v in d["terrain"].items()})
        self.ocean = self.terrain.ocean
        self.oceanf = self.ocean.float().flatten()
        self.climate.load_state_dict(d["climate"])
        self.agents.load_state_dict(d["agents"])
        self.materials.load_state_dict(d["materials"])
        self.species.load_state_dict(d["species"])
        self.qd.load_state_dict(d["qd"])
        self.lineage.load_state_dict(d["lineage"])
        self.bank.load_state_dict(d["bank"])
        self.social.load_state_dict(d["social"])
        self.focus.load_state_dict(d["focus"])
        self.production_log = deque(d.get("production_log", []), maxlen=200_000)
        self.rng.salt = int(d.get("rng_salt", 0))


# ============================================================== F5 focus
class FocusRecorder:
    """F5 scientific-focus individuals: full internal recording every tick (spec §11-12)."""

    def __init__(self, world: World, k: int, ticks: int):
        self.w = world
        self.k = int(k)
        self.T = int(ticks)
        H = world.layout.hidden
        self.slots = torch.full((self.k,), -1, dtype=torch.long, device=world.device)
        self.ids = np.full(self.k, -1, dtype=np.int64)
        self.u = np.zeros((self.k, self.T, N_INPUTS), np.float32)
        self.x = np.zeros((self.k, self.T, H), np.float32)
        self.y = np.zeros((self.k, self.T, len(O)), np.float32)
        self.h = np.zeros((self.k, self.T, 8), np.float32)  # m, dW, energy, integrity, tbody-topt, shelter, t_env, dmg
        self.ctx = np.zeros((self.k, self.T, 3), np.float32)  # resource here, light here, crowd proxy
        self.valid = np.zeros((self.k, self.T), bool)
        self.ptr = 0
        self.ticks = np.zeros(self.T, np.int64)

    def is_focus(self, slot: int) -> bool:
        return bool((self.slots == slot).any())

    def refill(self) -> None:
        A = self.w.agents
        need = torch.nonzero(self.slots < 0).flatten()
        if need.numel() == 0:
            return
        taken = set(self.slots[self.slots >= 0].tolist())
        cand = [s for s in torch.nonzero(A.active).flatten().tolist() if s not in taken]
        gen = self.w.rng.np("focus", self.w.tick)
        gen.shuffle(cand)
        for j, s in zip(need.tolist(), cand):
            self.slots[j] = s
            self.ids[j] = int(A.id[s])
            self.valid[j] = False

    def on_deaths(self, dead_slots: torch.Tensor) -> None:
        if dead_slots.numel() == 0:
            return
        hit = (self.slots.unsqueeze(1) == dead_slots.unsqueeze(0)).any(1)
        if bool(hit.any()):
            self.slots[hit] = -1

    def record(self, u, x, y, m, dW, ecap, shelter, t_env, cell, dmg) -> None:
        if self.w.tick % 50 == 0:
            self.refill()
        s = self.slots
        ok = s >= 0
        ss = s.clamp(min=0)
        A = self.w.agents
        ph = A.phys
        hom = torch.stack([m[ss], dW[ss], A.energy[ss] / ecap[ss], A.integrity[ss], A.tbody[ss] - ph["t_opt"][ss],
                           shelter[ss], t_env[ss], dmg[ss]], 1)
        ctx = torch.stack([self.w.R.flatten()[cell[ss]], self.w.light.flatten()[cell[ss]],
                           A.last_u[ss, I["crowd"]]], 1)
        blob = torch.cat([u[ss], x[ss], y[ss], hom, ctx], 1).cpu().numpy()
        I_, H_ = u.shape[1], x.shape[1]
        p = self.ptr
        okn = ok.cpu().numpy()
        self.u[:, p] = blob[:, :I_]
        self.x[:, p] = blob[:, I_:I_ + H_]
        self.y[:, p] = blob[:, I_ + H_:I_ + H_ + len(O)]
        self.h[:, p] = blob[:, I_ + H_ + len(O):I_ + H_ + len(O) + 8]
        self.ctx[:, p] = blob[:, -3:]
        self.valid[:, p] = okn
        self.ticks[p] = self.w.tick
        self.ptr = (p + 1) % self.T

    def series(self, j: int) -> dict:
        """Chronological contiguous recording for focus individual j."""
        order = np.r_[self.ptr:self.T, 0:self.ptr]
        v = self.valid[j, order]
        return {"u": self.u[j, order][v], "x": self.x[j, order][v], "y": self.y[j, order][v],
                "h": self.h[j, order][v], "ctx": self.ctx[j, order][v], "ticks": self.ticks[order][v],
                "id": int(self.ids[j])}

    def state_dict(self) -> dict:
        return {"slots": self.slots, "ids": self.ids.copy()}

    def load_state_dict(self, d: dict) -> None:
        self.slots = d["slots"].to(self.w.device).clone()
        self.ids = np.asarray(d["ids"]).copy()
        self.valid[:] = False


# ============================================================== ring buffer
class RingBuffer:
    """Recent raw states at reduced rate (spec §280-284). Frozen around events."""

    def __init__(self, world: World, frames: int):
        self.w = world
        self.n = int(frames)
        self.frames: deque = deque(maxlen=self.n)

    def capture(self, tick: int) -> None:
        w = self.w
        A = w.agents
        act = torch.nonzero(A.active).flatten()
        if act.numel() == 0:
            self.frames.append({"tick": tick, "n": 0})
            return
        lat, lon = w.grid.latlon(A.pos[act])
        r_here, light_here, hurt, crowd = w._last_context
        emit = w._last_emit[act]
        frame = {
            "tick": tick, "n": int(act.numel()),
            "id": A.id[act].cpu().numpy(), "lat": lat.cpu().numpy().astype(np.float32),
            "lon": lon.cpu().numpy().astype(np.float32), "species": A.species[act].cpu().numpy().astype(np.int32),
            "bits": w._last_bits[act].cpu().numpy().astype(np.int32),
            "energy": A.energy[act].cpu().numpy().astype(np.float16),
            "emit": (emit * 255).clamp(0, 255).to(torch.uint8).cpu().numpy(),
            "turn": torch.tanh(A.last_y[act, O["turn"]]).cpu().numpy().astype(np.float16),
            "thrust": torch.sigmoid(A.last_y[act, O["thrust"]]).cpu().numpy().astype(np.float16),
            "r_here": r_here[act].cpu().numpy().astype(np.float16),
            "light": light_here[act].cpu().numpy().astype(np.float16),
            "hurt": hurt[act].cpu().numpy(), "crowd": crowd[act].cpu().numpy().astype(np.float16),
            "cell": w._last_cell[act].cpu().numpy().astype(np.int32),
            "recv": w.sound.view(w.cfg.emit_bands, -1)[:, w._last_cell[act]].t().cpu().numpy().astype(np.float16),
        }
        self.frames.append(frame)

    def window(self, since_tick: int | None = None) -> list[dict]:
        if since_tick is None:
            return list(self.frames)
        return [f for f in self.frames if f["tick"] >= since_tick]
