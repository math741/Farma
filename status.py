"""Epistemic tags and the feature status registry (spec §3, §525-530).

Two separate axes are tracked everywhere in Genesis:

* :class:`Epistemic` — *what kind of statement* something is
  (simulated state, programmed law, measured quantity, inferred label …).
* :class:`Validation` — *how much evidence* supports a scientific claim.

Software features additionally carry a :class:`FeatureStatus`
(NOT IMPLEMENTED / UNVALIDATED / TESTED). The registry below is the single
source of truth the viewer displays; it never claims more than exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum


class Epistemic(str, Enum):
    SIMULATED = "SIMULATED"        # direct state of the simulation
    PROGRAMMED = "PROGRAMMED"      # a law / initial condition written by the programmer
    EMERGENT = "EMERGENT"          # pattern not explicitly coded, produced by dynamics
    MEASURED = "MEASURED"          # output of an external measurement / probe
    INFERRED = "INFERRED"          # label assigned by an observer-side detector
    HYPOTHESIS = "HYPOTHESIS"      # candidate interpretation awaiting tests
    EXTERNAL = "EXTERNAL"          # caused by an intervention from outside the cosmos


class Validation(str, Enum):
    NOT_VALIDATED = "NOT VALIDATED"
    CANDIDATE = "CANDIDATE"
    TESTED = "TESTED"
    REPLICATED = "REPLICATED"
    CAUSALLY_SUPPORTED = "CAUSALLY SUPPORTED"
    REJECTED = "REJECTED"

    @property
    def rank(self) -> int:
        order = {
            Validation.REJECTED: -1,
            Validation.NOT_VALIDATED: 0,
            Validation.CANDIDATE: 1,
            Validation.TESTED: 2,
            Validation.REPLICATED: 3,
            Validation.CAUSALLY_SUPPORTED: 4,
        }
        return order[self]


class FeatureStatus(str, Enum):
    NOT_IMPLEMENTED = "NOT IMPLEMENTED"
    UNVALIDATED = "UNVALIDATED"      # implemented, no automated verification yet
    TESTED = "TESTED"                # covered by automated tests


@dataclass
class Feature:
    key: str
    name: str
    phase: str
    status: FeatureStatus
    notes: str = ""
    tests: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


NI = FeatureStatus.NOT_IMPLEMENTED
UV = FeatureStatus.UNVALIDATED
TE = FeatureStatus.TESTED


def _f(key, name, phase, status, notes="", tests=None) -> Feature:
    return Feature(key, name, phase, status, notes, list(tests or []))


#: The honest map of what exists. Tests named here live in ``tests/``.
FEATURES: list[Feature] = [
    # Ω0 — cosmos core
    _f("cosmos.galaxy", "Procedural galaxy (spiral disk, IMF, stellar stages, remnants, central black hole)", "Ω0", TE,
       tests=["test_cosmos.py"]),
    _f("cosmos.systems", "Planetary systems (rocky, gas/ice giants, moons, belts, habitable zone)", "Ω0", TE,
       tests=["test_cosmos.py"]),
    _f("cosmos.f0_life", "F0 aggregate biospheres on unfocused life-bearing worlds", "Ω0", TE,
       notes="Population-field model only; no individuals exist at F0.", tests=["test_cosmos.py"]),
    _f("world.terrain", "Spherical terrain (fractal noise, oceans, deposits)", "Ω0", TE, tests=["test_world.py"]),
    _f("world.climate", "Energy-balance climate on a coarse grid (day/night, seasons, moisture, clouds)", "Ω0", TE,
       tests=["test_world.py"]),
    _f("world.persistence", "Atomic checkpoints, backups, crash recovery, no silent reset", "Ω0", TE,
       tests=["test_checkpoint.py", "test_runtime.py"]),
    _f("viewer.3d", "3D observatory (galaxy → system → planet → agent), WebGL2", "Ω0", TE,
       notes="Protocol and boot are tested headlessly; visual quality is not machine-verified.",
       tests=["test_server.py", "test_viewer_headless.py"]),
    _f("viewer.webgpu", "WebGPU renderer", "Ω0", NI,
       notes="Viewer reports WebGPU availability but renders with WebGL2."),
    # Ω1 — embodied organisms
    _f("life.development", "Developmental genome → morphogen lattice → body (no predefined tissues)", "Ω1", TE,
       tests=["test_life.py"]),
    _f("life.body_physics", "Body-derived locomotion, sensing, emission, grasp, armor, insulation", "Ω1", TE,
       tests=["test_life.py"]),
    _f("life.evolution", "Reproduction, self-adaptive mutation, crossover, speciation, seed bank", "Ω1", TE,
       tests=["test_world.py"]),
    _f("life.qd", "MAP-Elites archive, novelty / weirdness archive (observer-side)", "Ω1", TE,
       tests=["test_world.py"]),
    # Ω2 — learning / memory / affect
    _f("mind.substrate", "Leaky recurrent signal substrate with evolvable neuromodulated plasticity", "Ω2", TE,
       tests=["test_life.py"]),
    _f("mind.learning_probe", "Lifetime-learning probe (plastic vs frozen, randomized contingency)", "Ω2", TE,
       tests=["test_science.py"]),
    _f("mind.memory_probe", "Memory probe (delays, decoders, shuffled control, ablation)", "Ω2", TE,
       tests=["test_science.py"]),
    _f("mind.self_world", "Self/world differentiation probe (yoked control)", "Ω2", TE, tests=["test_science.py"]),
    _f("mind.body_schema", "Body-schema probe (consequence-violation response vs external jump)", "Ω2", TE,
       tests=["test_science.py"]),
    _f("affect.A0_A4", "Functional-affect pipeline levels A0–A4 + causal ablation", "Ω2", TE,
       notes="Synthetic positive/negative controls only validate the pipeline, never a claim.",
       tests=["test_science.py"]),
    _f("affect.A5_A9", "Affect levels A5–A9 (context, anticipation, learned, social, architecture)", "Ω2", NI),
    # Ω3 — society / communication / culture
    _f("society.graph", "Social graph from interaction history (proximity, gifts, strikes, mating)", "Ω3", TE,
       tests=["test_world.py"]),
    _f("society.communication", "Vibration channels + information/causal/stability/social-use metrics", "Ω3", TE,
       tests=["test_science.py"]),
    _f("society.compositionality", "Signal compositionality test", "Ω3", NI),
    _f("society.social_learning", "Imitation / teaching probes", "Ω3", NI),
    _f("society.culture_lineage", "Cultural lineage of material techniques (who produced what after whom)", "Ω3", TE,
       tests=["test_world.py"]),
    _f("society.marks", "Persistent marks (external symbol substrate) + writing-like detector", "Ω3", UV),
    # Ω4 — construction / settlement / technology
    _f("build.physics", "Digging, placing, structure decay, shelter, storage, trail compaction", "Ω4", TE,
       tests=["test_construction.py"]),
    _f("build.detectors", "House-like / storage-like / village-like / city-like / trail detectors", "Ω4", TE,
       tests=["test_construction.py"]),
    _f("tech.chemistry", "Open-ended hidden material chemistry; technology graph from real events", "Ω4", TE,
       tests=["test_construction.py"]),
    # Ω5 — civilization / science / history
    _f("civ.levels", "Civilization levels C0–C8 assessed from evidence", "Ω5", TE, tests=["test_world.py"]),
    _f("science.agent_science", "Detection of agent experimentation (S1)", "Ω5", UV),
    _f("science.S2_S5", "Agent predictive theories / engineering from model (S2–S5)", "Ω5", NI),
    _f("history.events", "Historical event engine with candidate causes; news since last visit", "Ω5", TE,
       tests=["test_world.py", "test_server.py"]),
    _f("history.forks", "Prime timeline protection, scientific forks, replay copies", "Ω5", TE,
       tests=["test_runtime.py"]),
    _f("history.replay", "Deterministic replay of frozen discovery windows", "Ω5", TE, tests=["test_runtime.py"]),
    # Ω6-Ω8 — space
    _f("space.orbits", "Keplerian orbits for system view", "Ω6", TE, tests=["test_cosmos.py"]),
    _f("space.spaceflight", "Spaceflight engine (launch, orbit, transfer)", "Ω6", NI,
       notes="No launch physics exists yet; no spaceflight event can be emitted."),
    _f("space.multiplanet", "Multiple simultaneous F4 worlds with different pressures (xenos)", "Ω6", TE,
       tests=["test_runtime.py"]),
    _f("contact.first_contact", "Detection / first contact between civilizations", "Ω7", NI),
    _f("contact.expedition", "Expedition (transplant foreign lineages into a marked fork)", "Ω7", TE,
       tests=["test_runtime.py"]),
    _f("galaxy.history", "Galactic timeline (F0 impacts, blooms, collapses, promotions)", "Ω8", TE,
       tests=["test_cosmos.py"]),
    # Ω9-Ω10
    _f("observer.X1", "Intervention-detection probe (observer level X1)", "Ω9", NI),
    _f("bridge.reality", "Reality bridge (real sensors → marked fork field source)", "Ω10", UV,
       notes="Viewer can stream microphone RMS (scalar only, local processing) into a marked fork."),
    _f("xr.webxr", "WebXR immersive-vr / immersive-ar (portal) session", "Ω0", UV,
       notes="Not verified on a Quest 3S device."),
    _f("xr.robots", "Physical robot avatars", "Ω10", NI),
    _f("audio.sonification", "Planetary ambience + signal sonification (not speech)", "Ω0", UV),
    _f("ops.autotune", "Hardware detection + measured auto-tuning", "Ω0", TE, tests=["test_runtime.py"]),
    _f("ops.oom", "OOM recovery (reduce science batch, then active fidelity, logged)", "Ω0", TE,
       tests=["test_runtime.py"]),
    _f("ops.remote", "HTTPS tunnel, token auth (read-only vs admin)", "Ω0", TE,
       notes="Tunnel binary download not exercised in tests.", tests=["test_server.py"]),
    _f("ops.security", "Agents only touch the world interface; Scientist DSL rejects code", "Ω0", TE,
       tests=["test_security.py"]),
    _f("ops.torch_compile", "torch.compile of the substrate step (benchmarked, opt-in)", "Ω0", UV),
    _f("ops.triton", "Custom Triton kernels", "Ω0", NI, notes="Not needed: no measured bottleneck requires them."),
    _f("fidelity.F0_F4_F5", "Fidelity levels F0 (aggregate), F4 (developmental), F5 (scientific focus)", "Ω0", TE,
       tests=["test_world.py"]),
    _f("fidelity.F1_F3", "Intermediate fidelity tiers F1–F3", "Ω0", NI,
       notes="Every individual in an active world currently runs at F4."),
]


def feature_table() -> list[dict]:
    return [f.to_dict() for f in FEATURES]


def feature(key: str) -> Feature:
    for f in FEATURES:
        if f.key == key:
            return f
    raise KeyError(key)
