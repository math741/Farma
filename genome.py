"""Developmental genome (spec §18-19, §29-30, §42).

A genome is a flat float vector with named segments:

* ``dev.*``   — weights of a gene-regulatory network run by every cell of the
                developing lattice (growth, differentiation, morphogens);
* ``br.*``    — initial weights / time constants of the signal substrate;
* ``pl.*``    — coefficients of the evolvable plasticity rule;
* ``mod``     — how the global modulatory broadcast gates gain / exploration;
* ``phys``    — physiology: thermal optimum and tolerance, lifespan,
                reproductive investment, chemical signature, mutation rate.

The genome encodes rules, never the final anatomy: bodies are grown.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .interface import N_INPUTS, N_OUTPUTS, O

PHYS_GENES = ("t_opt", "t_tol", "lifespan", "repro_invest", "scent0", "scent1", "log_sigma", "sex", "mut_frac",
              "reserve")
MOD_GENES = ("gain_mod", "explore_mod", "noise_base", "decay")


@dataclass
class GenomeLayout:
    hidden: int
    dev_channels: int
    dev_hidden: int

    def __post_init__(self):
        H, C, Hd = self.hidden, self.dev_channels, self.dev_hidden
        self.n_in = N_INPUTS
        self.n_out = N_OUTPUTS
        segs = [
            ("dev_w1", (Hd, C * 4)), ("dev_b1", (Hd,)), ("dev_w2", (C, Hd)), ("dev_b2", (C,)),
            ("br_w", (H, N_INPUTS + H)), ("br_b", (H,)), ("br_tau", (H,)), ("br_wo", (N_OUTPUTS, H)),
            ("br_bo", (N_OUTPUTS,)),
            ("pl_A", (H,)), ("pl_B", (H,)), ("pl_C", (H,)), ("pl_D", (H,)), ("pl_eta", (H,)),
            ("mod", (len(MOD_GENES),)), ("phys", (len(PHYS_GENES),)),
        ]
        self.slices: dict[str, tuple[int, int, tuple]] = {}
        o = 0
        for name, shape in segs:
            n = int(math.prod(shape))
            self.slices[name] = (o, o + n, shape)
            o += n
        self.size = o

    def get(self, g: torch.Tensor, name: str) -> torch.Tensor:
        a, b, shape = self.slices[name]
        return g[:, a:b].reshape(g.shape[0], *shape)

    def segment_bounds(self) -> list[tuple[int, int]]:
        """Crossover blocks: each segment, with the big weight matrices split by rows."""
        out = []
        for name, (a, b, shape) in self.slices.items():
            if len(shape) == 2 and shape[0] > 1:
                row = shape[1]
                for r in range(shape[0]):
                    out.append((a + r * row, a + (r + 1) * row))
            else:
                out.append((a, b))
        return out

    def phys(self, g: torch.Tensor, name: str) -> torch.Tensor:
        a, _, _ = self.slices["phys"]
        return g[:, a + PHYS_GENES.index(name)]

    def mod(self, g: torch.Tensor, name: str) -> torch.Tensor:
        a, _, _ = self.slices["mod"]
        return g[:, a + MOD_GENES.index(name)]

    def to_dict(self) -> dict:
        return {"hidden": self.hidden, "dev_channels": self.dev_channels, "dev_hidden": self.dev_hidden,
                "size": self.size, "segments": {k: [v[0], v[1], list(v[2])] for k, v in self.slices.items()}}


# ------------------------------------------------------------- physiology
def decode_physiology(layout: GenomeLayout, g: torch.Tensor, lifespan_days: float, day_ticks: int) -> dict:
    t_opt = 12.0 + 16.0 * torch.tanh(layout.phys(g, "t_opt"))
    t_tol = 4.0 + 14.0 * torch.sigmoid(layout.phys(g, "t_tol"))
    life = lifespan_days * day_ticks * (0.5 + 1.5 * torch.sigmoid(layout.phys(g, "lifespan")))
    invest = 0.2 + 0.45 * torch.sigmoid(layout.phys(g, "repro_invest"))
    scent = torch.tanh(torch.stack([layout.phys(g, "scent0"), layout.phys(g, "scent1")], 1))
    sigma = torch.exp(layout.phys(g, "log_sigma")).clamp(0.004, 0.25)
    sex = torch.sigmoid(layout.phys(g, "sex"))
    mut_frac = 0.03 + 0.3 * torch.sigmoid(layout.phys(g, "mut_frac"))
    return {"t_opt": t_opt, "t_tol": t_tol, "lifespan": life, "invest": invest, "scent": scent, "sigma": sigma,
            "sex": sex, "mut_frac": mut_frac}


# ----------------------------------------------------------- founders
def proto_genome(layout: GenomeLayout, gen: torch.Generator, device) -> torch.Tensor:
    """A random founder genome with mild structural priors (PROGRAMMED initial
    condition): small random development rules biased toward growth, random
    substrate wiring, and output biases that make feeding and reproduction
    initially likely and every other effector initially rare."""
    G = layout.size
    g = torch.zeros(1, G, device=device)

    def fill(name, std, mean=0.0):
        a, b, _ = layout.slices[name]
        g[:, a:b] = mean + std * torch.randn(1, b - a, generator=gen, device=device)

    H = layout.hidden
    fill("dev_w1", 0.35)
    fill("dev_b1", 0.1)
    fill("dev_w2", 0.25)
    fill("dev_b2", 0.15)
    a, _, _ = layout.slices["dev_b2"]
    g[:, a] += 0.35                      # alive channel: tendency to grow
    fill("br_w", 1.0 / math.sqrt(N_INPUTS + H))
    fill("br_b", 0.1)
    fill("br_tau", 1.0, 0.6)
    fill("br_wo", 1.2 / math.sqrt(H))
    a, _, _ = layout.slices["br_bo"]
    bias = torch.full((N_OUTPUTS,), -1.6, device=device)
    bias[O["turn"]] = 0.0
    bias[O["thrust"]] = 0.2
    bias[O["eat"]] = 1.2
    bias[O["reproduce"]] = 1.0
    for k in ("emit0", "emit1", "emit2", "emit3"):
        bias[O[k]] = -1.2
    g[:, a:a + N_OUTPUTS] = bias + 0.3 * torch.randn(N_OUTPUTS, generator=gen, device=device)
    for k in ("pl_A", "pl_B", "pl_C", "pl_D"):
        fill(k, 0.5)
    fill("pl_eta", 0.5, -2.0)
    fill("mod", 0.4)
    a, _, _ = layout.slices["phys"]
    phys = torch.tensor([0.0, 0.6, 0.0, 0.0, 0.0, 0.0, math.log(0.03), -1.0, -0.5, 0.0], device=device)
    g[:, a:a + len(PHYS_GENES)] = phys
    g[:, a + PHYS_GENES.index("scent0")] = torch.randn(1, generator=gen, device=device)
    g[:, a + PHYS_GENES.index("scent1")] = torch.randn(1, generator=gen, device=device)
    g[:, a + PHYS_GENES.index("t_opt")] = 0.3 * torch.randn(1, generator=gen, device=device)
    return g


# ------------------------------------------------------------ variation
def mutate(layout: GenomeLayout, parents: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Self-adaptive Gaussian mutation: each child's step size is the parent's
    log-normally perturbed; a genome-encoded fraction of genes is changed."""
    n, G = parents.shape
    dev = parents.device
    a_ls = layout.slices["phys"][0] + PHYS_GENES.index("log_sigma")
    a_mf = layout.slices["phys"][0] + PHYS_GENES.index("mut_frac")
    child = parents.clone()
    child[:, a_ls] = child[:, a_ls] + 0.22 * torch.randn(n, generator=gen, device=dev)
    child[:, a_ls] = child[:, a_ls].clamp(math.log(0.004), math.log(0.25))
    sigma = torch.exp(child[:, a_ls]).unsqueeze(1)
    frac = (0.03 + 0.3 * torch.sigmoid(child[:, a_mf])).unsqueeze(1)
    mask = torch.rand(n, G, generator=gen, device=dev) < frac
    noise = torch.randn(n, G, generator=gen, device=dev) * sigma
    child = child + noise * mask
    return child


def crossover(layout: GenomeLayout, a: torch.Tensor, b: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Block-uniform crossover (whole segments / weight rows inherited together)."""
    bounds = layout.segment_bounds()
    n = a.shape[0]
    pick = torch.rand(n, len(bounds), generator=gen, device=a.device) < 0.5
    child = a.clone()
    lens = torch.tensor([e - s for s, e in bounds], device=a.device)
    block_of_gene = torch.repeat_interleave(torch.arange(len(bounds), device=a.device), lens)
    take_b = pick[:, block_of_gene]
    child[take_b] = b[take_b]
    return child


def genome_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """RMS per-gene difference."""
    return ((a - b) ** 2).mean(-1).sqrt()
