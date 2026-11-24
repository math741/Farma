"""SignalEngine / LearningEngine — the processing substrate (spec §26-31, §38).

A batched leaky recurrent network per organism with primitives the spec asks
for: emission and propagation (weighted signals), delay and memory (per-unit
time constants), amplification and inhibition (signed weights), oscillation
(recurrence + leak), plasticity (an evolvable local rule) and modulation
(a global broadcast computed by two opponent units).

How much substrate an organism has is set by its body (conductive nodes);
inactive units are masked. Nothing here is called a neuron or a brain, and no
learning signal is supplied by the programmer: whatever drives the
modulatory broadcast — homeostatic error, a sensory cue, nothing — is wired by
evolution. Plasticity:

    ΔW = η · m · (A·post⊗pre + B·pre + C·post + D) − λ (W − W₀)
"""

from __future__ import annotations

import torch

from .genome import GenomeLayout


class SubstrateParams:
    """Decoded, per-organism constant parameters (views into the genome)."""

    def __init__(self, layout: GenomeLayout, genomes: torch.Tensor, units: torch.Tensor):
        L = layout
        H = L.hidden
        self.W0 = L.get(genomes, "br_w")                    # (N,H,I+H)
        self.b = L.get(genomes, "br_b")
        self.alpha = 0.05 + 0.95 * torch.sigmoid(L.get(genomes, "br_tau"))
        self.Wo = L.get(genomes, "br_wo")                   # (N,O,H)
        self.bo = L.get(genomes, "br_bo")
        self.A = L.get(genomes, "pl_A")
        self.B = L.get(genomes, "pl_B")
        self.C = L.get(genomes, "pl_C")
        self.D = L.get(genomes, "pl_D")
        self.eta = 0.02 * torch.sigmoid(L.get(genomes, "pl_eta"))
        self.gain_mod = torch.tanh(L.mod(genomes, "gain_mod"))
        self.explore_mod = torch.tanh(L.mod(genomes, "explore_mod"))
        self.noise_base = 0.12 * torch.sigmoid(L.mod(genomes, "noise_base"))
        self.decay = 0.002 * torch.sigmoid(L.mod(genomes, "decay"))
        ar = torch.arange(H, device=genomes.device).view(1, H)
        self.mask = (ar < units.view(-1, 1)).float()        # body-determined substrate size


def substrate_step(p: SubstrateParams, W: torch.Tensor, x: torch.Tensor, m_prev: torch.Tensor, u: torch.Tensor,
                   noise: torch.Tensor | None, plastic: bool = True, clamp_dir: torch.Tensor | None = None,
                   recurrent_scale: float = 1.0, consolidate: bool = False,
                   clamp_mu: torch.Tensor | None = None):
    """One update of every organism's substrate.

    Args:
        p: decoded parameters. W: (N,H,I+H) current plastic weights. x: (N,H) state.
        m_prev: (N,) previous modulatory broadcast. u: (N,I) raw inputs.
        noise: (N,H) standard normal or None. plastic: apply the plasticity rule.
        clamp_dir: optional (N,H) unit directions whose projection is clamped (causal
            ablation of a candidate latent, used only by the Scientist); the projection is
            held at ``clamp_mu`` (N,) if given (fluctuation ablation), else at 0.
        recurrent_scale: 0 ablates recurrence and leak memory (Scientist memory ablation).
        consolidate: apply the (batched, every few ticks) decay toward W0 and weight bound.
    Returns: (W_new, x_new, y, m, dW_norm)

    The plasticity rule factorises as ΔW = a ⊗ z + c·1ᵀ with
    a = η m (A·post + B) and c = η m (C·post + D), so it is applied as one fused
    batched outer-product update.
    """
    gain = (1.0 + p.gain_mod * m_prev).clamp(0.2, 3.0).unsqueeze(1)
    ug = u * gain
    xr = x * recurrent_scale
    z = torch.cat([ug, xr], 1)                                   # (N, I+H)
    pre = torch.bmm(W, z.unsqueeze(2)).squeeze(2) + p.b
    if noise is not None:
        sigma = p.noise_base * (1.0 + p.explore_mod * m_prev).clamp(0.0, 3.0)
        pre = pre + noise * sigma.unsqueeze(1)
    alpha = p.alpha if recurrent_scale > 0 else torch.ones_like(p.alpha)
    x_new = ((1 - alpha) * xr + alpha * torch.tanh(pre)) * p.mask
    if clamp_dir is not None:
        proj = (x_new * clamp_dir).sum(1, keepdim=True)
        if clamp_mu is not None:
            proj = proj - clamp_mu.view(-1, 1)
        x_new = x_new - proj * clamp_dir
    m = torch.tanh(2.0 * (x_new[:, 0] - x_new[:, 1]))
    y = torch.bmm(p.Wo, x_new.unsqueeze(2)).squeeze(2) + p.bo
    if plastic:
        coef = p.eta * m.unsqueeze(1) * p.mask                  # (N,H)
        a = coef * (p.A * x_new + p.B)
        c = coef * (p.C * x_new + p.D)
        W_new = torch.baddbmm(W + c.unsqueeze(2), a.unsqueeze(2), z.unsqueeze(1))
        if consolidate:
            W_new = W_new + (8.0 * p.decay).view(-1, 1, 1) * (p.W0 - W_new)
            W_new = W_new.clamp_(-4.0, 4.0)
        dW_norm = a.abs().mean(1) * z.abs().mean(1) + c.abs().mean(1)
    else:
        W_new = W
        dW_norm = torch.zeros_like(m)
    return W_new, x_new, y, m, dW_norm
