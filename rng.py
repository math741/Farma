"""Counter-based deterministic randomness.

Every random draw in Genesis is a pure function of ``(cosmos seed, world id,
stream name, tick)``. Nothing depends on hidden generator state, so a world
restored from a snapshot at tick *t* replays bit-identically on the same
hardware/software stack, and forks diverge only where an intervention or an
explicit re-seed (``salt``) makes them diverge.
"""

from __future__ import annotations

import hashlib

import numpy as np

MASK64 = (1 << 64) - 1


def splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def _part_to_int(p) -> int:
    if isinstance(p, (int, np.integer)):
        return int(p) & MASK64
    data = str(p).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def derive_seed(*parts) -> int:
    """Mix any number of ints/strings into a 63-bit non-negative seed."""
    h = 0x6A09E667F3BCC908
    for p in parts:
        h = splitmix64(h ^ _part_to_int(p))
    return h & ((1 << 63) - 1)


def np_rng(*parts) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(derive_seed(*parts)))


class StreamRNG:
    """Factory of per-tick torch generators for one world.

    ``gen("mutation", tick)`` always returns a generator in the same state for
    the same arguments. Creating generators is cheap compared with a tick.
    """

    def __init__(self, seed: int, device, salt: int = 0):
        self.seed = int(seed)
        self.salt = int(salt)
        self.device = device

    def seed_for(self, stream: str, tick: int, *extra) -> int:
        return derive_seed(self.seed, self.salt, stream, int(tick), *extra)

    def gen(self, stream: str, tick: int, *extra):
        import torch

        g = torch.Generator(device=self.device)
        g.manual_seed(self.seed_for(stream, tick, *extra))
        return g

    def np(self, stream: str, tick: int, *extra) -> np.random.Generator:
        return np.random.Generator(np.random.PCG64(self.seed_for(stream, tick, *extra)))
