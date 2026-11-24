"""SocialEngine — the social graph is built only from interaction history (spec §50-53).

Edges accumulate counts of physical interactions between persistent agent
IDs: co-location (sampled), energy transfers, strikes, mating and kinship.
Groups are found by weighted label propagation on the living subgraph and
their persistence is measured across windows — a group is never declared.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

EDGE_TYPES = ("proximity", "give", "strike", "mate", "kin")
T_PROX, T_GIVE, T_STRIKE, T_MATE, T_KIN = range(5)
WEIGHTS = np.array([1.0, 4.0, -3.0, 5.0, 2.0])


class SocialGraph:
    def __init__(self, max_edges: int = 400_000):
        self.max_edges = max_edges
        self.edges: dict[tuple[int, int], np.ndarray] = {}
        self.last: dict[tuple[int, int], int] = {}
        self.totals = np.zeros(len(EDGE_TYPES), dtype=np.int64)
        self.prev_groups: dict[int, int] = {}
        self.group_persistence: list[float] = []

    def ingest(self, a: np.ndarray, b: np.ndarray, types: np.ndarray, tick: int) -> None:
        if a.size == 0:
            return
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        keep = lo != hi
        lo, hi, types = lo[keep], hi[keep], types[keep]
        key = lo.astype(np.int64) * (1 << 31) + hi.astype(np.int64)
        comb = key * 8 + types
        uniq, counts = np.unique(comb, return_counts=True)
        for c, n in zip(uniq.tolist(), counts.tolist()):
            k, t = divmod(c, 8)
            pair = divmod(k, 1 << 31)
            e = self.edges.get(pair)
            if e is None:
                e = np.zeros(len(EDGE_TYPES), dtype=np.int32)
                self.edges[pair] = e
            e[t] += n
            self.last[pair] = tick
            self.totals[t] += n
        if len(self.edges) > self.max_edges:
            self._evict(tick)

    def _evict(self, tick: int) -> None:
        items = sorted(self.last.items(), key=lambda kv: kv[1])
        drop = len(self.edges) - int(self.max_edges * 0.8)
        for pair, _ in items[:drop]:
            self.edges.pop(pair, None)
            self.last.pop(pair, None)

    def neighbors(self, agent_id: int, limit: int = 24) -> list[dict]:
        out = []
        for (a, b), e in self.edges.items():
            if a == agent_id or b == agent_id:
                other = b if a == agent_id else a
                w = float((e * WEIGHTS).sum())
                out.append({"id": int(other), "weight": round(w, 2),
                            **{EDGE_TYPES[i]: int(e[i]) for i in range(len(EDGE_TYPES))}})
        out.sort(key=lambda r: -abs(r["weight"]))
        return out[:limit]

    def groups(self, alive: set[int], min_weight: float = 3.0, iters: int = 8) -> dict[int, int]:
        """Weighted label propagation over positive ties among living agents."""
        adj: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for (a, b), e in self.edges.items():
            if a in alive and b in alive:
                w = float((e * WEIGHTS).sum())
                if w >= min_weight:
                    adj[a].append((b, w))
                    adj[b].append((a, w))
        labels = {n: n for n in adj}
        nodes = sorted(adj)
        for _ in range(iters):
            changed = 0
            for n in nodes:
                score: dict[int, float] = defaultdict(float)
                for m, w in adj[n]:
                    score[labels[m]] += w
                if score:
                    best = max(sorted(score), key=lambda lab: score[lab])
                    if best != labels[n]:
                        labels[n] = best
                        changed += 1
            if not changed:
                break
        return labels

    def group_stats(self, alive: set[int]) -> dict:
        labels = self.groups(alive)
        sizes: dict[int, int] = defaultdict(int)
        for lab in labels.values():
            sizes[lab] += 1
        big = {lab for lab, n in sizes.items() if n >= 4}
        # persistence: Jaccard of co-membership between consecutive windows
        pers = 0.0
        if self.prev_groups and labels:
            common = [n for n in labels if n in self.prev_groups]
            if len(common) >= 4:
                same_now = same_prev = both = 0
                common = common[:400]
                for i in range(len(common)):
                    for j in range(i + 1, len(common)):
                        a, b = common[i], common[j]
                        sn = labels[a] == labels[b]
                        sp = self.prev_groups[a] == self.prev_groups[b]
                        same_now += sn
                        same_prev += sp
                        both += sn and sp
                union = same_now + same_prev - both
                pers = both / union if union else 0.0
        self.prev_groups = labels
        self.group_persistence.append(pers)
        self.group_persistence = self.group_persistence[-50:]
        return {"n_groups": len(big), "largest": max(sizes.values()) if sizes else 0,
                "grouped_agents": sum(sizes[lab] for lab in big), "persistence": round(pers, 3),
                "edges": len(self.edges), "totals": {EDGE_TYPES[i]: int(self.totals[i]) for i in range(5)}}

    def state_dict(self) -> dict:
        keys = np.array(list(self.edges.keys()), dtype=np.int64).reshape(-1, 2)
        vals = np.array(list(self.edges.values()), dtype=np.int32).reshape(-1, len(EDGE_TYPES))
        last = np.array([self.last[k] for k in self.edges.keys()], dtype=np.int64)
        return {"keys": keys, "vals": vals, "last": last, "totals": self.totals.copy(),
                "persistence": list(self.group_persistence)}

    def load_state_dict(self, d: dict) -> None:
        keys, vals, last = d["keys"], d["vals"], d["last"]
        self.edges = {(int(k[0]), int(k[1])): np.array(v, dtype=np.int32) for k, v in zip(keys, vals)}
        self.last = {(int(k[0]), int(k[1])): int(t) for k, t in zip(keys, last)}
        self.totals = np.asarray(d["totals"], dtype=np.int64).copy()
        self.group_persistence = list(d.get("persistence", []))
