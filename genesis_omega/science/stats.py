"""Statistics used by the Scientist (spec §230-233, §399-405).

Small, dependency-free estimators: plug-in mutual information with
Miller–Madow bias correction, conditional MI over confound strata,
permutation tests, cross-validated logistic/ridge decoders and binomial
tests. All randomness is seeded by the caller.
"""

from __future__ import annotations

import math

import numpy as np


def entropy_counts(counts: np.ndarray) -> float:
    n = counts.sum()
    if n <= 0:
        return 0.0
    p = counts[counts > 0] / n
    return float(-(p * np.log2(p)).sum())


def mutual_information(x: np.ndarray, y: np.ndarray, kx: int | None = None, ky: int | None = None,
                       correct: bool = True) -> float:
    """I(X;Y) in bits for integer-coded arrays (Miller–Madow corrected, floored at 0)."""
    x = np.asarray(x, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    n = x.size
    if n == 0:
        return 0.0
    kx = int(x.max()) + 1 if kx is None else kx
    ky = int(y.max()) + 1 if ky is None else ky
    joint = np.bincount(x * ky + y, minlength=kx * ky).reshape(kx, ky).astype(np.float64)
    hx = entropy_counts(joint.sum(1))
    hy = entropy_counts(joint.sum(0))
    hxy = entropy_counts(joint.ravel())
    mi = hx + hy - hxy
    if correct:
        bx = (joint.sum(1) > 0).sum()
        by = (joint.sum(0) > 0).sum()
        bxy = (joint > 0).sum()
        mi -= (bx + by - bxy - 1) / (2 * n * math.log(2))
    return max(0.0, float(mi))


def conditional_mi(x, y, z, kx=None, ky=None) -> float:
    """I(X;Y|Z) = Σ_z p(z) I(X;Y|Z=z)."""
    x, y, z = (np.asarray(a, dtype=np.int64) for a in (x, y, z))
    n = x.size
    if n == 0:
        return 0.0
    kx = int(x.max()) + 1 if kx is None else kx
    ky = int(y.max()) + 1 if ky is None else ky
    tot = 0.0
    for zv in np.unique(z):
        m = z == zv
        if m.sum() < 8:
            continue
        tot += m.sum() / n * mutual_information(x[m], y[m], kx, ky)
    return tot


def permutation_test(stat_fn, x, y, z, n_perm: int, rng: np.random.Generator) -> tuple[float, float, np.ndarray]:
    """Shuffle x within strata of z; returns (observed, p-value, null distribution)."""
    obs = stat_fn(x, y, z)
    null = np.empty(n_perm)
    x = np.asarray(x)
    z = np.asarray(z)
    strata = [np.where(z == v)[0] for v in np.unique(z)]
    for i in range(n_perm):
        xp = x.copy()
        for idx in strata:
            xp[idx] = x[idx][rng.permutation(idx.size)]
        null[i] = stat_fn(xp, y, z)
    p = (1 + (null >= obs).sum()) / (n_perm + 1)
    return float(obs), float(p), null


def js_similarity(p: np.ndarray, q: np.ndarray) -> float:
    """1 - Jensen-Shannon divergence (base 2) between two distributions."""
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-12))).sum())

    return 1.0 - 0.5 * (kl(p, m) + kl(q, m))


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X ≥ k) for X ~ Binomial(n, p) (exact, log-space)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    logs = []
    lp, lq = math.log(max(p, 1e-300)), math.log(max(1 - p, 1e-300))
    for i in range(k, n + 1):
        logs.append(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
    mx = max(logs)
    return float(min(1.0, math.exp(mx) * sum(math.exp(v - mx) for v in logs)))


def standardize(X: np.ndarray, mu=None, sd=None):
    mu = X.mean(0) if mu is None else mu
    sd = X.std(0) + 1e-6 if sd is None else sd
    return (X - mu) / sd, mu, sd


def logistic_fit(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 300, lr: float = 0.5) -> np.ndarray:
    """Binary logistic regression by gradient descent (X already standardised)."""
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    for _ in range(iters):
        z = np.clip(Xb @ w, -30, 30)
        p = 1 / (1 + np.exp(-z))
        g = Xb.T @ (p - y) / n
        g[:-1] += l2 * w[:-1]
        w -= lr * g
    return w


def logistic_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return (Xb @ w > 0).astype(np.int64)


def cv_decode_accuracy(X: np.ndarray, y: np.ndarray, folds: int, rng: np.random.Generator) -> float:
    """Cross-validated accuracy of a linear decoder for a binary label."""
    n = X.shape[0]
    if n < folds * 2 or len(np.unique(y)) < 2:
        return 0.5
    idx = rng.permutation(n)
    parts = np.array_split(idx, folds)
    correct = 0
    for k in range(folds):
        te = parts[k]
        tr = np.concatenate([parts[j] for j in range(folds) if j != k])
        Xtr, mu, sd = standardize(X[tr])
        Xte, _, _ = standardize(X[te], mu, sd)
        w = logistic_fit(Xtr, y[tr].astype(np.float64))
        correct += int((logistic_predict(Xte, w) == y[te]).sum())
    return correct / n


def ridge_cv_r2(X: np.ndarray, y: np.ndarray, folds: int, rng: np.random.Generator, lam: float = 1.0) -> float:
    """Cross-validated R² of ridge regression."""
    n = X.shape[0]
    if n < folds * 3:
        return 0.0
    idx = rng.permutation(n)
    parts = np.array_split(idx, folds)
    ss_res = ss_tot = 0.0
    for k in range(folds):
        te = parts[k]
        tr = np.concatenate([parts[j] for j in range(folds) if j != k])
        Xtr, mu, sd = standardize(X[tr])
        Xte, _, _ = standardize(X[te], mu, sd)
        ym = y[tr].mean()
        A = Xtr.T @ Xtr + lam * np.eye(X.shape[1])
        w = np.linalg.solve(A, Xtr.T @ (y[tr] - ym))
        pred = Xte @ w + ym
        ss_res += float(((y[te] - pred) ** 2).sum())
        ss_tot += float(((y[te] - y[tr].mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def mean_ci(values: np.ndarray, z: float = 1.96) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return 0.0, 0.0, 0.0
    m = float(v.mean())
    se = float(v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else 0.0
    return m, m - z * se, m + z * se


def welch_t(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch's t statistic and a normal-approximation two-sided p-value."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.size < 2 or b.size < 2:
        return 0.0, 1.0
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    t = (a.mean() - b.mean()) / math.sqrt(max(va + vb, 1e-12))
    p = math.erfc(abs(t) / math.sqrt(2))
    return float(t), float(p)
