"""Predicting the size of the one-round drift (Fig S1).
(i)  closed form for small heterogeneity:  d b = n * eta * s^2 * phi(b) * (pi - q) / (pi (1 - pi))
(ii) pair mean-update approximation: iterate a deterministic expected update for aligned boundaries. This ignores orientation differences and is not the exact expectation of the nonlinear multi-step process."""
import json, sys, numpy as np
sys.path.insert(0, ".")
from scipy.stats import norm
from network_model import FIT
D = json.load(open("../results/drift.json"))["grid"]
eta, n, s = FIT["eta"], FIT["n"], FIT["s_b"]
rng = np.random.default_rng(0)
rows = []
for i, q in enumerate(D["q"]):
    for j, p0 in enumerate(D["pi"]):
        b0 = norm.ppf(1 - p0)
        closed = n * eta * s**2 * norm.pdf(b0) * (p0 - q) / (p0 * (1 - p0))
        bi = b0 + s * rng.standard_normal(200000); bj = b0 + s * rng.standard_normal(200000); b = bi.copy()
        for _ in range(n):
            strict = b > bj
            p = np.where(strict, q * (norm.cdf(b) - norm.cdf(bj)) / (1 - norm.cdf(bj)), (1 - q) * (norm.cdf(bj) - norm.cdf(b)) / norm.cdf(bj))
            b = b + eta * p * (bj - b)
        rows.append(dict(q=q, pi=p0, sim=D["db"][i][j], closed=float(closed), pair=float((b - bi).mean())))
sim = np.array([r["sim"] for r in rows]); cl = np.array([r["closed"] for r in rows]); pr = np.array([r["pair"] for r in rows])
st = lambda a: dict(r=float(np.corrcoef(a, sim)[0, 1]), slope=float(np.polyfit(a, sim, 1)[0]), mae=float(np.abs(a - sim).mean()))
out = {"rows": rows, "closed": st(cl), "pair": st(pr), "sim_range": [float(sim.min()), float(sim.max())]}
json.dump(out, open("../results/drift_size.json", "w"), indent=1)
print("closed", out["closed"]); print("pair", out["pair"]); print("range", out["sim_range"])
print("central cells (pi 5-50%, q 20-80%) closed ratio:", np.round([r["closed"]/r["sim"] for r in rows if 0.05<=r["pi"]<=0.5 and 0.2<=r["q"]<=0.8 and abs(r["sim"])>0.01],2))
