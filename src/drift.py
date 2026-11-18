"""Direction of drift under example-based exchange (Fig 2).

When agents exchange labelled examples of which a share q are positive cases, a
population whose category covers a share pi of people drifts broader when q > pi and
narrower when q < pi.  Small-gap argument: for a pair with boundaries b1 < b2 the strict
agent is corrected on a positive example with probability q*G/(1-Phi(b1)) and the broad
agent on a negative one with probability (1-q)*G/Phi(b2), G = Phi(b2)-Phi(b1).  With
b1 ~ b2 the net drift of b is proportional to (1-q)/(1-pi) - q/pi, which is zero at pi = q.
"""
import json, sys, numpy as np, networkx as nx
sys.path.insert(0, ".")
from scipy.stats import norm
from network_model import run, examples, FIT
from chain_model import PHI_DSM

out = {}
# 1. one-round drift over a grid of initial prevalence and positive share
def one_round(pi0, q, M=40000, seed=0):
    rng = np.random.default_rng(seed)
    b = norm.ppf(1 - pi0) + FIT["s_b"] * rng.standard_normal(M)
    phi = PHI_DSM + FIT["s_phi"] * rng.standard_normal(M)
    t = rng.permutation(M); bt, pt = b[t], phi[t]
    x, lab = examples(rng, pt, bt, FIT["n"], q)
    b2, phi2 = b.copy(), phi.copy()
    for j in range(FIT["n"]):
        u = np.stack([np.cos(phi2), np.sin(phi2)], -1)
        dis = ((x[:, j] * u).sum(-1) > b2) != lab[:, j]
        phi2 = np.where(dis, phi2 + FIT["eta"] * (pt - phi2), phi2)
        b2 = np.where(dis, b2 + FIT["eta"] * (bt - b2), b2)
    return float(100 * ((1 - norm.cdf(b2)).mean() - (1 - norm.cdf(b)).mean())), float((b2 - b).mean())
PI = [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
Q = [0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
out["grid"] = {"pi": PI, "q": Q, "dprev": [[one_round(p, q)[0] for p in PI] for q in Q],
               "db": [[one_round(p, q)[1] for p in PI] for q in Q]}
# sign agreement with the small-gap prediction, using the population's mean prevalence
agree = tot = 0
for i, q in enumerate(Q):
    for j, p in enumerate(PI):
        if abs(q - p) < 1e-9: continue
        tot += 1; agree += (np.sign(-out["grid"]["db"][i][j]) == np.sign(q - p))
out["grid"]["sign_agreement"] = [int(agree), int(tot)]

# 2. equilibrium prevalence vs q with reversion to the fitted lay prior (no experts)
QS = [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9]
out["equil"] = {str(lam): [float(run("complete", f=0, lam=lam, pos_frac=q, T=300, seed=1)["final_prev"].mean()) for q in QS] for lam in (0.1, 0.02)}
out["equil"]["q"] = QS
r0 = run("complete", f=0, lam=1.0, T=5, seed=1); out["equil"]["prior"] = float(r0["mean_prev_t"][0])

# 3. expert dose-response by placement on scale-free networks
FS = [0.01, 0.02, 0.05, 0.1, 0.2]
out["dose"] = {pl: [float(run("scalefree", f=f, placement=pl, lam=0.1, T=300, seed=3)["final_prev"].mean()) for f in FS] for pl in ("hubs", "random")}
out["dose"]["f"] = FS
out["dose"]["none"] = float(run("scalefree", f=0, lam=0.1, T=300, seed=3)["final_prev"].mean())

# 4. one illustrative network, same graph, experts at hubs vs at random
out["net"] = {}
for pl in ("hubs", "random"):
    r = run("scalefree", N=40, R=1, T=300, f=0.1, placement=pl, lam=0.1, seed=11, return_state=True)
    g = r["graphs"][0]
    if "pos" not in out["net"]:
        pos = nx.spring_layout(g, seed=4, iterations=200)
        out["net"]["pos"] = [[float(pos[i][0]), float(pos[i][1])] for i in range(40)]
        out["net"]["edges"] = [[int(a), int(b)] for a, b in g.edges()]
    out["net"][pl] = {"expert": r["expert"][0].astype(int).tolist(), "prev": [None if np.isnan(v) else float(v) for v in r["prev"][0]],
                      "mean": float(np.nanmean(r["prev"][0]))}
json.dump(out, open("../results/drift.json", "w"), indent=1)
print("sign agreement", out["grid"]["sign_agreement"])
print("dprev grid (rows q, cols pi):"); print(np.round(out["grid"]["dprev"], 2))
print("equil", {k: np.round(v, 3).tolist() for k, v in out["equil"].items() if k not in ("q",)} )
print("dose", {k: np.round(v, 3).tolist() for k, v in out["dose"].items()})
print("net means", out["net"]["hubs"]["mean"], out["net"]["random"]["mean"])
