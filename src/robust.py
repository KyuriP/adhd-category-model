"""Robustness of the network results: sensitivity to q and lambda, network size, other topologies;
and a quantitative prediction of the size of the drift."""
import json, sys, numpy as np
sys.path.insert(0, ".")
from scipy.stats import norm
from network_model import run, FIT

out = {}
m = lambda r: float(r["final_prev"].mean())
se = lambda r: float(r["final_prev"].std() / np.sqrt(len(r["final_prev"])))

# 1. sensitivity of expert placement to q and lambda (scale-free, 5% experts)
QS, LS = [0.2, 0.35, 0.5, 0.7, 0.9], [0.02, 0.05, 0.1, 0.2, 0.4]
out["sens"] = {"q": QS, "lam": LS, "none": [], "random": [], "hubs": []}
for lam in LS:
    rows = {"none": [], "random": [], "hubs": []}
    for q in QS:
        rows["none"].append(m(run("scalefree", f=0, lam=lam, pos_frac=q, T=300, seed=21)))
        rows["random"].append(m(run("scalefree", f=0.05, placement="random", lam=lam, pos_frac=q, T=300, seed=21)))
        rows["hubs"].append(m(run("scalefree", f=0.05, placement="hubs", lam=lam, pos_frac=q, T=300, seed=21)))
    for k in rows: out["sens"][k].append(rows[k])
    print("lam", lam, {k: np.round(v, 3).tolist() for k, v in rows.items()}, flush=True)

# 3a. network size
out["size"] = {"N": [100, 200, 500, 1000], "none": [], "random": [], "hubs": []}
for N in out["size"]["N"]:
    R = 20 if N <= 200 else (10 if N == 500 else 6)
    for pl, f in (("none", 0), ("random", 0.05), ("hubs", 0.05)):
        r = run("scalefree", N=N, R=R, f=f, placement=pl if f else "random", lam=0.1, T=300, seed=22)
        out["size"][pl].append([m(r), se(r)])
    print("N", N, {k: np.round(out["size"][k][-1], 4).tolist() for k in ("none", "random", "hubs")}, flush=True)

# 3b. other topologies, 5% experts
out["topo"] = {}
for kind, pls in (("scalefree", ["random", "hubs"]), ("sfclustered", ["random", "hubs"]), ("modular", ["random", "hubs", "onecomm"]),
                  ("ring", ["random"]), ("complete", ["random"])):
    d = {"none": [m(r0 := run(kind, f=0, lam=0.1, T=300, seed=23)), se(r0)]}
    for pl in pls:
        r = run(kind, f=0.05, placement=pl, lam=0.1, T=300, seed=23); d[pl] = [m(r), se(r)]
    out["topo"][kind] = d
    print(kind, {k: np.round(v, 4).tolist() for k, v in d.items()}, flush=True)
json.dump(out, open("../results/robust.json", "w"), indent=1)
