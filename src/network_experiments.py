"""Network experiments with the Study-3-fitted parameters. Writes results/network.json."""
import json, sys
import numpy as np
import networkx as nx
sys.path.insert(0, ".")
from network_model import run

TOPO = ["complete", "ring", "star", "scalefree"]
ms = lambda a: [float(np.mean(a)), float(np.std(a) / np.sqrt(len(a)))]     # mean, SE over replicates
out = {}

# E1 peer interaction alone: does the category broaden without any expert or seed?
out["ratchet"] = {}
for q in ["base", 0.5, 0.9]:
    for lam in [0.0, 0.1]:
        for k in TOPO:
            r = run(k, f=0, lam=lam, pos_frac=q, T=300, seed=1)
            out["ratchet"][f"{q}|{lam}|{k}"] = ms(r["final_prev"]) + [float(r["mean_prev_t"][0])]

# E2 how many DSM-anchored experts hold the boundary?
out["experts"] = {}
FS = [0, 0.02, 0.05, 0.1, 0.2, 0.4]
for q in [0.5, "base"]:
    for k in TOPO:
        for f in FS:
            r = run(k, f=f, lam=0.1, pos_frac=q, T=300, seed=2)
            out["experts"][f"{q}|{k}|{f}"] = ms(r["final_prev"])

# E3 where should experts sit? 5% experts, hubs vs random vs periphery
out["placement"] = {}
for k in ["scalefree", "star"]:
    for pl in ["random", "hubs", "periphery"]:
        for lam in [0.1, 0.02]:
            r = run(k, f=0.05, placement=pl, lam=lam, T=400, seed=3)
            out["placement"][f"{k}|{pl}|{lam}"] = ms(r["final_prev"])

# E4 one-off campaign: everyone reads the DSM once, then only peers
out["campaign"] = {}
for k in ["complete", "ring"]:
    for lam in [0.0, 0.05, 0.2]:
        r = run(k, f=0, lam=lam, campaign=True, T=60, seed=4)
        r0 = run(k, f=0, lam=lam, campaign=False, T=60, seed=4)
        out["campaign"][f"{k}|{lam}"] = {"with": r["mean_prev_t"].tolist(), "without": r0["mean_prev_t"].tolist()}

# E5 pockets: implied prevalence by graph distance to the nearest expert (5% experts)
out["distance"] = {}
for k in ["ring", "scalefree"]:
    r = run(k, f=0.05, lam=0.1, T=300, seed=5, return_state=True)
    acc = {}
    for g, ex, pv in zip(r["graphs"], r["expert"], r["prev"]):
        d = nx.multi_source_dijkstra_path_length(g, set(np.where(ex)[0].tolist()))
        for i, di in d.items():
            if di > 0:
                acc.setdefault(min(int(di), 8), []).append(pv[i])
    out["distance"][k] = {str(d): [float(np.mean(v)), len(v)] for d, v in sorted(acc.items())}

json.dump(out, open("../results/network.json", "w"), indent=1)
for sec in ("ratchet", "experts", "placement", "distance"):
    print("==", sec)
    for k, v in out[sec].items():
        print("  ", k, np.round(v, 3).tolist() if isinstance(v, list) else {a: round(b[0], 3) for a, b in v.items()})
print("== campaign (t=0,1,5,20,60): with / without")
for k, v in out["campaign"].items():
    print("  ", k, np.round(np.array(v["with"])[[0, 1, 5, 20, 60]], 3).tolist(), "/", np.round(np.array(v["without"])[[0, 1, 5, 20, 60]], 3).tolist())
