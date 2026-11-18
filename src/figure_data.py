"""Simulation output for Figs 3 and 4 -> ../results/figure_data.json

1. Chain experiment: distribution of readers' implied prevalence by generation (fitted model),
   and the spread of generation means across simulated experiments of 220 chains.
2. Network: implied prevalence of every lay agent by graph distance to the nearest expert
   (scale-free, 200 agents, 20 networks, experts on 5% of nodes, q = 0.5, lambda = 0.1).
"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import networkx as nx
from scipy.stats import norm, gaussian_kde
from chain_model import simulate
import network_model as nm

out = {}

# ---- 1. chain experiment
fit = json.load(open("../results/fit_s3adj_balanced_lag.json"))["params"]
p = [fit[k] for k in ("mu_b", "s_b", "eta", "sig_p", "v", "sig_r", "c_s")]
big = simulate(p, "balanced", chains=20000, seed=11)
grid = np.linspace(0, 0.5, 101)
ridge = []
for g in range(7):
    prev = 1 - norm.cdf(big["b"][:, g])
    ridge.append(gaussian_kde(prev, bw_method=0.12)(grid).tolist())
prior = 1 - norm.cdf(fit["mu_b"] + fit["s_b"] * np.random.default_rng(5).standard_normal(20000))
out["ridge"] = {"x": grid.tolist(), "dens": ridge, "prior": gaussian_kde(prior, bw_method=0.12)(grid).tolist(),
                "median": [float(np.median(1 - norm.cdf(big["b"][:, g]))) for g in range(7)]}

reps = {k: [] for k in "VSP"}
for r in range(400):
    s = simulate(p, "balanced", chains=220, seed=1000 + r)
    for k in "VSP":
        reps[k].append(s[k].mean(0))
out["band"] = {k: {q: np.quantile(np.array(v), float(q) / 100, axis=0).tolist() for q in ("5", "25", "50", "75", "95")}
               for k, v in reps.items()}

# ---- 2. network locality
loc = {}
for pl in ("hubs", "random"):
    r = nm.run("scalefree", N=200, R=20, T=200, f=0.05, placement=pl, lam=0.1, pos_frac=0.5, seed=3, return_state=True)
    d_all, v_all = [], []
    for k, G in enumerate(r["graphs"]):
        ex = np.flatnonzero(r["expert"][k])
        dist = nx.multi_source_dijkstra_path_length(G, set(ex.tolist()))
        for i in range(G.number_of_nodes()):
            if i in set(ex.tolist()):
                continue
            d_all.append(min(dist.get(i, 99), 4))
            v_all.append(float(r["prev"][k, i]))
    d_all, v_all = np.array(d_all), np.array(v_all)
    loc[pl] = {str(d): {"n": int((d_all == d).sum()),
                        "q": np.quantile(v_all[d_all == d], [0.1, 0.25, 0.5, 0.75, 0.9]).tolist() if (d_all == d).any() else None,
                        "sample": np.random.default_rng(d).choice(v_all[d_all == d], size=min(120, int((d_all == d).sum())), replace=False).tolist() if (d_all == d).any() else []}
               for d in (1, 2, 3, 4)}
r0 = nm.run("scalefree", N=200, R=20, T=200, f=0.0, lam=0.1, pos_frac=0.5, seed=3, return_state=True)
loc["none"] = float(np.nanmean(r0["prev"]))
out["locality"] = loc
json.dump(out, open("../results/figure_data.json", "w"))
print({pl: {d: (v["n"], round(v["q"][2], 3) if v["q"] else None) for d, v in loc[pl].items()} for pl in ("hubs", "random")}, loc["none"])
print("medians", [round(m, 3) for m in out["ridge"]["median"]])
