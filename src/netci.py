"""Monte Carlo intervals for the main network numbers: 40 scale-free networks of 200 agents, q = 0.5,
lambda = 0.1, 300 rounds. The same networks are used in every condition, so differences are paired.
-> ../results/netci.json"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import network_model as nm

def per_net(**k):
    return nm.run("scalefree", N=200, R=40, T=300, lam=0.1, pos_frac=0.5, seed=31, **k)["final_prev"]

ci = lambda v: [float(100 * v.mean()), float(100 * (v.mean() - 1.96 * v.std(ddof=1) / np.sqrt(len(v)))), float(100 * (v.mean() + 1.96 * v.std(ddof=1) / np.sqrt(len(v))))]
none = per_net(f=0.0)
out = {"none": ci(none)}
for pl, f in (("random", 0.05), ("random", 0.10), ("random", 0.20), ("hubs", 0.02), ("hubs", 0.05)):
    v = per_net(f=f, placement=pl)
    out[f"{pl}_{f}"] = ci(v); out[f"reduction_{pl}_{f}"] = ci(none - v)
    print(pl, f, out[f"{pl}_{f}"], out[f"reduction_{pl}_{f}"], flush=True)
h2, r10 = per_net(f=0.02, placement="hubs"), per_net(f=0.10, placement="random")
out["hubs2_minus_random10"] = ci(r10 - h2)
r5, h5 = per_net(f=0.05, placement="random"), per_net(f=0.05, placement="hubs")
red_r, red_h = none - r5, none - h5
out["ratio_hubs_random_5"] = float(red_h.mean() / red_r.mean())
bs = np.random.default_rng(0).integers(0, len(none), (2000, len(none)))
rat = red_h[bs].mean(1) / red_r[bs].mean(1)
out["ratio_hubs_random_5_ci"] = [float(np.quantile(rat, 0.025)), float(np.quantile(rat, 0.975))]
print(out["hubs2_minus_random10"], out["ratio_hubs_random_5"], out["ratio_hubs_random_5_ci"])
json.dump(out, open("../results/netci.json", "w"), indent=1)
