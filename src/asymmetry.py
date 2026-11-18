"""Strict experts vs broad sources, on the prevalence scale.
Broad sources sit as far below the mean lay prior as the DSM boundary is above it.
Equal offset distances are not equal on the prevalence scale, so the paper uses
asymmetry_b.py instead; this one is kept for reference. -> ../results/asymmetry.json
"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
from scipy.stats import norm
import network_model as nm
from chain_model import B_DSM

mu = nm.FIT["mu_b"]; b_broad = mu - (B_DSM - mu)
out = {"b_strict": B_DSM, "b_broad": b_broad, "prev_strict": float(1 - norm.cdf(B_DSM)), "prev_broad": float(1 - norm.cdf(b_broad)), "rows": []}
for q in (0.2, 0.5, 0.9):
    for lam in (0.02, 0.1):
        none = nm.run("scalefree", N=200, R=20, T=200, f=0.0, lam=lam, pos_frac=q, seed=3)["final_prev"].mean()
        row = {"q": q, "lam": lam, "none": float(none)}
        for pl in ("hubs", "random"):
            for kind, bb in (("strict", B_DSM), ("broad", b_broad)):
                v = nm.run("scalefree", N=200, R=20, T=200, f=0.05, placement=pl, lam=lam, pos_frac=q, seed=3, src_b=bb)["final_prev"].mean()
                row[f"{kind}_{pl}"] = float(v)
        row["ratio_hubs"] = (row["broad_hubs"] - none) / (none - row["strict_hubs"])
        row["ratio_random"] = (row["broad_random"] - none) / (none - row["strict_random"])
        out["rows"].append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
json.dump(out, open("../results/asymmetry.json", "w"), indent=1)
