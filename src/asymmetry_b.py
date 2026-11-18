"""Strict experts vs broad sources (Fig 5c), measured on the offset scale, where both are
equally far from the lay prior. 40 networks, same networks and expert positions for both source
types, so differences are paired. -> ../results/asymmetry_b.json"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import network_model as nm
from chain_model import B_DSM
mu = nm.FIT["mu_b"]; bb = mu - (B_DSM - mu); rows = []
for q, lam in ((0.2, 0.1), (0.5, 0.1), (0.9, 0.1), (0.5, 0.02)):
    def mb(**k):
        r = nm.run("scalefree", N=200, R=40, T=300, lam=lam, pos_frac=q, seed=3, return_state=True, **k)
        B = np.where(r["expert"], np.nan, r["b"])
        return np.nanmean(B, axis=1)                      # mean lay offset per network
    none = mb(f=0.0); row = {"q": q, "lam": lam, "none": float(none.mean())}
    ci = lambda v: [float(v.mean()), float(v.mean() - 1.96 * v.std(ddof=1) / np.sqrt(len(v))), float(v.mean() + 1.96 * v.std(ddof=1) / np.sqrt(len(v)))]
    for pl in ("hubs", "random"):
        st, br = mb(f=0.05, placement=pl, src_b=B_DSM), mb(f=0.05, placement=pl, src_b=bb)
        row["strict_" + pl] = float(st.mean()); row["broad_" + pl] = float(br.mean())
        row["strict_shift_" + pl] = ci(st - none); row["broad_shift_" + pl] = ci(none - br)
        row["diff_" + pl] = ci((none - br) - (st - none))          # broad push minus strict pull
        row["ratio_" + pl] = (none.mean() - br.mean()) / (st.mean() - none.mean())
    rows.append(row); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
json.dump({"b_broad": bb, "rows": rows}, open("../results/asymmetry_b.json", "w"), indent=1)
