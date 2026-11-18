"""One scale-free network of 150 agents for Fig 4a,b: same graph, 5% experts at the hubs or at random.
-> ../results/net_snapshot.json"""
import json, sys
sys.path.insert(0, ".")
import numpy as np
import networkx as nx
from network_model import run

N = 150
out = {}
for pl in ("hubs", "random"):
    r = run("scalefree", N=N, R=1, T=300, f=0.05, placement=pl, lam=0.1, seed=5, return_state=True)
    g = r["graphs"][0]
    if "pos" not in out:
        pos = nx.kamada_kawai_layout(g)
        out["pos"] = [[float(pos[i][0]), float(pos[i][1])] for i in range(N)]
        out["edges"] = [[int(a), int(b)] for a, b in g.edges()]
        out["deg"] = [int(g.degree(i)) for i in range(N)]
    p = r["prev"][0]
    out[pl] = {"expert": r["expert"][0].astype(int).tolist(),
               "prev": [None if np.isnan(v) else float(v) for v in p], "mean": float(np.nanmean(p))}
    print(pl, out[pl]["mean"], flush=True)
json.dump(out, open("../results/net_snapshot.json", "w"), indent=1)
