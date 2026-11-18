"""Fit chain-model variants to Study 2 and Study 3 by simulated method of moments."""
import json, sys, time
import numpy as np
from scipy.optimize import differential_evolution
sys.path.insert(0, ".")
from chain_model import simulate, summarise

T = json.load(open("../results/targets.json"))
BOUNDS = [(-1, 3), (0.01, 2), (0.0, 1.0), (0.01, 3), (-1, 3), (0.05, 3), (-3, 3)]
NAMES = ["mu_b", "s_b", "eta", "sig_p", "v", "sig_r", "c_s"]

def parameter_names(variant):
    return NAMES + (['b_lay'] if variant in ('register','register_trend') else []) + (['trend'] if variant=='register_trend' else [])


def loss(p, t, variant, use_lag, chains=3000, seed=7):
    generations=len(t['V_mean'])
    out=simulate(p, variant, chains=chains, seed=seed, generations=generations)
    s = summarise({k:v[:,:generations] for k,v in out.items()})
    L = 0.0
    for k in "VSP":
        L += (((s[k + "_mean"] - np.array(t[k + "_mean"])) / np.array(t[k + "_se"])) ** 2).sum()
    L += (((s["P_sd"] - np.array(t["P_sd"])) / np.array(t["P_sd_se"])) ** 2).sum()
    if use_lag:
        for k in "VSP":
            L += ((s[k + "_lag1"] - t[k + "_lag1"]) / t[k + "_lag1_se"]) ** 2
    return L


def fit(study, variant, use_lag):
    t = T[str(study)]
    bounds = BOUNDS + ([(-1, 3)] if variant in ('register','register_trend') else []) + ([(-0.5,0.5)] if variant=='register_trend' else [])
    import pathlib
    path = pathlib.Path(f"../results-original/fit_s{study}_{variant}_{'lag' if use_lag else 'nolag'}.json")
    previous = json.load(open(path))["params"] if path.exists() else None
    names = parameter_names(variant)
    starts=pathlib.Path('../fit_start_values.json')
    if starts.exists():
        previous=json.load(open(starts)).get(f'{study}:{variant}',previous)
    x0 = [previous[k] for k in names] if previous else None
    r = differential_evolution(loss, bounds, args=(t, variant, use_lag), seed=3,
                               maxiter=60, popsize=12, tol=1e-6, polish=False, x0=x0)
    return r


if __name__ == "__main__":
    study, variant, use_lag = sys.argv[1], sys.argv[2], sys.argv[3] == "lag"
    t0 = time.time()
    r = fit(study, variant, use_lag)
    s = summarise(simulate(r.x, variant, chains=20000, seed=11, generations=len(T[str(study)]['V_mean'])))
    res = {"study": study, "variant": variant, "use_lag": use_lag, "generations": len(T[str(study)]["V_mean"]), "loss": float(r.fun),
           "params": dict(zip(parameter_names(variant), map(float, r.x))),
           "pred": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in s.items()},
           "secs": time.time() - t0,
           "optimizer": {"seed": 3, "maxiter": 60, "popsize": 12, "nit": int(r.nit),
                         "nfev": int(r.nfev), "converged": bool(r.success), "message": str(r.message)},
           "validation_loss_20000": float(loss(r.x, T[str(study)], variant, use_lag, chains=20000, seed=17)),
           "target_version": "2026-11-18-chain-bootstrap-2000"}
    json.dump(res, open(f"../results/fit_s{study}_{variant}_{'lag' if use_lag else 'nolag'}.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("study", "variant", "use_lag", "loss", "params", "secs")}))
    print({k: np.round(v, 3).tolist() if isinstance(v, list) else round(v, 3) for k, v in res["pred"].items()})
