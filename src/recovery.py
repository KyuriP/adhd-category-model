"""Parameter and model recovery.
Synthetic experiments of the size of Study 3 (220 chains, 7 generations) are generated from the fitted
partial-copying model and from the fitted no-lineage model, and each is refit with both models using the
same moments, weights and optimiser as fit.py. -> ../results/recovery.json
"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
from multiprocessing import Pool
from chain_model import simulate, summarise
import fit as F
import argparse

T3 = F.T["3adj"]
NAMES = F.NAMES


def synth_targets(params, variant, seed):
    s = summarise(simulate(params, variant, chains=220, seed=seed))
    t = dict(T3)                                   # keep the data's standard errors and n
    for k in "VSP":
        t[k + "_mean"] = np.asarray(s[k + "_mean"]).tolist()
        t[k + "_lag1"] = float(s[k + "_lag1"])
    t["P_sd"] = np.asarray(s["P_sd"]).tolist()
    return t


def job(args):
    gen_model, rep, fit_model = args
    src = json.load(open(f"../results/fit_s3adj_{'balanced' if gen_model == 'partial' else 'register'}_lag.json"))["params"]
    names = NAMES + (["b_lay"] if gen_model == "lineage_free" else [])
    p = [src[k] for k in names]
    t = synth_targets(p, "balanced" if gen_model == "partial" else "register", seed=5000 + rep)
    variant = "balanced" if fit_model == "partial" else "register"
    F.T["synthetic"] = t
    r = F.fit("synthetic", variant, True)
    est = dict(zip(NAMES + (["b_lay"] if variant == "register" else []), map(float, r.x)))
    check=[float(F.loss(r.x,t,variant,True,chains=20000,seed=s)) for s in (17,19,23)]
    return {"generated_by": gen_model, "rep": rep, "fit_with": fit_model, "loss": float(r.fun), "est": est, "true": src,
            'simulation_check':{'seeds':[17,19,23],'chains':20000,'losses':check,'mean_loss':float(np.mean(check))},
            'optimizer':{'converged':bool(r.success),'nit':int(r.nit),'nfev':int(r.nfev),'maxiter':60,'popsize':12},
            'synthetic_seed':5000+rep,'target_version':'2026-11-18-chain-bootstrap-2000',
            'scope':'Conditional on generating parameters and the observed-data bootstrap weights; no covariate-adjustment or model-selection uncertainty.'}


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--replicates',type=int,default=10)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    jobs = [(g, r, f) for g in ("partial", "lineage_free") for r in range(args.replicates) for f in ("partial", "lineage_free")]
    with Pool(args.workers) as pool:
        res = []
        for out in pool.imap_unordered(job, jobs):
            res.append(out)
            print(out["generated_by"], out["rep"], out["fit_with"], round(out["loss"], 1),
                  {k: round(v, 3) for k, v in out["est"].items() if k in ("eta", "mu_b", "s_b")}, flush=True)
            json.dump(res, open("../results/recovery.json", "w"), indent=1)
