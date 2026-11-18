"""Approximate standard errors for the simulated-moments estimates (Study 3, adjusted means, 31 moments).
Cov ~ (J' W J)^-1 with J the Jacobian of the simulated moments (central differences, common random numbers)
and W the diagonal weight matrix used in the fit. The sandwich covariance uses the joint whole-chain bootstrap covariance of the moments. Local approximation onlys and omit simulation and model-selection uncertainty."""
import json, sys, numpy as np
sys.path.insert(0, ".")
from chain_model import simulate, summarise
from scipy.stats import norm
T = json.load(open("../results/targets.json"))["3adj"]
NAMES = ["mu_b", "s_b", "eta", "sig_p", "v", "sig_r", "c_s"]
def moments(p, variant):
    s = summarise(simulate(p, variant, chains=40000, seed=5))
    return np.concatenate([s["V_mean"], s["S_mean"], s["P_mean"], s["P_sd"], [s["V_lag1"], s["S_lag1"], s["P_lag1"]]])
se = np.concatenate([T["V_se"], T["S_se"], T["P_se"], T["P_sd_se"], [T[k + "_lag1_se"] for k in "VSP"]])
out = {}
for variant in ("balanced", "positive", "base"):
    fit = json.load(open(f"../results/fit_s3adj_{variant}_lag.json")); p0 = np.array([fit["params"][k] for k in NAMES])
    J = np.zeros((31, 7))
    for k in range(7):
        h = 0.05 if NAMES[k] != "eta" else 0.04
        a, b = p0.copy(), p0.copy(); a[k] += h; b[k] -= h
        J[:, k] = (moments(a, variant) - moments(b, variant)) / (2 * h)
    W = np.diag(1 / se**2)
    bread = np.linalg.pinv(J.T @ W @ J)
    omega = np.array(T["moment_cov"])
    cov = bread @ J.T @ W @ omega @ W @ J @ bread
    sd = np.sqrt(np.maximum(0, np.diag(cov)))
    prev = 1 - norm.cdf(p0[0]); prev_se = norm.pdf(p0[0]) * sd[0]
    out[variant] = {"est": dict(zip(NAMES, p0.tolist())), "se": dict(zip(NAMES, sd.tolist())), "loss": fit["loss"],
                    "prior_prev": float(prev), "prior_prev_se": float(prev_se)}
    print(variant, {k: f"{e:.2f} ({s:.2f})" for k, e, s in zip(NAMES, p0, sd)}, f"prior prev {100*prev:.1f} ({100*prev_se:.1f})", flush=True)
json.dump(out, open("../results/param_se.json", "w"), indent=1)
