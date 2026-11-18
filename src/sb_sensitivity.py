"""How the predicted long-run drift depends on prior heterogeneity s_b, which mostly sets its size.
Fully connected population, no experts, lambda = 0.02, q in {0.05, 0.5, 0.9}, s_b at the estimate and at
plus and minus one standard error. -> ../results/sb_sensitivity.json"""
import sys, json
sys.path.insert(0, ".")
import network_model as nm
out = {}
base = dict(nm.FIT)
fit=json.load(open('../results/fit_s3adj_balanced_lag.json'))['params']
se=json.load(open('../results/param_se.json'))['balanced']['se']['s_b']
for k in ('mu_b','s_b','eta'): nm.FIT[k]=fit[k]
for sb in (max(.01,fit['s_b']-se),fit['s_b'],fit['s_b']+se):
    nm.FIT["s_b"] = sb
    out[str(sb)] = {str(q): float(nm.run("complete", N=200, R=10, T=300, f=0.0, lam=0.02, pos_frac=q, seed=41)["final_prev"].mean()) for q in (0.05, 0.5, 0.9)}
    print(sb, out[str(sb)], flush=True)
nm.FIT.update(base)
out={'params':fit,'s_b_se':se,'results':out,'settings':{'N':200,'R':10,'T':300,'lambda':.02,'seed':41}}
json.dump(out, open("../results/sb_sensitivity.json", "w"), indent=1)
