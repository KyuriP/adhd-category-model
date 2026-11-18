"""Same-day comparisons (Table S4).

Study 2: examples vs words instruction, both run on the same days. Slope per generation by
condition and the difference, SEs clustered by chain.
Study 4: criteria vs a gen-7 Study 2 text on one day. Observed difference vs what the
Study 2 and Study 3 fits predict.
-> ../results/conditions.json
"""
import sys, json
sys.path.insert(0, ".")
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from load import LIK
from targets import logit
from chain_model import simulate
from preprocess import study_data

def agreement(s):
    v=s.map(LIK)
    return v.ge(4).astype(float).where(v.notna())

out = {"study2": {}, "study4": {}}


def frame(s):
    d = study_data(s)
    sd = "Self.Diagnosis.Qs_1" if s == 1 else "Self.Diagnosis.Q_1"
    x = pd.DataFrame({"chain": d.chain if "chain" in d else d["Assigned_Chain"], "gen": d.gen.astype(int),
                      "V": agreement(d["Other.Diagnosis.Qs_1"]),
                      "S": agreement(d[sd]),
                      "P": logit(pd.to_numeric(d["Q_Prevalence.Rate"], errors="coerce"))})
    if s in (1, 2):
        x["cond"] = d["prompt"]
    if "Progress" in d:
        x = x[d["Progress"].values == 100]
    return x


x2 = frame(2)
x2["ex"] = (x2.cond == "examples").astype(float)
for y in "VSP":
    xx = x2.dropna(subset=[y])
    m = smf.ols(f"{y} ~ C(gen) + ex + ex:gen", xx).fit(cov_type="cluster", cov_kwds={"groups": xx.chain})
    r = {"diff": [float(m.params["ex:gen"]), *map(float, m.conf_int().loc["ex:gen"])], "p": float(m.pvalues["ex:gen"])}
    for c in ("words", "examples"):
        z = xx[xx.cond == c]
        mm = smf.ols(f"{y} ~ gen", z).fit(cov_type="cluster", cov_kwds={"groups": z.chain})
        r[c] = [float(mm.params["gen"]), *map(float, mm.conf_int().loc["gen"])]
    out["study2"][y] = r
# Descriptive signs only. Outcomes are correlated, so this is not a sign test.
out["study2"]["all_three_positive"] = all(out["study2"][y]["diff"][0] > 0 for y in "VSP")

x4 = study_data(4)
x4 = pd.DataFrame({"gen": x4.gen.astype(int), "prog": x4.Progress,
                   "V": agreement(x4["Other.Diagnosis.Qs_1"]),
                   "S": agreement(x4["Self.Diagnosis.Q_1"]),
                   "P": logit(pd.to_numeric(x4["Q_Prevalence.Rate"], errors="coerce"))})
x4 = x4[x4.prog == 100]
obs = {}
for y in "VSP":
    a, b = x4[x4.gen == 1][y].dropna(), x4[x4.gen == 7][y].dropna()
    dlt = b.mean() - a.mean(); se = np.sqrt(a.var() / len(a) + b.var() / len(b))
    obs[y] = [float(dlt), float(dlt - 1.96 * se), float(dlt + 1.96 * se), int(len(a)), int(len(b))]
out["study4"]["observed"] = obs
pred = {}
for key in ("2", "3adj"):
    f = json.load(open(f"../results/fit_s{key}_balanced_lag.json"))["params"]
    p = [f[k] for k in ("mu_b", "s_b", "eta", "sig_p", "v", "sig_r", "c_s")]
    s = simulate(p, "balanced", n=10, chains=40000, seed=5, generations=8)
    # A new reader of the generation-7 text receives an eighth reading in a chain.
    pred[key] = {y: float(s[y][:, 7].mean() - s[y][:, 0].mean()) for y in "VSP"}
    pred[key]["eta"] = f["eta"]
out["study4"]["predicted"] = pred
out['settings']={'study2_chains':int(x2.chain.nunique()),'study2_n':len(x2),'prediction_chains':40000,'prediction_seed':5,'text_generation':7,'reader_generation':8,'parameter_uncertainty_included':False}
json.dump(out, open("../results/conditions.json", "w"), indent=1)
print(json.dumps(out, indent=1))
