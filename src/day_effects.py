"""Is the generational drift a day/sample effect?

1. Composition: do participant characteristics change across generations?
2. Adjustment: does the drift (linear trend over generations) shrink when we
   control for those characteristics?
3. Within-generation timing: inside a generation, do ratings depend on the hour
   of day or on which day (for generations collected over several days)?
All models use chain-clustered standard errors.
"""
import json, sys
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
sys.path.insert(0, ".")
from load import LIK
from preprocess import study_data

def logit(p):
    p = np.clip(p, 0.5, 99.5) / 100
    return np.log(p / (1 - p))

YN = lambda x: (x.astype(str) == "Yes").astype(float)
OUT = {}
for s in (2, 3):
    d = study_data(s)
    cond = d["prompt"] if s == 2 else d["motivation"]
    x = pd.DataFrame({
        "chain": d.chain, "gen": d.gen.astype(int), "cond": cond,
        "V": (d["Other.Diagnosis.Qs_1"].map(LIK) >= 4).astype(float),
        "S": (d["Self.Diagnosis.Q_1"].map(LIK) >= 4).astype(float),
        "P": logit(pd.to_numeric(d["Q_Prevalence.Rate"], errors="coerce")),
        "age": pd.to_numeric(d["Age.Screener"], errors="coerce"),
        "female": (d.Gender == "Female").astype(float),
        "past_dx": YN(d.Past_MH_diagnosis), "nhs": YN(d.Health_Access),
        "know_tx": YN(d.Prior_Know1), "know_wait": YN(d.Prior_Know2), "follows_nd": YN(d.Prior_Know3),
        "t": pd.to_datetime(d.StartDate),
    })
    x["hour"] = x.t.dt.hour + x.t.dt.minute / 60
    x["date"] = x.t.dt.normalize()
    x["day_in_gen"] = x.groupby("gen").date.transform(lambda z: (z - z.min()).dt.days)
    covs = ["age", "female", "past_dx", "nhs", "know_tx", "know_wait", "follows_nd"]
    res = {"composition": {}, "trend": {}, "within": {}}

    # 1. composition by generation
    for c in covs:
        g = x.groupby("gen")[c].mean()
        if c == "age":
            p = stats.f_oneway(*[v.dropna() for _, v in x.groupby("gen")[c]]).pvalue
        else:
            p = stats.chi2_contingency(pd.crosstab(x.gen, x[c]))[1]
        res["composition"][c] = {"by_gen": g.round(3).tolist(), "p": float(p)}

    # 2. drift with and without covariates (linear in generation, gens 2-7 vs 1 handled by trend)
    xx = x.dropna(subset=covs + ["P"])
    for y in ("V", "S", "P"):
        f0 = f"{y} ~ gen + C(cond)"
        f1 = f0 + " + " + " + ".join(covs)
        m0 = smf.ols(f0, xx).fit(cov_type="cluster", cov_kwds={"groups": xx.chain})
        m1 = smf.ols(f1, xx).fit(cov_type="cluster", cov_kwds={"groups": xx.chain})
        res["trend"][y] = {"raw": [float(m0.params["gen"]), *m0.conf_int().loc["gen"].tolist()],
                           "adjusted": [float(m1.params["gen"]), *m1.conf_int().loc["gen"].tolist()],
                           "cov_p": {c: float(m1.pvalues[c]) for c in covs}}

    # 3. within-generation timing (generation x condition fixed effects)
    for y in ("V", "S", "P"):
        m = smf.ols(f"{y} ~ C(gen)*C(cond) + hour + day_in_gen", x.dropna(subset=[y])).fit(
            cov_type="cluster", cov_kwds={"groups": x.dropna(subset=[y]).chain})
        res["within"][y] = {k: [float(m.params[k]), float(m.pvalues[k])] for k in ("hour", "day_in_gen")}
    res["multi_day_gens"] = {int(g): int(v) for g, v in x.groupby("gen").date.nunique().items() if v > 1}
    OUT[s] = res

    print(f"=== Study {s}   generations spanning >1 day: {res['multi_day_gens']}")
    print(" composition (mean by gen 1..7, p):")
    for c, v in res["composition"].items():
        print(f"   {c:11s} {v['by_gen']}  p={v['p']:.3f}")
    print(" drift per generation, raw vs adjusted [95% CI]:")
    for y, v in res["trend"].items():
        r, a = v["raw"], v["adjusted"]
        print(f"   {y}: raw {r[0]:+.4f} [{r[1]:+.4f},{r[2]:+.4f}]  adj {a[0]:+.4f} [{a[1]:+.4f},{a[2]:+.4f}]  "
              f"strong covariates: {[c for c,p in v['cov_p'].items() if p < .01]}")
    print(" within-generation timing (coef, p):", json.dumps(res["within"]))
json.dump(OUT, open("../results/day_effects.json", "w"), indent=1)
