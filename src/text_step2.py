"""Step 2: use the texts people wrote as a less noisy measure of their boundary.

Each participant at generation g writes the text read at generation g+1, so
writer (chain c, gen g) <-> stimulus of (c, g+1). Writers exist for g = 1..6.

1. Writer composite = mean of z-scored Vincent rating, self rating, logit prevalence
   (the writer's own answers, i.e. a noisy reading of their boundary).
2. Text score = cross-fitted ridge prediction of the writer composite from the
   TF-IDF of the writer's text (folds grouped by chain, so a chain's texts never
   train the model that scores them).
3. Tests
   validity     : text score vs writer's own composite
   transmission : writer's text score vs the READER's composite (next generation)
   lineage      : text score at g vs text score at g+1 in the same chain
   baseline     : writer composite vs reader composite (what the ratings alone show)
"""
import json, sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
sys.path.insert(0, ".")
from load import load


def logit(p):
    p = np.clip(p, 0.5, 99.5) / 100
    return np.log(p / (1 - p))


def zs(x):
    return (x - x.mean()) / x.std()


def boot_r(a, b, groups, n=2000, seed=0):
    """Pearson r with a 95% CI from a bootstrap over chains."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": a, "b": b, "g": groups}).dropna()
    r0 = stats.pearsonr(df.a, df.b)[0]
    ug = df.g.unique(); idx = {g: np.where(df.g.values == g)[0] for g in ug}
    rs = []
    for _ in range(n):
        pick = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        rs.append(stats.pearsonr(df.a.values[pick], df.b.values[pick])[0])
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return dict(r=float(r0), lo=float(lo), hi=float(hi), n=int(len(df)))


df = load()
res = {}
for s in (2, 3):
    d = df[df.study == s].copy()
    d["P"] = logit(d.prev)
    for v in ("vincent_dx", "self_dx", "P"):
        # remove generation x condition means so drift does not create spurious correlation
        d[v + "_c"] = d[v] - d.groupby(["cond", "gen"])[v].transform("mean")
    d["comp"] = zs(d[["vincent_dx_c", "self_dx_c", "P_c"]].apply(zs).mean(1))
    nxt = d[["chain", "gen", "stim", "comp"]].copy(); nxt["gen"] -= 1
    w = d[d.gen <= 6].merge(nxt.rename(columns={"stim": "text", "comp": "reader_comp"}), on=["chain", "gen"])
    w = w[w.text.str.split().str.len() >= 3].reset_index(drop=True)

    oof = np.zeros(len(w))
    for tr, te in GroupKFold(5).split(w, groups=w.chain):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, stop_words="english")
        Xtr = vec.fit_transform(w.text.iloc[tr])
        m = RidgeCV(alphas=np.logspace(-1, 3, 20)).fit(Xtr, w.comp.iloc[tr])
        oof[te] = m.predict(vec.transform(w.text.iloc[te]))
    w["text_score"] = oof
    # residualise text score on gen x cond as well
    w["text_score"] = w.text_score - w.groupby(["cond", "gen"]).text_score.transform("mean")

    # lineage of text scores along the chain
    lag = w[["chain", "gen", "text_score"]].copy(); lag["gen"] -= 1
    L = w.merge(lag.rename(columns={"text_score": "next_score"}), on=["chain", "gen"])

    # interpretable words: full-data fit
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True, stop_words="english")
    X = vec.fit_transform(w.text); m = RidgeCV(alphas=np.logspace(-1, 3, 20)).fit(X, w.comp)
    terms = np.array(vec.get_feature_names_out()); o = np.argsort(m.coef_)
    res[s] = {
        "n_writers": int(len(w)),
        "validity": boot_r(w.text_score, w.comp, w.chain),
        "transmission": boot_r(w.text_score, w.reader_comp, w.chain),
        "baseline_ratings": boot_r(w.comp, w.reader_comp, w.chain),
        "text_lineage": boot_r(L.text_score, L.next_score, L.chain),
        "broad_terms": terms[o[::-1][:15]].tolist(),
        "strict_terms": terms[o[:15]].tolist(),
    }
    print(f"== Study {s}  writers={len(w)}")
    for k in ("validity", "transmission", "baseline_ratings", "text_lineage"):
        r = res[s][k]; print(f"  {k:17s} r={r['r']:+.3f}  [{r['lo']:+.3f}, {r['hi']:+.3f}]  n={r['n']}")
    print("  broad :", ", ".join(res[s]["broad_terms"]))
    print("  strict:", ", ".join(res[s]["strict_terms"]))
json.dump(res, open("../results/step2_text.json", "w"), indent=1)
