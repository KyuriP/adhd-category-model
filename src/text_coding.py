"""Code each text for features that bear on where the reader should put the boundary,
then test (a) how the features change over generations and (b) whether the features of
the text a person READ predict that person's ratings (within generation x condition).

Features (rule-based; word lists are hand-made and rough):
  dsm_n      number of distinct DSM symptom ideas mentioned (0-18)
  extra_n    number of distinct non-DSM traits (hyperfocus, sensory overload, emotions,
             dopamine, lateness, impulsive spending, racing thoughts, boredom, isolation)
  threshold  any strictness cue: counts, durations, persistence, severity, impairment, diagnosis
  common     claims that ADHD is common or that everyone has some of it
  you        addresses the reader / invites self-identification ("you may have ADHD")
  hedge      hedges per 10 words (may, might, can, could, sometimes)
  words      length in words
"""
import json, re, sys
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
sys.path.insert(0, ".")
from load import load

def R(*p):
    return re.compile(r"\b(?:" + "|".join(p) + r")", re.I)

DSM = {  # 9 inattention + 9 hyperactivity/impulsivity ideas
    "careless": R(r"careless", r"detail", r"mistake"),
    "sustain": R(r"concentrat", r"focus", r"attention span", r"pay(?:ing)? attention", r"sustain"),
    "listen": R(r"listen", r"zone out", r"daydream", r"mind wander"),
    "follow": R(r"finish", r"complet", r"follow(?:ing)? through", r"follow instructions"),
    "organise": R(r"organi[sz]", r"disorgani[sz]", r"plan"),
    "avoid": R(r"avoid", r"procrastinat", r"put(?:ting)? off", r"delay"),
    "lose": R(r"los(?:e|ing) (?:things|items|keys|phone)", r"misplac"),
    "distract": R(r"distract"),
    "forget": R(r"forget", r"forgot"),
    "fidget": R(r"fidget", r"squirm", r"tap"),
    "seat": R(r"sit still", r"stay seated", r"sitting still", r"stay still", r"keep still"),
    "restless": R(r"restless", r"on the move", r"on the go", r"driven by a motor", r"keep moving"),
    "quiet": R(r"quiet(?:ly)?"),
    "hyper": R(r"hyperactiv", r"hyper\b", r"overactive", r"energy"),
    "talk": R(r"talk(?:s|ing)? (?:a lot|excessive|too much|quickly|fast)", r"talkative", r"chatter"),
    "blurt": R(r"blurt", r"answer before"),
    "wait": R(r"wait(?:ing)? (?:their|your|his|her) turn", r"impatien", r"can'?t wait"),
    "interrupt": R(r"interrupt", r"intrud"),
}
EXTRA = {
    "hyperfocus": R(r"hyper ?fo[c]?us", r"hyper ?fixat", r"fixat"),
    "sensory": R(r"overstimulat", r"sensory", r"hypersensitiv", r"noise"),
    "emotion": R(r"emotion", r"mood", r"overwhelm", r"anxi", r"frustrat", r"feelings"),
    "dopamine": R(r"dopamine", r"reward"),
    "late": R(r"\blate\b", r"time management", r"time blind", r"punctual", r"on time"),
    "spending": R(r"spend", r"buy", r"shopping"),
    "racing": R(r"racing", r"thoughts", r"mind (?:is|goes)"),
    "bored": R(r"bor(?:ed|ing|edom)"),
    "isolate": R(r"isolat", r"social"),
}
THRESH = R(r"at least", r"six\b", r"6\b", r"months?", r"persistent", r"persist", r"severe", r"severity",
           r"significant", r"impair", r"interfer", r"criteria", r"diagnos", r"clinical", r"professional",
           r"settings?", r"before (?:the )?age", r"childhood")
COMMON = R(r"common", r"many people", r"lots of people", r"millions", r"1 in", r"one in", r"\d+ ?%",
           r"everyone", r"we all", r"most people", r"a lot of people", r"more people")
YOU = R(r"you may have", r"you might have", r"do you", r"if you", r"are you", r"you could have", r"you have adhd")
HEDGE = R(r"may\b", r"might\b", r"can\b", r"could\b", r"sometimes")


def code(t):
    w = max(len(t.split()), 1)
    return {
        "dsm_n": sum(bool(p.search(t)) for p in DSM.values()),
        "extra_n": sum(bool(p.search(t)) for p in EXTRA.values()),
        "threshold": float(bool(THRESH.search(t))),
        "common": float(bool(COMMON.search(t))),
        "you": float(bool(YOU.search(t))),
        "hedge": 10 * len(HEDGE.findall(t)) / w,
        "words": w,
    }


def logit(p):
    p = np.clip(p, 0.5, 99.5) / 100
    return np.log(p / (1 - p))


FEATS = ["dsm_n", "extra_n", "threshold", "common", "you", "hedge", "words"]
if __name__ == "__main__":
    df = load()
    out = {}
    for s in (2, 3):
        d = df[df.study == s].copy().reset_index(drop=True)
        d = pd.concat([d, pd.DataFrame([code(t) for t in d.stim])], axis=1)
        d["V"] = (d.vincent_dx >= 4).astype(float); d["S"] = (d.self_dx >= 4).astype(float); d["P"] = logit(d.prev)
        d["lwords"] = np.log(d.words)
        res = {"by_gen": d.groupby("gen")[FEATS].mean().round(3).to_dict(orient="list")}
        # reader models: features of the text READ, generations 2-7 only (gen 1 all read the DSM)
        r = d[d.gen >= 2].copy()
        Z = ["dsm_n", "extra_n", "threshold", "common", "you", "hedge", "lwords"]
        for z in Z:
            r[z + "_z"] = (r[z] - r[z].mean()) / r[z].std()
        rhs = " + ".join(z + "_z" for z in Z)
        res["reader"] = {}
        for y in ("V", "S", "P"):
            m = smf.ols(f"{y} ~ {rhs} + C(gen)*C(cond)", r.dropna(subset=[y])).fit(
                cov_type="cluster", cov_kwds={"groups": r.dropna(subset=[y]).chain})
            res["reader"][y] = {z: [float(m.params[z + "_z"]), *map(float, m.conf_int().loc[z + "_z"]), float(m.pvalues[z + "_z"])] for z in Z}
            ft = m.f_test(" = ".join(z + "_z" for z in Z) + " = 0")
            res["reader"][y]["joint_p"] = float(ft.pvalue)
        out[s] = res
        print(f"=== Study {s}")
        print(pd.DataFrame(res["by_gen"], index=range(1, 8)).round(2).to_string())
        for y in ("V", "S", "P"):
            row = res["reader"][y]
            print(f" reader {y}: joint p={row['joint_p']:.3f}  " + "  ".join(
                f"{z}:{v[0]:+.3f}{'*' if v[3] < .05 else ''}" for z, v in row.items() if z != "joint_p"))
        d.to_pickle(f"../results/coded_s{s}.pkl")
    json.dump(out, open("../results/text_coding.json", "w"), indent=1)
