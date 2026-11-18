"""Load and harmonise the four ADHD transmission-chain studies."""
import re
import numpy as np
import pandas as pd
from preprocess import study_data

DATA = "../data"
LIK = {"Strongly Disagree": 1, "Strongly disagree": 1, "Somewhat disagree": 2,
       "Neither agree nor disagree": 3, "Somewhat agree": 4, "Strongly agree": 5}


def strip_html(s):
    if not isinstance(s, str):
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load(exclude=True):
    out = []
    for i in (1, 2, 3, 4):
        d = study_data(i)
        sd = "Self.Diagnosis.Qs_1" if i == 1 else "Self.Diagnosis.Q_1"
        if i == 1:
            cond = d["prompt"]
        elif i == 2:
            cond = d["prompt"]
        elif i == 3:
            cond = d["motivation"]
        else:
            cond = pd.Series("exp4", index=d.index)
        chain = d["chain"] if "chain" in d else d["Assigned_Chain"]
        stim = d["stimulus"] if "stimulus" in d else d["Assigned_Text"]
        x = pd.DataFrame({
            "study": i,
            "chain": chain,
            "gen": d["gen"].astype(int),
            "cond": cond,
            "self_dx": d[sd].map(LIK),
            "vincent_dx": d["Other.Diagnosis.Qs_1"].map(LIK),
            "vincent_help": d["Other.Diagnosis.Qs_2"].map(LIK),
            "prev": pd.to_numeric(d["Q_Prevalence.Rate"], errors="coerce"),
            "treat": pd.to_numeric(d["Q_Treatment.Rate"], errors="coerce"),
            "stim": stim.map(strip_html),
            "ai": d.get("AI_Self_Report", d.get("Q8", pd.Series("No", index=d.index))).astype(str).str.startswith("Yes"),
            "inject": d.get("prompt_inject", pd.Series(False, index=d.index)).fillna(False).astype(bool),
            "progress": d["Progress"],
        })
        out.append(x)
    df = pd.concat(out, ignore_index=True)
    df.loc[(df.prev < 0) | (df.prev > 100), "prev"] = np.nan
    if exclude:
        df = df[(df.progress == 100)]
    return df
