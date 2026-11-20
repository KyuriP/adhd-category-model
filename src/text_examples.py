"""Exploratory coding of the texts read in Studies 2 and 3: positive versus limiting statements.
The counts describe the texts; they are not an estimate of q and are not used in any fit.

Each text is split into sentences. A sentence that says who does not have ADHD, or states a
condition for diagnosis, counts as limiting; otherwise a sentence describing people with ADHD
or their behaviour counts as positive. Everything else is left uncoded. Word lists are hand-made.
-> ../results/text_examples.json
"""
import json, re, sys
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from load import strip_html
from preprocess import study_data

LIMIT = re.compile(r"\b(not everyone|not all|not every|doesn'?t (?:necessarily )?mean|does not (?:necessarily )?mean|isn'?t (?:just|simply|the same|laziness|a lack)|is not (?:just|simply|the same|laziness|caused|a lack)|everyone (?:gets|has|can|experiences|is|struggles|loses|forgets)|we all|all of us|from time to time|occasionally|once in a while|normal (?:to|for|behaviou?r|level)|typical (?:for|of) (?:their|a person'?s|someone'?s) age|more than (?:most|other|is typical|usual|average|normal|their peers)|beyond what|out of proportion|only (?:diagnosed|if|when|counts)|must (?:be|have|persist|occur|last|start|begin)|at least (?:six|6|five|5)|six months|6 months|before (?:the )?age|by age|two or more (?:settings|places|environments)|more than one setting|multiple settings|significant(?:ly)? (?:impair|interfer|affect|disrupt)|interfere[sd]? with|rule[sd]? out|better explained|not (?:due|caused) to|rather than|unlike)\b", re.I)
POS = re.compile(r"\b(people|someone|somebody|individuals?|persons?|those|children|kids|adults|sufferers|patients|they|he|she|you)\b[^.!?]{0,40}\b(with adhd|who have adhd|who has adhd|have adhd|has adhd|adhd)\b|\badhd\b[^.!?]{0,30}\b(is|causes|means|can|may|might|makes|affects|involves|presents|shows|leads|results)\b|\b(symptoms?|signs?|traits?|characteristics?)\b[^.!?]{0,20}\b(include|are|of adhd|such as|like)\b|\b(struggle|struggles|difficulty|difficulties|trouble|hard time|unable|forget|fidget|interrupt|impulsiv|restless|hyperactiv|inattent|distracted|lose|loses|losing)\b", re.I)


def code(text):
    sents = [s for s in re.split(r"(?<=[.!?])\s+", strip_html(text)) if len(s) > 3]
    lim = sum(bool(LIMIT.search(s)) for s in sents)
    pos = sum(bool(POS.search(s)) and not LIMIT.search(s) for s in sents)
    return len(sents), pos, lim


out = {}
for study in (2, 3):
    d = study_data(study)
    c = pd.DataFrame([code(t) for t in d["stimulus"]], columns=["n", "pos", "limit"])
    c["gen"] = d["gen"].astype(int).values
    g = c.groupby("gen").agg(texts=("n", "size"), sentences=("n", "sum"), pos=("pos", "sum"), limit=("limit", "sum"),
                             any_limit=("limit", lambda v: float((v > 0).mean())))
    g["positive_share"] = g["pos"] / (g["pos"] + g["limit"])
    out[str(study)] = {int(k): {kk: (float(vv) if kk in ("any_limit", "positive_share") else int(vv)) for kk, vv in row.items()}
                       for k, row in g.iterrows()}
    print("Study", study); print(g.round(3))
json.dump(out, open("../results/text_examples.json", "w"), indent=1)
