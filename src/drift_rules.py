"""Does the sign of the drift depend on the update rule? (label-only check)
One round of random pairwise exchange on the 8 x 8 grid of category prevalence pi and positive share q
(as in Figure 2b), for three rules applied to each of n = 10 labelled examples:
  copy        main rule: on disagreement move a fraction eta of the way to the source's offset
  perceptron  label only: on disagreement move the offset a fixed step d toward classifying the example
              correctly (down after a missed positive, up after a false positive)
  margin      label only: on disagreement move the offset a fraction eta of the way to the example itself
Offsets only (prevalence depends on the offset alone). -> ../results/drift_rules.json
"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
from scipy.stats import norm

PI = [0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.95]
M, n, eta, d, s_b = 40000, 10, 0.28, 0.05, 0.31


def one_round(pi, q, rule, rng):
    b = norm.ppf(1 - pi) + s_b * rng.standard_normal(M)
    src = b[rng.permutation(M)]
    lab = rng.random((M, n)) < q
    pb = norm.cdf(src)[:, None]
    U = rng.random((M, n))
    x = np.clip(np.where(lab, norm.ppf(pb + U * (1 - pb)), norm.ppf(U * pb)), -8, 8)
    b0 = b.copy()
    for j in range(n):
        dis = (x[:, j] > b) != lab[:, j]
        if rule == "copy":
            b = np.where(dis, b + eta * (src - b), b)
        elif rule == "perceptron":
            b = np.where(dis, b + np.where(lab[:, j], -d, d), b)
        else:
            b = np.where(dis, b + eta * (x[:, j] - b), b)
    return float((b - b0).mean())


out = {"pi": PI, "rules": {}}
for rule in ("copy", "perceptron", "margin"):
    rng = np.random.default_rng(1)
    g = [[one_round(pi, q, rule, rng) for pi in PI] for q in PI]          # rows: q, columns: pi
    match = sum(np.sign(-g[i][j]) == np.sign(PI[i] - PI[j]) for i in range(8) for j in range(8) if i != j)
    out["rules"][rule] = {"db": g, "sign_match": int(match), "cells": 56}
    print(rule, "sign matches", match, "/ 56", flush=True)
json.dump(out, open("../results/drift_rules.json", "w"), indent=1)
