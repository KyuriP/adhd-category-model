"""Chain-mode version of the linear-boundary convention model.

Each agent holds a linear boundary in a 2D trait space (x1 = inattention,
x2 = hyperactivity), parameterised by angle phi and offset b: a point x is
'ADHD' if u(phi).x > b.  The population of people is x ~ N(0, I), so the
fraction of the population an agent counts as ADHD is 1 - Phi(b).

Transmission is the same update rule as the naming-game model, but run down
a chain instead of between peers: agent k starts from its own prior boundary,
sees n examples written by agent k-1 (the teacher), and on every example
where its own label disagrees with the teacher's label it moves its
(phi, b) a fraction eta of the way toward the teacher's.  Generation 1's
teacher is the DSM boundary.

Variants
  'balanced'  : every text shows n/2 positive and n/2 negative examples
  'positive'  : the DSM text is balanced (it states a threshold), lay texts
                from generation 2 on only show positive examples (symptom
                lists), so a learner who is already more inclusive than the
                teacher never disagrees and never moves
  'register'  : no lineage. From generation 2 on every learner is taught by a
                generic 'lay' boundary b_lay shared by all chains (tests
                whether a population-wide shift explains the data as well)

Measurement
  prevalence report: logit(report) = logit(1 - Phi(b)) + N(0, sig_p)
  Vincent >= 4     : u.xV - b + N(0, sig_r) > 0, xV = (v, v)
  self >= 4        : u.x_self - b + c_s + N(0, sig_r) > 0, x_self ~ N(0, I)
"""
import numpy as np
from scipy.stats import norm, truncnorm

B_DSM = norm.ppf(0.95)          # DSM boundary: ~5% of population
PHI_DSM = np.pi / 4
N_GEN = 7


def _examples(rng, phi_t, b_t, n, positive_only):
    """Points labelled by the teacher. Returns x (C, n, 2), label (C, n)."""
    C = phi_t.shape[0]
    u = np.stack([np.cos(phi_t), np.sin(phi_t)], -1)          # (C,2)
    w = np.stack([-np.sin(phi_t), np.cos(phi_t)], -1)
    if positive_only:
        lab = np.ones((C, n), bool)
    else:
        lab = np.zeros((C, n), bool)
        lab[:, : n // 2] = True
    U = rng.random((C, n))
    # truncated normal along u: above b for positives, below for negatives
    pb = norm.cdf(b_t)[:, None]
    t = np.where(lab, norm.ppf(pb + U * (1 - pb)), norm.ppf(U * pb))
    t = np.clip(t, -8, 8)
    s = rng.standard_normal((C, n))
    x = t[..., None] * u[:, None, :] + s[..., None] * w[:, None, :]
    return x, lab


def simulate(p, variant="balanced", n=10, chains=4000, seed=1, generations=N_GEN):
    mu_b, s_b, eta, sig_p, v, sig_r, c_s = p[:7]
    b_lay = p[7] if len(p) > 7 else 0.0
    trend = p[8] if variant == 'register_trend' else 0.0
    s_phi = np.deg2rad(15)
    rng = np.random.default_rng(seed)
    C = chains
    out = {k: np.zeros((C, generations)) for k in ("b", "V", "S", "P")}
    phi_t = np.full(C, PHI_DSM)
    b_t = np.full(C, B_DSM)
    xV = np.array([v, v])
    for g in range(generations):
        phi = PHI_DSM + s_phi * rng.standard_normal(C)
        b = mu_b + s_b * rng.standard_normal(C)
        if variant in ("register", "register_trend") and g > 0:
            phi_t = np.full(C, PHI_DSM)
            b_t = np.full(C, b_lay + trend * (g - 1))
        pos_only = variant == "positive" and g > 0
        if variant == "base" and g > 0:      # lay texts sample people at the population base rate
            from network_model import examples
            x, lab = examples(rng, phi_t, b_t, n, "base")
        else:
            x, lab = _examples(rng, phi_t, b_t, n, pos_only)
        for j in range(n):                                     # sequential nudges
            u = np.stack([np.cos(phi), np.sin(phi)], -1)
            mine = (x[:, j] * u).sum(-1) > b
            dis = mine != lab[:, j]
            phi = np.where(dis, phi + eta * (phi_t - phi), phi)
            b = np.where(dis, b + eta * (b_t - b), b)
        u = np.stack([np.cos(phi), np.sin(phi)], -1)
        prev = np.clip(1 - norm.cdf(b), 1e-4, 1 - 1e-4)
        out["b"][:, g] = b
        out["P"][:, g] = np.log(prev / (1 - prev)) + sig_p * rng.standard_normal(C)
        out["V"][:, g] = ((u @ xV) - b + sig_r * rng.standard_normal(C)) > 0
        xs = rng.standard_normal((C, 2))
        out["S"][:, g] = ((u * xs).sum(-1) - b + c_s + sig_r * rng.standard_normal(C)) > 0
        phi_t, b_t = phi, b
    return out


def lag1(a):
    a = a - a.mean(0, keepdims=True)
    x, y = a[:, :-1].ravel(), a[:, 1:].ravel()
    sx, sy = x.std(), y.std()
    return 0.0 if sx == 0 or sy == 0 else float((x * y).mean() / (sx * sy))


def summarise(out):
    s = {}
    for k in "VSP":
        s[k + "_mean"] = out[k].mean(0)
        s[k + "_lag1"] = lag1(out[k])
    s["P_sd"] = out["P"].std(0)
    s["b_lag1"] = lag1(out["b"])
    s["b_mean"] = out["b"].mean(0)
    return s
