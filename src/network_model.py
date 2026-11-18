"""Network version of the boundary model, using fixed scenario parameters near the Study 3 fit.

Agents hold a linear boundary (angle phi, offset b) in a 2D trait space; a person
x ~ N(0, I) is 'ADHD' for an agent if u(phi).x > b, so the agent's implied
prevalence is 1 - Phi(b).

Each round, every lay agent reads one 'post' from a random neighbour: n examples
labelled by that neighbour. On every example where the reader's own label differs,
it moves (phi, b) a fraction eta toward the neighbour's (same rule as the chain fit).
Then the reader relaxes a fraction lam back toward its own lay prior.

  lam  = reversion to the personal lay prior per round. NOT identified by the chain
         data (each participant was observed once), so it is swept.
  pos_frac = share of a post's examples that are positive cases. Also not identified:
         the chain fit used 0.5, and positive-only fitted equally well. Swept.
  Expert agents sit at the DSM boundary and never update.

Settings (rounded, from an earlier fit without the lag moments): lay prior offset mu_b = 0.85
(about 20% prevalence), spread s_b = 0.31, eta = 0.28, n = 10 examples.
"""
import numpy as np
import networkx as nx
from scipy.stats import norm
from chain_model import B_DSM, PHI_DSM

FIT = dict(mu_b=0.85, s_b=0.31, eta=0.28, n=10, s_phi=np.deg2rad(15))


def examples(rng, phi_t, b_t, n, pos_frac):
    """n people labelled by the teacher. pos_frac = share of examples that are positive
    ('this is ADHD'); pos_frac='base' samples people from the population, so positives
    appear at the teacher's own base rate."""
    C = phi_t.shape[0]
    u = np.stack([np.cos(phi_t), np.sin(phi_t)], -1)
    w = np.stack([-np.sin(phi_t), np.cos(phi_t)], -1)
    pb = norm.cdf(b_t)[:, None]
    q = (1 - pb) if pos_frac == "base" else pos_frac
    lab = rng.random((C, n)) < q
    U = rng.random((C, n))
    t = np.clip(np.where(lab, norm.ppf(pb + U * (1 - pb)), norm.ppf(U * pb)), -8, 8)
    s = rng.standard_normal((C, n))
    return t[..., None] * u[:, None, :] + s[..., None] * w[:, None, :], lab


def make_graph(kind, N, seed):
    if kind == "complete":
        return nx.complete_graph(N)
    if kind == "ring":
        return nx.watts_strogatz_graph(N, 4, 0.0, seed=seed)
    if kind == "star":
        return nx.star_graph(N - 1)            # node 0 is the hub
    if kind == "scalefree":
        return nx.barabasi_albert_graph(N, 2, seed=seed)
    if kind == "modular":          # 25-agent communities, dense inside, sparse between
        return nx.random_partition_graph([25] * (N // 25), 0.3, 0.004, seed=seed)
    if kind == "sfclustered":      # scale-free with triangles (Holme-Kim)
        return nx.powerlaw_cluster_graph(N, 2, 0.5, seed=seed)
    raise ValueError(kind)


def build(kind, N, R, seed):
    """R independent graphs stacked into one padded neighbour table."""
    graphs = [make_graph(kind, N, seed + r) for r in range(R)]
    deg = np.array([[g.degree(i) for i in range(N)] for g in graphs])         # (R, N)
    maxd = max(1, int(deg.max()))
    nb = np.zeros((R * N, maxd), dtype=np.int64)
    for r, g in enumerate(graphs):
        for i in range(N):
            v = np.fromiter(g.neighbors(i), dtype=np.int64)
            nb[r * N + i, : len(v)] = v + r * N
    return graphs, deg, nb


def place_experts(graphs, deg, f, placement, rng):
    R, N = deg.shape
    k = int(round(f * N))
    ex = np.zeros((R, N), bool)
    for r in range(R):
        if k == 0:
            continue
        if placement == "random":
            ex[r, rng.choice(N, k, replace=False)] = True
        elif placement == "hubs":
            ex[r, np.argsort(-deg[r], kind="stable")[:k]] = True
        elif placement == "onecomm":      # all experts inside the first community (modular graphs)
            ex[r, :k] = True
        elif placement == "periphery":
            ex[r, np.argsort(deg[r], kind="stable")[:k]] = True
    return ex


def run(kind="complete", N=200, R=20, T=200, f=0.0, placement="random", lam=0.1,
        eta=None, pos_frac=0.5, campaign=False, seed=0, record_every=1, return_state=False, src_b=None):
    p = dict(FIT)
    if eta is not None:
        p["eta"] = eta
    rng = np.random.default_rng(seed)
    graphs, deg, nb = build(kind, N, R, seed)
    ex = place_experts(graphs, deg, f, placement, rng).ravel()
    M = R * N
    prior_b = p["mu_b"] + p["s_b"] * rng.standard_normal(M)
    prior_phi = PHI_DSM + p["s_phi"] * rng.standard_normal(M)
    b, phi = prior_b.copy(), prior_phi.copy()
    b[ex], phi[ex] = (B_DSM if src_b is None else src_b), PHI_DSM
    degf = deg.ravel()
    lay = ~ex

    def read(b, phi, bt, pt):
        x, lab = examples(rng, pt, bt, p["n"], pos_frac)
        for j in range(p["n"]):
            u = np.stack([np.cos(phi), np.sin(phi)], -1)
            dis = ((x[:, j] * u).sum(-1) > b) != lab[:, j]
            phi = np.where(dis, phi + p["eta"] * (pt - phi), phi)
            b = np.where(dis, b + p["eta"] * (bt - b), b)
        return b, phi

    if campaign:   # everyone reads the DSM criteria once at t = 0
        b2, phi2 = read(b, phi, np.full(M, B_DSM), np.full(M, PHI_DSM))
        b, phi = np.where(lay, b2, b), np.where(lay, phi2, phi)

    traj = []
    for t in range(T + 1):
        if t % record_every == 0:
            prev = 1 - norm.cdf(b)
            traj.append(np.where(lay, prev, np.nan).reshape(R, N))
        if t == T:
            break
        teacher = nb[np.arange(M), (rng.random(M) * degf).astype(np.int64)]
        # An isolated agent has no exposure, rather than reading an unrelated node.
        teacher = np.where(degf == 0, np.arange(M), teacher)
        b2, phi2 = read(b, phi, b[teacher], phi[teacher])
        b2 = b2 + lam * (prior_b - b2)
        phi2 = phi2 + lam * (prior_phi - phi2)
        b, phi = np.where(lay, b2, b), np.where(lay, phi2, phi)
    traj = np.array(traj)                                   # (time, R, N)
    out = {"mean_prev_t": np.nanmean(traj, axis=(1, 2)),
           "final_prev": np.nanmean(traj[-max(1, len(traj) // 5):], axis=(0, 2))}   # per replicate
    if return_state:
        prev_state = np.mean(np.nan_to_num(traj[-max(1, len(traj) // 5):]), axis=0)
        prev_state[ex.reshape(R, N)] = np.nan
        out.update(graphs=graphs, expert=ex.reshape(R, N), prev=prev_state,
                   b=b.reshape(R, N))
    return out
