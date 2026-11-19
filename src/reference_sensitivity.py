"""Refit the chain models at diagnostic references of 2.5%, 5% and 10%,
and rerun the expert-placement comparison at each reference.
Same targets, bounds and search budget as the main fits; failed fits are kept in the output.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
import sys
import time
import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chain_model import simulate, summarise
import network_model as nm

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / 'results'
NAMES = ['mu_b', 's_b', 'eta', 'sig_p', 'v', 'sig_r', 'c_s']
BOUNDS = [(-1, 3), (.01, 2), (0, 1), (.01, 3), (-1, 3), (.05, 3), (-3, 3)]


def loss(p, target, variant, reference, chains=3000, seed=7):
    s = summarise(simulate(p, variant, chains=chains, seed=seed,
                          diagnostic_prev=reference))
    value = 0.
    for k in 'VSP':
        value += np.sum(((s[k+'_mean']-target[k+'_mean']) / target[k+'_se'])**2)
        value += ((s[k+'_lag1']-target[k+'_lag1']) / target[k+'_lag1_se'])**2
    value += np.sum(((s['P_sd']-target['P_sd']) / target['P_sd_se'])**2)
    return float(value)


def fit_case(reference, variant):
    start = time.time()
    target = json.loads((RES/'targets.json').read_text())['3adj']
    names = NAMES + (['b_lay', 'trend'] if variant == 'register_trend' else [])
    base = json.loads((RES/f'fit_s3adj_{variant}_lag.json').read_text())
    x0 = [base['params'][k] for k in names]
    if reference == .05:
        params = x0
        fit_loss = base['loss']
        optimizer = base['optimizer']
    else:
        bounds = BOUNDS + ([(-1, 3), (-.5, .5)] if variant == 'register_trend' else [])
        r = differential_evolution(loss, bounds,
            args=(target, variant, reference), seed=3, maxiter=60,
            popsize=12, tol=1e-6, polish=False, x0=x0)
        params, fit_loss = r.x, float(r.fun)
        optimizer = dict(seed=3, maxiter=60, popsize=12, nit=int(r.nit),
                         nfev=int(r.nfev), converged=bool(r.success), message=str(r.message))
    checks = [loss(params, target, variant, reference, 20000, seed) for seed in (17,19,23)]
    p = dict(zip(names, map(float, params)))
    return dict(reference=reference, variant=variant, params=p, loss=fit_loss,
                initial_params=dict(zip(names, x0)), optimizer=optimizer,
                simulation_check=dict(chains=20000, seeds=[17,19,23], losses=checks,
                                      mean_loss=float(np.mean(checks))),
                prior_at_mean=float(norm.sf(p['mu_b'])),
                mean_prior=float(norm.sf(p['mu_b']/np.sqrt(1+p['s_b']**2))),
                secs=time.time()-start)


def network_case(row):
    old = dict(nm.FIT)
    nm.FIT.update({k:row['params'][k] for k in ('mu_b','s_b','eta')})
    try:
        cases = {}
        for name, f, place in [('none',0,'random'),('hubs2',.02,'hubs'),
                                ('random10',.1,'random'),('hubs5',.05,'hubs')]:
            r = nm.run('scalefree', N=200, R=20, T=600, f=f, placement=place,
                       pos_frac=.5, lam=.1, seed=53, src_b=norm.isf(row['reference']))
            cases[name] = r['final_prev'].tolist()
        difference = np.array(cases['random10'])-cases['hubs2']
        se = difference.std(ddof=1)/np.sqrt(len(difference))
        return dict(reference=row['reference'], cases=cases,
                    hubs2_advantage_pp=[float(100*v) for v in
                      (difference.mean(),difference.mean()-1.96*se,difference.mean()+1.96*se)])
    finally:
        nm.FIT.update(old)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--restart',action='store_true',help='Recompute all added fits and network cases')
    ap.add_argument('--reuse-fits',action='store_true',help='Keep saved reference fits and rerun network cases')
    args=ap.parse_args()
    path=RES/'reference_sensitivity.json'
    out=json.loads(path.read_text()) if path.exists() and not args.restart else dict(fits=[],networks=[])
    if args.reuse_fits:
        if len(out['fits'])!=6:raise ValueError('Six saved reference fits are required')
        out['networks']=[]
    out['settings']=dict(references=[.025,.05,.1], network=dict(N=200,R=20,T=600,
        last_rounds=120,q=.5,reversion=.1,seed=53), target_version='2026-11-18-chain-bootstrap-2000')
    done={(r['reference'],r['variant']) for r in out['fits']}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(fit_case,ref,v) for ref in (.025,.05,.1)
              for v in ('balanced','register_trend') if (ref,v) not in done]
        for f in as_completed(jobs):
            row=f.result();out['fits'].append(row)
            path.write_text(json.dumps(out,indent=2))
            print('fit',row['reference'],row['variant'],row['simulation_check']['mean_loss'],flush=True)
    done={r['reference'] for r in out['networks']}
    for row in sorted(out['fits'],key=lambda r:r['reference']):
        if row['variant']=='balanced' and row['reference'] not in done:
            result=network_case(row);out['networks'].append(result)
            path.write_text(json.dumps(out,indent=2))
            print('network',result['reference'],result['hubs2_advantage_pp'],flush=True)


if __name__=='__main__':main()
