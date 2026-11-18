import json, sys, numpy as np
from scipy.optimize import minimize
sys.path.insert(0,'.')
from fit import loss, T
from chain_model import simulate, summarise
best=json.load(open('../results/fit_s3_balanced_nolag.json'))['params']
x0=np.array([best[k] for k in ['mu_b','s_b','sig_p','v','sig_r','c_s']])
t=T['3']; rows=[]
for eta in [0.0,0.05,0.1,0.2,0.3,0.36,0.45,0.6,0.8]:
    f=lambda q: loss([q[0],abs(q[1]),eta,abs(q[2]),q[3],abs(q[4]),q[5]],t,'balanced',False)
    r=minimize(f,x0,method='Nelder-Mead',options={'maxfev':500,'xatol':1e-3,'fatol':1e-2})
    q=r.x; p=[q[0],abs(q[1]),eta,abs(q[2]),q[3],abs(q[4]),q[5]]
    s=summarise(simulate(p,'balanced',chains=20000,seed=11))
    rows.append(dict(eta=eta,loss=float(r.fun),b_lag1=s['b_lag1'],V_lag1=s['V_lag1'],P_lag1=s['P_lag1'],S_lag1=s['S_lag1'],mu_b=q[0]))
    print({k:round(v,3) for k,v in rows[-1].items()},flush=True)
json.dump(rows,open('../results/profile_s3.json','w'))
