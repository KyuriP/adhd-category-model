"""Fit targets: the 31 moments per dataset, with SEs and covariance from 2,000 bootstraps over chains.
Adjusted means: OLS on generation + covariates, predicted over the whole sample.
"""
import json, sys
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
from load import LIK
from preprocess import study_data
COVS = ['age', 'female', 'past_dx', 'nhs', 'know_tx', 'know_wait', 'follows_nd']
YN = lambda x: (x.astype(str) == 'Yes').astype(float)

def logit(p):
    p = np.clip(p, 0.5, 99.5) / 100
    return np.log(p / (1 - p))

def frame(study, clean=False):
    d = study_data(study)
    if clean:
        flag = ((d.AI_Self_Report == 'Yes') | d.prompt_inject.fillna(False).astype(bool)
                | d.filter(like='_pasted').fillna(False).astype(bool).any(axis=1))
        first = d.loc[flag].groupby('chain').gen.min(); cutoff = d.chain.map(first)
        d = d.loc[cutoff.isna() | (d.gen < cutoff)].copy()
    x = pd.DataFrame({
        'chain': d.chain, 'gen': d.gen.astype(int),
        'cond': d['prompt'] if study == 2 else d['motivation'],
        'V': d['Other.Diagnosis.Qs_1'].map(LIK), 'S': d['Self.Diagnosis.Q_1'].map(LIK),
        'P': logit(pd.to_numeric(d['Q_Prevalence.Rate'], errors='coerce')),
        'age': pd.to_numeric(d['Age.Screener'], errors='coerce'),
        'female': (d.Gender == 'Female').astype(float),
        'past_dx': YN(d.Past_MH_diagnosis), 'nhs': YN(d.Health_Access),
        'know_tx': YN(d.Prior_Know1), 'know_wait': YN(d.Prior_Know2), 'follows_nd': YN(d.Prior_Know3)})
    for k in 'VS': x[k] = x[k].ge(4).astype(float).where(x[k].notna())
    if x[['V','S','P'] + COVS].isna().any().any():
        raise ValueError('Missing analysis values require an explicit missing-data policy')
    if x.duplicated(['chain','gen']).any(): raise ValueError('Duplicate chain-generation')
    return x.reset_index(drop=True)

def estimate(x, adjusted=False, bootstraps=2000, seed=20261118):
    y = x[['V','S','P']].to_numpy(float); gen = x.gen.to_numpy() - 1
    generations = int(gen.max()) + 1
    if set(gen) != set(range(generations)): raise ValueError('Generations must be consecutive from 1')
    chain, names = pd.factorize(x.chain, sort=True); cond, _ = pd.factorize(x.cond, sort=True)
    nc = len(names); group = gen * (cond.max()+1) + cond; ng = group.max()+1
    ix = np.full((nc,generations), -1, dtype=int); ix[chain,gen] = np.arange(len(x))
    pairs = [(ix[:,g],ix[:,g+1]) for g in range(generations-1)]
    a = np.concatenate([u[(u>=0)&(v>=0)] for u,v in pairs])
    b = np.concatenate([v[(u>=0)&(v>=0)] for u,v in pairs])
    design = np.column_stack([np.ones(len(x)), *(gen==g for g in range(1,generations)),
        *(cond==c for c in range(1,cond.max()+1)), x[COVS].to_numpy()]).astype(float)
    def vector(counts):
        w = counts[chain]; n = np.bincount(gen, weights=w, minlength=generations)
        means = np.array([np.bincount(gen,weights=w*y[:,k],minlength=generations)/n for k in range(3)])
        sd = np.sqrt(np.bincount(gen,weights=w*(y[:,2]-means[2,gen])**2,minlength=generations)/(n-1))
        if adjusted:
            beta = np.linalg.lstsq(design*np.sqrt(w[:,None]),y*np.sqrt(w[:,None]),rcond=None)[0]
            pred = np.tile(np.average(design,axis=0,weights=w),(generations,1))
            pred[:,1:generations] = np.vstack([np.zeros(generations-1),np.eye(generations-1)])
            means = (pred@beta).T
        gn = np.bincount(group,weights=w,minlength=ng)
        gm = np.column_stack([np.bincount(group,weights=w*y[:,k],minlength=ng)/gn for k in range(3)])
        residual=y-gm[group]; pw=w[a]; u,v=residual[a],residual[b]
        u=u-np.average(u,axis=0,weights=pw);v=v-np.average(v,axis=0,weights=pw)
        lag=(pw[:,None]*u*v).sum(0)/np.sqrt((pw[:,None]*u**2).sum(0)*(pw[:,None]*v**2).sum(0))
        return np.concatenate([means.ravel(),sd,lag])
    point=vector(np.ones(nc));rng=np.random.default_rng(seed)
    draws=np.array([vector(rng.multinomial(nc,np.full(nc,1/nc))) for _ in range(bootstraps)])
    if not np.isfinite(draws).all():raise ValueError('Non-finite bootstrap moments')
    cov=np.cov(draws,rowvar=False,ddof=1);se=np.sqrt(np.diag(cov))
    out={'n':len(x),'n_chains':nc,'by_gen_n':x.groupby('gen').size().tolist(),
         'adjusted':adjusted,'bootstrap_replicates':bootstraps,'bootstrap_seed':seed,
         'moment_cov':cov.tolist(),'P_sd':point[3*generations:4*generations].tolist(),'P_sd_se':se[3*generations:4*generations].tolist()}
    for k,key in enumerate('VSP'):
        sl=slice(k*generations,(k+1)*generations)
        out[key+'_mean']=point[sl].tolist();out[key+'_se']=se[sl].tolist()
        out[key+'_lag1']=float(point[4*generations+k]);out[key+'_lag1_se']=float(se[4*generations+k]);out[key+'_lag1_n']=len(a)
    return out

if __name__=='__main__':
    T,A={},{}
    for study in (2,3):
        x=frame(study);T[str(study)]=estimate(x,seed=20261118+study)
        adj=estimate(x,adjusted=True,seed=20261118+study)
        A[str(study)]={k:{'raw':T[str(study)][k+'_mean'],'adj':adj[k+'_mean']} for k in 'VSP'}
        if study==3:T['3adj']=adj
        T[f'{study}clean']=estimate(frame(study,clean=True),adjusted=True,seed=20261218+study)
        print(study,T[str(study)]['n'],T[f'{study}clean']['n'],flush=True)
    json.dump(T,open('../results/targets.json','w'))
    json.dump(A,open('../results/adjusted_means.json','w'))
