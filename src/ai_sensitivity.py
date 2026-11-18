"""Sensitivity check: drop participants touched by AI (self-reported AI use, hidden trap word,
or copy-paste), and everyone downstream of them in their chain."""
import json, sys, numpy as np, pandas as pd, statsmodels.formula.api as smf
sys.path.insert(0, '.')
from load import LIK
from preprocess import study_data
from text_coding import code
def logit(p): p=np.clip(p,0.5,99.5)/100; return np.log(p/(1-p))
YN=lambda x:(x.astype(str)=='Yes').astype(float)
T=json.load(open('../results/targets.json')); out={}
for s in (2,3):
    d=study_data(s)
    flag=(d.AI_Self_Report=='Yes')|d.prompt_inject.fillna(False).astype(bool)|d.filter(like='_pasted').fillna(False).astype(bool).any(axis=1)
    first=d[flag].groupby('chain').gen.min(); f=d.chain.map(first)
    x=pd.DataFrame({'chain':d.chain,'gen':d.gen,'cond':d['prompt'] if s==2 else d['motivation'],
      'V':(d['Other.Diagnosis.Qs_1'].map(LIK)>=4).astype(float),'S':(d['Self.Diagnosis.Q_1'].map(LIK)>=4).astype(float),
      'P':logit(pd.to_numeric(d['Q_Prevalence.Rate'],errors='coerce')),'age':pd.to_numeric(d['Age.Screener'],errors='coerce'),
      'female':(d.Gender=='Female').astype(float),'past_dx':YN(d.Past_MH_diagnosis),'nhs':YN(d.Health_Access),
      'know_tx':YN(d.Prior_Know1),'know_wait':YN(d.Prior_Know2),'follows_nd':YN(d.Prior_Know3),'stim':d.stimulus.fillna('')})
    x=x[f.isna()|(d.gen<f)].dropna(subset=['P','age']).copy()
    import re; x['stim']=x.stim.map(lambda t: re.sub(r'\s+',' ',re.sub(r'<[^>]+>',' ',t)).strip())
    covs='age + female + past_dx + nhs + know_tx + know_wait + follows_nd'
    res={'n':int(len(x)),'by_gen_n':x.groupby('gen').size().tolist(),'trend':{},'adj':{},'raw':{}}
    for y in 'VSP':
        m0=smf.ols(f'{y} ~ gen + C(cond)',x).fit(cov_type='cluster',cov_kwds={'groups':x.chain})
        m1=smf.ols(f'{y} ~ gen + C(cond) + {covs}',x).fit(cov_type='cluster',cov_kwds={'groups':x.chain})
        mg=smf.ols(f'{y} ~ C(gen) + C(cond) + {covs}',x).fit()
        res['trend'][y]={'raw':[float(m0.params.gen),*m0.conf_int().loc['gen']],'adj':[float(m1.params.gen),*m1.conf_int().loc['gen']]}
        res['adj'][y]=[float(mg.predict(x.assign(gen=g)).mean()) for g in range(1,8)]
        res['raw'][y]=x.groupby('gen')[y].mean().tolist()
    # text -> reader (gens 2-7)
    r=x[x.gen>=2].copy(); r=pd.concat([r.reset_index(drop=True),pd.DataFrame([code(t) for t in r.stim])],axis=1); r['lwords']=np.log(r.words)
    Z=['dsm_n','extra_n','threshold','common','you','hedge','lwords']
    for z in Z: r[z+'_z']=(r[z]-r[z].mean())/r[z].std()
    m=smf.ols('V ~ '+' + '.join(z+'_z' for z in Z)+' + C(gen)*C(cond)',r).fit(cov_type='cluster',cov_kwds={'groups':r.chain})
    res['reader_V']={z:[float(m.params[z+'_z']),float(m.pvalues[z+'_z'])] for z in Z}
    res['reader_V_joint_p']=float(m.f_test(' = '.join(z+'_z' for z in Z)+' = 0').pvalue)
    # targets.py independently recomputes all 31 moments after the filter.
    out[s]=res
    print(f'== Study {s}: n={len(x)} by gen {res["by_gen_n"]}')
    for y in 'VSP':
        a=res['trend'][y]; print(f'   {y}: raw {a["raw"][0]:+.4f} [{a["raw"][1]:+.4f},{a["raw"][2]:+.4f}]  adj {a["adj"][0]:+.4f} [{a["adj"][1]:+.4f},{a["adj"][2]:+.4f}]  adj means {np.round(res["adj"][y],3).tolist()}')
    print('   reader V joint p=%.3f'%res['reader_V_joint_p'], {k:(round(v[0],3),round(v[1],3)) for k,v in res['reader_V'].items()})
json.dump(out,open('../results/ai_sensitivity.json','w'),indent=1)
