"""Does a text keep the DSM's strictness cues, and do readers of such texts draw stricter boundaries?"""
import sys, json, re, numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
sys.path.insert(0,'.')
from load import load
PAT = re.compile(r"\b(at least|six|6\b|months?|persistent|persist|severe|severity|significant(ly)?|impair\w*|interfer\w*|criteria|diagnos\w*|clinical\w*|professional|multiple settings|two or more|more than one setting|before (the )?age|childhood)\b", re.I)
def logit(p): p=np.clip(p,0.5,99.5)/100; return np.log(p/(1-p))
df=load(); out={}
for s in (2,3):
    d=df[(df.study==s)].copy(); d['P']=logit(d.prev); d['V']=(d.vincent_dx>=4).astype(int)
    d['cue']=d.stim.str.count(PAT)  # cues in the text this person READ
    d['any']=(d.cue>0).astype(int)
    share=d.groupby('gen')['any'].mean()
    r=d[d.gen>=2]
    mP=smf.ols('P ~ any + C(gen)*C(cond)',r).fit(cov_type='cluster',cov_kwds={'groups':r.chain})
    mV=smf.logit('V ~ any + C(gen) + C(cond)',r).fit(disp=0,cov_type='cluster',cov_kwds={'groups':r.chain})
    out[s]=dict(share_by_gen=share.round(3).tolist(),
      P_coef=float(mP.params['any']),P_ci=mP.conf_int().loc['any'].tolist(),P_p=float(mP.pvalues['any']),
      V_or=float(np.exp(mV.params['any'])),V_ci=np.exp(mV.conf_int().loc['any']).tolist(),V_p=float(mV.pvalues['any']),
      n=int(len(r)), n_any=int(r['any'].sum()))
    print(s, json.dumps(out[s]))
    # retention: given writer read cues, did they keep them?
    nxt=d[['chain','gen','any']].copy(); nxt['gen']-=1
    m=d.merge(nxt,on=['chain','gen'],suffixes=('','_wrote'))
    print('  P(wrote cue | read cue) by gen:', m.groupby(['gen','any'])['any_wrote'].mean().unstack().round(2).to_dict())
json.dump(out,open('../results/step2_threshold.json','w'),indent=1)
