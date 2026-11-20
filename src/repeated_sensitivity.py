"""Repeated-learning runs for Fig 3 and Tables S10-S12.

20 populations of 200 people, 600 rounds; summaries are means over the last 120 rounds.
Broad and strict sources are equally far from the mean prior on the offset scale, and
each source run is compared with a no-source run on the same lay people.
Results are checkpointed so a stopped run can be resumed.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import json
import sys
import time
import numpy as np
from scipy.special import ndtri

sys.path.insert(0,str(Path(__file__).resolve().parent))
from repeated_learning import run

RES=Path(__file__).resolve().parents[1]/'results'


def ci(values):
    values=np.asarray(values);mean=float(values.mean())
    se=float(values.std(ddof=1)/np.sqrt(len(values)))
    return [mean,mean-1.96*se,mean+1.96*se]


def scenarios():
    rows=[]
    for aligned in (True,False):
        for rule in ('copy','perceptron','margin'):
            for lam in (0.,.02,.1):
                for q in (.05,.2,.5,.9):
                    rows.append(dict(group='drift',rule=rule,aligned=aligned,lam=lam,q=q))
    for rule in ('copy','perceptron','margin'):
        for q in (.2,.5,.9):
            rows.append(dict(group='sources',rule=rule,aligned=False,lam=.1,q=q))
    # Vary the strength of the two alternative updates, without fitting it to data.
    for rule,param,values in [('perceptron','step',(.025,.1)),('margin','eta',(.14,.56))]:
        for value in values:
            for q in (.05,.5,.9):
                rows.append(dict(group='step',rule=rule,aligned=False,lam=.1,q=q,**{param:value}))
            rows.append(dict(group='sources_step',rule=rule,aligned=False,lam=.1,q=.5,**{param:value}))
    return rows


def case_key(cfg):return json.dumps(cfg,sort_keys=True)


def calculate(cfg):
    start=time.time();kw={k:v for k,v in cfg.items() if k!='group'}
    source_case=cfg['group'].startswith('sources')
    if source_case:kw['kind']='scalefree'
    no=run(**kw)
    out=dict(key=case_key(cfg),settings=cfg,trace=no['trace'],
        mean_prevalence=ci(no['prev'].mean(axis=1)),
        change_pp=ci(100*(no['prev']-no['initial_prev']).mean(axis=1)),
        last_window_change_pp=ci(100*(no['prev']-no['prev_previous']).mean(axis=1)),
        prevalence_replicates=no['prev'].mean(axis=1).tolist())
    if source_case:
        out['sources']={}
        for place in ('hubs','random'):
            strict=run(**kw,f=.05,placement=place,source_b=ndtri(.95))
            broad=run(**kw,f=.05,placement=place,source_b=2*.85-ndtri(.95))
            mask=~strict['expert']
            assert np.array_equal(mask,~broad['expert'])
            mean=lambda a:np.sum(np.where(mask,a,0),axis=1)/mask.sum(axis=1)
            narrowing=mean(strict['b']-no['b'])
            broadening=mean(no['b']-broad['b'])
            out['sources'][place]=dict(strict_shift=ci(narrowing),broad_shift=ci(broadening),
                broad_minus_strict=ci(broadening-narrowing),
                ratio=float(broadening.mean()/narrowing.mean()),
                strict_replicates=narrowing.tolist(),broad_replicates=broadening.tolist())
    out['secs']=time.time()-start
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--restart',action='store_true',help='Recompute every planned scenario')
    args=ap.parse_args();path=RES/'repeated_sensitivity.json'
    out=json.loads(path.read_text()) if path.exists() and not args.restart else dict(cases=[])
    out['design']=dict(N=200,R=20,T=600,last_rounds=120,seed=71,mu_b=.85,s_b=.31,
        angle_spread_degrees=15,eta=.28,step=.05,examples=10,
        source_fraction=.05,diagnostic_reference=.05,
        comparisons='Same initial populations, graphs and source locations across source conditions; matched lay-agent baseline')
    out['planned_scenarios']=scenarios()
    done={c['key'] for c in out['cases']}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(calculate,cfg) for cfg in scenarios() if case_key(cfg) not in done]
        for f in as_completed(jobs):
            c=f.result();out['cases'].append(c);path.write_text(json.dumps(out,indent=2))
            print(len(out['cases']),len(scenarios()),c['key'],
                  'change pp',round(c['change_pp'][0],3),flush=True)
    assert len(out['cases'])==len(out['planned_scenarios'])


if __name__=='__main__':main()
