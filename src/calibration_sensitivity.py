"""Expert placement again, but with the parameters from the main chain fit instead of the rounded ones."""
import json,sys
sys.path.insert(0,'.')
import network_model as nm
p=json.load(open('../results/fit_s3adj_balanced_lag.json'))['params']
for k in ('mu_b','s_b','eta'):nm.FIT[k]=p[k]
out={'params':{k:p[k] for k in ('mu_b','s_b','eta')},'settings':{'N':200,'R':20,'T':300,'q':.5,'lambda':.1,'seed':3},'cases':{}}
for name,f,pl in [('none',0,'random'),('hubs2',.02,'hubs'),('random10',.1,'random'),('hubs5',.05,'hubs')]:
 r=nm.run('scalefree',f=f,placement=pl,T=300,seed=3)
 vals=r['final_prev'];out['cases'][name]={'mean':float(vals.mean()),'replicate_values':vals.tolist()}
 print(name,float(vals.mean()),flush=True)
json.dump(out,open('../results/calibration_sensitivity.json','w'),indent=2)
