"""Re-evaluate the fitted losses with 20,000 chains at three new seeds.
Same data, so this only checks simulation noise, not out-of-sample fit.
"""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'.')
from fit import loss,T,parameter_names
seeds=[17,19,23];out={}
for path in sorted(Path('../results').glob('fit_s*_lag.json')):
 d=json.load(open(path))
 if not d.get('target_version'):continue
 names=parameter_names(d['variant'])
 p=[d['params'][k] for k in names]
 values=[float(loss(p,T[d['study']],d['variant'],True,chains=20000,seed=s)) for s in seeds]
 out[path.stem]={'chains':20000,'seeds':seeds,'losses':values,'mean_loss':float(np.mean(values))}
 d['validation_loss_20000']=values[0];d['simulation_check']=out[path.stem]
 json.dump(d,open(path,'w'),indent=1)
 print(path.stem,round(d['loss'],2),np.round(values,2).tolist(),flush=True)
json.dump(out,open('../results/simulation_check.json','w'),indent=2)
