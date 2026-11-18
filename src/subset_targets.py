"""Targets for generations 1-5 only (drop gens 6 and 7)."""
import json
import sys
sys.path.insert(0, '.')
from targets import frame, estimate
path='../results/targets.json'
targets=json.load(open(path))
x=frame(3)
targets['3adj_gens5']=estimate(x.loc[x.gen<=5].copy(),adjusted=True,seed=20261125)
json.dump(targets,open(path,'w'))
print('Five-generation target',targets['3adj_gens5']['n'],len(targets['3adj_gens5']['moment_cov']),flush=True)
