<<<<<<< HEAD
"""Checks for the reference refits, the label-only rules and the matched populations."""
=======
<<<<<<< HEAD
"""Checks for the reference refits, the label-only rules and the matched populations."""
=======
"""Checks for reference changes, label-only information access and matched populations."""
>>>>>>> 20bc25dc208597d275872f1132daa9fbab6be5c8
>>>>>>> e0be0d0288f614a42c68d8d84b30840184505552
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from chain_model import simulate
from repeated_learning import learn,run
import network_model as nm


class SensitivityChecks(unittest.TestCase):
    def test_reference_is_irrelevant_without_learning(self):
        p=[.85,.31,0,1.15,.5,.5,-.28]
        low=simulate(p,chains=100,seed=4,diagnostic_prev=.025)
        high=simulate(p,chains=100,seed=4,diagnostic_prev=.1)
        for k in low:np.testing.assert_array_equal(low[k],high[k])
    def test_label_only_rules_do_not_read_source_parameters(self):
        rng=np.random.default_rng(17)
        b=rng.normal(size=30);phi=rng.normal(size=30)
        x=rng.normal(size=(30,10,2));labels=rng.random((30,10))<.5
        for aligned in (True,False):
            for rule in ('perceptron','margin'):
                a=learn(b,phi,x,labels,rule,aligned=aligned,bt=np.full(30,99.),pt=np.zeros(30))
                z=learn(b,phi,x,labels,rule,aligned=aligned,bt=np.full(30,-99.),pt=np.ones(30))
                for av,zv in zip(a,z):np.testing.assert_array_equal(av,zv)
    def test_copying_control_matches_existing_population_model(self):
        old=nm.run('complete',N=12,R=2,T=240,seed=71,lam=.1,pos_frac=.5)
        new=run('copy',N=12,R=2,T=240,seed=71,lam=.1,q=.5)
        np.testing.assert_allclose(old['mean_prev_t'][::20],
            [a['mean_prevalence'] for a in new['trace']],atol=1e-12)
    def test_source_conditions_keep_initial_populations_and_fixed_sources(self):
        kw=dict(N=12,R=2,T=240,seed=71,kind='scalefree')
        no=run(**kw)
        source=run(**kw,rule='margin',f=.25,source_b=1.5)
        np.testing.assert_array_equal(no['initial_prev'],source['initial_prev'])
        np.testing.assert_allclose(source['b'][source['expert']],1.5,atol=1e-12)


if __name__=='__main__':unittest.main()
