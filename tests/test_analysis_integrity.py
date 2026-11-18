"""Sanity checks on the sample, targets and model."""
import os
from pathlib import Path
import sys
import unittest
import numpy as np
import networkx as nx
import statsmodels.formula.api as smf
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from targets import frame, estimate, COVS
from chain_model import simulate
import network_model as nm

class IntegrityChecks(unittest.TestCase):
    def test_published_sample_and_cleaning(self):
        s2,s3=frame(2),frame(3)
        self.assertEqual((len(s2),s2.chain.nunique()),(1519,217))
        self.assertEqual((len(s3),s3.chain.nunique()),(1540,220))
        self.assertNotIn(119,s2.chain.tolist())
        self.assertLess(s3.age.max(),120)
        self.assertEqual(len(frame(3,clean=True)),1168)
    def test_standardisation_and_joint_uncertainty(self):
        x=frame(3);t=estimate(x,adjusted=True,bootstraps=50,seed=91)
        for k in 'VSP':
            m=smf.ols(f'{k} ~ C(gen) + C(cond) + '+' + '.join(COVS),x).fit()
            expected=[m.predict(x.assign(gen=g)).mean() for g in range(1,8)]
            np.testing.assert_allclose(t[k+'_mean'],expected,rtol=1e-10)
        cov=np.array(t['moment_cov'])
        self.assertEqual(cov.shape,(31,31))
        self.assertGreaterEqual(np.linalg.eigvalsh(cov).min(),-1e-12)
        self.assertTrue(np.all(np.diag(cov)>0))
    def test_no_learning_keeps_each_prior(self):
        p=[.85,.31,0,1.15,.5,.5,-.28]
        a=simulate(p,'balanced',chains=100,seed=4)
        b=simulate(p,'positive',chains=100,seed=4)
        np.testing.assert_array_equal(a['b'],b['b'])
    def test_isolated_nodes_do_not_read_other_graphs(self):
        old=nm.make_graph
        try:
            nm.make_graph=lambda kind,N,seed:nx.empty_graph(N)
            a=nm.run(N=5,R=2,T=3,lam=0,seed=18)
            np.testing.assert_allclose(a['mean_prev_t'],a['mean_prev_t'][0],atol=1e-14)
        finally:nm.make_graph=old

if __name__=='__main__':unittest.main()
