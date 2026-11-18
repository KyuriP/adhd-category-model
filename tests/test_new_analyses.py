"""Checks for the gen-8 reading, the trend model and the gens 1-5 targets."""
from pathlib import Path
import sys, unittest
import numpy as np
import statsmodels.formula.api as smf
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from chain_model import simulate, B_DSM
from targets import frame, estimate, COVS
import network_model as nm

class NewAnalysisChecks(unittest.TestCase):
    def test_extra_reader_preserves_original_seven_generations(self):
        p=[.83,.28,.31,1.15,.5,.49,-.27]
        old=simulate(p,chains=60,seed=5)
        extra=simulate(p,chains=60,seed=5,generations=8)
        for k in old: np.testing.assert_array_equal(old[k],extra[k][:,:7])
        self.assertEqual(extra['b'].shape,(60,8))
    def test_zero_source_trend_is_constant_source(self):
        p=[.83,.28,.31,1.15,.5,.49,-.27,1.2]
        constant=simulate(p,'register',chains=60,seed=9)
        zero=simulate(p+[0.],'register_trend',chains=60,seed=9)
        trend=simulate(p+[-.05],'register_trend',chains=60,seed=9)
        for k in constant:
            np.testing.assert_array_equal(constant[k],zero[k])
            np.testing.assert_array_equal(constant[k][:,:2],trend[k][:,:2])
        self.assertFalse(np.array_equal(constant['b'][:,2:],trend['b'][:,2:]))
    def test_five_generation_targets_restandardise_and_resize_covariance(self):
        x=frame(3).query('gen <= 5');t=estimate(x,adjusted=True,bootstraps=50,seed=25)
        self.assertEqual((t['n'],t['n_chains']),(1100,220))
        self.assertEqual(np.asarray(t['moment_cov']).shape,(23,23))
        for k in 'VSP':
            m=smf.ols(f'{k} ~ C(gen) + C(cond) + '+' + '.join(COVS),x).fit()
            expected=[m.predict(x.assign(gen=g)).mean() for g in range(1,6)]
            np.testing.assert_allclose(t[k+'_mean'],expected,rtol=1e-10)
            self.assertEqual(t[k+'_lag1_n'],880)
    def test_same_day_denominators_omit_missing_responses(self):
        import json
        from preprocess import study_data
        from load import LIK
        x=study_data(4);x=x[x.Progress==100]
        r=json.loads((ROOT/'results/conditions.json').read_text())['study4']['observed']
        for key,col in [('V','Other.Diagnosis.Qs_1'),('S','Self.Diagnosis.Q_1')]:
            valid=x.assign(score=x[col].map(LIK)).dropna(subset=['score'])
            groups=[valid.loc[valid.gen==g,'score'].ge(4) for g in (1,7)]
            self.assertEqual([len(z) for z in groups],r[key][3:])
            self.assertEqual(sum(r[key][3:]),215)
            self.assertAlmostEqual(groups[1].mean()-groups[0].mean(),r[key][0])
    def test_fixed_source_offset_stays_fixed(self):
        for source in [B_DSM,2*nm.FIT['mu_b']-B_DSM]:
            r=nm.run('scalefree',N=12,R=2,T=5,f=.25,src_b=source,return_state=True,seed=4)
            np.testing.assert_array_equal(r['b'][r['expert']],source)

if __name__=='__main__':unittest.main()
