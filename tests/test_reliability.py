import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from reliability_study import observations, error_metrics
from inverse_pulsatile_physics import analytical_fit, TRUE_COEFFICIENTS


class ReliabilityTests(unittest.TestCase):
    def test_counts_nested_locations_and_paired_noise(self):
        x,y=observations(96,.1,42)
        for n in (12,24,48):
            a,b=observations(n,.1,42)
            np.testing.assert_array_equal(a,x[:n])
            np.testing.assert_array_equal(b,y[:n])
            self.assertFalse(np.any((a[:,1]>=.3)&(a[:,1]<=.65)))

    def test_noiseless_analytical_recovery(self):
        for n in (12,24,48,96):
            x,y=observations(n,0,42)
            c,_=analytical_fit(x,y)
            np.testing.assert_allclose(c,TRUE_COEFFICIENTS,atol=2e-6)

    def test_metric_definitions(self):
        truth={'velocity':np.ones((3,2))*1.5,'flow':np.ones(3),'shear':np.ones(3)*2}
        pred={k:v*1.1 for k,v in truth.items()}
        values=error_metrics(pred,truth,np.array([False,True,False]))
        for value in values.values(): self.assertAlmostEqual(value,10)


if __name__=='__main__': unittest.main()
