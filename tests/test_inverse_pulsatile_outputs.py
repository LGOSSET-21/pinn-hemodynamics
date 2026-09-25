import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from inverse_pulsatile_outputs import evaluate
from inverse_pulsatile_physics import simulate


class EvaluationTests(unittest.TestCase):
    def test_metrics_use_learned_coefficients_and_signed_shear(self):
        class Steady(torch.nn.Module):
            def forward(self,x):
                return 1.5*(1-x[:,0:1]**2)
        x,y=simulate(42,0)
        fields,metrics=evaluate(Steady(),np.array([6.,0.,0.]),x,y)
        np.testing.assert_allclose(fields["force_pred"],6.)
        np.testing.assert_allclose(fields["flow_pred"],0.75,atol=1e-4)
        np.testing.assert_allclose(fields["shear_pred"],3.,atol=1e-6)
        self.assertGreater(metrics["pinn"]["force_relative_l2"],0.3)
        self.assertLess(metrics["analytical"]["force_relative_l2"],1e-5)


if __name__=="__main__":
    unittest.main()
