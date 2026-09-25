import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from pulsatile_outputs import evaluate


class OutputTests(unittest.TestCase):
    def test_flow_and_shear_use_correct_radius_weight_and_sign(self):
        class Steady(torch.nn.Module):
            def forward(self,x):
                return 1.5*(1-x[:,0:1]**2)
        fields,metrics=evaluate(Steady())
        np.testing.assert_allclose(fields["flow_pred"],0.75,atol=1e-4)
        np.testing.assert_allclose(fields["shear_pred"],3.,atol=1e-6)
        hidden=fields["hidden"]
        self.assertTrue(hidden.any())
        self.assertTrue(np.all(fields["t"][hidden]>=0.3))
        self.assertTrue(np.all(fields["t"][hidden]<=0.65))
        self.assertTrue(np.isfinite(metrics["hidden_velocity_rmse"]))


if __name__=="__main__":
    unittest.main()
