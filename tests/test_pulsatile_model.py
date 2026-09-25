import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from train_pulsatile import PulsatileNet, residual


class ModelTests(unittest.TestCase):
    def test_hard_conditions_for_untrained_model(self):
        model=PulsatileNet().double()
        x=torch.tensor([[0.,0.23],[1.,0.41],[0.6,0.12],[0.6,1.12]],dtype=torch.float64,requires_grad=True)
        u=model(x)
        grad=torch.autograd.grad(u.sum(),x,create_graph=True)[0]
        self.assertAlmostEqual(grad[0,0].item(),0,places=12)
        self.assertAlmostEqual(u[1].item(),0,places=12)
        self.assertAlmostEqual(u[2].item(),u[3].item(),places=12)

    def test_residual_includes_time_derivative_and_forcing(self):
        # Champ arbitraire, calcul indépendant : u=(1-r²)*(1+t).
        class Known(torch.nn.Module):
            def forward(self,x):
                return (1-x[:,0:1]**2)*(1+x[:,1:2])
        x=torch.tensor([[0.25,0.2],[0.75,0.6]],dtype=torch.float64,requires_grad=True)
        r,t=x[:,0:1],x[:,1:2]
        expected=r*(9/(2*torch.pi)*(1-r**2)+4*(1+t)-6-4*torch.cos(2*torch.pi*t))
        torch.testing.assert_close(residual(Known(),x),expected)


if __name__ == "__main__":
    unittest.main()
