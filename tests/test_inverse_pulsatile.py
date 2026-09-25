import sys
import unittest
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from inverse_pulsatile_physics import field,force,flow,shear,simulate,analytical_fit,describe
from train_inverse_pulsatile import inverse_residual


class InversePhysicsTests(unittest.TestCase):
    def test_nonzero_sine_component_satisfies_pde(self):
        c=np.array([5.,-1.,2.])
        r=np.array([0.2,0.5,0.8]); t=0.17; h=1e-4
        u=field(r,t,c)
        ut=(field(r,t+h,c)-field(r,t-h,c))/(2*h)
        ur=(field(r+h,t,c)-field(r-h,t,c))/(2*h)
        urr=(field(r+h,t,c)-2*u+field(r-h,t,c))/h**2
        np.testing.assert_allclose(9/(2*np.pi)*ut-urr-ur/r,force(t,c),atol=3e-6)

    def test_identify_coefficients_without_noise(self):
        x,y=simulate(42,0)
        c,condition=analytical_fit(x,y)
        np.testing.assert_allclose(c,[6,3.2,2.4],atol=1e-5)
        self.assertLess(condition,100)
        self.assertFalse(np.any((x[:,1]>=0.3)&(x[:,1]<=0.65)))
        info=describe(c)
        self.assertAlmostEqual(info["amplitude"],4,places=5)
        self.assertAlmostEqual(info["peak_phase_cycles"],0.10241638235,places=5)

    def test_rank_deficient_observations_are_rejected(self):
        with self.assertRaises(ValueError):
            analytical_fit(np.array([[1.,0.],[1.,0.2],[1.,0.7]]),np.zeros((3,1)))

    def test_integrated_flow_and_wall_derivative(self):
        c=np.array([5.,-1.,2.]); t=np.linspace(0,1,13); r=np.linspace(0,1,10001)
        q=2*np.trapz(field(r[None,:],t[:,None],c)*r,r,axis=1)
        np.testing.assert_allclose(q,flow(t,c),atol=2e-8)
        h=1e-5
        np.testing.assert_allclose(-(field(1+h,t,c)-field(1-h,t,c))/(2*h),shear(t,c),atol=1e-8)

    def test_residual_trains_all_force_coefficients(self):
        class Known(torch.nn.Module):
            def forward(self,x):
                return (1-x[:,0:1]**2)*(1+x[:,1:2])
        x=torch.tensor([[0.25,0.1],[0.75,0.2]],dtype=torch.float64,requires_grad=True)
        c=torch.tensor([2.,0.5,0.5],dtype=torch.float64,requires_grad=True)
        res=inverse_residual(Known(),c,x)
        r,t=x[:,0:1],x[:,1:2]
        expected=r*(9/(2*torch.pi)*(1-r**2)+4*(1+t)
                    -2-0.5*torch.cos(2*torch.pi*t)-0.5*torch.sin(2*torch.pi*t))
        torch.testing.assert_close(res,expected)
        grad=torch.autograd.grad(res.sum(),c)[0]
        self.assertTrue(torch.all(grad.abs()>0.01))


if __name__=="__main__":
    unittest.main()
