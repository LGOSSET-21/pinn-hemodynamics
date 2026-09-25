"""Contrôles de physique indépendants des courbes d'apprentissage."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pulsatile_physics import reference, reference_flow, reference_shear, observations


class PhysicsTests(unittest.TestCase):
    def test_stationary_limit_and_integrals(self):
        r = np.array([0., 0.5, 1.])
        np.testing.assert_allclose(reference(r, 0.2, amplitude=0), [1.5, 1.125, 0], atol=1e-12)
        self.assertAlmostEqual(float(reference_flow(0.2, amplitude=0)), 0.75)
        self.assertAlmostEqual(float(reference_shear(0.2, amplitude=0)), 3.)

    def test_reference_satisfies_unsteady_equation(self):
        r = np.array([0.2, 0.5, 0.8])
        t, h = 0.173, 1e-4
        u = reference(r, t)
        ur = (reference(r+h,t)-reference(r-h,t))/(2*h)
        urr = (reference(r+h,t)-2*u+reference(r-h,t))/h**2
        ut = (reference(r,t+h)-reference(r,t-h))/(2*h)
        residual = 9/(2*np.pi)*ut-urr-ur/r-(6+4*np.cos(2*np.pi*t))
        np.testing.assert_allclose(residual, 0, atol=3e-6)

    def test_periodicity_boundaries_and_derived_quantities(self):
        t = np.linspace(0,1,13)
        np.testing.assert_allclose(reference(0.4,t), reference(0.4,t+1), atol=1e-12)
        np.testing.assert_allclose(reference(1.,t), 0, atol=1e-12)
        h=1e-5
        np.testing.assert_allclose((reference(h,t)-reference(-h,t))/(2*h),0,atol=1e-12)
        r=np.linspace(0,1,10001)
        q=2*np.trapz(reference(r[None,:],t[:,None])*r,r,axis=1)
        np.testing.assert_allclose(q, reference_flow(t),atol=2e-8)
        shear=-(reference(1+h,t)-reference(1-h,t))/(2*h)
        np.testing.assert_allclose(shear,reference_shear(t),atol=1e-8)

    def test_observations_exclude_hidden_interval(self):
        x,y=observations(42)
        self.assertEqual(x.shape,(36,2))
        self.assertEqual(y.shape,(36,1))
        self.assertFalse(np.any((x[:,1]>=0.3)&(x[:,1]<=0.65)))
        np.testing.assert_array_equal(y, observations(42)[1])


if __name__ == "__main__":
    unittest.main()
