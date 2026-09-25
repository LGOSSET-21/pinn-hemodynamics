"""Référence de Womersley pour un tube rigide : alpha=3, G=6+4*cos(2*pi*t).

t = temps / période ; r = rayon / R ; u = vitesse / Uref.
L'équation est alpha²/(2*pi)*u_t = u_rr + u_r/r + G(t).
La série de Bessel ci-dessous est destinée à ce cas modéré alpha=3 uniquement.
"""
import numpy as np

ALPHA = 3.0
MEAN_FORCE = 6.0
AMPLITUDE = 4.0
VELOCITY_SCALE = 1.5
K = np.sqrt(-1j) * ALPHA


def bessel(z, order=0):
    """J0 ou J1 par série convergente, sans ajouter de dépendance SciPy."""
    z = np.asarray(z, dtype=complex)
    term = np.ones_like(z) if order == 0 else z / 2
    total = term.copy()
    for m in range(1, 40):
        term = term * (-(z*z)/4) / (m*(m+order))
        total = total + term
    return total


def reference(r, t, amplitude=AMPLITUDE):
    """Solution déjà périodique : partie moyenne + oscillation complexe."""
    r, t = np.asarray(r), np.asarray(t)
    harmonic = amplitude/(1j*ALPHA**2) * (1-bessel(K*r)/bessel(K))
    return MEAN_FORCE/4*(1-r*r) + np.real(harmonic*np.exp(2j*np.pi*t))


def reference_flow(t, amplitude=AMPLITUDE):
    """Q*=Q/(pi*R²*Uref)=2*int(r*u)dr, intégré analytiquement."""
    harmonic = amplitude/(1j*ALPHA**2)*(1-2*bessel(K,1)/(K*bessel(K)))
    return MEAN_FORCE/8 + np.real(harmonic*np.exp(2j*np.pi*np.asarray(t)))


def reference_shear(t, amplitude=AMPLITUDE):
    """tau*=-u_r(1,t) : effort axial signé du fluide sur la paroi."""
    harmonic = -amplitude/(1j*ALPHA**2)*K*bessel(K,1)/bessel(K)
    return MEAN_FORCE/2 + np.real(harmonic*np.exp(2j*np.pi*np.asarray(t)))


def observations(seed=42):
    """36 mesures ; aucune phase dans [0.30,0.65], bruit sigma=0.03*1.5."""
    r,t=np.meshgrid(np.linspace(0.08,0.92,6),[0.,0.1,0.2,0.75,0.85,0.95])
    x=np.column_stack([r.ravel(),t.ravel()])
    rng=np.random.default_rng(seed)
    y=reference(x[:,0],x[:,1])+rng.normal(0,0.03*VELOCITY_SCALE,len(x))
    return x.astype(np.float32),y[:,None].astype(np.float32)
