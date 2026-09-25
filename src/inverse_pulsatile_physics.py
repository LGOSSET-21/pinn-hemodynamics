"""Simulation et référence analytique ; jamais utilisées comme cibles de physique."""
import numpy as np
from pulsatile_physics import ALPHA,K,bessel

TRUE_COEFFICIENTS=np.array([6.,3.2,2.4])
SCALE=1.5


def basis(r,t):
    """Réponse à trois forçages unitaires : constant, cosinus, sinus."""
    r,t=np.broadcast_arrays(np.asarray(r),np.asarray(t))
    h=(1-bessel(K*r)/bessel(K))/(1j*ALPHA**2)*np.exp(2j*np.pi*t)
    return np.stack(((1-r*r)/4,h.real,h.imag),axis=-1)


def field(r,t,coefficients):
    return basis(r,t)@np.asarray(coefficients)


def force(t,coefficients):
    a0,ac,ass=coefficients
    return a0+ac*np.cos(2*np.pi*np.asarray(t))+ass*np.sin(2*np.pi*np.asarray(t))


def flow(t,coefficients):
    a0,ac,ass=coefficients
    h=(1-2*bessel(K,1)/(K*bessel(K)))/(1j*ALPHA**2)
    return a0/8+np.real((ac-1j*ass)*h*np.exp(2j*np.pi*np.asarray(t)))


def shear(t,coefficients):
    a0,ac,ass=coefficients
    h=-K*bessel(K,1)/(1j*ALPHA**2*bessel(K))
    return a0/2+np.real((ac-1j*ass)*h*np.exp(2j*np.pi*np.asarray(t)))


def simulate(seed=42,noise=0.03):
    r,t=np.meshgrid(np.linspace(0.08,0.92,6),[0.,0.1,0.2,0.75,0.85,0.95])
    x=np.column_stack((r.ravel(),t.ravel()))
    y=field(x[:,0],x[:,1],TRUE_COEFFICIENTS)
    y=y+np.random.default_rng(seed).normal(0,noise*SCALE,len(x))
    return x.astype(np.float32),y[:,None].astype(np.float32)


def analytical_fit(x,y):
    """Moindres carrés sur les mêmes observations, sans coefficients vrais."""
    design=basis(x[:,0],x[:,1])
    coefficients,_,rank,singular=np.linalg.lstsq(design,np.asarray(y).ravel(),rcond=None)
    if rank<3:
        raise ValueError("Ces observations ne permettent pas d'identifier trois coefficients.")
    return coefficients,float(singular[0]/singular[-1])


def describe(coefficients):
    a0,ac,ass=map(float,coefficients)
    amplitude=float(np.hypot(ac,ass))
    # G=a0+amplitude*cos(2*pi*(t-phase)). Phase modulo un cycle.
    phase=float(np.arctan2(ass,ac)/(2*np.pi)%1) if amplitude>1e-8 else None
    return dict(mean=a0,amplitude=amplitude,peak_phase_cycles=phase)
