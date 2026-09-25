"""Apprendre un cycle pulsatile ; lancer : python src/train_pulsatile.py.

La référence analytique fournit seulement les observations et la vérification.
L'équation et le forçage G(t) sont connus partout, même aux instants sans mesures.
"""
import argparse
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from pulsatile_physics import ALPHA, observations, VELOCITY_SCALE


class PulsatileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(3,32),nn.Tanh(),nn.Linear(32,32),
                                   nn.Tanh(),nn.Linear(32,1))

    def forward(self,x):
        r,t=x[:,0:1],x[:,1:2]
        # Entrée temporelle périodique ; pas de condition initiale de démarrage.
        features=torch.cat((r**2,torch.sin(2*torch.pi*t),torch.cos(2*torch.pi*t)),dim=1)
        # Symétrie au centre et vitesse nulle à la paroi imposées exactement.
        return (1-r**2)*self.network(features)


def gradient(y,x):
    return torch.autograd.grad(y,x,torch.ones_like(y),create_graph=True)[0]


def residual(model,x):
    """Équation multipliée par r pour ne pas diviser par zéro au centre."""
    r,t=x[:,0:1],x[:,1:2]
    d=gradient(model(x),x)
    urr=gradient(d[:,0:1],x)[:,0:1]
    force=6+4*torch.cos(2*torch.pi*t)
    return ALPHA**2/(2*torch.pi)*r*d[:,1:2]-(r*urr+d[:,0:1])-r*force


def train(seed=42,steps=4000):
    torch.manual_seed(seed)
    model=PulsatileNet()
    x_obs,y_obs=observations(seed)
    x_obs,y_obs=torch.tensor(x_obs),torch.tensor(y_obs)
    # Positions de physique réparties dans tout le cycle, sans valeurs de vitesse.
    x_phys=torch.quasirandom.SobolEngine(2,scramble=True,seed=seed).draw(384)
    x_phys[:,0]=0.005+0.99*x_phys[:,0]
    history=[]

    def losses():
        x=x_phys.detach().clone().requires_grad_(True)
        data=((model(x_obs)-y_obs)**2).mean()/VELOCITY_SCALE**2
        physics=(residual(model,x)**2).mean()/10**2
        return data,physics

    optimizer=torch.optim.Adam(model.parameters(),lr=0.002)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=max(steps//3,1),gamma=0.4)
    start=perf_counter()
    for step in range(steps):
        data,physics=losses()
        loss=data+physics
        if not torch.isfinite(loss):
            raise RuntimeError("L'entraînement a divergé (erreur non finie).")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        history.append([data.item(),physics.item()])
        if step==0 or (step+1)%1000==0:
            print(f"Adam {step+1}/{steps} — mesures {data.item():.6f}, physique {physics.item():.6f}",flush=True)

    # Raffiner la même perte, sans consulter les valeurs exactes d'évaluation.
    print("Affinage L-BFGS…",flush=True)
    optimizer=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=250,
                               tolerance_grad=1e-8,line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        data,physics=losses()
        loss=data+physics
        if not torch.isfinite(loss):
            raise RuntimeError("L'affinage a divergé.")
        loss.backward()
        return loss
    optimizer.step(closure)
    data,physics=losses()
    print(f"Perte finale — mesures {data.item():.6f}, physique {physics.item():.6f}",flush=True)
    return model,np.asarray(history),perf_counter()-start


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps",type=int,default=4000)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--no-show",action="store_true")
    args=parser.parse_args()
    if args.steps<1 or args.seed<0:
        parser.error("--steps doit être positif et --seed positif ou nul.")
    torch.set_num_threads(1)
    model,history,seconds=train(args.seed,args.steps)
    # Import tardif pour garder les tests du modèle indépendants de l'affichage.
    from pulsatile_outputs import export_results
    folder=Path(__file__).resolve().parents[1]/"results"/(
        "pulsatile_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    print("Validation et création des graphiques…",flush=True)
    export_results(model,history,seconds,args,folder)


if __name__=="__main__":
    main()
