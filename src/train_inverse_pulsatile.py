"""Retrouver le forçage pulsatile inconnu : python src/train_inverse_pulsatile.py."""
import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from train_pulsatile import PulsatileNet,gradient
from pulsatile_physics import ALPHA


def inverse_residual(model,coefficients,x):
    r,t=x[:,0:1],x[:,1:2]
    d=gradient(model(x),x)
    urr=gradient(d[:,0:1],x)[:,0:1]
    a0,ac,ass=coefficients.unbind()
    # Les coefficients ci-dessous sont des paramètres appris, pas les valeurs vraies.
    driving=a0+ac*torch.cos(2*torch.pi*t)+ass*torch.sin(2*torch.pi*t)
    return ALPHA**2/(2*torch.pi)*r*d[:,1:2]-r*urr-d[:,0:1]-r*driving


def fit(x_observed,y_observed,seed=42,steps=4000,lbfgs_steps=300):
    """N'accepte aucune valeur vraie du forçage, ni profil exact."""
    torch.manual_seed(seed)
    model=PulsatileNet()
    coefficients=torch.nn.Parameter(torch.tensor([2.,0.5,0.5]))
    x_obs=torch.tensor(x_observed,dtype=torch.float32)
    y_obs=torch.tensor(y_observed,dtype=torch.float32)
    x_phys=torch.quasirandom.SobolEngine(2,scramble=True,seed=seed).draw(384)
    x_phys[:,0]=0.005+0.99*x_phys[:,0]
    optimizer=torch.optim.Adam([
        {"params":model.parameters(),"lr":0.002},
        {"params":[coefficients],"lr":0.01},
    ])
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=max(steps//3,1),gamma=0.4)
    history=[]
    start=perf_counter()
    def losses():
        x=x_phys.detach().clone().requires_grad_(True)
        data=((model(x_obs)-y_obs)**2).mean()/1.5**2
        physics=(inverse_residual(model,coefficients,x)**2).mean()/10**2
        return data,physics
    for step in range(steps):
        data,physics=losses()
        loss=data+physics
        if not torch.isfinite(loss):
            raise RuntimeError("Erreur non finie pendant l'entraînement.")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        history.append([data.item(),physics.item(),*coefficients.detach().tolist()])
        if step==0 or (step+1)%1000==0:
            values=", ".join(f"{v:.3f}" for v in coefficients.detach().tolist())
            print(f"Adam {step+1}/{steps} — coefficients [{values}]",flush=True)
    print("Affinage conjoint de la vitesse et du forçage…",flush=True)
    optimizer=torch.optim.LBFGS(list(model.parameters())+[coefficients],max_iter=lbfgs_steps,
                               tolerance_grad=1e-8,line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        data,physics=losses()
        loss=data+physics
        if not torch.isfinite(loss):
            raise RuntimeError("Erreur non finie pendant l'affinage.")
        loss.backward()
        return loss
    optimizer.step(closure)
    return model,coefficients.detach().numpy(),np.asarray(history),perf_counter()-start


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps",type=int,default=4000)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--noise",type=float,default=0.03)
    parser.add_argument("--study",action="store_true",help="9 essais : bruits 0,3,10 %, graines consécutives")
    parser.add_argument("--no-show",action="store_true")
    args=parser.parse_args()
    if args.steps<1 or args.seed<0 or not np.isfinite(args.noise) or args.noise<0:
        parser.error("Étapes positives, graine positive ou nulle, bruit fini positif ou nul requis.")
    torch.set_num_threads(1)
    from inverse_pulsatile_physics import simulate
    from inverse_pulsatile_outputs import export_case,write_study
    folder=Path(__file__).resolve().parents[1]/"results"/(
        "inverse_pulsatile_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    folder.mkdir(parents=True,exist_ok=False)
    cases=[(noise,args.seed+k) for noise in (0.,0.03,0.1) for k in range(3)] if args.study else [(args.noise,args.seed)]
    records=[]
    for i,(noise,seed) in enumerate(cases,1):
        print(f"Expérience {i}/{len(cases)} — bruit {noise:.0%}, graine {seed}",flush=True)
        x,y=simulate(seed,noise)
        model,c,history,seconds=fit(x,y,seed,args.steps)
        records.append(export_case(model,c,history,x,y,seconds,seed,noise,args.steps,
                                   folder/f"case_{i:02d}",args.no_show or args.study,
                                   make_plots=not args.study))
    (folder/"summary.json").write_text(json.dumps(records,indent=2)+"\n")
    write_study(records,folder)
    print(f"Résultats : {folder}",flush=True)
    if not args.no_show and not args.study:
        import matplotlib.pyplot as plt
        plt.show()


if __name__=="__main__":
    main()
