"""Run a resumable, paired synthetic benchmark; presentation language: English."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from inverse_pulsatile_physics import (field,flow,shear,analytical_fit,
                                      TRUE_COEFFICIENTS,SCALE)
from train_pulsatile import PulsatileNet
from train_inverse_pulsatile import fit

ROOT=Path(__file__).resolve().parents[1]
COUNTS=(12,24,48,96)
NOISES=(0.,.03,.1,.2)
SEEDS=(42,43,44)
PROTOCOL=dict(version=1,counts=COUNTS,noise=NOISES,seeds=SEEDS,steps=1000,
              lbfgs_max_iter=100,hidden=[.3,.65],velocity_scale=SCALE,
              observed_phases=[0.,.1,.2,.75,.85,.95],alpha=3.,
              location_design='nested base-2 van der Corput radii, six fixed phases',
              methods=['pinn','data_only','analytical'])


def observations(count,noise,seed):
    if count not in COUNTS or noise<0 or not np.isfinite(noise):
        raise ValueError('Unsupported count or noise level')
    def vdc(index):
        result=0.;denominator=1
        while index:
            index,remainder=divmod(index,2);denominator*=2
            result+=remainder/denominator
        return result
    radii=.08+.84*np.array([vdc(i) for i in range(1,17)])
    x=np.column_stack((np.repeat(radii,6),np.tile(PROTOCOL['observed_phases'],16)))
    y=field(x[:,0],x[:,1],TRUE_COEFFICIENTS)
    y+=np.random.default_rng(seed).normal(0,noise*SCALE,96)
    return x[:count].astype(np.float32),y[:count,None].astype(np.float32)


def error_metrics(pred,truth,hidden):
    return dict(velocity=float(np.sqrt(np.mean((pred['velocity'][hidden]-truth['velocity'][hidden])**2))/SCALE*100),
                flow=float(np.linalg.norm(pred['flow']-truth['flow'])/np.linalg.norm(truth['flow'])*100),
                shear=float(np.linalg.norm(pred['shear']-truth['shear'])/np.linalg.norm(truth['shear'])*100))


def fit_data_only(x,y,seed,steps,lbfgs_steps):
    torch.manual_seed(seed)
    model=PulsatileNet()
    x,y=torch.tensor(x),torch.tensor(y)
    start=perf_counter()
    def loss(): return ((model(x)-y)**2).mean()/SCALE**2
    opt=torch.optim.Adam(model.parameters(),lr=.002)
    schedule=torch.optim.lr_scheduler.StepLR(opt,max(steps//3,1),gamma=.4)
    for _ in range(steps):
        opt.zero_grad();value=loss()
        if not torch.isfinite(value): raise RuntimeError('Non-finite data-only loss')
        value.backward();opt.step();schedule.step()
    opt=torch.optim.LBFGS(model.parameters(),max_iter=lbfgs_steps,tolerance_grad=1e-8,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();value=loss()
        if not torch.isfinite(value): raise RuntimeError('Non-finite data-only refinement loss')
        value.backward();return value
    opt.step(closure)
    return model,perf_counter()-start


def neural_fields(model,r,t):
    rr,tt=np.meshgrid(r,t)
    x=torch.tensor(np.column_stack((rr.ravel(),tt.ravel())),dtype=torch.float32)
    model.eval()
    with torch.no_grad(): u=model(x).numpy().reshape(rr.shape)
    wall=torch.tensor(np.column_stack((np.ones_like(t),t)),dtype=torch.float32,requires_grad=True)
    tau=-torch.autograd.grad(model(wall).sum(),wall)[0][:,0].detach().numpy()
    return dict(velocity=u,flow=2*np.trapz(u*r,r,axis=1),shear=tau)


def run_case(folder,count,noise,seed):
    folder.mkdir(parents=True,exist_ok=True)
    x,y=observations(count,noise,seed)
    model,c,history,pinn_seconds=fit(x,y,seed,PROTOCOL['steps'],PROTOCOL['lbfgs_max_iter'])
    plain,plain_seconds=fit_data_only(x,y,seed,PROTOCOL['steps'],PROTOCOL['lbfgs_max_iter'])
    start=perf_counter();coefficients,condition=analytical_fit(x,y);analytical_seconds=perf_counter()-start
    r,t=np.linspace(0,1,121),np.linspace(0,1,101)
    rr,tt=np.meshgrid(r,t);hidden=(t>=.3)&(t<=.65)
    truth=dict(velocity=field(rr,tt,TRUE_COEFFICIENTS),flow=flow(t,TRUE_COEFFICIENTS),shear=shear(t,TRUE_COEFFICIENTS))
    predicted=dict(pinn=neural_fields(model,r,t),data_only=neural_fields(plain,r,t),
                   analytical=dict(velocity=field(rr,tt,coefficients),flow=flow(t,coefficients),shear=shear(t,coefficients)))
    metrics={name:error_metrics(values,truth,hidden) for name,values in predicted.items()}
    if not all(np.isfinite(v) for m in metrics.values() for v in m.values()):
        raise RuntimeError('Non-finite evaluation metrics')
    record=dict(count=count,noise=noise,seed=seed,metrics=metrics,
                seconds=dict(pinn=pinn_seconds,data_only=plain_seconds,analytical=analytical_seconds),
                coefficients=c.tolist(),analytical_coefficients=coefficients.tolist(),
                analytical_design_condition=condition,case=folder.name)
    np.savez_compressed(folder/'predictions.npz',r=r,t=t,hidden=hidden,x=x,y=y,history=history,
                        **{f'{name}_{key}':a for name,values in dict(truth=truth,**predicted).items() for key,a in values.items()})
    torch.save(dict(pinn=model.state_dict(),data_only=plain.state_dict(),coefficients=c.tolist()),folder/'models.pt')
    # metrics.json is the completion marker: write it only after both larger artifacts.
    temp=folder/'metrics.tmp';temp.write_text(json.dumps(record,indent=2),encoding='utf-8')
    temp.replace(folder/'metrics.json')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'results/reliability_v1')
    args=parser.parse_args();torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    protocol=dict(PROTOCOL,torch_version=torch.__version__,numpy_version=np.__version__)
    protocol['source_sha256']={name:hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest() for name in
        ('reliability_study.py','train_inverse_pulsatile.py','train_pulsatile.py','inverse_pulsatile_physics.py','pulsatile_physics.py')}
    serialized=json.dumps(protocol,indent=2)
    manifest=args.output/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text())!=json.loads(serialized):
        raise ValueError('Protocol/source changed. Use a new output directory; do not mix checkpoints.')
    manifest.write_text(serialized,encoding='utf-8')
    cases=[(n,b,42) for n in (12,96) for b in (0.,.2)] if args.pilot else [(n,b,s) for n in COUNTS for b in NOISES for s in SEEDS]
    for index,(n,b,s) in enumerate(cases,1):
        folder=args.output/f'n{n:03d}_noise{round(b*100):02d}_seed{s}'
        if (folder/'metrics.json').exists():
            print(f'REUSE {folder.name}',flush=True);continue
        print(f'CASE {index}/{len(cases)}: {n} observations, {b:.0%} noise, seed {s}',flush=True)
        record=run_case(folder,n,b,s)
        print('COMPLETE '+json.dumps(record['metrics'])+' seconds='+json.dumps(record['seconds']),flush=True)
        records=[json.loads(p.read_text()) for p in sorted(args.output.glob('*/metrics.json'))]
        (args.output/'summary.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(f'SAVED {args.output}',flush=True)


if __name__=='__main__': main()
