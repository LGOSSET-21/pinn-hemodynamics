"""Resume the fixed study using independent CPU processes; preserve pilot runs."""
import argparse
import contextlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import multiprocessing
import numpy as np
import torch
from reliability_study import ROOT,PROTOCOL,COUNTS,NOISES,SEEDS,run_case


def worker(folder,n,b,seed):
    torch.set_num_threads(1)
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'training.log').open('w') as log,contextlib.redirect_stdout(log):
        return run_case(folder,n,b,seed)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--output',type=Path,default=ROOT/'results/reliability_v1')
    args=parser.parse_args()
    if not 1<=args.workers<=3: parser.error('Use 1–3 independent workers')
    manifest=json.loads((args.output/'protocol.json').read_text())
    for key,value in json.loads(json.dumps(PROTOCOL)).items():
        if manifest[key]!=value: raise ValueError('Protocol mismatch')
    for name,digest in manifest['source_sha256'].items():
        if hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Training source changed: '+name)
    if manifest['torch_version']!=torch.__version__: raise ValueError('Torch version changed')
    if manifest['numpy_version']!=np.__version__: raise ValueError('NumPy version changed')
    (args.output/'execution.json').write_text(json.dumps(dict(pilot_workers=1,study_workers=args.workers,torch_threads_per_worker=1,
         note='Fitting times include uncontrolled concurrent CPU contention; do not interpret them as controlled speed benchmarks.'),indent=2))
    pending=[]
    for n in COUNTS:
        for b in NOISES:
            for seed in SEEDS:
                folder=args.output/f'n{n:03d}_noise{round(b*100):02d}_seed{seed}'
                if not (folder/'metrics.json').exists():pending.append((folder,n,b,seed))
    print(f'{len(pending)} cases pending; {args.workers} workers.',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(worker,*case):case[0].name for case in pending}
        for future in as_completed(futures):
            record=future.result()
            records=[json.loads(p.read_text()) for p in sorted(args.output.glob('*/metrics.json'))]
            (args.output/'summary.json').write_text(json.dumps(records,indent=2))
            print(f"{len(records)}/48 COMPLETE {record['case']} PINN hidden error={record['metrics']['pinn']['velocity']:.3f}%",flush=True)
    print('Full study complete.',flush=True)
