"""Export English heatmaps and an offline explorer from saved benchmark cases."""
import argparse
import json
import os
from html import escape
from urllib.parse import quote
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
METHODS=['pinn','data_only','analytical']
LABELS=['Inverse PINN','Data-only NN (no PDE loss)','Analytical least squares']
METRICS=['velocity','flow','shear']
TITLES=['Hidden-phase velocity error','Flow error','Wall-shear error']


def aggregate(records):
    cells=[]
    for n in (12,24,48,96):
        for noise in (0.,.03,.1,.2):
            group=[r for r in records if r['count']==n and r['noise']==noise]
            values={m:{k:dict(mean=float(np.mean([r['metrics'][m][k] for r in group])) if group else None,
                                sd=float(np.std([r['metrics'][m][k] for r in group],ddof=1)) if len(group)>1 else None)
                       for k in METRICS} for m in METHODS}
            cells.append(dict(count=n,noise=noise,repeats=len(group),values=values))
    return cells


def export(source,output):
    records=[json.loads(p.read_text()) for p in sorted(source.glob('*/metrics.json'))]
    if not records: raise ValueError('No completed cases to export')
    cells=aggregate(records)
    cases=[];reference=None
    for record in records:
        with np.load(source/record['case']/'predictions.npz') as z:
            rounded=lambda a:np.asarray(a,dtype=float).round(6).tolist()
            if reference is None:
                reference=dict(r=rounded(z['r'][::4]),t=rounded(z['t'][::2]),
                    velocity=rounded(z['truth_velocity'][::2,::4]),flow=rounded(z['truth_flow'][::2]),shear=rounded(z['truth_shear'][::2]))
            detail={m:dict(velocity=rounded(z[m+'_velocity'][::2,::4]),flow=rounded(z[m+'_flow'][::2]),shear=rounded(z[m+'_shear'][::2])) for m in METHODS}
            cases.append(dict(**record,fields=detail,observations=dict(x=rounded(z['x']),y=rounded(z['y']))))
    data=dict(cells=cells,cases=cases,reference=reference,methods=METHODS,labels=LABELS,metrics=METRICS,titles=TITLES,
              protocol=json.loads((source/'protocol.json').read_text()))
    template=(ROOT/'visualization/templates/reliability.html.in').read_text(encoding='utf-8')
    html=template.replace('__STUDY_DATA__',json.dumps(data,separators=(',',':'),ensure_ascii=False))
    html=html.replace('__D3_LIBRARY__',(ROOT/'visualization/vendor/d3.v7.9.0.min.js').read_text())
    html=html.replace('__FLOW_MATH__',(ROOT/'visualization/flow_math.js').read_text())
    for name in ('REPORT.md','reliability-map.png','reliability-map.pdf'):
        relative=quote(os.path.relpath(source/name,output.parent))
        html=html.replace('../results/reliability_v1/'+name,escape(relative,quote=True))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(html,encoding='utf-8')
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    fig,axes=plt.subplots(3,3,figsize=(14,12),layout='constrained')
    for j,key in enumerate(METRICS):
        vmax=max(c['values'][m][key]['mean'] or 0 for c in cells for m in METHODS)
        norm=Normalize(0,max(vmax,1e-6))
        for i,method in enumerate(METHODS):
            grid=np.array([[next(c for c in cells if c['count']==n and c['noise']==b)['values'][method][key]['mean'] for n in (12,24,48,96)] for b in (0.,.03,.1,.2)],dtype=float)
            ax=axes[i,j];im=ax.imshow(grid,cmap='YlGnBu',norm=norm,aspect='auto')
            ax.set_xticks(range(4),[12,24,48,96]);ax.set_yticks(range(4),['0%','3%','10%','20%'])
            ax.set(xlabel='Number of velocity observations',ylabel='Noise standard deviation / 1.5',title=f'{LABELS[i]}\n{TITLES[j]}')
            for y in range(4):
                for x in range(4):
                    v=grid[y,x]
                    repeats=next(c for c in cells if c['count']==(12,24,48,96)[x] and c['noise']==(0.,.03,.1,.2)[y])['repeats']
                    label=(f'{v:.2f}'+(f'\n{repeats}/3 runs' if repeats<3 else '')) if np.isfinite(v) else 'Pending'
                    ax.text(x,y,label,ha='center',va='center',fontsize=10,color='white' if v>vmax*.55 else '#152c3d')
        fig.colorbar(im,ax=axes[:,j],shrink=.65,label='Mean error (%) — common scale within column')
    fig.suptitle(f'Reconstruction reliability · {len(records)}/48 completed runs\nVelocity: hidden-phase RMSE / 1.5; flow and wall shear: full-cycle relative L2. Synthetic data.',fontsize=14)
    fig.savefig(source/'reliability-map.png',dpi=180)
    fig.savefig(source/'reliability-map.pdf');plt.close(fig)
    lines=['# Reconstruction reliability study','',f'Completed runs: **{len(records)} / 48**.','',
       '## Protocol','',
       'Observation counts: 12, 24, 48, 96. Gaussian noise standard deviation: 0%, 3%, 10%, 20% of the fixed velocity scale 1.5. Seeds: 42, 43, 44.',
       'Six fixed observed phases: 0, 0.1, 0.2, 0.75, 0.85, 0.95. No velocity observations in [0.30, 0.65]. Increasing the count adds nested radial locations, not additional observed phases.',
       'All methods receive exactly the same observations. Both networks use the same architecture, initialization, periodic features and exact centre symmetry/no-slip wall conditions. The data-only network omits the PDE loss.',
       'Both networks: 1,000 Adam steps, then at most 100 L-BFGS iterations with strong-Wolfe line search. Equal iteration limits do not imply equal computational cost or equal convergence.',
       'The analytical baseline uses the correct harmonic solution family. Geometry, frequency and alpha=3 are known. The PINN jointly learns the velocity and three forcing coefficients.',
       '', '## Metrics','',
       'Velocity error: RMSE over the hidden phases divided by 1.5, expressed as a percentage. Flow and signed wall-shear errors: full-cycle relative L2, expressed as percentages. Evaluation grid: 101 phases × 121 radial positions. Flow is integrated radially; wall shear is evaluated by automatic differentiation for the networks.',
       'Mean ± sample standard deviation across seeds; this is not a confidence interval. Seeds jointly change noise and neural initialization. No clinical acceptance threshold is implied.',
       '', '## Results','', '| Observations | Noise | Repeats | Method | Velocity (%) | Flow (%) | Wall shear (%) |', '|---:|---:|---:|---|---:|---:|---:|']
    for cell in cells:
        for method,label in zip(METHODS,LABELS):
            def val(k):
                v=cell['values'][method][k]
                if v['mean'] is None:return 'Pending'
                return f"{v['mean']:.4f}"+(f" ± {v['sd']:.4f}" if v['sd'] is not None else ' (one run)')
            lines.append(f"| {cell['count']} | {cell['noise']:.0%} | {cell['repeats']} | {label} | {val('velocity')} | {val('flow')} | {val('shear')} |")
    lines+=['','## Measured comparisons','']
    complete=[c for c in cells if c['repeats']==3]
    for key,title in zip(METRICS,TITLES):
        if complete:
            wins=sum(c['values']['pinn'][key]['mean']<c['values']['data_only'][key]['mean'] for c in complete)
            analytic=sum(c['values']['analytical'][key]['mean']<c['values']['pinn'][key]['mean'] for c in complete)
            lines.append(f'- {title}: the PINN has lower mean error than the data-only network in {wins}/{len(complete)} completed cells; analytical least squares has lower mean error than the PINN in {analytic}/{len(complete)} cells.')
    for method,label in zip(METHODS,LABELS):
        timing=np.array([r['seconds'][method] for r in records])
        lines.append(f'- {label}: median fitting time {np.median(timing):.3f} s (range {timing.min():.3f}–{timing.max():.3f} s). CPU, one Torch thread per process. Evaluation/export time excluded.')
    lines.append('The pilot ran sequentially; remaining cases used up to three concurrent processes. Recorded times include uncontrolled CPU contention and are not a controlled method-speed benchmark.')
    lines+=['','## Limits and interpretation','',
      'These maps characterize one synthetic geometry, forcing family, sampling design and fixed optimization budget. They do not establish universal PINN superiority or validate a medical application. More observations do not have to improve every individual noisy run.',
      'Three repetitions provide only a preliminary view of variability. Optimization error, noise effects and model assumptions can all affect the result. A failure at this budget is not proof that a method cannot solve the problem.',
      'The interactive page downsamples fields for display and linearly interpolates between saved samples. Metrics use the original evaluation grid. Particle motion and axial distances are illustrative, not tracked blood cells.',
      '', '## Reproduction','',
      '`python src/reliability_study.py --pilot` runs the four-corner pilot. `python src/reliability_study.py` resumes all 48 paired cases. Protocol and source hashes prevent mixing incompatible checkpoints.',
      '`python src/reliability_report.py` rebuilds the English page and figures. Each case stores its observations and fields in predictions.npz, network weights in models.pt and metrics/timing in metrics.json.','']
    (source/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    (source/'aggregate.json').write_text(json.dumps(cells,indent=2),encoding='utf-8')
    print(f'Exported {len(records)} runs: {output}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'results/reliability_v1')
    parser.add_argument('--output',type=Path,default=ROOT/'visualization/reliability.html')
    a=parser.parse_args();export(a.source,a.output)
