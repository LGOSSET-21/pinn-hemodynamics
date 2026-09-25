"""Assembler la vue interactive à partir de neuf résultats déjà enregistrés."""
import argparse
import json
from html import escape
from pathlib import Path

import numpy as np


def build(source, template, output, standalone=False):
    records=json.loads((source/"summary.json").read_text())
    if len(records)!=9 or {r["steps"] for r in records}!={2000}:
        raise ValueError("La visualisation attend les neuf essais comparables à 2000 étapes.")
    cases=[]
    reference=None
    velocity_min=0.; velocity_max=0.; error_max=0.
    rounded=lambda a:np.asarray(a,dtype=np.float64).round(6).tolist()
    for index,record in enumerate(records,1):
        folder=source/f"case_{index:02d}"
        with np.load(folder/"predictions.npz") as z:
            if z["pred"].shape!=(101,121) or not all(np.isfinite(z[k]).all() for k in z.files):
                raise ValueError("Grille inattendue ou données non finies.")
            current=dict(r=rounded(z["r"][::4]),t=rounded(z["t"][::2]),
                         velocity=rounded(z["truth"][::2,::4]),
                         force=rounded(z["force_true"][::2]),
                         flow=rounded(z["flow_true"][::2]),
                         shear=rounded(z["shear_true"][::2]))
            if reference is None:
                reference=current
            elif current!=reference:
                raise ValueError("Les références des scénarios diffèrent.")
            obs=json.loads((folder/"observations.json").read_text())
            phases=np.array(obs["r_t"])[:,1]
            if np.any((phases>=0.3)&(phases<=0.65)):
                raise ValueError("Une observation entre dans la plage cachée.")
            cases.append(dict(noise=record["noise"],seed=record["seed"],
                              source=str(folder.relative_to(source.parent.parent)),
                              velocity=rounded(z["pred"][::2,::4]),
                              force=rounded(z["force_pred"][::2]),
                              flow=rounded(z["flow_pred"][::2]),
                              shear=rounded(z["shear_pred"][::2]),
                              observations=obs,metrics=record["pinn"]))
            velocity_min=min(velocity_min,float(z["truth"].min()),float(z["pred"].min()))
            velocity_max=max(velocity_max,float(z["truth"].max()),float(z["pred"].max()))
            error_max=max(error_max,float(np.abs(z["pred"]-z["truth"]).max()))
    data=dict(reference=reference,cases=cases,velocity_min=velocity_min,
              velocity_max=velocity_max,error_max=error_max,hidden=[0.3,0.65])
    html=template.read_text().replace("__FLOW_DATA__",json.dumps(data,ensure_ascii=False,separators=(",",":")))
    html=html.replace("__FLOW_MATH__",(template.parent/"flow_math.js").read_text())
    if len(html.encode())>=1_000_000 or "__FLOW_" in html:
        raise ValueError("Fragment trop grand ou incomplet.")
    if standalone:
        styles=(template.parent/"standalone.css").read_text(encoding="utf-8")
        library=(template.parent/"vendor/d3.v7.9.0.min.js").read_text(encoding="utf-8")
        html=html.replace('<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>',
                          '<script>'+library+'</script>')
        html=html.replace('</h2>', '</h2><p class="intro">Un écoulement pulsatile, quelques mesures et les lois de la physique. Explore comment le réseau reconstruit ce qu’il n’a pas observé.</p>',1)
        html=html.replace('<h2>', '<style>'+styles+'</style><h2>',1)
        html=html.replace('<div class="sr-only">', '<p class="lab-footer">9 expériences · 2 000 étapes chacune · Données simulées · Aucun entraînement lancé par cette page</p><div class="sr-only">',1)
        html=(template.parent/"standalone-shell.template.html").read_text(encoding="utf-8").replace('__FLOW_DOCUMENT__',escape(html))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(html,encoding="utf-8")
    print(f"9 scénarios, 51 phases × 31 rayons ; {len(html.encode()):,} octets ; {output}")


if __name__=="__main__":
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--standalone",action="store_true",help="Page autonome utilisable hors connexion")
    args=parser.parse_args()
    build(root/"results/inverse_pulsatile_20260913_165408_169589",
          root/"visualization/flow-fragment.template.html",args.output,args.standalone)
