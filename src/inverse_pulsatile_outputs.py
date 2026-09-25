"""Évaluer après l'entraînement : le forçage vrai intervient seulement ici."""
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from inverse_pulsatile_physics import (TRUE_COEFFICIENTS,SCALE,field,force,flow,
                                       shear,analytical_fit,describe)


def evaluate(model,coefficients,x_obs,y_obs):
    fitted,condition=analytical_fit(x_obs,y_obs)
    r,t=np.linspace(0,1,121),np.linspace(0,1,101)
    rr,tt=np.meshgrid(r,t)
    x=torch.tensor(np.column_stack((rr.ravel(),tt.ravel())),dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred=model(x).numpy().reshape(rr.shape)
    wall=torch.tensor(np.column_stack((np.ones_like(t),t)),dtype=torch.float32,requires_grad=True)
    shear_pred=-torch.autograd.grad(model(wall).sum(),wall)[0][:,0].detach().numpy()
    f=dict(r=r,t=t,hidden=(t>=0.30)&(t<=0.65),pred=pred,
           truth=field(rr,tt,TRUE_COEFFICIENTS),analytical=field(rr,tt,fitted),
           force_pred=force(t,coefficients),force_true=force(t,TRUE_COEFFICIENTS),
           force_analytical=force(t,fitted),flow_pred=2*np.trapz(pred*r,r,axis=1),
           flow_true=flow(t,TRUE_COEFFICIENTS),flow_analytical=flow(t,fitted),
           shear_pred=shear_pred,shear_true=shear(t,TRUE_COEFFICIENTS),
           shear_analytical=shear(t,fitted))
    hidden=f["hidden"]
    relative=lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(b))
    true_description=describe(TRUE_COEFFICIENTS)
    metrics=dict(true_coefficients=TRUE_COEFFICIENTS.tolist(),truth=true_description,
                 analytical_design_condition=condition)
    for name,c,suffix,velocity in (("pinn",coefficients,"pred",pred),
                                    ("analytical",fitted,"analytical",f["analytical"])):
        desc=describe(c)
        phase=desc["peak_phase_cycles"]
        phase_error=abs((phase-true_description["peak_phase_cycles"]+0.5)%1-0.5) if phase is not None else None
        record=dict(coefficients=np.asarray(c).tolist(),**desc,
            phase_error_cycles=phase_error,
            force_relative_l2=relative(f["force_"+suffix],f["force_true"]),
            hidden_force_relative_l2=relative(f["force_"+suffix][hidden],f["force_true"][hidden]),
            velocity_rmse=float(np.sqrt(np.mean((velocity-f["truth"])**2))),
            hidden_velocity_rmse=float(np.sqrt(np.mean((velocity[hidden]-f["truth"][hidden])**2))),
            flow_relative_l2=relative(f["flow_"+suffix],f["flow_true"]),
            shear_relative_l2=relative(f["shear_"+suffix],f["shear_true"]))
        record["accuracy_targets_met"]=(record["force_relative_l2"]<0.1 and
            record["hidden_velocity_rmse"]/SCALE<0.05 and
            record["flow_relative_l2"]<0.1 and record["shear_relative_l2"]<0.1)
        if not all(np.isfinite(v) for v in record.values() if isinstance(v,(float,int))):
            raise RuntimeError("Résultat non fini.")
        metrics[name]=record
    return f,metrics


def export_case(model,c,history,x,y,seconds,seed,noise,steps,folder,no_show,make_plots=True):
    f,metrics=evaluate(model,c,x,y)
    metrics.update(seed=seed,noise=noise,steps=steps,training_seconds=seconds,
                   torch_version=torch.__version__,numpy_version=np.__version__)
    from train_inverse_pulsatile import inverse_residual
    check=torch.quasirandom.SobolEngine(2,scramble=True,seed=seed+1).draw(1024)
    check[:,0]=0.005+0.99*check[:,0]
    check.requires_grad_(True)
    metrics["weighted_residual_rmse"]=float(torch.sqrt((inverse_residual(model,torch.tensor(c),check)**2).mean()).detach())
    folder.mkdir(parents=True,exist_ok=False)
    (folder/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    (folder/"observations.json").write_text(json.dumps(dict(r_t=x.tolist(),u=y.tolist()),indent=2)+"\n")
    np.savez_compressed(folder/"predictions.npz",**f,history=history)
    torch.save(dict(state_dict=model.state_dict(),coefficients=c.tolist(),
                    seed=seed,noise=noise,steps=steps,alpha=3.),folder/"model.pt")
    if make_plots:
        print("Création des comparaisons…",flush=True)
        fig,axs=plt.subplots(2,2,figsize=(12,8),layout="constrained")
        for ax,key,title,ylabel in ((axs[0,0],"force","Forçage inconnu reconstruit","G(t)"),
                                    (axs[1,0],"flow","Débit","Q*"),
                                    (axs[1,1],"shear","Cisaillement signé à la paroi","τ*")):
            ax.plot(f["t"],f[key+"_true"],"k-",lw=2,label="Vérité simulée")
            ax.plot(f["t"],f[key+"_pred"],"--",color="#d97823",label="PINN inverse")
            ax.plot(f["t"],f[key+"_analytical"],":",color="#167b8c",label="Ajustement analytique")
            ax.axvspan(0.30,0.65,color="gray",alpha=0.15,label="Aucune vitesse observée")
            ax.set(title=title,xlabel="Fraction de cycle",ylabel=ylabel)
        index=np.argmin(abs(f["t"]-0.45))
        for key,label,style,color in (("truth","Vérité simulée","-","black"),
                                      ("pred","PINN inverse","--","#d97823"),
                                      ("analytical","Ajustement analytique",":","#167b8c")):
            axs[0,1].plot(f["r"],f[key][index],style,color=color,label=label)
        axs[0,1].set(title="Profil à t/T=0,45 : aucune mesure",xlabel="Rayon r",ylabel="Vitesse u")
        for ax in axs.flat:
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        fig.suptitle(f"36 vitesses simulées • bruit {noise:.0%} • forçage appris • graine {seed}")
        fig.savefig(folder/"comparison.png",dpi=140)
        evo,axes=plt.subplots(1,3,figsize=(12,3.8),layout="constrained")
        for i,(ax,label) in enumerate(zip(axes,["Moyenne a0","Coefficient cosinus ac","Coefficient sinus as"])):
            ax.plot(np.arange(1,len(history)+1),history[:,i+2],label="Pendant Adam")
            ax.axhline(TRUE_COEFFICIENTS[i],color="black",ls="--",label="Vrai (évaluation)")
            ax.scatter([len(history)+1],[c[i]],c="#d97823",zorder=5,label="Après L-BFGS")
            ax.set(title=label,xlabel="Étape Adam",ylabel="Coefficient")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        evo.savefig(folder/"coefficients.png",dpi=140)
        if no_show:
            plt.close(fig); plt.close(evo)
    p=metrics["pinn"]
    print(f"Moyenne {p['mean']:.4f}, amplitude {p['amplitude']:.4f}, "
          f"phase du maximum {p['peak_phase_cycles']} cycle",flush=True)
    print(f"Erreur L2 forçage : {100*p['force_relative_l2']:.2f} % ; "
          f"RMSE vitesse cachée : {p['hidden_velocity_rmse']:.5f}",flush=True)
    return metrics


def write_study(records,folder):
    lines=["# Identification du forçage pulsatile", "",
           f"Étapes Adam par essai : {sorted(set(row['steps'] for row in records))} ; affinage L-BFGS : au plus 300 itérations.",
           "Données simulées ; alpha et fréquence connus. G=a0+ac*cos(2πt)+as*sin(2πt).",
           "Vrai : moyenne 6, amplitude 4, maximum à 0,102416 cycle.",
           "36 observations ; aucune vitesse dans [0,30;0,65]. Le forçage vrai n'est pas fourni à l'entraînement.",
           "", "| Bruit | n | L2 forçage PINN (%) | L2 forçage analytique (%) | RMSE vitesse cachée PINN |",
           "|---|---:|---:|---:|---:|"]
    for noise in sorted(set(row["noise"] for row in records)):
        subset=[row for row in records if row["noise"]==noise]
        def cell(method,key,scale=1):
            values=np.array([row[method][key] for row in subset])*scale
            return f"{values.mean():.5f}"+(f" ± {values.std(ddof=1):.5f}" if len(values)>1 else "")
        lines.append(f"| {noise:.0%} | {len(subset)} | {cell('pinn','force_relative_l2',100)} | "
                     f"{cell('analytical','force_relative_l2',100)} | {cell('pinn','hidden_velocity_rmse')} |")
    lines += ["", "Moyenne ± écart-type si n>1, pas un intervalle de confiance.",
              "Les graines font varier ensemble bruit et initialisation. L'ajustement analytique utilise les mêmes observations.",
              "Les pertes physiques sont évaluées partout, mais avec les coefficients appris. La famille harmonique est connue.",
              "", "## Résultats individuels", "",
              "| Cas | Bruit | Graine | Moyenne | Amplitude | Maximum (cycle) | L2 débit (%) | L2 cisaillement (%) |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for index,row in enumerate(records,1):
        p=row["pinn"]; phase=p["peak_phase_cycles"]
        phase_text=f"{phase:.5f}" if phase is not None else "indéfinie"
        lines.append(f"| [cas {index}](case_{index:02d}/metrics.json) | {row['noise']:.0%} | {row['seed']} | "
                     f"{p['mean']:.5f} | {p['amplitude']:.5f} | {phase_text} | "
                     f"{100*p['flow_relative_l2']:.5f} | {100*p['shear_relative_l2']:.5f} |")
    lines += ["", "Chaque cas conserve observations, prédictions, historique, métriques et poids.",
              "En exécution simple, comparison.png et coefficients.png sont également produits.",
              "L'étude évite de régénérer ces figures à chaque cas. Même nombre d'étapes pour tous ses cas.",
              "Ces résultats ne constituent pas une validation médicale ni une supériorité générale sur les méthodes classiques."]
    (folder/"REPORT.md").write_text("\n".join(lines)+"\n")
