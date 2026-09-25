"""Évaluation après apprentissage, figures et animation de l'expérience pulsatile."""
import json

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch

from pulsatile_physics import reference,reference_flow,reference_shear,observations,VELOCITY_SCALE


def evaluate(model):
    r,t=np.linspace(0,1,121),np.linspace(0,1,101)
    rr,tt=np.meshgrid(r,t)
    x=torch.tensor(np.column_stack((rr.ravel(),tt.ravel())),dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred=model(x).numpy().reshape(rr.shape)
    truth=reference(rr,tt)
    flow_pred=2*np.trapz(pred*r,r,axis=1)
    wall=torch.tensor(np.column_stack((np.ones_like(t),t)),dtype=torch.float32,requires_grad=True)
    grad=torch.autograd.grad(model(wall).sum(),wall)[0]
    shear_pred=-grad[:,0].detach().numpy()
    flow_true,shear_true=reference_flow(t),reference_shear(t)
    hidden=(t>=0.30)&(t<=0.65)
    rmse=lambda a,b: float(np.sqrt(np.mean((a-b)**2)))
    relative=lambda a,b: float(np.linalg.norm(a-b)/np.linalg.norm(b))
    metrics={
        "velocity_rmse":rmse(pred,truth),
        "hidden_velocity_rmse":rmse(pred[hidden],truth[hidden]),
        "hidden_velocity_rmse_over_scale":rmse(pred[hidden],truth[hidden])/VELOCITY_SCALE,
        "flow_relative_l2":relative(flow_pred,flow_true),
        "shear_relative_l2":relative(shear_pred,shear_true),
        "hidden_flow_relative_l2":relative(flow_pred[hidden],flow_true[hidden]),
        "hidden_shear_relative_l2":relative(shear_pred[hidden],shear_true[hidden]),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise RuntimeError("Métriques non finies : vérifier l'entraînement.")
    metrics["accuracy_targets_met"]=(metrics["hidden_velocity_rmse_over_scale"]<0.05
        and metrics["flow_relative_l2"]<0.10 and metrics["shear_relative_l2"]<0.10)
    fields=dict(r=r,t=t,pred=pred,truth=truth,flow_pred=flow_pred,flow_true=flow_true,
                shear_pred=shear_pred,shear_true=shear_true,hidden=hidden)
    return fields,metrics


def export_results(model,history,seconds,args,folder):
    f,metrics=evaluate(model)
    metrics.update(training_seconds=seconds,settings=vars(args),
                   torch_version=torch.__version__,numpy_version=np.__version__,
                   data_description="36 observations simulées ; sigma=0.045 ; phases cachées [0.30,0.65]",
                   physics_description="alpha=3 ; G=6+4*cos(2*pi*t) connu sur tout le cycle")
    # Résidu vérifié en d'autres points que ceux de l'entraînement.
    from train_pulsatile import residual
    check=torch.quasirandom.SobolEngine(2,scramble=True,seed=args.seed+1).draw(1024)
    check[:,0]=0.005+0.99*check[:,0]
    check.requires_grad_(True)
    metrics["weighted_residual_rmse"]=float(torch.sqrt((residual(model,check)**2).mean()).detach())
    folder.mkdir(parents=True,exist_ok=False)
    (folder/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    x_obs,y_obs=observations(args.seed)
    (folder/"observations.json").write_text(json.dumps({"r_t":x_obs.tolist(),"u":y_obs.tolist()},indent=2)+"\n")
    np.savez_compressed(folder/"predictions.npz",**f,adam_history=history)
    torch.save({"state_dict":model.state_dict(),"settings":vars(args),
                "architecture":"PulsatileNet: (1-r²)*MLP(r²,sin(2*pi*t),cos(2*pi*t))"},folder/"model.pt")
    r,t=f["r"],f["t"]
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    for phase,color in ((0.10,"#136f9c"),(0.45,"#c55a11"),(0.85,"#5b7e32")):
        index=np.argmin(abs(t-phase))
        label=f"t/T={phase:.2f}"+(" — sans mesures" if phase==0.45 else "")
        axes[0,0].plot(r,f["truth"][index],color=color,label=label+" (réf.)")
        axes[0,0].plot(r,f["pred"][index],"--",color=color)
    axes[0,0].set(title="Profils : référence pleine, PINN en tirets",xlabel="Rayon r",ylabel="Vitesse u")
    for column,label in ((0,"Mesures"),(1,"Équation")):
        axes[0,1].semilogy(np.arange(1,len(history)+1),history[:,column],label=label)
    axes[0,1].set(title="Apprentissage Adam (avant affinage L-BFGS)",xlabel="Étape",ylabel="Perte mise à l'échelle")
    for ax,key,ylabel,title in ((axes[1,0],"flow","Q* = Q / (π R² Uref)","Débit au cours du cycle"),
                                (axes[1,1],"shear","τ* = −∂u/∂r à la paroi","Cisaillement signé à la paroi")):
        ax.plot(t,f[key+"_true"],color="#136f9c",label="Référence")
        ax.plot(t,f[key+"_pred"],"--",color="#d97823",label="PINN")
        ax.axvspan(0.30,0.65,color="gray",alpha=0.15,label="Aucune mesure de vitesse")
        ax.set(title=title,xlabel="Fraction de cycle t/T",ylabel=ylabel)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Écoulement pulsatile simulé • 36 mesures bruitées • forçage connu")
    fig.savefig(folder/"comparison.png",dpi=150)

    heat,axs=plt.subplots(1,3,figsize=(14,4.8),layout="constrained")
    vmin=min(f["truth"].min(),f["pred"].min())
    vmax=max(f["truth"].max(),f["pred"].max())
    for ax,key,title in ((axs[0],"truth","Référence"),(axs[1],"pred","PINN")):
        im=ax.pcolormesh(t,r,f[key].T,shading="auto",cmap="viridis",vmin=vmin,vmax=vmax)
        ax.set_title(title)
    heat.colorbar(im,ax=list(axs[:2]),label="Vitesse u — même échelle")
    axs[1].scatter(x_obs[:,1],x_obs[:,0],s=15,c="white",edgecolors="black",linewidths=0.5,label="Mesures")
    axs[1].legend(fontsize=8,loc="upper right")
    error=f["pred"]-f["truth"]
    bound=max(abs(error).max(),1e-8)
    im=axs[2].pcolormesh(t,r,error.T,shading="auto",cmap="RdBu_r",vmin=-bound,vmax=bound)
    heat.colorbar(im,ax=axs[2],label="PINN − référence")
    axs[2].set_title("Erreur signée")
    for ax in axs:
        ax.axvline(0.30,color="white",ls="--",lw=1.5)
        ax.axvline(0.65,color="white",ls="--",lw=1.5)
        ax.set(xlabel="Fraction de cycle t/T",ylabel="Rayon r")
    heat.suptitle("Entre les lignes : aucun exemple de vitesse ; équation disponible partout")
    heat.savefig(folder/"spacetime.png",dpi=150)

    animation_fig,ax=plt.subplots(figsize=(8,4.8))
    exact_line,=ax.plot(r,f["truth"][0],color="#136f9c",lw=2,label="Référence")
    pred_line,=ax.plot(r,f["pred"][0],"--",color="#d97823",lw=2,label="PINN")
    points=ax.scatter([],[],c="black",s=35,label="Mesures à cet instant",zorder=4)
    ax.set(xlim=(0,1),ylim=(min(-0.05,vmin-0.1),vmax+0.15),xlabel="Rayon r",ylabel="Vitesse u")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower left",fontsize=9)
    animation_fig.tight_layout()
    # Le titre animé occupe deux lignes ; réserver sa place avant le rendu.
    animation_fig.subplots_adjust(top=0.82)
    def frame(index):
        phase=t[index]
        exact_line.set_ydata(f["truth"][index])
        pred_line.set_ydata(f["pred"][index])
        match=np.isclose(x_obs[:,1],phase,atol=1e-6)
        points.set_offsets(np.column_stack((x_obs[match,0],y_obs[match,0])))
        status="PLAGE SANS MESURES" if f["hidden"][index] else (
            "phase observée" if match.any() else "instant sans mesure")
        ax.set_title(f"Cycle pulsatile simulé • t/T = {phase:.2f}\n{status}")
        return exact_line,pred_line,points
    # 100 images, 1 cycle en 5 secondes de lecture (pas une durée physique).
    animation=FuncAnimation(animation_fig,frame,frames=range(100),interval=50,blit=False)
    print("Enregistrement de l'animation (100 images)…",flush=True)
    animation.save(folder/"profiles.gif",writer=PillowWriter(fps=20),dpi=100)
    plt.close(animation_fig)
    text=("# Résultat pulsatile\n\nDonnées entièrement simulées.\n\n"
          "- profiles.gif : cycle animé, observations affichées seulement à leurs phases.\n"
          "- spacetime.png : référence, reconstruction et erreur ; zone temporelle cachée indiquée.\n"
          "- comparison.png : profils, apprentissage, débit et cisaillement.\n"
          "- metrics.json : erreurs, réglages et temps de calcul.\n"
          "- model.pt : poids sauvegardés ; aucun rechargement automatique au prochain lancement.\n"
          "- observations.json et predictions.npz : données numériques reproductibles.\n\n"
          "Le forçage et l'équation sont connus partout. Ce résultat teste une reconstruction\n"
          "dans un régime périodique, pas une prédiction d'un futur inconnu.\n")
    (folder/"README.md").write_text(text)
    print(f"RMSE vitesse aux instants cachés : {metrics['hidden_velocity_rmse']:.6f} "
          f"({100*metrics['hidden_velocity_rmse_over_scale']:.2f} % de l'échelle 1,5)")
    print(f"Erreur relative L2 débit : {100*metrics['flow_relative_l2']:.2f} %")
    print(f"Erreur relative L2 cisaillement : {100*metrics['shear_relative_l2']:.2f} %")
    print(f"Cibles de précision atteintes : {metrics['accuracy_targets_met']}")
    print(f"Résultats enregistrés : {folder}")
    if args.no_show:
        plt.close(fig)
        plt.close(heat)
    else:
        plt.show()
