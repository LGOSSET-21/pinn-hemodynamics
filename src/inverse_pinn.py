"""Identifier un coefficient physique à partir de vitesses simulées bruitées.

Lancer : python src/inverse_pinn.py
Étude répétée : python src/inverse_pinn.py --study --no-show
"""

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from train_pinn import derivative


class Profile(nn.Module):
    """Une courbe dont la pente au centre et la valeur à la paroi sont nulles."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 24), nn.Tanh(),
            nn.Linear(24, 24), nn.Tanh(), nn.Linear(24, 1),
        )

    def forward(self, r):
        # (1-r²) impose u(1)=0 ; utiliser r² comme entrée impose u'(0)=0.
        # La fonction apprise peut encore varier avec r : le profil parabolique
        # n'est pas imposé. Les deux réseaux utilisent cette même construction.
        return (1 - r**2) * self.network(r**2)


def simulate(n, noise, seed, true_a):
    """Le simulateur connaît A ; l'entraînement reçoit seulement r et u mesuré."""
    rng = np.random.default_rng(seed)
    r = np.linspace(0.05, 0.95, n)
    clean = true_a / 4 * (1 - r**2)
    # Bruit gaussien additif : écart-type en fraction de la vitesse centrale.
    observed = clean + rng.normal(0, noise * true_a / 4, n)
    return torch.tensor(r[:, None], dtype=torch.float32), torch.tensor(
        observed[:, None], dtype=torch.float32
    )


def fit(r_obs, u_obs, steps, seed, physics):
    """Aucune solution exacte ni valeur vraie de A dans cette fonction."""
    torch.manual_seed(seed)
    model = Profile()
    # exp(log_a) garantit un coefficient positif. Initialisation volontairement
    # éloignée de la valeur utilisée par le simulateur.
    log_a = nn.Parameter(torch.tensor(math.log(2.0)))
    groups = [{"params": model.parameters(), "lr": 0.003}]
    if physics:
        groups.append({"params": [log_a], "lr": 0.01})
    optimizer = torch.optim.Adam(groups)
    positions = torch.linspace(0, 1, 66)[1:-1].reshape(-1, 1)
    history = {"data": [], "physics": [], "a": []}
    start = perf_counter()
    for step in range(steps):
        loss_data = ((model(r_obs) - u_obs)**2).mean()
        loss_physics = torch.tensor(0.0)
        if physics:
            r = positions.clone().requires_grad_(True)
            du = derivative(model(r), r)
            residual = r * derivative(du, r) + du + log_a.exp() * r
            loss_physics = (residual**2).mean()
        loss = loss_data + loss_physics
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history["data"].append(loss_data.item())
        history["physics"].append(loss_physics.item())
        history["a"].append(log_a.exp().item())
    return model, log_a.exp().item() if physics else None, history, perf_counter() - start


def experiment(args, noise, seed, folder):
    r_obs, u_obs = simulate(args.points, noise, seed, args.true_a)
    pinn, a, history, seconds_pinn = fit(r_obs, u_obs, args.steps, seed, True)
    baseline, _, _, seconds_nn = fit(r_obs, u_obs, args.steps, seed, False)

    # Référence simple : ajuster directement u=A*(1-r²)/4 par moindres carrés.
    # Elle reçoit les mêmes observations, pas A vrai. Pour ce modèle simple,
    # c'est un concurrent essentiel : un PINN n'est pas nécessairement meilleur.
    basis = (1 - r_obs**2) / 4
    a_ls = max(0.0, ((basis * u_obs).sum() / (basis**2).sum()).item())
    grid = torch.linspace(0, 1, 401).reshape(-1, 1)
    with torch.no_grad():
        truth = args.true_a / 4 * (1 - grid**2)
        pred_pinn = pinn(grid)
        pred_nn = baseline(grid)
        pred_ls = a_ls / 4 * (1 - grid**2)
        rmse = lambda prediction: torch.sqrt(((prediction - truth)**2).mean()).item()
        rmse_data = torch.sqrt(((pinn(r_obs) - u_obs)**2).mean()).item()
    check = torch.linspace(0.001, 0.999, 400).reshape(-1, 1).requires_grad_(True)
    du = derivative(pinn(check), check)
    residual = check * derivative(du, check) + du + a * check
    residual_rmse = torch.sqrt((residual**2).mean()).item()
    result = {
        "seed": seed, "noise_fraction": noise, "points": args.points,
        "steps": args.steps, "true_a": args.true_a, "estimated_a_pinn": a,
        "estimated_a_least_squares": a_ls,
        "a_relative_error_percent": 100 * abs(a - args.true_a) / args.true_a,
        "rmse_pinn": rmse(pred_pinn), "rmse_network": rmse(pred_nn),
        "rmse_least_squares": rmse(pred_ls), "rmse_pinn_on_noisy_data": rmse_data,
        "weighted_physics_residual_rmse": residual_rmse,
        "seconds_pinn": seconds_pinn, "seconds_network": seconds_nn,
    }
    # L'évaluation utilise la vérité simulée uniquement après l'entraînement.
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("Résultat non fini : l'entraînement a divergé.")

    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (folder / "observations.json").write_text(json.dumps({
        "r": r_obs.flatten().tolist(), "u_observed": u_obs.flatten().tolist()
    }, indent=2) + "\n")
    torch.save({"model_state_dict": pinn.state_dict(), "a": a,
                "architecture": "Profile: (1-r²)*MLP(r²), widths 1-24-24-1, Tanh",
                "settings": vars(args), "seed": seed, "noise_fraction": noise},
               folder / "pinn_weights.pt")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = grid.flatten().numpy()
    for prediction, label, style in (
        (truth, "Vérité simulée", "k-"), (pred_pinn, "PINN inverse", "--"),
        (pred_nn, "Réseau sans équation", ":"), (pred_ls, "Ajustement analytique", "-."),
    ):
        axes[0, 0].plot(x, prediction.flatten().numpy(), style, label=label)
    axes[0, 0].scatter(r_obs.numpy(), u_obs.numpy(), c="black", s=24,
                       label="Observations bruitées", zorder=5)
    axes[0, 0].set(xlabel="Position r", ylabel="Vitesse adimensionnée u", title="Reconstruction du profil")
    axes[0, 1].plot(range(1, args.steps + 1), history["a"], label="A appris")
    axes[0, 1].axhline(args.true_a, color="black", linestyle="--", label="A vrai (vérification)")
    axes[0, 1].axhline(a_ls, color="gray", linestyle=":", label="A analytique ajusté")
    axes[0, 1].set(xlabel="Étape", ylabel="Coefficient A", title="Identification du coefficient")
    for key, label in (("data", "Mesures"), ("physics", "Équation")):
        axes[1, 0].semilogy(range(1, args.steps + 1), history[key], label=label)
    axes[1, 0].set(xlabel="Étape", ylabel="Erreur au carré", title="Les deux erreurs du PINN")
    for prediction, label in ((pred_pinn, "PINN"), (pred_nn, "Réseau sans équation"),
                              (pred_ls, "Ajustement analytique")):
        axes[1, 1].plot(x, (prediction - truth).flatten().numpy(), label=label)
    axes[1, 1].set(xlabel="Position r", ylabel="Prédiction − vérité", title="Erreur sur le profil exact")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"Données simulées • {args.points} mesures • bruit {noise:.0%} • graine {seed}")
    fig.tight_layout()
    fig.savefig(folder / "comparison.png", dpi=160)
    if args.no_show or args.study:
        plt.close(fig)
    print(f"Bruit {noise:.0%}, graine {seed} : A = {a:.4f} (vrai {args.true_a:g}) ; "
          f"RMSE PINN {result['rmse_pinn']:.5f}, réseau {result['rmse_network']:.5f}, "
          f"analytique {result['rmse_least_squares']:.5f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=12)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--true-a", type=float, default=6.0)
    parser.add_argument("--study", action="store_true", help="3 bruits × 3 graines")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed doit être positif ou nul.")
    if args.points < 2 or args.steps < 1 or not math.isfinite(args.noise) or args.noise < 0:
        parser.error("Il faut au moins 2 points, 1 étape et un bruit fini positif ou nul.")
    if not math.isfinite(args.true_a) or args.true_a <= 0:
        parser.error("--true-a doit être fini et strictement positif.")
    # Pour ce petit réseau, un seul fil de calcul évite des surcoûts sur CPU.
    torch.set_num_threads(1)
    folder = Path(__file__).resolve().parents[1] / "results" / (
        "inverse_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    cases = [(noise, args.seed + offset) for noise in (0.0, 0.05, 0.10)
             for offset in range(3)] if args.study else [(args.noise, args.seed)]
    results = []
    for index, (noise, seed) in enumerate(cases, 1):
        print(f"Expérience {index}/{len(cases)} : entraînement…", flush=True)
        results.append(experiment(args, noise, seed, folder / f"case_{index:02d}"))
    (folder / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    if args.study:
        lines = ["# Étude du PINN inverse", "", "Données simulées ; 3 répétitions par bruit.",
                 "Les écarts-types décrivent ces répétitions, pas un intervalle de confiance.", "",
                 "| Bruit | RMSE PINN | RMSE réseau | RMSE analytique | Erreur relative A |",
                 "|---|---|---|---|---|"]
        for noise in (0.0, 0.05, 0.10):
            subset = [row for row in results if row["noise_fraction"] == noise]
            cells = []
            for key in ("rmse_pinn", "rmse_network", "rmse_least_squares", "a_relative_error_percent"):
                values = np.array([row[key] for row in subset])
                cells.append(f"{values.mean():.5f} ± {values.std(ddof=1):.5f}")
            lines.append(f"| {noise:.0%} | " + " | ".join(cells) + " |")
        lines += ["", "Erreur relative A en %. Moyenne ± écart-type, n=3.",
                  "Les deux réseaux partagent architecture, initialisation, conditions aux limites",
                  "et nombre d'étapes. Le PINN utilise en plus l'équation et apprend A.",
                  "Les temps de calcul ne sont donc pas égaux. Aucun hyperparamètre n'est",
                  "optimisé sur ces résultats. L'ajustement analytique exploite la solution connue."]
        (folder / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"Résultats et poids sauvegardés : {folder}")
    if not args.no_show and not args.study:
        plt.show()


if __name__ == "__main__":
    main()
