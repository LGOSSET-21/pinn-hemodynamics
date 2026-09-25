"""Retrouver Poiseuille avec l'équation physique et deux conditions aux limites."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


def derivative(values, positions):
    """Dériver les sorties du réseau par rapport à leurs positions."""
    # create_graph=True permet ensuite de dériver à nouveau et d'entraîner
    # les paramètres à partir de ces dérivées. Le réseau traite chaque point séparément.
    return torch.autograd.grad(
        values, positions, grad_outputs=torch.ones_like(values), create_graph=True
    )[0]


def main():
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(1, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 64 positions intérieures où vérifier la physique : aucune vitesse cible !
    positions = torch.linspace(0, 1, 66)[1:-1].reshape(-1, 1)
    history = {"Physique": [], "Centre": [], "Paroi": []}

    for step in range(4000):
        r = positions.clone().requires_grad_(True)
        center = torch.tensor([[0.0]], requires_grad=True)
        wall = torch.tensor([[1.0]])
        u = model(r)
        du = derivative(u, r)
        d2u = derivative(du, r)

        # Équation cylindrique normalisée : u'' + u'/r + 4 = 0.
        # On multiplie par r : r*u'' + u' + 4*r = 0 pour éviter la division
        # par zéro. C'est équivalent pour r > 0, mais pondère le résidu par r.
        residual = r * d2u + du + 4 * r
        loss_physics = torch.mean(residual**2)
        loss_center = torch.mean(derivative(model(center), center)**2)  # u'(0) = 0
        loss_wall = torch.mean(model(wall)**2)                       # u(1) = 0
        loss = loss_physics + loss_center + loss_wall

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for name, value in zip(history, (loss_physics, loss_center, loss_wall)):
            history[name].append(value.item())
        if step == 0 or (step + 1) % 1000 == 0:
            print(f"Étape {step + 1:4d}/4000 — physique : {loss_physics.item():.6f}"
                    f" | centre : {loss_center.item():.6f} | paroi : {loss_wall.item():.6f}",
                    flush=True)

    # L'entraînement est terminé. La solution exacte apparaît seulement ici
    # pour vérifier le résultat ; elle n'a pas servi à calculer la perte.
    model.eval()
    r_plot = torch.linspace(0, 1, 201).reshape(-1, 1)
    with torch.no_grad():
        u_pred = model(r_plot)
        u_exact = 1 - r_plot**2
        rmse = torch.sqrt(torch.mean((u_pred - u_exact)**2)).item()
        at_08 = model(torch.tensor([[0.8]])).item()
        wall_error = abs(model(torch.tensor([[1.0]])).item())
    center = torch.tensor([[0.0]], requires_grad=True)
    center_error = abs(derivative(model(center), center).item())
    # Vérification du résidu sur d'autres positions que celles d'entraînement.
    r_check = torch.linspace(0.001, 0.999, 200).reshape(-1, 1).requires_grad_(True)
    du_check = derivative(model(r_check), r_check)
    residual_check = r_check * derivative(du_check, r_check) + du_check + 4 * r_check
    physics_rmse = torch.sqrt(torch.mean(residual_check**2)).item()
    print(f"À r = 0.8 : PINN = {at_08:.4f} ; valeur exacte = 0.3600")
    print(f"Erreur RMSE sur la vitesse : {rmse:.6f}")
    print(f"Vérification : |u'(0)| = {center_error:.6f} ; |u(1)| = {wall_error:.6f}")
    print(f"RMSE du résidu multiplié par r sur une autre grille : {physics_rmse:.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(r_plot.numpy(), u_exact.numpy(), label="Solution exacte (vérification)")
    axes[0].plot(r_plot.numpy(), u_pred.numpy(), "--", label="PINN")
    axes[0].set_title("Profil appris sans exemples de vitesse")
    axes[0].set_xlabel("Position radiale normalisée r")
    axes[0].set_ylabel("Vitesse normalisée u")
    for name, values in history.items():
        axes[1].semilogy(range(1, len(values) + 1), values, label=name)
    axes[1].set_title("Les trois erreurs d'entraînement")
    axes[1].set_xlabel("Étape d'entraînement")
    axes[1].set_ylabel("Erreur au carré (échelle logarithmique)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    output = Path(__file__).resolve().parents[1] / "results" / "pinn_fit.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(f"Graphique enregistré : {output}")
    plt.show()


if __name__ == "__main__":
    main()
