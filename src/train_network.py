"""Apprendre le profil de Poiseuille à partir d'exemples (pas encore un PINN)."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


def main():
    # Fixer le hasard permet de reproduire le même entraînement.
    torch.manual_seed(42)

    # 1. Préparer 20 exemples exacts : une position et sa vitesse.
    # Chaque ligne contient une seule valeur, d'où la forme (20, 1).
    r_train = torch.linspace(0, 1, 20).reshape(-1, 1)
    u_train = 1 - r_train**2

    # 2. Construire le réseau : 1 entrée (r), deux couches de 32 neurones,
    # puis 1 sortie (la vitesse prédite). Tanh permet d'apprendre une courbe.
    model = nn.Sequential(
        nn.Linear(1, 32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, 1),
    )

    # L'erreur est la moyenne des écarts au carré entre prédictions et cibles.
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = []

    # 3. Répéter : prédire, mesurer l'erreur, calculer les gradients,
    # puis ajuster les paramètres pour réduire l'erreur.
    for step in range(2000):
        prediction = model(r_train)
        loss = loss_function(prediction, u_train)
        optimizer.zero_grad()  # Effacer les gradients de l'étape précédente.
        loss.backward()       # Calculer comment chaque paramètre agit sur l'erreur.
        optimizer.step()      # Ajuster les paramètres.
        losses.append(loss.item())

        if step == 0 or (step + 1) % 400 == 0:
            print(f"Étape {step + 1:4d}/2000 — erreur moyenne : {loss.item():.6f}")

    # 4. Évaluer sur une grille plus fine, sans calculer de gradients.
    r_plot = torch.linspace(0, 1, 200).reshape(-1, 1)
    model.eval()
    with torch.no_grad():
        u_pred = model(r_plot)
        at_08 = model(torch.tensor([[0.8]])).item()
        rmse = torch.sqrt(torch.mean((u_pred - (1 - r_plot**2))**2)).item()
    print(f"À r = 0.8 : réseau = {at_08:.4f} ; valeur exacte = 0.3600")
    print(f"Erreur quadratique moyenne en racine sur la grille : {rmse:.6f}")

    # 5. Comparer les courbes et observer l'évolution de l'erreur.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(r_plot.numpy(), (1 - r_plot**2).numpy(), label="Solution exacte")
    axes[0].plot(r_plot.numpy(), u_pred.numpy(), "--", label="Prédiction du réseau")
    axes[0].scatter(r_train.numpy(), u_train.numpy(), s=22, label="20 exemples")
    axes[0].set_title("Le réseau apprend le profil de Poiseuille")
    axes[0].set_xlabel("Position radiale normalisée r")
    axes[0].set_ylabel("Vitesse normalisée u")
    axes[0].legend()
    axes[1].semilogy(range(1, len(losses) + 1), losses)
    axes[1].set_title("Erreur pendant l'entraînement")
    axes[1].set_xlabel("Étape d'entraînement")
    axes[1].set_ylabel("Erreur moyenne au carré (échelle logarithmique)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Enregistrer le résultat dans le projet, quel que soit le dossier de lancement.
    output = Path(__file__).resolve().parents[1] / "results" / "network_fit.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(f"Graphique enregistré : {output}")
    plt.show()


if __name__ == "__main__":
    main()
