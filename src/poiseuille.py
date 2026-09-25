"""Tracer le profil de vitesse normalisé d'un écoulement de Poiseuille."""

import matplotlib.pyplot as plt
import numpy as np


def main():
    # r est la distance au centre du tube divisée par son rayon :
    # r = 0 au centre et r = 1 à la paroi.
    r = np.linspace(0, 1, 200)

    # La vitesse est divisée par sa valeur maximale au centre du tube.
    # Elle vaut donc 1 au centre et 0 à la paroi (adhérence).
    u = 1 - r**2
    print("Vitesse normalisée à mi-rayon :", 1 - 0.5**2)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(r, u, linewidth=2, label=r"$u(r) = 1 - r^2$")
    ax.set_title("Profil de vitesse normalisé de Poiseuille")
    ax.set_xlabel("Position radiale normalisée r (centre → paroi)")
    ax.set_ylabel("Vitesse normalisée u(r)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    # Ouvre une fenêtre contenant le graphique.
    plt.show()


if __name__ == "__main__":
    main()
