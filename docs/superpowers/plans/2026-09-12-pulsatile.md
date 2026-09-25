# Pulsatile PINN Implementation Plan

> Exécution dans cette session selon executing-plans ; répertoire local non Git.

**Goal:** produire et vérifier les quatre sorties approuvées pour un PINN pulsatile.
**Architecture:** référence physique séparée de l'entraînement, sorties séparées.
**Tech Stack:** Python, NumPy, PyTorch, Matplotlib, Pillow, unittest.
**Spec:** ../specs/2026-09-12-pulsatile-design.md

## Global Constraints

Ne pas modifier sources/. Conserver les anciens scripts. Données simulées explicitement
signalées, aucune mesure dans [0,30;0,65], forçage connu dans tout le cycle.

## Task 1 : référence et tests physiques

- [x] Créer tests/test_pulsatile.py : limite G1=0 => u=1,5*(1-r²), Q*=0,75,
  tau*=3 ; vérifier l'équation par différences finies à r=0,2/0,5/0,8.
- [x] Exécuter unittest et constater l'absence du module.
- [x] Créer src/pulsatile_physics.py avec reference(r,t,amplitude=4),
  reference_flow(t), reference_shear(t), observations(seed).
- [x] Exécuter les tests jusqu'à validation des conditions et du masque.

## Task 2 : entraînement et évaluation

- [x] Tester la périodicité, l'adhérence et la symétrie d'un réseau non entraîné.
- [x] Créer src/train_pulsatile.py avec PulsatileNet et train(seed,steps).
  Calculer les dérivées spatiales/temporelles par autodifférentiation, avec résidu
  multiplié par r. Perte = mesures/1,5² + physique/10².
- [x] Entraîner avec Adam puis L-BFGS ; rapporter RMSE cachée, débit, cisaillement.

## Task 3 : sorties et guide

- [x] Créer src/pulsatile_outputs.py : courbes, cartes avec même échelle, animation
  Pillow, métriques et poids dans results/pulsatile_DATE_HEURE.
- [x] Exécuter le script complet sans affichage et vérifier les fichiers.
- [x] Inspecter graphiques et images de l'animation ; valider les seuils de la spec.
- [x] Ajouter PULSATILE_GUIDE.md et les commandes dans README.md.

## Vérification finale

7 tests réussis. Exécution complète : results/pulsatile_20260912_164517_509466_verified.
RMSE cachée / 1,5 = 0,39098 % ; L2 débit = 0,33193 % ; L2 cisaillement = 0,31497 %.
Relecture indépendante : aucune anomalie signalée. Figures inspectées ; GIF de 100 images.
