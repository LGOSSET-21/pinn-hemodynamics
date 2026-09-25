# Inverse Pulsatile Implementation Plan

**Goal:** réunir identification et reconstruction temporelle selon la proposition approuvée.
**Architecture:** base analytique et simulation dans inverse_pulsatile_physics.py ;
entraînement dans train_inverse_pulsatile.py ; sortie dans inverse_pulsatile_outputs.py.
**Tech Stack:** Python, PyTorch, NumPy, Matplotlib, unittest.
**Spec:** ../specs/2026-09-13-inverse-pulsatile.md

## Global Constraints
Conserver les anciens résultats et scripts. Pas de vérité dans la perte. Alpha connu.
36 observations, aucune dans [0.30,0.65]. Unités adimensionnées explicitement documentées.

- [x] Écrire les tests avant code : PDE pour un coefficient sinusoïdal non nul,
  ajustement exact sans bruit, refus d'un plan de mesure de rang insuffisant,
  gradient non nul du résidu vers les trois coefficients, débit et cisaillement.
- [x] Vérifier les échecs dus aux modules absents, puis créer base/simulateur,
  résidu paramétré et apprentissage Adam + L-BFGS avec observations seules.
- [x] Exécuter les tests physiques et un entraînement complet par défaut.
- [x] Exporter les quatre courbes comparatives et l'historique de coefficients.
  Vérifier visuellement et comparer les métriques aux objectifs définis avant calcul.
- [x] Exécuter 9 expériences : bruit 0,3,10 %, graines 42,43,44 ; résumer moyenne/écart-type.
- [x] Relire le code et la séparation entraînement/évaluation ; compléter guide,
  README et RESEARCH_LOG avec les chiffres effectivement obtenus.

Validation finale : 13 tests réussis ; essai principal à 4000 étapes et étude de 9 essais à 2000 étapes terminés. Figures inspectées, journaux et rapports mis à jour.
