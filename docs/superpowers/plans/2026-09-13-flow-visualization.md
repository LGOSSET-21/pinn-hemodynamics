# Flow visualization implementation plan

**Goal:** explorer les résultats existants avec tube animé et courbes synchronisées.
**Spec:** ../specs/2026-09-13-flow-visualization.md
**Architecture:** template littéral HTML + fonctions mathématiques JS + générateur
Python lisant les sorties locales, sans appels réseau pour les données.

- [x] Tester interpolation et intégration JS avec un champ linéaire connu.
- [x] Extraire les neuf cas à budget comparable, vérifier grilles, phases et sources.
- [x] Créer animation comparative, sélecteurs de bruit/graine, lecture et curseur.
- [x] Ajouter les trois graphiques synchronisés et leur lecture au survol.
- [x] Vérifier syntaxe, données et taille du fragment ; conserver la provenance dans le journal.
