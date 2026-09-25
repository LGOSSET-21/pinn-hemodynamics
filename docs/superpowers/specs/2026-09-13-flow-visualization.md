# Visualisation interactive — périmètre approuvé

Deux coupes longitudinales schématiques (référence/PINN), mêmes particules
initiales et même échelle de vitesse. Aucun nouveau modèle ni entraînement.
Sources : les neuf cas inverse_pulsatile_20260913_165408_169589, tous à 2000 étapes.
Choix du bruit 0/3/10 % et des graines 42/43/44, lecture/pause et curseur de phase.
Couleurs vitesse ou erreur, observations radiales affichées uniquement à leur phase
sur un plan axial illustratif. Courbes G, Q et tau synchronisées avec la phase.

Le champ est pleinement développé : indépendant de la coordonnée axiale.
Les particules sont des traceurs illustratifs, pas des cellules ; déplacements
obtenus en intégrant la vitesse dans le temps, avec une échelle axiale illustrative.
Les positions initiales sont identiques dans les deux panneaux. Une nouvelle
sélection repart sur la même initialisation pour permettre la comparaison.

Vue inline, fragment local durable <1 Mo, aucune requête de données externe.
Données sous-échantillonnées et interpolées uniquement pour l'affichage ;
métriques reprises des évaluations complètes. Pausée au départ, contrôles clavier,
responsive, palette adaptée au thème. Source et générateur conservés dans le projet.
Tests de l'interpolation et de l'intégration, contrôle de provenance et syntaxe JS.
