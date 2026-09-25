# Identifier un forçage pulsatile — conception approuvée en conversation

Conserver alpha=3, tube rigide et régime périodique de l'expérience précédente.
Apprendre les trois coefficients de G(t)=a0+ac*cos(2*pi*t)+as*sin(2*pi*t),
en même temps que la vitesse. Vrai triplet du simulateur : (6,3.2,2.4),
amplitude 4, maximum à 0.102416 cycle. Initialisation (2,0.5,0.5).
La fonction d'entraînement ne reçoit que les observations, jamais le vrai triplet.
Réutiliser PulsatileNet et gradient, sans appeler le résidu à forçage connu.

36 mesures aux phases précédentes, aucune dans [0.30,0.65]. Bruit par défaut
sigma=0.03*1.5. Alpha, fréquence et forme harmonique sont connus ; phase,
amplitude et moyenne inconnues. Comparaison équitable à un ajustement linéaire
de la solution de Womersley sur les mêmes observations, avec contrôle du rang.

Sorties : forçage comparé, profil caché, débit et cisaillement, évolution des
trois coefficients ; données, poids, métriques et rapport lisible. Pas besoin
de refaire l'animation de l'expérience précédente. Étude 3 bruits x 3 graines,
enregistrée séparément ; distinguer incertitude due au bruit et approximation.
Objectifs de contrôle pour l'essai par défaut : erreur relative L2 forçage <10 %,
vitesse cachée RMSE/1.5 <5 %, débit et cisaillement L2 <10 %. Ces seuils ne sont
pas une garantie pour l'étude à fort bruit.
