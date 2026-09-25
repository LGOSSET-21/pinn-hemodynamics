# PINN pulsatile : conception

Périmètre approuvé dans la conversation : animation du profil, carte espace-temps,
reconstruction à des instants sans mesures, débit et cisaillement comparés.

Tube droit rigide, fluide newtonien, écoulement axisymétrique pleinement développé.
Temps t en fraction de période, rayon r et vitesse u adimensionnés. Équation :
`alpha²/(2*pi)*u_t = u_rr + u_r/r + G(t)`, alpha=3 et G(t)=6+4*cos(2*pi*t).
Référence périodique de Womersley calculée avec les fonctions de Bessel complexes.
Pas de simulation du démarrage : on étudie un régime déjà périodique.

Réseau (r², sin(2*pi*t), cos(2*pi*t)) -> (1-r²)*MLP : périodicité,
symétrie et adhérence imposées. Aucune référence analytique dans la perte physique.
36 mesures simulées à 6 rayons et 6 phases, bruit de 3 % d'une échelle fixe 1,5.
Aucune mesure dans l'intervalle temporel [0,30;0,65]. L'équation et la force motrice
connue sont disponibles partout, y compris dans cet intervalle : interpolation
physiquement contrainte, pas prédiction d'une force inconnue ni extrapolation.

Sorties : comparison.png, spacetime.png, profiles.gif, metrics.json,
observations.json, predictions.npz, model.pt et README du résultat.
Débit Q*=2*int_0^1(r*u)dr = Q/(pi*R²*Uref).
Cisaillement signé exercé par le fluide sur la paroi tau*=-u_r(1,t),
à multiplier par mu*Uref/R pour retrouver des pascals.

Validation : tests physiques indépendants (limite stationnaire, résidu par
différences finies, périodicité, conditions aux limites, débit/cisaillement),
test du masque temporel, entraînement complet et métriques sur instants cachés.
Cibles pratiques par défaut : RMSE vitesse cachée / 1,5 < 5 %, erreur relative L2
sur débit et cisaillement < 10 %. Ne pas modifier les cibles après lecture des résultats.
Résultats dans un dossier horodaté, sources et anciens scripts préservés.
Bibliothèques existantes uniquement ; série de Bessel limitée à alpha=3.
