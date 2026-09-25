# Identification du forçage pulsatile

Étapes Adam par essai : [4000] ; affinage L-BFGS : au plus 300 itérations.
Données simulées ; alpha et fréquence connus. G=a0+ac*cos(2πt)+as*sin(2πt).
Vrai : moyenne 6, amplitude 4, maximum à 0,102416 cycle.
36 observations ; aucune vitesse dans [0,30;0,65]. Le forçage vrai n'est pas fourni à l'entraînement.

| Bruit | n | L2 forçage PINN (%) | L2 forçage analytique (%) | RMSE vitesse cachée PINN |
|---|---:|---:|---:|---:|
| 3% | 1 | 1.04135 | 0.81369 | 0.00844 |

Moyenne ± écart-type si n>1, pas un intervalle de confiance.
Les graines font varier ensemble bruit et initialisation. L'ajustement analytique utilise les mêmes observations.
Les pertes physiques sont évaluées partout, mais avec les coefficients appris. La famille harmonique est connue.

## Résultats individuels

| Cas | Bruit | Graine | Moyenne | Amplitude | Maximum (cycle) | L2 débit (%) | L2 cisaillement (%) |
|---|---|---|---:|---:|---:|---:|---:|
| [cas 1](case_01/metrics.json) | 3% | 42 | 6.02331 | 3.93050 | 0.10486 | 0.65880 | 0.84555 |

Chaque cas conserve observations, prédictions, historique, métriques et poids.
En exécution simple, comparison.png et coefficients.png sont également produits.
L'étude évite de régénérer ces figures à chaque cas. Même nombre d'étapes pour tous ses cas.
Ces résultats ne constituent pas une validation médicale ni une supériorité générale sur les méthodes classiques.
