# Identification du forçage pulsatile

Budget vérifié : 2 000 étapes Adam par essai, puis affinage L-BFGS (au plus 300 itérations).

Données simulées ; alpha et fréquence connus. G=a0+ac*cos(2πt)+as*sin(2πt).
Vrai : moyenne 6, amplitude 4, maximum à 0,102416 cycle.
36 observations ; aucune vitesse dans [0,30;0,65]. Le forçage vrai n'est pas fourni à l'entraînement.

| Bruit | n | L2 forçage PINN (%) | L2 forçage analytique (%) | RMSE vitesse cachée PINN |
|---|---:|---:|---:|---:|
| 0% | 3 | 0.12324 ± 0.09602 | 0.00000 ± 0.00000 | 0.00119 ± 0.00055 |
| 3% | 3 | 1.16794 ± 0.47066 | 0.88820 ± 0.24237 | 0.00610 ± 0.00195 |
| 10% | 3 | 4.40394 ± 2.51952 | 2.96066 ± 0.80790 | 0.02978 ± 0.01727 |

Moyenne ± écart-type si n>1, pas un intervalle de confiance.
Les graines font varier ensemble bruit et initialisation. L'ajustement analytique utilise les mêmes observations.
Les pertes physiques sont évaluées partout, mais avec les coefficients appris. La famille harmonique est connue.

## Résultats individuels

| Cas | Bruit | Graine | Moyenne | Amplitude | Maximum (cycle) | L2 débit (%) | L2 cisaillement (%) |
|---|---|---|---:|---:|---:|---:|---:|
| [cas 1](case_01/metrics.json) | 0% | 42 | 6.00258 | 3.99348 | 0.10268 | 0.03637 | 0.07907 |
| [cas 2](case_02/metrics.json) | 0% | 43 | 6.00513 | 3.99676 | 0.10321 | 0.11875 | 0.17583 |
| [cas 3](case_03/metrics.json) | 0% | 44 | 5.99934 | 3.99672 | 0.10240 | 0.03758 | 0.04949 |
| [cas 4](case_04/metrics.json) | 3% | 42 | 6.01150 | 3.93767 | 0.10376 | 0.46722 | 0.64415 |
| [cas 5](case_05/metrics.json) | 3% | 43 | 5.94924 | 3.98845 | 0.10503 | 0.59128 | 0.89744 |
| [cas 6](case_06/metrics.json) | 3% | 44 | 6.02693 | 4.06014 | 0.09683 | 0.73969 | 1.18719 |
| [cas 7](case_07/metrics.json) | 10% | 42 | 6.01312 | 3.97417 | 0.10949 | 1.72991 | 1.73254 |
| [cas 8](case_08/metrics.json) | 10% | 43 | 5.87905 | 3.96230 | 0.11718 | 1.97881 | 3.14278 |
| [cas 9](case_09/metrics.json) | 10% | 44 | 6.06171 | 4.20674 | 0.07860 | 3.29422 | 4.74475 |

Chaque cas conserve observations, prédictions, historique, métriques et poids.
En exécution simple, comparison.png et coefficients.png sont également produits.
L'étude évite de régénérer ces figures à chaque cas. Même nombre d'étapes pour tous ses cas.
Ces résultats ne constituent pas une validation médicale ni une supériorité générale sur les méthodes classiques.
