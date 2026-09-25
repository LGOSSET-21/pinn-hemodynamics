# Étude du PINN inverse

Données simulées ; 3 répétitions par bruit.
Les écarts-types décrivent ces répétitions, pas un intervalle de confiance.

| Bruit | RMSE PINN | RMSE réseau | RMSE analytique | Erreur relative A |
|---|---|---|---|---|
| 0% | 0.00027 ± 0.00011 | 0.00244 ± 0.00052 | 0.00000 ± 0.00000 | 0.04987 ± 0.01981 |
| 5% | 0.01812 ± 0.01025 | 0.03743 ± 0.01843 | 0.01812 ± 0.01028 | 1.61884 ± 0.96522 |
| 10% | 0.03624 ± 0.02053 | 0.08008 ± 0.03990 | 0.03624 ± 0.02056 | 3.24456 ± 1.86869 |

Erreur relative A en %. Moyenne ± écart-type, n=3.
Les deux réseaux partagent architecture, initialisation, conditions aux limites
et nombre d'étapes. Le PINN utilise en plus l'équation et apprend A.
Les temps de calcul ne sont donc pas égaux. Aucun hyperparamètre n'est
optimisé sur ces résultats. L'ajustement analytique exploite la solution connue.
