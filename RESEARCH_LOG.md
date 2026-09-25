# Journal de recherche — PINNs et hémodynamique

Dernière mise à jour : 13 septembre 2026 (Hong Kong).

## Objectif et contexte pour reprendre le projet

Louis souhaite comprendre les PINNs en profondeur et conserver une trace des
expériences pour pouvoir présenter le travail plus tard. Il travaille sur Mac,
dans VS Code, et apprend Python et les notions d'entraînement progressivement.
Expliquer chaque nouvelle hypothèse et distinguer code, terminal et résultats.

Projet pédagogique assisté par IA : les scripts ont été préparés avec l'assistant,
puis exécutés et discutés avec Louis. Ne pas présenter ces exercices comme une
nouvelle méthode scientifique, une validation clinique ou une recherche autonome
sans préciser les contributions. Toutes les observations utilisées sont simulées.

Ce journal est la référence durable du projet. Lors d'une reprise, le lire avec
README.md et les métriques des expériences concernées. Pour les étapes suivantes,
ajouter la question, les hypothèses, les paramètres, le chemin des sorties,
les résultats et leurs limites. Ne pas remplacer les résultats anciens.

## Étapes réalisées

### 9–10 septembre : profil exact et premier réseau

1. Création du projet Python, environnement `.venv`, NumPy et Matplotlib.
2. `src/poiseuille.py` trace `u(r)=1-r²` entre le centre et la paroi.
3. `src/train_network.py` apprend 20 exemples exacts calculés avec cette formule.
   Deux couches de 32 neurones, Tanh, Adam, 2 000 étapes.
4. Compréhension des poids, biais, prédiction, fonction d'erreur et gradients.

Résultat discuté et vérifié dans la conversation : à r=0,8, le réseau prédit
environ 0,3564 pour une valeur exacte de 0,36 ; RMSE du profil environ 0,002563.
Ces nombres proviennent du compte rendu d'exécution dans la conversation,
pas d'un fichier de métriques structuré pour cette première étape.

### 10 septembre : premier PINN direct

`src/train_pinn.py` cherche le profil à partir de l'équation
`u''+u'/r+4=0`, avec `u'(0)=0` et `u(1)=0`, sans vitesses d'entraînement.
Le résidu est multiplié par r pour éviter la division par zéro ; cela modifie
la pondération près du centre. Les conditions aux limites sont pénalisées.

Résultat d'exécution discuté : u(0,8)≈0,3604 ; RMSE vitesse≈0,000418.
La référence analytique sert à vérifier le résultat, pas à entraîner ce PINN.
Une erreur physique faible aux points vérifiés ne prouve pas une exactitude partout.

### 10–12 septembre : problème inverse et bruit

`src/inverse_pinn.py` apprend simultanément le profil et A dans
`u''+u'/r+A=0` à partir de vitesses bruitées. A vrai=6 ; initialisation A=2.
L'échelle de vitesse est désormais indépendante de la vitesse centrale,
qui vaut 1,5 dans ce cas. A est un coefficient adimensionné combinant gradient
de pression et viscosité, pas une pression en pascals.

Comparaisons : PINN, réseau sans équation intérieure et ajustement analytique
par moindres carrés. Les deux réseaux imposent les mêmes conditions aux limites.
Les trois méthodes reçoivent les mêmes observations, sans la valeur vraie de A.

Étude enregistrée : [rapport de 9 expériences](results/inverse_20260910_161917_740504/REPORT.md).
12 observations, 3 graines (42, 43, 44) par niveau de bruit :

| Bruit | RMSE PINN moyenne ± écart-type | RMSE réseau | RMSE analytique |
|---|---:|---:|---:|
| 0 % | 0,00027 ± 0,00011 | 0,00244 ± 0,00052 | ≈0 |
| 5 % | 0,01812 ± 0,01025 | 0,03743 ± 0,01843 | 0,01812 ± 0,01028 |
| 10 % | 0,03624 ± 0,02053 | 0,08008 ± 0,03990 | 0,03624 ± 0,02056 |

Essais supplémentaires exécutés le 12 septembre (graine 42) :

| Mesures | Bruit | A estimé | RMSE PINN | RMSE réseau | Dossier dans results/ |
|---|---|---:|---:|---:|---|
| 12 | 5 % | 5,9052 | 0,01701 | 0,05091 | inverse_20260912_161502_125967 |
| 12 | 10 % | 5,8087 | 0,03401 | 0,11361 | inverse_20260912_162101_600821 |
| 4 | 10 % | 5,9877 | 0,00428 | 0,20816 | inverse_20260912_162239_134506 |

Le dossier inverse_20260912_162432_514315 répète le dernier essai avec la même
graine : ce n'est pas une répétition statistique indépendante. Le meilleur
résultat avec 4 mesures ne prouve pas que moins de mesures aide ; positions et
réalisation du bruit changent, et l'amplitude ajustée est ici particulièrement favorable.
Il manque une étude répétée comparant systématiquement 4 et 12 mesures.

Conclusion limitée à ces essais : l'équation améliore la reconstruction par
rapport au réseau sans équation intérieure, mais la méthode analytique est aussi
précise et beaucoup plus directe sur ce problème. Trois graines ne constituent
pas une étude exhaustive d'incertitude.

### 12 septembre : écoulement pulsatile

`src/train_pulsatile.py` prédit u(r,t). Référence de Womersley pour un tube droit
rigide, fluide newtonien, régime déjà périodique et pleinement développé :

```text
alpha²/(2*pi)*u_t = u_rr + u_r/r + G(t)
alpha=3 ; G(t)=6+4*cos(2*pi*t)
```

36 mesures simulées, bruit sigma=0,045 ; aucune mesure dans [0,30;0,65] du cycle.
L'équation ET le forçage sont connus partout, y compris dans cet intervalle.
C'est une reconstruction à des phases non observées, pas une prédiction d'un
futur ou d'un forçage inconnu. Le réseau impose périodicité, symétrie et adhérence.
4 000 étapes Adam puis affinage L-BFGS, graine 42.

Résultat à utiliser pour une présentation :
[dossier vérifié](results/pulsatile_20260912_164517_509466_verified/README.md).

| Critère | Résultat |
|---|---:|
| RMSE vitesse sur tout le cycle | 0,006653 |
| RMSE vitesse sur phases cachées | 0,005865 |
| RMSE cachée / échelle 1,5 | 0,391 % |
| Erreur relative L2 débit, tout le cycle | 0,332 % |
| Erreur relative L2 cisaillement, tout le cycle | 0,315 % |
| Erreur relative L2 débit, phases cachées | 0,461 % |
| Erreur relative L2 cisaillement, phases cachées | 0,349 % |

Sources exactes : [metrics.json](results/pulsatile_20260912_164517_509466_verified/metrics.json).
Les pourcentages L2 ne sont pas des erreurs maximales instantanées.

Les dossiers pulsatile_20260912_172149_756756 et pulsatile_20260912_172700_030418
contiennent les mêmes métriques avec la même graine. Ils confirment la reproduction
du résultat, pas la robustesse à d'autres bruits. Les temps d'entraînement varient
fortement : environ 264 s dans le premier essai, puis 37 s dans ces deux essais.
Ne pas traiter ces durées comme un benchmark contrôlé ni inclure l'export graphique.

Sorties disponibles : profiles.gif, spacetime.png, comparison.png, observations,
prédictions numériques, métriques et poids sauvegardés. Le dossier `_verified`
contient une correction de marge du titre animé ; ce n'est pas un nouvel entraînement.
Sept tests physiques et fonctionnels ont passé le 12 septembre : référence,
limite stationnaire, équation par différences finies, conditions, périodicité,
séparation des observations et calculs du débit/cisaillement. Relecture indépendante
du code : aucune anomalie signalée après contrôle des formules.

## Vocabulaire déjà abordé

- RMSE : Root Mean Square Error, racine de l'erreur quadratique moyenne.
- Entraîner modifie les paramètres ; prédire seul ne les modifie pas.
- Même graine et mêmes paramètres : même réalisation, pas un nouvel essai indépendant.
- Résidu physique et erreur par rapport à la vérité mesurent des choses différentes.
- Peu d'erreur sur des mesures bruitées ne garantit pas une bonne reconstruction.
- Le cisaillement dépend de la pente à la paroi, pas seulement de la vitesse.

## Transition proposée le 13 septembre — mise en œuvre ci-dessous

Prochaine question : **peut-on retrouver le forçage pulsatile inconnu à partir
de quelques vitesses, tout en reconstruisant le cycle ?**

Réunir les deux expériences déjà comprises : identification de paramètre et
dynamique temporelle. Conserver alpha et la géométrie connus. Apprendre les
coefficients de `G(t)=a0+ac*cos(2*pi*t)+as*sin(2*pi*t)` en même temps que u(r,t).
En déduire moyenne, amplitude et phase. Ne pas apprendre simultanément viscosité
et forçage sans examiner l'identifiabilité. Des observations à plusieurs phases
seront nécessaires pour contraindre la variation temporelle.

Évaluation à prévoir avant toute affirmation : forçage reconstruit comparé au vrai,
erreurs de vitesse, débit et cisaillement ; plusieurs graines et niveaux de bruit ;
comparaison à un ajustement inverse de la solution analytique de Womersley.
Les valeurs vraies des paramètres et la référence ne doivent pas entrer dans la
perte physique ; les coefficients entraînables figurent bien dans le résidu.
Cette étape a ensuite été autorisée par Louis et mise en œuvre ; voir les résultats ci-dessous.

### 13 septembre : forçage pulsatile inconnu

Nouveau lancement : `python src/train_inverse_pulsatile.py`.
La géométrie, alpha=3 et la fréquence restent connus ; on apprend a0, ac et as
en même temps que le champ de vitesse. Le simulateur utilise (6 ; 3,2 ; 2,4),
soit une moyenne de 6, une amplitude de 4 et un maximum à 0,102416 cycle.
L'initialisation vaut (2 ; 0,5 ; 0,5). La perte physique ne reçoit pas le vrai forçage.
La forme harmonique est connue, ce n'est pas une découverte libre d'équation.

36 observations, bruit sigma=0,045 (3 % de l'échelle 1,5), mêmes phases cachées
[0,30;0,65]. Comparaison à un ajustement linéaire de la solution de Womersley,
sur les mêmes observations ; rang 3 et conditionnement de la matrice ≈5,94.

Premier essai, graine 42, 4 000 étapes Adam puis affinage L-BFGS :

| Quantité | Vrai | PINN | Ajustement analytique |
|---|---:|---:|---:|
| Moyenne du forçage | 6 | 6,02331 | 6,03898 |
| Amplitude | 4 | 3,93050 | 3,99820 |
| Phase du maximum (cycle) | 0,102416 | 0,104862 | 0,104558 |
| Erreur relative L2 du forçage | — | 1,04135 % | 0,81369 % |
| RMSE vitesse aux phases cachées | — | 0,008442 | 0,011173 |
| Erreur relative L2 débit | — | 0,65880 % | 0,71039 % |
| Erreur relative L2 cisaillement | — | 0,84555 % | 0,72383 % |

Source : [métriques du premier essai](results/inverse_pulsatile_20260913_165145_051547_verified/case_01/metrics.json).
Le PINN respecte les seuils définis avant le calcul. L'analytique estime mieux
le forçage dans cet essai, mais le PINN a une RMSE cachée un peu plus faible :
la conclusion dépend de la grandeur évaluée.

Treize tests ont passé après ajout de cette étape (six nouveaux tests, sept
existants). La relecture indépendante n'a trouvé ni fuite de vérité dans la
perte, ni incohérence de phase/signe du cisaillement. Un caractère de titre
non pris en charge par la police a été remplacé ; les figures finales gardent
les mêmes couleurs pour chaque méthode.

Une étude de 9 essais (bruits 0 %, 3 %, 10 % ; graines 42,43,44) a été terminée
avec **2 000 étapes Adam par essai**, puis le même affinage L-BFGS. Elle utilise
donc un budget différent de l'essai principal à 4 000 étapes. Ses conclusions
doivent être comparées à réglages identiques au sein de l'étude.

| Bruit | L2 forçage PINN (%) | L2 forçage analytique (%) | RMSE vitesse cachée PINN |
|---|---:|---:|---:|
| 0 % | 0,12324 ± 0,09602 | ≈0 | 0,00119 ± 0,00055 |
| 3 % | 1,16794 ± 0,47066 | 0,88820 ± 0,24237 | 0,00610 ± 0,00195 |
| 10 % | 4,40394 ± 2,51952 | 2,96066 ± 0,80790 | 0,02978 ± 0,01727 |

Moyenne ± écart-type, n=3 par bruit. Sources :
[rapport d'étude](results/inverse_pulsatile_20260913_165408_169589/REPORT.md) et
[données complètes](results/inverse_pulsatile_20260913_165408_169589/summary.json).
L'erreur sur le forçage et sa variabilité augmentent avec le bruit dans cette
étude. À 10 %, les erreurs individuelles vont de 1,91 % à 6,95 %. L'ajustement
analytique a une erreur moyenne de forçage inférieure à celle du PINN aux trois
niveaux ; aucun avantage général du PINN n'est donc démontré ici.

La référence sait exploiter directement la famille exacte de solutions.
Cette étude mesure ensemble l'effet du bruit et de l'initialisation, avec un
budget numérique fini ; elle ne permet pas d'attribuer tout l'écart au bruit.
Les treize tests ont repassé après les dernières corrections. L'expérience
principale et les neuf cas ont été exécutés jusqu'au bout ; les poids, données,
métriques, figures principales et rapports sont conservés.

Pour reproduire l'étude :
`python src/train_inverse_pulsatile.py --study --steps 2000 --no-show`.
Pour l'expérience principale : `python src/train_inverse_pulsatile.py`.

Guide : [identifier le forçage pulsatile](INVERSE_PULSATILE_GUIDE.md).

Étape ultérieure plus ambitieuse : géométrie avec rétrécissement. Définir d'abord
si l'on choisit un canal 2D ou un tube axisymétrique ; ne pas confondre les deux.
Prévoir une référence numérique indépendante et la vérification des équations
avant toute visualisation interprétée comme un résultat hémodynamique.

## Trame pour présenter le travail plus tard

1. Question : reconstruire un écoulement et certains paramètres avec peu de données.
2. Méthode : réseau supervisé, ajout de l'équation, problème inverse, ajout du temps.
3. Vérification : solutions exactes, tests physiques, bruit, mesures rares, phases cachées.
4. Résultats : afficher les comparaisons et leurs paramètres, puis l'animation pulsatile.
5. Limites : simulation favorable, petit nombre de graines, forçage connu au stade pulsatile,
   aucune validation médicale, aucune supériorité générale sur les méthodes classiques.
6. Contributions : expliciter l'aide de l'IA pour le code et les analyses, et ce que Louis
   a exécuté, étudié et compris. La présentation finale reste à rédiger.

## Références et documents

- [Guide des PINNs du projet](UNDERSTANDING_PINNS.md)
- [Guide pulsatile du projet](PULSATILE_GUIDE.md)
- [Présentation des PINNs par Raissi et ses coauteurs](https://maziarraissi.github.io/PINNs/)
- [Identification d'équations, article des auteurs](https://arxiv.org/abs/1711.10566)
- [Validation expérimentale d'un écoulement pulsatile avec Womersley](https://pubmed.ncbi.nlm.nih.gov/11672627/)

Ces références donnent le contexte scientifique ; nos résultats viennent des scripts
et fichiers locaux cités ci-dessus. Ne pas attribuer nos chiffres à ces publications.


## 13 septembre 2026 — Exploration interactive du tube pulsatile

Comparaison animée de la référence et du PINN inverse, avec lecture/pause,
curseur temporel, bruit (0, 3, 10 %), graines (42, 43, 44), vitesse ou erreur
de vitesse, et courbes synchronisées du forçage G, du débit Q* et du cisaillement τ*.
La plage 30–65 % du cycle ne contient aucune observation de vitesse.

Source exclusive : `results/inverse_pulsatile_20260913_165408_169589`, neuf cas
à 2 000 étapes chacun. Aucun nouvel entraînement. Les métriques conservées
proviennent des grilles originales ; l'affichage utilise 51 phases et 31 rayons,
avec interpolation linéaire. Les traceurs et la distance axiale sont illustratifs,
ils ne représentent pas des cellules sanguines ni une géométrie mesurée.

Sources reproductibles : `visualization/tube-flow.template.html`,
`visualization/flow_math.js` et `src/build_flow_visualization.py`.
Vue affichée dans la conversation :
`/Users/louisgosset/.codex/visualizations/2026/09/09/01a08577-65d6-7f51-9c1c-0aac0206e85e/tube-pulsatile-interactif.html`.

Vérifications : cinq contrôles mathématiques JS réussis, syntaxe JS valide,
conformité des neuf profils et des 27 courbes aux sorties NPZ à l'arrondi près,
fragment inférieur à 1 Mo et revue statique indépendante sans anomalie.
Pas de vérification visuelle dans un navigateur.


## 13 septembre 2026 — Correction de l’ouverture locale et présentation

La capture de Louis montrait l’ouverture du modèle non assemblé dans Safari :
emplacements de données non remplacés, styles de la conversation absents et
absence de déclaration UTF-8. La version utilisable est désormais
`visualization/index.html`, page autonome avec données et D3 intégrés.
L'ancienne adresse `visualization/tube-flow.template.html` redirige vers elle ;
le modèle éditable est `visualization/flow-fragment.template.html`.

Présentation : typographie système, commandes regroupées, palette turquoise et
orange, vues du tube sur fond clair, courbes empilées sur écran étroit.
Les résultats scientifiques et les neuf scénarios restent les mêmes.

Validation : trois tests d’export réussis (UTF-8/styles, données intégrées sans
script distant, ancienne adresse), cinq contrôles mathématiques JS réussis,
syntaxe de tous les scripts exportés vérifiée. Le navigateur automatisé refuse
les adresses file:// : le rendu dans Safari et les clics n'ont donc pas été
validés visuellement pendant cette correction.


## 13 September 2026 — Reconstruction reliability study (English deliverables)

Approved study: counts 12/24/48/96 × noise SD 0/3/10/20% of 1.5 × seeds
42/43/44. Four corner cases at seed 42 form the pilot and are retained in the
full study. Reference targets are never used in the PDE loss. Both neural
methods share observations, initial weights, architecture and boundary/periodic
constraints; the data-only NN removes the PDE loss. Both receive 1,000 Adam
steps and at most 100 L-BFGS iterations. Analytical least squares is the strong
correct-family baseline. The observation-free interval remains [0.30, 0.65].

Sources: `src/reliability_study.py`, `src/run_reliability_parallel.py`,
`src/reliability_report.py`, `visualization/reliability.template.html`.
Resumable case artifacts and protocol/source hashes: `results/reliability_v1/`.
English explorer: `visualization/reliability.html`; English reading guide:
`RELIABILITY_GUIDE.md`. Figures and measured conclusions are rebuilt from saved
cases, with explicit incomplete-run labels.

Validation during implementation: 21 Python tests pass, exported pilot scripts
parse, no remote script dependency, static figure visually inspected. A
read-only independent review checked pairing, truth leakage, metric definitions
and resume behaviour. Its resume-version and custom-link findings were fixed.
The previous file:// browser-policy denial still prevents automated page
interaction validation; no workaround was attempted.

Pilot observation (one seed only): PINN errors increase with noise; for that
seed at 20% noise, additional radial observations reduce its hidden-velocity
error. The analytical method is more accurate in all four pilot corners.
Full-study conclusions must use the completed three-seed aggregates, not these
pilot findings alone. Timing includes sequential pilot and concurrent later
cases, so reported durations are not controlled speed comparisons.


## 14 September 2026 — Study complete and template-opening fix

All 48 paired cases are complete, giving three seeds in each of 16 cells.
All three metrics favour the PINN over the data-only network in mean error
in 16/16 cells. Analytical least squares has lower mean error than the PINN
in 16/16 cells for all three metrics. These comparisons concern the fixed
synthetic benchmark and optimization budget only. Full means and sample SDs
are in `results/reliability_v1/REPORT.md`.

The English offline page, PNG and PDF now include all 48 runs. The final
static figure was visually inspected; all 144 exported velocity fields were
checked against the saved full-grid predictions at display precision.
JavaScript syntax and complete experiment/seed coverage pass verification.

Louis opened `visualization/reliability.template.html`, which still contained
unassembled placeholders. That old address now redirects to `reliability.html`.
The editable source moved to `visualization/templates/reliability.html.in`
to keep it separate from pages intended to open in a browser. The report
generator uses this new source. Browser interaction testing remains unavailable
because the previous automatic browser policy denied local file URLs.


## 14 September 2026 — English project note for a controls professor

Created a three-page email attachment, `output/pdf/Louis_Gosset_Project_Note_Control_and_PINNs.pdf`,
focusing on offline field reconstruction and unknown-input estimation. It
presents the 48-case benchmark, a common-scale velocity map and a numerical
mean/SD comparison, then connects future work to state-space estimation and
measurement identifiability. It explicitly separates completed work from
proposed observers/control and discloses AI assistance.

MECH3610 course context verified against the official HKUST 2026-27 Control
Principles listing. PINN methodological reference: Raissi et al. (2019),
doi:10.1016/j.jcp.2018.10.045. The results remain local project results.
All three final PDF pages were rendered and visually reviewed. Nothing was
sent to the professor.
