# Reconstruction reliability study

Completed runs: **48 / 48**.

## Protocol

Observation counts: 12, 24, 48, 96. Gaussian noise standard deviation: 0%, 3%, 10%, 20% of the fixed velocity scale 1.5. Seeds: 42, 43, 44.
Six fixed observed phases: 0, 0.1, 0.2, 0.75, 0.85, 0.95. No velocity observations in [0.30, 0.65]. Increasing the count adds nested radial locations, not additional observed phases.
All methods receive exactly the same observations. Both networks use the same architecture, initialization, periodic features and exact centre symmetry/no-slip wall conditions. The data-only network omits the PDE loss.
Both networks: 1,000 Adam steps, then at most 100 L-BFGS iterations with strong-Wolfe line search. Equal iteration limits do not imply equal computational cost or equal convergence.
The analytical baseline uses the correct harmonic solution family. Geometry, frequency and alpha=3 are known. The PINN jointly learns the velocity and three forcing coefficients.

## Metrics

Velocity error: RMSE over the hidden phases divided by 1.5, expressed as a percentage. Flow and signed wall-shear errors: full-cycle relative L2, expressed as percentages. Evaluation grid: 101 phases × 121 radial positions. Flow is integrated radially; wall shear is evaluated by automatic differentiation for the networks.
Mean ± sample standard deviation across seeds; this is not a confidence interval. Seeds jointly change noise and neural initialization. No clinical acceptance threshold is implied.

## Results

| Observations | Noise | Repeats | Method | Velocity (%) | Flow (%) | Wall shear (%) |
|---:|---:|---:|---|---:|---:|---:|
| 12 | 0% | 3 | Inverse PINN | 0.1923 ± 0.0314 | 0.1419 ± 0.0450 | 0.1898 ± 0.1001 |
| 12 | 0% | 3 | Data-only NN (no PDE loss) | 3.0337 ± 2.2999 | 2.2460 ± 1.6846 | 6.0151 ± 1.2348 |
| 12 | 0% | 3 | Analytical least squares | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| 12 | 3% | 3 | Inverse PINN | 2.0887 ± 2.1565 | 2.2674 ± 1.5441 | 2.9215 ± 1.6970 |
| 12 | 3% | 3 | Data-only NN (no PDE loss) | 22.3918 ± 14.0965 | 15.4203 ± 6.5889 | 31.1974 ± 9.7080 |
| 12 | 3% | 3 | Analytical least squares | 0.9542 ± 0.4979 | 1.0772 ± 0.1881 | 1.1548 ± 0.1974 |
| 12 | 10% | 3 | Inverse PINN | 6.1573 ± 7.2805 | 7.0529 ± 5.3182 | 9.6364 ± 5.6954 |
| 12 | 10% | 3 | Data-only NN (no PDE loss) | 92.2928 ± 71.4047 | 83.4382 ± 63.2070 | 110.0258 ± 35.1492 |
| 12 | 10% | 3 | Analytical least squares | 3.1808 ± 1.6595 | 3.5906 ± 0.6271 | 3.8494 ± 0.6580 |
| 12 | 20% | 3 | Inverse PINN | 14.3296 ± 13.2690 | 15.1729 ± 9.4893 | 19.6802 ± 9.9716 |
| 12 | 20% | 3 | Data-only NN (no PDE loss) | 65.4519 ± 43.9377 | 94.2879 ± 44.2811 | 246.2482 ± 233.6631 |
| 12 | 20% | 3 | Analytical least squares | 6.3616 ± 3.3190 | 7.1813 ± 1.2541 | 7.6988 ± 1.3160 |
| 24 | 0% | 3 | Inverse PINN | 0.1048 ± 0.0151 | 0.0908 ± 0.0212 | 0.1423 ± 0.0217 |
| 24 | 0% | 3 | Data-only NN (no PDE loss) | 3.3921 ± 3.2781 | 2.3266 ± 2.5431 | 4.1480 ± 0.0915 |
| 24 | 0% | 3 | Analytical least squares | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| 24 | 3% | 3 | Inverse PINN | 1.4997 ± 0.5174 | 1.5319 ± 0.3487 | 1.9159 ± 0.5306 |
| 24 | 3% | 3 | Data-only NN (no PDE loss) | 14.7572 ± 7.0269 | 14.1817 ± 5.7867 | 23.3865 ± 10.2717 |
| 24 | 3% | 3 | Analytical least squares | 0.4449 ± 0.3417 | 0.6278 ± 0.2881 | 0.6694 ± 0.2953 |
| 24 | 10% | 3 | Inverse PINN | 5.1059 ± 2.4041 | 5.1137 ± 1.6471 | 6.2038 ± 2.2434 |
| 24 | 10% | 3 | Data-only NN (no PDE loss) | 72.9198 ± 74.1531 | 62.4440 ± 58.7849 | 98.8372 ± 44.1901 |
| 24 | 10% | 3 | Analytical least squares | 1.4830 ± 1.1389 | 2.0926 ± 0.9605 | 2.2312 ± 0.9842 |
| 24 | 20% | 3 | Inverse PINN | 10.6389 ± 5.4612 | 10.4264 ± 3.8050 | 12.5003 ± 5.1399 |
| 24 | 20% | 3 | Data-only NN (no PDE loss) | 193.4561 ± 94.0991 | 166.2905 ± 96.2211 | 182.9297 ± 90.9151 |
| 24 | 20% | 3 | Analytical least squares | 2.9659 ± 2.2777 | 4.1851 ± 1.9210 | 4.4625 ± 1.9684 |
| 48 | 0% | 3 | Inverse PINN | 0.1204 ± 0.0264 | 0.1097 ± 0.0235 | 0.1589 ± 0.0247 |
| 48 | 0% | 3 | Data-only NN (no PDE loss) | 3.4847 ± 2.2987 | 2.2533 ± 1.7415 | 3.7491 ± 0.8567 |
| 48 | 0% | 3 | Analytical least squares | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| 48 | 3% | 3 | Inverse PINN | 1.2051 ± 1.1899 | 1.2260 ± 0.9860 | 1.3796 ± 1.2717 |
| 48 | 3% | 3 | Data-only NN (no PDE loss) | 11.1812 ± 5.9919 | 9.6167 ± 5.0119 | 14.4040 ± 3.2090 |
| 48 | 3% | 3 | Analytical least squares | 0.9755 ± 0.7179 | 1.0685 ± 0.4313 | 1.1568 ± 0.4810 |
| 48 | 10% | 3 | Inverse PINN | 4.1097 ± 4.5513 | 4.2662 ± 3.6638 | 4.7801 ± 4.6349 |
| 48 | 10% | 3 | Data-only NN (no PDE loss) | 31.4167 ± 2.4896 | 22.9448 ± 2.2239 | 28.1372 ± 6.1637 |
| 48 | 10% | 3 | Analytical least squares | 3.2518 ± 2.3931 | 3.5617 ± 1.4375 | 3.8559 ± 1.6035 |
| 48 | 20% | 3 | Inverse PINN | 8.8167 ± 10.0212 | 9.0659 ± 8.1673 | 10.1545 ± 10.3046 |
| 48 | 20% | 3 | Data-only NN (no PDE loss) | 209.4130 ± 230.5248 | 165.8982 ± 187.6803 | 145.6193 ± 131.6577 |
| 48 | 20% | 3 | Analytical least squares | 6.5036 ± 4.7862 | 7.1233 ± 2.8750 | 7.7118 ± 3.2070 |
| 96 | 0% | 3 | Inverse PINN | 0.1484 ± 0.0470 | 0.1170 ± 0.0759 | 0.1729 ± 0.0790 |
| 96 | 0% | 3 | Data-only NN (no PDE loss) | 3.1727 ± 2.4211 | 2.6845 ± 1.2116 | 5.0425 ± 1.2122 |
| 96 | 0% | 3 | Analytical least squares | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| 96 | 3% | 3 | Inverse PINN | 1.4561 ± 0.9947 | 1.3611 ± 0.8666 | 1.5596 ± 1.0242 |
| 96 | 3% | 3 | Data-only NN (no PDE loss) | 5.6495 ± 1.3017 | 4.5820 ± 0.6046 | 7.8386 ± 3.6137 |
| 96 | 3% | 3 | Analytical least squares | 0.8418 ± 0.6855 | 0.8276 ± 0.5996 | 0.9053 ± 0.6471 |
| 96 | 10% | 3 | Inverse PINN | 5.4089 ± 3.8527 | 4.8277 ± 3.1988 | 5.3600 ± 3.6815 |
| 96 | 10% | 3 | Data-only NN (no PDE loss) | 36.0906 ± 36.2688 | 32.4617 ± 26.6514 | 44.1621 ± 19.0193 |
| 96 | 10% | 3 | Analytical least squares | 2.8060 ± 2.2850 | 2.7588 ± 1.9988 | 3.0178 ± 2.1570 |
| 96 | 20% | 3 | Inverse PINN | 10.1888 ± 7.6447 | 9.3713 ± 6.5709 | 10.5537 ± 7.8148 |
| 96 | 20% | 3 | Data-only NN (no PDE loss) | 120.9247 ± 130.0515 | 105.4407 ± 113.0767 | 117.4864 ± 82.1741 |
| 96 | 20% | 3 | Analytical least squares | 5.6120 ± 4.5699 | 5.5176 ± 3.9975 | 6.0356 ± 4.3139 |

## Measured comparisons

- Hidden-phase velocity error: the PINN has lower mean error than the data-only network in 16/16 completed cells; analytical least squares has lower mean error than the PINN in 16/16 cells.
- Flow error: the PINN has lower mean error than the data-only network in 16/16 completed cells; analytical least squares has lower mean error than the PINN in 16/16 cells.
- Wall-shear error: the PINN has lower mean error than the data-only network in 16/16 completed cells; analytical least squares has lower mean error than the PINN in 16/16 cells.
- Inverse PINN: median fitting time 40.897 s (range 16.533–87.893 s). CPU, one Torch thread per process. Evaluation/export time excluded.
- Data-only NN (no PDE loss): median fitting time 8.208 s (range 3.701–20.854 s). CPU, one Torch thread per process. Evaluation/export time excluded.
- Analytical least squares: median fitting time 0.003 s (range 0.001–0.025 s). CPU, one Torch thread per process. Evaluation/export time excluded.
The pilot ran sequentially; remaining cases used up to three concurrent processes. Recorded times include uncontrolled CPU contention and are not a controlled method-speed benchmark.

## Limits and interpretation

These maps characterize one synthetic geometry, forcing family, sampling design and fixed optimization budget. They do not establish universal PINN superiority or validate a medical application. More observations do not have to improve every individual noisy run.
Three repetitions provide only a preliminary view of variability. Optimization error, noise effects and model assumptions can all affect the result. A failure at this budget is not proof that a method cannot solve the problem.
The interactive page downsamples fields for display and linearly interpolates between saved samples. Metrics use the original evaluation grid. Particle motion and axial distances are illustrative, not tracked blood cells.

## Reproduction

`python src/reliability_study.py --pilot` runs the four-corner pilot. `python src/reliability_study.py` resumes all 48 paired cases. Protocol and source hashes prevent mixing incompatible checkpoints.
`python src/reliability_report.py` rebuilds the English page and figures. Each case stores its observations and fields in predictions.npz, network weights in models.pt and metrics/timing in metrics.json.
