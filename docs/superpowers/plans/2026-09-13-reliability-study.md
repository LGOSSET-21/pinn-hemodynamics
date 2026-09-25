# Reliability study implementation plan

**Goal:** an English, reproducible comparison of reconstruction error across observation counts and noise levels.
**Architecture:** a resumable experiment runner reuses the validated inverse PINN and analytical physics; a separate exporter creates an offline interactive report and publication-sized figures from saved results.
**Tech stack:** Python, NumPy, PyTorch, Matplotlib, standalone HTML/JavaScript.
**Spec:** the user's approved 4 × 4 study with three seeds and a four-corner pilot, as detailed below.

## Fixed protocol

- Observation counts: 12, 24, 48, 96; Gaussian noise standard deviation: 0, 3, 10, 20% of the fixed velocity scale 1.5.
- Seeds 42, 43, 44 vary observation noise and neural initialization together. Locations are deterministic and nested across counts; six observed phases are fixed. No observation in [0.30, 0.65].
- Three methods receive identical observations: inverse PINN, data-only network, analytical least squares. Both networks share architecture, exact symmetry/no-slip/periodicity, initialization and an Adam budget of 1000 steps followed by at most 100 L-BFGS iterations. The data-only network has no differential-equation loss. Equal iteration counts do not mean equal runtime.
- Reference solutions are used to simulate observations and evaluate predictions, never as hidden training targets. The analytical baseline knows the correct solution family; this is a favourable synthetic benchmark, not a general superiority test.
- Metrics: hidden-phase velocity RMSE / 1.5 × 100; full-cycle flow and signed wall-shear relative L2 errors × 100. Show mean and sample SD across seeds; no binary success threshold.
- Pilot: the four corners with seed 42; retain these runs in the full 48-case study. Checkpoints may be reused only when the complete protocol matches.
- Save observations, predictions, neural weights, times, and metrics. All new presentation text is English. Missing/failed runs must remain explicitly incomplete.

## Implementation and validation

- [x] Add deterministic nested sampling and metric tests (`tests/test_reliability.py`); verify noise-free analytical recovery and hidden-phase exclusion.
- [x] Implement `src/reliability_study.py`, including data-only training, case checkpoints, protocol checks, four-corner pilot and full continuation.
- [x] Run pilot; inspect numerical validity, observation counts and measured runtime before continuing all 48 cases.
- [x] Implement `src/reliability_report.py` and `visualization/templates/reliability.html.in`: nine heatmaps, cell selection, seed selection, synchronized profiles/flow/shear and illustrative tube playback. Export a static PNG/PDF figure and English methods/results report.
- [x] Verify aggregation, all completed cases, exported script syntax and existing physics tests. Preserve the measured conclusions and limitations in the research log.

No repository commits: this workspace has no Git checkout. Browser file:// access was denied earlier; do not bypass that denial. Report visual validation limits honestly.
