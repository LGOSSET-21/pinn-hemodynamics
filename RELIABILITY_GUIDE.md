# Reconstruction Reliability: A Guide to the Experiment

Open **[the offline explorer](visualization/reliability.html)**. It uses saved results and does not train a model in your browser.

## Research question

How do measurement count and observation noise affect reconstruction of an unobserved part of a periodic flow cycle? Does adding the governing equation improve on a neural network trained only on measurements?

## Reading the maps

Each row of panels is a method; each column is an error metric. Inside a panel, columns vary the observation count and rows vary the noise. A cell reports the mean error and sample standard deviation over three seeds. Incomplete cells explicitly show their available repetition count.

The colour scale is shared between methods for each metric. A lighter cell means lower error. Colours do not label an experiment as clinically acceptable or unacceptable. The static figure shows means; the interactive map and report also show variability.

- **Hidden-phase velocity error:** RMSE in the unobserved interval, divided by the fixed scale 1.5, multiplied by 100. This is not pointwise percentage error and remains meaningful at the zero-velocity wall.
- **Flow error:** full-cycle relative L2 error of the radially integrated velocity.
- **Wall-shear error:** full-cycle relative L2 error of the negative radial velocity derivative at the wall.

Select a cell, choose a seed, and use the time slider. Between 30% and 65% of the cycle, there were no measured velocities. The profile plot shows what the method reconstructs there. The animated tracers illustrate the saved velocity field; they do not represent individual blood cells.

## What makes the comparison fair — and what remains unequal

The three methods receive exactly the same observations. Samples are nested across measurement counts. All cases use the same six observed phases, so more measurements improve radial sampling without filling in the hidden temporal interval.

Both neural networks start with the same weights for a given seed. They share the architecture, exact centre symmetry and wall condition, and periodic time representation. The data-only network does not use a PDE residual. Its name does not mean that all physical structure has been removed.

Both networks receive 1,000 Adam steps and up to 100 L-BFGS iterations. The inverse PINN additionally learns three forcing coefficients and computes a differential-equation residual. Equal iteration limits do not imply equal runtime, equal convergence or the best possible tuning of each method.

Analytical least squares knows the exact harmonic solution family for this simple geometry. It is a deliberately strong, inexpensive baseline. If it wins, the appropriate conclusion is that a neural network is unnecessary for this particular inverse problem under these assumptions.

## How to discuss the outcome

First compare methods at a fixed observation count and noise level. Then examine trends across noise and count, checking standard deviations and individual seeds before making claims. A worse case may reflect noise, optimization error or assumptions; this study does not separate all three causes.

An improvement in velocity alone is insufficient evidence for accurate wall shear, which depends on a derivative. Likewise, one good seed is insufficient evidence for robustness.

The final measured comparisons and fitting times are generated in **[REPORT.md](results/reliability_v1/REPORT.md)**. Timing includes CPU contention because the pilot was sequential and later cases ran concurrently; it is not a controlled speed benchmark.

## Reproduce or resume

From the project directory with the Python environment activated:

```bash
python src/reliability_study.py --pilot
python src/reliability_study.py
python src/reliability_report.py
```

The full study resumes completed cases after checking the protocol, source hashes and dependency versions. Alternatively, after the pilot, `python src/run_reliability_parallel.py --workers 3` runs up to three cases concurrently. Do not start two runners on the same output folder at the same time.

Changing the training protocol requires a new output folder, passed with `--output`. Pass that folder to the report generator with `--source`. Preserve the original study when comparing a changed protocol.

## Scope

This is an educational synthetic benchmark with known geometry, frequency and Womersley parameter. It contains no patient data and provides no medical validation. Three seeds offer preliminary evidence, not a confidence interval or a universal guarantee. The implementation and analysis were developed with AI assistance; the scientific presentation should distinguish implemented experiments, measured findings and the author's own interpretation.

## Completed study

All 48 runs are complete. Across all 16 configurations, the PINN has lower
mean error than the data-only network for each of the three metrics. The
analytical fit has lower mean error than the PINN in all 16 configurations
for each metric. This demonstrates the benefit of the PDE constraint over
the chosen neural baseline, while also showing that the classical correct-family
fit is preferable for this simple synthetic problem at the tested budget.

Open `visualization/reliability.html`. The similarly named `.template.html`
address is a compatibility redirect; editable page source lives in
`visualization/templates/reliability.html.in`.
