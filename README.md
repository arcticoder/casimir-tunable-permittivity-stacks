# Casimir Tunable Permittivity Stacks — Research Prototype

## Related Repositories

- `energy`: Central hub for related research and reproducibility artifacts.
- `lqg-ftl-metric-engineering`: Projects that use tunable-permittivity models for exploratory, model-level investigations.
- `casimir-ultra-smooth-fabrication-platform`: Fabrication and quality-control artifacts relevant to prototype devices.

This repository contains research-stage scripts, derivations, and prototype software exploring tunable permittivity stacks and their effect in specific model scenarios. Results coming from the repository are conditional on the models, numerical choices, and experimental/test configurations used; they are not production guarantees.

## Overview

The project investigates whether controlled permittivity stacks can alter model-level exotic-energy estimates in targeted simulations. The repository provides example workflows, UQ mitigation strategies, and proof-of-concept validation artifacts. Independent reproduction and expanded validations are recommended before making operational claims.

## Objectives & Caveats (summary)

- **Objective (research):** Explore model sensitivities of permittivity stacks in electromagnetic and spacetime-coupled simulations.
- **Model-Level Results:** Some simulation configurations produce field outputs that are FTL-compatible at the model level; these are mathematical/model outputs that require further physical, experimental, and safety validation before being interpreted as evidence of feasibility.
- **Caveats:** Reported tuning accuracy, ranges, and response-times are specific to controlled experimental setups or simulation parameters and should be reproduced with the provided artifacts.

## Representative (example) Specifications

Numeric summaries below are example results from selected tests and simulations; consult `docs/` for the exact inputs and scripts used.

- **Tuning Range (example):** ε_r ≈ 1.5 to 15.0 (dependent on material stack and configuration)
- **Tuning Accuracy (example):** <1% in controlled demonstrations (reproducibility dependent on setup)
- **Response Time (example):** ~85–100 ms in selected actuation setups

## Advanced Mathematical & UQ Framework (summary)

The codebase includes advanced modeling patterns (tensor state estimation, coupled-field formulations) and UQ workflows (PCE + GP + Sobol with numerical safeguards). These are research tools; the repository documents mitigations for numerical instability and suggests practices for independent verification.

## Validation, UQ & Reproducibility Guidance

- **Reproducibility:** Re-run example scripts with the `data/` inputs and record environment metadata (Python version, package versions). Prefer deterministic symbolic checks where available.
- **UQ Checks:** Use the provided PCE/GP/Sobol scripts with seeds and the recommended regularization parameters from `docs/UQ_CRITICAL_RESOLUTION_REPORT.md`.
- **Numerical Stability:** For ill-conditioned problems, increase numeric precision and use provided SVD/Tikhonov fallbacks; run the `uq_critical_validation.py` suite before generalizing results.

## Example Usage & Quick Start

See `docs/technical-documentation.md` and the `src/digital_twin` demonstration scripts for runnable examples and the exact inputs used in tests.

## Scope, Validation & Limitations

- **Scope:** Research and prototype experiments; intended for researchers and maintainers.
- **Validation:** The repository contains targeted validation suites and mitigation examples; these are not exhaustive. CI tests cover a small set of algebraic checks and example inputs when configured.
- **Limitations:** Reported outcomes depend on fabrication fidelity, environmental control, numerical methods, and model assumptions. Do not extrapolate experimental or numeric results beyond the documented test conditions without additional validation.

## Contributing

When contributing numeric claims or performance numbers, include the exact input files, random seeds, environment metadata, and UQ artifacts necessary to reproduce the results.

---

This repository documents research-stage investigations and prototype artifacts. Numeric summaries and model-level findings are provisional and require reproducible artifacts, independent review, and engineering validation before being treated as deployment-ready specifications.
