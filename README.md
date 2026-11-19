# adhd-category-model

Code for "Partial copying and the broadening of diagnostic categories through peer transmission" (Kyuri Park).

Agents hold a category boundary and move toward a source's boundary when they disagree about an example. I fit this to the ADHD transmission-chain data of Katiyar et al. and then run the same rule on networks.

## Data

The data are not included. Download `Study1_Data.csv` to `Study4_Data.csv` from the authors' Zenodo record (doi:10.5281/zenodo.23073308) and put them in `data/`.

## Running

Python 3.12; exact versions are in `requirements-lock.txt`.

```sh
pip install -r requirements-lock.txt
./reproduce.sh            # everything, including all fits (slow)
FAST=1 ./reproduce.sh     # reuse saved fits, profile and recovery; rerun predictions
python -m unittest discover -s tests
```

Scripts run from `src/` and write to `results/`. `paper/make_figs.py`, `paper/make_supp.py` and `paper/sensitivity_material.py` turn the results into `figures.tex` and `supp.tex`; compile `paper/main.tex` with `latexmk -pdf`.

The full run repeats the main fits, the learning-rate profile, 40 recovery fits and the four additional reference fits. The fast run keeps those fitted results and reruns the data preparation, uncertainty calculations, predictions and figures. Set `RECOVERY_WORKERS` or `SENSITIVITY_WORKERS` to control parallel processes. The sensitivity scripts resume saved checkpoints when run separately without `--restart`.

## What's where

- `chain_model.py`, `fit.py`: the chain model and the simulated-moments fit (differential evolution, 60 iterations, 3,000 chains per evaluation).
- `preprocess.py`, `load.py`, `targets.py`: data cleaning and the 31 fit targets. I follow the original authors' exclusions (Study 2 chain 119) and their age fix in Study 3. Standard errors come from 2,000 bootstrap resamples of whole chains.
- `param_se.py`, `simulation_check.py`, `recovery.py`, `profile.py`: standard errors, checks against simulation noise, model recovery, learning-rate profile.
- `conditions.py`: the same-day comparisons (Study 2 instruction, Study 4).
- `drift.py`, `drift_size.py`, `drift_rules.py`: direction and size of change after one round of exchange.
- `repeated_learning.py`, `repeated_sensitivity.py`: 600-round checks of copying and two label-only rules, including source comparisons and changes in update strength. All 97 settings are retained.
- `reference_sensitivity.py`: refits at diagnostic references of 2.5%, 5% and 10%, and the expert-placement comparison at each reference.
- `network_model.py`, `network_experiments.py`, `robust.py`, `netci.py`, `asymmetry_b.py`, `calibration_sensitivity.py`, `sb_sensitivity.py`, `figure_data.py`, `network_snapshot.py`: network simulations.
- `day_effects.py`, `ai_sensitivity.py`: demographic trends and the AI-use exclusion check.
- `text_coding.py`, `text_step2.py`, `threshold.py`, `asymmetry.py`: exploratory analyses, not inputs to the manuscript figures.
- `results-original/` and `fit_start_values.json`: earlier fits, used as starting points for the optimiser.

## Caveats

The network runs use rounded parameters from an earlier fit (`calibration_sensitivity.py` shows the main contrast barely changes with the final ones). The chain data don't identify reversion or the share of positive examples, so those are scenario settings. The diagnostic reference is an assumption, not a prevalence estimate. Refitting it at 2.5%, 5% and 10% gives similar initial breadth but different learning rates.

Broadening with many positive examples survives the tested label-only rules. A general advantage for broad sources does not. The fixed-step learner can remove or reverse that advantage. The chain data also do not identify copying as the cause of the observed changes. A changing shared source fits at least as well. All these qualifications are included in the manuscript.

The new network summaries average rounds 481 to 600. They do not assert equilibrium. Simulation intervals describe variation between populations at fixed settings. Fitted parameter intervals use a local approximation and exclude model-selection and simulation uncertainty.

## License

GPL-3.0, see `LICENSE`.
