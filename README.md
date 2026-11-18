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
FAST=1 ./reproduce.sh     # reuse the saved fits, redo the rest
python -m unittest discover -s tests
```

Scripts run from `src/` and write to `results/`. `paper/make_figs.py` and `paper/make_supp.py` turn the results into `figs.tex`, `figs2.tex` and `supp.tex`; compile `paper/main.tex` with `latexmk -pdf`.

## What's where

- `chain_model.py`, `fit.py`: the chain model and the simulated-moments fit (differential evolution, 60 iterations, 3,000 chains per evaluation).
- `preprocess.py`, `load.py`, `targets.py`: data cleaning and the 31 fit targets. I follow the original authors' exclusions (Study 2 chain 119) and their age fix in Study 3. Standard errors come from 2,000 bootstrap resamples of whole chains.
- `param_se.py`, `simulation_check.py`, `recovery.py`, `profile.py`: standard errors, checks against simulation noise, model recovery, learning-rate profile.
- `conditions.py`: the same-day comparisons (Study 2 instruction, Study 4).
- `drift.py`, `drift_size.py`, `drift_rules.py`: direction and size of the drift under repeated exchange, including the two label-only learners.
- `network_model.py`, `network_experiments.py`, `robust.py`, `netci.py`, `asymmetry_b.py`, `calibration_sensitivity.py`, `sb_sensitivity.py`, `figure_data.py`, `network_snapshot.py`: network simulations.
- `day_effects.py`, `ai_sensitivity.py`, `text_coding.py`, `text_step2.py`, `threshold.py`, `asymmetry.py`: side analyses, not needed for the paper.
- `results-original/` and `fit_start_values.json`: earlier fits, used as starting points for the optimiser.

## Caveats

The network runs use rounded parameters from an earlier fit (`calibration_sensitivity.py` shows the main contrast barely changes with the final ones). The chain data don't identify reversion or the share of positive examples, so those are scenario settings. The 5% diagnostic boundary is an assumption, not a prevalence estimate.

## License

GPL-3.0, see `LICENSE`.
