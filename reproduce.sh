#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/src"
py() { python3 -I "$@"; }
py targets.py
py subset_targets.py
py day_effects.py
py ai_sensitivity.py
if [ "${FAST:-0}" != "1" ]; then
  for study in 3adj 3 2 3clean; do
    for variant in balanced positive base register; do
      py fit.py "$study" "$variant" lag
    done
  done
  for variant in balanced register; do py fit.py 3adj_gens5 "$variant" lag; done
  py fit.py 3adj register_trend lag
  py fit.py 3 balanced nolag
  py profile.py
  py recovery.py --replicates 10 --workers "${RECOVERY_WORKERS:-3}"
fi
py simulation_check.py
py param_se.py
py conditions.py
py text_examples.py
py network_experiments.py
py drift.py
py drift_size.py
py drift_rules.py
py robust.py
py figure_data.py
py network_snapshot.py
py calibration_sensitivity.py
py asymmetry.py
py asymmetry_b.py
py netci.py
py sb_sensitivity.py
if [ "${FAST:-0}" != "1" ]; then
  py reference_sensitivity.py --restart --workers "${SENSITIVITY_WORKERS:-2}"
else
  py reference_sensitivity.py --reuse-fits --workers "${SENSITIVITY_WORKERS:-2}"
fi
py repeated_sensitivity.py --restart --workers "${SENSITIVITY_WORKERS:-2}"
cd ../paper
py make_figs.py
py make_supp.py
