"""Exclusions from the original authors' scripts. The raw CSVs are not changed."""
from pathlib import Path
import pandas as pd
DATA = Path(__file__).resolve().parents[1] / 'data'
def study_data(study):
    d = pd.read_csv(DATA / f'Study{study}_Data.csv')
    if study == 2:  # as in their Study 2 Analyses.R (repeat participant)
        d = d.loc[d['chain'] != 119].copy()
    if study == 3:  # age typo, as in their Study 3 Demographics Table.R
        d.loc[pd.to_numeric(d['Age.Screener'], errors='coerce') == 300, 'Age.Screener'] = 30
    return d.reset_index(drop=True)
