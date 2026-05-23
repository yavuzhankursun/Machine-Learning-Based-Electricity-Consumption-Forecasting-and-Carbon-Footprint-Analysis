"""
ASAMA 2 — 7 MODEL KARSILASTIRMASI (Turkiye elektrik talebi, TWh)
=================================================================
Elektrik talebi hedefini 7 modelle walk-forward CV ile tahmin eder.

  Girdi (RF/XGB/LGBM/CatBoost/LSTM): electricity_demand lag_1, lag_2 + 14 feature
  Girdi (ARIMA/SARIMA):              electricity_demand serisi (univariate baseline)

  → Tum modeller AYNI test noktalari uzerinde (karsilastirilabilir R²).

CIKTI:
  pipeline/sonuclar/model_karsilastirma/asama2_model_karsilastirma.csv
  pipeline/sonuclar/model_karsilastirma/asama2_model_karsilastirma.json

Mevcut asama2_egit.py / forecast ciktilarina DOKUNMAZ.
"""

import os, sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from karsilastirma_ortak import (yedi_modeli_calistir, en_iyi_model,
                                 MODEL_SIRASI, TF_MEVCUT, SM_MEVCUT)

ULKE  = 'Turkey'
HEDEF = 'electricity_demand'
LAG_SAYISI     = 2
MIN_EGITIM_YIL = 15
N_TRIAL_TREE = 30
N_TRIAL_LSTM = 12
LSTM_EPOCHS  = 150

FEATURE_COLS = [
    'Population', 'GDP_Per_Capita', 'GDP_Growth',
    'renewables_share_elec', 'Urbanization_Rate',
    'Industry_Value_Added_Pct_GDP', 'Population_Density',
    'fossil_fuel_consumption', 'nuclear_consumption',
    'coal_co2', 'gas_co2', 'co2_per_capita',
    'Electricity_Loss_Pct', 'temperature_change_from_ghg',
]

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
ISLENMIS_DIR = os.path.join(PIPELINE_DIR, 'veri', 'islenmis')
CIKTI_DIR    = os.path.join(PIPELINE_DIR, 'sonuclar', 'model_karsilastirma')
os.makedirs(CIKTI_DIR, exist_ok=True)
CSV_YOL  = os.path.join(CIKTI_DIR, 'asama2_model_karsilastirma.csv')
JSON_YOL = os.path.join(CIKTI_DIR, 'asama2_model_karsilastirma.json')


def main():
    print("=" * 72)
    print("ASAMA 2 — 7 MODEL KARSILASTIRMASI (Turkiye elektrik talebi)")
    print(f"  Modeller: {', '.join(MODEL_SIRASI)}")
    print(f"  TensorFlow(LSTM)={TF_MEVCUT}  statsmodels(ARIMA/SARIMA)={SM_MEVCUT}")
    print("  Degerlendirme: Walk-Forward Expanding Window CV")
    print("=" * 72)

    tr = pd.concat([
        pd.read_csv(os.path.join(ISLENMIS_DIR, 'train.csv')),
        pd.read_csv(os.path.join(ISLENMIS_DIR, 'val.csv')),
        pd.read_csv(os.path.join(ISLENMIS_DIR, 'test.csv')),
    ])
    tr = tr[tr['country'] == ULKE].sort_values('year').reset_index(drop=True)
    print(f"  {ULKE}: {len(tr)} yil ({tr['year'].min()}-{tr['year'].max()})")

    kullanilan = [f for f in FEATURE_COLS
                  if f in tr.columns and tr[f].notna().sum() >= 5]
    atlanan = [f for f in FEATURE_COLS if f not in kullanilan]
    print(f"  Kullanilan feature ({len(kullanilan)}): {kullanilan}")
    if atlanan:
        print(f"  Atlanan (yetersiz): {atlanan}")

    tr[kullanilan] = tr[kullanilan].ffill().bfill()
    for k in range(1, LAG_SAYISI + 1):
        tr[f'{HEDEF}_lag{k}'] = tr[HEDEF].shift(k)
    lag_cols = [f'{HEDEF}_lag{k}' for k in range(1, LAG_SAYISI + 1)]
    input_cols = lag_cols + kullanilan
    tr = tr.dropna(subset=input_cols + [HEDEF]).reset_index(drop=True)

    X = tr[input_cols].values
    y = tr[HEDEF].values
    print(f"  Kullanilabilir: {len(y)} yil ({tr['year'].min()}-{tr['year'].max()})")
    print(f"  Girdi: {len(lag_cols)} lag + {len(kullanilan)} feature = {len(input_cols)}")
    print(f"  Walk-Forward CV: min_train={MIN_EGITIM_YIL}, "
          f"{len(y)-MIN_EGITIM_YIL} out-of-sample tahmin\n")

    t0 = time.time()
    s = yedi_modeli_calistir(X, y, MIN_EGITIM_YIL,
                             n_trials_tree=N_TRIAL_TREE,
                             n_trials_lstm=N_TRIAL_LSTM,
                             lstm_epochs=LSTM_EPOCHS, etiket=HEDEF)
    best = en_iyi_model(s)
    print(f"\n  EN IYI MODEL: {best} (R²={s[best]['r2']:.4f})" if best else "  tum NaN")
    print(f"  Sure: {(time.time()-t0)/60:.1f}dk")

    temiz = {m: {k: v for k, v in d.items() if k not in ('trues', 'preds')}
             for m, d in s.items()}
    out = {'hedef': HEDEF, 'birim': 'TWh', 'en_iyi': best,
           'input_cols': input_cols, 'n_obs': int(len(y)),
           'min_train': MIN_EGITIM_YIL, 'modeller': temiz}
    _kaydet(out)
    print(f"\n  CSV : {CSV_YOL}\n  JSON: {JSON_YOL}")


def _nan(o):
    if isinstance(o, float) and (o != o):
        return None
    raise TypeError


def _kaydet(out):
    tmp = JSON_YOL + '.tmp'
    json.dump(out, open(tmp, 'w'), ensure_ascii=False, indent=1, default=_nan)
    os.replace(tmp, JSON_YOL)
    rows = []
    for model in MODEL_SIRASI:
        md = out['modeller'].get(model, {})
        rows.append({'model': model, 'r2': md.get('r2'), 'mae': md.get('mae'),
                     'rmse': md.get('rmse'), 'mape': md.get('mape'),
                     'cv_n': md.get('cv_n', 0), 'en_iyi': model == out.get('en_iyi'),
                     'order': md.get('order', ''), 'seasonal_order': md.get('seasonal_order', '')})
    tmp = CSV_YOL + '.tmp'
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, CSV_YOL)


if __name__ == '__main__':
    main()
