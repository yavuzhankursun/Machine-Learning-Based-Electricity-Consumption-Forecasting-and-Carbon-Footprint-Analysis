"""
ASAMA 3 — 7 MODEL KARSILASTIRMASI (Turkiye karbon ayak izi, Mt CO2)
====================================================================
Toplam CO2 emisyonunu 7 modelle walk-forward CV ile tahmin eder.

  Girdi (RF/XGB/LGBM/CatBoost/LSTM): co2 lag_1, lag_2 + leakage'siz feature'lar
  Girdi (ARIMA/SARIMA):              co2 serisi (univariate baseline)

  → Tum modeller AYNI test noktalari uzerinde (karsilastirilabilir R²).
  → LEAKAGE GUARD: co2 turevi feature'lar (coal_co2, gas_co2, co2_per_capita...)
    YASAK_LEAKAGE ile engellenir.

CIKTI:
  pipeline/sonuclar/model_karsilastirma/asama3_model_karsilastirma.csv
  pipeline/sonuclar/model_karsilastirma/asama3_model_karsilastirma.json

Mevcut asama3_egit.py / forecast ciktilarina DOKUNMAZ.
"""

import os, sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from karsilastirma_ortak import (yedi_modeli_calistir, en_iyi_model,
                                 MODEL_SIRASI, TF_MEVCUT, SM_MEVCUT)

ULKE  = 'Turkey'
HEDEF = 'co2'
LAG_SAYISI     = 2
MIN_EGITIM_YIL = 10
N_TRIAL_TREE = 30
N_TRIAL_LSTM = 12
LSTM_EPOCHS  = 150

YASAK_LEAKAGE = {
    'co2', 'co2_per_capita', 'co2_per_gdp',
    'coal_co2', 'oil_co2', 'gas_co2', 'cement_co2', 'flaring_co2',
    'consumption_co2', 'trade_co2', 'cumulative_co2',
    'total_ghg', 'ghg_per_capita', 'methane', 'nitrous_oxide',
    'temperature_change_from_co2', 'share_global_co2',
    'co2_growth_abs', 'co2_growth_prct',
}

FEATURE_COLS = [
    'electricity_demand',
    'Population', 'Urbanization_Rate', 'Population_Density',
    'GDP_Per_Capita', 'Industry_Value_Added_Pct_GDP',
    'fossil_fuel_consumption', 'nuclear_consumption',
    'primary_energy_consumption', 'energy_per_capita', 'energy_per_gdp',
    'renewables_share_elec', 'nuclear_share_elec', 'fossil_share_elec',
    'temperature_change_from_ghg',
]

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
ISLENMIS_DIR = os.path.join(PIPELINE_DIR, 'veri', 'islenmis')
HAM_DIR      = os.path.join(PIPELINE_DIR, 'veri', 'ham')
CIKTI_DIR    = os.path.join(PIPELINE_DIR, 'sonuclar', 'model_karsilastirma')
os.makedirs(CIKTI_DIR, exist_ok=True)
CSV_YOL  = os.path.join(CIKTI_DIR, 'asama3_model_karsilastirma.csv')
JSON_YOL = os.path.join(CIKTI_DIR, 'asama3_model_karsilastirma.json')


def main():
    print("=" * 72)
    print("ASAMA 3 — 7 MODEL KARSILASTIRMASI (Turkiye karbon ayak izi)")
    print(f"  Modeller: {', '.join(MODEL_SIRASI)}")
    print(f"  TensorFlow(LSTM)={TF_MEVCUT}  statsmodels(ARIMA/SARIMA)={SM_MEVCUT}")
    print("  Degerlendirme: Walk-Forward Expanding Window CV")
    print("=" * 72)

    # LEAKAGE assertion
    bulunan = [f for f in FEATURE_COLS if f in YASAK_LEAKAGE]
    if bulunan:
        raise ValueError(f"LEAKAGE: {bulunan} co2 turevi, FEATURE_COLS'tan cikar")
    print(f"  ✓ LEAKAGE KONTROL: {len(FEATURE_COLS)} feature temiz")

    # CO2 (owid)
    owid = pd.read_csv(os.path.join(HAM_DIR, 'owid_co2_emissions.csv'), comment='#')
    owid = owid.rename(columns={'Country': 'country', 'Year': 'year'})
    co2 = owid[owid['country'] == ULKE][['year', 'co2']].dropna().sort_values('year')

    # Feature'lar
    tr = pd.concat([
        pd.read_csv(os.path.join(ISLENMIS_DIR, 'train.csv')),
        pd.read_csv(os.path.join(ISLENMIS_DIR, 'val.csv')),
        pd.read_csv(os.path.join(ISLENMIS_DIR, 'test.csv')),
    ])
    tr = tr[tr['country'] == ULKE].sort_values('year').reset_index(drop=True)

    df = co2.merge(tr[['year'] + [c for c in FEATURE_COLS if c in tr.columns]],
                   on='year', how='inner').sort_values('year').reset_index(drop=True)
    print(f"  {ULKE} CO2: {len(df)} yil ({df['year'].min()}-{df['year'].max()}) | "
          f"araliq {df['co2'].min():.1f}-{df['co2'].max():.1f} Mt")

    kullanilan = [f for f in FEATURE_COLS
                  if f in df.columns and df[f].notna().sum() >= 5]
    df[kullanilan] = df[kullanilan].ffill().bfill()
    for k in range(1, LAG_SAYISI + 1):
        df[f'{HEDEF}_lag{k}'] = df[HEDEF].shift(k)
    lag_cols = [f'{HEDEF}_lag{k}' for k in range(1, LAG_SAYISI + 1)]
    input_cols = lag_cols + kullanilan
    df = df.dropna(subset=input_cols + [HEDEF]).reset_index(drop=True)

    X = df[input_cols].values
    y = df[HEDEF].values
    print(f"  Kullanilabilir: {len(y)} yil | girdi: {len(lag_cols)} lag + "
          f"{len(kullanilan)} feature = {len(input_cols)}")
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
    out = {'hedef': HEDEF, 'birim': 'Mt CO2', 'en_iyi': best,
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
