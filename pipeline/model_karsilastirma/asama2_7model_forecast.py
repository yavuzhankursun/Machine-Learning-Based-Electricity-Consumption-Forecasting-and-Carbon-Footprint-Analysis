"""
AŞAMA 2 — 7 MODEL KARSILASTIRMASI + FORECAST (Turkiye elektrik tuketimi, TWh)
=============================================================================
Turkiye'nin 2026-2030 elektrik tuketimini 7 modelle tahmin eder. EN IYI R²'li
model nihai forecast'i uretir (→ Aşama 3'e girdi olur).

  7 MODEL : RF · XGBoost · LightGBM · CatBoost · LSTM · ARIMA · SARIMA
  GIRDI (tree/LSTM): electricity_demand lag_1, lag_2 + 18 korele parametre
       - korele parametrelerin GECMIS gercek halleri (CV icin, train/val/test)
       - korele parametrelerin GELECEK tahminleri (forecast icin, AŞAMA 1 ciktisi)
  GIRDI (ARIMA/SARIMA): electricity_demand serisi (univariate baseline)
  CV       : Walk-Forward Expanding Window (tum modeller AYNI test noktalari)
  FORECAST : En iyi model; tree/LSTM iteratif + differencing (trend ekstrapolasyon)

GIRDI DOSYALARI:
  veri/islenmis/train.csv, val.csv, test.csv                          (gecmis gercek)
  asama1_feature_tahmini/ciktilar/feature_tahminleri_2021_2030.csv    (AŞAMA 1 forecast)

CIKTI DOSYALARI:
  sonuclar/model_karsilastirma/asama2_model_karsilastirma.csv / .json (R² tablosu)
  asama2_elektrik_tahmini/ciktilar/elektrik_tahmin_2026_2030.csv      (7 model + en iyi)
    → en iyi model sutunu AŞAMA 3'e girdi olur.
"""

import os, sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from karsilastirma_ortak import (yedi_modeli_calistir, en_iyi_model, MODEL_SIRASI,
                                 FEATURE_MODELLER, TF_MEVCUT, SM_MEVCUT,
                                 _final_predict_fn, forecast_arima_seri,
                                 iteratif_forecast)

# ─── Sabitler ─────────────────────────────────────────────────────────────────
ULKE          = 'Turkey'
HEDEF         = 'electricity_demand'
BIRIM         = 'TWh'
LAG_SAYISI    = 2
MIN_EGITIM_YIL = 15
FC_ILK, FC_SON = 2026, 2030
N_TRIAL_TREE  = 30
N_TRIAL_LSTM  = 12
LSTM_EPOCHS   = 150
FARK_AL       = True

# Aşama 1'in urettigi 18 korele parametre (elektrigi etkileyenlerin tamami)
FEATURE_COLS = [
    'Population', 'GDP_Per_Capita', 'GDP_Growth',
    'renewables_share_elec', 'Urbanization_Rate',
    'Industry_Value_Added_Pct_GDP', 'Population_Density',
    'fossil_fuel_consumption', 'nuclear_consumption',
    'primary_energy_consumption', 'energy_per_capita', 'energy_per_gdp',
    'nuclear_share_elec', 'fossil_share_elec',
    'coal_co2', 'gas_co2', 'co2_per_capita',
    'temperature_change_from_ghg',
]

# ─── Yollar ───────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
ISLENMIS_DIR = os.path.join(PIPELINE_DIR, 'veri', 'islenmis')
ASAMA1_DIR   = os.path.join(PIPELINE_DIR, 'asama1_feature_tahmini', 'ciktilar')
KARS_DIR     = os.path.join(PIPELINE_DIR, 'sonuclar', 'model_karsilastirma')
ASAMA2_CIKTI = os.path.join(PIPELINE_DIR, 'asama2_elektrik_tahmini', 'ciktilar')
os.makedirs(KARS_DIR, exist_ok=True)
os.makedirs(ASAMA2_CIKTI, exist_ok=True)

CSV_YOL      = os.path.join(KARS_DIR, 'asama2_model_karsilastirma.csv')
JSON_YOL     = os.path.join(KARS_DIR, 'asama2_model_karsilastirma.json')
FORECAST_CSV = os.path.join(ASAMA2_CIKTI, 'elektrik_tahmin_2026_2030.csv')
ASAMA1_FC    = os.path.join(ASAMA1_DIR, 'feature_tahminleri_2021_2030.csv')


def _nan(o):
    if isinstance(o, float) and (o != o):
        return None
    raise TypeError


def main():
    print("=" * 74)
    print("ASAMA 2 — 7 MODEL KARSILASTIRMASI + FORECAST (Turkiye elektrik)")
    print(f"  Modeller: {', '.join(MODEL_SIRASI)}")
    print(f"  TensorFlow(LSTM)={TF_MEVCUT}  statsmodels(ARIMA/SARIMA)={SM_MEVCUT}")
    print(f"  CV: Walk-Forward | Forecast: en iyi model {FC_ILK}-{FC_SON} | differencing={FARK_AL}")
    print("=" * 74)

    # Gecmis gercek veri
    tr = pd.concat([
        pd.read_csv(os.path.join(ISLENMIS_DIR, 'train.csv')),
        pd.read_csv(os.path.join(ISLENMIS_DIR, 'val.csv')),
        pd.read_csv(os.path.join(ISLENMIS_DIR, 'test.csv')),
    ])
    tr = tr[tr['country'] == ULKE].sort_values('year').reset_index(drop=True)

    # AŞAMA 1 forecast (gelecek feature degerleri)
    if not os.path.exists(ASAMA1_FC):
        raise RuntimeError(f"AŞAMA 1 ciktisi yok: {ASAMA1_FC}\n  Once asama1_7model_forecast.py calistirin.")
    a1 = pd.read_csv(ASAMA1_FC)
    a1 = a1[a1['country'] == ULKE].sort_values('year').reset_index(drop=True)

    # Kullanilabilir feature (>=5 dolu yil)
    kullanilan = [f for f in FEATURE_COLS
                  if f in tr.columns and tr[f].notna().sum() >= 5]
    atlanan = [f for f in FEATURE_COLS if f not in kullanilan]
    print(f"  {ULKE}: {len(tr)} yil ({tr['year'].min()}-{tr['year'].max()})")
    print(f"  Kullanilan feature ({len(kullanilan)})")
    if atlanan:
        print(f"  Atlanan (yetersiz veri): {atlanan}")

    # LEAKAGE GUARD: ffill (gecmisten); bastaki NaN'lar sabit 0 ile (bfill gelecek
    # bilgisini gecmise tasirdi → walk-forward CV'de leakage). Bkz. code review H1.
    tr[kullanilan] = tr[kullanilan].ffill().fillna(0.0)
    for k in range(1, LAG_SAYISI + 1):
        tr[f'{HEDEF}_lag{k}'] = tr[HEDEF].shift(k)
    lag_cols = [f'{HEDEF}_lag{k}' for k in range(1, LAG_SAYISI + 1)]
    input_cols = lag_cols + kullanilan
    tr = tr.dropna(subset=input_cols + [HEDEF]).reset_index(drop=True)

    X = tr[input_cols].values
    y = tr[HEDEF].values
    print(f"  Girdi: {len(lag_cols)} lag + {len(kullanilan)} feature = {len(input_cols)}")
    print(f"  Walk-Forward CV: min_train={MIN_EGITIM_YIL}, "
          f"{len(y)-MIN_EGITIM_YIL} out-of-sample tahmin\n")

    # ── 7 model CV ──
    t0 = time.time()
    s = yedi_modeli_calistir(X, y, MIN_EGITIM_YIL, n_trials_tree=N_TRIAL_TREE,
                             n_trials_lstm=N_TRIAL_LSTM, lstm_epochs=LSTM_EPOCHS,
                             etiket=HEDEF, fark_al=FARK_AL)
    best = en_iyi_model(s)
    print(f"\n  EN IYI MODEL: {best} (R²={s[best]['r2']:.4f})" if best else "  tum NaN")

    # ── Forecast (7 model) ──
    # Aşama 1 forecast'inden 2026-2030 feature degerleri
    feat_fc = {}
    for _, row in a1[a1['year'].between(FC_ILK, FC_SON)].iterrows():
        yil = int(row['year'])
        feat_fc[yil] = {}
        for f in kullanilan:
            v = row.get(f, np.nan)
            if pd.isna(v):  # Aşama 1'de yoksa son gercek deger
                v = tr[f].dropna().iloc[-1] if f in tr.columns and tr[f].notna().any() else 0.0
            feat_fc[yil][f] = float(v)
    fc_yillar = list(range(FC_ILK, FC_SON + 1))
    gecmis = dict(zip(tr['year'].astype(int), tr[HEDEF].astype(float)))

    tum_fc = {}   # {model: {yil: tahmin}}
    for model in MODEL_SIRASI:
        if s.get(model, {}).get('r2') != s.get(model, {}).get('r2'):
            continue  # NaN model, forecast yok
        if model in FEATURE_MODELLER:
            predict_fn = _final_predict_fn(model, s[model].get('params'), X, y,
                                           lstm_epochs=max(LSTM_EPOCHS, 200),
                                           fark_al=FARK_AL)
            if predict_fn is None:
                continue
            fc = iteratif_forecast(
                predict_fn, gecmis, fc_yillar,
                x_kurucu=lambda yil, lags: [list(lags) + [feat_fc[yil][f] for f in kullanilan]],
                lag_sayisi=LAG_SAYISI, fark_al=FARK_AL)
        else:  # ARIMA/SARIMA univariate
            arr = forecast_arima_seri(y, s[model].get('order'),
                                      s[model].get('seasonal_order'), len(fc_yillar))
            fc = {yil: float(v) for yil, v in zip(fc_yillar, arr)}
        tum_fc[model] = fc

    print(f"\n  Forecast {FC_ILK}-{FC_SON} (en iyi = {best}):")
    if best in tum_fc:
        for yil in fc_yillar:
            print(f"    {yil}: {tum_fc[best][yil]:.2f} {BIRIM}")
    print(f"  Sure: {(time.time()-t0)/60:.1f}dk")

    # ── Kaydet ──
    temiz = {m: {k: v for k, v in d.items() if k not in ('trues', 'preds')}
             for m, d in s.items()}
    out = {'hedef': HEDEF, 'birim': BIRIM, 'en_iyi': best,
           'input_cols': input_cols, 'n_obs': int(len(y)),
           'min_train': MIN_EGITIM_YIL, 'fark_al': FARK_AL, 'modeller': temiz}
    _kaydet_kars(out)
    _kaydet_forecast(tum_fc, best, fc_yillar)
    print(f"\n  R² tablosu : {CSV_YOL}")
    print(f"  Forecast   : {FORECAST_CSV}")


def _kaydet_kars(out):
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


def _kaydet_forecast(tum_fc, best, fc_yillar):
    """elektrik_tahmin_2026_2030.csv: her model sutunu + en_iyi sutunu (Aşama 3 bunu okur)."""
    rows = {'year': fc_yillar}
    for model in MODEL_SIRASI:
        if model in tum_fc:
            rows[f'{model}_{BIRIM}'] = [round(tum_fc[model][y], 4) for y in fc_yillar]
    # En iyi modelin tahmini ayri sutun (Aşama 3 net okusun)
    if best in tum_fc:
        rows['EN_IYI_TWh'] = [round(tum_fc[best][y], 4) for y in fc_yillar]
        rows['EN_IYI_MODEL'] = [best] * len(fc_yillar)
    tmp = FORECAST_CSV + '.tmp'
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, FORECAST_CSV)


if __name__ == '__main__':
    main()
