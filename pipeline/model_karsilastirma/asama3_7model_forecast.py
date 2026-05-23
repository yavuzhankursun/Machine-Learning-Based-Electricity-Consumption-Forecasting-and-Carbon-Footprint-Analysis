"""
AŞAMA 3 — 7 MODEL KARSILASTIRMASI + FORECAST (Turkiye karbon ayak izi, Mt CO2)
==============================================================================
Turkiye'nin 2026-2030 toplam CO2 emisyonunu 7 modelle tahmin eder.

  7 MODEL : RF · XGBoost · LightGBM · CatBoost · LSTM · ARIMA · SARIMA
  GIRDI (tree/LSTM): co2 lag_1, lag_2 + electricity_demand + leakage'siz korele parametreler
       - electricity_demand FORECAST'i ← AŞAMA 2'nin EN IYI modelinden
       - diger parametre forecast'leri ← AŞAMA 1'den
  GIRDI (ARIMA/SARIMA): co2 serisi (univariate baseline)
  CV       : Walk-Forward Expanding Window (tum modeller AYNI test noktalari)
  FORECAST : En iyi model; tree/LSTM iteratif + differencing

  ⚠️ LEAKAGE GUARD: co2 turevi feature'lar (coal_co2, gas_co2, co2_per_capita,
     total_ghg, ...) YASAKTIR — assertion ile kontrol edilir.

GIRDI DOSYALARI:
  veri/ham/owid_co2_emissions.csv                                    (gecmis gercek CO2)
  veri/islenmis/train.csv, val.csv, test.csv                         (gecmis gercek feature)
  asama1_feature_tahmini/ciktilar/feature_tahminleri_2021_2030.csv   (AŞAMA 1 forecast)
  asama2_elektrik_tahmini/ciktilar/elektrik_tahmin_2026_2030.csv     (AŞAMA 2 en iyi forecast)

CIKTI DOSYALARI:
  sonuclar/model_karsilastirma/asama3_model_karsilastirma.csv / .json
  asama3_karbon_tahmini/ciktilar/karbon_tahmin_2026_2030.csv
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
HEDEF         = 'co2'
BIRIM         = 'Mt CO2'
LAG_SAYISI    = 2
MIN_EGITIM_YIL = 10
FC_ILK, FC_SON = 2026, 2030
N_TRIAL_TREE  = 30
N_TRIAL_LSTM  = 12
LSTM_EPOCHS   = 150
FARK_AL       = True

# LEAKAGE GUARD: bu listedeki hicbir feature kullanilmaz (co2 turevi/bileseni)
YASAK_LEAKAGE = {
    'co2', 'co2_per_capita', 'co2_per_gdp',
    'coal_co2', 'oil_co2', 'gas_co2', 'cement_co2', 'flaring_co2',
    'consumption_co2', 'trade_co2', 'cumulative_co2',
    'total_ghg', 'ghg_per_capita', 'methane', 'nitrous_oxide',
    'temperature_change_from_co2', 'share_global_co2',
    'co2_growth_abs', 'co2_growth_prct',
}

# Karbonu etkileyen leakage'siz parametreler (18 - co2 turevleri + electricity_demand)
FEATURE_COLS = [
    'electricity_demand',                    # ← AŞAMA 2 forecast
    'Population', 'GDP_Per_Capita', 'GDP_Growth',
    'renewables_share_elec', 'Urbanization_Rate',
    'Industry_Value_Added_Pct_GDP', 'Population_Density',
    'fossil_fuel_consumption', 'nuclear_consumption',
    'primary_energy_consumption', 'energy_per_capita', 'energy_per_gdp',
    'nuclear_share_elec', 'fossil_share_elec',
    'temperature_change_from_ghg',
    # NOT: coal_co2 / gas_co2 / co2_per_capita BILEREK YOK (leakage)
]

# ─── Yollar ───────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
ISLENMIS_DIR = os.path.join(PIPELINE_DIR, 'veri', 'islenmis')
HAM_DIR      = os.path.join(PIPELINE_DIR, 'veri', 'ham')
ASAMA1_DIR   = os.path.join(PIPELINE_DIR, 'asama1_feature_tahmini', 'ciktilar')
ASAMA2_DIR   = os.path.join(PIPELINE_DIR, 'asama2_elektrik_tahmini', 'ciktilar')
KARS_DIR     = os.path.join(PIPELINE_DIR, 'sonuclar', 'model_karsilastirma')
ASAMA3_CIKTI = os.path.join(PIPELINE_DIR, 'asama3_karbon_tahmini', 'ciktilar')
os.makedirs(KARS_DIR, exist_ok=True)
os.makedirs(ASAMA3_CIKTI, exist_ok=True)

CSV_YOL      = os.path.join(KARS_DIR, 'asama3_model_karsilastirma.csv')
JSON_YOL     = os.path.join(KARS_DIR, 'asama3_model_karsilastirma.json')
FORECAST_CSV = os.path.join(ASAMA3_CIKTI, 'karbon_tahmin_2026_2030.csv')
ASAMA1_FC    = os.path.join(ASAMA1_DIR, 'feature_tahminleri_2021_2030.csv')
ASAMA2_FC    = os.path.join(ASAMA2_DIR, 'elektrik_tahmin_2026_2030.csv')


def _nan(o):
    if isinstance(o, float) and (o != o):
        return None
    raise TypeError


def main():
    print("=" * 74)
    print("ASAMA 3 — 7 MODEL KARSILASTIRMASI + FORECAST (Turkiye karbon)")
    print(f"  Modeller: {', '.join(MODEL_SIRASI)}")
    print(f"  TensorFlow(LSTM)={TF_MEVCUT}  statsmodels(ARIMA/SARIMA)={SM_MEVCUT}")
    print(f"  CV: Walk-Forward | Forecast: en iyi model {FC_ILK}-{FC_SON} | differencing={FARK_AL}")
    print("=" * 74)

    # LEAKAGE assertion
    bulunan = [f for f in FEATURE_COLS if f in YASAK_LEAKAGE]
    if bulunan:
        raise ValueError(f"❌ LEAKAGE: {bulunan} co2 turevi — FEATURE_COLS'tan cikar!")
    print(f"  ✓ LEAKAGE KONTROL: {len(FEATURE_COLS)} feature temiz")

    # Gecmis gercek CO2 (owid)
    owid = pd.read_csv(os.path.join(HAM_DIR, 'owid_co2_emissions.csv'), comment='#')
    owid = owid.rename(columns={'Country': 'country', 'Year': 'year'})
    co2 = owid[owid['country'] == ULKE][['year', 'co2']].dropna().sort_values('year')

    # Gecmis gercek feature
    tr = pd.concat([
        pd.read_csv(os.path.join(ISLENMIS_DIR, 'train.csv')),
        pd.read_csv(os.path.join(ISLENMIS_DIR, 'val.csv')),
        pd.read_csv(os.path.join(ISLENMIS_DIR, 'test.csv')),
    ])
    tr = tr[tr['country'] == ULKE].sort_values('year').reset_index(drop=True)

    df = co2.merge(tr[['year'] + [c for c in FEATURE_COLS if c in tr.columns]],
                   on='year', how='inner').sort_values('year').reset_index(drop=True)
    print(f"  {ULKE} CO2: {len(df)} yil ({df['year'].min()}-{df['year'].max()}) | "
          f"{df['co2'].min():.1f}-{df['co2'].max():.1f} {BIRIM}")

    # AŞAMA 1 + AŞAMA 2 forecast dosyalari
    for yol, ad in [(ASAMA1_FC, 'AŞAMA 1'), (ASAMA2_FC, 'AŞAMA 2')]:
        if not os.path.exists(yol):
            raise RuntimeError(f"{ad} ciktisi yok: {yol}")
    a1 = pd.read_csv(ASAMA1_FC); a1 = a1[a1['country'] == ULKE].sort_values('year')
    a2 = pd.read_csv(ASAMA2_FC)
    # AŞAMA 2'nin en iyi elektrik sutunu
    if 'EN_IYI_TWh' in a2.columns:
        elec_col = 'EN_IYI_TWh'
    else:
        elec_col = [c for c in a2.columns if c.endswith('_TWh')][0]
    print(f"  Forecast elektrik kaynagi: AŞAMA 2 ({elec_col}, en iyi model)")

    kullanilan = [f for f in FEATURE_COLS
                  if f in df.columns and df[f].notna().sum() >= 5]
    atlanan = [f for f in FEATURE_COLS if f not in kullanilan]
    if atlanan:
        print(f"  Atlanan (yetersiz veri): {atlanan}")

    # LEAKAGE GUARD: ffill (gecmisten); bastaki NaN'lar sabit 0 ile (bfill gelecek
    # bilgisini gecmise tasirdi → walk-forward CV'de leakage). Bkz. code review H1.
    df[kullanilan] = df[kullanilan].ffill().fillna(0.0)
    for k in range(1, LAG_SAYISI + 1):
        df[f'{HEDEF}_lag{k}'] = df[HEDEF].shift(k)
    lag_cols = [f'{HEDEF}_lag{k}' for k in range(1, LAG_SAYISI + 1)]
    input_cols = lag_cols + kullanilan
    df = df.dropna(subset=input_cols + [HEDEF]).reset_index(drop=True)

    X = df[input_cols].values
    y = df[HEDEF].values
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

    # ── Forecast feature degerleri (2026-2030) ──
    a2_elec = dict(zip(a2['year'].astype(int), a2[elec_col].astype(float)))
    feat_fc = {}
    for yil in range(FC_ILK, FC_SON + 1):
        feat_fc[yil] = {}
        a1_row = a1[a1['year'] == yil]
        for f in kullanilan:
            if f == 'electricity_demand':
                v = a2_elec.get(yil, np.nan)              # AŞAMA 2'den
            elif len(a1_row) and f in a1_row.columns:
                v = a1_row[f].values[0]                   # AŞAMA 1'den
            else:
                v = np.nan
            if pd.isna(v):
                v = df[f].dropna().iloc[-1] if f in df.columns and df[f].notna().any() else 0.0
            feat_fc[yil][f] = float(v)
    fc_yillar = list(range(FC_ILK, FC_SON + 1))
    gecmis = dict(zip(df['year'].astype(int), df[HEDEF].astype(float)))

    tum_fc = {}
    for model in MODEL_SIRASI:
        if s.get(model, {}).get('r2') != s.get(model, {}).get('r2'):
            continue
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
        else:
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
    out = {'hedef': HEDEF, 'birim': BIRIM, 'en_iyi': best, 'input_cols': input_cols,
           'n_obs': int(len(y)), 'min_train': MIN_EGITIM_YIL, 'fark_al': FARK_AL,
           'elektrik_kaynagi': elec_col, 'modeller': temiz}
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
    rows = {'year': fc_yillar}
    for model in MODEL_SIRASI:
        if model in tum_fc:
            rows[f'{model}_MtCO2'] = [round(tum_fc[model][y], 4) for y in fc_yillar]
    if best in tum_fc:
        rows['EN_IYI_MtCO2'] = [round(tum_fc[best][y], 4) for y in fc_yillar]
        rows['EN_IYI_MODEL'] = [best] * len(fc_yillar)
    tmp = FORECAST_CSV + '.tmp'
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, FORECAST_CSV)


if __name__ == '__main__':
    main()
