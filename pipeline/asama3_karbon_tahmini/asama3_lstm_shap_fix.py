"""
LSTM SHAP Post-Hoc Fix
======================
asama3_egit.py'de LSTM modeli için shap.Explainer "unknown rank" hatası verdi
(TF Sequential modelin input shape'ini SHAP otomatik tespit edemedi).

Bu script kayıtlı asama3_lstm.keras modelinden SHAP grafiklerini POST-HOC üretir.
Model tekrar train edilmez — sadece SHAP analizi.

Çözüm: numpy wrapper ile shap.KernelExplainer kullan.
"""

import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import load_model

# ─── Yollar ───────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR  = os.path.dirname(SCRIPT_DIR)
ISLENMIS_DIR  = os.path.join(PIPELINE_DIR, 'veri', 'islenmis')
HAM_DIR       = os.path.join(PIPELINE_DIR, 'veri', 'ham')
ASAMA1_DIR    = os.path.join(PIPELINE_DIR, 'asama1_feature_tahmini', 'ciktilar')
ASAMA2_DIR    = os.path.join(PIPELINE_DIR, 'asama2_elektrik_tahmini', 'ciktilar')
MODELLER_DIR  = os.path.join(SCRIPT_DIR, 'modeller')
GORSELLER_DIR = os.path.join(PIPELINE_DIR, 'sonuclar', 'gorseller')

# Asama 3 ile aynı feature listesi
FEATURE_COLS = [
    'electricity_demand', 'Population', 'Urbanization_Rate', 'Population_Density',
    'GDP_Per_Capita', 'Industry_Value_Added_Pct_GDP',
    'fossil_fuel_consumption', 'nuclear_consumption',
    'primary_energy_consumption', 'energy_per_capita', 'energy_per_gdp',
    'renewables_share_elec', 'nuclear_share_elec', 'fossil_share_elec',
    'temperature_change_from_ghg',
]
LAG_COLS = ['co2_lag1', 'co2_lag2']
INPUT_COLS = LAG_COLS + FEATURE_COLS

ULKE = 'Turkey'
HEDEF = 'co2'

print("=" * 70)
print("LSTM SHAP POST-HOC FIX")
print("=" * 70)

# ─── Veri hazirla (asama3_egit.py ile aynı) ─────────────────────────────────
print("\nVeri yukleniyor...")
owid_co2 = pd.read_csv(os.path.join(HAM_DIR, 'owid_co2_emissions.csv'), comment='#')
owid_co2 = owid_co2.rename(columns={'Country':'country','Year':'year'})
tr_co2 = owid_co2[owid_co2['country']==ULKE][['year','co2']].dropna().sort_values('year')

train_all = pd.read_csv(os.path.join(ISLENMIS_DIR, 'train.csv'))
val_all   = pd.read_csv(os.path.join(ISLENMIS_DIR, 'val.csv'))
test_all  = pd.read_csv(os.path.join(ISLENMIS_DIR, 'test.csv'))
turkey_feat = pd.concat([
    train_all[train_all['country']==ULKE],
    val_all[val_all['country']==ULKE],
    test_all[test_all['country']==ULKE],
]).sort_values('year').reset_index(drop=True)

turkey_df = tr_co2.merge(
    turkey_feat[['year']+[c for c in FEATURE_COLS if c in turkey_feat.columns]],
    on='year', how='inner').sort_values('year').reset_index(drop=True)

KULLANILAN = [f for f in FEATURE_COLS if f in turkey_df.columns
              and turkey_df[f].notna().sum() >= 5]
turkey_df[KULLANILAN] = turkey_df[KULLANILAN].ffill().bfill()

for k in [1, 2]:
    turkey_df[f'{HEDEF}_lag{k}'] = turkey_df[HEDEF].shift(k)

INPUT_COLS_GERCEK = LAG_COLS + KULLANILAN
turkey_df = turkey_df.dropna(subset=INPUT_COLS_GERCEK + [HEDEF]).reset_index(drop=True)
tum_X = turkey_df[INPUT_COLS_GERCEK].values
tum_y = turkey_df[HEDEF].values

print(f"  Veri: {len(turkey_df)} yil, {len(INPUT_COLS_GERCEK)} feature")

# ─── Model + scaler yukle ──────────────────────────────────────────────────
print("\nModel + scaler yukleniyor...")
lstm_model = load_model(os.path.join(MODELLER_DIR, 'asama3_lstm.keras'))
sf_l = joblib.load(os.path.join(MODELLER_DIR, 'asama3_lstm_feat_scaler.pkl'))
sh_l = joblib.load(os.path.join(MODELLER_DIR, 'asama3_lstm_hedef_scaler.pkl'))
print(f"  ✓ asama3_lstm.keras + 2 scaler")

# Scaled input
X_scaled = sf_l.transform(tum_X)
print(f"  Scaled input shape: {X_scaled.shape}")

# ─── SHAP wrapper: 2D input → LSTM 3D + scaler unwrap ──────────────────────
def lstm_predict_wrapper(X_2d):
    """SHAP'in cagiracagi predict fonksiyonu. 2D input alir, scaled output verir."""
    X_3d = X_2d.reshape(-1, 1, X_2d.shape[-1])
    p_s = lstm_model.predict(X_3d, verbose=0).ravel()
    # Scaled hedef → orijinal birim (Mt CO2)
    return sh_l.inverse_transform(p_s.reshape(-1,1)).ravel()

# Sanity check
test_pred = lstm_predict_wrapper(X_scaled[:3])
print(f"  Sanity check: ilk 3 tahmin = {test_pred.round(2)}")

# ─── SHAP KernelExplainer ──────────────────────────────────────────────────
print("\nSHAP KernelExplainer...")
n_baseline = min(15, len(X_scaled))  # SHAP icin baseline (shap.utils.sample)
baseline = shap.utils.sample(X_scaled, n_baseline, random_state=42)
explainer = shap.KernelExplainer(lstm_predict_wrapper, baseline)

print(f"  Baseline boyutu: {baseline.shape}")
print(f"  SHAP degerleri hesaplaniyor ({len(X_scaled)} satir × {len(INPUT_COLS_GERCEK)} feature)...")
shap_values = explainer.shap_values(X_scaled, nsamples=80, silent=True)
shap_values = np.array(shap_values)
print(f"  ✓ SHAP shape: {shap_values.shape}")

# ─── Grafikleri ciz ─────────────────────────────────────────────────────────
explanation = shap.Explanation(
    values=shap_values,
    data=X_scaled,
    feature_names=INPUT_COLS_GERCEK,
)

# 1. SHAP bar
yol_bar = os.path.join(GORSELLER_DIR, 'asama3_LSTM_shap_bar.png')
plt.figure(figsize=(10, max(5, len(INPUT_COLS_GERCEK)*0.4)))
shap.plots.bar(explanation, show=False, max_display=len(INPUT_COLS_GERCEK))
plt.title(f'LSTM (Turkiye) — SHAP Bar (mean |SHAP|)\nExplainer: kernel (post-hoc)')
plt.tight_layout()
plt.savefig(yol_bar, dpi=150, bbox_inches='tight')
plt.close('all')
print(f"  ✓ SHAP bar: {yol_bar}")

# 2. SHAP beeswarm
yol_beeswarm = os.path.join(GORSELLER_DIR, 'asama3_LSTM_shap_beeswarm.png')
plt.figure(figsize=(11, max(5, len(INPUT_COLS_GERCEK)*0.4)))
shap.plots.beeswarm(explanation, show=False, max_display=len(INPUT_COLS_GERCEK))
plt.title(f'LSTM (Turkiye) — SHAP Beeswarm (her nokta=bir yil)\nExplainer: kernel')
plt.tight_layout()
plt.savefig(yol_beeswarm, dpi=150, bbox_inches='tight')
plt.close('all')
print(f"  ✓ SHAP beeswarm: {yol_beeswarm}")

print("\n" + "=" * 70)
print("LSTM SHAP POST-HOC FIX TAMAMLANDI")
print("=" * 70)
print(f"\n2 grafik kaydedildi:")
print(f"  {yol_bar}")
print(f"  {yol_beeswarm}")
