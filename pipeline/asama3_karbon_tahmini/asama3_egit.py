"""
AŞAMA 3 — Karbon Ayak İzi Tahmini (Walk-Forward CV + Optuna + SHAP)
--------------------------------------------------------------------
Türkiye'nin 2026-2030 toplam CO2 emisyonunu (milyon ton) tahmin eder.

PIPELINE:
  Aşama 1 → Feature tahminleri (Population, GDP, vs.)
  Aşama 2 → Elektrik tüketimi tahmini (TWh)
  Aşama 3 → Karbon ayak izi tahmini (milyon ton CO2)  ← BU

DATA LEAKAGE GÜVENCESİ (kod boyunca uygulanır):
  ✓ YASAK_LEAKAGE listesi: co2/co2_per_capita/co2_per_gdp/coal_co2/oil_co2/
    gas_co2/cement_co2/flaring_co2/consumption_co2/trade_co2/cumulative_co2/
    total_ghg/ghg_per_capita/methane/nitrous_oxide/temperature_change_from_co2/
    share_global_co2 — kodda assertion ile kontrol edilir
  ✓ co2 lag'leri yalnızca t-1, t-2 (geçmiş)
  ✓ Forecast 2026-2030: model kendi co2 tahminini lag'ler (iteratif)
  ✓ Walk-forward CV: train[0..t-1] → tahmin[t]; gelecek bilgi sızmaz
  ✓ MinMaxScaler her fold'da SADECE train fold'a fit edilir
  ✓ Aşama 1 ve 2 forecast'leri 2030 yılına kadar zaten tahmin edilmiş
    (gerçek 2026-2030 değerleri kullanılmaz)

CHECKPOINT SİSTEMİ:
  ✓ Her model bittiğinde tahminler + metrikler + grafikler atomic kaydedilir
  ✓ Yeniden başlatıldığında tamamlanmış modeller atlanır

GİRDİLER (15 leakage'siz feature):
  - co2_lag1, co2_lag2 (geçmiş CO2; iteratif forecast)
  - electricity_demand (Aşama 2 tahmini, 2026-2030)
  Demografi: Population, Urbanization_Rate, Population_Density
  Ekonomi:   GDP_Per_Capita, Industry_Value_Added_Pct_GDP
  Enerji:    fossil_fuel_consumption, nuclear_consumption,
             primary_energy_consumption, energy_per_capita,
             energy_per_gdp (⭐ STIRPAT)
  Mix:       renewables_share_elec, nuclear_share_elec, fossil_share_elec
  İklim:     temperature_change_from_ghg
  ÇIKARILDI: GDP_Growth (R²=-0.25), Electricity_Loss_Pct (Türkiye 2030 NaN),
             coal_co2/gas_co2/co2_per_capita (LEAKAGE)

MODELLER: Random Forest | XGBoost | LSTM
"""

import os, time, warnings
import numpy as np
import pandas as pd
import joblib
import optuna
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.ensemble        import RandomForestRegressor
from sklearn.preprocessing   import MinMaxScaler
from sklearn.metrics         import (mean_squared_error, mean_absolute_error,
                                      r2_score, confusion_matrix)
from sklearn.model_selection import TimeSeriesSplit
from xgboost                 import XGBRegressor

try:
    import tensorflow as tf
    from tensorflow.keras.models    import Sequential
    from tensorflow.keras.layers    import LSTM as LSTM_Layer, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_MEVCUT = True
except ImportError:
    TF_MEVCUT = False
    print("[UYARI] TensorFlow bulunamadi — LSTM atlaniyor.")

# ═════════════════════════════════════════════════════════════════════════════
# SABITLER + LEAKAGE GUVENLIGI
# ═════════════════════════════════════════════════════════════════════════════

FC_ILK = 2026;  FC_SON = 2030
LAG_SAYISI     = 2
MIN_EGITIM_YIL = 10
OPTUNA_TRIAL   = 40
LSTM_EPOCHS    = 200
LSTM_PATIENCE  = 20

ULKE  = 'Turkey'
HEDEF = 'co2'  # milyon ton CO2

# LEAKAGE GUARD: bu listede olan hicbir feature kullanilmaz
YASAK_LEAKAGE = {
    'co2', 'co2_per_capita', 'co2_per_gdp',
    'coal_co2', 'oil_co2', 'gas_co2', 'cement_co2', 'flaring_co2',
    'consumption_co2', 'trade_co2', 'cumulative_co2',
    'total_ghg', 'ghg_per_capita',
    'methane', 'nitrous_oxide',
    'temperature_change_from_co2', 'share_global_co2',
    'co2_growth_abs', 'co2_growth_prct',
}

# Asama1 R²>=0.4 olan + electricity_demand (Asama 2)
# Toplam 15 feature + 2 lag = 17 girdi
FEATURE_COLS = [
    # — Aşama 2 çıktısı —
    'electricity_demand',           # asama2 tahmini, korr=0.990
    # — Demografi (Aşama 1) —
    'Population',                   # asama1 R²=0.999
    'Urbanization_Rate',            # asama1 R²=0.989
    'Population_Density',           # asama1 R²=0.997
    # — Ekonomi (Aşama 1) —
    'GDP_Per_Capita',               # asama1 R²=0.744
    'Industry_Value_Added_Pct_GDP', # asama1 R²=0.500 (fabrika katma deger)
    # — Enerji miktar (Aşama 1) —
    'fossil_fuel_consumption',      # asama1 R²=0.960 (TWh)
    'nuclear_consumption',          # asama1 R²=1.000 (TWh, Turkey'de 0)
    'primary_energy_consumption',   # asama1 R²=0.979 (YENI)
    'energy_per_capita',            # asama1 R²=0.961 (YENI)
    'energy_per_gdp',               # asama1 R²=0.757 (YENI ⭐STIRPAT)
    # — Enerji mix payları (Aşama 1) —
    'renewables_share_elec',        # asama1 R²=0.574
    'nuclear_share_elec',           # asama1 R²=1.000 (YENI, Turkey'de 0)
    'fossil_share_elec',            # asama1 R²=0.550 (YENI)
    # — İklim (sınırda, ters yön sinyal) —
    'temperature_change_from_ghg',  # asama1 R²=0.838
]

# LEAKAGE ASSERTION — kodu calismadan once kontrol et
def _leakage_kontrol():
    bulunan = [f for f in FEATURE_COLS if f in YASAK_LEAKAGE]
    if bulunan:
        raise ValueError(
            f"❌ LEAKAGE TESPIT EDILDI! Bu feature'lar co2 turevi/bileseni:\n"
            f"   {bulunan}\n"
            f"   FEATURE_COLS'tan cikar veya YASAK_LEAKAGE listesini guncelle.")
    print(f"  ✓ LEAKAGE KONTROL: {len(FEATURE_COLS)} feature temiz, "
          f"hicbiri co2 turevi degil")

LAG_COLS = [f'{HEDEF}_lag{k}' for k in range(1, LAG_SAYISI+1)]

# ═════════════════════════════════════════════════════════════════════════════
# YOLLAR
# ═════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR  = os.path.dirname(SCRIPT_DIR)
ISLENMIS_DIR  = os.path.join(PIPELINE_DIR, 'veri', 'islenmis')
HAM_DIR       = os.path.join(PIPELINE_DIR, 'veri', 'ham')
ASAMA1_DIR    = os.path.join(PIPELINE_DIR, 'asama1_feature_tahmini', 'ciktilar')
ASAMA2_DIR    = os.path.join(PIPELINE_DIR, 'asama2_elektrik_tahmini', 'ciktilar')
MODELLER_DIR  = os.path.join(SCRIPT_DIR, 'modeller')
CIKTILAR_DIR  = os.path.join(SCRIPT_DIR, 'ciktilar')
GORSELLER_DIR = os.path.join(PIPELINE_DIR, 'sonuclar', 'gorseller')
for d in [MODELLER_DIR, CIKTILAR_DIR, GORSELLER_DIR]:
    os.makedirs(d, exist_ok=True)

METRIK_CSV  = os.path.join(CIKTILAR_DIR, 'asama3_metrikler.csv')
TAHMIN_CSV  = os.path.join(CIKTILAR_DIR, 'karbon_tahmin_2026_2030.csv')
CV_CSV      = os.path.join(CIKTILAR_DIR, 'asama3_cv_detay.csv')

# ═════════════════════════════════════════════════════════════════════════════
# YARDIMCI (matematik + zaman)
# ═════════════════════════════════════════════════════════════════════════════

def mape(y_true, y_pred):
    t, p = np.array(y_true, float), np.array(y_pred, float)
    mask = t != 0
    return np.mean(np.abs((t[mask] - p[mask]) / t[mask])) * 100 if mask.sum() else np.nan

def metrikleri_hesapla(y_true, y_pred):
    """R², MAE, RMSE, MAPE — tum metrikler tek dict'te."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    mp   = float(mape(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred)) if len(y_true) > 2 else float('nan')
    return {'r2': round(r2,4), 'mae': round(mae,4),
            'rmse': round(rmse,4), 'mape': round(mp,4)}

def hata_bandi_hesapla(y_true, y_pred):
    pct = np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true)) * 100
    return {
        'Mukemmel (<%5)' : int(np.sum(pct < 5)),
        'Iyi (%5-10)'    : int(np.sum((pct>=5)  & (pct<10))),
        'Orta (%10-20)'  : int(np.sum((pct>=10) & (pct<20))),
        'Zayif (>%20)'   : int(np.sum(pct >= 20)),
    }

def format_sure(saniye):
    if saniye < 60:    return f"{saniye:.1f}sn"
    if saniye < 3600:  return f"{saniye/60:.1f}dk"
    return f"{saniye/3600:.2f}saat"

# ═════════════════════════════════════════════════════════════════════════════
# GRAFIKLER
# ═════════════════════════════════════════════════════════════════════════════

def ciz_walkforward(yillar_g, y_g, cv_yillar, cv_dict, fc_yillar, fc_dict, yol):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(yillar_g, y_g, 'k-o', ms=4, lw=2, label='Gercek', zorder=5)
    renkler = ['steelblue', 'seagreen', 'tomato']
    for (isim, preds), renk in zip(cv_dict.items(), renkler):
        ax.plot(cv_yillar, preds, '--s', color=renk, ms=5, label=f'{isim} (CV)')
    for (isim, preds), renk in zip(fc_dict.items(), renkler):
        ax.plot(fc_yillar, preds, ':^', color=renk, ms=5, alpha=0.7, label=f'{isim} (Forecast)')
    ax.axvspan(cv_yillar[0], cv_yillar[-1]+0.9, alpha=0.08, color='blue', label='Walk-Forward CV')
    ax.axvspan(FC_ILK, FC_SON+0.9, alpha=0.08, color='red', label='Forecast')
    ax.set_title(f'Asama 3 — Karbon Ayak Izi (Turkey): Gercek vs Modeller\n'
                 f'Walk-Forward CV ({len(cv_yillar)} out-of-sample tahmin)')
    ax.set_xlabel('Yil'); ax.set_ylabel('CO2 (milyon ton)'); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()

def ciz_residual(cv_yillar, y_g, y_t, isim, yol):
    res = np.array(y_g) - np.array(y_t)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(cv_yillar, res, color='steelblue', edgecolor='k')
    axes[0].axhline(0, color='red', lw=1.5)
    axes[0].set_xlabel('Yil'); axes[0].set_ylabel('Residual (Mt CO2)')
    axes[0].set_title(f'{isim} — Residual (Walk-Forward CV)')
    axes[1].scatter(y_t, res, color='steelblue', edgecolors='k', s=60)
    axes[1].axhline(0, color='red', lw=1.5)
    axes[1].set_xlabel('Tahmin'); axes[1].set_ylabel('Residual')
    axes[1].set_title(f'{isim} — Residual vs Tahmin')
    plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()

def ciz_residual_dist(trues, preds, isim, yol):
    """Residual histogram + Q-Q plot."""
    res = np.array(trues) - np.array(preds)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(res, bins=min(15, max(5, len(res)//2)), color='steelblue', edgecolor='k')
    axes[0].axvline(0, color='red', lw=2, label='Sifir hata')
    axes[0].axvline(np.mean(res), color='orange', ls='--', lw=2,
                    label=f'Bias={np.mean(res):.3f}')
    axes[0].set_xlabel('Residual (Mt CO2)'); axes[0].set_ylabel('Frekans')
    axes[0].set_title(f'{isim} — Residual Dagilimi'); axes[0].legend(fontsize=8)
    from scipy import stats
    stats.probplot(res, dist='norm', plot=axes[1])
    axes[1].set_title(f'{isim} — Q-Q Plot (Normallik)')
    plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()

def ciz_hata_bandi(bandlar, isim, yol):
    fig, ax = plt.subplots(figsize=(8, 4))
    renkler = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
    bars = ax.bar(list(bandlar.keys()), list(bandlar.values()), color=renkler, edgecolor='k')
    for bar, val in zip(bars, bandlar.values()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                str(val), ha='center', va='bottom', fontweight='bold')
    ax.set_title(f'{isim} — Hata Bandi (Walk-Forward CV)')
    ax.set_ylabel('Tahmin Sayisi')
    ax.set_ylim(0, max(bandlar.values())+1.5 if max(bandlar.values())>0 else 2)
    plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()

def ciz_confusion_matrix_regresyon(trues, preds, isim, yol):
    """Regresyon icin confusion matrix.
    SADECE BU FONKSIYONA OZEL: trues quintile'lara bolunur (small / mid-small /
    mid / mid-large / large), confusion matrix olusturulur.
    Bu binning baska analizde kullanilmaz."""
    if len(trues) < 5:
        return
    sinir = np.quantile(trues, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    sinif_isimleri = ['small', 'mid-small', 'mid', 'mid-large', 'large']

    def _sinifla(d):
        for i in range(5):
            if d <= sinir[i+1]:
                return i
        return 4

    y_t = np.array([_sinifla(v) for v in trues])
    y_p = np.array([_sinifla(v) for v in preds])
    cm = confusion_matrix(y_t, y_p, labels=list(range(5)))

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(cm, cmap='Blues', aspect='equal')
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(sinif_isimleri, rotation=30, ha='right')
    ax.set_yticklabels(sinif_isimleri)
    ax.set_xlabel('Tahmin Sinifi')
    ax.set_ylabel('Gercek Sinifi')
    ax.set_title(f'{isim} — Confusion Matrix (Quintile bins)\n'
                 f'5 buyukluk sinifi: small / mid-small / mid / mid-large / large')
    for i in range(5):
        for j in range(5):
            v = cm[i, j]
            ax.text(j, i, str(v), ha='center', va='center',
                    color='white' if v > cm.max()/2 else 'black',
                    fontsize=11, fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Tahmin Sayisi')
    dogru = np.trace(cm); toplam = cm.sum()
    acc = dogru/toplam if toplam else 0
    ax.text(0.02, -0.18, f'Dogru sinif: {dogru}/{toplam} (acc={acc:.2f})',
            transform=ax.transAxes, fontsize=10, color='darkgreen')
    plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()

def ciz_cv_scatter(trues, preds, isim, yol):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(trues, preds, color='steelblue', edgecolors='k', s=60, zorder=5)
    mn, mx = min(trues.min(), preds.min()), max(trues.max(), preds.max())
    margin = (mx - mn) * 0.05
    ax.plot([mn-margin, mx+margin], [mn-margin, mx+margin], 'r--', lw=1.5, label='Ideal (y=x)')
    r2 = r2_score(trues, preds)
    ax.set_xlabel('Gercek (Mt CO2)'); ax.set_ylabel('Tahmin (Mt CO2)')
    ax.set_title(f'{isim} — Walk-Forward CV\nR² = {r2:.4f}')
    ax.legend(); plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()

def ciz_learning_curve(tum_X, tum_y, model_factory, isim, min_train, yol):
    sizes, rmses = [], []
    for size in range(min_train, len(tum_y)-1):
        X_tr, y_tr = tum_X[:size], tum_y[:size]
        X_te, y_te = tum_X[size:size+1], tum_y[size:size+1]
        sc = MinMaxScaler()
        m = model_factory()
        m.fit(sc.fit_transform(X_tr), y_tr)
        pred = m.predict(sc.transform(X_te))[0]
        sizes.append(size)
        rmses.append(abs(y_te[0] - pred))
    if len(sizes) < 2: return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sizes, rmses, 'b-o', ms=4)
    ax.set_xlabel('Egitim Seti Boyutu (yil)'); ax.set_ylabel('Mutlak Hata (Mt CO2)')
    ax.set_title(f'{isim} — Learning Curve')
    plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()

def ciz_feature_importance(model, names, isim, yol):
    if not hasattr(model, 'feature_importances_'):
        return
    imp = model.feature_importances_
    idx = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(9, max(5, len(names)*0.35)))
    ax.barh([names[i] for i in idx], imp[idx], color='coral', edgecolor='k')
    for i, val in enumerate(imp[idx]):
        ax.text(val + imp.max()*0.01, i, f'{val:.3f}', va='center', fontsize=8)
    ax.set_xlabel('Feature Importance (built-in)')
    ax.set_title(f'{isim} — Built-in Feature Importance\n'
                 f'(Egitim sirasinda modelin verdigi agirliklar)')
    plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()

# ─── SHAP ────────────────────────────────────────────────────────────────────

def _shap_explainer(model, X_s, feature_names):
    """Model tipine gore en uygun shap explainer."""
    if isinstance(model, (XGBRegressor, RandomForestRegressor)):
        return shap.TreeExplainer(model), 'tree'
    try:
        return shap.Explainer(model.predict, X_s, feature_names=feature_names), 'auto'
    except Exception:
        baseline = shap.utils.sample(X_s, min(50, len(X_s)))
        return shap.KernelExplainer(model.predict, baseline), 'kernel'

def _shap_degerleri(explainer, kind, X_s):
    if kind == 'tree':
        return np.array(explainer.shap_values(X_s))
    elif kind == 'auto':
        sv = explainer(X_s)
        return sv.values if hasattr(sv, 'values') else np.array(sv)
    else:
        return np.array(explainer.shap_values(X_s, silent=True))

def ciz_shap_bar(model, X_s, names, isim, yol):
    """SHAP bar — native shap.plots.bar."""
    try:
        explainer, kind = _shap_explainer(model, X_s, names)
        sv_arr = _shap_degerleri(explainer, kind, X_s)
        explanation = shap.Explanation(values=sv_arr, data=X_s, feature_names=names)
        plt.figure(figsize=(9, max(4, len(names)*0.4)))
        shap.plots.bar(explanation, show=False, max_display=len(names))
        plt.title(f'{isim} — SHAP Bar (mean |SHAP|)\nExplainer: {kind}')
        plt.tight_layout(); plt.savefig(yol, dpi=150, bbox_inches='tight'); plt.close('all')
    except Exception as e:
        print(f"    [SHAP bar atlandi: {e}]")

def ciz_shap_beeswarm(model, X_s, names, isim, yol):
    """SHAP beeswarm — native shap.plots.beeswarm (baloncuklu)."""
    try:
        explainer, kind = _shap_explainer(model, X_s, names)
        sv_arr = _shap_degerleri(explainer, kind, X_s)
        explanation = shap.Explanation(values=sv_arr, data=X_s, feature_names=names)
        plt.figure(figsize=(10, max(5, len(names)*0.4)))
        shap.plots.beeswarm(explanation, show=False, max_display=len(names))
        plt.title(f'{isim} — SHAP Beeswarm (her nokta=bir yil)\nExplainer: {kind}')
        plt.tight_layout(); plt.savefig(yol, dpi=150, bbox_inches='tight'); plt.close('all')
    except Exception as e:
        print(f"    [SHAP beeswarm atlandi: {e}]")

def ciz_forecast_karsilastirma(fc_yillar, tahminler, yol):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(fc_yillar)); w = 0.25
    renkler = ['steelblue', 'seagreen', 'tomato']
    for i, (isim, deg) in enumerate(tahminler.items()):
        ax.bar(x+i*w, deg, w, label=isim, color=renkler[i%len(renkler)], edgecolor='k')
    ax.set_xticks(x+w); ax.set_xticklabels([str(y) for y in fc_yillar])
    ax.set_xlabel('Yil'); ax.set_ylabel('CO2 (milyon ton)')
    ax.set_title('2026-2030 Karbon Ayak Izi Tahmini — Model Karsilastirmasi')
    ax.legend(); plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# CHECKPOINT SISTEMI
# ═════════════════════════════════════════════════════════════════════════════

def checkpoint_kaydet(metrik_listesi, tahminler_dict, cv_detay):
    """Atomic CSV yazimi: once .tmp, sonra rename."""
    if metrik_listesi:
        df = pd.DataFrame(metrik_listesi)
        tmp = METRIK_CSV + '.tmp'
        df.to_csv(tmp, index=False)
        os.replace(tmp, METRIK_CSV)
    if tahminler_dict:
        rows = {'year': list(range(FC_ILK, FC_SON+1))}
        for isim, fc in tahminler_dict.items():
            rows[f'{isim}_MtCO2'] = [round(fc[y], 2) for y in range(FC_ILK, FC_SON+1)]
        df = pd.DataFrame(rows)
        tmp = TAHMIN_CSV + '.tmp'
        df.to_csv(tmp, index=False)
        os.replace(tmp, TAHMIN_CSV)
    if cv_detay:
        df = pd.DataFrame(cv_detay)
        tmp = CV_CSV + '.tmp'
        df.to_csv(tmp, index=False)
        os.replace(tmp, CV_CSV)

def checkpoint_yukle():
    """Onceden tamamlanmis modelleri tespit et."""
    tamamlanmis = set()
    if os.path.exists(MODELLER_DIR):
        for fname in os.listdir(MODELLER_DIR):
            if fname.startswith('asama3_') and (fname.endswith('.pkl') or fname.endswith('.keras')):
                if 'rf.pkl' in fname:    tamamlanmis.add('RF')
                elif 'xgboost.pkl' in fname: tamamlanmis.add('XGBoost')
                elif 'lstm.keras' in fname:  tamamlanmis.add('LSTM')
    return tamamlanmis

# ═════════════════════════════════════════════════════════════════════════════
# ANA SCRIPT
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("ASAMA 3 — KARBON AYAK IZI TAHMINI (Walk-Forward CV)")
print("=" * 70)

# Leakage assertion
print("\n[LEAKAGE GUVENCESI]")
_leakage_kontrol()
print(f"  ✓ Hedef: {HEDEF} (milyon ton CO2)")
print(f"  ✓ Yasak liste: {len(YASAK_LEAKAGE)} feature engellenir")
print(f"  ✓ co2 lag'leri yalnizca t-1, t-2 (gecmis)")
print(f"  ✓ Forecast: model kendi co2 tahminini lag'ler (iteratif)")
print(f"  ✓ Walk-forward CV + fold ici scaler fit")
print()

# ═════════════════════════════════════════════════════════════════════════════
# VERİ YÜKLEME
# ═════════════════════════════════════════════════════════════════════════════

print("Veri yukleniyor...")

# CO2 verisi (owid)
owid_co2 = pd.read_csv(os.path.join(HAM_DIR, 'owid_co2_emissions.csv'), comment='#')
owid_co2 = owid_co2.rename(columns={'Country':'country','Year':'year'})
tr_co2 = owid_co2[owid_co2['country'] == ULKE][['year', 'co2']].dropna().sort_values('year')

# Feature verileri
train_all = pd.read_csv(os.path.join(ISLENMIS_DIR, 'train.csv'))
val_all   = pd.read_csv(os.path.join(ISLENMIS_DIR, 'val.csv'))
test_all  = pd.read_csv(os.path.join(ISLENMIS_DIR, 'test.csv'))
turkey_feat = pd.concat([
    train_all[train_all['country'] == ULKE],
    val_all[val_all['country'] == ULKE],
    test_all[test_all['country'] == ULKE],
]).sort_values('year').reset_index(drop=True)

# CO2 + featurelar birlestir
turkey_df = tr_co2.merge(turkey_feat[['year'] + [c for c in FEATURE_COLS if c in turkey_feat.columns]],
                          on='year', how='inner').sort_values('year').reset_index(drop=True)

print(f"  {ULKE} CO2 verisi: {len(turkey_df)} yil ({turkey_df['year'].min()}-{turkey_df['year'].max()})")
print(f"  CO2 araligi: {turkey_df['co2'].min():.1f} - {turkey_df['co2'].max():.1f} milyon ton")

# Kullanilabilir featurelar (en az 5 dolu yil)
KULLANILAN = []
print(f"\n  Feature doluluk + correlation:")
for f in FEATURE_COLS:
    if f in turkey_df.columns:
        dolu = turkey_df[f].notna().sum()
        if dolu >= 5:
            KULLANILAN.append(f)
            korr = turkey_df['co2'].corr(turkey_df[f])
            print(f"    {f:<35}: {dolu:>2} dolu | korr={korr:>+.4f}")
        else:
            print(f"    {f:<35}: {dolu:>2} dolu → ATLA (yetersiz)")

INPUT_COLS = LAG_COLS + KULLANILAN
print(f"\n  Toplam girdi: {len(LAG_COLS)} lag + {len(KULLANILAN)} feature = {len(INPUT_COLS)}")

# NaN doldur
turkey_df[KULLANILAN] = turkey_df[KULLANILAN].ffill().bfill()

# Lag ekle
for k in range(1, LAG_SAYISI+1):
    turkey_df[f'{HEDEF}_lag{k}'] = turkey_df[HEDEF].shift(k)
turkey_df = turkey_df.dropna(subset=INPUT_COLS + [HEDEF]).reset_index(drop=True)

tum_yillar = turkey_df['year'].values
tum_X      = turkey_df[INPUT_COLS].values
tum_y      = turkey_df[HEDEF].values

print(f"  Kullanilabilir: {len(turkey_df)} yil ({tum_yillar[0]:.0f}-{tum_yillar[-1]:.0f})")

# Walk-forward fold'lari
cv_folds = []
for test_idx in range(MIN_EGITIM_YIL, len(tum_yillar)):
    cv_folds.append((list(range(test_idx)), test_idx))
cv_yillar = [int(tum_yillar[t]) for _, t in cv_folds]
print(f"  Walk-Forward CV: {len(cv_folds)} fold ({cv_yillar[0]}-{cv_yillar[-1]})")

# Forecast featurelari (2026-2030)
asama1_df = pd.read_csv(os.path.join(ASAMA1_DIR, 'feature_tahminleri_2021_2030.csv'))
asama1_tr = asama1_df[asama1_df['country'] == ULKE].sort_values('year')

asama2_fc = pd.read_csv(os.path.join(ASAMA2_DIR, 'elektrik_tahmin_2026_2030.csv'))
if 'LSTM_TWh' in asama2_fc.columns:
    elec_col = 'LSTM_TWh'
elif 'XGBoost_TWh' in asama2_fc.columns:
    elec_col = 'XGBoost_TWh'
else:
    elec_col = [c for c in asama2_fc.columns if 'TWh' in c][0]
print(f"  Forecast elektrik kaynagi: Aşama 2 ({elec_col})")

# Forecast feature matrisi
fc_rows = []
for _, row in asama1_tr[asama1_tr['year'] >= FC_ILK].iterrows():
    yil = int(row['year'])
    r = {'year': yil}
    for f in KULLANILAN:
        if f == 'electricity_demand':
            elec_row = asama2_fc[asama2_fc['year'] == yil]
            r[f] = float(elec_row[elec_col].values[0]) if len(elec_row) > 0 else np.nan
        elif f in row.index:
            r[f] = float(row[f]) if not pd.isna(row[f]) else np.nan
        else:
            r[f] = np.nan
    fc_rows.append(r)
fc_asama = pd.DataFrame(fc_rows)

# Eksik featurelar icin son bilinen
for f in KULLANILAN:
    if f in fc_asama.columns and fc_asama[f].isna().any():
        son = turkey_df[f].dropna().iloc[-1] if f in turkey_df.columns else 0.0
        fc_asama[f] = fc_asama[f].fillna(son)

print(f"  Forecast featurelari: {len(fc_asama)} yil")

# Checkpoint kontrol
print("\nCheckpoint kontrolu...")
TAMAMLANMIS = checkpoint_yukle()
if TAMAMLANMIS:
    print(f"  [RESUME] {len(TAMAMLANMIS)} model mevcut: {sorted(TAMAMLANMIS)}")
else:
    print(f"  [TEMIZ BASLANGIC]")

# ═════════════════════════════════════════════════════════════════════════════
# YARDIMCI: WALK-FORWARD + ITERATIF FORECAST
# ═════════════════════════════════════════════════════════════════════════════

tscv = TimeSeriesSplit(n_splits=min(5, len(tum_yillar)-2))

def optuna_rf(trial):
    params = dict(
        n_estimators=trial.suggest_int('n_estimators', 50, 400),
        max_depth=trial.suggest_int('max_depth', 2, 10),
        min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 5),
        max_features=trial.suggest_categorical('max_features', ['sqrt','log2', 0.7]),
        random_state=42, n_jobs=-1)
    rmses = []
    for tr_i, vl_i in tscv.split(tum_X):
        sc = MinMaxScaler()  # LEAKAGE GUARD: fold ici fit
        m = RandomForestRegressor(**params)
        m.fit(sc.fit_transform(tum_X[tr_i]), tum_y[tr_i])
        rmses.append(np.sqrt(mean_squared_error(tum_y[vl_i], m.predict(sc.transform(tum_X[vl_i])))))
    return np.mean(rmses)

def optuna_xgb(trial):
    params = dict(
        n_estimators=trial.suggest_int('n_estimators', 50, 300),
        max_depth=trial.suggest_int('max_depth', 2, 5),
        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        subsample=trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
        reg_alpha=trial.suggest_float('reg_alpha', 0.0, 2.0),
        reg_lambda=trial.suggest_float('reg_lambda', 0.5, 3.0),
        random_state=42, verbosity=0)
    rmses = []
    for tr_i, vl_i in tscv.split(tum_X):
        sc = MinMaxScaler()  # LEAKAGE GUARD
        m = XGBRegressor(**params)
        m.fit(sc.fit_transform(tum_X[tr_i]), tum_y[tr_i],
              eval_set=[(sc.transform(tum_X[vl_i]), tum_y[vl_i])], verbose=False)
        rmses.append(np.sqrt(mean_squared_error(tum_y[vl_i], m.predict(sc.transform(tum_X[vl_i])))))
    return np.mean(rmses)

def walk_forward(model_factory):
    """LEAKAGE GUARD: her fold'da scaler sadece train'e fit."""
    preds, trues = [], []
    for train_idx, test_idx in cv_folds:
        sc = MinMaxScaler()
        m = model_factory()
        m.fit(sc.fit_transform(tum_X[train_idx]), tum_y[train_idx])
        preds.append(m.predict(sc.transform(tum_X[test_idx:test_idx+1]))[0])
        trues.append(tum_y[test_idx])
    return np.array(trues), np.array(preds)

def walk_forward_lstm(lstm_factory):
    preds, trues = [], []
    for train_idx, test_idx in cv_folds:
        X_tr, y_tr = tum_X[train_idx], tum_y[train_idx]
        X_te = tum_X[test_idx:test_idx+1]
        sc_f, sc_h = MinMaxScaler(), MinMaxScaler()
        X_tr_s = sc_f.fit_transform(X_tr)
        X_te_s = sc_f.transform(X_te)
        y_tr_s = sc_h.fit_transform(y_tr.reshape(-1,1)).ravel()
        nf = X_tr_s.shape[1]
        m = lstm_factory(nf)
        es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        m.fit(X_tr_s.reshape(-1,1,nf), y_tr_s, epochs=100, batch_size=4,
              callbacks=[es], verbose=0)
        p_s = m.predict(X_te_s.reshape(1,1,nf), verbose=0).ravel()[0]
        preds.append(sc_h.inverse_transform([[p_s]])[0][0])
        trues.append(tum_y[test_idx])
    return np.array(trues), np.array(preds)

def iteratif_forecast(model, feat_scaler, hedef_scaler=None, is_lstm=False):
    """ITERATIF: model kendi co2 tahminini lag olarak kullanir
    (gercek 2026-2030 co2 degerlerini ASLA kullanmaz)."""
    co2_hist = dict(zip(turkey_df['year'], turkey_df[HEDEF]))
    fc = {}
    # Tahminin tarihsel araligi (asiri patlamayi engelle)
    co2_min = float(np.nanmin(turkey_df[HEDEF].values))
    co2_max = float(np.nanmax(turkey_df[HEDEF].values))
    co2_range = co2_max - co2_min
    alt = co2_min - 5 * co2_range
    ust = co2_max + 5 * co2_range

    for _, row in fc_asama.iterrows():
        yil = int(row['year'])
        lags = []
        for k in range(1, LAG_SAYISI+1):
            v = co2_hist.get(yil-k, np.nan)
            if not np.isfinite(v):
                bilinen = sorted([val for yr, val in co2_hist.items() if yr < yil and np.isfinite(val)])
                v = bilinen[-1] if bilinen else 0.0
            v = float(np.clip(v, alt, ust))
            lags.append(v)
        feats = [float(row.get(f, 0.0)) if not pd.isna(row.get(f, np.nan)) else 0.0
                 for f in KULLANILAN]
        x = np.nan_to_num(np.array(lags + feats).reshape(1,-1),
                          nan=0.0, posinf=1e6, neginf=-1e6)
        x_s = np.nan_to_num(feat_scaler.transform(x), nan=0.0, posinf=1.0, neginf=0.0)
        try:
            if is_lstm:
                p_s = model.predict(x_s.reshape(1,1,-1), verbose=0).ravel()[0]
                pred = hedef_scaler.inverse_transform([[p_s]])[0][0]
            else:
                pred = model.predict(x_s)[0]
            if not np.isfinite(pred):
                pred = lags[0]
        except Exception:
            pred = lags[0]
        pred = float(np.clip(pred, alt, ust))
        co2_hist[yil] = pred
        fc[yil] = pred
    return fc

# ═════════════════════════════════════════════════════════════════════════════
# MODELLERI EGIT
# ═════════════════════════════════════════════════════════════════════════════

t_overall = time.time()
tum_cv, tum_fc, tum_metrik = {}, {}, []
cv_detay = []

# ─── MODEL 1: RANDOM FOREST ──────────────────────────────────────────────────
if 'RF' in TAMAMLANMIS:
    print("\n[1/3] RF → [SKIP] model dosyasi var")
else:
    t_rf = time.time()
    print("\n" + "="*70)
    print("  MODEL 1: RANDOM FOREST")
    print("="*70)
    print(f"  Optuna ({OPTUNA_TRIAL} trial)...")
    rf_study = optuna.create_study(direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=42))
    rf_study.optimize(optuna_rf, n_trials=OPTUNA_TRIAL)
    rf_best = rf_study.best_params
    print(f"  En iyi params: {rf_best}")

    rf_trues, rf_preds = walk_forward(lambda: RandomForestRegressor(**rf_best))
    rf_m = metrikleri_hesapla(rf_trues, rf_preds)
    print(f"  CV: R²={rf_m['r2']:.4f}  MAE={rf_m['mae']:.2f}  "
          f"RMSE={rf_m['rmse']:.2f}  MAPE={rf_m['mape']:.2f}%")
    print(f"  Hata Bandi: {hata_bandi_hesapla(rf_trues, rf_preds)}")

    sc_rf = MinMaxScaler()
    rf_final = RandomForestRegressor(**rf_best)
    rf_final.fit(sc_rf.fit_transform(tum_X), tum_y)
    joblib.dump(rf_final, os.path.join(MODELLER_DIR, 'asama3_rf.pkl'))
    joblib.dump(sc_rf, os.path.join(MODELLER_DIR, 'asama3_rf_scaler.pkl'))
    rf_fc = iteratif_forecast(rf_final, sc_rf)
    print("  Forecast 2026-2030:")
    for y, v in rf_fc.items(): print(f"    {y}: {v:.2f} Mt CO2")

    tum_cv['RF'] = (rf_trues, rf_preds)
    tum_fc['RF'] = rf_fc
    tum_metrik.append({'model': 'RF', **rf_m, 'optuna_cv_rmse': round(rf_study.best_value,4)})

    # Grafikler
    base = os.path.join(GORSELLER_DIR, 'asama3_RF')
    ciz_residual(cv_yillar, rf_trues, rf_preds, 'RF', f'{base}_residual.png')
    ciz_residual_dist(rf_trues, rf_preds, 'RF', f'{base}_residual_dist.png')
    ciz_hata_bandi(hata_bandi_hesapla(rf_trues, rf_preds), 'RF', f'{base}_hata_bandi.png')
    ciz_confusion_matrix_regresyon(rf_trues, rf_preds, 'RF', f'{base}_confusion_matrix.png')
    ciz_cv_scatter(rf_trues, rf_preds, 'RF', f'{base}_cv_scatter.png')
    ciz_learning_curve(tum_X, tum_y, lambda: RandomForestRegressor(**rf_best),
                       'RF', MIN_EGITIM_YIL, f'{base}_learning_curve.png')
    ciz_feature_importance(rf_final, INPUT_COLS, 'RF', f'{base}_feature_importance.png')
    ciz_shap_bar(rf_final, sc_rf.transform(tum_X), INPUT_COLS, 'RF', f'{base}_shap_bar.png')
    ciz_shap_beeswarm(rf_final, sc_rf.transform(tum_X), INPUT_COLS, 'RF', f'{base}_shap_beeswarm.png')

    # CV detay
    for yil, g, t in zip(cv_yillar, rf_trues, rf_preds):
        cv_detay.append({'model':'RF','year':yil,'gercek_MtCO2':round(g,2),
                         'tahmin_MtCO2':round(t,2),'hata_pct':round(abs(g-t)/g*100,2)})
    checkpoint_kaydet(tum_metrik, tum_fc, cv_detay)
    print(f"  ✓ RF tamamlandi ({format_sure(time.time()-t_rf)}) — checkpoint atildi")

# ─── MODEL 2: XGBOOST ────────────────────────────────────────────────────────
if 'XGBoost' in TAMAMLANMIS:
    print("\n[2/3] XGBoost → [SKIP] model dosyasi var")
else:
    t_xgb = time.time()
    print("\n" + "="*70)
    print("  MODEL 2: XGBOOST")
    print("="*70)
    print(f"  Optuna ({OPTUNA_TRIAL} trial)...")
    xgb_study = optuna.create_study(direction='minimize',
                  sampler=optuna.samplers.TPESampler(seed=42))
    xgb_study.optimize(optuna_xgb, n_trials=OPTUNA_TRIAL)
    xgb_best = xgb_study.best_params
    print(f"  En iyi params: {xgb_best}")

    xgb_trues, xgb_preds = walk_forward(lambda: XGBRegressor(**xgb_best, verbosity=0))
    xgb_m = metrikleri_hesapla(xgb_trues, xgb_preds)
    print(f"  CV: R²={xgb_m['r2']:.4f}  MAE={xgb_m['mae']:.2f}  "
          f"RMSE={xgb_m['rmse']:.2f}  MAPE={xgb_m['mape']:.2f}%")
    print(f"  Hata Bandi: {hata_bandi_hesapla(xgb_trues, xgb_preds)}")

    sc_xgb = MinMaxScaler()
    xgb_final = XGBRegressor(**xgb_best, verbosity=0)
    xgb_final.fit(sc_xgb.fit_transform(tum_X), tum_y)
    joblib.dump(xgb_final, os.path.join(MODELLER_DIR, 'asama3_xgboost.pkl'))
    joblib.dump(sc_xgb, os.path.join(MODELLER_DIR, 'asama3_xgboost_scaler.pkl'))
    xgb_fc = iteratif_forecast(xgb_final, sc_xgb)
    print("  Forecast 2026-2030:")
    for y, v in xgb_fc.items(): print(f"    {y}: {v:.2f} Mt CO2")

    tum_cv['XGBoost'] = (xgb_trues, xgb_preds)
    tum_fc['XGBoost'] = xgb_fc
    tum_metrik.append({'model': 'XGBoost', **xgb_m, 'optuna_cv_rmse': round(xgb_study.best_value,4)})

    base = os.path.join(GORSELLER_DIR, 'asama3_XGBoost')
    ciz_residual(cv_yillar, xgb_trues, xgb_preds, 'XGBoost', f'{base}_residual.png')
    ciz_residual_dist(xgb_trues, xgb_preds, 'XGBoost', f'{base}_residual_dist.png')
    ciz_hata_bandi(hata_bandi_hesapla(xgb_trues, xgb_preds), 'XGBoost', f'{base}_hata_bandi.png')
    ciz_confusion_matrix_regresyon(xgb_trues, xgb_preds, 'XGBoost', f'{base}_confusion_matrix.png')
    ciz_cv_scatter(xgb_trues, xgb_preds, 'XGBoost', f'{base}_cv_scatter.png')
    ciz_learning_curve(tum_X, tum_y, lambda: XGBRegressor(**xgb_best, verbosity=0),
                       'XGBoost', MIN_EGITIM_YIL, f'{base}_learning_curve.png')
    ciz_feature_importance(xgb_final, INPUT_COLS, 'XGBoost', f'{base}_feature_importance.png')
    ciz_shap_bar(xgb_final, sc_xgb.transform(tum_X), INPUT_COLS, 'XGBoost', f'{base}_shap_bar.png')
    ciz_shap_beeswarm(xgb_final, sc_xgb.transform(tum_X), INPUT_COLS, 'XGBoost', f'{base}_shap_beeswarm.png')

    for yil, g, t in zip(cv_yillar, xgb_trues, xgb_preds):
        cv_detay.append({'model':'XGBoost','year':yil,'gercek_MtCO2':round(g,2),
                         'tahmin_MtCO2':round(t,2),'hata_pct':round(abs(g-t)/g*100,2)})
    checkpoint_kaydet(tum_metrik, tum_fc, cv_detay)
    print(f"  ✓ XGBoost tamamlandi ({format_sure(time.time()-t_xgb)}) — checkpoint atildi")

# ─── MODEL 3: LSTM ───────────────────────────────────────────────────────────
if 'LSTM' in TAMAMLANMIS:
    print("\n[3/3] LSTM → [SKIP] model dosyasi var")
elif TF_MEVCUT:
    t_lstm = time.time()
    print("\n" + "="*70)
    print("  MODEL 3: LSTM")
    print("="*70)
    n_feat = tum_X.shape[1]

    def lstm_factory(nf, units1=64, units2=32, dropout=0.2):
        m = Sequential([
            LSTM_Layer(units1, return_sequences=True, input_shape=(1, nf)),
            Dropout(dropout),
            LSTM_Layer(units2, return_sequences=False),
            Dropout(dropout),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        m.compile(optimizer='adam', loss='mse')
        return m

    def optuna_lstm(trial):
        u1 = trial.suggest_categorical('units1', [32, 64, 128])
        u2 = trial.suggest_categorical('units2', [16, 32, 64])
        dr = trial.suggest_float('dropout', 0.1, 0.4)
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        ba = trial.suggest_categorical('batch', [4, 8])
        rmses = []
        for tr_i, vl_i in tscv.split(tum_X):
            sf, sh = MinMaxScaler(), MinMaxScaler()
            Xtr = sf.fit_transform(tum_X[tr_i]); Xvl = sf.transform(tum_X[vl_i])
            ytr = sh.fit_transform(tum_y[tr_i].reshape(-1,1)).ravel()
            m = lstm_factory(n_feat, u1, u2, dr)
            m.optimizer.learning_rate.assign(lr)
            yvl_s = sh.transform(tum_y[vl_i].reshape(-1,1)).ravel()
            es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            m.fit(Xtr.reshape(-1,1,n_feat), ytr,
                  validation_data=(Xvl.reshape(-1,1,n_feat), yvl_s),
                  epochs=100, batch_size=ba, callbacks=[es], verbose=0)
            p = sh.inverse_transform(m.predict(Xvl.reshape(-1,1,n_feat), verbose=0).reshape(-1,1)).ravel()
            rmses.append(np.sqrt(mean_squared_error(tum_y[vl_i], p)))
        return np.mean(rmses)

    print(f"  Optuna ({OPTUNA_TRIAL} trial)...")
    lstm_study = optuna.create_study(direction='minimize',
                   sampler=optuna.samplers.TPESampler(seed=42))
    lstm_study.optimize(optuna_lstm, n_trials=OPTUNA_TRIAL)
    lp = lstm_study.best_params
    print(f"  En iyi params: {lp}")

    lstm_trues, lstm_preds = walk_forward_lstm(
        lambda nf: lstm_factory(nf, lp['units1'], lp['units2'], lp['dropout']))
    lstm_m = metrikleri_hesapla(lstm_trues, lstm_preds)
    print(f"  CV: R²={lstm_m['r2']:.4f}  MAE={lstm_m['mae']:.2f}  "
          f"RMSE={lstm_m['rmse']:.2f}  MAPE={lstm_m['mape']:.2f}%")
    print(f"  Hata Bandi: {hata_bandi_hesapla(lstm_trues, lstm_preds)}")

    # Final training
    sf_l, sh_l = MinMaxScaler(), MinMaxScaler()
    Xa = sf_l.fit_transform(tum_X)
    ya = sh_l.fit_transform(tum_y.reshape(-1,1)).ravel()
    lstm_final = lstm_factory(n_feat, lp['units1'], lp['units2'], lp['dropout'])
    es = EarlyStopping(monitor='loss', patience=LSTM_PATIENCE, restore_best_weights=True)
    lstm_final.fit(Xa.reshape(-1,1,n_feat), ya, epochs=LSTM_EPOCHS, batch_size=lp['batch'],
                    callbacks=[es], verbose=0)
    lstm_final.save(os.path.join(MODELLER_DIR, 'asama3_lstm.keras'))
    joblib.dump(sf_l, os.path.join(MODELLER_DIR, 'asama3_lstm_feat_scaler.pkl'))
    joblib.dump(sh_l, os.path.join(MODELLER_DIR, 'asama3_lstm_hedef_scaler.pkl'))

    lstm_fc = iteratif_forecast(lstm_final, sf_l, sh_l, is_lstm=True)
    print("  Forecast 2026-2030:")
    for y, v in lstm_fc.items(): print(f"    {y}: {v:.2f} Mt CO2")

    tum_cv['LSTM'] = (lstm_trues, lstm_preds)
    tum_fc['LSTM'] = lstm_fc
    tum_metrik.append({'model': 'LSTM', **lstm_m, 'optuna_cv_rmse': round(lstm_study.best_value,4)})

    base = os.path.join(GORSELLER_DIR, 'asama3_LSTM')
    ciz_residual(cv_yillar, lstm_trues, lstm_preds, 'LSTM', f'{base}_residual.png')
    ciz_residual_dist(lstm_trues, lstm_preds, 'LSTM', f'{base}_residual_dist.png')
    ciz_hata_bandi(hata_bandi_hesapla(lstm_trues, lstm_preds), 'LSTM', f'{base}_hata_bandi.png')
    ciz_confusion_matrix_regresyon(lstm_trues, lstm_preds, 'LSTM', f'{base}_confusion_matrix.png')
    ciz_cv_scatter(lstm_trues, lstm_preds, 'LSTM', f'{base}_cv_scatter.png')
    # SHAP — LSTM icin KernelExplainer (yavas ama calisir)
    ciz_shap_bar(lstm_final, sf_l.transform(tum_X), INPUT_COLS, 'LSTM', f'{base}_shap_bar.png')
    ciz_shap_beeswarm(lstm_final, sf_l.transform(tum_X), INPUT_COLS, 'LSTM', f'{base}_shap_beeswarm.png')

    for yil, g, t in zip(cv_yillar, lstm_trues, lstm_preds):
        cv_detay.append({'model':'LSTM','year':yil,'gercek_MtCO2':round(g,2),
                         'tahmin_MtCO2':round(t,2),'hata_pct':round(abs(g-t)/g*100,2)})
    checkpoint_kaydet(tum_metrik, tum_fc, cv_detay)
    print(f"  ✓ LSTM tamamlandi ({format_sure(time.time()-t_lstm)}) — checkpoint atildi")
else:
    print("\n[3/3] LSTM atlandi (TensorFlow yuklu degil)")

# ═════════════════════════════════════════════════════════════════════════════
# FINAL: Tum modellerin karsilastirma grafigi + ozet
# ═════════════════════════════════════════════════════════════════════════════

if tum_cv:
    cv_dict = {isim: preds for isim, (_, preds) in tum_cv.items()}
    fc_dict = {isim: [fc[y] for y in range(FC_ILK, FC_SON+1)] for isim, fc in tum_fc.items()}
    fc_yillar = list(range(FC_ILK, FC_SON+1))

    ciz_walkforward(turkey_df['year'].values, turkey_df[HEDEF].values,
                    cv_yillar, cv_dict, fc_yillar, fc_dict,
                    os.path.join(GORSELLER_DIR, 'asama3_tum_modeller.png'))
    ciz_forecast_karsilastirma(fc_yillar, fc_dict,
        os.path.join(GORSELLER_DIR, 'asama3_forecast_karsilastirma.png'))

# Final flush
checkpoint_kaydet(tum_metrik, tum_fc, cv_detay)

toplam_sure = time.time() - t_overall
print("\n" + "=" * 70)
print(f"ASAMA 3 TAMAMLANDI ({format_sure(toplam_sure)})")
print("=" * 70)

# Ozet
if os.path.exists(METRIK_CSV):
    metrik_df = pd.read_csv(METRIK_CSV)
    pd.set_option('display.width', 140)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print("\nMetrikler (Walk-Forward CV):")
    cols = ['model','r2','mae','rmse','mape','optuna_cv_rmse']
    cols = [c for c in cols if c in metrik_df.columns]
    print(metrik_df[cols].to_string(index=False))

if os.path.exists(TAHMIN_CSV):
    fc_df = pd.read_csv(TAHMIN_CSV)
    print("\nForecast (2026-2030):")
    print(fc_df.to_string(index=False))

print(f"\nGirdiler ({len(INPUT_COLS)}): {INPUT_COLS}")
print(f"Walk-Forward CV: {len(cv_folds)} out-of-sample tahmin")
print("\nKaydedilen dosyalar:")
print(f"  ciktilar/karbon_tahmin_2026_2030.csv")
print(f"  ciktilar/asama3_metrikler.csv")
print(f"  ciktilar/asama3_cv_detay.csv")
print(f"  modeller/asama3_rf.pkl, asama3_xgboost.pkl, asama3_lstm.keras")
print(f"  sonuclar/gorseller/asama3_* (her model icin 9 grafik)")
