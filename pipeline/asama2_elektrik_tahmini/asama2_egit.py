"""
AŞAMA 2 — Elektrik Tüketimi Tahmini (Walk-Forward CV + Optuna + SHAP)
----------------------------------------------------------------------
Türkiye'nin 2026-2030 elektrik tüketimini (TWh) tahmin eder.

DEĞERLENDİRME: Walk-Forward Expanding Window Cross-Validation
  - Min 15 yıl eğitim → sonraki yılı tahmin et → pencereyi genişlet
  - ~19 out-of-sample tahmin → anlamlı R² ve MAPE
  - Zaman serisi sırası ASLA bozulmaz

GİRDİLER:
  - electricity_demand lag_1, lag_2  (geçmiş elektrik)
  - Tüm 14 feature (temel + ek parametreler)
  → Forecast (2026-2030) için AŞAMA 1 TAHMİNLERİ kullanılır

MODELLER: Random Forest | XGBoost | LSTM
"""

import os, warnings
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
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score
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

# ─── Sabitler ─────────────────────────────────────────────────────────────────
FC_ILK = 2026;  FC_SON = 2030
LAG_SAYISI     = 2
MIN_EGITIM_YIL = 15          # walk-forward icin min egitim penceresi
OPTUNA_TRIAL   = 40
LSTM_EPOCHS    = 200
LSTM_PATIENCE  = 20

ULKE  = 'Turkey'
HEDEF = 'electricity_demand'

FEATURE_COLS = [
    'Population', 'GDP_Per_Capita', 'GDP_Growth',
    'renewables_share_elec', 'Urbanization_Rate',
    'Industry_Value_Added_Pct_GDP', 'Population_Density',
    'fossil_fuel_consumption', 'nuclear_consumption',
    'coal_co2', 'gas_co2', 'co2_per_capita',
    'Electricity_Loss_Pct', 'temperature_change_from_ghg',
]
LAG_COLS = [f'{HEDEF}_lag{k}' for k in range(1, LAG_SAYISI+1)]

# ─── Yollar ───────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR  = os.path.dirname(SCRIPT_DIR)
ISLENMIS_DIR  = os.path.join(PIPELINE_DIR, 'veri', 'islenmis')
ASAMA1_DIR    = os.path.join(PIPELINE_DIR, 'asama1_feature_tahmini', 'ciktilar')
MODELLER_DIR  = os.path.join(SCRIPT_DIR, 'modeller')
CIKTILAR_DIR  = os.path.join(SCRIPT_DIR, 'ciktilar')
GORSELLER_DIR = os.path.join(PIPELINE_DIR, 'sonuclar', 'gorseller')
for d in [MODELLER_DIR, CIKTILAR_DIR, GORSELLER_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Yardımcı ────────────────────────────────────────────────────────────────

def mape(y_true, y_pred):
    t, p = np.array(y_true, float), np.array(y_pred, float)
    mask = t != 0
    return np.mean(np.abs((t[mask] - p[mask]) / t[mask])) * 100 if mask.sum() else np.nan

def metrikler(y_true, y_pred):
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    m     = mape(y_true, y_pred)
    r2    = r2_score(y_true, y_pred) if len(y_true) > 2 else np.nan
    rmsle = np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred))**2))
    return dict(rmse=round(rmse,4), mae=round(mae,4), mape=round(m,4),
                r2=round(r2,4), rmsle=round(rmsle,4))

def hata_bandi(y_true, y_pred):
    pct = np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true)) * 100
    return {
        'Mukemmel (<%5)' : int(np.sum(pct < 5)),
        'Iyi (%5-10)'    : int(np.sum((pct>=5)  & (pct<10))),
        'Orta (%10-20)'  : int(np.sum((pct>=10) & (pct<20))),
        'Zayif (>%20)'   : int(np.sum(pct >= 20)),
    }

# ─── Grafikler ────────────────────────────────────────────────────────────────

def ciz_walkforward(yillar_g, y_g, cv_yillar, cv_tahminler_dict, fc_yillar, fc_dict, yol):
    """Gercek + walk-forward CV tahminleri + forecast"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(yillar_g, y_g, 'k-o', ms=4, lw=2, label='Gercek', zorder=5)
    renkler = ['steelblue', 'seagreen', 'tomato']
    for (isim, preds), renk in zip(cv_tahminler_dict.items(), renkler):
        ax.plot(cv_yillar, preds, '--s', color=renk, ms=5, label=f'{isim} (CV)')
    for (isim, preds), renk in zip(fc_dict.items(), renkler):
        ax.plot(fc_yillar, preds, ':^', color=renk, ms=5, alpha=0.7, label=f'{isim} (Forecast)')
    ax.axvspan(cv_yillar[0], cv_yillar[-1]+0.9, alpha=0.08, color='blue', label='Walk-Forward CV')
    ax.axvspan(FC_ILK, FC_SON+0.9, alpha=0.08, color='red', label='Forecast')
    ax.set_title(f'Asama 2 — Elektrik Tuketimi: Gercek vs Modeller\n'
                 f'Walk-Forward CV ({len(cv_yillar)} out-of-sample tahmin)')
    ax.set_xlabel('Yil'); ax.set_ylabel('TWh'); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()

def ciz_residual(cv_yillar, y_gercek, y_tahmin, model_isim, yol):
    res = np.array(y_gercek) - np.array(y_tahmin)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(cv_yillar, res, color='steelblue', edgecolor='k')
    axes[0].axhline(0, color='red', lw=1.5)
    axes[0].set_xlabel('Yil'); axes[0].set_ylabel('Residual (TWh)')
    axes[0].set_title(f'{model_isim} — Residual (Walk-Forward CV)')
    axes[1].scatter(y_tahmin, res, color='steelblue', edgecolors='k', s=60)
    axes[1].axhline(0, color='red', lw=1.5)
    axes[1].set_xlabel('Tahmin (TWh)'); axes[1].set_ylabel('Residual (TWh)')
    axes[1].set_title(f'{model_isim} — Residual vs Tahmin')
    plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()

def ciz_hata_bandi(bandlar, model_isim, yol):
    fig, ax = plt.subplots(figsize=(8, 4))
    renkler = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
    bars = ax.bar(list(bandlar.keys()), list(bandlar.values()), color=renkler, edgecolor='k')
    for bar, val in zip(bars, bandlar.values()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                str(val), ha='center', va='bottom', fontweight='bold')
    ax.set_title(f'{model_isim} — Hata Bandi (Walk-Forward CV)')
    ax.set_ylabel('Tahmin Sayisi')
    ax.set_ylim(0, max(bandlar.values())+1.5 if max(bandlar.values())>0 else 2)
    plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()

def ciz_shap(model, X_s, feature_names, model_isim, yol):
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_s)
        mean_abs  = np.abs(shap_vals).mean(axis=0)
        idx = np.argsort(mean_abs)
        fig, ax = plt.subplots(figsize=(9, max(5, len(feature_names)*0.35)))
        ax.barh([feature_names[i] for i in idx], mean_abs[idx],
                color='steelblue', edgecolor='k')
        for i, val in enumerate(mean_abs[idx]):
            ax.text(val + mean_abs.max()*0.01, i, f'{val:.3f}', va='center', fontsize=8)
        ax.set_xlabel('Ortalama |SHAP Degeri|')
        ax.set_title(f'{model_isim} — SHAP Feature Importance\n'
                     f'(Hangi feature elektrik tahminini ne kadar etkiliyor?)')
        plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()
    except Exception as e:
        print(f"    [SHAP cizilmedi: {e}]")

def ciz_forecast_karsilastirma(fc_yillar, tahminler_dict, yol):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(fc_yillar)); w = 0.25
    renkler = ['steelblue', 'seagreen', 'tomato']
    for i, (isim, degerler) in enumerate(tahminler_dict.items()):
        ax.bar(x+i*w, degerler, w, label=isim, color=renkler[i%len(renkler)], edgecolor='k')
    ax.set_xticks(x+w); ax.set_xticklabels([str(y) for y in fc_yillar])
    ax.set_xlabel('Yil'); ax.set_ylabel('Tahmin (TWh)')
    ax.set_title('2026-2030 Elektrik Tahmini — Model Karsilastirmasi')
    ax.legend(); plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()

def ciz_cv_scatter(trues, preds, model_isim, yol):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(trues, preds, color='steelblue', edgecolors='k', s=60, zorder=5)
    mn, mx = min(trues.min(), preds.min()), max(trues.max(), preds.max())
    margin = (mx - mn) * 0.05
    ax.plot([mn-margin, mx+margin], [mn-margin, mx+margin], 'r--', lw=1.5, label='Ideal (y=x)')
    r2 = r2_score(trues, preds)
    ax.set_xlabel('Gercek (TWh)'); ax.set_ylabel('Tahmin (TWh)')
    ax.set_title(f'{model_isim} — Walk-Forward CV\nR² = {r2:.4f}')
    ax.legend(); plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()

def ciz_learning_curve(tum_X, tum_y, model_factory, model_isim, min_train, yol):
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
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sizes, rmses, 'b-o', ms=4)
    ax.set_xlabel('Egitim Seti Boyutu (yil)'); ax.set_ylabel('Mutlak Hata (TWh)')
    ax.set_title(f'{model_isim} — Learning Curve\n(Egitim boyutu arttikca hata nasil degisiyor?)')
    plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()

def ciz_feature_importance(model, feature_names, model_isim, yol):
    try:
        imp = model.feature_importances_
        idx = np.argsort(imp)
        fig, ax = plt.subplots(figsize=(9, max(5, len(feature_names)*0.35)))
        ax.barh([feature_names[i] for i in idx], imp[idx], color='coral', edgecolor='k')
        for i, val in enumerate(imp[idx]):
            ax.text(val + imp.max()*0.01, i, f'{val:.3f}', va='center', fontsize=8)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'{model_isim} — Feature Importance (Built-in)\n'
                     f'(Modelin ogrenme sirasinda verdigi agirliklar)')
        plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()
    except Exception as e:
        print(f"    [Feature importance cizilmedi: {e}]")

# ═════════════════════════════════════════════════════════════════════════════
# VERİ HAZIRLAMA
# ═════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("ASAMA 2 — ELEKTRIK TUKETIMI TAHMINI")
print("  Walk-Forward Expanding Window CV")
print("=" * 65)

# Turkey verisini yukle
train_all = pd.read_csv(os.path.join(ISLENMIS_DIR, 'train.csv'))
val_all   = pd.read_csv(os.path.join(ISLENMIS_DIR, 'val.csv'))
test_all  = pd.read_csv(os.path.join(ISLENMIS_DIR, 'test.csv'))
asama1_df = pd.read_csv(os.path.join(ASAMA1_DIR,   'feature_tahminleri_2021_2030.csv'))

# Tum Turkey verisi birlestir (1990-2025)
turkey_df = pd.concat([
    train_all[train_all['country'] == ULKE],
    val_all[val_all['country'] == ULKE],
    test_all[test_all['country'] == ULKE],
]).sort_values('year').reset_index(drop=True)

print(f"  {ULKE}: {len(turkey_df)} yil ({turkey_df['year'].min()}-{turkey_df['year'].max()})")

# Kullanilabilir featurelar (en az 5 dolu yil)
KULLANILAN = []
for f in FEATURE_COLS:
    if f in turkey_df.columns:
        dolu = turkey_df[f].notna().sum()
        if dolu >= 5:
            KULLANILAN.append(f)
            print(f"    {f:<35}: {dolu}/{len(turkey_df)} dolu")
        else:
            print(f"    {f:<35}: {dolu}/{len(turkey_df)} -> ATLA")

INPUT_COLS = LAG_COLS + KULLANILAN
print(f"\n  Toplam girdi: {len(LAG_COLS)} lag + {len(KULLANILAN)} feature = {len(INPUT_COLS)}")

# NaN doldur (feature icin ffill+bfill)
turkey_df[KULLANILAN] = turkey_df[KULLANILAN].ffill().bfill()

# Lag ekle
for k in range(1, LAG_SAYISI+1):
    turkey_df[f'{HEDEF}_lag{k}'] = turkey_df[HEDEF].shift(k)

# Lag NaN olan satirlari at (ilk 2 yil)
turkey_df = turkey_df.dropna(subset=INPUT_COLS + [HEDEF]).reset_index(drop=True)
tum_yillar = turkey_df['year'].values
tum_X      = turkey_df[INPUT_COLS].values
tum_y      = turkey_df[HEDEF].values

print(f"  Kullanilabilir veri: {len(turkey_df)} yil ({tum_yillar[0]:.0f}-{tum_yillar[-1]:.0f})")

# Walk-forward split noktalarini belirle
# Min 15 yil egitim, sonra her yil 1 tahmin
ilk_test_idx = MIN_EGITIM_YIL  # index 15 = 16. yil
cv_folds = []
for test_idx in range(ilk_test_idx, len(tum_yillar)):
    train_idx = list(range(0, test_idx))
    cv_folds.append((train_idx, test_idx))

cv_yillar = [int(tum_yillar[t]) for _, t in cv_folds]
print(f"  Walk-Forward CV: {len(cv_folds)} fold (test yillari: {cv_yillar[0]}-{cv_yillar[-1]})")

# Asama1 forecast verileri
fc_asama1 = asama1_df[
    (asama1_df['country'] == ULKE) & (asama1_df['year'] >= FC_ILK)
].sort_values('year').reset_index(drop=True)

if len(fc_asama1) == 0:
    raise RuntimeError(f"{ULKE} asama1 tahminlerinde yok! asama1_sadece_turkey.py calistirin.")

# Eksik featurelar icin son bilinen degeri kullan
for f in KULLANILAN:
    if f not in fc_asama1.columns or fc_asama1[f].isna().all():
        son = turkey_df[f].dropna().iloc[-1] if f in turkey_df.columns else 0.0
        fc_asama1[f] = son
        print(f"    {f}: asama1'de yok, son bilinen deger ({son:.2f})")

# ═════════════════════════════════════════════════════════════════════════════
# OPTUNA: Tum veri uzerinde en iyi hiperparametre bul (TimeSeriesSplit)
# ═════════════════════════════════════════════════════════════════════════════

tscv = TimeSeriesSplit(n_splits=min(5, len(tum_yillar)-2))

def optuna_rf(trial):
    params = dict(
        n_estimators     = trial.suggest_int('n_estimators', 50, 400),
        max_depth        = trial.suggest_int('max_depth', 2, 10),
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5),
        max_features     = trial.suggest_categorical('max_features', ['sqrt','log2', 0.7]),
        random_state=42, n_jobs=-1,
    )
    rmses = []
    for tr_i, vl_i in tscv.split(tum_X):
        sc = MinMaxScaler()
        m = RandomForestRegressor(**params)
        m.fit(sc.fit_transform(tum_X[tr_i]), tum_y[tr_i])
        rmses.append(np.sqrt(mean_squared_error(tum_y[vl_i], m.predict(sc.transform(tum_X[vl_i])))))
    return np.mean(rmses)

def optuna_xgb(trial):
    params = dict(
        n_estimators     = trial.suggest_int('n_estimators', 50, 300),
        max_depth        = trial.suggest_int('max_depth', 2, 5),
        learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        subsample        = trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0),
        reg_alpha        = trial.suggest_float('reg_alpha', 0.0, 2.0),
        reg_lambda       = trial.suggest_float('reg_lambda', 0.5, 3.0),
        random_state=42, verbosity=0,
    )
    rmses = []
    for tr_i, vl_i in tscv.split(tum_X):
        sc = MinMaxScaler()
        m = XGBRegressor(**params)
        m.fit(sc.fit_transform(tum_X[tr_i]), tum_y[tr_i],
              eval_set=[(sc.transform(tum_X[vl_i]), tum_y[vl_i])], verbose=False)
        rmses.append(np.sqrt(mean_squared_error(tum_y[vl_i], m.predict(sc.transform(tum_X[vl_i])))))
    return np.mean(rmses)

# ═════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD CV + FORECAST — her model icin
# ═════════════════════════════════════════════════════════════════════════════

def iteratif_forecast(model, turkey_df, fc_asama1, feat_scaler,
                      hedef_scaler=None, is_lstm=False):
    elec_hist = dict(zip(turkey_df['year'], turkey_df[HEDEF]))
    fc = {}
    for _, row in fc_asama1.iterrows():
        yil = int(row['year'])
        lags = []
        for k in range(1, LAG_SAYISI+1):
            v = elec_hist.get(yil-k, np.nan)
            if np.isnan(v):
                bilinen = sorted([val for yr, val in elec_hist.items() if yr < yil])
                v = bilinen[-1] if bilinen else 0.0
            lags.append(v)
        feats = [float(row.get(f, 0.0)) if not pd.isna(row.get(f, np.nan)) else 0.0
                 for f in KULLANILAN]
        x = np.array(lags + feats).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        x_s = feat_scaler.transform(x)
        x_s = np.nan_to_num(x_s, nan=0.0, posinf=1.0, neginf=0.0)

        if is_lstm:
            pred_s = model.predict(x_s.reshape(1, 1, -1), verbose=0).ravel()[0]
            pred   = hedef_scaler.inverse_transform([[pred_s]])[0][0]
        else:
            pred = model.predict(x_s)[0]

        elec_hist[yil] = pred
        fc[yil] = pred
    return fc

def walk_forward_cv(model_factory, isim):
    """Walk-forward expanding window: her fold'da egit, 1 yil tahmin et."""
    cv_preds = []
    cv_trues = []
    for train_idx, test_idx in cv_folds:
        X_tr, y_tr = tum_X[train_idx], tum_y[train_idx]
        X_te, y_te = tum_X[test_idx:test_idx+1], tum_y[test_idx:test_idx+1]
        sc = MinMaxScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)
        m = model_factory()
        m.fit(X_tr_s, y_tr)
        pred = m.predict(X_te_s)[0]
        cv_preds.append(pred)
        cv_trues.append(y_te[0])
    return np.array(cv_trues), np.array(cv_preds)

def walk_forward_lstm(lstm_factory, hedef_scaler_full):
    """Walk-forward for LSTM (scaled target)."""
    cv_preds, cv_trues = [], []
    for train_idx, test_idx in cv_folds:
        X_tr, y_tr = tum_X[train_idx], tum_y[train_idx]
        X_te, y_te = tum_X[test_idx:test_idx+1], tum_y[test_idx:test_idx+1]
        sc_f = MinMaxScaler()
        sc_h = MinMaxScaler()
        X_tr_s = sc_f.fit_transform(X_tr)
        X_te_s = sc_f.transform(X_te)
        y_tr_s = sc_h.fit_transform(y_tr.reshape(-1,1)).ravel()
        n_feat = X_tr_s.shape[1]
        m = lstm_factory(n_feat)
        es = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        m.fit(X_tr_s.reshape(-1,1,n_feat), y_tr_s,
              epochs=100, batch_size=4, callbacks=[es], verbose=0)
        pred_s = m.predict(X_te_s.reshape(1,1,n_feat), verbose=0).ravel()[0]
        pred = sc_h.inverse_transform([[pred_s]])[0][0]
        cv_preds.append(pred)
        cv_trues.append(y_te[0])
    return np.array(cv_trues), np.array(cv_preds)

tum_cv       = {}   # {model: (trues, preds)}
tum_fc       = {}   # {model: {yil: tahmin}}
tum_metrik   = []

# ═════════════════════════════════════════════════════════════════════════════
# MODEL 1: RANDOM FOREST
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  MODEL 1: RANDOM FOREST")
print("="*65)

print(f"  Optuna ({OPTUNA_TRIAL} trial)...")
rf_study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
rf_study.optimize(optuna_rf, n_trials=OPTUNA_TRIAL)
rf_best = rf_study.best_params
print(f"  En iyi params: {rf_best}")
print(f"  Optuna CV RMSE: {rf_study.best_value:.4f}")

print(f"  Walk-Forward CV ({len(cv_folds)} fold)...")
rf_trues, rf_preds = walk_forward_cv(
    lambda: RandomForestRegressor(**rf_best), 'RF')
rf_m = metrikler(rf_trues, rf_preds)
print(f"  SONUC: R2={rf_m['r2']:.4f}  MAPE={rf_m['mape']:.2f}%  "
      f"RMSE={rf_m['rmse']:.2f}  RMSLE={rf_m['rmsle']:.4f}")
print(f"  Hata Bandi: {hata_bandi(rf_trues, rf_preds)}")

# Final model (tum veri) + forecast
final_sc_rf = MinMaxScaler()
rf_final = RandomForestRegressor(**rf_best)
rf_final.fit(final_sc_rf.fit_transform(tum_X), tum_y)
joblib.dump(rf_final, os.path.join(MODELLER_DIR, 'asama2_rf.pkl'))
rf_fc = iteratif_forecast(rf_final, turkey_df, fc_asama1, final_sc_rf)
print(f"  Forecast 2026-2030:")
for y, v in rf_fc.items(): print(f"    {y}: {v:.2f} TWh")

tum_cv['RF']  = (rf_trues, rf_preds)
tum_fc['RF']  = rf_fc
tum_metrik.append({'model': 'RF', **rf_m, 'optuna_cv_rmse': round(rf_study.best_value,4)})

# Grafikler
ciz_residual(cv_yillar, rf_trues, rf_preds, 'RF',
             os.path.join(GORSELLER_DIR, 'asama2_RF_residual.png'))
ciz_hata_bandi(hata_bandi(rf_trues, rf_preds), 'RF',
               os.path.join(GORSELLER_DIR, 'asama2_RF_hata_bandi.png'))
ciz_shap(rf_final, final_sc_rf.transform(tum_X), INPUT_COLS, 'RF',
         os.path.join(GORSELLER_DIR, 'asama2_RF_shap.png'))
ciz_cv_scatter(rf_trues, rf_preds, 'RF', os.path.join(GORSELLER_DIR, 'asama2_RF_cv_scatter.png'))
ciz_feature_importance(rf_final, INPUT_COLS, 'RF', os.path.join(GORSELLER_DIR, 'asama2_RF_feature_importance.png'))
ciz_learning_curve(tum_X, tum_y, lambda: RandomForestRegressor(**rf_best), 'RF', MIN_EGITIM_YIL, os.path.join(GORSELLER_DIR, 'asama2_RF_learning_curve.png'))

# ═════════════════════════════════════════════════════════════════════════════
# MODEL 2: XGBOOST
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  MODEL 2: XGBOOST")
print("="*65)

print(f"  Optuna ({OPTUNA_TRIAL} trial)...")
xgb_study = optuna.create_study(direction='minimize',
                                  sampler=optuna.samplers.TPESampler(seed=42))
xgb_study.optimize(optuna_xgb, n_trials=OPTUNA_TRIAL)
xgb_best = xgb_study.best_params
print(f"  En iyi params: {xgb_best}")
print(f"  Optuna CV RMSE: {xgb_study.best_value:.4f}")

print(f"  Walk-Forward CV ({len(cv_folds)} fold)...")
xgb_trues, xgb_preds = walk_forward_cv(
    lambda: XGBRegressor(**xgb_best, verbosity=0), 'XGBoost')
xgb_m = metrikler(xgb_trues, xgb_preds)
print(f"  SONUC: R2={xgb_m['r2']:.4f}  MAPE={xgb_m['mape']:.2f}%  "
      f"RMSE={xgb_m['rmse']:.2f}  RMSLE={xgb_m['rmsle']:.4f}")
print(f"  Hata Bandi: {hata_bandi(xgb_trues, xgb_preds)}")

# Final model + forecast
final_sc_xgb = MinMaxScaler()
xgb_final = XGBRegressor(**xgb_best, verbosity=0)
xgb_final.fit(final_sc_xgb.fit_transform(tum_X), tum_y)
joblib.dump(xgb_final, os.path.join(MODELLER_DIR, 'asama2_xgboost.pkl'))
xgb_fc = iteratif_forecast(xgb_final, turkey_df, fc_asama1, final_sc_xgb)
print(f"  Forecast 2026-2030:")
for y, v in xgb_fc.items(): print(f"    {y}: {v:.2f} TWh")

tum_cv['XGBoost']  = (xgb_trues, xgb_preds)
tum_fc['XGBoost']  = xgb_fc
tum_metrik.append({'model': 'XGBoost', **xgb_m, 'optuna_cv_rmse': round(xgb_study.best_value,4)})

ciz_residual(cv_yillar, xgb_trues, xgb_preds, 'XGBoost',
             os.path.join(GORSELLER_DIR, 'asama2_XGBoost_residual.png'))
ciz_hata_bandi(hata_bandi(xgb_trues, xgb_preds), 'XGBoost',
               os.path.join(GORSELLER_DIR, 'asama2_XGBoost_hata_bandi.png'))
ciz_shap(xgb_final, final_sc_xgb.transform(tum_X), INPUT_COLS, 'XGBoost',
         os.path.join(GORSELLER_DIR, 'asama2_XGBoost_shap.png'))
ciz_cv_scatter(xgb_trues, xgb_preds, 'XGBoost', os.path.join(GORSELLER_DIR, 'asama2_XGBoost_cv_scatter.png'))
ciz_feature_importance(xgb_final, INPUT_COLS, 'XGBoost', os.path.join(GORSELLER_DIR, 'asama2_XGBoost_feature_importance.png'))
ciz_learning_curve(tum_X, tum_y, lambda: XGBRegressor(**xgb_best, verbosity=0), 'XGBoost', MIN_EGITIM_YIL, os.path.join(GORSELLER_DIR, 'asama2_XGBoost_learning_curve.png'))

# ═════════════════════════════════════════════════════════════════════════════
# MODEL 3: LSTM
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  MODEL 3: LSTM")
print("="*65)

if TF_MEVCUT:
    def lstm_factory(n_feat, units1=64, units2=32, dropout=0.2):
        m = Sequential([
            LSTM_Layer(units1, return_sequences=True, input_shape=(1, n_feat)),
            Dropout(dropout),
            LSTM_Layer(units2, return_sequences=False),
            Dropout(dropout),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        m.compile(optimizer='adam', loss='mse')
        return m

    # Optuna LSTM
    n_feat_all = tum_X.shape[1]
    def optuna_lstm(trial):
        units1  = trial.suggest_categorical('units1', [32, 64, 128])
        units2  = trial.suggest_categorical('units2', [16, 32, 64])
        dropout = trial.suggest_float('dropout', 0.1, 0.4)
        lr      = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        batch   = trial.suggest_categorical('batch', [4, 8, 16])

        rmses = []
        for tr_i, vl_i in tscv.split(tum_X):
            sc_f = MinMaxScaler(); sc_h = MinMaxScaler()
            X_tr_s = sc_f.fit_transform(tum_X[tr_i])
            X_vl_s = sc_f.transform(tum_X[vl_i])
            y_tr_s = sc_h.fit_transform(tum_y[tr_i].reshape(-1,1)).ravel()
            y_vl   = tum_y[vl_i]
            m = lstm_factory(n_feat_all, units1, units2, dropout)
            m.optimizer.learning_rate.assign(lr)
            es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            y_vl_s = sc_h.transform(y_vl.reshape(-1,1)).ravel()
            m.fit(X_tr_s.reshape(-1,1,n_feat_all), y_tr_s,
                  validation_data=(X_vl_s.reshape(-1,1,n_feat_all), y_vl_s),
                  epochs=100, batch_size=batch, callbacks=[es], verbose=0)
            p_s = m.predict(X_vl_s.reshape(-1,1,n_feat_all), verbose=0).ravel()
            p   = sc_h.inverse_transform(p_s.reshape(-1,1)).ravel()
            rmses.append(np.sqrt(mean_squared_error(y_vl, p)))
        return np.mean(rmses)

    print(f"  Optuna ({OPTUNA_TRIAL} trial)...")
    lstm_study = optuna.create_study(direction='minimize',
                                      sampler=optuna.samplers.TPESampler(seed=42))
    lstm_study.optimize(optuna_lstm, n_trials=OPTUNA_TRIAL)
    lp = lstm_study.best_params
    print(f"  En iyi params: {lp}")
    print(f"  Optuna CV RMSE: {lstm_study.best_value:.4f}")

    # Walk-forward LSTM
    print(f"  Walk-Forward CV ({len(cv_folds)} fold)...")
    lstm_trues, lstm_preds = walk_forward_lstm(
        lambda nf: lstm_factory(nf, lp['units1'], lp['units2'], lp['dropout']),
        None)
    lstm_m = metrikler(lstm_trues, lstm_preds)
    print(f"  SONUC: R2={lstm_m['r2']:.4f}  MAPE={lstm_m['mape']:.2f}%  "
          f"RMSE={lstm_m['rmse']:.2f}  RMSLE={lstm_m['rmsle']:.4f}")
    print(f"  Hata Bandi: {hata_bandi(lstm_trues, lstm_preds)}")

    # Final LSTM + forecast
    final_sc_lstm_f = MinMaxScaler()
    final_sc_lstm_h = MinMaxScaler()
    X_all_s = final_sc_lstm_f.fit_transform(tum_X)
    y_all_s = final_sc_lstm_h.fit_transform(tum_y.reshape(-1,1)).ravel()

    lstm_final = lstm_factory(n_feat_all, lp['units1'], lp['units2'], lp['dropout'])
    es = EarlyStopping(monitor='loss', patience=LSTM_PATIENCE, restore_best_weights=True)
    lstm_final.fit(X_all_s.reshape(-1,1,n_feat_all), y_all_s,
                    epochs=LSTM_EPOCHS, batch_size=lp['batch'],
                    callbacks=[es], verbose=1)
    lstm_final.save(os.path.join(MODELLER_DIR, 'asama2_lstm.keras'))
    joblib.dump(final_sc_lstm_f, os.path.join(MODELLER_DIR, 'asama2_lstm_feat_scaler.pkl'))
    joblib.dump(final_sc_lstm_h, os.path.join(MODELLER_DIR, 'asama2_lstm_hedef_scaler.pkl'))

    lstm_fc = iteratif_forecast(lstm_final, turkey_df, fc_asama1,
                                 final_sc_lstm_f, final_sc_lstm_h, is_lstm=True)
    print(f"  Forecast 2026-2030:")
    for y, v in lstm_fc.items(): print(f"    {y}: {v:.2f} TWh")

    tum_cv['LSTM']  = (lstm_trues, lstm_preds)
    tum_fc['LSTM']  = lstm_fc
    tum_metrik.append({'model': 'LSTM', **lstm_m, 'optuna_cv_rmse': round(lstm_study.best_value,4)})

    ciz_residual(cv_yillar, lstm_trues, lstm_preds, 'LSTM',
                 os.path.join(GORSELLER_DIR, 'asama2_LSTM_residual.png'))
    ciz_hata_bandi(hata_bandi(lstm_trues, lstm_preds), 'LSTM',
                   os.path.join(GORSELLER_DIR, 'asama2_LSTM_hata_bandi.png'))
    ciz_cv_scatter(lstm_trues, lstm_preds, 'LSTM', os.path.join(GORSELLER_DIR, 'asama2_LSTM_cv_scatter.png'))
else:
    print("  LSTM atlandi (TensorFlow yuklu degil).")

# ═════════════════════════════════════════════════════════════════════════════
# BİRLEŞİK GRAFİKLER VE KAYIT
# ═════════════════════════════════════════════════════════════════════════════

# Walk-forward + forecast grafigi
cv_tahmin_dict = {isim: preds for isim, (_, preds) in tum_cv.items()}
fc_dict = {isim: [fc[y] for y in range(FC_ILK, FC_SON+1)] for isim, fc in tum_fc.items()}
fc_yillar = list(range(FC_ILK, FC_SON+1))

ciz_walkforward(turkey_df['year'].values, turkey_df[HEDEF].values,
                cv_yillar, cv_tahmin_dict, fc_yillar, fc_dict,
                os.path.join(GORSELLER_DIR, 'asama2_tum_modeller.png'))

ciz_forecast_karsilastirma(fc_yillar, fc_dict,
    os.path.join(GORSELLER_DIR, 'asama2_forecast_karsilastirma.png'))

# Kaydet
metrik_df = pd.DataFrame(tum_metrik)
metrik_df.to_csv(os.path.join(CIKTILAR_DIR, 'asama2_metrikler.csv'), index=False)

fc_rows = {'year': fc_yillar}
for isim, fc in tum_fc.items():
    fc_rows[f'{isim}_TWh'] = [fc[y] for y in fc_yillar]
fc_df = pd.DataFrame(fc_rows)
fc_df.to_csv(os.path.join(CIKTILAR_DIR, 'elektrik_tahmin_2026_2030.csv'), index=False)

# Walk-forward CV detaylari
cv_detail_rows = []
for isim, (trues, preds) in tum_cv.items():
    for yil, g, t in zip(cv_yillar, trues, preds):
        cv_detail_rows.append({'model': isim, 'year': yil,
                               'gercek_TWh': round(g,2), 'tahmin_TWh': round(t,2),
                               'hata_pct': round(abs(g-t)/g*100, 2)})
cv_detail_df = pd.DataFrame(cv_detail_rows)
cv_detail_df.to_csv(os.path.join(CIKTILAR_DIR, 'asama2_cv_detay.csv'), index=False)

# ─── Özet ─────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("ASAMA 2 TAMAMLANDI")
print("="*65)
print(f"\n  Featurelar ({len(INPUT_COLS)}): {INPUT_COLS}")
print(f"  Walk-Forward CV: {len(cv_folds)} out-of-sample tahmin ({cv_yillar[0]}-{cv_yillar[-1]})")

pd.set_option('display.width', 140)
pd.set_option('display.float_format', '{:.4f}'.format)
print("\nMetrikler (Walk-Forward CV):")
print(metrik_df.to_string(index=False))
print("\nForecast (2026-2030):")
print(fc_df.to_string(index=False))

print("\nKaydedilen dosyalar:")
print(f"  ciktilar/elektrik_tahmin_2026_2030.csv")
print(f"  ciktilar/asama2_metrikler.csv")
print(f"  ciktilar/asama2_cv_detay.csv (her yilin tahmin detayi)")
print(f"  modeller/asama2_rf.pkl, asama2_xgboost.pkl, asama2_lstm.keras")
print(f"  sonuclar/gorseller/asama2_*")
