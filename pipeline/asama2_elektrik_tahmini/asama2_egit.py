"""
AŞAMA 2 — Elektrik Tüketimi Tahmini (Optuna + SHAP + Kapsamlı Değerlendirme)
------------------------------------------------------------------------------
Türkiye'nin 2026-2030 elektrik tüketimini (TWh) tahmin eder.

GİRDİLER:
  - electricity_demand lag_1, lag_2  (geçmiş elektrik, sadece geçmişe bakar)
  - Population, GDP_Per_Capita, GDP_Growth, renewables_share_elec,
    Urbanization_Rate, Industry_Value_Added_Pct_GDP, Population_Density
    → val/test için GERÇEK değerler, forecast için AŞAMA 1 TAHMİNLERİ

DATA LEAKAGE ÖNLEMLERİ:
  1. MinMaxScaler SADECE TRAIN verisine (1990-2020) fit edilir.
  2. Val, test, forecast sadece .transform() görür.
  3. TimeSeriesSplit — rastgele fold yasak.
  4. Optuna CV objective'i TimeSeriesSplit val RMSE'sini minimize eder.
  5. Her Optuna trial ve CV fold'unda scaler sıfırdan fit edilir.
  6. Forecast (2026-2030) lag değerleri iteratif hesaplanır — gelecek sızmaz.
  7. Stage 2 val/test değerlendirmesi GERÇEK feature değerleriyle yapılır
     (Stage 1 hatasını Stage 2 değerlendirmesine karıştırmamak için).

MODELLER: Random Forest | XGBoost | LSTM
DEĞERLENDİRME: CV | Learning Curve | Residual | Hata Bandı | SHAP
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
from sklearn.base            import clone
from xgboost                 import XGBRegressor

try:
    import tensorflow as tf
    from tensorflow.keras.models    import Sequential
    from tensorflow.keras.layers    import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TF_MEVCUT = True
except ImportError:
    TF_MEVCUT = False
    print("[UYARI] TensorFlow bulunamadi — LSTM atlaniyor.")

# ─── Sabitler ─────────────────────────────────────────────────────────────────
TRAIN_SON  = 2020
VAL_ILK    = 2021;  VAL_SON  = 2023
TEST_ILK   = 2024;  TEST_SON = 2025
FC_ILK     = 2026;  FC_SON   = 2030

LAG_SAYISI   = 2
N_CV_SPLITS  = 3
OPTUNA_TRIAL = 40
LSTM_EPOCHS  = 200
LSTM_BATCH   = 8
LSTM_PATIENCE= 20

HEDEF = 'electricity_demand'
FEATURE_COLS = [
    'Population', 'GDP_Per_Capita', 'GDP_Growth',
    'renewables_share_elec', 'Urbanization_Rate',
    'Industry_Value_Added_Pct_GDP', 'Population_Density',
]
LAG_COLS   = [f'{HEDEF}_lag{k}' for k in range(1, LAG_SAYISI+1)]
INPUT_COLS = LAG_COLS + FEATURE_COLS

# ─── Yollar ───────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR  = os.path.dirname(SCRIPT_DIR)
ISLENMIS_DIR  = os.path.join(PIPELINE_DIR, 'veri', 'islenmis')
ASAMA1_DIR    = os.path.join(PIPELINE_DIR, 'asama1_feature_tahmini', 'ciktilar')
MODELLER_DIR  = os.path.join(SCRIPT_DIR, 'modeller')
CIKTILAR_DIR  = os.path.join(SCRIPT_DIR, 'ciktilar')
GORSELLER_DIR = os.path.join(PIPELINE_DIR, 'sonuclar', 'gorseller')

# ─── Yardımcı Fonksiyonlar ────────────────────────────────────────────────────

def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask   = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def metrik_yazdir(y_true, y_pred, etiket):
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    mape_ = mape(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)
    print(f"    {etiket:<25} MAPE:{mape_:6.2f}%  RMSE:{rmse:7.2f}  R2:{r2:7.4f}")
    return dict(rmse=round(rmse,4), mae=round(mae,4),
                mape=round(mape_,4), r2=round(r2,4))

def hata_bandi(y_true, y_pred):
    pct = np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true)) * 100
    return {
        'Mukemmel (<%5)' : int(np.sum(pct < 5)),
        'Iyi    (%5-10)' : int(np.sum((pct>=5)  & (pct<10))),
        'Orta  (%10-20)' : int(np.sum((pct>=10) & (pct<20))),
        'Zayif  (>%20)'  : int(np.sum(pct >= 20)),
    }

# ─── Grafik Fonksiyonları ─────────────────────────────────────────────────────

def ciz_zaman_serisi(gercek_yillar, gercek_elec, tahminler_dict, kayit_yolu):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(gercek_yillar, gercek_elec, 'k-o', ms=4, lw=2, label='Gercek', zorder=5)
    renkler = ['steelblue', 'seagreen', 'tomato']
    for (isim, (yillar, degerler)), renk in zip(tahminler_dict.items(), renkler):
        ax.plot(yillar, degerler, '--s', color=renk, ms=4, label=isim)
    ax.axvspan(VAL_ILK,  VAL_SON+0.9,  alpha=0.10, color='green',  label='Validation')
    ax.axvspan(TEST_ILK, TEST_SON+0.9, alpha=0.10, color='orange', label='Test')
    ax.axvspan(FC_ILK,   FC_SON+0.9,   alpha=0.10, color='red',    label='Forecast')
    ax.set_title('Asama 2 — Elektrik Tuketimi: Gercek vs Modeller')
    ax.set_xlabel('Yil'); ax.set_ylabel('TWh')
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(kayit_yolu, dpi=150); plt.close()

def ciz_residual(yillar, y_gercek, y_tahmin, model_isim, bolum, kayit_yolu):
    res = np.array(y_gercek) - np.array(y_tahmin)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(yillar, res, color='steelblue', edgecolor='k')
    axes[0].axhline(0, color='red', lw=1.5)
    axes[0].set_xlabel('Yil'); axes[0].set_ylabel('Residual (TWh)')
    axes[0].set_title(f'{model_isim} — Residual ({bolum})')
    axes[1].scatter(y_tahmin, res, color='steelblue', edgecolors='k', s=60)
    axes[1].axhline(0, color='red', lw=1.5)
    axes[1].set_xlabel('Tahmin (TWh)'); axes[1].set_ylabel('Residual (TWh)')
    axes[1].set_title(f'{model_isim} — Residual vs Tahmin ({bolum})')
    plt.tight_layout(); plt.savefig(kayit_yolu, dpi=150); plt.close()

def ciz_hata_bandi(bandlar, model_isim, bolum, kayit_yolu):
    fig, ax = plt.subplots(figsize=(8, 4))
    renkler = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
    bars = ax.bar(list(bandlar.keys()), list(bandlar.values()),
                  color=renkler, edgecolor='k')
    for bar, val in zip(bars, bandlar.values()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                str(val), ha='center', va='bottom', fontweight='bold')
    ax.set_title(f'{model_isim} — Hata Bandi ({bolum})\n'
                 f'(Regression Confusion Matrix Karsiligi)')
    ax.set_ylabel('Tahmin Sayisi')
    mx = max(bandlar.values()) if max(bandlar.values()) > 0 else 1
    ax.set_ylim(0, mx + 1.5)
    plt.tight_layout(); plt.savefig(kayit_yolu, dpi=150); plt.close()

def ciz_cv(model_isim, cv_tr, cv_vl, kayit_yolu):
    folds = list(range(1, len(cv_tr)+1))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(folds, cv_tr, 'b-o', label='Train RMSE')
    ax.plot(folds, cv_vl, 'r-o', label='Val RMSE')
    ax.set_xticks(folds)
    ax.set_xlabel('CV Fold (TimeSeriesSplit)'); ax.set_ylabel('RMSE (TWh)')
    ax.set_title(f'{model_isim} — Cross-Validation')
    ax.legend()
    plt.tight_layout(); plt.savefig(kayit_yolu, dpi=150); plt.close()

def ciz_learning_curve(X_s, y, model_factory, model_isim, tscv, kayit_yolu):
    n = len(y)
    boyutlar = sorted(set([int(n*r) for r in [0.4,0.55,0.7,0.85,1.0] if int(n*r)>3]))
    tr_means, vl_means = [], []
    for boyut in boyutlar:
        Xs, ys = X_s[:boyut], y[:boyut]
        fold_tr, fold_vl = [], []
        for tr_idx, vl_idx in tscv.split(Xs):
            if len(tr_idx)<2 or len(vl_idx)<1: continue
            sc = MinMaxScaler()
            Xft_s = sc.fit_transform(Xs[tr_idx])
            Xfv_s = sc.transform(Xs[vl_idx])
            m = model_factory()
            m.fit(Xft_s, ys[tr_idx])
            fold_tr.append(np.sqrt(mean_squared_error(ys[tr_idx], m.predict(Xft_s))))
            fold_vl.append(np.sqrt(mean_squared_error(ys[vl_idx], m.predict(Xfv_s))))
        if fold_tr:
            tr_means.append(np.mean(fold_tr))
            vl_means.append(np.mean(fold_vl))
    if len(tr_means) < 2: return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(boyutlar[:len(tr_means)], tr_means, 'b-o', label='Train RMSE')
    ax.plot(boyutlar[:len(vl_means)], vl_means, 'r-o', label='Val RMSE')
    ax.set_xlabel('Egitim Ornegi Sayisi'); ax.set_ylabel('RMSE (TWh)')
    ax.set_title(f'{model_isim} — Learning Curve')
    ax.legend()
    plt.tight_layout(); plt.savefig(kayit_yolu, dpi=150); plt.close()

def ciz_shap(model, X_s, feature_names, model_isim, kayit_yolu):
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_s)
        mean_abs  = np.abs(shap_vals).mean(axis=0)
        sorted_idx = np.argsort(mean_abs)
        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.barh([feature_names[i] for i in sorted_idx],
                       mean_abs[sorted_idx],
                       color='steelblue', edgecolor='k')
        for bar, val in zip(bars, mean_abs[sorted_idx]):
            ax.text(val + mean_abs.max()*0.01,
                    bar.get_y()+bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=8)
        ax.set_xlabel('Ortalama |SHAP Degeri|')
        ax.set_title(f'{model_isim} — SHAP Feature Importance\n'
                     f'(Hangi feature elektrik tahminini ne kadar etkiliyor?)')
        plt.tight_layout(); plt.savefig(kayit_yolu, dpi=150); plt.close()
    except Exception as e:
        print(f"    [SHAP cizilmedi: {e}]")

def ciz_lstm_loss(history, kayit_yolu):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history['loss'],     'b-', label='Train Loss (MSE)')
    ax.plot(history.history['val_loss'], 'r-', label='Val Loss (MSE)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (MSE)')
    ax.set_title('LSTM — Learning Curve (Epoch bazli)')
    ax.legend()
    plt.tight_layout(); plt.savefig(kayit_yolu, dpi=150); plt.close()

def ciz_forecast_karsilastirma(fc_yillar, tahminler_dict, kayit_yolu):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(fc_yillar)); w = 0.25
    renkler = ['steelblue', 'seagreen', 'tomato']
    for i, (isim, degerler) in enumerate(tahminler_dict.items()):
        ax.bar(x+i*w, degerler, w, label=isim,
               color=renkler[i%len(renkler)], edgecolor='k')
    ax.set_xticks(x+w); ax.set_xticklabels([str(y) for y in fc_yillar])
    ax.set_xlabel('Yil'); ax.set_ylabel('Tahmin (TWh)')
    ax.set_title('2026-2030 Elektrik Tahmini — Model Karsilastirmasi')
    ax.legend()
    plt.tight_layout(); plt.savefig(kayit_yolu, dpi=150); plt.close()

# ─── Veri Hazırla ─────────────────────────────────────────────────────────────
print("=" * 65)
print("ASAMA 2 — ELEKTRIK TUKETIMI TAHMINI")
print("=" * 65)

train_df  = pd.read_csv(os.path.join(ISLENMIS_DIR, 'turkiye_train_1990_2020.csv'))
val_df    = pd.read_csv(os.path.join(ISLENMIS_DIR, 'turkiye_val_2021_2023.csv'))
test_df   = pd.read_csv(os.path.join(ISLENMIS_DIR, 'turkiye_test_2024_2025.csv'))
asama1_df = pd.read_csv(os.path.join(ASAMA1_DIR,   'feature_tahminleri_2021_2030.csv'))

gercek_df = pd.concat([train_df, val_df, test_df]).sort_values('year').reset_index(drop=True)

# Lag feature ekle — shift() sadece geçmişe bakar
for k in range(1, LAG_SAYISI+1):
    gercek_df[f'{HEDEF}_lag{k}'] = gercek_df[HEDEF].shift(k)
gercek_df = gercek_df.dropna().reset_index(drop=True)

# Split
def bolum_al(df, yil_min, yil_max):
    m = (df['year'] >= yil_min) & (df['year'] <= yil_max)
    return df[m][INPUT_COLS].values, df[m][HEDEF].values, df[m]['year'].values

train_X, train_y, train_yil = bolum_al(gercek_df, gercek_df['year'].min(), TRAIN_SON)
val_X,   val_y,   val_yil   = bolum_al(gercek_df, VAL_ILK, VAL_SON)
test_X,  test_y,  test_yil  = bolum_al(gercek_df, TEST_ILK, TEST_SON)

print(f"  Train: {train_X.shape}  Val: {val_X.shape}  Test: {test_X.shape}")
print(f"  Girdiler: {INPUT_COLS}")

# ── SCALER: sadece TRAIN'e fit ────────────────────────────────────────────────
feat_scaler  = MinMaxScaler()
train_X_s    = feat_scaler.fit_transform(train_X)   # fit + transform
val_X_s      = feat_scaler.transform(val_X)          # sadece transform
test_X_s     = feat_scaler.transform(test_X)          # sadece transform

hedef_scaler = MinMaxScaler()
train_y_s    = hedef_scaler.fit_transform(train_y.reshape(-1,1)).ravel()

joblib.dump(feat_scaler,  os.path.join(MODELLER_DIR, 'asama2_feat_scaler.pkl'))
joblib.dump(hedef_scaler, os.path.join(MODELLER_DIR, 'asama2_hedef_scaler.pkl'))
print("  [OK] Scaler sadece TRAIN verisine fit edildi.")

tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)

# ── Forecast Feature Matrisi (2026-2030) ─────────────────────────────────────
fc_asama1 = asama1_df[asama1_df['year'] >= FC_ILK].reset_index(drop=True)

def iteratif_forecast(model, gercek_df, fc_asama1, feat_scaler,
                      hedef_scaler=None, is_lstm=False):
    """
    Her yıl için lag değerini bir önceki adımın tahmininden alır.
    Feature değerlerini Aşama 1 çıktısından alır.
    Gelecek veri sızmaz.
    """
    elec_hist = dict(zip(gercek_df['year'], gercek_df[HEDEF]))
    fc_tahmin = {}

    for _, row in fc_asama1.iterrows():
        yil = int(row['year'])
        lag_vals  = [elec_hist.get(yil-k, np.nan) for k in range(1, LAG_SAYISI+1)]
        feat_vals = [row[f] for f in FEATURE_COLS]
        x_row = np.array(lag_vals + feat_vals).reshape(1, -1)
        x_row_s = feat_scaler.transform(x_row)

        if is_lstm:
            x_row_s = x_row_s.reshape(1, 1, -1)
            pred_s  = model.predict(x_row_s, verbose=0).ravel()[0]
            pred    = hedef_scaler.inverse_transform([[pred_s]])[0][0]
        else:
            pred = model.predict(x_row_s)[0]

        elec_hist[yil] = pred
        fc_tahmin[yil] = pred

    return fc_tahmin

# Depolama
tum_val_tahmin  = {}
tum_test_tahmin = {}
tum_fc_tahmin   = {}
tum_metrikler   = []

# ═══════════════════════════════════════════════════════════════════════════════
# OPTUNA OBJECTIVE (RF ve XGBoost için ortak yapı)
# ═══════════════════════════════════════════════════════════════════════════════

def optuna_xgb(trial, X_raw, y, tscv):
    params = dict(
        n_estimators     = trial.suggest_int  ('n_estimators', 50, 300),
        max_depth        = trial.suggest_int  ('max_depth', 2, 5),
        learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        subsample        = trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0),
        reg_alpha        = trial.suggest_float('reg_alpha', 0.0, 2.0),
        reg_lambda       = trial.suggest_float('reg_lambda', 0.5, 3.0),
        random_state=42, verbosity=0,
    )
    rmse_list = []
    for tr_idx, vl_idx in tscv.split(X_raw):
        sc = MinMaxScaler()
        Xft = sc.fit_transform(X_raw[tr_idx])
        Xfv = sc.transform(X_raw[vl_idx])
        m = XGBRegressor(**params)
        m.fit(Xft, y[tr_idx], eval_set=[(Xfv, y[vl_idx])], verbose=False)
        rmse_list.append(np.sqrt(mean_squared_error(y[vl_idx], m.predict(Xfv))))
    return np.mean(rmse_list)

def optuna_rf(trial, X_raw, y, tscv):
    params = dict(
        n_estimators  = trial.suggest_int('n_estimators', 50, 400),
        max_depth     = trial.suggest_int('max_depth', 2, 10),
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5),
        max_features  = trial.suggest_categorical('max_features', ['sqrt','log2', 0.7]),
        random_state  = 42, n_jobs=-1,
    )
    rmse_list = []
    for tr_idx, vl_idx in tscv.split(X_raw):
        sc = MinMaxScaler()
        Xft = sc.fit_transform(X_raw[tr_idx])
        Xfv = sc.transform(X_raw[vl_idx])
        m = RandomForestRegressor(**params)
        m.fit(Xft, y[tr_idx])
        rmse_list.append(np.sqrt(mean_squared_error(y[vl_idx], m.predict(Xfv))))
    return np.mean(rmse_list)

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 1: RANDOM FOREST
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  MODEL 1: RANDOM FOREST")
print("="*65)

print(f"  Optuna ({OPTUNA_TRIAL} trial)...")
rf_study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
rf_study.optimize(lambda t: optuna_rf(t, train_X, train_y, tscv),
                  n_trials=OPTUNA_TRIAL)
rf_best = rf_study.best_params
print(f"  En iyi params: {rf_best}  |  CV RMSE: {rf_study.best_value:.4f}")

# CV ile fold RMSE
cv_tr, cv_vl = [], []
print(f"  TimeSeriesSplit CV:")
for fi, (tr_idx, vl_idx) in enumerate(tscv.split(train_X_s)):
    sc = MinMaxScaler()
    Xft = sc.fit_transform(train_X[tr_idx])
    Xfv = sc.transform(train_X[vl_idx])
    m = RandomForestRegressor(**rf_best)
    m.fit(Xft, train_y[tr_idx])
    cv_tr.append(np.sqrt(mean_squared_error(train_y[tr_idx], m.predict(Xft))))
    cv_vl.append(np.sqrt(mean_squared_error(train_y[vl_idx], m.predict(Xfv))))
    print(f"    Fold {fi+1}: TrainRMSE={cv_tr[-1]:.2f}  ValRMSE={cv_vl[-1]:.2f}")

# Final model
rf_model = RandomForestRegressor(**rf_best)
rf_model.fit(train_X_s, train_y)
joblib.dump(rf_model, os.path.join(MODELLER_DIR, 'asama2_rf.pkl'))

rf_val_pred  = rf_model.predict(val_X_s)
rf_test_pred = rf_model.predict(test_X_s)
rf_fc        = iteratif_forecast(rf_model, gercek_df, fc_asama1, feat_scaler)

print("  Metrikler:")
rf_val_m  = metrik_yazdir(val_y,  rf_val_pred,  'Validation (2021-2023)')
rf_test_m = metrik_yazdir(test_y, rf_test_pred, 'Test       (2024-2025)')

print("  Hata Bandi:")
rfvb = hata_bandi(val_y, rf_val_pred)
rftb = hata_bandi(test_y, rf_test_pred)
print(f"    Val: {rfvb}")
print(f"    Test: {rftb}")

print("  Forecast 2026-2030:")
for y,v in rf_fc.items(): print(f"    {y}: {v:.2f} TWh")

tum_val_tahmin['RF']  = rf_val_pred
tum_test_tahmin['RF'] = rf_test_pred
tum_fc_tahmin['RF']   = [rf_fc[y] for y in range(FC_ILK, FC_SON+1)]

# Grafikler
ciz_cv('RF', cv_tr, cv_vl,
       os.path.join(GORSELLER_DIR, 'asama2_RF_cv.png'))
ciz_learning_curve(train_X, train_y, lambda: RandomForestRegressor(**rf_best),
                   'RF', tscv,
                   os.path.join(GORSELLER_DIR, 'asama2_RF_learning_curve.png'))
ciz_residual(val_yil, val_y, rf_val_pred, 'RF', 'Validation',
             os.path.join(GORSELLER_DIR, 'asama2_RF_residual_val.png'))
ciz_residual(test_yil, test_y, rf_test_pred, 'RF', 'Test',
             os.path.join(GORSELLER_DIR, 'asama2_RF_residual_test.png'))
ciz_hata_bandi(rfvb,  'RF', 'Validation',
               os.path.join(GORSELLER_DIR, 'asama2_RF_hata_bandi_val.png'))
ciz_hata_bandi(rftb,  'RF', 'Test',
               os.path.join(GORSELLER_DIR, 'asama2_RF_hata_bandi_test.png'))
ciz_shap(rf_model, train_X_s, INPUT_COLS, 'RF',
         os.path.join(GORSELLER_DIR, 'asama2_RF_shap.png'))

tum_metrikler.append({
    'model': 'RF',
    **{f'val_{k}': v for k,v in rf_val_m.items()},
    **{f'test_{k}': v for k,v in rf_test_m.items()},
    'cv_val_rmse_ort': round(np.mean(cv_vl),4),
    'optuna_cv_rmse' : round(rf_study.best_value,4),
})

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 2: XGBOOST
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  MODEL 2: XGBOOST")
print("="*65)

print(f"  Optuna ({OPTUNA_TRIAL} trial)...")
xgb_study = optuna.create_study(direction='minimize',
                                  sampler=optuna.samplers.TPESampler(seed=42))
xgb_study.optimize(lambda t: optuna_xgb(t, train_X, train_y, tscv),
                    n_trials=OPTUNA_TRIAL)
xgb_best = xgb_study.best_params
print(f"  En iyi params: {xgb_best}  |  CV RMSE: {xgb_study.best_value:.4f}")

cv_tr, cv_vl = [], []
print(f"  TimeSeriesSplit CV:")
for fi, (tr_idx, vl_idx) in enumerate(tscv.split(train_X_s)):
    sc = MinMaxScaler()
    Xft = sc.fit_transform(train_X[tr_idx])
    Xfv = sc.transform(train_X[vl_idx])
    m = XGBRegressor(**xgb_best, verbosity=0)
    m.fit(Xft, train_y[tr_idx], eval_set=[(Xfv, train_y[vl_idx])], verbose=False)
    cv_tr.append(np.sqrt(mean_squared_error(train_y[tr_idx], m.predict(Xft))))
    cv_vl.append(np.sqrt(mean_squared_error(train_y[vl_idx], m.predict(Xfv))))
    print(f"    Fold {fi+1}: TrainRMSE={cv_tr[-1]:.2f}  ValRMSE={cv_vl[-1]:.2f}")

xgb_model = XGBRegressor(**xgb_best, verbosity=0)
xgb_model.fit(train_X_s, train_y, eval_set=[(val_X_s, val_y)], verbose=False)
joblib.dump(xgb_model, os.path.join(MODELLER_DIR, 'asama2_xgboost.pkl'))

xgb_val_pred  = xgb_model.predict(val_X_s)
xgb_test_pred = xgb_model.predict(test_X_s)
xgb_fc        = iteratif_forecast(xgb_model, gercek_df, fc_asama1, feat_scaler)

print("  Metrikler:")
xgb_val_m  = metrik_yazdir(val_y,  xgb_val_pred,  'Validation (2021-2023)')
xgb_test_m = metrik_yazdir(test_y, xgb_test_pred, 'Test       (2024-2025)')

print("  Hata Bandi:")
xvb = hata_bandi(val_y, xgb_val_pred)
xtb = hata_bandi(test_y, xgb_test_pred)
print(f"    Val: {xvb}")
print(f"    Test: {xtb}")

print("  Forecast 2026-2030:")
for y,v in xgb_fc.items(): print(f"    {y}: {v:.2f} TWh")

tum_val_tahmin['XGBoost']  = xgb_val_pred
tum_test_tahmin['XGBoost'] = xgb_test_pred
tum_fc_tahmin['XGBoost']   = [xgb_fc[y] for y in range(FC_ILK, FC_SON+1)]

ciz_cv('XGBoost', cv_tr, cv_vl,
       os.path.join(GORSELLER_DIR, 'asama2_XGBoost_cv.png'))
ciz_learning_curve(train_X, train_y, lambda: XGBRegressor(**xgb_best, verbosity=0),
                   'XGBoost', tscv,
                   os.path.join(GORSELLER_DIR, 'asama2_XGBoost_learning_curve.png'))
ciz_residual(val_yil, val_y, xgb_val_pred, 'XGBoost', 'Validation',
             os.path.join(GORSELLER_DIR, 'asama2_XGBoost_residual_val.png'))
ciz_residual(test_yil, test_y, xgb_test_pred, 'XGBoost', 'Test',
             os.path.join(GORSELLER_DIR, 'asama2_XGBoost_residual_test.png'))
ciz_hata_bandi(xvb, 'XGBoost', 'Validation',
               os.path.join(GORSELLER_DIR, 'asama2_XGBoost_hata_bandi_val.png'))
ciz_hata_bandi(xtb, 'XGBoost', 'Test',
               os.path.join(GORSELLER_DIR, 'asama2_XGBoost_hata_bandi_test.png'))
ciz_shap(xgb_model, train_X_s, INPUT_COLS, 'XGBoost',
         os.path.join(GORSELLER_DIR, 'asama2_XGBoost_shap.png'))

tum_metrikler.append({
    'model': 'XGBoost',
    **{f'val_{k}': v for k,v in xgb_val_m.items()},
    **{f'test_{k}': v for k,v in xgb_test_m.items()},
    'cv_val_rmse_ort': round(np.mean(cv_vl),4),
    'optuna_cv_rmse' : round(xgb_study.best_value,4),
})

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL 3: LSTM
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  MODEL 3: LSTM")
print("="*65)

if TF_MEVCUT:
    val_y_s   = hedef_scaler.transform(val_y.reshape(-1,1)).ravel()
    train_y_s2= hedef_scaler.transform(train_y.reshape(-1,1)).ravel()

    # LSTM girdisi: (samples, timesteps=1, features)
    Xt = train_X_s.reshape(train_X_s.shape[0], 1, train_X_s.shape[1])
    Xv = val_X_s.reshape(val_X_s.shape[0],   1, val_X_s.shape[1])
    Xe = test_X_s.reshape(test_X_s.shape[0], 1, test_X_s.shape[1])

    def yeni_lstm(n_feat, units1=64, units2=32, dropout=0.2):
        m = Sequential([
            LSTM(units1, return_sequences=True, input_shape=(1, n_feat)),
            Dropout(dropout),
            LSTM(units2, return_sequences=False),
            Dropout(dropout),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        m.compile(optimizer='adam', loss='mse')
        return m

    n_feat = train_X_s.shape[1]

    # Optuna — LSTM hiperparametre araması
    def optuna_lstm(trial):
        units1   = trial.suggest_categorical('units1', [32, 64, 128])
        units2   = trial.suggest_categorical('units2', [16, 32, 64])
        dropout  = trial.suggest_float('dropout', 0.1, 0.4)
        lr       = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        batch    = trial.suggest_categorical('batch', [4, 8, 16])

        rmse_list = []
        for tr_idx, vl_idx in tscv.split(train_X_s):
            Xft = train_X_s[tr_idx].reshape(-1, 1, n_feat)
            Xfv = train_X_s[vl_idx].reshape(-1, 1, n_feat)
            yft = train_y_s2[tr_idx]
            yfv = train_y_s2[vl_idx]
            lstm_cv = yeni_lstm(n_feat, units1, units2, dropout)
            lstm_cv.optimizer.learning_rate.assign(lr)
            es = EarlyStopping(monitor='val_loss', patience=10,
                               restore_best_weights=True)
            lstm_cv.fit(Xft, yft, validation_data=(Xfv, yfv),
                        epochs=100, batch_size=batch,
                        callbacks=[es], verbose=0)
            pred_s = lstm_cv.predict(Xfv, verbose=0).ravel()
            pred   = hedef_scaler.inverse_transform(pred_s.reshape(-1,1)).ravel()
            orig   = hedef_scaler.inverse_transform(yfv.reshape(-1,1)).ravel()
            rmse_list.append(np.sqrt(mean_squared_error(orig, pred)))
        return np.mean(rmse_list)

    print(f"  Optuna ({OPTUNA_TRIAL} trial)...")
    lstm_study = optuna.create_study(direction='minimize',
                                      sampler=optuna.samplers.TPESampler(seed=42))
    lstm_study.optimize(optuna_lstm, n_trials=OPTUNA_TRIAL)
    lp = lstm_study.best_params
    print(f"  En iyi params: {lp}  |  CV RMSE: {lstm_study.best_value:.4f}")

    # CV
    cv_tr, cv_vl = [], []
    print(f"  TimeSeriesSplit CV:")
    for fi, (tr_idx, vl_idx) in enumerate(tscv.split(train_X_s)):
        Xft = train_X_s[tr_idx].reshape(-1,1,n_feat)
        Xfv = train_X_s[vl_idx].reshape(-1,1,n_feat)
        yft = train_y_s2[tr_idx]; yfv = train_y_s2[vl_idx]
        m_cv = yeni_lstm(n_feat, lp['units1'], lp['units2'], lp['dropout'])
        es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        m_cv.fit(Xft, yft, validation_data=(Xfv,yfv),
                 epochs=LSTM_EPOCHS, batch_size=lp['batch'],
                 callbacks=[es], verbose=0)
        p_tr = hedef_scaler.inverse_transform(m_cv.predict(Xft,verbose=0).reshape(-1,1)).ravel()
        p_vl = hedef_scaler.inverse_transform(m_cv.predict(Xfv,verbose=0).reshape(-1,1)).ravel()
        o_tr = hedef_scaler.inverse_transform(yft.reshape(-1,1)).ravel()
        o_vl = hedef_scaler.inverse_transform(yfv.reshape(-1,1)).ravel()
        cv_tr.append(np.sqrt(mean_squared_error(o_tr, p_tr)))
        cv_vl.append(np.sqrt(mean_squared_error(o_vl, p_vl)))
        print(f"    Fold {fi+1}: TrainRMSE={cv_tr[-1]:.2f}  ValRMSE={cv_vl[-1]:.2f}")

    # Final LSTM
    lstm_path = os.path.join(MODELLER_DIR, 'asama2_lstm.keras')
    lstm_final = yeni_lstm(n_feat, lp['units1'], lp['units2'], lp['dropout'])
    es_f = EarlyStopping(monitor='val_loss', patience=LSTM_PATIENCE,
                          restore_best_weights=True)
    mc   = ModelCheckpoint(lstm_path, monitor='val_loss',
                            save_best_only=True, verbose=0)
    history = lstm_final.fit(Xt, train_y_s2,
                              validation_data=(Xv, val_y_s),
                              epochs=LSTM_EPOCHS, batch_size=lp['batch'],
                              callbacks=[es_f, mc], verbose=1)

    lstm_val_pred  = hedef_scaler.inverse_transform(
        lstm_final.predict(Xv, verbose=0).reshape(-1,1)).ravel()
    lstm_test_pred = hedef_scaler.inverse_transform(
        lstm_final.predict(Xe, verbose=0).reshape(-1,1)).ravel()
    lstm_fc = iteratif_forecast(lstm_final, gercek_df, fc_asama1,
                                 feat_scaler, hedef_scaler, is_lstm=True)

    print("  Metrikler:")
    lstm_val_m  = metrik_yazdir(val_y,  lstm_val_pred,  'Validation (2021-2023)')
    lstm_test_m = metrik_yazdir(test_y, lstm_test_pred, 'Test       (2024-2025)')

    print("  Hata Bandi:")
    lvb = hata_bandi(val_y, lstm_val_pred)
    ltb = hata_bandi(test_y, lstm_test_pred)
    print(f"    Val: {lvb}"); print(f"    Test: {ltb}")

    print("  Forecast 2026-2030:")
    for y,v in lstm_fc.items(): print(f"    {y}: {v:.2f} TWh")

    tum_val_tahmin['LSTM']  = lstm_val_pred
    tum_test_tahmin['LSTM'] = lstm_test_pred
    tum_fc_tahmin['LSTM']   = [lstm_fc[y] for y in range(FC_ILK, FC_SON+1)]

    ciz_lstm_loss(history,
                  os.path.join(GORSELLER_DIR, 'asama2_LSTM_learning_curve.png'))
    ciz_cv('LSTM', cv_tr, cv_vl,
           os.path.join(GORSELLER_DIR, 'asama2_LSTM_cv.png'))
    ciz_residual(val_yil, val_y, lstm_val_pred, 'LSTM', 'Validation',
                 os.path.join(GORSELLER_DIR, 'asama2_LSTM_residual_val.png'))
    ciz_residual(test_yil, test_y, lstm_test_pred, 'LSTM', 'Test',
                 os.path.join(GORSELLER_DIR, 'asama2_LSTM_residual_test.png'))
    ciz_hata_bandi(lvb, 'LSTM', 'Validation',
                   os.path.join(GORSELLER_DIR, 'asama2_LSTM_hata_bandi_val.png'))
    ciz_hata_bandi(ltb, 'LSTM', 'Test',
                   os.path.join(GORSELLER_DIR, 'asama2_LSTM_hata_bandi_test.png'))

    tum_metrikler.append({
        'model': 'LSTM',
        **{f'val_{k}': v for k,v in lstm_val_m.items()},
        **{f'test_{k}': v for k,v in lstm_test_m.items()},
        'cv_val_rmse_ort': round(np.mean(cv_vl),4),
        'optuna_cv_rmse' : round(lstm_study.best_value,4),
    })
else:
    print("  LSTM atlandi (TensorFlow yuklu degil).")

# ─── Birleşik Grafikler ───────────────────────────────────────────────────────
birlesik = {}
for isim in tum_val_tahmin:
    yillar  = list(val_yil) + list(test_yil) + list(range(FC_ILK, FC_SON+1))
    degerler= list(tum_val_tahmin[isim]) + list(tum_test_tahmin[isim]) + tum_fc_tahmin[isim]
    birlesik[isim] = (yillar, degerler)

ciz_zaman_serisi(gercek_df['year'].values, gercek_df[HEDEF].values,
                 birlesik,
                 os.path.join(GORSELLER_DIR, 'asama2_tum_modeller.png'))
ciz_forecast_karsilastirma(
    list(range(FC_ILK, FC_SON+1)),
    {isim: tum_fc_tahmin[isim] for isim in tum_fc_tahmin},
    os.path.join(GORSELLER_DIR, 'asama2_forecast_karsilastirma.png'))

# ─── Kaydet ───────────────────────────────────────────────────────────────────
metrik_df = pd.DataFrame(tum_metrikler)
metrik_df.to_csv(os.path.join(CIKTILAR_DIR, 'asama2_metrikler.csv'), index=False)

fc_df = pd.DataFrame({'year': list(range(FC_ILK, FC_SON+1)),
                       **{f'{isim}_TWh': tum_fc_tahmin[isim]
                          for isim in tum_fc_tahmin}})
fc_df.to_csv(os.path.join(CIKTILAR_DIR, 'elektrik_tahmin_2026_2030.csv'), index=False)

# ─── Özet ─────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("ASAMA 2 TAMAMLANDI")
print("="*65)
pd.set_option('display.width', 120)
pd.set_option('display.float_format', '{:.4f}'.format)
print("\nMetrikler:")
print(metrik_df.to_string(index=False))
print("\nForecast (2026-2030):")
print(fc_df.to_string(index=False))
