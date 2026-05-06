"""
ASAMA 1 — Sadece Turkey icin feature tahmini.
Mevcut 66 ulkenin sonuclarina Turkey'yi ekler.
Diger ulkelere DOKUNMAZ.
"""

import os, sys, warnings
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

from sklearn.linear_model    import LinearRegression
from sklearn.preprocessing   import PolynomialFeatures, MinMaxScaler
from sklearn.pipeline        import make_pipeline
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base            import clone
from xgboost                 import XGBRegressor

# --- Sabitler -----------------------------------------------------------------
TRAIN_SON  = 2020
VAL_ILK    = 2021;  VAL_SON  = 2023
TEST_ILK   = 2024;  TEST_SON = 2025
FC_ILK     = 2026;  FC_SON   = 2030
TAHMIN_YILLARI = list(range(VAL_ILK, FC_SON + 1))

LAG_SAYISI   = 2
N_CV_SPLITS  = 3
OPTUNA_TRIAL = 40
MIN_NOKTA    = 6

ULKE = 'Turkey'

# --- Yollar -------------------------------------------------------------------
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR  = os.path.dirname(SCRIPT_DIR)
ISLENMIS_DIR  = os.path.join(PIPELINE_DIR, 'veri', 'islenmis')
MODELLER_DIR  = os.path.join(SCRIPT_DIR, 'modeller')
CIKTILAR_DIR  = os.path.join(SCRIPT_DIR, 'ciktilar')
GORSELLER_DIR = os.path.join(PIPELINE_DIR, 'sonuclar', 'gorseller')

for d in [MODELLER_DIR, CIKTILAR_DIR, GORSELLER_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Yardimci ----------------------------------------------------------------

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true, float), np.array(y_pred, float)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() else np.nan

def lag_df_olustur(yillar, degerler, n_lag=2):
    df = pd.DataFrame({'year': yillar, 'value': degerler})
    for k in range(1, n_lag + 1):
        df[f'lag_{k}'] = df['value'].shift(k)
    return df.dropna().reset_index(drop=True)

def X_y_ayir(df):
    X = df[['year'] + [f'lag_{k}' for k in range(1, LAG_SAYISI + 1)]].values
    y = df['value'].values
    return X, y

def optuna_objective(trial, X_raw, y, tscv):
    params = dict(
        n_estimators     = trial.suggest_int('n_estimators', 50, 300),
        max_depth        = trial.suggest_int('max_depth', 2, 4),
        learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        subsample        = trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0),
        reg_alpha        = trial.suggest_float('reg_alpha', 0.0, 2.0),
        reg_lambda       = trial.suggest_float('reg_lambda', 0.5, 3.0),
        random_state=42, verbosity=0,
    )
    fold_rmse = []
    for tr_i, vl_i in tscv.split(X_raw):
        sc = MinMaxScaler()
        m = XGBRegressor(**params)
        m.fit(sc.fit_transform(X_raw[tr_i]), y[tr_i],
              eval_set=[(sc.transform(X_raw[vl_i]), y[vl_i])], verbose=False)
        fold_rmse.append(np.sqrt(mean_squared_error(y[vl_i], m.predict(sc.transform(X_raw[vl_i])))))
    return np.mean(fold_rmse)

# --- Veri Yukle ---------------------------------------------------------------
print("=" * 70)
print(f"ASAMA 1 — SADECE {ULKE} FEATURE TAHMINI")
print("=" * 70)

train_df = pd.read_csv(os.path.join(ISLENMIS_DIR, 'train.csv'))
val_df   = pd.read_csv(os.path.join(ISLENMIS_DIR, 'val.csv'))
test_df  = pd.read_csv(os.path.join(ISLENMIS_DIR, 'test.csv'))

feat_conf    = pd.read_csv(os.path.join(ISLENMIS_DIR, 'feature_listesi.csv'))
ALL_FEATURES = feat_conf['tum_features'].dropna().tolist()

# Mevcut tahmin dosyasini oku (66 ulke)
tahmin_csv = os.path.join(CIKTILAR_DIR, 'feature_tahminleri_2021_2030.csv')
mevcut_df  = pd.read_csv(tahmin_csv) if os.path.exists(tahmin_csv) else pd.DataFrame()

# Turkey'nin mevcut tahminlerini sil (varsa tekrar olusturacagiz)
if len(mevcut_df) > 0 and 'country' in mevcut_df.columns:
    mevcut_df = mevcut_df[mevcut_df['country'] != ULKE].copy()

print(f"  Feature sayisi: {len(ALL_FEATURES)}")
print(f"  Mevcut tahmin : {len(mevcut_df)} satir ({mevcut_df['country'].nunique() if len(mevcut_df) > 0 else 0} ulke)")
print()

# --- Turkey verisi -----------------------------------------------------------
tr_train = train_df[train_df['country'] == ULKE].sort_values('year')
tr_val   = val_df[val_df['country'] == ULKE].sort_values('year')
tr_test  = test_df[test_df['country'] == ULKE].sort_values('year')

print(f"  {ULKE} Train: {len(tr_train)} satir ({tr_train['year'].min()}-{tr_train['year'].max()})")
print(f"  {ULKE} Val  : {len(tr_val)} satir")
print(f"  {ULKE} Test : {len(tr_test)} satir")
print()

tahmin_dict = {}
metrik_listesi = []

for feat_idx, feature in enumerate(ALL_FEATURES):
    print(f"  [{feat_idx+1}/{len(ALL_FEATURES)}] {feature}", end=" ... ")

    if feature not in tr_train.columns or tr_train[feature].isna().all():
        print("SKIP (tum NaN)")
        continue

    tr_u = tr_train[tr_train[feature].notna()].copy()
    if len(tr_u) < 3:
        print(f"SKIP (sadece {len(tr_u)} satir)")
        continue

    # Lag
    lag_tr = lag_df_olustur(tr_u['year'].values, tr_u[feature].values, LAG_SAYISI)
    if len(lag_tr) < 2:
        print("SKIP (lag sonrasi < 2)")
        continue
    X_tr, y_tr = X_y_ayir(lag_tr)

    # Val/test lag
    gercek_u = pd.concat([tr_u, tr_val, tr_test]).sort_values('year')
    gercek_u = gercek_u[gercek_u[feature].notna()] if feature in gercek_u.columns else gercek_u
    lag_all  = lag_df_olustur(gercek_u['year'].values, gercek_u[feature].values, LAG_SAYISI)
    lag_val_ = lag_all[(lag_all['year'] >= VAL_ILK) & (lag_all['year'] <= VAL_SON)]
    lag_tst_ = lag_all[(lag_all['year'] >= TEST_ILK) & (lag_all['year'] <= TEST_SON)]

    X_val_, y_val_ = (X_y_ayir(lag_val_) if len(lag_val_) > 0 else (np.empty((0, 3)), np.array([])))
    X_tst_, y_tst_ = (X_y_ayir(lag_tst_) if len(lag_tst_) > 0 else (np.empty((0, 3)), np.array([])))

    yeterli = len(X_tr) >= MIN_NOKTA

    if yeterli:
        n_splits_u = min(N_CV_SPLITS, len(X_tr) - 1)
        tscv_u     = TimeSeriesSplit(n_splits=n_splits_u)
        study = optuna.create_study(direction='minimize',
                                     sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lambda trial: optuna_objective(trial, X_tr, y_tr, tscv_u),
                       n_trials=OPTUNA_TRIAL, show_progress_bar=False)
        adaylar = {
            'Linear' : make_pipeline(PolynomialFeatures(1), LinearRegression()),
            'Poly2'  : make_pipeline(PolynomialFeatures(2), LinearRegression()),
            'XGBoost': XGBRegressor(**study.best_params, verbosity=0),
        }
    else:
        adaylar = {'Linear': make_pipeline(PolynomialFeatures(1), LinearRegression())}

    # Model secimi
    if len(X_val_) > 0:
        en_iyi_isim, en_iyi_rmse, en_iyi_model = None, float('inf'), None
        for isim, m in adaylar.items():
            sc_t = MinMaxScaler()
            m.fit(sc_t.fit_transform(X_tr), y_tr)
            rmse_t = np.sqrt(mean_squared_error(y_val_, m.predict(sc_t.transform(X_val_))))
            if rmse_t < en_iyi_rmse:
                en_iyi_rmse, en_iyi_isim, en_iyi_model = rmse_t, isim, m
    else:
        en_iyi_isim  = 'Linear'
        en_iyi_model = make_pipeline(PolynomialFeatures(1), LinearRegression())

    # Final model
    final_sc = MinMaxScaler()
    final_m  = clone(en_iyi_model)
    final_m.fit(final_sc.fit_transform(X_tr), y_tr)

    # 2021-2030 iteratif tahmin
    history_u = dict(zip(gercek_u['year'].values.tolist(), gercek_u[feature].values.tolist()))
    tahmin_u  = {}
    for yil in TAHMIN_YILLARI:
        lag1 = history_u.get(yil - 1, np.nan)
        lag2 = history_u.get(yil - 2, np.nan)
        if np.isnan(lag1):
            bilinen = [v for k, v in history_u.items() if k < yil]
            lag1 = bilinen[-1] if bilinen else 0.0
        if np.isnan(lag2):
            lag2 = lag1
        pred = final_m.predict(final_sc.transform(np.array([[yil, lag1, lag2]])))[0]
        tahmin_u[yil]  = pred
        history_u[yil] = pred

    # Global dict'e kaydet
    iso = tr_u['iso_code'].iloc[0] if 'iso_code' in tr_u.columns else 'TUR'
    for yil in TAHMIN_YILLARI:
        key = (ULKE, yil)
        if key not in tahmin_dict:
            tahmin_dict[key] = {'country': ULKE, 'iso_code': iso, 'year': yil}
        tahmin_dict[key][feature] = round(float(tahmin_u[yil]), 6)

    # Metrik
    val_pred = final_m.predict(final_sc.transform(X_val_)) if len(X_val_) > 0 else np.array([])
    tst_pred = final_m.predict(final_sc.transform(X_tst_)) if len(X_tst_) > 0 else np.array([])
    val_mape = mape(y_val_, val_pred) if len(y_val_) > 0 else np.nan
    tst_mape = mape(y_tst_, tst_pred) if len(y_tst_) > 0 else np.nan

    print(f"{en_iyi_isim} | Val MAPE: {val_mape:.2f}% | Forecast 2030: {tahmin_u[2030]:,.2f}")

    # Model kaydet (mevcut pkl'ye Turkey ekle)
    pkl_yol = os.path.join(MODELLER_DIR, f'asama1_{feature}.pkl')
    if os.path.exists(pkl_yol):
        ulke_modeller = joblib.load(pkl_yol)
    else:
        ulke_modeller = {}
    ulke_modeller[ULKE] = (final_m, final_sc)
    joblib.dump(ulke_modeller, pkl_yol)

# --- Sonuclari birlestir ve kaydet -------------------------------------------
turkey_rows = list(tahmin_dict.values())
turkey_df   = pd.DataFrame(turkey_rows)

# Mevcut 66 ulke + Turkey birlestir
birlesik = pd.concat([mevcut_df, turkey_df], ignore_index=True)
birlesik = birlesik.sort_values(['country', 'year']).reset_index(drop=True)

# Kolon sirasi
cols = ['country', 'iso_code', 'year'] + [f for f in ALL_FEATURES if f in birlesik.columns]
birlesik = birlesik[[c for c in cols if c in birlesik.columns]]

birlesik.to_csv(tahmin_csv, index=False)

print()
print("=" * 70)
print(f"{ULKE} ASAMA 1 TAMAMLANDI")
print("=" * 70)
print(f"  Toplam tahmin: {len(birlesik)} satir ({birlesik['country'].nunique()} ulke)")
tr_check = birlesik[birlesik['country'] == ULKE]
print(f"  {ULKE}: {len(tr_check)} satir ({tr_check['year'].min()}-{tr_check['year'].max()})")
print(f"\n  Kaydedildi: {tahmin_csv}")
