"""
ASAMA 1 — Feature Tahmini (Tum Ulkeler, Per-Country, Optuna + SHAP)
---------------------------------------------------------------------
Tum ulkeler icin tum korele parametrelerin 2021-2030 tahminini yapar.
Her ulke x her feature icin ayri model egitilir.

GIRTI:
  pipeline/veri/islenmis/train.csv, val.csv, test.csv  (tum ulkeler)
  pipeline/veri/islenmis/feature_listesi.csv

CIKTI:
  ciktilar/feature_tahminleri_2021_2030.csv  -> tum ulkeler x 10 yil
  ciktilar/asama1_metrikler.csv              -> val/test metrikleri
  modeller/asama1_{feature}.pkl              -> {country: (model, scaler)} dict
  sonuclar/gorseller/asama1_{feature}_*.png  -> SADECE TURKIYE icin

OVERFITTING ONLEMLERI:
  - max_depth 2-4 kisitlandi
  - Optuna TimeSeriesSplit CV RMSE minimize eder (train RMSE degil)
  - Her fold'da scaler sifirdan fit edilir
  - Az veri (<6 nokta): sadece Linear regresyon, Optuna yok

DEGERLENDIRME (Turkiye icin):
  - TimeSeriesSplit CV
  - Learning Curve
  - Residual analizi
  - Hata bandi analizi (<%5 / %5-10 / %10-20 / >%20)
  - SHAP feature importance (XGBoost secilirse)
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
TAHMIN_YILLARI = list(range(VAL_ILK, FC_SON + 1))   # 2021-2030

LAG_SAYISI   = 2
N_CV_SPLITS  = 3
OPTUNA_TRIAL = 40
MIN_NOKTA    = 6    # CV icin minimum veri noktasi (az ise sadece Linear)

TURKIYE_ADI  = 'Turkey'   # ayrintili grafik kaydedilecek ulke

# --- Yollar -------------------------------------------------------------------
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR  = os.path.dirname(SCRIPT_DIR)
ISLENMIS_DIR  = os.path.join(PIPELINE_DIR, 'veri', 'islenmis')
MODELLER_DIR  = os.path.join(SCRIPT_DIR, 'modeller')
CIKTILAR_DIR  = os.path.join(SCRIPT_DIR, 'ciktilar')
GORSELLER_DIR = os.path.join(PIPELINE_DIR, 'sonuclar', 'gorseller')

for d in [MODELLER_DIR, CIKTILAR_DIR, GORSELLER_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Yardimci Fonksiyonlar ----------------------------------------------------

def mape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def metrik_hesapla(y_true, y_pred):
    if len(y_true) == 0:
        return dict(rmse=np.nan, mae=np.nan, mape=np.nan, r2=np.nan)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    mape_ = mape(y_true, y_pred)
    r2    = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    return dict(rmse=round(rmse, 4), mae=round(mae, 4),
                mape=round(mape_, 4), r2=round(r2, 4))

def hata_bandi(y_true, y_pred):
    if len(y_true) == 0:
        return {'Mukemmel (<%5)': 0, 'Iyi (%5-10)': 0,
                'Orta (%10-20)': 0, 'Zayif (>%20)': 0}
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask   = y_true != 0
    pct    = np.zeros(len(y_true))
    pct[mask] = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100
    return {
        'Mukemmel (<%5)' : int(np.sum(pct < 5)),
        'Iyi    (%5-10)' : int(np.sum((pct >= 5)  & (pct < 10))),
        'Orta  (%10-20)' : int(np.sum((pct >= 10) & (pct < 20))),
        'Zayif  (>%20)'  : int(np.sum(pct >= 20)),
    }

def lag_df_olustur(yillar, degerler, n_lag=2):
    """year + lag_1 + lag_2 ozellikleri — shift() geriye bakar, leakage yok."""
    df = pd.DataFrame({'year': yillar, 'value': degerler})
    for k in range(1, n_lag + 1):
        df[f'lag_{k}'] = df['value'].shift(k)
    return df.dropna().reset_index(drop=True)

def X_y_ayir(df):
    X = df[['year'] + [f'lag_{k}' for k in range(1, LAG_SAYISI + 1)]].values
    y = df['value'].values
    return X, y

def optuna_objective(trial, X_raw, y, tscv):
    """TimeSeriesSplit CV RMSE minimize — overfitting riskine karsi."""
    params = dict(
        n_estimators     = trial.suggest_int  ('n_estimators', 50, 300),
        max_depth        = trial.suggest_int  ('max_depth', 2, 4),
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
        Xf_s = sc.fit_transform(X_raw[tr_i])
        Xv_s = sc.transform(X_raw[vl_i])
        m = XGBRegressor(**params)
        m.fit(Xf_s, y[tr_i], eval_set=[(Xv_s, y[vl_i])], verbose=False)
        fold_rmse.append(np.sqrt(mean_squared_error(y[vl_i], m.predict(Xv_s))))
    return np.mean(fold_rmse)

# --- Grafik Fonksiyonlari (sadece Turkiye icin cagrilir) ----------------------

def ciz_zaman_serisi(yillar_g, y_g, yillar_t, y_t, feature, yol):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(yillar_g, y_g, 'k-o', ms=4, lw=2, label='Gercek')
    ax.plot(yillar_t, y_t, 'r--s', ms=4, label='Tahmin (2021-2030)')
    ax.axvspan(VAL_ILK,  VAL_SON  + 0.9, alpha=0.12, color='green',  label='Validation')
    ax.axvspan(TEST_ILK, TEST_SON + 0.9, alpha=0.12, color='orange', label='Test')
    ax.axvspan(FC_ILK,   FC_SON   + 0.9, alpha=0.12, color='red',    label='Forecast')
    ax.set_title(f'Asama 1 | {feature} (Turkiye): Gercek vs Tahmin')
    ax.set_xlabel('Yil'); ax.set_ylabel(feature); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()

def ciz_residual(y_g, y_t, feature, bolum, yol):
    if len(y_g) == 0:
        return
    res = np.array(y_g) - np.array(y_t)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_t, res, color='steelblue', edgecolors='k', s=60)
    axes[0].axhline(0, color='red', lw=1.5)
    axes[0].set_xlabel('Tahmin'); axes[0].set_ylabel('Residual')
    axes[0].set_title(f'{feature} — Residual Scatter ({bolum})')
    axes[1].hist(res, bins=max(3, len(res) // 2), color='steelblue', edgecolor='k')
    axes[1].axvline(0, color='red', lw=1.5)
    axes[1].set_xlabel('Residual'); axes[1].set_ylabel('Frekans')
    axes[1].set_title(f'{feature} — Hata Dagilimi ({bolum})')
    plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()

def ciz_hata_bandi(bandlar, feature, bolum, yol):
    fig, ax = plt.subplots(figsize=(8, 4))
    renkler = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
    bars = ax.bar(list(bandlar.keys()), list(bandlar.values()),
                  color=renkler, edgecolor='k')
    for bar, val in zip(bars, bandlar.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                str(val), ha='center', va='bottom', fontweight='bold')
    ax.set_title(f'{feature} — Hata Bandi ({bolum}) | Turkiye\n'
                 f'(Regression Confusion Matrix Karsiligi)')
    ax.set_ylabel('Tahmin Sayisi')
    ax.set_ylim(0, max(bandlar.values()) + 1.5 if max(bandlar.values()) > 0 else 2)
    plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()

def ciz_cv(cv_tr, cv_vl, feature, yol):
    folds = list(range(1, len(cv_tr) + 1))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(folds, cv_tr, 'b-o', label='Train RMSE')
    ax.plot(folds, cv_vl, 'r-o', label='Val RMSE')
    ax.set_title(f'{feature} — Cross-Validation (Turkiye, TimeSeriesSplit)')
    ax.set_xlabel('Fold'); ax.set_ylabel('RMSE'); ax.legend()
    plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()

def ciz_learning_curve(X_s, y, feature, tscv, yol):
    n = len(y)
    boyutlar = sorted(set(
        [int(n * r) for r in [0.4, 0.55, 0.7, 0.85, 1.0] if int(n * r) > 3]
    ))
    tr_means, vl_means, gec_boyut = [], [], []
    for boyut in boyutlar:
        X_sub, y_sub = X_s[:boyut], y[:boyut]
        fold_tr, fold_vl = [], []
        for tr_i, vl_i in tscv.split(X_sub):
            if len(tr_i) < 2 or len(vl_i) < 1:
                continue
            sc = MinMaxScaler()
            Xft = sc.fit_transform(X_sub[tr_i])
            Xfv = sc.transform(X_sub[vl_i])
            m = XGBRegressor(max_depth=3, n_estimators=100,
                             reg_alpha=0.5, reg_lambda=1.0,
                             verbosity=0, random_state=42)
            m.fit(Xft, y_sub[tr_i])
            fold_tr.append(np.sqrt(mean_squared_error(y_sub[tr_i], m.predict(Xft))))
            fold_vl.append(np.sqrt(mean_squared_error(y_sub[vl_i], m.predict(Xfv))))
        if fold_tr:
            tr_means.append(np.mean(fold_tr))
            vl_means.append(np.mean(fold_vl))
            gec_boyut.append(boyut)
    if len(tr_means) < 2:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gec_boyut, tr_means, 'b-o', label='Train RMSE')
    ax.plot(gec_boyut, vl_means, 'r-o', label='Val RMSE')
    ax.set_title(f'{feature} — Learning Curve (Turkiye)')
    ax.set_xlabel('Egitim Ornegi Sayisi'); ax.set_ylabel('RMSE'); ax.legend()
    plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()

def ciz_shap(model, X_s, feature_names, feature, yol):
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_s)
        mean_abs  = np.abs(shap_vals).mean(axis=0)
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(feature_names, mean_abs, color='steelblue', edgecolor='k')
        for bar, val in zip(bars, mean_abs):
            ax.text(val + mean_abs.max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=9)
        ax.set_title(f'{feature} — SHAP Feature Importance (Turkiye)')
        ax.set_xlabel('Ort |SHAP Degeri|')
        plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()
    except Exception as e:
        print(f"    [SHAP cizilmedi: {e}]")

# --- Veri Yukle ---------------------------------------------------------------
print("=" * 70)
print("ASAMA 1 — TUM ULKELER FEATURE TAHMINI (Optuna + SHAP)")
print("=" * 70)

train_df = pd.read_csv(os.path.join(ISLENMIS_DIR, 'train.csv'))
val_df   = pd.read_csv(os.path.join(ISLENMIS_DIR, 'val.csv'))
test_df  = pd.read_csv(os.path.join(ISLENMIS_DIR, 'test.csv'))

feat_conf    = pd.read_csv(os.path.join(ISLENMIS_DIR, 'feature_listesi.csv'))
ALL_FEATURES = feat_conf['tum_features'].dropna().tolist()

ulkeler = sorted(train_df['country'].unique())
print(f"  Ulke sayisi   : {len(ulkeler)}")
print(f"  Feature sayisi: {len(ALL_FEATURES)}")
print(f"  Featureler    : {ALL_FEATURES}")
print()

# --- Ana Dongu: Her Feature x Her Ulke ----------------------------------------

tahmin_dict_global = {}   # {(country, year): {feature: val, ...}}
metrik_listesi     = []

for feat_idx, feature in enumerate(ALL_FEATURES):
    print(f"\n{'='*70}")
    print(f"  [{feat_idx+1}/{len(ALL_FEATURES)}] FEATURE: {feature}")
    print(f"{'='*70}")

    ulke_modeller = {}   # {country: (model, scaler)}

    for ulke in ulkeler:
        tr_u  = train_df[train_df['country'] == ulke].sort_values('year')
        val_u = val_df  [val_df  ['country'] == ulke].sort_values('year')
        tst_u = test_df [test_df ['country'] == ulke].sort_values('year')

        # Bu feature bu ulkede var mi?
        if feature not in tr_u.columns or tr_u[feature].isna().all():
            continue
        tr_u = tr_u[tr_u[feature].notna()]
        if len(tr_u) < 3:
            continue

        # Lag ozellikleri olustur
        lag_tr = lag_df_olustur(tr_u['year'].values, tr_u[feature].values, LAG_SAYISI)
        if len(lag_tr) < 2:
            continue
        X_tr, y_tr = X_y_ayir(lag_tr)

        # Val / test lag (gercek tarihsel veriden)
        gercek_u = pd.concat([tr_u, val_u, tst_u]).sort_values('year')
        if feature in gercek_u.columns:
            gercek_u = gercek_u[gercek_u[feature].notna()]
        lag_all  = lag_df_olustur(gercek_u['year'].values,
                                   gercek_u[feature].values, LAG_SAYISI)
        lag_val_ = lag_all[(lag_all['year'] >= VAL_ILK)  & (lag_all['year'] <= VAL_SON)]
        lag_tst_ = lag_all[(lag_all['year'] >= TEST_ILK) & (lag_all['year'] <= TEST_SON)]

        X_val_, y_val_  = (X_y_ayir(lag_val_)  if len(lag_val_)  > 0
                           else (np.empty((0, 3)), np.array([])))
        X_tst_, y_tst_  = (X_y_ayir(lag_tst_)  if len(lag_tst_)  > 0
                           else (np.empty((0, 3)), np.array([])))

        yeterli = len(X_tr) >= MIN_NOKTA

        # Aday modeller
        if yeterli:
            n_splits_u = min(N_CV_SPLITS, len(X_tr) - 1)
            tscv_u     = TimeSeriesSplit(n_splits=n_splits_u)
            study = optuna.create_study(
                direction='minimize',
                sampler=optuna.samplers.TPESampler(seed=42)
            )
            study.optimize(
                lambda trial: optuna_objective(trial, X_tr, y_tr, tscv_u),
                n_trials=OPTUNA_TRIAL,
                show_progress_bar=False,
            )
            adaylar = {
                'Linear' : make_pipeline(PolynomialFeatures(1), LinearRegression()),
                'Poly2'  : make_pipeline(PolynomialFeatures(2), LinearRegression()),
                'XGBoost': XGBRegressor(**study.best_params, verbosity=0),
            }
        else:
            adaylar = {
                'Linear': make_pipeline(PolynomialFeatures(1), LinearRegression()),
            }

        # Val RMSE ile en iyi model sec
        if len(X_val_) > 0:
            en_iyi_isim, en_iyi_rmse, en_iyi_model = None, float('inf'), None
            for isim, m in adaylar.items():
                sc_t = MinMaxScaler()
                m.fit(sc_t.fit_transform(X_tr), y_tr)
                rmse_t = np.sqrt(mean_squared_error(
                    y_val_, m.predict(sc_t.transform(X_val_))))
                if rmse_t < en_iyi_rmse:
                    en_iyi_rmse  = rmse_t
                    en_iyi_isim  = isim
                    en_iyi_model = m
        else:
            en_iyi_isim  = 'Linear'
            en_iyi_model = make_pipeline(PolynomialFeatures(1), LinearRegression())

        # Final model: tum train uzerinde
        final_sc = MinMaxScaler()
        Xtr_s_f  = final_sc.fit_transform(X_tr)
        final_m  = clone(en_iyi_model)
        final_m.fit(Xtr_s_f, y_tr)

        ulke_modeller[ulke] = (final_m, final_sc)

        # Tahmin 2021-2030 iteratif
        history_u = dict(zip(gercek_u['year'].values.tolist(),
                             gercek_u[feature].values.tolist()))
        tahmin_u  = {}
        for yil in TAHMIN_YILLARI:
            lag1 = history_u.get(yil - 1, np.nan)
            lag2 = history_u.get(yil - 2, np.nan)
            # Lag eksikse son bilinen degeri kullan
            if np.isnan(lag1):
                bilinen = [v for k, v in history_u.items() if k < yil]
                lag1 = bilinen[-1] if bilinen else 0.0
            if np.isnan(lag2):
                lag2 = lag1
            x_row   = np.array([[yil, lag1, lag2]])
            pred    = final_m.predict(final_sc.transform(x_row))[0]
            tahmin_u[yil]  = pred
            history_u[yil] = pred

        # Global tahmin dict'e kaydet
        iso = tr_u['iso_code'].iloc[0] if 'iso_code' in tr_u.columns else ''
        for yil in TAHMIN_YILLARI:
            key = (ulke, yil)
            if key not in tahmin_dict_global:
                tahmin_dict_global[key] = {'country': ulke, 'iso_code': iso, 'year': yil}
            tahmin_dict_global[key][feature] = round(float(tahmin_u[yil]), 6)

        # --- Turkiye icin ayrintili analiz ve grafikler -----------------------
        is_turkey = (ulke == TURKIYE_ADI)

        if is_turkey:
            tscv_tr = TimeSeriesSplit(n_splits=min(N_CV_SPLITS, len(X_tr) - 1))
            cv_tr_rmse, cv_vl_rmse = [], []
            for tr_i, vl_i in tscv_tr.split(X_tr):
                sc_f = MinMaxScaler()
                Xft  = sc_f.fit_transform(X_tr[tr_i])
                Xfv  = sc_f.transform(X_tr[vl_i])
                fm   = clone(en_iyi_model)
                fm.fit(Xft, y_tr[tr_i])
                cv_tr_rmse.append(np.sqrt(mean_squared_error(
                    y_tr[tr_i], fm.predict(Xft))))
                cv_vl_rmse.append(np.sqrt(mean_squared_error(
                    y_tr[vl_i], fm.predict(Xfv))))

            # Tahmin metrikleri
            val_pred  = (final_m.predict(final_sc.transform(X_val_))
                         if len(X_val_) > 0 else np.array([]))
            test_pred = (final_m.predict(final_sc.transform(X_tst_))
                         if len(X_tst_) > 0 else np.array([]))

            val_m  = metrik_hesapla(y_val_, val_pred)
            test_m = metrik_hesapla(y_tst_, test_pred)
            val_b  = hata_bandi(y_val_, val_pred)
            test_b = hata_bandi(y_tst_, test_pred)

            print(f"  Turkiye | {en_iyi_isim} | "
                  f"Val MAPE:{val_m['mape']:.2f}%  R2:{val_m['r2']:.4f} | "
                  f"Test MAPE:{test_m['mape']:.2f}%  R2:{test_m['r2']:.4f}")
            print(f"  Forecast 2026-2030:")
            for y in range(FC_ILK, FC_SON + 1):
                print(f"    {y}: {tahmin_u[y]:,.4f}")

            metrik_listesi.append({
                'feature': feature, 'country': ulke,
                'secilen_model': en_iyi_isim,
                'val_mape': val_m['mape'], 'val_r2': val_m['r2'],
                'test_mape': test_m['mape'], 'test_r2': test_m['r2'],
                'cv_val_rmse_ort': round(np.mean(cv_vl_rmse), 4) if cv_vl_rmse else np.nan,
                'cv_train_rmse_ort': round(np.mean(cv_tr_rmse), 4) if cv_tr_rmse else np.nan,
            })

            # Grafikler
            all_t = [tahmin_u[y] for y in TAHMIN_YILLARI]
            ciz_zaman_serisi(
                gercek_u['year'].values, gercek_u[feature].values,
                TAHMIN_YILLARI, all_t, feature,
                os.path.join(GORSELLER_DIR, f'asama1_{feature}_zaman.png'))

            if len(val_pred) > 0:
                ciz_residual(y_val_, val_pred, feature, 'Validation',
                    os.path.join(GORSELLER_DIR, f'asama1_{feature}_residual_val.png'))
                ciz_hata_bandi(val_b, feature, 'Validation',
                    os.path.join(GORSELLER_DIR, f'asama1_{feature}_hata_bandi_val.png'))
            if len(test_pred) > 0:
                ciz_residual(y_tst_, test_pred, feature, 'Test',
                    os.path.join(GORSELLER_DIR, f'asama1_{feature}_residual_test.png'))
                ciz_hata_bandi(test_b, feature, 'Test',
                    os.path.join(GORSELLER_DIR, f'asama1_{feature}_hata_bandi_test.png'))
            if cv_tr_rmse:
                ciz_cv(cv_tr_rmse, cv_vl_rmse, feature,
                    os.path.join(GORSELLER_DIR, f'asama1_{feature}_cv.png'))
                ciz_learning_curve(Xtr_s_f, y_tr, feature, tscv_tr,
                    os.path.join(GORSELLER_DIR, f'asama1_{feature}_learning_curve.png'))
            if en_iyi_isim == 'XGBoost':
                ciz_shap(final_m, Xtr_s_f, ['year', 'lag_1', 'lag_2'], feature,
                    os.path.join(GORSELLER_DIR, f'asama1_{feature}_shap.png'))

        else:
            # Turkiye disi: sadece val metrigi (varsa)
            if len(X_val_) > 0:
                val_pred = final_m.predict(final_sc.transform(X_val_))
                vm = metrik_hesapla(y_val_, val_pred)
                metrik_listesi.append({
                    'feature': feature, 'country': ulke,
                    'secilen_model': en_iyi_isim,
                    'val_mape': vm['mape'], 'val_r2': vm['r2'],
                    'test_mape': np.nan, 'test_r2': np.nan,
                    'cv_val_rmse_ort': np.nan, 'cv_train_rmse_ort': np.nan,
                })

    # Feature basina 1 model dosyasi: {country: (model, scaler)}
    joblib.dump(ulke_modeller,
                os.path.join(MODELLER_DIR, f'asama1_{feature}.pkl'))
    print(f"\n  -> {len(ulke_modeller)} ulke modeli kaydedildi: asama1_{feature}.pkl")

# --- Cikti CSV'leri -----------------------------------------------------------
tahmin_rows = list(tahmin_dict_global.values())
tahmin_df   = (pd.DataFrame(tahmin_rows)
               .sort_values(['country', 'year'])
               .reset_index(drop=True))

# Sutun sirasi: country, iso_code, year, feature1, feature2, ...
cols = ['country', 'iso_code', 'year'] + [f for f in ALL_FEATURES
                                           if f in tahmin_df.columns]
tahmin_df = tahmin_df[[c for c in cols if c in tahmin_df.columns]]

tahmin_df.to_csv(
    os.path.join(CIKTILAR_DIR, 'feature_tahminleri_2021_2030.csv'), index=False)
pd.DataFrame(metrik_listesi).to_csv(
    os.path.join(CIKTILAR_DIR, 'asama1_metrikler.csv'), index=False)

# --- Ozet ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("ASAMA 1 TAMAMLANDI")
print("=" * 70)
print(f"  Tahmin: {tahmin_df.shape[0]} satir "
      f"({tahmin_df['country'].nunique()} ulke x 10 yil)")

metrik_df = pd.DataFrame(metrik_listesi)
if len(metrik_df) > 0:
    tr_ozet = metrik_df[metrik_df['country'] == TURKIYE_ADI]
    if len(tr_ozet) > 0:
        print("\nTurkiye Ozet Metrikleri:")
        pd.set_option('display.width', 120)
        pd.set_option('display.float_format', '{:.4f}'.format)
        print(tr_ozet[['feature', 'secilen_model',
                        'val_mape', 'val_r2',
                        'test_mape', 'test_r2',
                        'cv_val_rmse_ort']].to_string(index=False))

print("\nKaydedilen dosyalar:")
print(f"  ciktilar/feature_tahminleri_2021_2030.csv")
print(f"  ciktilar/asama1_metrikler.csv")
print(f"  modeller/asama1_{{feature}}.pkl  (her feature icin 1 dosya)")
print(f"  sonuclar/gorseller/asama1_*  (Turkiye grafikleri)")
