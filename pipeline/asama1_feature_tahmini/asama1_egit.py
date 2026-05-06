"""
AŞAMA 1 — Feature Tahmini (Walk-Forward CV + Optuna)
-----------------------------------------------------
Tüm ülkeler için 19 parametrenin 2021-2030 tahminini yapar.
Her ülke × her feature için ayrı model eğitilir.

DEĞERLENDİRME: Walk-Forward Expanding Window Cross-Validation
  - Min 10 yıl eğitim → sonraki yılı tahmin et → pencereyi genişlet
  - Çok sayıda out-of-sample tahmin → anlamlı R²

DATA LEAKAGE GÜVENCESİ (kod boyunca uygulanır):
  ✓ Her ülke kendi başına eğitilir (ülkeler birbirini etkilemez)
  ✓ Walk-forward CV: train[0..t-1] → tahmin[t]; gelecek bilgi sızmaz
  ✓ MinMaxScaler her fold'da YALNIZCA train fold'a fit edilir
  ✓ Lag feature'lar yalnızca t-1, t-2 (geçmiş)
  ✓ Optuna TimeSeriesSplit kullanır (random shuffle YOK)
  ✓ Forecast 2026-2030: model kendi tahminini lag'ler (iteratif, gerçek değer kullanmaz)

CHECKPOINT SİSTEMİ:
  ✓ Her feature bittiğinde tahminler + metrikler CSV'ye atomic yazılır
  ✓ Yeniden başlatıldığında tamamlanmış feature'lar otomatik atlanır
  ✓ Model dosyası (.pkl) varsa o feature resume ediliyor

GİRDİ:  veri/islenmis/train.csv, val.csv, test.csv
ÇIKTI:  ciktilar/feature_tahminleri_2021_2030.csv
        ciktilar/asama1_metrikler.csv
        modeller/asama1_{feature}.pkl
        sonuclar/gorseller/asama1_* (Türkiye grafikleri)
"""

import os, time, warnings
import numpy as np
import pandas as pd
import joblib
import optuna
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.linear_model    import LinearRegression
from sklearn.preprocessing   import PolynomialFeatures, MinMaxScaler
from sklearn.pipeline        import make_pipeline
from sklearn.metrics         import (mean_squared_error, mean_absolute_error,
                                      r2_score, confusion_matrix)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base            import clone
from xgboost                 import XGBRegressor

# ─── Sabitler ─────────────────────────────────────────────────────────────────
TAHMIN_YILLARI = list(range(2021, 2031))
LAG_SAYISI     = 2
OPTUNA_TRIAL   = 15   # 40→15: ~2.5x hizlanma (Population 2.21 saat surdu)
MIN_EGITIM     = 10   # walk-forward icin min egitim noktasi
MIN_NOKTA      = 6    # model egitmek icin min veri
TSCV_SPLITS    = 3    # min(5,...) → min(3,...): %40 hizlanma

TURKIYE_ADI = 'Turkey'

# ─── Yollar ───────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR  = os.path.dirname(SCRIPT_DIR)
ISLENMIS_DIR  = os.path.join(PIPELINE_DIR, 'veri', 'islenmis')
MODELLER_DIR  = os.path.join(SCRIPT_DIR, 'modeller')
CIKTILAR_DIR  = os.path.join(SCRIPT_DIR, 'ciktilar')
GORSELLER_DIR = os.path.join(PIPELINE_DIR, 'sonuclar', 'gorseller')
for d in [MODELLER_DIR, CIKTILAR_DIR, GORSELLER_DIR]:
    os.makedirs(d, exist_ok=True)

TAHMIN_CSV  = os.path.join(CIKTILAR_DIR, 'feature_tahminleri_2021_2030.csv')
METRIK_CSV  = os.path.join(CIKTILAR_DIR, 'asama1_metrikler.csv')

# ─── Yardımcı (matematik + zaman) ────────────────────────────────────────────

def lag_df_olustur(yillar, degerler, n_lag=2):
    df = pd.DataFrame({'year': yillar, 'value': degerler})
    for k in range(1, n_lag + 1):
        df[f'lag_{k}'] = df['value'].shift(k)
    return df.dropna().reset_index(drop=True)

def X_y_ayir(df):
    X = df[['year'] + [f'lag_{k}' for k in range(1, LAG_SAYISI + 1)]].values
    y = df['value'].values
    return X, y

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
    return {
        'wf_cv_r2'  : round(r2,   4),
        'wf_cv_mae' : round(mae,  4),
        'wf_cv_rmse': round(rmse, 4),
        'wf_cv_mape': round(mp,   4),
    }

def format_sure(saniye):
    if saniye < 60:    return f"{saniye:.1f}sn"
    if saniye < 3600:  return f"{saniye/60:.1f}dk"
    return f"{saniye/3600:.2f}saat"

# ─── Optuna + Walk-Forward CV ────────────────────────────────────────────────

def optuna_objective(trial, X_raw, y, tscv):
    params = dict(
        n_estimators     = trial.suggest_int('n_estimators', 50, 150),  # 300→150: ~%30 hizlanma
        max_depth        = trial.suggest_int('max_depth', 2, 4),
        learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        subsample        = trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0),
        reg_alpha        = trial.suggest_float('reg_alpha', 0.0, 2.0),
        reg_lambda       = trial.suggest_float('reg_lambda', 0.5, 3.0),
        random_state=42, verbosity=0,
    )
    rmses = []
    for tr_i, vl_i in tscv.split(X_raw):
        # LEAKAGE GUARD: scaler SADECE train fold'a fit ediliyor
        sc = MinMaxScaler()
        m = XGBRegressor(**params)
        m.fit(sc.fit_transform(X_raw[tr_i]), y[tr_i],
              eval_set=[(sc.transform(X_raw[vl_i]), y[vl_i])], verbose=False)
        rmses.append(np.sqrt(mean_squared_error(y[vl_i], m.predict(sc.transform(X_raw[vl_i])))))
    return np.mean(rmses)

def walk_forward_cv(X_all, y_all, model_factory, min_train=10):
    """Walk-forward expanding window CV.
    LEAKAGE GUARD: her fold'da scaler sadece train indekslerine fit edilir.
    Test indeksi (gelecek) train'i etkilemez."""
    preds, trues = [], []
    for test_idx in range(min_train, len(y_all)):
        train_idx = list(range(test_idx))   # 0..t-1
        X_tr, y_tr = X_all[train_idx], y_all[train_idx]
        X_te = X_all[test_idx:test_idx+1]   # sadece t
        sc = MinMaxScaler()
        m = model_factory()
        m.fit(sc.fit_transform(X_tr), y_tr)
        pred = m.predict(sc.transform(X_te))[0]
        preds.append(pred)
        trues.append(y_all[test_idx])
    return np.array(trues), np.array(preds)

# ─── Türkiye Grafikleri ──────────────────────────────────────────────────────

def ciz_zaman_serisi(yillar_g, y_g, yillar_t, y_t, cv_yillar, cv_preds, feature, yol):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(yillar_g, y_g, 'k-o', ms=4, lw=2, label='Gercek')
    if len(cv_yillar) > 0:
        ax.plot(cv_yillar, cv_preds, 'b--s', ms=4, label='Walk-Forward CV')
    ax.plot(yillar_t, y_t, 'r--^', ms=4, label='Forecast (2021-2030)')
    ax.set_title(f'Asama 1 | {feature} (Turkiye): Gercek vs Tahmin')
    ax.set_xlabel('Yil'); ax.set_ylabel(feature); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()

def ciz_cv_scatter(trues, preds, feature, r2, yol):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(trues, preds, color='steelblue', edgecolors='k', s=60, zorder=5)
    mn, mx = min(trues.min(), preds.min()), max(trues.max(), preds.max())
    margin = (mx - mn) * 0.05
    ax.plot([mn-margin, mx+margin], [mn-margin, mx+margin], 'r--', lw=1.5, label='Ideal')
    ax.set_xlabel('Gercek'); ax.set_ylabel('Tahmin')
    ax.set_title(f'{feature} (Turkiye) — Walk-Forward CV\nR² = {r2:.4f}')
    ax.legend(); plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()

def ciz_hata_bandi(trues, preds, feature, yol):
    """Regresyon icin 'confusion matrix' muadili: hata yuzdesi bandlari."""
    pct = np.abs((trues - preds) / trues) * 100
    bandlar = {
        'Mukemmel (<%5)': int(np.sum(pct < 5)),
        'Iyi (%5-10)':    int(np.sum((pct>=5)  & (pct<10))),
        'Orta (%10-20)':  int(np.sum((pct>=10) & (pct<20))),
        'Zayif (>%20)':   int(np.sum(pct >= 20)),
    }
    fig, ax = plt.subplots(figsize=(8, 4))
    renkler = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
    bars = ax.bar(list(bandlar.keys()), list(bandlar.values()), color=renkler, edgecolor='k')
    for bar, val in zip(bars, bandlar.values()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                str(val), ha='center', va='bottom', fontweight='bold')
    ax.set_title(f'{feature} (Turkiye) — Hata Bandi (Walk-Forward CV)')
    ax.set_ylabel('Tahmin Sayisi')
    ax.set_ylim(0, max(bandlar.values())+1.5 if max(bandlar.values())>0 else 2)
    plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()

def ciz_residual_dist(trues, preds, feature, yol):
    """Residual dagilimi — regresyon icin confusion matrix muadili (hata profili).
    Histogram + KDE + sifir cizgisi → hatalarin merkezi (bias) ve yayilim (variance)
    gorselleri."""
    res = trues - preds
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Histogram
    axes[0].hist(res, bins=min(15, max(5, len(res)//2)), color='steelblue', edgecolor='k')
    axes[0].axvline(0, color='red', lw=2, label='Sifir hata')
    axes[0].axvline(np.mean(res), color='orange', ls='--', lw=2,
                    label=f'Ortalama bias={np.mean(res):.3f}')
    axes[0].set_xlabel('Residual (Gercek - Tahmin)'); axes[0].set_ylabel('Frekans')
    axes[0].set_title(f'{feature} — Residual Dagilimi'); axes[0].legend(fontsize=8)
    # Q-Q (residual normal mi?)
    from scipy import stats
    stats.probplot(res, dist='norm', plot=axes[1])
    axes[1].set_title(f'{feature} — Q-Q Plot (Normallik kontrolu)')
    plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()

def ciz_confusion_matrix_regresyon(trues, preds, feature, yol):
    """Regresyon icin confusion matrix.
    SADECE BU FONKSIYONA OZEL: trues ve preds quintile sinirlarina gore 5 sinifa
    atilir (small / mid-small / mid / mid-large / large), sonra confusion matrix
    olusturulur. Bu binning sistemi BASKA ANALIZDE KULLANILMAZ — yalnizca
    confusion matrix gorseli icindir.
    """
    if len(trues) < 5:
        return  # quintile icin cok az veri

    # Quintile sınırları GERCEK degerlerden cikarilir (5 esit yogunluk siniri)
    sinir = np.quantile(trues, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    sinif_isimleri = ['small', 'mid-small', 'mid', 'mid-large', 'large']

    def _sinifla(d):
        for i in range(5):
            if d <= sinir[i+1]:
                return i
        return 4

    y_t_sinif = np.array([_sinifla(v) for v in trues])
    y_p_sinif = np.array([_sinifla(v) for v in preds])

    cm = confusion_matrix(y_t_sinif, y_p_sinif, labels=list(range(5)))

    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(cm, cmap='Blues', aspect='equal')
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(sinif_isimleri, rotation=30, ha='right')
    ax.set_yticklabels(sinif_isimleri)
    ax.set_xlabel('Tahmin Sinifi')
    ax.set_ylabel('Gercek Sinifi')
    ax.set_title(f'{feature} (Turkiye) — Confusion Matrix (Quintile bins)\n'
                 f'5 buyukluk sinifi: small / mid-small / mid / mid-large / large')

    # Hucre degerlerini yaz
    for i in range(5):
        for j in range(5):
            val = cm[i, j]
            ax.text(j, i, str(val), ha='center', va='center',
                    color='white' if val > cm.max()/2 else 'black',
                    fontsize=11, fontweight='bold')

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Tahmin Sayisi')

    # Diyagonal vurgu — ideal: tum noktalar diyagonal uzerinde
    dogru = np.trace(cm); toplam = cm.sum()
    accuracy = dogru / toplam if toplam else 0
    ax.text(0.02, -0.18, f'Dogru sinif: {dogru}/{toplam} (accuracy={accuracy:.2f})',
            transform=ax.transAxes, fontsize=10, color='darkgreen')

    plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()

def ciz_learning_curve_a1(X_all, y_all, model_factory, feature, min_train, yol):
    sizes, errors = [], []
    for size in range(min_train, len(y_all)-1):
        X_tr, y_tr = X_all[:size], y_all[:size]
        X_te, y_te = X_all[size:size+1], y_all[size:size+1]
        sc = MinMaxScaler()
        m = model_factory()
        m.fit(sc.fit_transform(X_tr), y_tr)
        pred = m.predict(sc.transform(X_te))[0]
        sizes.append(size)
        errors.append(abs(y_te[0] - pred))
    if len(sizes) < 2: return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sizes, errors, 'b-o', ms=4)
    ax.set_xlabel('Egitim Seti Boyutu (yil)'); ax.set_ylabel('Mutlak Hata')
    ax.set_title(f'{feature} (Turkiye) — Learning Curve\n'
                 f'(Egitim verisi arttikca hata nasil degisiyor?)')
    plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()

def _shap_explainer_olustur(model, X_s, feature_names):
    """Model tipine gore en uygun shap explainer'i dondurur.
    XGBoost  → TreeExplainer (hizli)
    sklearn  → shap.Explainer auto-detect veya KernelExplainer fallback
    """
    import shap
    # XGBoost (TreeExplainer en hizli ve dogru)
    if isinstance(model, XGBRegressor):
        return shap.TreeExplainer(model), 'tree'
    # sklearn Pipeline veya LinearRegression — shap.Explainer auto-detect
    try:
        # Yeni API: feature_names parametresi
        explainer = shap.Explainer(model.predict, X_s, feature_names=feature_names)
        return explainer, 'auto'
    except Exception:
        # Fallback: KernelExplainer (model-agnostic, yavas ama her zaman calisir)
        baseline = shap.utils.sample(X_s, min(50, len(X_s)))
        return shap.KernelExplainer(model.predict, baseline), 'kernel'

def _shap_degerleri(explainer, kind, X_s):
    """Explainer tipine gore SHAP degerlerini hesapla, numpy array olarak don."""
    if kind == 'tree':
        sv = explainer.shap_values(X_s)
        return np.array(sv)
    elif kind == 'auto':
        sv = explainer(X_s)
        return sv.values if hasattr(sv, 'values') else np.array(sv)
    else:  # kernel
        sv = explainer.shap_values(X_s, silent=True)
        return np.array(sv)

def ciz_shap_bar(model, X_s, feature_names, feature, yol):
    """SHAP bar chart — native shap.plots.bar.
    Tum modeller icin calisir (XGBoost, Linear, Poly2)."""
    try:
        import shap
        explainer, kind = _shap_explainer_olustur(model, X_s, feature_names)
        sv_arr = _shap_degerleri(explainer, kind, X_s)
        # shap.Explanation objesi olustur (native shap.plots.bar bunu ister)
        explanation = shap.Explanation(
            values=sv_arr,
            data=X_s,
            feature_names=feature_names,
        )
        plt.figure(figsize=(8, max(3, len(feature_names)*0.6)))
        shap.plots.bar(explanation, show=False, max_display=len(feature_names))
        plt.title(f'{feature} (Turkiye) — SHAP Bar (mean |SHAP|)\n'
                  f'Explainer: {kind}')
        plt.tight_layout()
        plt.savefig(yol, dpi=150, bbox_inches='tight')
        plt.close('all')
    except Exception as e:
        print(f"    [SHAP bar atlandi: {e}]")

def ciz_shap_beeswarm(model, X_s, feature_names, feature, yol):
    """SHAP beeswarm (baloncuklu) — native shap.plots.beeswarm.
    Her nokta bir gozlem; renk feature degeri, x-eksen SHAP."""
    try:
        import shap
        explainer, kind = _shap_explainer_olustur(model, X_s, feature_names)
        sv_arr = _shap_degerleri(explainer, kind, X_s)
        explanation = shap.Explanation(
            values=sv_arr,
            data=X_s,
            feature_names=feature_names,
        )
        plt.figure(figsize=(9, max(4, len(feature_names)*0.6)))
        shap.plots.beeswarm(explanation, show=False, max_display=len(feature_names))
        plt.title(f'{feature} (Turkiye) — SHAP Beeswarm (her nokta=bir yil)\n'
                  f'Explainer: {kind}')
        plt.tight_layout()
        plt.savefig(yol, dpi=150, bbox_inches='tight')
        plt.close('all')
    except Exception as e:
        print(f"    [SHAP beeswarm atlandi: {e}]")

def ciz_feature_importance_builtin(model, feature_names, feature, yol):
    """Model'in built-in feature_importances_ - SHAP'tan farkli, training-time bazli."""
    try:
        if not hasattr(model, 'feature_importances_'):
            return
        imp = model.feature_importances_
        idx = np.argsort(imp)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh([feature_names[i] for i in idx], imp[idx], color='coral', edgecolor='k')
        for i, val in enumerate(imp[idx]):
            ax.text(val + imp.max()*0.01, i, f'{val:.4f}', va='center', fontsize=9)
        ax.set_xlabel('Feature Importance (built-in)')
        ax.set_title(f'{feature} (Turkiye) — Built-in Feature Importance\n'
                     f'(Egitim sirasinda modelin verdigi agirliklar)')
        plt.tight_layout(); plt.savefig(yol, dpi=150); plt.close()
    except Exception as e:
        print(f"    [Built-in importance atlandi: {e}]")

# ─── Checkpoint sistemi ──────────────────────────────────────────────────────

def checkpoint_kaydet(tahmin_dict_global, metrik_listesi, all_features):
    """Atomic CSV yazimi: once .tmp, sonra rename."""
    if tahmin_dict_global:
        rows = list(tahmin_dict_global.values())
        df = pd.DataFrame(rows).sort_values(['country', 'year']).reset_index(drop=True)
        cols = ['country', 'iso_code', 'year'] + [f for f in all_features if f in df.columns]
        df = df[[c for c in cols if c in df.columns]]
        tmp = TAHMIN_CSV + '.tmp'
        df.to_csv(tmp, index=False)
        os.replace(tmp, TAHMIN_CSV)

    if metrik_listesi:
        df = pd.DataFrame(metrik_listesi)
        tmp = METRIK_CSV + '.tmp'
        df.to_csv(tmp, index=False)
        os.replace(tmp, METRIK_CSV)

def checkpoint_yukle():
    """Onceden tamamlanmis tahminleri ve metrikleri yukle, idempotent."""
    tahmin_dict_global = {}
    metrik_listesi = []
    tamamlanmis_features = set()  # model dosyasi olan feature'lar

    if os.path.exists(TAHMIN_CSV):
        try:
            df = pd.read_csv(TAHMIN_CSV)
            for _, row in df.iterrows():
                key = (row['country'], int(row['year']))
                tahmin_dict_global[key] = row.to_dict()
            print(f"  [RESUME] {TAHMIN_CSV} yuklendi: "
                  f"{len(tahmin_dict_global)} satir, "
                  f"{df['country'].nunique()} ulke")
        except Exception as e:
            print(f"  [RESUME UYARI] Tahmin CSV okunamadi: {e}")

    if os.path.exists(METRIK_CSV):
        try:
            df = pd.read_csv(METRIK_CSV)
            metrik_listesi = df.to_dict('records')
            print(f"  [RESUME] {METRIK_CSV} yuklendi: {len(metrik_listesi)} satir")
        except Exception as e:
            print(f"  [RESUME UYARI] Metrik CSV okunamadi: {e}")

    # Hangi feature'larin model dosyasi var? (bu = tamamlanmis demek)
    if os.path.exists(MODELLER_DIR):
        for fname in os.listdir(MODELLER_DIR):
            if fname.startswith('asama1_') and fname.endswith('.pkl'):
                feat_name = fname[len('asama1_'):-len('.pkl')]
                tamamlanmis_features.add(feat_name)
        if tamamlanmis_features:
            print(f"  [RESUME] {len(tamamlanmis_features)} feature modeli mevcut, "
                  f"atlanacak: {sorted(tamamlanmis_features)}")

    return tahmin_dict_global, metrik_listesi, tamamlanmis_features

# ═════════════════════════════════════════════════════════════════════════════
# ANA SCRIPT
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("ASAMA 1 — TUM ULKELER FEATURE TAHMINI (Walk-Forward CV)")
print("=" * 70)

# Data leakage guvence log'u
print("\n[LEAKAGE GUVENCESI] Bu kod su mekanizmalari uygular:")
print("  ✓ Her ulke kendi basina egitilir, ulkeler arasi sızma yok")
print("  ✓ Walk-forward CV: train[0..t-1] → tahmin[t]; gelecek bilgisi sızmaz")
print("  ✓ MinMaxScaler her fold'da SADECE train fold'a fit edilir")
print("  ✓ Lag feature'lar yalnizca t-1, t-2 (gecmis), gelecek lag yok")
print("  ✓ Optuna TimeSeriesSplit: random shuffle YOK, zaman sirasi korunur")
print("  ✓ Forecast 2026-2030: model kendi tahminini lag'ler (iteratif)")
print()

# Veri yukle
print("Veri yukleniyor...")
train_df = pd.read_csv(os.path.join(ISLENMIS_DIR, 'train.csv'))
val_df   = pd.read_csv(os.path.join(ISLENMIS_DIR, 'val.csv'))
test_df  = pd.read_csv(os.path.join(ISLENMIS_DIR, 'test.csv'))

# Walk-forward icin tum veri birlestirilir (CV zaten zamansal split yapar)
all_df = pd.concat([train_df, val_df, test_df]).sort_values(
    ['country', 'year']).reset_index(drop=True)
print(f"  Toplam: {len(all_df)} satir, {all_df['country'].nunique()} ulke, "
      f"{all_df['year'].min()}-{all_df['year'].max()}")

feat_conf    = pd.read_csv(os.path.join(ISLENMIS_DIR, 'feature_listesi.csv'))
ALL_FEATURES = feat_conf["tum_features"].dropna().tolist()
print(f"  Feature sayisi: {len(ALL_FEATURES)}")

# Sadece Turkey mi, tum ulkeler mi?
SADECE_TURKEY = False  # False = tum 213 ulke (saatler)

# ─── Checkpoint yukle ────────────────────────────────────────────────────────
print("\nCheckpoint kontrolu...")
tahmin_dict_global, metrik_listesi, TAMAMLANMIS_FEATURES = checkpoint_yukle()

if SADECE_TURKEY:
    ulkeler = [TURKIYE_ADI]
    print(f"\n  MOD: SADECE TURKEY ({len(ulkeler)} ulke)")
else:
    ulkeler = sorted(all_df['country'].unique())
    print(f"\n  MOD: TUM ULKELER ({len(ulkeler)} ulke)")

print(f"  Egitilecek feature: {len(ALL_FEATURES)}")
print(f"  Tamamlanmis (atlanacak): {len(TAMAMLANMIS_FEATURES)}")
print(f"  Yapilacak: {len([f for f in ALL_FEATURES if f not in TAMAMLANMIS_FEATURES])}")

# ─── Ana Döngü ───────────────────────────────────────────────────────────────
t_overall_start = time.time()
feature_sureleri = []

for feat_idx, feature in enumerate(ALL_FEATURES):

    if feature in TAMAMLANMIS_FEATURES:
        print(f"\n[{feat_idx+1}/{len(ALL_FEATURES)}] {feature} → "
              f"[SKIP] model dosyasi var, resume.")
        continue

    t_feat_start = time.time()
    print(f"\n{'='*70}")
    print(f"  [{feat_idx+1}/{len(ALL_FEATURES)}] FEATURE: {feature}")
    if feature_sureleri:
        ort = sum(feature_sureleri) / len(feature_sureleri)
        kalan = (len(ALL_FEATURES) - feat_idx) * ort
        print(f"  Toplam gecen: {format_sure(time.time()-t_overall_start)} | "
              f"Ortalama feature: {format_sure(ort)} | "
              f"Tahmini kalan: ~{format_sure(kalan)}")
    print(f"{'='*70}")

    ulke_modeller = {}
    ulke_sayac = 0
    ulke_basarili = 0

    for ulke in ulkeler:
        ulke_sayac += 1
        u_df = all_df[all_df['country'] == ulke].sort_values('year')

        # Feature var mi, yeterli veri var mi?
        if feature not in u_df.columns or u_df[feature].isna().all():
            continue
        u_df = u_df[u_df[feature].notna()]
        if len(u_df) < 3:
            continue

        # Lag olustur
        lag_df = lag_df_olustur(u_df['year'].values, u_df[feature].values, LAG_SAYISI)
        if len(lag_df) < 4:
            continue
        X_all, y_all = X_y_ayir(lag_df)

        yeterli = len(X_all) >= MIN_NOKTA

        # ── Optuna (TimeSeriesSplit, leakage'siz) ──
        if yeterli:
            n_splits = min(TSCV_SPLITS, len(X_all) - 2)
            tscv = TimeSeriesSplit(n_splits=n_splits)
            study = optuna.create_study(direction='minimize',
                        sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(
                lambda trial: optuna_objective(trial, X_all, y_all, tscv),
                n_trials=OPTUNA_TRIAL, show_progress_bar=False)
            adaylar = {
                'Linear' : make_pipeline(PolynomialFeatures(1), LinearRegression()),
                'Poly2'  : make_pipeline(PolynomialFeatures(2), LinearRegression()),
                'XGBoost': XGBRegressor(**study.best_params, verbosity=0),
            }
        else:
            adaylar = {
                'Linear': make_pipeline(PolynomialFeatures(1), LinearRegression()),
            }

        # ── Walk-Forward CV: en iyi modeli sec ──
        en_iyi_isim, en_iyi_r2 = None, -999
        en_iyi_metrik = None
        en_iyi_trues, en_iyi_preds = None, None
        min_tr = max(min(MIN_EGITIM, len(X_all) - 3), 3)

        cv_sonuclari = {}
        for isim, m_template in adaylar.items():
            trues, preds = walk_forward_cv(
                X_all, y_all,
                lambda mt=m_template: clone(mt) if hasattr(mt, 'get_params') else mt,
                min_train=min_tr
            )
            if len(trues) > 2:
                m = metrikleri_hesapla(trues, preds)
                cv_sonuclari[isim] = (trues, preds, m)
                if m['wf_cv_r2'] > en_iyi_r2:
                    en_iyi_r2     = m['wf_cv_r2']
                    en_iyi_isim   = isim
                    en_iyi_metrik = m
                    en_iyi_trues  = trues
                    en_iyi_preds  = preds

        if en_iyi_isim is None:
            en_iyi_isim   = 'Linear'
            en_iyi_metrik = {'wf_cv_r2': float('nan'), 'wf_cv_mae': float('nan'),
                             'wf_cv_rmse': float('nan'), 'wf_cv_mape': float('nan')}

        # ── Final model: tum veri uzerinde egit (forecast icin) ──
        final_sc = MinMaxScaler()
        if en_iyi_isim == 'XGBoost' and yeterli:
            final_m = XGBRegressor(**study.best_params, verbosity=0)
        elif en_iyi_isim == 'Poly2':
            final_m = make_pipeline(PolynomialFeatures(2), LinearRegression())
        else:
            final_m = make_pipeline(PolynomialFeatures(1), LinearRegression())

        final_m.fit(final_sc.fit_transform(X_all), y_all)
        ulke_modeller[ulke] = (final_m, final_sc)
        ulke_basarili += 1

        # ── 2021-2030 ITERATIF tahmin (leakage guvenligi: model kendi tahminini lag'ler) ──
        history_u = dict(zip(u_df['year'].values.tolist(), u_df[feature].values.tolist()))
        tahmin_u = {}
        # NaN/inf koruma: feature'in tarihsel min-max'i (Poly2 patlamasin diye sınır)
        feat_min = float(np.nanmin(u_df[feature].values))
        feat_max = float(np.nanmax(u_df[feature].values))
        feat_range = max(feat_max - feat_min, 1e-6)
        # Tahminin max/min sınırı: tarihsel aralığın 5x'i (gercek dunya icin makul)
        tahmin_alt = feat_min - 5 * feat_range
        tahmin_ust = feat_max + 5 * feat_range

        for yil in TAHMIN_YILLARI:
            lag1 = history_u.get(yil - 1, np.nan)
            lag2 = history_u.get(yil - 2, np.nan)
            if not np.isfinite(lag1):
                bilinen = [v for k, v in history_u.items() if k < yil and np.isfinite(v)]
                lag1 = bilinen[-1] if bilinen else 0.0
            if not np.isfinite(lag2):
                lag2 = lag1
            # Cok buyuk lag'leri sinirla (Poly2 patlamasin)
            lag1 = float(np.clip(lag1, tahmin_alt, tahmin_ust))
            lag2 = float(np.clip(lag2, tahmin_alt, tahmin_ust))

            x_in = np.array([[float(yil), lag1, lag2]])
            x_in = np.nan_to_num(x_in, nan=0.0, posinf=1e6, neginf=-1e6)
            try:
                pred = final_m.predict(final_sc.transform(x_in))[0]
            except Exception:
                pred = lag1  # patlarsa son lag'i kullan
            # Tahmin patlarsa veya nan ise: son bilinen degeri kullan
            if not np.isfinite(pred):
                pred = lag1
            # Tahminin tarihsel araliga yakin olmasini sagla (Poly2 dispersi engelle)
            pred = float(np.clip(pred, tahmin_alt, tahmin_ust))
            tahmin_u[yil] = pred
            history_u[yil] = pred  # iteratif: tahmini lag olarak kullan

        # Global tahmin dict'e kaydet
        iso = u_df['iso_code'].iloc[0] if 'iso_code' in u_df.columns else ''
        for yil in TAHMIN_YILLARI:
            key = (ulke, yil)
            if key not in tahmin_dict_global:
                tahmin_dict_global[key] = {'country': ulke, 'iso_code': iso, 'year': yil}
            tahmin_dict_global[key][feature] = round(float(tahmin_u[yil]), 6)

        # ── Metriği listeye ekle (her ulke icin) ──
        metrik_satir = {
            'feature': feature, 'country': ulke,
            'secilen_model': en_iyi_isim,
            **en_iyi_metrik,
            'wf_cv_n': len(en_iyi_trues) if en_iyi_trues is not None else 0,
        }
        metrik_listesi.append(metrik_satir)

        # ── Türkiye: detaylı analiz + grafikler ──
        if ulke == TURKIYE_ADI:
            print(f"  [TURKEY] Modeller test edildi:")
            for isim, (_, _, m) in cv_sonuclari.items():
                marker = " ***" if isim == en_iyi_isim else ""
                print(f"    {isim:<10} R²={m['wf_cv_r2']:.4f}  "
                      f"MAE={m['wf_cv_mae']:.4f}  "
                      f"RMSE={m['wf_cv_rmse']:.4f}  "
                      f"MAPE={m['wf_cv_mape']:.2f}%{marker}")
            print(f"  [TURKEY] SECILEN: {en_iyi_isim} | R²={en_iyi_r2:.4f}")
            print(f"  [TURKEY] Forecast 2026-2030:")
            for y in range(2026, 2031):
                print(f"    {y}: {tahmin_u[y]:,.4f}")

            cv_yillar_tr = lag_df['year'].values[min_tr:]

            # Grafikler
            all_tahmin = [tahmin_u[y] for y in TAHMIN_YILLARI]
            ciz_zaman_serisi(
                u_df['year'].values, u_df[feature].values,
                TAHMIN_YILLARI, all_tahmin,
                cv_yillar_tr, en_iyi_preds, feature,
                os.path.join(GORSELLER_DIR, f'asama1_{feature}_zaman.png'))
            if en_iyi_trues is not None and len(en_iyi_trues) > 2:
                ciz_cv_scatter(
                    en_iyi_trues, en_iyi_preds, feature, en_iyi_r2,
                    os.path.join(GORSELLER_DIR, f'asama1_{feature}_cv_scatter.png'))
                ciz_hata_bandi(
                    en_iyi_trues, en_iyi_preds, feature,
                    os.path.join(GORSELLER_DIR, f'asama1_{feature}_hata_bandi.png'))
                ciz_residual_dist(
                    en_iyi_trues, en_iyi_preds, feature,
                    os.path.join(GORSELLER_DIR, f'asama1_{feature}_residual_dist.png'))
                ciz_confusion_matrix_regresyon(
                    en_iyi_trues, en_iyi_preds, feature,
                    os.path.join(GORSELLER_DIR, f'asama1_{feature}_confusion_matrix.png'))
                ciz_learning_curve_a1(
                    X_all, y_all,
                    lambda: clone(final_m) if hasattr(final_m, 'get_params') else final_m,
                    feature, min_tr,
                    os.path.join(GORSELLER_DIR, f'asama1_{feature}_learning_curve.png'))
                # SHAP her model icin (XGBoost / Linear / Poly2)
                feature_names = ['year', 'lag_1', 'lag_2']
                X_s = final_sc.transform(X_all)
                ciz_shap_bar(
                    final_m, X_s, feature_names, feature,
                    os.path.join(GORSELLER_DIR, f'asama1_{feature}_shap_bar.png'))
                ciz_shap_beeswarm(
                    final_m, X_s, feature_names, feature,
                    os.path.join(GORSELLER_DIR, f'asama1_{feature}_shap_beeswarm.png'))
                # Built-in feature_importances_ sadece tree-based modeller'de var
                if hasattr(final_m, 'feature_importances_'):
                    ciz_feature_importance_builtin(
                        final_m, feature_names, feature,
                        os.path.join(GORSELLER_DIR, f'asama1_{feature}_importance.png'))

    # Model kaydet (TUM ulke modelleri tek pkl'de)
    joblib.dump(ulke_modeller, os.path.join(MODELLER_DIR, f'asama1_{feature}.pkl'))

    # ── CHECKPOINT FLUSH (bu feature bittiğinde) ──
    checkpoint_kaydet(tahmin_dict_global, metrik_listesi, ALL_FEATURES)

    # Süre raporla
    feat_sure = time.time() - t_feat_start
    feature_sureleri.append(feat_sure)
    print(f"\n  -> {ulke_basarili}/{ulke_sayac} ulke modeli kaydedildi "
          f"({format_sure(feat_sure)} surdu) ✓ checkpoint atildi")

# ─── Final özet ──────────────────────────────────────────────────────────────
toplam_sure = time.time() - t_overall_start
print("\n" + "=" * 70)
print(f"ASAMA 1 TAMAMLANDI ({format_sure(toplam_sure)})")
print("=" * 70)

# Final flush (zaten checkpoint'te yazıldı, ama emin olmak için)
checkpoint_kaydet(tahmin_dict_global, metrik_listesi, ALL_FEATURES)

# Yenide oku ve özetle
tahmin_df = pd.read_csv(TAHMIN_CSV)
metrik_df = pd.read_csv(METRIK_CSV)

print(f"  Tahmin: {tahmin_df.shape[0]} satir "
      f"({tahmin_df['country'].nunique()} ulke x 10 yil)")

# Turkiye ozet
if len(metrik_df) > 0:
    tr_ozet = metrik_df[metrik_df['country'] == TURKIYE_ADI]
    if len(tr_ozet) > 0:
        pd.set_option('display.width', 160)
        pd.set_option('display.float_format', '{:.4f}'.format)
        print(f"\nTurkiye Walk-Forward CV Metrikleri:")
        cols = ['feature', 'secilen_model', 'wf_cv_r2', 'wf_cv_mae',
                'wf_cv_rmse', 'wf_cv_mape', 'wf_cv_n']
        cols = [c for c in cols if c in tr_ozet.columns]
        print(tr_ozet[cols].to_string(index=False))
        print(f"\n  Ortalama R²: {tr_ozet['wf_cv_r2'].mean():.4f}")
        print(f"  Ortalama MAE: {tr_ozet['wf_cv_mae'].mean():.4f}")

print("\nKaydedilen dosyalar:")
print(f"  ciktilar/feature_tahminleri_2021_2030.csv")
print(f"  ciktilar/asama1_metrikler.csv")
print(f"  modeller/asama1_{{feature}}.pkl ({len(ALL_FEATURES)} dosya)")
print(f"  sonuclar/gorseller/asama1_* (Turkiye grafikleri)")
