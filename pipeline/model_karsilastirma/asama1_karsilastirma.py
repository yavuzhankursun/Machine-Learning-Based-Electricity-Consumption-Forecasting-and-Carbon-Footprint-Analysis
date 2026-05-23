"""
ASAMA 1 — 7 MODEL KARSILASTIRMASI (Turkiye, 19 feature)
========================================================
Turkiye'nin 19 girdi degiskeninin HER BIRI icin 7 modeli walk-forward CV ile
degerlendirir ve R²/MAE/RMSE/MAPE tablosunu CSV + JSON olarak yazar.

  Girdi (RF/XGB/LGBM/CatBoost/LSTM): [year, lag_1, lag_2]
  Girdi (ARIMA/SARIMA):              feature'in kendi zaman serisi (univariate)

  → Tum modeller AYNI test noktalarinda degerlendirilir (karsilastirilabilir R²).

CIKTI:
  pipeline/sonuclar/model_karsilastirma/asama1_model_karsilastirma.csv
  pipeline/sonuclar/model_karsilastirma/asama1_model_karsilastirma.json

Mevcut asama1_egit.py / forecast ciktilarina DOKUNMAZ.
"""

import os, sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from karsilastirma_ortak import (yedi_modeli_calistir, en_iyi_model,
                                 MODEL_SIRASI, TF_MEVCUT, SM_MEVCUT)

# ── Sabitler ────────────────────────────────────────────────────────────────────
ULKE        = 'Turkey'
LAG_SAYISI  = 2
N_TRIAL_TREE = 15   # tree modelleri icin Optuna trial (19 feature x 4 model)
N_TRIAL_LSTM = 0    # 0 = sabit LSTM mimarisi (hiz icin; 19 feature cok)
LSTM_EPOCHS  = 100

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
ISLENMIS_DIR = os.path.join(PIPELINE_DIR, 'veri', 'islenmis')
CIKTI_DIR    = os.path.join(PIPELINE_DIR, 'sonuclar', 'model_karsilastirma')
os.makedirs(CIKTI_DIR, exist_ok=True)

CSV_YOL  = os.path.join(CIKTI_DIR, 'asama1_model_karsilastirma.csv')
JSON_YOL = os.path.join(CIKTI_DIR, 'asama1_model_karsilastirma.json')


def lag_matrisi(yillar, degerler, n_lag=2):
    df = pd.DataFrame({'year': yillar, 'value': degerler})
    for k in range(1, n_lag + 1):
        df[f'lag_{k}'] = df['value'].shift(k)
    df = df.dropna().reset_index(drop=True)
    X = df[['year'] + [f'lag_{k}' for k in range(1, n_lag + 1)]].values
    y = df['value'].values
    return X, y


def main():
    print("=" * 72)
    print("ASAMA 1 — 7 MODEL KARSILASTIRMASI (Turkiye)")
    print(f"  Modeller: {', '.join(MODEL_SIRASI)}")
    print(f"  TensorFlow(LSTM)={TF_MEVCUT}  statsmodels(ARIMA/SARIMA)={SM_MEVCUT}")
    print("  Degerlendirme: Walk-Forward Expanding Window CV (tum modeller ayni noktalar)")
    print("=" * 72)

    # Veri
    tr = pd.concat([
        pd.read_csv(os.path.join(ISLENMIS_DIR, 'train.csv')),
        pd.read_csv(os.path.join(ISLENMIS_DIR, 'val.csv')),
        pd.read_csv(os.path.join(ISLENMIS_DIR, 'test.csv')),
    ])
    tr = tr[tr['country'] == ULKE].sort_values('year').reset_index(drop=True)
    feats = pd.read_csv(os.path.join(ISLENMIS_DIR, 'feature_listesi.csv'))['tum_features'].dropna().tolist()
    print(f"  {ULKE}: {len(tr)} yil ({tr['year'].min()}-{tr['year'].max()}) | {len(feats)} feature\n")

    # Resume: tamamlanmis feature'lar
    sonuc_json = {}
    if os.path.exists(JSON_YOL):
        try:
            sonuc_json = json.load(open(JSON_YOL))
            print(f"  [RESUME] {len(sonuc_json)} feature mevcut, atlanacak: {sorted(sonuc_json)}\n")
        except Exception:
            sonuc_json = {}

    t0 = time.time()
    for i, feat in enumerate(feats):
        if feat in sonuc_json:
            print(f"[{i+1}/{len(feats)}] {feat} → [SKIP] mevcut")
            continue
        if feat not in tr.columns:
            print(f"[{i+1}/{len(feats)}] {feat} → [ATLA] kolon yok")
            continue

        seri = tr[['year', feat]].dropna()
        if len(seri) < 8:
            print(f"[{i+1}/{len(feats)}] {feat} → [ATLA] yetersiz veri ({len(seri)})")
            continue

        X, y = lag_matrisi(seri['year'].values, seri[feat].values, LAG_SAYISI)
        if len(y) < 8:
            print(f"[{i+1}/{len(feats)}] {feat} → [ATLA] lag sonrasi yetersiz ({len(y)})")
            continue

        # min_train: cogu nokta out-of-sample olsun
        min_train = max(min(10, len(y) - 4), 4)
        gecen = time.time() - t0
        print(f"[{i+1}/{len(feats)}] {feat}  (n={len(y)}, min_train={min_train}, "
              f"gecen={gecen/60:.1f}dk)")

        s = yedi_modeli_calistir(X, y, min_train,
                                 n_trials_tree=N_TRIAL_TREE,
                                 n_trials_lstm=N_TRIAL_LSTM,
                                 lstm_epochs=LSTM_EPOCHS, etiket=feat)
        best = en_iyi_model(s)
        # trues/preds'i JSON'dan cikar (CSV'de gereksiz, dosya sismesin)
        temiz = {m: {k: v for k, v in d.items() if k not in ('trues', 'preds')}
                 for m, d in s.items()}
        sonuc_json[feat] = {'en_iyi': best, 'modeller': temiz}
        print(f"    → EN IYI: {best} (R²={s[best]['r2']:.4f})" if best else "    → tum modeller NaN")

        # Checkpoint (her feature sonrasi)
        _kaydet(sonuc_json)

    print(f"\nTAMAMLANDI ({(time.time()-t0)/60:.1f}dk)")
    _ozet(sonuc_json)


def _kaydet(sonuc_json):
    # JSON (atomic)
    tmp = JSON_YOL + '.tmp'
    json.dump(sonuc_json, open(tmp, 'w'), ensure_ascii=False, indent=1, default=_nan)
    os.replace(tmp, JSON_YOL)
    # CSV (uzun format: feature, model, ...)
    rows = []
    for feat, d in sonuc_json.items():
        for model in MODEL_SIRASI:
            md = d['modeller'].get(model, {})
            rows.append({
                'feature': feat, 'model': model,
                'r2': md.get('r2'), 'mae': md.get('mae'),
                'rmse': md.get('rmse'), 'mape': md.get('mape'),
                'cv_n': md.get('cv_n', 0),
                'en_iyi': model == d.get('en_iyi'),
                'order': md.get('order', ''),
                'seasonal_order': md.get('seasonal_order', ''),
            })
    df = pd.DataFrame(rows)
    tmp = CSV_YOL + '.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, CSV_YOL)


def _nan(o):
    if isinstance(o, float) and (o != o):
        return None
    raise TypeError


def _ozet(sonuc_json):
    print("\n" + "=" * 72)
    print("OZET — feature basina en iyi model ve R²")
    print("=" * 72)
    rows = []
    for feat, d in sonuc_json.items():
        best = d.get('en_iyi')
        r2 = d['modeller'].get(best, {}).get('r2') if best else None
        rows.append((feat, best, r2))
    rows.sort(key=lambda x: (x[2] if x[2] is not None else -99), reverse=True)
    for feat, best, r2 in rows:
        print(f"  {feat:<35} {str(best):<10} R²={r2 if r2 is not None else float('nan'):.4f}")
    print(f"\n  CSV : {CSV_YOL}")
    print(f"  JSON: {JSON_YOL}")


if __name__ == '__main__':
    main()
