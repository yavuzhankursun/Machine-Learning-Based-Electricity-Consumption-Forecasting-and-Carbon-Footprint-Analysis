"""
AŞAMA 1 — 7 MODEL KARSILASTIRMASI + FORECAST (Turkiye, korele parametreler)
===========================================================================
Elektrik tuketimini ve karbon ayak izini ETKILEYEN tum korele parametrelerin
2021-2030 gelecek tahminini uretir. HER feature icin 7 model walk-forward CV
ile degerlendirilir, EN IYI R²'li model ile gelecek tahmini yapilir.

  7 MODEL : RF · XGBoost · LightGBM · CatBoost · LSTM · ARIMA · SARIMA
  GIRDI   : feature'in KENDI gecmis zaman serisi (univariate self-prediction)
            Tree/LSTM  → X = [year, lag_1, lag_2]
            ARIMA/SARIMA → feature serisi (univariate)
  CV      : Walk-Forward Expanding Window (tum modeller AYNI test noktalari)
  FORECAST: En iyi model ile iteratif (tree/LSTM) veya cok-adim (ARIMA/SARIMA).
            Model kendi tahminini bir sonraki yilin lag'i olarak kullanir → leakage yok.

KAPSAM : 18 feature (Electricity_Loss_Pct cikarildi — Turkiye'de veri yok).
         nuclear_consumption / nuclear_share_elec sabit 0 (Akkuyu oncesi) — tutuldu.

GIRDI DOSYALARI:
  veri/islenmis/train.csv, val.csv, test.csv   (gecmis gercek 1990-2025)
  veri/islenmis/feature_listesi.csv            (feature isimleri)

CIKTI DOSYALARI:
  sonuclar/model_karsilastirma/asama1_model_karsilastirma.csv / .json  (R² tablosu)
  asama1_feature_tahmini/ciktilar/feature_tahminleri_2021_2030.csv     (Turkiye, en iyi model)
    → bu dosya AŞAMA 2'ye girdi olur.

CHECKPOINT: Her feature sonrasi atomic kayit. Yeniden baslatildiginda tamamlananlar atlanir.
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
LAG_SAYISI    = 2
TAHMIN_SON    = 2030
N_TRIAL_TREE  = 20      # tree modelleri icin Optuna trial
N_TRIAL_LSTM  = 0       # 0 = sabit LSTM mimarisi (18 feature, hiz icin)
LSTM_EPOCHS   = 80
FARK_AL       = True    # feature-tabanli modeller hedefin farkini ogrenir (trend ekstrapolasyon)

# Turkiye'de tahmin edilemeyen / bos feature (kullanici karari ile cikarildi)
CIKARILAN     = {'Electricity_Loss_Pct'}

# ─── Yollar ───────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
ISLENMIS_DIR = os.path.join(PIPELINE_DIR, 'veri', 'islenmis')
KARS_DIR     = os.path.join(PIPELINE_DIR, 'sonuclar', 'model_karsilastirma')
ASAMA1_CIKTI = os.path.join(PIPELINE_DIR, 'asama1_feature_tahmini', 'ciktilar')
os.makedirs(KARS_DIR, exist_ok=True)
os.makedirs(ASAMA1_CIKTI, exist_ok=True)

CSV_YOL      = os.path.join(KARS_DIR, 'asama1_model_karsilastirma.csv')
JSON_YOL     = os.path.join(KARS_DIR, 'asama1_model_karsilastirma.json')
FORECAST_CSV = os.path.join(ASAMA1_CIKTI, 'feature_tahminleri_2021_2030.csv')


# ─── Yardimci ──────────────────────────────────────────────────────────────────

def lag_matrisi(yillar, degerler, n_lag=2):
    df = pd.DataFrame({'year': yillar, 'value': degerler})
    for k in range(1, n_lag + 1):
        df[f'lag_{k}'] = df['value'].shift(k)
    df = df.dropna().reset_index(drop=True)
    X = df[['year'] + [f'lag_{k}' for k in range(1, n_lag + 1)]].values
    y = df['value'].values
    return X, y


def _nan(o):
    if isinstance(o, float) and (o != o):
        return None
    raise TypeError


def feature_forecast_uret(best, sonuc_best, seri, feat):
    """En iyi model ile feature'in gelecek tahminini uret.
    Donus: {yil: deger} 2021-2030 (gercek olan yillar gercek, gelecek tahmin)."""
    gercek = dict(zip(seri['year'].astype(int), seri[feat].astype(float)))
    son_gercek = max(gercek)
    fc_yillar = list(range(son_gercek + 1, TAHMIN_SON + 1))  # yalniz gelecek

    if not fc_yillar:
        gelecek = {}
    elif best in FEATURE_MODELLER:
        X, y = lag_matrisi(seri['year'].values, seri[feat].values, LAG_SAYISI)
        predict_fn = _final_predict_fn(best, sonuc_best.get('params'), X, y,
                                       lstm_epochs=max(LSTM_EPOCHS, 150),
                                       fark_al=FARK_AL)
        if predict_fn is None:  # LSTM yoksa → fallback
            gelecek = {y_: gercek[son_gercek] for y_ in fc_yillar}
        else:
            # Asama 1 univariate: X = [year, lag_1, lag_2]
            gelecek = iteratif_forecast(
                predict_fn, gercek, fc_yillar,
                x_kurucu=lambda yil, lags: [[yil, lags[0], lags[1]]],
                lag_sayisi=LAG_SAYISI, fark_al=FARK_AL)
    else:  # ARIMA / SARIMA — univariate cok-adim
        y_seri = seri[feat].astype(float).values
        fc_arr = forecast_arima_seri(y_seri, sonuc_best.get('order'),
                                     sonuc_best.get('seasonal_order'), len(fc_yillar))
        gelecek = {yil: float(v) for yil, v in zip(fc_yillar, fc_arr)}

    # 2021-2030 ciktisi: gercek varsa gercek, yoksa tahmin
    out = {}
    for yil in range(2021, TAHMIN_SON + 1):
        out[yil] = float(gercek[yil]) if yil in gercek else float(gelecek.get(yil, np.nan))
    return out


# ─── Checkpoint ────────────────────────────────────────────────────────────────

def checkpoint_yukle():
    sonuc_json, tahmin_dict = {}, {}
    if os.path.exists(JSON_YOL):
        try:
            sonuc_json = json.load(open(JSON_YOL))
        except Exception:
            sonuc_json = {}
    if os.path.exists(FORECAST_CSV):
        try:
            df = pd.read_csv(FORECAST_CSV)
            tr = df[df['country'] == ULKE]
            for _, row in tr.iterrows():
                tahmin_dict[int(row['year'])] = row.to_dict()
        except Exception:
            tahmin_dict = {}
    return sonuc_json, tahmin_dict


def kaydet(sonuc_json, tahmin_turkiye, iso):
    """R² tablosu (CSV+JSON) + feature_tahminleri (Turkiye satirlari guncellenir)."""
    # 1) JSON (R² tablosu)
    tmp = JSON_YOL + '.tmp'
    json.dump(sonuc_json, open(tmp, 'w'), ensure_ascii=False, indent=1, default=_nan)
    os.replace(tmp, JSON_YOL)

    # 2) CSV (uzun format: feature, model, metrikler)
    rows = []
    for feat, d in sonuc_json.items():
        for model in MODEL_SIRASI:
            md = d['modeller'].get(model, {})
            rows.append({'feature': feat, 'model': model,
                         'r2': md.get('r2'), 'mae': md.get('mae'),
                         'rmse': md.get('rmse'), 'mape': md.get('mape'),
                         'cv_n': md.get('cv_n', 0), 'en_iyi': model == d.get('en_iyi'),
                         'order': md.get('order', ''),
                         'seasonal_order': md.get('seasonal_order', '')})
    tmp = CSV_YOL + '.tmp'
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, CSV_YOL)

    # 3) feature_tahminleri_2021_2030.csv — Turkiye satirlarini guncelle, digerleri korunur
    if tahmin_turkiye:
        if os.path.exists(FORECAST_CSV):
            df = pd.read_csv(FORECAST_CSV)
            diger = df[df['country'] != ULKE].copy()
        else:
            diger = pd.DataFrame()
        tr_rows = []
        for yil in range(2021, TAHMIN_SON + 1):
            r = {'country': ULKE, 'iso_code': iso, 'year': yil}
            r.update(tahmin_turkiye.get(yil, {}))
            tr_rows.append(r)
        tr_df = pd.DataFrame(tr_rows)
        birlesik = pd.concat([diger, tr_df], ignore_index=True) if len(diger) else tr_df
        birlesik = birlesik.sort_values(['country', 'year']).reset_index(drop=True)
        tmp = FORECAST_CSV + '.tmp'
        birlesik.to_csv(tmp, index=False)
        os.replace(tmp, FORECAST_CSV)


# ═══════════════════════════════════════════════════════════════════════════════
# ANA
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    tek_feature = sys.argv[1] if len(sys.argv) > 1 else None  # tek feature test modu

    print("=" * 74)
    print("ASAMA 1 — 7 MODEL KARSILASTIRMASI + FORECAST (Turkiye)")
    print(f"  Modeller: {', '.join(MODEL_SIRASI)}")
    print(f"  TensorFlow(LSTM)={TF_MEVCUT}  statsmodels(ARIMA/SARIMA)={SM_MEVCUT}")
    print(f"  CV: Walk-Forward Expanding Window | Forecast: en iyi model, 2021-{TAHMIN_SON}")
    print("=" * 74)

    tr = pd.concat([
        pd.read_csv(os.path.join(ISLENMIS_DIR, 'train.csv')),
        pd.read_csv(os.path.join(ISLENMIS_DIR, 'val.csv')),
        pd.read_csv(os.path.join(ISLENMIS_DIR, 'test.csv')),
    ])
    tr = tr[tr['country'] == ULKE].sort_values('year').reset_index(drop=True)
    iso = tr['iso_code'].iloc[0] if 'iso_code' in tr.columns and len(tr) else 'TUR'

    feats_all = pd.read_csv(os.path.join(ISLENMIS_DIR, 'feature_listesi.csv'))[
        'tum_features'].dropna().tolist()
    feats = [f for f in feats_all if f not in CIKARILAN]
    if tek_feature:
        feats = [f for f in feats if f == tek_feature]
        print(f"  [TEK FEATURE TEST MODU] → {tek_feature}")

    print(f"  {ULKE}: {len(tr)} yil ({tr['year'].min()}-{tr['year'].max()})")
    print(f"  Tahmin edilecek feature: {len(feats)} (cikarilan: {sorted(CIKARILAN)})\n")

    sonuc_json, tahmin_turkiye = checkpoint_yukle()
    if sonuc_json:
        print(f"  [RESUME] {len(sonuc_json)} feature mevcut, atlanacak.\n")

    t0 = time.time()
    for i, feat in enumerate(feats, 1):
        if feat in sonuc_json and not tek_feature:
            print(f"[{i}/{len(feats)}] {feat} → [SKIP] mevcut")
            continue
        if feat not in tr.columns:
            print(f"[{i}/{len(feats)}] {feat} → [ATLA] kolon yok")
            continue

        seri = tr[['year', feat]].dropna()
        if len(seri) < 8:
            print(f"[{i}/{len(feats)}] {feat} → [ATLA] yetersiz veri ({len(seri)})")
            continue

        X, y = lag_matrisi(seri['year'].values, seri[feat].values, LAG_SAYISI)
        if len(y) < 8:
            print(f"[{i}/{len(feats)}] {feat} → [ATLA] lag sonrasi yetersiz ({len(y)})")
            continue

        min_train = max(min(10, len(y) - 4), 4)
        gecen = (time.time() - t0) / 60
        print(f"[{i}/{len(feats)}] {feat}  (n={len(y)}, min_train={min_train}, "
              f"gecen={gecen:.1f}dk)")

        s = yedi_modeli_calistir(X, y, min_train, n_trials_tree=N_TRIAL_TREE,
                                 n_trials_lstm=N_TRIAL_LSTM, lstm_epochs=LSTM_EPOCHS,
                                 etiket=feat, fark_al=FARK_AL)
        best = en_iyi_model(s)
        if best is None:
            print(f"    → tum modeller NaN, feature atlandi")
            continue
        print(f"    → EN IYI: {best} (R²={s[best]['r2']:.4f})")

        # Forecast (en iyi model)
        fc = feature_forecast_uret(best, s[best], seri, feat)
        g2026 = ", ".join(f"{yil}:{fc[yil]:.2f}" for yil in range(2026, TAHMIN_SON + 1))
        print(f"    forecast 2026-{TAHMIN_SON}: {g2026}")

        # Sonuca yaz (trues/preds JSON'dan cikar — dosya sismesin)
        temiz = {m: {k: v for k, v in d.items() if k not in ('trues', 'preds')}
                 for m, d in s.items()}
        sonuc_json[feat] = {'en_iyi': best, 'modeller': temiz}
        for yil in range(2021, TAHMIN_SON + 1):
            tahmin_turkiye.setdefault(yil, {})[feat] = round(fc[yil], 6)

        kaydet(sonuc_json, tahmin_turkiye, iso)
        print(f"    ✓ checkpoint atildi")

    print(f"\n{'='*74}")
    print(f"ASAMA 1 TAMAMLANDI ({(time.time()-t0)/60:.1f}dk)")
    print(f"{'='*74}")
    _ozet(sonuc_json)
    print(f"\n  R² tablosu : {CSV_YOL}")
    print(f"  Forecast   : {FORECAST_CSV}")


def _ozet(sonuc_json):
    print("\nOZET — feature basina en iyi model + R²:")
    rows = []
    for feat, d in sonuc_json.items():
        best = d.get('en_iyi')
        r2 = d['modeller'].get(best, {}).get('r2') if best else None
        rows.append((feat, best, r2))
    rows.sort(key=lambda x: (x[2] if x[2] is not None else -99), reverse=True)
    for feat, best, r2 in rows:
        r2s = f"{r2:.4f}" if r2 is not None else "nan"
        print(f"  {feat:<35} {str(best):<10} R²={r2s}")


if __name__ == '__main__':
    main()
