"""
7-MODEL PIPELINE — GORSEL URETIMI (Turkiye)
============================================
3 asamanin CV + forecast sonuclarindan tez icin gorseller uretir.

URETILEN GORSELLER (sonuclar/gorseller/7model_*.png):
  Her asama icin:
    * 7 model R² karsilastirma bar (en iyi vurgulu)
    * 7 model MAPE karsilastirma bar
    * Forecast cizgi (gecmis gercek + 7 model 2026-2030)
  Asama 1 ek:
    * 18 feature × en iyi model R² yatay bar
  Asama 2/3 ek (en iyi model yeniden egitilir):
    * En iyi model CV scatter (gercek vs tahmin)
    * SHAP bar + beeswarm (en iyi tree model — feature etkisi)
    * Built-in feature importance

GIRDI: model_karsilastirma/*.json, forecast CSV'leri, ham/islenmis veri.
Mevcut egit.py gorsellerine DOKUNMAZ — dosya adlari '7model_' onekli.
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from karsilastirma_ortak import (MODEL_SIRASI, MODEL_CLS, FEATURE_MODELLER, TF_MEVCUT,
                                 walk_forward_feature, walk_forward_lstm,
                                 tune_rf, tune_xgb, tune_lgbm, tune_catboost)
from sklearn.preprocessing import MinMaxScaler

# ─── Yollar ───────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
ISLENMIS_DIR = os.path.join(PIPELINE_DIR, 'veri', 'islenmis')
HAM_DIR      = os.path.join(PIPELINE_DIR, 'veri', 'ham')
KARS_DIR     = os.path.join(PIPELINE_DIR, 'sonuclar', 'model_karsilastirma')
ASAMA1_DIR   = os.path.join(PIPELINE_DIR, 'asama1_feature_tahmini', 'ciktilar')
ASAMA2_DIR   = os.path.join(PIPELINE_DIR, 'asama2_elektrik_tahmini', 'ciktilar')
ASAMA3_DIR   = os.path.join(PIPELINE_DIR, 'asama3_karbon_tahmini', 'ciktilar')
GORSEL_DIR   = os.path.join(PIPELINE_DIR, 'sonuclar', 'gorseller')
os.makedirs(GORSEL_DIR, exist_ok=True)

ULKE = 'Turkey'
RENK7 = {'RF': '#1f77b4', 'XGBoost': '#ff7f0e', 'LightGBM': '#2ca02c',
         'CatBoost': '#d62728', 'LSTM': '#9467bd', 'ARIMA': '#8c564b', 'SARIMA': '#e377c2'}


def _yol(ad):
    return os.path.join(GORSEL_DIR, f'7model_{ad}.png')


# ═══════════════════════════════════════════════════════════════════════════════
# ORTAK GRAFIKLER (CSV/JSON'dan — yeniden egitim yok)
# ═══════════════════════════════════════════════════════════════════════════════

def ciz_r2_bar(modeller, en_iyi, baslik, ad, metr='r2', metr_ad='R²'):
    """7 model metrik karsilastirma yatay bar."""
    isim, deg = [], []
    for m in MODEL_SIRASI:
        v = modeller.get(m, {}).get(metr)
        if v is not None and v == v:
            isim.append(m); deg.append(v)
    if not isim:
        return
    sira = np.argsort(deg)
    isim = [isim[i] for i in sira]; deg = [deg[i] for i in sira]
    renkler = ['gold' if m == en_iyi else RENK7.get(m, 'steelblue') for m in isim]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(isim, deg, color=renkler, edgecolor='k')
    for b, v, m in zip(bars, deg, isim):
        etk = f'{v:.4f}' + ('  ★ EN IYI' if m == en_iyi else '')
        ax.text(v + (max(deg)-min(deg, default=0))*0.01, b.get_y()+b.get_height()/2,
                etk, va='center', fontsize=9, fontweight='bold' if m == en_iyi else 'normal')
    ax.set_xlabel(metr_ad); ax.set_title(baslik)
    plt.tight_layout(); plt.savefig(_yol(ad), dpi=150); plt.close()


def ciz_forecast_cizgi(gecmis_yil, gecmis_deg, fc_df, yil_col, model_cols,
                       en_iyi_col, birim, baslik, ad):
    """Gecmis gercek + 7 model forecast cizgi grafigi."""
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(gecmis_yil, gecmis_deg, 'k-o', ms=4, lw=2, label='Gercek (gecmis)', zorder=10)
    fc_yil = fc_df[yil_col].values
    for col in model_cols:
        if col not in fc_df.columns:
            continue
        m = col.split('_')[0]
        ax.plot(fc_yil, fc_df[col].values, '--s', ms=4, alpha=0.55,
                color=RENK7.get(m, 'gray'), label=f'{m} (forecast)')
    if en_iyi_col in fc_df.columns:
        ax.plot(fc_yil, fc_df[en_iyi_col].values, '-^', ms=7, lw=2.5, color='red',
                label='EN IYI MODEL', zorder=9)
    ax.axvspan(fc_yil.min()-0.4, fc_yil.max()+0.4, alpha=0.07, color='red')
    ax.set_xlabel('Yil'); ax.set_ylabel(birim); ax.set_title(baslik)
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(_yol(ad), dpi=150); plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# CV SCATTER + SHAP (en iyi model yeniden egitilir — Asama 2/3)
# ═══════════════════════════════════════════════════════════════════════════════

def _en_iyi_tree(modeller):
    """R²'si en yuksek TREE modeli (SHAP TreeExplainer icin)."""
    aday = {m: modeller[m]['r2'] for m in MODEL_CLS
            if m in modeller and modeller[m].get('r2') == modeller[m].get('r2')}
    return max(aday, key=aday.get) if aday else None


def _params_bul(modeller, model, X, y):
    """JSON'da params varsa onu, yoksa yeniden tune et."""
    p = modeller.get(model, {}).get('params')
    if p:
        return p
    tuner = {'RF': tune_rf, 'XGBoost': tune_xgb, 'LightGBM': tune_lgbm,
             'CatBoost': tune_catboost}.get(model)
    return tuner(X, y, 20) if tuner else {}


def ciz_cv_scatter(X, y, min_train, best, modeller, birim, baslik, ad, fark_al=True):
    """En iyi modeli walk-forward ile yeniden calistir → gercek vs tahmin scatter."""
    try:
        if best in MODEL_CLS:
            params = _params_bul(modeller, best, X, y)
            trues, preds = walk_forward_feature(lambda: MODEL_CLS[best](**params),
                                                X, y, min_train, fark_al=fark_al)
        elif best == 'LSTM' and TF_MEVCUT:
            p = modeller.get('LSTM', {}).get('params', {}) or {}
            trues, preds = walk_forward_lstm(
                X, y, min_train, units1=p.get('units1', 64), units2=p.get('units2', 32),
                dropout=p.get('dropout', 0.2), lr=p.get('lr', 1e-3),
                epochs=150, batch=p.get('batch', 4), fark_al=fark_al)
        else:
            return  # ARIMA/SARIMA scatter ayri ele alinmaz
        from sklearn.metrics import r2_score
        r2 = r2_score(trues, preds)
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        ax.scatter(trues, preds, color='steelblue', edgecolors='k', s=70, zorder=5)
        mn, mx = min(trues.min(), preds.min()), max(trues.max(), preds.max())
        mg = (mx-mn)*0.05
        ax.plot([mn-mg, mx+mg], [mn-mg, mx+mg], 'r--', lw=1.5, label='Ideal (y=x)')
        ax.set_xlabel(f'Gercek ({birim})'); ax.set_ylabel(f'Tahmin ({birim})')
        ax.set_title(f'{baslik}\n{best} — Walk-Forward CV | R²={r2:.4f}')
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(_yol(ad), dpi=150); plt.close()
    except Exception as e:
        print(f"  [CV scatter atlandi ({ad}): {e}]")


def ciz_shap(X, y, input_cols, modeller, baslik, ad, fark_al=True):
    """En iyi TREE modelini final egit → SHAP bar + beeswarm + built-in importance."""
    try:
        import shap
        tree = _en_iyi_tree(modeller)
        if tree is None:
            print(f"  [SHAP atlandi ({ad}): tree model yok]")
            return
        params = _params_bul(modeller, tree, X, y)
        # differencing'li final egitim (forecast ile tutarli)
        if fark_al and len(y) >= 3:
            Xf, yf = X[1:], y[1:] - y[:-1]
        else:
            Xf, yf = X, y
        sc = MinMaxScaler()
        Xs = sc.fit_transform(Xf)
        m = MODEL_CLS[tree](**params)
        m.fit(Xs, yf)
        explainer = shap.TreeExplainer(m)
        sv = explainer.shap_values(Xs)
        expl = shap.Explanation(values=np.array(sv), data=Xs, feature_names=input_cols)

        # Bar
        plt.figure(figsize=(9, max(4, len(input_cols)*0.4)))
        shap.plots.bar(expl, show=False, max_display=len(input_cols))
        plt.title(f'{baslik}\nSHAP (mean |Δ|) — {tree} | hedefin farki uzerinde')
        plt.tight_layout(); plt.savefig(_yol(ad+'_shap_bar'), dpi=150, bbox_inches='tight'); plt.close('all')

        # Beeswarm
        plt.figure(figsize=(10, max(5, len(input_cols)*0.4)))
        shap.plots.beeswarm(expl, show=False, max_display=len(input_cols))
        plt.title(f'{baslik}\nSHAP Beeswarm — {tree}')
        plt.tight_layout(); plt.savefig(_yol(ad+'_shap_beeswarm'), dpi=150, bbox_inches='tight'); plt.close('all')

        # Built-in importance
        if hasattr(m, 'feature_importances_'):
            imp = m.feature_importances_
            idx = np.argsort(imp)
            plt.figure(figsize=(9, max(4, len(input_cols)*0.4)))
            plt.barh([input_cols[i] for i in idx], imp[idx], color='coral', edgecolor='k')
            plt.xlabel('Feature Importance (built-in)')
            plt.title(f'{baslik}\n{tree} — Built-in Feature Importance')
            plt.tight_layout(); plt.savefig(_yol(ad+'_importance'), dpi=150, bbox_inches='tight'); plt.close('all')
        print(f"  ✓ SHAP + importance ({tree}) → {ad}")
    except Exception as e:
        print(f"  [SHAP atlandi ({ad}): {e}]")


# ═══════════════════════════════════════════════════════════════════════════════
# ASAMA 1
# ═══════════════════════════════════════════════════════════════════════════════

def asama1_gorseller():
    print("\n[ASAMA 1] gorseller...")
    jp = os.path.join(KARS_DIR, 'asama1_model_karsilastirma.json')
    if not os.path.exists(jp):
        print("  asama1 JSON yok, atlandi."); return
    j = json.load(open(jp))

    # 1) Feature × en iyi model R² yatay bar
    rows = []
    for feat, d in j.items():
        best = d.get('en_iyi')
        r2 = d['modeller'].get(best, {}).get('r2') if best else None
        if r2 is not None and r2 == r2:
            rows.append((feat, best, r2))
    rows.sort(key=lambda x: x[2])
    fig, ax = plt.subplots(figsize=(10, max(5, len(rows)*0.42)))
    renkler = [RENK7.get(b, 'gray') for _, b, _ in rows]
    bars = ax.barh([f for f, _, _ in rows], [r for _, _, r in rows], color=renkler, edgecolor='k')
    for b, (feat, best, r2) in zip(bars, rows):
        ax.text(max(r2, 0)+0.01, b.get_y()+b.get_height()/2, f'{best} ({r2:.3f})',
                va='center', fontsize=8)
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('R² (Walk-Forward CV)')
    ax.set_title('Asama 1 — 18 Korele Parametre: Feature Basina En Iyi Model ve R²')
    # Model renk lejandi
    from matplotlib.patches import Patch
    kullanilan_m = sorted({b for _, b, _ in rows})
    ax.legend(handles=[Patch(color=RENK7.get(m, 'gray'), label=m) for m in kullanilan_m],
              fontsize=8, loc='lower right', title='En iyi model')
    plt.tight_layout(); plt.savefig(_yol('asama1_feature_en_iyi_r2'), dpi=150); plt.close()

    # 2) Model genel performansi (feature'lar arasi ortalama R², R²>=0 olanlar)
    model_r2 = {m: [] for m in MODEL_SIRASI}
    for feat, d in j.items():
        for m in MODEL_SIRASI:
            v = d['modeller'].get(m, {}).get('r2')
            if v is not None and v == v and v > -1:  # asiri negatifleri katma
                model_r2[m].append(v)
    ort = {m: np.mean(v) for m, v in model_r2.items() if v}
    if ort:
        ms = sorted(ort, key=ort.get)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        bars = ax.barh(ms, [ort[m] for m in ms],
                       color=[RENK7.get(m, 'gray') for m in ms], edgecolor='k')
        for b, m in zip(bars, ms):
            ax.text(ort[m]+0.005, b.get_y()+b.get_height()/2,
                    f'{ort[m]:.3f} (n={len(model_r2[m])})', va='center', fontsize=8)
        ax.set_xlabel('Ortalama R² (18 feature uzerinde)')
        ax.set_title('Asama 1 — Modellerin Genel Performansi\n(feature tahmininde ortalama basari)')
        plt.tight_layout(); plt.savefig(_yol('asama1_model_genel'), dpi=150); plt.close()

    # 3) Temsili feature forecast cizgileri
    ft = os.path.join(ASAMA1_DIR, 'feature_tahminleri_2021_2030.csv')
    tr_hist = pd.concat([pd.read_csv(os.path.join(ISLENMIS_DIR, f'{s}.csv'))
                         for s in ['train', 'val', 'test']])
    tr_hist = tr_hist[tr_hist['country'] == ULKE].sort_values('year')
    if os.path.exists(ft):
        a1 = pd.read_csv(ft); a1 = a1[a1['country'] == ULKE].sort_values('year')
        onemli = ['Population', 'GDP_Per_Capita', 'fossil_fuel_consumption',
                  'primary_energy_consumption', 'renewables_share_elec', 'energy_per_capita']
        onemli = [f for f in onemli if f in a1.columns and f in tr_hist.columns]
        n = len(onemli)
        fig, axes = plt.subplots((n+1)//2, 2, figsize=(14, 3.2*((n+1)//2)))
        axes = np.array(axes).ravel()
        for i, f in enumerate(onemli):
            ax = axes[i]
            h = tr_hist[['year', f]].dropna()
            ax.plot(h['year'], h[f], 'k-o', ms=3, label='Gercek')
            fc = a1[a1['year'] >= 2026]
            ax.plot(fc['year'], fc[f], 'r--^', ms=5, label='Forecast')
            best = j.get(f, {}).get('en_iyi', '?')
            ax.set_title(f'{f}  (en iyi: {best})', fontsize=10)
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
        for k in range(n, len(axes)):
            axes[k].axis('off')
        fig.suptitle('Asama 1 — Temsili Korele Parametre Forecast (2026-2030)', fontsize=13)
        plt.tight_layout(); plt.savefig(_yol('asama1_forecast_ornek'), dpi=150); plt.close()
    print("  ✓ Asama 1 gorselleri")


# ═══════════════════════════════════════════════════════════════════════════════
# ASAMA 2 / 3 ORTAK
# ═══════════════════════════════════════════════════════════════════════════════

def asama23_gorseller(no, hedef_ad, birim, json_ad, fc_csv, fc_birim_suffix,
                      gecmis_yukle):
    print(f"\n[ASAMA {no}] gorseller...")
    jp = os.path.join(KARS_DIR, json_ad)
    if not os.path.exists(jp):
        print(f"  asama{no} JSON yok, atlandi."); return
    j = json.load(open(jp))
    modeller = j['modeller']; en_iyi = j['en_iyi']

    # 1) R² + 2) MAPE bar
    ciz_r2_bar(modeller, en_iyi, f'Asama {no} — {hedef_ad}: 7 Model R² Karsilastirma',
               f'asama{no}_r2', 'r2', 'R² (Walk-Forward CV)')
    ciz_r2_bar(modeller, en_iyi, f'Asama {no} — {hedef_ad}: 7 Model MAPE Karsilastirma',
               f'asama{no}_mape', 'mape', 'MAPE (%)')

    # 3) Forecast cizgi
    if os.path.exists(fc_csv):
        fc_df = pd.read_csv(fc_csv)
        gh_yil, gh_deg = gecmis_yukle()
        model_cols = [f'{m}_{fc_birim_suffix}' for m in MODEL_SIRASI]
        en_iyi_col = f'EN_IYI_{fc_birim_suffix}'
        ciz_forecast_cizgi(gh_yil, gh_deg, fc_df, 'year', model_cols, en_iyi_col,
                           birim, f'Asama {no} — {hedef_ad}: Gercek + 7 Model Forecast (2026-2030)',
                           f'asama{no}_forecast')

    # 4) CV scatter + 5) SHAP (en iyi model / en iyi tree yeniden egitilir)
    X, y, input_cols, min_train = _veri_hazirla(no)
    if X is not None:
        ciz_cv_scatter(X, y, min_train, en_iyi, modeller, birim,
                       f'Asama {no} — {hedef_ad}', f'asama{no}_cv_scatter')
        ciz_shap(X, y, input_cols, modeller, f'Asama {no} — {hedef_ad}', f'asama{no}')
    print(f"  ✓ Asama {no} gorselleri")


def _veri_hazirla(no):
    """Asama 2/3 icin X, y, input_cols, min_train yeniden kur (CV scatter + SHAP icin)."""
    try:
        tr = pd.concat([pd.read_csv(os.path.join(ISLENMIS_DIR, f'{s}.csv'))
                        for s in ['train', 'val', 'test']])
        tr = tr[tr['country'] == ULKE].sort_values('year').reset_index(drop=True)
        if no == 2:
            from asama2_7model_forecast import FEATURE_COLS, HEDEF, LAG_SAYISI, MIN_EGITIM_YIL
            df = tr
        else:
            from asama3_7model_forecast import FEATURE_COLS, HEDEF, LAG_SAYISI, MIN_EGITIM_YIL
            owid = pd.read_csv(os.path.join(HAM_DIR, 'owid_co2_emissions.csv'), comment='#')
            owid = owid.rename(columns={'Country': 'country', 'Year': 'year'})
            co2 = owid[owid['country'] == ULKE][['year', 'co2']].dropna()
            df = co2.merge(tr[['year'] + [c for c in FEATURE_COLS if c in tr.columns]],
                           on='year', how='inner').sort_values('year').reset_index(drop=True)
        kullanilan = [f for f in FEATURE_COLS if f in df.columns and df[f].notna().sum() >= 5]
        df[kullanilan] = df[kullanilan].ffill().bfill()
        for k in range(1, LAG_SAYISI+1):
            df[f'{HEDEF}_lag{k}'] = df[HEDEF].shift(k)
        lag_cols = [f'{HEDEF}_lag{k}' for k in range(1, LAG_SAYISI+1)]
        input_cols = lag_cols + kullanilan
        df = df.dropna(subset=input_cols+[HEDEF]).reset_index(drop=True)
        return df[input_cols].values, df[HEDEF].values, input_cols, MIN_EGITIM_YIL
    except Exception as e:
        print(f"  [veri hazirlama hatasi (asama{no}): {e}]")
        return None, None, None, None


def _gecmis_elektrik():
    tr = pd.concat([pd.read_csv(os.path.join(ISLENMIS_DIR, f'{s}.csv'))
                    for s in ['train', 'val', 'test']])
    tr = tr[tr['country'] == ULKE].sort_values('year')
    h = tr[['year', 'electricity_demand']].dropna()
    return h['year'].values, h['electricity_demand'].values


def _gecmis_co2():
    owid = pd.read_csv(os.path.join(HAM_DIR, 'owid_co2_emissions.csv'), comment='#')
    owid = owid.rename(columns={'Country': 'country', 'Year': 'year'})
    h = owid[owid['country'] == ULKE][['year', 'co2']].dropna().sort_values('year')
    return h['year'].values, h['co2'].values


if __name__ == '__main__':
    asama1_gorseller()
    asama23_gorseller(2, 'Elektrik Tuketimi', 'TWh', 'asama2_model_karsilastirma.json',
                      os.path.join(ASAMA2_DIR, 'elektrik_tahmin_2026_2030.csv'), 'TWh',
                      _gecmis_elektrik)
    asama23_gorseller(3, 'Karbon Ayak Izi', 'Mt CO2', 'asama3_model_karsilastirma.json',
                      os.path.join(ASAMA3_DIR, 'karbon_tahmin_2026_2030.csv'), 'MtCO2',
                      _gecmis_co2)
    print(f"\nTUM GORSELLER: {GORSEL_DIR}/7model_*.png")
    png = [f for f in os.listdir(GORSEL_DIR) if f.startswith('7model_')]
    print(f"Uretilen: {len(png)} dosya")
