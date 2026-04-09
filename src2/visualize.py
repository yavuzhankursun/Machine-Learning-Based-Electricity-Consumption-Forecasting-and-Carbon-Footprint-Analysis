"""
Per-Country Gorsellestirme

Her ulke icin 3 ayri PNG:
  1. {CODE}_forecast.png   - Tahmin grafigi (train/val/test/gelecek)
  2. {CODE}_analysis.png   - Hata analizi (confusion matrix karsiligi)
  3. {CODE}_crossval.png   - Walk-forward cross-validation

Cikti dizini: output2/figures/{forecast,analysis,crossval}/
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import r2_score

OUTPUT_BASE = os.path.join(os.path.dirname(__file__), '..', 'output2', 'figures')

COLORS = {
    'train'      : '#2196F3',   # mavi
    'val_actual' : '#FF9800',   # turuncu
    'val_pred'   : '#FF5722',   # kirmizi-turuncu
    'test_actual': '#4CAF50',   # yesil
    'test_pred'  : '#F44336',   # kirmizi
    'forecast'   : '#9C27B0',   # mor
}

REGION_ALPHA = 0.07


def _save(fig, subdir, filename):
    path = os.path.join(OUTPUT_BASE, subdir, filename)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1: Tahmin grafigi
# ─────────────────────────────────────────────────────────────────────────────

def plot_forecast(country_hist: pd.DataFrame, iso: str, country_name: str,
                  val_r2: float, test_r2: float):
    """
    Tek ulke icin zaman serisi tahmini.
    Gosterilen:
      - Mavi   : egitim donemi gercek degerler
      - Turuncu: val gercek / tahmin (2016-2018)
      - Yesil  : test gercek (2019-2022)
      - Kirmizi: test tahmini (2019-2022)
      - Mor    : gelecek tahmin (2023-2028)
    """
    fig, ax = plt.subplots(figsize=(13, 5))

    # Arka plan bolgeleri
    ax.axvspan(1993, 2015.5, alpha=REGION_ALPHA, color=COLORS['train'],   label='_nolegend_')
    ax.axvspan(2015.5, 2018.5, alpha=REGION_ALPHA, color=COLORS['val_actual'], label='_nolegend_')
    ax.axvspan(2018.5, 2022.5, alpha=REGION_ALPHA, color=COLORS['test_actual'], label='_nolegend_')
    ax.axvspan(2022.5, 2028.5, alpha=REGION_ALPHA, color=COLORS['forecast'],    label='_nolegend_')

    def get(split):
        s = country_hist[country_hist['split'] == split].sort_values('year')
        return s['year'].values, s['value'].values

    # Train
    tr_x, tr_y = get('train')
    ax.plot(tr_x, tr_y, 'o-', color=COLORS['train'], lw=2, ms=4, label='Train (gercek)')

    # Val actual
    va_x, va_y = get('val_actual')
    if len(va_x):
        # Train'in son noktasindan val'a baglanti
        ax.plot([tr_x[-1], va_x[0]], [tr_y[-1], va_y[0]],
                '-', color=COLORS['train'], lw=1.5, alpha=0.4)
        ax.plot(va_x, va_y, 's-', color=COLORS['val_actual'], lw=2, ms=6,
                label='Val (gercek)')

    # Val pred
    vp_x, vp_y = get('val_pred')
    if len(vp_x):
        ax.plot(vp_x, vp_y, 's--', color=COLORS['val_pred'], lw=2, ms=6,
                label='Val (tahmin)')

    # Test actual
    ta_x, ta_y = get('test_actual')
    if len(ta_x):
        last_va = va_y[-1] if len(va_y) else tr_y[-1]
        last_va_x = va_x[-1] if len(va_x) else tr_x[-1]
        ax.plot([last_va_x, ta_x[0]], [last_va, ta_y[0]],
                '-', color=COLORS['val_actual'], lw=1.5, alpha=0.4)
        ax.plot(ta_x, ta_y, '^-', color=COLORS['test_actual'], lw=2, ms=6,
                label='Test (gercek)')

    # Test pred
    tp_x, tp_y = get('test_pred')
    if len(tp_x):
        ax.plot(tp_x, tp_y, '^--', color=COLORS['test_pred'], lw=2, ms=6,
                label='Test (tahmin)')

    # Forecast
    fc_x, fc_y = get('forecast')
    if len(fc_x):
        last_x = ta_x[-1] if len(ta_x) else tr_x[-1]
        last_y = ta_y[-1] if len(ta_y) else tr_y[-1]
        ax.plot([last_x, fc_x[0]], [last_y, fc_y[0]],
                '--', color=COLORS['forecast'], lw=1.5, alpha=0.5)
        ax.plot(fc_x, fc_y, 'D--', color=COLORS['forecast'], lw=2, ms=7,
                label='Gelecek tahmin (2023-2028)')
        for x, y in zip(fc_x, fc_y):
            ax.annotate(f'{y:.0f}', (x, y), textcoords='offset points',
                        xytext=(0, 10), ha='center', fontsize=8,
                        color=COLORS['forecast'], fontweight='bold')

    # Bolgeler etiketi
    for yr, lbl in [(2004, 'TRAIN\n1993-2015'), (2017, 'VAL\n2016-18'),
                    (2020.5, 'TEST\n2019-22'), (2025.5, 'TAHMIN\n2023-28')]:
        ax.text(yr, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else 0,
                lbl, ha='center', va='bottom', fontsize=7.5, alpha=0.6)

    ax.set_title(f'{country_name} ({iso})\nVal R²={val_r2:.4f}  |  Test R²={test_r2:.4f}',
                 fontsize=12)
    ax.set_xlabel('Yil')
    ax.set_ylabel('Elektrik Tuketimi (kWh/kisi)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    _save(fig, 'forecast', f'{iso}_forecast.png')


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2: Hata analizi (regresyon icin confusion matrix karsiligi)
# ─────────────────────────────────────────────────────────────────────────────

def plot_analysis(country_hist: pd.DataFrame, iso: str, country_name: str):
    """
    4 alt grafik:
      TL: Actual vs Predicted scatter (val+test)
      TR: Zaman serisi residual (hata)
      BL: Hata histogrami
      BR: Yuzde hata (MAPE) yila gore
    """
    # Val ve test icin gercek/tahmin cifilerini birlestirir
    def pairs(actual_split, pred_split):
        a = country_hist[country_hist['split'] == actual_split].sort_values('year')
        p = country_hist[country_hist['split'] == pred_split].sort_values('year')
        merged = a.merge(p, on=['iso_code', 'country', 'year'], suffixes=('_a', '_p'))
        return merged['value_a'].values, merged['value_p'].values, merged['year'].values

    va_act, va_pred, va_yr = pairs('val_actual',  'val_pred')
    te_act, te_pred, te_yr = pairs('test_actual', 'test_pred')

    all_act  = np.concatenate([va_act, te_act])
    all_pred = np.concatenate([va_pred, te_pred])
    all_yr   = np.concatenate([va_yr, te_yr])
    resid    = all_act - all_pred
    pct_err  = np.where(all_act > 10, (all_pred - all_act) / all_act * 100, np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f'{country_name} ({iso}) — Hata Analizi', fontsize=13)

    # TL: Actual vs Predicted scatter
    ax = axes[0, 0]
    colors_pts = [COLORS['val_pred']] * len(va_act) + [COLORS['test_pred']] * len(te_act)
    ax.scatter(all_act, all_pred, c=colors_pts, alpha=0.8, s=60, edgecolors='white', lw=0.5)
    mn, mx = min(all_act.min(), all_pred.min()), max(all_act.max(), all_pred.max())
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1.5, label='Ideal (y=x)')
    ax.set_xlabel('Gercek (kWh/kisi)')
    ax.set_ylabel('Tahmin (kWh/kisi)')
    ax.set_title('Gercek vs Tahmin')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    # Val/Test legend
    ax.scatter([], [], c=COLORS['val_pred'],  s=40, label='Val')
    ax.scatter([], [], c=COLORS['test_pred'], s=40, label='Test')
    ax.legend(fontsize=8)

    # TR: Residuals over time
    ax = axes[0, 1]
    ax.bar(all_yr, resid,
           color=[COLORS['val_pred'] if y <= 2018 else COLORS['test_pred'] for y in all_yr],
           alpha=0.8, edgecolor='white')
    ax.axhline(0, color='black', lw=1.5, linestyle='--')
    ax.set_xlabel('Yil')
    ax.set_ylabel('Gercek - Tahmin (kWh/kisi)')
    ax.set_title('Kalıntılar (Residuals) Zaman Serisinde')
    ax.grid(True, alpha=0.3, axis='y')

    # BL: Error histogram
    ax = axes[1, 0]
    ax.hist(resid, bins=min(15, len(resid)), color='#607D8B', edgecolor='white',
            alpha=0.85)
    ax.axvline(0, color='red', lw=1.5, linestyle='--', label='Hatasiz')
    ax.axvline(resid.mean(), color='orange', lw=1.5, linestyle='-',
               label=f'Ort={resid.mean():.1f}')
    ax.set_xlabel('Hata (kWh/kisi)')
    ax.set_ylabel('Frekans')
    ax.set_title('Hata Dagilimi')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # BR: % error by year
    ax = axes[1, 1]
    valid = ~np.isnan(pct_err)
    bar_colors = [COLORS['val_pred'] if y <= 2018 else COLORS['test_pred']
                  for y, v in zip(all_yr, valid) if v]
    ax.bar(all_yr[valid], pct_err[valid], color=bar_colors,
           alpha=0.85, edgecolor='white')
    ax.axhline(0, color='black', lw=1, linestyle='--')
    ax.set_xlabel('Yil')
    ax.set_ylabel('Yuzde Hata (%)')
    ax.set_title('Yila Gore % Hata\n(+ : fazla tahmin, - : az tahmin)')
    ax.grid(True, alpha=0.3, axis='y')

    patches = [mpatches.Patch(color=COLORS['val_pred'],  label='Val (2016-18)'),
               mpatches.Patch(color=COLORS['test_pred'], label='Test (2019-22)')]
    fig.legend(handles=patches, loc='lower center', ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    _save(fig, 'analysis', f'{iso}_analysis.png')


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3: Walk-forward Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

def plot_crossval(country_hist: pd.DataFrame, cv_preds: pd.DataFrame,
                  fold_stats: pd.DataFrame, iso: str, country_name: str):
    """
    Walk-forward cross-validation grafigi.

    Sol panel : Egitim serisinde her fold'un val tahminleri vs gercek
    Sag panel : Her fold icin Train R2 vs Val R2 (yan yana bar)
                Train R2 >> Val R2 → overfitting var
                Train R2 ~ Val R2  → model genelliyor
    """
    cv_c = cv_preds[cv_preds['iso_code'] == iso].sort_values(['fold', 'year'])
    if cv_c.empty:
        return

    folds = sorted(cv_c['fold'].unique())
    n     = len(folds)
    cmap  = plt.cm.get_cmap('tab10', n)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{country_name} ({iso}) — Walk-Forward Cross-Validation ({n} Fold)',
                 fontsize=12)

    # ── Sol: zaman serisinde fold tahminleri ──────────────────────────────
    ax = axes[0]
    tr = country_hist[(country_hist['iso_code'] == iso) &
                      (country_hist['split'] == 'train')].sort_values('year')
    ax.plot(tr['year'], tr['value'], 'o-', color=COLORS['train'],
            lw=1.5, ms=3, alpha=0.5, label='Gercek deger')

    for i, fold in enumerate(folds):
        fc = cv_c[cv_c['fold'] == fold].sort_values('year')
        vr = fold_stats[fold_stats['fold'] == fold]['val_r2'].values[0]
        ax.scatter(fc['year'], fc['actual'],    s=55, marker='x',
                   color=cmap(i), zorder=4)
        ax.scatter(fc['year'], fc['predicted'], s=80, marker='o',
                   color=cmap(i), zorder=5,
                   label=f'Fold {fold+1} tahmin  (Val R²={vr:.4f})')
        # gercek ile tahmin arasini baglayan dikey cizgiler
        for _, row in fc.iterrows():
            ax.plot([row['year'], row['year']],
                    [row['actual'], row['predicted']],
                    '-', color=cmap(i), lw=1, alpha=0.5)

    ax.set_xlabel('Yil')
    ax.set_ylabel('kWh/kisi')
    ax.set_title('Her Fold: Tahmin (daire) vs Gercek (X)')
    ax.legend(fontsize=7.5, loc='upper left')
    ax.grid(True, alpha=0.3)

    # ── Sag: Train R2 vs Val R2 yan yana bar ─────────────────────────────
    ax = axes[1]
    x       = np.arange(n)
    width   = 0.35
    labels  = [f'Fold {f+1}\n{fold_stats[fold_stats["fold"]==f]["val_years"].values[0]}'
               for f in folds]
    tr_r2s  = [fold_stats[fold_stats['fold'] == f]['train_r2'].values[0] for f in folds]
    va_r2s  = [fold_stats[fold_stats['fold'] == f]['val_r2'].values[0]   for f in folds]

    bars_tr = ax.bar(x - width/2, tr_r2s, width, label='Train R²',
                     color='#2196F3', alpha=0.85, edgecolor='white')
    bars_va = ax.bar(x + width/2, va_r2s, width, label='Val R²',
                     color='#FF5722', alpha=0.85, edgecolor='white')

    # Deger etiketleri
    for bar, v in zip(bars_tr, tr_r2s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{v:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold',
                color='#2196F3')
    for bar, v in zip(bars_va, va_r2s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{v:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold',
                color='#FF5722')

    # Ortalama cizgileri
    ax.axhline(np.mean(tr_r2s), color='#2196F3', lw=1.5, linestyle='--',
               alpha=0.7, label=f'Ort. Train R²={np.mean(tr_r2s):.4f}')
    ax.axhline(np.mean(va_r2s), color='#FF5722', lw=1.5, linestyle='--',
               alpha=0.7, label=f'Ort. Val R²={np.mean(va_r2s):.4f}')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('R²')
    ax.set_ylim(max(0, min(tr_r2s + va_r2s) - 0.05), 1.02)
    ax.set_title('Her Fold: Train R² vs Val R²\n'
                 '(fark buyukse overfitting, yakınsa model genelliyor)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    _save(fig, 'crossval', f'{iso}_crossval.png')


# ─────────────────────────────────────────────────────────────────────────────
# Ana cagri: tum ulkeleri isler
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_country_plots(history_df: pd.DataFrame,
                                cv_preds: pd.DataFrame,
                                fold_stats: pd.DataFrame,
                                val_r2s: dict,
                                test_r2s: dict):
    """
    Tum ulkeler icin 3 plot tipi olustur.

    history_df : build_country_history() ciktisi
    cv_preds   : compute_cv_preds()[0] ciktisi
    fold_stats : compute_cv_preds()[1] ciktisi
    val_r2s    : {iso_code: r2_float}
    test_r2s   : {iso_code: r2_float}
    """
    countries = history_df[['iso_code', 'country']].drop_duplicates()
    total = len(countries)

    for i, (_, row) in enumerate(countries.iterrows(), 1):
        iso  = row['iso_code']
        name = row['country']
        ch   = history_df[history_df['iso_code'] == iso]

        vr = val_r2s.get(iso, 0.0)
        tr = test_r2s.get(iso, 0.0)

        plot_forecast(ch, iso, name, vr, tr)
        plot_analysis(ch, iso, name)
        plot_crossval(ch, cv_preds, fold_stats, iso, name)

        if i % 25 == 0 or i == total:
            print(f"  {i}/{total} ulke grafigi tamamlandi")

    print(f"  Tum grafikler: {OUTPUT_BASE}/")
