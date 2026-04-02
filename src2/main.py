"""
Ana Pipeline — Kisi Basina Elektrik Tuketimi Tahmini

Adimlar:
  1. Veri yukle (150 ulke, 1990-2022)
  2. Feature engineering (data leakage YOK)
  3. Model egitimi
       a) Random Forest (baseline)
       b) XGBoost + Optuna (n_trials=30)
       c) LSTM (epochs=200, batch=32, early_stopping=20)
  4. Walk-forward cross-validation (n_splits=5)
  5. 2023-2028 tahmini (recursive forecasting)
  6. Per-ulke grafik olustur
       - {CODE}_forecast.png  : zaman serisi (train/val/test/tahmin)
       - {CODE}_analysis.png  : hata analizi (4 panel)
       - {CODE}_crossval.png  : walk-forward CV
  7. Sonuclari kaydet

Ciktilar: ml-aaaa/output2/
"""
import os, sys, time, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from data_loader      import load_data
from feature_engineering import prepare_data
from model_rf         import train_random_forest
from model_xgboost    import train_xgboost
from model_lstm       import train_lstm
from forecast         import forecast_all_countries, build_country_history
from crossval         import compute_cv_preds
from visualize        import generate_all_country_plots
from metrics          import mape

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output2')


def save_results(forecast_df: pd.DataFrame, metrics_rows: list):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    forecast_df.to_csv(os.path.join(OUTPUT_DIR, 'all_forecasts.csv'), index=False)
    pd.DataFrame(metrics_rows).to_csv(os.path.join(OUTPUT_DIR, 'metrics.csv'), index=False)
    print(f"\n  Sonuclar kaydedildi: {OUTPUT_DIR}/")


def compute_per_country_r2(data: dict, result: dict):
    """Her ulke icin val ve test R2'sini hesapla."""
    from sklearn.metrics import r2_score

    val_df  = data['val_df'].reset_index(drop=True).copy()
    test_df = data['test_df'].reset_index(drop=True).copy()
    val_df['pred']  = result['pred_val']
    test_df['pred'] = result['pred_test']

    val_r2s  = {}
    test_r2s = {}
    for iso in val_df['iso_code'].unique():
        v = val_df[val_df['iso_code'] == iso]
        t = test_df[test_df['iso_code'] == iso]
        val_r2s[iso]  = r2_score(v['elec_per_capita'].values, v['pred'].values)
        test_r2s[iso] = r2_score(t['elec_per_capita'].values, t['pred'].values)

    return val_r2s, test_r2s


def main():
    t0 = time.time()
    print("=" * 70)
    print("KISI BASINA ELEKTRIK TUKETIMI — ML TAHMIN PIPELINE")
    print("=" * 70)

    # ── 1. Veri ────────────────────────────────────────────────────────────
    print("\n[1/7] Veri yukleniyor...")
    df_raw = load_data()

    # ── 2. Feature engineering ─────────────────────────────────────────────
    print("\n[2/7] Feature engineering...")
    data = prepare_data(df_raw)

    # ── 3. Model egitimi ────────────────────────────────────────────────────
    print("\n[3/7] Model egitimi...")

    rf_result  = train_random_forest(data)
    xgb_result = train_xgboost(data, n_trials=30)
    lstm_result = train_lstm(data, epochs=200, batch_size=32)

    results = [rf_result, xgb_result]
    if lstm_result:
        results.append(lstm_result)

    # Metrics ozeti
    metrics_rows = []
    print("\n  --- Model Karsilastirma ---")
    for r in results:
        print(f"  {r['name']:15s} | "
              f"Val MAPE={r['metrics_val']['mape']:5.2f}% | "
              f"Test MAPE={r['metrics_test']['mape']:5.2f}% | "
              f"Test R2={r['metrics_test']['r2']:.4f}")
        metrics_rows.append({
            'model'     : r['name'],
            'val_mape'  : r['metrics_val']['mape'],
            'val_r2'    : r['metrics_val']['r2'],
            'test_mape' : r['metrics_test']['mape'],
            'test_r2'   : r['metrics_test']['r2'],
            'test_mae'  : r['metrics_test']['mae'],
            'test_rmse' : r['metrics_test']['rmse'],
        })

    # En iyi model (val MAPE'ye gore)
    best = min(results, key=lambda r: r['metrics_val']['mape'])
    print(f"\n  En iyi model: {best['name']} (Val MAPE={best['metrics_val']['mape']:.2f}%)")

    # ── 4. Cross-validation ─────────────────────────────────────────────────
    print("\n[4/7] Walk-forward cross-validation...")
    cv_preds, fold_stats = compute_cv_preds(data, best, n_splits=5)

    # ── 5. Recursive forecasting 2023-2028 ──────────────────────────────────
    print("\n[5/7] 2023-2028 recursive tahmini...")
    forecast_df = forecast_all_countries(data, best)

    # Turkiye ozeti
    tur = forecast_df[forecast_df['iso_code'] == 'TUR']
    if len(tur):
        print("\n  Turkiye 2023-2028 tahmini:")
        for _, row in tur.iterrows():
            print(f"    {int(row['year'])}: {row['predicted_elec_per_capita']:.0f} kWh/kisi")

    # ── 6. Per-ulke gecmis + tahmin tablosu ─────────────────────────────────
    print("\n[6/7] Ulke bazli zaman serileri hazirlaniyor...")
    history_df = build_country_history(data, best, forecast_df)

    # Per-ulke R2 hesapla
    val_r2s, test_r2s = compute_per_country_r2(data, best)

    # ── 7. Grafik olustur ───────────────────────────────────────────────────
    print("\n[7/7] Per-ulke grafikler olusturuluyor (150 ulke x 3 grafik)...")
    generate_all_country_plots(history_df, cv_preds, fold_stats, val_r2s, test_r2s)

    # ── Sonuclari kaydet ────────────────────────────────────────────────────
    save_results(forecast_df, metrics_rows)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"PIPELINE TAMAMLANDI — {elapsed:.0f} saniye")
    print(f"  Grafik dizini : output2/figures/")
    print(f"  Tahminler     : output2/all_forecasts.csv")
    print(f"  Metrikler     : output2/metrics.csv")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
