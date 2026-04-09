"""
Ana Pipeline - Tum adimlari sirasiyla calistirir
1. Veri birlestirme ve temizleme
2. Feature engineering
3. Model egitimi (RF, XGBoost, LSTM)
4. Akademik dogrulama testleri
5. Turkiye tahmini (2025, 2030, 2035)
6. Karbon ayak izi hesaplamasi
7. Gorsellestirme
"""
import sys
import os
import warnings
import time

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# src/ dizinini path'e ekle
sys.path.insert(0, os.path.dirname(__file__))

from data_merge import load_and_merge, save_merged
from feature_engineering import prepare_data
from model_rf import train_random_forest
from model_xgboost import train_xgboost
from model_lstm import train_lstm
from model_comparison import (naive_baseline, comparison_table, correlation_analysis,
                              ablation_study, residual_analysis, overfitting_check,
                              diebold_mariano_test, run_all_tests)
from forecast import forecast_all_countries
from carbon_footprint import calculate_all_carbon
from visualize import generate_all_plots


def generate_dashboard_json(merged_df, forecast_df, carbon_df, emission_factors):
    """Dashboard icin harici JSON veri dosyasi olustur."""
    import json

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')

    countries = []
    for iso, grp in merged_df.groupby('iso_code'):
        grp = grp.sort_values('year')
        last = grp.iloc[-1]
        pop = last.get('Population', 0)
        elec_pc = (last['electricity_demand'] * 1e9 / pop) if pop > 0 else 0

        # Zaman serileri (compact format)
        ts = [{'y': int(r['year']), 'v': round(r['electricity_demand'], 2)} for _, r in grp.iterrows()]
        ts_g = [{'y': int(r['year']), 'v': round(r['GDP_Per_Capita'], 1)} for _, r in grp.iterrows()]
        ts_u = [{'y': int(r['year']), 'v': round(r['Urbanization_Rate'], 2)} for _, r in grp.iterrows()]
        ts_e = [{'y': int(r['year']), 'v': round(r['electricity_demand'] * 1e9 / r['Population'], 0) if r['Population'] > 0 else 0} for _, r in grp.iterrows()]

        # Ulke tahmini
        c_forecast = forecast_df[forecast_df['iso_code'] == iso][['year', 'predicted_demand_twh']].to_dict('records') if forecast_df is not None else []

        # Ulke karbon tahmini
        c_carbon = []
        if carbon_df is not None:
            cc = carbon_df[(carbon_df['iso_code'] == iso) & (carbon_df['type'] == 'predicted')]
            c_carbon = [{'y': int(r['year']), 'co2': round(r['co2_mt'], 2), 'ef': round(r['emission_factor'], 4)} for _, r in cc.iterrows()]

        countries.append({
            'c': iso, 'n': last['country'],
            'd': round(last['electricity_demand'], 2),
            'e': round(elec_pc, 0),
            'g': round(last.get('GDP_Per_Capita', 0), 0),
            'p': round(pop, 0),
            'u': round(last.get('Urbanization_Rate', 0), 1),
            'ef': emission_factors.get(iso, 0.45),
            'ts': ts, 'ts_g': ts_g, 'ts_u': ts_u, 'ts_e': ts_e,
            'forecast': c_forecast,
            'carbon': c_carbon,
        })

    data = {
        'countries': countries,
        'meta': {
            'total_countries': len(countries),
            'year_range': [int(merged_df['year'].min()), int(merged_df['year'].max())],
            'forecast_years': [2025, 2030, 2035],
        }
    }

    path = os.path.join(output_dir, 'dashboard_data.json')
    with open(path, 'w') as f:
        json.dump(data, f, separators=(',', ':'))

    size_kb = os.path.getsize(path) / 1024
    print(f"  Dashboard verisi: {path} ({size_kb:.0f} KB)")
    print(f"  Ulke sayisi: {len(countries)}")
    print(f"  Tahmin iceren ulke: {sum(1 for c in countries if c['forecast'])}")
    print(f"  Karbon iceren ulke: {sum(1 for c in countries if c['carbon'])}")


def main():
    start = time.time()
    print("=" * 80)
    print("MAKINE OGRENMESI ILE ENERJI TUKETIMI TAHMINI VE KARBON AYAK IZI ANALIZI")
    print("TUBiTAK 2209-A Projesi")
    print("=" * 80)

    # --- 1. Veri birlestirme ---
    print("\n[ADIM 1/7] Veri birlestirme ve temizleme...")
    df = load_and_merge()
    save_merged(df)

    # --- 2. Feature engineering ---
    print("\n[ADIM 2/7] Feature engineering...")
    data = prepare_data(df)

    # --- 3. Model egitimi ---
    print("\n[ADIM 3/7] Model egitimi...")

    # Naive baseline
    naive = naive_baseline(data)

    # Random Forest (baseline)
    rf_result = train_random_forest(data)

    # XGBoost (ana model)
    xgb_result = train_xgboost(data)

    # LSTM (deep learning)
    lstm_result = train_lstm(data)

    # Sonuclari topla
    results = [naive, rf_result, xgb_result]
    if lstm_result:
        results.append(lstm_result)

    # --- 4. Akademik dogrulama testleri ---
    print("\n[ADIM 4/7] Akademik dogrulama testleri...")
    test_output = run_all_tests(data, results)

    # Ablation study
    ablation_results = ablation_study(data, train_xgboost)

    # --- 5. Tum ulkeler tahmini ---
    print("\n[ADIM 5/8] Tum ulkeler enerji tuketimi tahmini...")
    tree_models = [r for r in results if r['name'] in ('Random Forest', 'XGBoost')]
    best_model = min(tree_models, key=lambda x: x['metrics_test']['mape'])
    print(f"\n  En iyi model: {best_model['name']} (Test MAPE: {best_model['metrics_test']['mape']:.2f}%)")
    forecast_df, turkey_results, turkey_history = forecast_all_countries(data, best_model)

    # --- 6. Tum ulkeler karbon ayak izi ---
    print("\n[ADIM 6/8] Karbon ayak izi hesaplamasi (tum ulkeler)...")
    carbon_df, emission_factors = calculate_all_carbon(forecast_df, df)

    # --- 7. Gorsellestirme ---
    print("\n[ADIM 7/8] Grafik olusturma...")
    # turkey_carbon sadece Turkiye icin
    turkey_carbon = carbon_df[carbon_df['iso_code'] == 'TUR'] if carbon_df is not None else None
    generate_all_plots(data, results, test_output['comparison_df'],
                       ablation_results, turkey_history, turkey_carbon)

    # --- 8. Dashboard verisi olustur ---
    print("\n[ADIM 8/8] Dashboard verisi olusturuluyor...")
    generate_dashboard_json(df, forecast_df, carbon_df, emission_factors)

    # --- Ozet ---
    elapsed = time.time() - start
    print("\n" + "=" * 80)
    print("PIPELINE TAMAMLANDI")
    print("=" * 80)
    print(f"\n  Sure: {elapsed:.1f} saniye")
    print(f"  Veri: {df.shape[0]} satir, {df.shape[1]} sutun, {df['iso_code'].nunique()} ulke")
    print(f"  Feature sayisi: {len(data['feature_cols'])}")
    print(f"  En iyi model: {best_model['name']}")
    print(f"    Test MAPE: {best_model['metrics_test']['mape']:.2f}%")
    print(f"    Test R2: {best_model['metrics_test']['r2']:.4f}")
    print(f"  Tahmin uretilen ulke: {forecast_df['iso_code'].nunique()}")
    print(f"  Karbon hesaplanan ulke: {carbon_df['iso_code'].nunique() if carbon_df is not None else 0}")
    if turkey_results:
        print(f"\n  Turkiye Tahminleri:")
        for r in turkey_results:
            print(f"    {r['year']}: {r['predicted_demand_twh']:.1f} TWh")
    print(f"\n  Cikti dosyalari: output/")
    print(f"  Grafikler: output/figures/")


if __name__ == '__main__':
    main()
