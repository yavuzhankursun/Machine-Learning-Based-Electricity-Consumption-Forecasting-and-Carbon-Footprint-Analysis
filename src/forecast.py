"""
Tum Ulkeler + Turkiye 2025/2030/2035 Enerji Tuketimi Tahmini
- Lineer trend ile bagimsiz degisken projeksiyonu
- En iyi model ile gelecek tahmini (213 ulke)
- Recursive forecasting
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')


def _project_features(df_country, feature_cols, target_years):
    """Bir ulkenin bagimsiz degiskenlerini lineer trend ile projekte et (sessiz)."""
    years = df_country['year'].values.reshape(-1, 1)
    projections = {y: {} for y in target_years}

    skip_cols = {'year', 'country_encoded',
                 'demand_lag_1', 'demand_lag_2', 'demand_lag_3',
                 'demand_rolling_mean_3', 'demand_rolling_std_3',
                 'demand_yoy_growth'}

    for col in feature_cols:
        if col in skip_cols:
            continue
        values = df_country[col].values
        mask = ~np.isnan(values)
        if mask.sum() < 3:
            val = values[mask].mean() if mask.any() else 0
            for y in target_years:
                projections[y][col] = val
            continue

        lr = LinearRegression()
        lr.fit(years[mask], values[mask])
        for y in target_years:
            projections[y][col] = lr.predict([[y]])[0]

    return projections


def _forecast_single_country(df_country, model, feature_cols, target_years, verbose=False):
    """Tek ulke icin recursive forecasting."""
    country_encoded = df_country['country_encoded'].iloc[0]
    country_code = df_country['iso_code'].iloc[0]

    projections = _project_features(df_country, feature_cols, target_years)

    # Tarihsel buyume orani
    demands = df_country['electricity_demand'].values
    growth_rates = []
    for i in range(1, len(demands)):
        if demands[i-1] > 0:
            growth_rates.append(demands[i] / demands[i-1] - 1)
    recent_growth = np.mean(growth_rates[-10:]) if len(growth_rates) >= 10 else (np.mean(growth_rates) if growth_rates else 0.02)

    # Recursive forecasting
    all_years = list(range(df_country['year'].max() + 1, max(target_years) + 1))
    demand_history = list(demands)
    results = []

    for year in all_years:
        feature_vector = []
        for col in feature_cols:
            if col == 'year':
                feature_vector.append(year)
            elif col == 'country_encoded':
                feature_vector.append(country_encoded)
            elif col == 'demand_lag_1':
                feature_vector.append(demand_history[-1])
            elif col == 'demand_lag_2':
                feature_vector.append(demand_history[-2] if len(demand_history) >= 2 else demand_history[-1])
            elif col == 'demand_lag_3':
                feature_vector.append(demand_history[-3] if len(demand_history) >= 3 else demand_history[-1])
            elif col == 'demand_rolling_mean_3':
                last3 = demand_history[-3:] if len(demand_history) >= 3 else demand_history
                feature_vector.append(np.mean(last3))
            elif col == 'demand_rolling_std_3':
                last3 = demand_history[-3:] if len(demand_history) >= 3 else demand_history
                feature_vector.append(np.std(last3) if len(last3) > 1 else 0)
            elif col == 'demand_yoy_growth':
                if len(demand_history) >= 2 and demand_history[-2] > 0:
                    feature_vector.append((demand_history[-1] - demand_history[-2]) / demand_history[-2])
                else:
                    feature_vector.append(0)
            else:
                closest_year = min(target_years, key=lambda y: abs(y - year))
                feature_vector.append(projections.get(closest_year, {}).get(col, 0))

        pred = model.predict(np.array([feature_vector]))[0]

        # Trend adjustment
        trend_pred = demand_history[-1] * (1 + recent_growth)
        years_ahead = year - 2023
        if years_ahead <= 3:
            w = 0.7
        elif years_ahead <= 8:
            w = 0.5
        else:
            w = 0.3
        pred = w * pred + (1 - w) * trend_pred
        pred = max(pred, 0.01)

        demand_history.append(pred)
        if year in target_years:
            results.append({'year': year, 'predicted_demand_twh': round(pred, 2)})

    return results, recent_growth


def forecast_all_countries(data, best_result, target_years=[2025, 2030, 2035]):
    """Tum 213 ulke icin gelecek yillarin elektrik tuketimini tahmin et."""
    print("\n=== TUM ULKELER ENERJI TUKETIMI TAHMINI ===")
    print(f"  Model: {best_result['name']}")

    df = data['full_df']
    feature_cols = data['feature_cols']
    model = best_result['model']

    countries = df['iso_code'].unique()
    print(f"  Ulke sayisi: {len(countries)}")

    all_forecasts = []
    turkey_results = None
    turkey_history = None

    for i, iso in enumerate(countries):
        df_c = df[df['iso_code'] == iso].sort_values('year')
        if len(df_c) < 5:
            continue

        results, growth = _forecast_single_country(df_c, model, feature_cols, target_years)
        country_name = df_c['country'].iloc[0]
        last_demand = df_c['electricity_demand'].iloc[-1]

        for r in results:
            all_forecasts.append({
                'iso_code': iso,
                'country': country_name,
                'year': r['year'],
                'predicted_demand_twh': r['predicted_demand_twh'],
                'last_actual_twh': round(last_demand, 2),
                'growth_rate': round(growth * 100, 2),
            })

        # Turkiye ozel rapor
        if iso == 'TUR':
            turkey_results = results
            turkey_history = pd.DataFrame({
                'year': df_c['year'].values.tolist() + [r['year'] for r in results],
                'electricity_demand_twh': df_c['electricity_demand'].values.tolist() +
                                          [r['predicted_demand_twh'] for r in results],
                'type': ['actual'] * len(df_c) + ['predicted'] * len(results)
            })

    # Kaydet
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    forecast_df = pd.DataFrame(all_forecasts)
    forecast_df.to_csv(os.path.join(OUTPUT_DIR, 'all_countries_forecast.csv'), index=False)

    if turkey_history is not None:
        turkey_history.to_csv(os.path.join(OUTPUT_DIR, 'turkey_forecast.csv'), index=False)

    # Ozet
    print(f"\n  Tahmin uretilen ulke: {forecast_df['iso_code'].nunique()}")
    print(f"  Toplam tahmin satiri: {len(forecast_df)}")

    # Turkiye detay
    if turkey_results:
        last_tr = df[df['iso_code'] == 'TUR']['electricity_demand'].iloc[-1]
        print(f"\n  Turkiye Tahminleri:")
        for r in turkey_results:
            change = ((r['predicted_demand_twh'] - last_tr) / last_tr) * 100
            print(f"    {r['year']}: {r['predicted_demand_twh']:.1f} TWh ({change:+.1f}%)")

    # En buyuk 10 ulke
    top10 = forecast_df[forecast_df['year'] == 2030].nlargest(10, 'predicted_demand_twh')
    print(f"\n  2030 En Buyuk 10 Tuketici:")
    for _, row in top10.iterrows():
        print(f"    {row['country']}: {row['predicted_demand_twh']:.1f} TWh")

    return forecast_df, turkey_results, turkey_history
