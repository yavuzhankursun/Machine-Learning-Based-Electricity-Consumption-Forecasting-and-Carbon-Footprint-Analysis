"""
Tum Ulkeler Icin Karbon Ayak Izi Hesaplamasi
- owid_co2_emissions.csv'den ulke bazli emisyon faktoru turet
- Tahmin edilen elektrik tuketiminden CO2 emisyonu hesapla
- Data leakage YOK: emisyon faktoru model feature'i degil, post-processing
"""
import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'collected')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')


def load_emission_factors():
    """owid_co2_emissions.csv'den ulke bazli emisyon faktoru hesapla."""
    path = os.path.join(DATA_DIR, 'owid_co2_emissions.csv')
    with open(path) as f:
        lines = [l for l in f if not l.startswith('#')]
    from io import StringIO
    co2_df = pd.read_csv(StringIO(''.join(lines)))

    # Emisyon faktoru = co2 (Mt) / energy_per_capita * population (TWh'ye cevir)
    # Daha basit: co2 (Mt CO2) / electricity_demand (TWh) ≈ Mt CO2 / TWh
    # Ama co2 toplam CO2 (sadece elektrik degil) oldugundan,
    # elektrik CO2 orani ~%40 olarak kabul edelim (IEA global ortalama)

    # Her ulke icin son 5 yilin ortalamasi
    emission_factors = {}
    for iso in co2_df['iso_code'].dropna().unique():
        if not isinstance(iso, str) or len(iso) != 3:
            continue
        country_data = co2_df[co2_df['iso_code'] == iso].sort_values('year')
        recent = country_data.tail(5)

        co2_vals = pd.to_numeric(recent['co2'], errors='coerce').dropna()
        energy_vals = pd.to_numeric(recent['energy_per_capita'], errors='coerce').dropna()

        if len(co2_vals) >= 2 and len(energy_vals) >= 2:
            avg_co2 = co2_vals.mean()  # Mt CO2/yil toplam
            # energy_per_capita kWh cinsinden → ulke toplam enerji
            # Basitlestirme: ulke bazli ortalama faktor
            # kg CO2 / kWh ≈ (co2 Mt * 1e9 kg) / (energy_per_capita kWh * population)
            pop_vals = None
            # Population burada yok, merged_clean'den aliriz
            # Su an icin co2_per_capita kullanarak yakla
            co2_pc = pd.to_numeric(recent['co2_per_capita'], errors='coerce').dropna()
            energy_pc = pd.to_numeric(recent['energy_per_capita'], errors='coerce').dropna()

            if len(co2_pc) >= 2 and len(energy_pc) >= 2:
                # co2_per_capita: ton CO2/kisi, energy_per_capita: kWh/kisi
                # emission_factor = co2_per_capita * 1000 / energy_per_capita (kg CO2 / kWh)
                avg_co2_pc = co2_pc.mean()
                avg_energy_pc = energy_pc.mean()
                if avg_energy_pc > 0:
                    ef = (avg_co2_pc * 1000) / avg_energy_pc  # kg CO2 / kWh
                    ef = np.clip(ef, 0.05, 2.0)  # Mantikli aralikta tut
                    emission_factors[iso] = round(ef, 4)

    # Global ortalama (eksik ulkeler icin)
    if emission_factors:
        global_avg = np.median(list(emission_factors.values()))
    else:
        global_avg = 0.45

    return emission_factors, global_avg


def calculate_all_carbon(forecast_df, merged_df=None):
    """Tum ulkeler icin karbon ayak izi hesapla."""
    print("\n=== TUM ULKELER KARBON AYAK IZI HESAPLAMASI ===")

    emission_factors, global_avg = load_emission_factors()
    print(f"  Emisyon faktoru bulunan ulke: {len(emission_factors)}")
    print(f"  Global medyan EF: {global_avg:.3f} kg CO2/kWh")

    results = []

    # Gercek veriler (gecmis)
    if merged_df is not None:
        for _, row in merged_df.iterrows():
            iso = row['iso_code']
            ef = emission_factors.get(iso, global_avg)
            demand_twh = row['electricity_demand']
            co2_mt = (demand_twh * 1e9 * ef) / 1e9  # TWh → kWh → kg CO2 → Mt
            results.append({
                'iso_code': iso,
                'country': row['country'],
                'year': int(row['year']),
                'electricity_demand_twh': round(demand_twh, 2),
                'emission_factor': round(ef, 4),
                'co2_mt': round(co2_mt, 2),
                'type': 'actual',
            })

    # Tahmin veriler (gelecek)
    for _, row in forecast_df.iterrows():
        iso = row['iso_code']
        ef_base = emission_factors.get(iso, global_avg)

        # Gelecekte emisyon faktoru yillik %1.5 azalir (yenilenebilir artisi)
        years_ahead = row['year'] - 2023
        ef = ef_base * (0.985 ** years_ahead)

        demand_twh = row['predicted_demand_twh']
        co2_mt = (demand_twh * 1e9 * ef) / 1e9

        results.append({
            'iso_code': iso,
            'country': row['country'],
            'year': int(row['year']),
            'electricity_demand_twh': round(demand_twh, 2),
            'emission_factor': round(ef, 4),
            'co2_mt': round(co2_mt, 2),
            'type': 'predicted',
        })

    carbon_df = pd.DataFrame(results)

    # Kaydet
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Tum ulkeler (sadece tahmin yillari — gecmis cok buyuk olur)
    predicted_carbon = carbon_df[carbon_df['type'] == 'predicted']
    predicted_carbon.to_csv(os.path.join(OUTPUT_DIR, 'all_countries_carbon.csv'), index=False)

    # Turkiye detay (gecmis + tahmin)
    turkey_carbon = carbon_df[carbon_df['iso_code'] == 'TUR']
    turkey_carbon.to_csv(os.path.join(OUTPUT_DIR, 'carbon_footprint.csv'), index=False)

    # Ozet
    if len(predicted_carbon) > 0:
        print(f"\n  Karbon tahmin uretilen ulke: {predicted_carbon['iso_code'].nunique()}")
        top10_2030 = predicted_carbon[predicted_carbon['year'] == 2030].nlargest(10, 'co2_mt')
        print(f"\n  2030 En Yuksek CO2 Emisyonu (Top 10):")
        for _, row in top10_2030.iterrows():
            print(f"    {row['country']}: {row['co2_mt']:.1f} Mt CO2 (EF: {row['emission_factor']:.3f})")

    # Turkiye detay
    tr_pred = turkey_carbon[turkey_carbon['type'] == 'predicted']
    if len(tr_pred) > 0:
        tr_last_actual = turkey_carbon[(turkey_carbon['type'] == 'actual') & (turkey_carbon['year'] == 2023)]
        base_co2 = tr_last_actual['co2_mt'].values[0] if len(tr_last_actual) > 0 else 0
        print(f"\n  Turkiye Karbon Ayak Izi:")
        for _, row in tr_pred.iterrows():
            change = ((row['co2_mt'] - base_co2) / base_co2 * 100) if base_co2 > 0 else 0
            print(f"    {row['year']}: {row['co2_mt']:.1f} Mt CO2 ({change:+.1f}%)")

    return carbon_df, emission_factors
