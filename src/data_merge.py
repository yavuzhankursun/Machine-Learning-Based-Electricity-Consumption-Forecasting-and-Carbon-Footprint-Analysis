"""
Veri Birlestirme ve Temizleme
- 5 CSV'yi country + year ile merge eder
- Leakage degiskenleri cikarir
- Eksik veri imputasyonu yapar
"""
import pandas as pd
import numpy as np
import os
from scipy import stats

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'collected')
EXTENDED_DIR = os.path.join(os.path.dirname(__file__), '..', 'collected', 'extended')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')


# --- Leakage riski tasiyan degiskenler (KESINLIKLE kullanilmayacak) ---
LEAKAGE_COLUMNS = [
    # CO2 - enerji tuketiminden turetilmis
    'co2', 'co2_per_capita', 'co2_per_gdp',
    'coal_co2', 'oil_co2', 'gas_co2', 'cement_co2',
    'energy_per_capita', 'energy_per_gdp',
    # Climate - enerji tuketiminin fonksiyonu
    'total_ghg', 'total_ghg_excluding_lucf',
    'ghg_per_capita', 'ghg_excluding_lucf_per_capita',
    'methane', 'methane_per_capita',
    'nitrous_oxide', 'nitrous_oxide_per_capita',
    # Energy - hedef veya bilesenler
    'electricity_generation',
    'fossil_fuel_consumption', 'nuclear_consumption',
    'primary_energy_consumption',
    'per_capita_electricity',
    # Industry - hedef veya turetilmis
    'Electric_Power_Consumption_kWh_Per_Capita',
    'Energy_Use_Per_Capita',
    'Electricity_Loss_Pct',
]


def read_csv_skip_comments(path):
    """Yorum satirlarini (#) atlayarak CSV oku."""
    with open(path) as f:
        lines = [line for line in f if not line.startswith('#')]
    from io import StringIO
    return pd.read_csv(StringIO(''.join(lines)))


def load_and_merge():
    """Genisletilmis veri varsa onu kullan, yoksa 5 CSV'yi birlestir."""

    # --- Genisletilmis veri kontrolu (1990-2025) ---
    extended_path = os.path.join(EXTENDED_DIR, 'merged_extended.csv')
    if os.path.exists(extended_path):
        print("  Genisletilmis veri kullaniliyor (1990-2025)...")
        merged = pd.read_csv(extended_path)
        merged['year'] = merged['year'].astype(int)

        # Leakage kontrolu
        for col in LEAKAGE_COLUMNS:
            if col in merged.columns:
                print(f"  LEAKAGE UYARISI: '{col}' siliniyor!")
                merged = merged.drop(columns=[col])

        # Temel temizlik
        merged = merged.dropna(subset=['electricity_demand', 'iso_code'])
        merged = merged[merged['electricity_demand'] > 0]
        merged = merged[merged['iso_code'].str.len() == 3]

        # Outlier detection (hedef degisken haric)
        numeric_cols_for_outlier = ['GDP_Per_Capita', 'GDP_Growth', 'Population_Density']
        numeric_cols_for_outlier = [c for c in numeric_cols_for_outlier if c in merged.columns]
        outlier_report = {}
        for col in numeric_cols_for_outlier:
            data_col = merged[col].dropna()
            if len(data_col) < 10:
                continue
            Q1 = data_col.quantile(0.25)
            Q3 = data_col.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            iqr_outliers = ((merged[col] < lower) | (merged[col] > upper)).sum()
            z_scores = np.abs(stats.zscore(data_col))
            zscore_outliers = (z_scores > 3).sum()
            outlier_report[col] = {'iqr_outliers': int(iqr_outliers), 'zscore_outliers': int(zscore_outliers)}
            merged[col] = merged[col].clip(lower=lower, upper=upper)

        if outlier_report:
            print(f"\nAykiri Deger Raporu (IQR + z-score):")
            for col, info in outlier_report.items():
                print(f"  {col}: IQR outliers={info['iqr_outliers']}, z-score outliers={info['zscore_outliers']}")

        # NaN doldurmak
        fill_cols = [c for c in merged.columns if c not in ('country', 'iso_code', 'year')]
        merged = merged.sort_values(['iso_code', 'year'])
        merged[fill_cols] = merged.groupby('iso_code')[fill_cols].transform(lambda x: x.ffill().bfill())
        for col in fill_cols:
            if merged[col].isna().sum() > 0:
                merged[col] = merged[col].fillna(merged[col].median())

        print(f"\nBirlesmis veri: {merged.shape[0]} satir, {merged.shape[1]} sutun")
        print(f"Ulke sayisi: {merged['iso_code'].nunique()}")
        print(f"Yil araligi: {merged['year'].min()} - {merged['year'].max()}")
        print(f"Sutunlar: {list(merged.columns)}")
        print(f"\nEksik deger sayisi:\n{merged.isna().sum()}")

        tr = merged[merged['iso_code'] == 'TUR']
        print(f"\nTurkiye: {len(tr)} satir, yillar: {tr['year'].min()}-{tr['year'].max()}")
        print(f"Turkiye electricity_demand araligi: {tr['electricity_demand'].min():.1f} - {tr['electricity_demand'].max():.1f} TWh")

        return merged

    # --- Fallback: 5 CSV birlestirme (eski yontem) ---
    print("  Genisletilmis veri bulunamadi, 5 CSV birlestiriliyor...")

    # 1. World Bank
    wb = read_csv_skip_comments(os.path.join(DATA_DIR, 'world_bank_indicators.csv'))
    wb = wb.rename(columns={'Country': 'country', 'Country_Code': 'iso_code', 'Year': 'year'})

    # 2. Climate / Temperature
    climate = read_csv_skip_comments(os.path.join(DATA_DIR, 'climate_temperature_data.csv'))

    # 3. CO2 Emissions
    co2 = read_csv_skip_comments(os.path.join(DATA_DIR, 'owid_co2_emissions.csv'))

    # 4. Industry / Energy Indicators
    industry = read_csv_skip_comments(os.path.join(DATA_DIR, 'industry_energy_indicators.csv'))
    industry = industry.rename(columns={'Country': 'country', 'Country_Code': 'iso_code', 'Year': 'year'})

    # 5. Energy Data (hedef degisken buradan)
    energy = read_csv_skip_comments(os.path.join(DATA_DIR, 'owid_energy_data.csv'))

    # --- Birlestirme (iso_code + year uzerinden) ---
    # iso_code + year en guvenilir key
    merged = energy[['country', 'year', 'iso_code', 'electricity_demand',
                      'renewables_share_elec']].copy()

    # World Bank
    wb_cols = ['iso_code', 'year', 'Population', 'Population_Density',
               'Urbanization_Rate', 'GDP_Growth', 'GDP_Per_Capita']
    merged = merged.merge(wb[wb_cols], on=['iso_code', 'year'], how='left')

    # Industry (sadece guvenli sutunlar)
    ind_cols = ['iso_code', 'year', 'Industry_Value_Added_Pct_GDP',
                'Renewable_Energy_Consumption_Pct']
    merged = merged.merge(industry[ind_cols], on=['iso_code', 'year'], how='left')

    # Climate (sadece sicaklik degisimi gostergeleri)
    clim_cols = ['iso_code', 'year',
                 'temperature_change_from_ghg',
                 'share_of_temperature_change_from_ghg']
    merged = merged.merge(climate[clim_cols], on=['iso_code', 'year'], how='left')

    # --- Leakage kontrolu ---
    for col in LEAKAGE_COLUMNS:
        if col in merged.columns:
            print(f"  LEAKAGE UYARISI: '{col}' siliniyor!")
            merged = merged.drop(columns=[col])

    # --- Temel temizlik ---
    # Hedef degisken eksikse o satiri at
    merged = merged.dropna(subset=['electricity_demand'])

    # electricity_demand <= 0 olanlari at
    merged = merged[merged['electricity_demand'] > 0]

    # iso_code eksik olanlari at (agregat bolge satirlari)
    merged = merged.dropna(subset=['iso_code'])
    merged = merged[merged['iso_code'].str.len() == 3]

    # Yil'i integer yap
    merged['year'] = merged['year'].astype(int)

    # --- Numerik sutunlari float'a cevir ---
    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = [c for c in merged.columns
                   if c not in numeric_cols and c not in ['country', 'iso_code']]
    for col in non_numeric:
        merged[col] = pd.to_numeric(merged[col], errors='coerce')

    # --- Aykiri deger tespiti (z-score + IQR) ---
    # NOT: electricity_demand (hedef degisken) CLIP EDILMEZ — buyuk ulkelerin
    # gercek degerleri IQR tarafindan "outlier" olarak isaretlenir ama bunlar dogrudur.
    # Sadece yardimci degiskenleri clip ediyoruz.
    numeric_cols_for_outlier = ['GDP_Per_Capita', 'GDP_Growth', 'Population_Density']
    outlier_report = {}
    for col in numeric_cols_for_outlier:
        if col not in merged.columns:
            continue
        data_col = merged[col].dropna()
        if len(data_col) < 10:
            continue

        # IQR method
        Q1 = data_col.quantile(0.25)
        Q3 = data_col.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        iqr_outliers = ((merged[col] < lower) | (merged[col] > upper)).sum()

        # z-score method
        z_scores = np.abs(stats.zscore(data_col))
        zscore_outliers = (z_scores > 3).sum()

        outlier_report[col] = {'iqr_outliers': int(iqr_outliers), 'zscore_outliers': int(zscore_outliers), 'lower': lower, 'upper': upper}

        # Clip extreme outliers (IQR method) - don't remove, just cap
        merged[col] = merged[col].clip(lower=lower, upper=upper)

    print(f"\nAykiri Deger Raporu (IQR + z-score):")
    for col, info in outlier_report.items():
        print(f"  {col}: IQR outliers={info['iqr_outliers']}, z-score outliers={info['zscore_outliers']}")

    # --- Eksik deger imputasyonu ---
    # Ulke bazinda forward-fill + backward-fill (zaman serisi icin en uygun)
    fill_cols = [c for c in merged.columns if c not in ['country', 'iso_code', 'year']]
    merged = merged.sort_values(['iso_code', 'year'])
    merged[fill_cols] = merged.groupby('iso_code')[fill_cols].transform(
        lambda x: x.ffill().bfill()
    )

    # Hala eksik olan numerik sutunlari global medyan ile doldur
    for col in fill_cols:
        if merged[col].isna().sum() > 0:
            merged[col] = merged[col].fillna(merged[col].median())

    print(f"\nBirlesmis veri: {merged.shape[0]} satir, {merged.shape[1]} sutun")
    print(f"Ulke sayisi: {merged['iso_code'].nunique()}")
    print(f"Yil araligi: {merged['year'].min()} - {merged['year'].max()}")
    print(f"\nSutunlar: {list(merged.columns)}")
    print(f"\nEksik deger sayisi:\n{merged.isna().sum()}")

    # Turkiye kontrolu
    tr = merged[merged['iso_code'] == 'TUR']
    print(f"\nTurkiye: {len(tr)} satir, yillar: {tr['year'].min()}-{tr['year'].max()}")
    print(f"Turkiye electricity_demand araligi: {tr['electricity_demand'].min():.1f} - {tr['electricity_demand'].max():.1f} TWh")

    return merged


def save_merged(df):
    """Birlesmis veriyi kaydet."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, 'merged_clean.csv')
    df.to_csv(path, index=False)
    print(f"\nKaydedildi: {path}")
    return path


if __name__ == '__main__':
    df = load_and_merge()
    save_merged(df)
