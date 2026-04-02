"""
Genisletilmis Veri Toplama — 1965-2023
OWID Energy + World Bank verilerini 1965'e kadar uzat.
Kucuk ulkelerdeki eksik yillari akilli imputasyon ile doldur.
"""
import pandas as pd
import numpy as np
import os
import urllib.request
import json

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'collected')
EXTENDED_DIR = os.path.join(os.path.dirname(__file__), '..', 'collected', 'extended')


def download_owid_energy():
    """OWID Energy tam veri setini indir (1965-2023)."""
    url = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"
    path = os.path.join(EXTENDED_DIR, 'owid_energy_full.csv')

    if os.path.exists(path):
        print(f"  Zaten mevcut: {path}")
        return pd.read_csv(path)

    print(f"  Indiriliyor: OWID Energy (tam veri)...")
    urllib.request.urlretrieve(url, path)
    print(f"  Kaydedildi: {path}")
    return pd.read_csv(path)


def download_owid_co2():
    """OWID CO2 tam veri setini indir."""
    url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
    path = os.path.join(EXTENDED_DIR, 'owid_co2_full.csv')

    if os.path.exists(path):
        print(f"  Zaten mevcut: {path}")
        return pd.read_csv(path)

    print(f"  Indiriliyor: OWID CO2 (tam veri)...")
    urllib.request.urlretrieve(url, path)
    print(f"  Kaydedildi: {path}")
    return pd.read_csv(path)


def download_world_bank_extended():
    """World Bank verilerini API uzerinden 1965-2023 arasi cek."""
    indicators = {
        'SP.POP.TOTL': 'Population',
        'EN.POP.DNST': 'Population_Density',
        'SP.URB.TOTL.IN.ZS': 'Urbanization_Rate',
        'NY.GDP.MKTP.KD.ZG': 'GDP_Growth',
        'NY.GDP.PCAP.CD': 'GDP_Per_Capita',
        'NV.IND.TOTL.ZS': 'Industry_Value_Added_Pct_GDP',
    }

    path = os.path.join(EXTENDED_DIR, 'world_bank_extended.csv')
    if os.path.exists(path):
        print(f"  Zaten mevcut: {path}")
        return pd.read_csv(path)

    print("  Indiriliyor: World Bank (1965-2023)...")
    all_data = []

    for indicator_code, col_name in indicators.items():
        try:
            # World Bank API: 50 yillik veri
            api_url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator_code}?date=1965:2023&format=json&per_page=20000"
            req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60) as response:
                data = json.loads(response.read().decode())

            if len(data) < 2 or data[1] is None:
                print(f"    {col_name}: Veri alinamadi")
                continue

            for item in data[1]:
                if item['country']['id'] and item['value'] is not None:
                    all_data.append({
                        'Country_Code': item['countryiso3code'] if item['countryiso3code'] else item['country']['id'],
                        'Country': item['country']['value'],
                        'Year': int(item['date']),
                        'indicator': col_name,
                        'value': item['value'],
                    })

            print(f"    {col_name}: {sum(1 for d in data[1] if d['value'] is not None)} deger")

        except Exception as e:
            print(f"    {col_name}: HATA - {e}")

    if not all_data:
        print("  World Bank API'den veri alinamadi, mevcut veriyi kullaniyoruz")
        return None

    df = pd.DataFrame(all_data)
    # Pivot: her indicator ayri sutun olsun
    pivoted = df.pivot_table(index=['Country_Code', 'Country', 'Year'],
                              columns='indicator', values='value').reset_index()
    pivoted.columns.name = None
    pivoted.to_csv(path, index=False)
    print(f"  Kaydedildi: {path} ({len(pivoted)} satir)")
    return pivoted


def build_extended_dataset():
    """Genisletilmis veri setini olustur."""
    print("\n=== GENISLETILMIS VERI TOPLAMA (1965-2023) ===\n")
    os.makedirs(EXTENDED_DIR, exist_ok=True)

    # 1. OWID Energy (ana kaynak — electricity_demand)
    print("[1/3] OWID Energy verisi...")
    energy = download_owid_energy()

    # Gerekli sutunlari sec
    energy_cols = ['country', 'year', 'iso_code',
                   'electricity_demand', 'electricity_generation',
                   'primary_energy_consumption',
                   'renewables_share_elec', 'renewables_share_energy',
                   'fossil_fuel_consumption', 'nuclear_consumption',
                   'per_capita_electricity', 'energy_per_capita',
                   'population', 'gdp']

    available_cols = [c for c in energy_cols if c in energy.columns]
    energy_slim = energy[available_cols].copy()

    # Sadece ulke verilerini al (bolge/dunya degil)
    energy_slim = energy_slim.dropna(subset=['iso_code'])
    energy_slim = energy_slim[energy_slim['iso_code'].str.len() == 3]
    energy_slim = energy_slim[energy_slim['year'] >= 1965]

    # electricity_demand olmayanlari at
    energy_slim = energy_slim.dropna(subset=['electricity_demand'])
    energy_slim = energy_slim[energy_slim['electricity_demand'] > 0]

    print(f"  OWID Energy: {energy_slim['iso_code'].nunique()} ulke, "
          f"{int(energy_slim['year'].min())}-{int(energy_slim['year'].max())}")

    # 2. World Bank (demografik + ekonomik)
    print("\n[2/3] World Bank verisi...")
    wb = download_world_bank_extended()

    # 3. OWID CO2 (emisyon faktoru icin — model feature degil!)
    print("\n[3/3] OWID CO2 verisi...")
    co2 = download_owid_co2()

    # --- Birlestir ---
    print("\n[Birlestirme]...")

    # Energy'den ana tablo
    merged = energy_slim[['country', 'year', 'iso_code', 'electricity_demand']].copy()

    # OWID'den population ve gdp (energy dataset icinde var)
    if 'population' in energy_slim.columns:
        merged['Population'] = energy_slim['population'].values
    if 'gdp' in energy_slim.columns:
        merged['GDP_total'] = energy_slim['gdp'].values

    # Renewables share (LEAKAGE DEGIL — oran, mutlak degil)
    if 'renewables_share_elec' in energy_slim.columns:
        merged['renewables_share_elec'] = energy_slim['renewables_share_elec'].values

    # World Bank merge (eger basarili indirildiyse)
    if wb is not None and len(wb) > 0:
        wb_cols = [c for c in wb.columns if c not in ('Country',)]
        merged = merged.merge(
            wb.rename(columns={'Country_Code': 'iso_code', 'Year': 'year'})[
                [c for c in ['iso_code', 'year', 'Population_Density', 'Urbanization_Rate',
                             'GDP_Growth', 'GDP_Per_Capita', 'Industry_Value_Added_Pct_GDP']
                 if c in wb.columns or c in ('iso_code', 'year')]
            ],
            on=['iso_code', 'year'], how='left'
        )
    else:
        # Mevcut world_bank_indicators.csv'den al
        print("  World Bank API basarisiz, mevcut CSV kullaniliyor...")
        wb_existing = pd.read_csv(os.path.join(DATA_DIR, 'world_bank_indicators.csv'),
                                   comment='#')
        wb_existing = wb_existing.rename(columns={'Country': 'country_wb', 'Country_Code': 'iso_code', 'Year': 'year'})
        merged = merged.merge(
            wb_existing[['iso_code', 'year', 'Population_Density', 'Urbanization_Rate',
                          'GDP_Growth', 'GDP_Per_Capita']],
            on=['iso_code', 'year'], how='left'
        )

    # GDP per capita: OWID'den de hesaplanabilir
    if 'GDP_Per_Capita' not in merged.columns or merged['GDP_Per_Capita'].isna().sum() > len(merged) * 0.5:
        if 'GDP_total' in merged.columns and 'Population' in merged.columns:
            gdp_pc = merged['GDP_total'] / merged['Population']
            if 'GDP_Per_Capita' in merged.columns:
                merged['GDP_Per_Capita'] = merged['GDP_Per_Capita'].fillna(gdp_pc)
            else:
                merged['GDP_Per_Capita'] = gdp_pc

    # Population: OWID'den de var
    if 'Population' in merged.columns and 'Population_Density' not in merged.columns:
        # Yuzolcumu sabit kabul et, population_density ~ population / country_area
        # Basitlestirme: null birak, sonra impute edilecek
        pass

    # GDP_total artik gereksiz
    if 'GDP_total' in merged.columns:
        merged = merged.drop(columns=['GDP_total'])

    # --- Akilli Imputasyon ---
    print("\n[Imputasyon] Eksik verileri doldurma...")

    merged = merged.sort_values(['iso_code', 'year'])

    # Ulke bazinda forward-fill + backward-fill
    fill_cols = [c for c in merged.columns if c not in ('country', 'iso_code', 'year')]
    merged[fill_cols] = merged.groupby('iso_code')[fill_cols].transform(
        lambda x: x.ffill().bfill()
    )

    # Hala NaN olan: bolgesel medyan veya global medyan
    for col in fill_cols:
        na_count = merged[col].isna().sum()
        if na_count > 0:
            global_median = merged[col].median()
            merged[col] = merged[col].fillna(global_median)
            if na_count > 100:
                print(f"  {col}: {na_count} eksik → global medyan ({global_median:.2f})")

    # Negatif electricity_demand temizle
    merged = merged[merged['electricity_demand'] > 0]

    # --- Rapor ---
    print(f"\n=== GENISLETILMIS VERI SETI ===")
    print(f"  Satir: {len(merged)}")
    print(f"  Ulke: {merged['iso_code'].nunique()}")
    print(f"  Yil: {int(merged['year'].min())}-{int(merged['year'].max())}")
    print(f"  Sutunlar: {list(merged.columns)}")
    print(f"\n  Eksik deger:")
    for col in merged.columns:
        na = merged[col].isna().sum()
        if na > 0:
            print(f"    {col}: {na}")
    if merged.isna().sum().sum() == 0:
        print(f"    Hic eksik deger yok!")

    # Turkiye kontrolu
    tr = merged[merged['iso_code'] == 'TUR']
    print(f"\n  Turkiye: {len(tr)} yil ({int(tr['year'].min())}-{int(tr['year'].max())})")
    print(f"  Turkiye electricity_demand: {tr['electricity_demand'].min():.1f} - {tr['electricity_demand'].max():.1f} TWh")

    # Kaydet
    out_path = os.path.join(EXTENDED_DIR, 'merged_extended.csv')
    merged.to_csv(out_path, index=False)
    print(f"\n  Kaydedildi: {out_path}")

    return merged


if __name__ == '__main__':
    build_extended_dataset()
