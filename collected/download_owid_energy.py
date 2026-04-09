"""
OWID Energy Data - Veri Toplama Scripti
Kaynak: Our World in Data - Energy Dataset
URL: https://github.com/owid/energy-data
"""

import pandas as pd
import os

def main():
    # 1. CSV'yi pandas ile indir
    url = 'https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv'
    print('Veri indiriliyor...')
    df = pd.read_csv(url)
    print(f'Ham veri: {df.shape[0]} satir, {df.shape[1]} sutun')

    # Mevcut sutunlari kontrol et
    needed = [
        'country', 'year', 'iso_code',
        'primary_energy_consumption',
        'electricity_demand', 'electricity_generation',
        'renewables_share_energy', 'renewables_share_elec',
        'fossil_fuel_consumption',
        'nuclear_consumption',
        'per_capita_electricity'
    ]
    for col in needed:
        if col in df.columns:
            print(f'  [OK] {col}')
        else:
            print(f'  [YOK] {col}')

    # 2. Sadece gercek ulkeleri al (iso_code NaN olmayanlari)
    df = df[df['iso_code'].notna()]
    print(f'Gercek ulkeler filtresi sonrasi: {df.shape[0]} satir')

    # 3. Yil filtresi: 2000-2023
    df = df[(df['year'] >= 2000) & (df['year'] <= 2023)]
    print(f'Yil filtresi (2000-2023) sonrasi: {df.shape[0]} satir')

    # 4. Ilgili sutunlari sec (mevcut olanlari)
    select_cols = [c for c in needed if c in df.columns]
    df = df[select_cols].copy()
    print(f'Secilen sutunlar: {select_cols}')

    # 5. Cikti dizinini olustur (varsa atla)
    out_path = 'C:/Users/LENOVO/Desktop/Tez Yeni/data/collected/owid_energy_data.csv'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 6. Metadata satiri + CSV olarak kaydet (UTF-8)
    with open(out_path, 'w', encoding='utf-8', newline='') as f:
        f.write('# Kaynak: Our World in Data - Energy Dataset - https://github.com/owid/energy-data\n')
        df.to_csv(f, index=False)

    print(f'\nKayit tamamlandi: {out_path}')
    print(f'Sonuc: {df.shape[0]} satir, {df.shape[1]} sutun')
    print(f'Ulke sayisi: {df["country"].nunique()}')
    print(f'Yil araligi: {df["year"].min()} - {df["year"].max()}')
    print(f'Dosya boyutu: {os.path.getsize(out_path) / 1024:.1f} KB')

if __name__ == '__main__':
    main()
