"""
Sanayi Uretim Endeksi ve Enerji Gostergeleri Toplama Scripti
Kaynak: World Bank WDI API
Yil Araligi: 2000-2023
"""

import requests
import pandas as pd
import time
import os
import sys

OUTPUT_PATH = r"C:/Users/LENOVO/Desktop/Tez Yeni/data/collected/industry_energy_indicators.csv"

# World Bank gostergeleri
INDICATORS = {
    "NV.IND.TOTL.ZS": "Industry_Value_Added_Pct_GDP",
    "EG.USE.ELEC.KH.PC": "Electric_Power_Consumption_kWh_Per_Capita",
    "EG.ELC.LOSS.ZS": "Electricity_Loss_Pct",
    "EG.FEC.RNEW.ZS": "Renewable_Energy_Consumption_Pct",
    "EG.USE.PCAP.KG.OE": "Energy_Use_Per_Capita",
}

BASE_URL = "http://api.worldbank.org/v2/country/all/indicator/{indicator}?format=json&per_page=20000&date=2000:2023"


def fetch_indicator(indicator_code, column_name):
    """World Bank API'den tek bir gosterge cek."""
    url = BASE_URL.format(indicator=indicator_code)
    print(f"  Cekiliyor: {indicator_code} ({column_name}) ...")

    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"    HATA: {indicator_code} cekilemedi - {e}")
        return pd.DataFrame()

    if len(data) < 2 or data[1] is None:
        print(f"    UYARI: {indicator_code} icin veri bulunamadi.")
        return pd.DataFrame()

    records = []
    for entry in data[1]:
        country_name = entry.get("country", {}).get("value", "")
        country_code = entry.get("countryiso3code", "") or entry.get("country", {}).get("id", "")
        year = entry.get("date", "")
        value = entry.get("value", None)

        records.append({
            "Country": country_name,
            "Country_Code": country_code,
            "Year": int(year) if year else None,
            column_name: value,
        })

    df = pd.DataFrame(records)
    non_null = df[column_name].notna().sum()
    print(f"    Basarili: {len(df)} satir, {non_null} dolu deger")
    return df


def fetch_eurostat_energy():
    """Eurostat'tan enerji uretim verisi cekmeye calis."""
    print("\n  Eurostat'tan enerji verisi deneniyor...")
    eurostat_url = (
        "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
        "nrg_ind_id?format=JSON&lang=en&freq=A&nrg_bal=PPRD&siec=TOTAL&unit=KTOE"
        "&time=2000&time=2005&time=2010&time=2015&time=2020&time=2023"
    )
    try:
        resp = requests.get(eurostat_url, timeout=30)
        if resp.status_code == 200:
            print("    Eurostat verisi alindi (ancak ana veri World Bank'tir).")
            return True
        else:
            print(f"    Eurostat erisim basarisiz (HTTP {resp.status_code}). World Bank verisi yeterli.")
            return False
    except Exception as e:
        print(f"    Eurostat erisim basarisiz: {e}. World Bank verisi yeterli.")
        return False


def main():
    print("=" * 65)
    print("SANAYI & ENERJI GOSTERGELERI TOPLAMA")
    print("Kaynak: World Bank WDI API | Yil: 2000-2023")
    print("=" * 65)

    all_dfs = []
    sources = ["World Bank WDI"]

    for indicator_code, column_name in INDICATORS.items():
        df = fetch_indicator(indicator_code, column_name)
        if not df.empty:
            all_dfs.append(df)
        time.sleep(1)  # API rate-limit'e saygi

    if not all_dfs:
        print("\nHICBIR VERI CEKILEMEDI! Cikiliyor.")
        sys.exit(1)

    # Ilk DataFrame'i temel al, digerlerini merge et
    print("\n  DataFrame'ler birlestiriliyor...")
    merged = all_dfs[0]
    for df in all_dfs[1:]:
        merged = pd.merge(
            merged, df,
            on=["Country", "Country_Code", "Year"],
            how="outer"
        )

    # Temizlik
    merged = merged.dropna(subset=["Year"])
    merged["Year"] = merged["Year"].astype(int)
    merged = merged.sort_values(["Country", "Year"]).reset_index(drop=True)

    # Aggrege ulkeleri/bolgeleri cikar (sadece gercek ulkeleri birak)
    # World Bank'in aggregate kodlari genelde 2 harfli olmayan veya ozel kodlardir
    # Ama biz tum verileri tutalim, kullanici filtreleyebilir

    # Eurostat denemesi
    eurostat_ok = fetch_eurostat_energy()
    if eurostat_ok:
        sources.append("Eurostat (referans)")

    # Sutun sirasi
    column_order = [
        "Country", "Country_Code", "Year",
        "Industry_Value_Added_Pct_GDP",
        "Electric_Power_Consumption_kWh_Per_Capita",
        "Electricity_Loss_Pct",
        "Renewable_Energy_Consumption_Pct",
        "Energy_Use_Per_Capita",
    ]
    # Sadece mevcut sutunlari al
    existing_cols = [c for c in column_order if c in merged.columns]
    merged = merged[existing_cols]

    # Metadata satiri ile CSV kaydet
    source_str = " + ".join(sources)
    metadata_line = f"# Kaynak: {source_str} - Sanayi ve Enerji Gostergeleri\n"

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(metadata_line)
        merged.to_csv(f, index=False)

    print("\n" + "=" * 65)
    print("SONUC OZETI")
    print("=" * 65)
    print(f"  Toplam satir    : {len(merged)}")
    print(f"  Benzersiz ulke  : {merged['Country'].nunique()}")
    print(f"  Yil araligi     : {merged['Year'].min()} - {merged['Year'].max()}")
    print(f"  Sutunlar        : {list(merged.columns)}")
    print(f"  Kaynaklar       : {source_str}")
    print(f"  Cikti dosyasi   : {OUTPUT_PATH}")

    # Her sutundaki dolu deger sayisi
    print("\n  Sutun bazinda dolu deger sayilari:")
    for col in merged.columns:
        if col not in ["Country", "Country_Code", "Year"]:
            count = merged[col].notna().sum()
            print(f"    {col}: {count}")

    # Ornek satirlar
    print("\n  Ornek satirlar (ilk 5):")
    print(merged.head().to_string(index=False))

    print(f"\n  CSV basariyla kaydedildi: {OUTPUT_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()
