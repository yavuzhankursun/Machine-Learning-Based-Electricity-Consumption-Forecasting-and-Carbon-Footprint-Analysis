"""
Veri Yukleme: electricity_per_capita-Temiz Veri.xlsx

Dataset: 150 ulke, 1990-2022, kisi basina elektrik tuketimi (kWh/kisi)
- 2023-2024 forward-fill oldugu icin KULLANILMIYOR
- World Bank bolge/aggregate kodlari filtreleniyor
- Yalnizca ISO country kodlari kalir
"""
import pandas as pd
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'electricity_per_capita-Temiz Veri.xlsx')

# Kullanilacak gercek veri yillari (2023-2024 forward-fill)
DATA_YEARS = list(range(1990, 2023))   # 1990-2022

# Tahmin edilecek gelecek yillar
FORECAST_YEARS = list(range(2023, 2029))   # 2023-2028 (6 yil)

# World Bank bolge/aggregate kodlari - bunlar ulke degil, cikarilir
WB_AGGREGATES = {
    'AFE','AFW','ARB','CEB','CSS','EAP','EAR','EAS','ECA','ECE','ECS',
    'EMU','EUU','FCS','HIC','HPC','IBD','IBT','IDA','IDB','IDX','LAC',
    'LCN','LDC','LIC','LMC','LMY','LTE','MEA','MIC','MNA','NAC','OED',
    'OSS','PRE','PSS','PST','SAS','SSA','SSF','SST','TEA','TEC','TLA',
    'TMN','TSA','TSS','UMC','WLD',
}


def load_data() -> pd.DataFrame:
    """
    Excel'i yukle, wide → long donusum yap.

    Returns
    -------
    df : DataFrame
        Sutunlar: country, iso_code, year, elec_per_capita
    """
    df_wide = pd.read_excel(DATA_PATH)

    # Aggregate bolgeler cikar
    df_wide = df_wide[~df_wide['Country Code'].isin(WB_AGGREGATES)].copy()

    # Wide → Long
    year_cols = [str(y) for y in DATA_YEARS]
    df = df_wide.melt(
        id_vars=['Country Name', 'Country Code'],
        value_vars=year_cols,
        var_name='year',
        value_name='elec_per_capita',
    )
    df['year'] = df['year'].astype(int)
    df = df.rename(columns={'Country Name': 'country', 'Country Code': 'iso_code'})
    df = df.sort_values(['iso_code', 'year']).reset_index(drop=True)

    # Sifir duzeltme (Namibia 1990 gibi veri giris hatasi)
    df.loc[df['elec_per_capita'] <= 0, 'elec_per_capita'] = np.nan
    df['elec_per_capita'] = (
        df.groupby('iso_code')['elec_per_capita']
        .transform(lambda x: x.bfill().ffill())
    )

    n_countries = df['iso_code'].nunique()
    print(f"Veri yuklendi: {n_countries} ulke, {df['year'].min()}-{df['year'].max()}, "
          f"{len(df)} satir")
    return df
