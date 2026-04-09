"""
Veri Hazırlama Scripti — Tüm Ülkeler, Ülkeye Özgü Split
---------------------------------------------------------
Her ülke kendi veri aralığından öğrenir. Bazı ülkede 1990'dan veri
varsa oradan başlar, bazısında 2005'ten başlıyorsa oradan başlar.

Birleştirilen datasetler:
  1. merged_clean.csv         → temel dataset (1990–2025, 213 ülke)
  2. owid_energy_data.csv     → enerji karışımı, fosil/nükleer tüketim
  3. owid_co2_emissions.csv   → kömür/gaz CO2 (enerji kaynağı göstergesi)
  4. industry_energy_indicators.csv → elektrik kayıp oranı, enerji yoğunluğu
  5. climate_temperature_data.csv   → GHG kaynaklı sıcaklık değişimi

Leakage riski taşıyan sütunlar (hedef değişkenle türetilmiş):
  DIŞLANDI: per_capita_electricity, electricity_generation,
             electricity_demand (owid'den), renewables_share_elec (tekrar),
             Industry_Value_Added_Pct_GDP (tekrar), primary_energy_consumption,
             Energy_Use_Per_Capita (elektrik içeriyor)

Split (ülkeye özgü):
  Train : her ülkenin veri başlangıcından 2020'ye kadar
  Val   : 2021–2023 (ülkede bu yıllar varsa)
  Test  : 2024–2025 (ülkede bu yıllar varsa)
  Forecast hedefi: 2026–2030
"""

import os
import pandas as pd
import numpy as np

# ─── Yollar ───────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))   # veri/islenmis/
PIPELINE_DIR  = os.path.dirname(os.path.dirname(SCRIPT_DIR)) # pipeline/
HAM_DIR       = os.path.join(PIPELINE_DIR, 'veri', 'ham')
CIKTI_DIR     = SCRIPT_DIR

HAM_DOSYALAR = {
    'merged_clean'    : os.path.join(HAM_DIR, 'merged_clean.csv'),
    'owid_energy'     : os.path.join(HAM_DIR, 'owid_energy_data.csv'),
    'owid_co2'        : os.path.join(HAM_DIR, 'owid_co2_emissions.csv'),
    'industry_energy' : os.path.join(HAM_DIR, 'industry_energy_indicators.csv'),
    'climate_temp'    : os.path.join(HAM_DIR, 'climate_temperature_data.csv'),
}

# ─── Eklenecek Feature'lar (leakage olmayan) ──────────────────────────────────
# Her sütun hangi datasetten geldiği ve neden dahil edildiği:
TRAIN_SON  = 2020
VAL_ILK    = 2021; VAL_SON  = 2023
TEST_ILK   = 2024; TEST_SON = 2025

EK_FEATURES = {
    # Enerji kaynağı karışımı — elektriği hangi kaynak üretiyor?
    'fossil_fuel_consumption' : 'owid_energy',   # fosil yakıt tüketimi (TWh)
    'nuclear_consumption'     : 'owid_energy',   # nükleer tüketim (TWh)
    # Fosil yakıt bileşimi — kömür mü, gaz mı ağırlıklı?
    'coal_co2'                : 'owid_co2',      # kömür kaynaklı CO2 (ton)
    'gas_co2'                 : 'owid_co2',      # gaz kaynaklı CO2 (ton)
    'co2_per_capita'          : 'owid_co2',      # kişi başına emisyon
    # Şebeke verimliliği
    'Electricity_Loss_Pct'    : 'industry_energy', # iletim kayıp oranı (%)
    # İklim değişimi
    'temperature_change_from_ghg': 'climate_temp', # GHG kaynaklı sıcaklık değişimi
}

# ─── Veri Oku ─────────────────────────────────────────────────────────────────
print("Veriler okunuyor...")

base_df = pd.read_csv(HAM_DOSYALAR['merged_clean'])
base_df.columns = base_df.columns.str.strip()
print(f"  merged_clean   : {base_df.shape}")

owid_energy = pd.read_csv(HAM_DOSYALAR['owid_energy'], comment='#')
owid_co2    = pd.read_csv(HAM_DOSYALAR['owid_co2'],    comment='#')
ind_energy  = pd.read_csv(HAM_DOSYALAR['industry_energy'], comment='#')
climate     = pd.read_csv(HAM_DOSYALAR['climate_temp'], comment='#')

# Sütun isimlerini normalize et
ind_energy  = ind_energy.rename(columns={'Country':'country','Country_Code':'iso_code','Year':'year'})
owid_energy = owid_energy.rename(columns={'Country':'country','Year':'year'})
owid_co2    = owid_co2.rename(columns={'Country':'country','Year':'year'})
climate     = climate.rename(columns={'Country':'country','Year':'year'})

# ─── Birleştir ────────────────────────────────────────────────────────────────
print("Datasetler birleştiriliyor...")

merge_key = ['country', 'year']

# owid_energy'den sadece istediğimiz sütunlar
df = base_df.merge(
    owid_energy[merge_key + ['fossil_fuel_consumption', 'nuclear_consumption']],
    on=merge_key, how='left'
)

# owid_co2'den
df = df.merge(
    owid_co2[merge_key + ['coal_co2', 'gas_co2', 'co2_per_capita']],
    on=merge_key, how='left'
)

# industry_energy'den
df = df.merge(
    ind_energy[merge_key + ['Electricity_Loss_Pct']],
    on=merge_key, how='left'
)

# climate'dan
df = df.merge(
    climate[merge_key + ['temperature_change_from_ghg']],
    on=merge_key, how='left'
)

print(f"  Birleşik dataset: {df.shape}")
print(f"  Ülke sayısı     : {df['country'].nunique()}")
print(f"  Yıl aralığı     : {df['year'].min()} - {df['year'].max()}")

# ─── Tüm Feature Listesi ──────────────────────────────────────────────────────
TEMEL_FEATURES = [
    'electricity_demand',
    'Population',
    'GDP_Per_Capita',
    'GDP_Growth',
    'renewables_share_elec',
    'Urbanization_Rate',
    'Industry_Value_Added_Pct_GDP',
    'Population_Density',
]

EK_FEATURE_LISTESI = list(EK_FEATURES.keys())
TUM_FEATURES = TEMEL_FEATURES + EK_FEATURE_LISTESI

KOLONLAR = ['country', 'year', 'iso_code'] + TUM_FEATURES

# Sadece mevcut kolonları al
mevcut_kolonlar = [c for c in KOLONLAR if c in df.columns]
df = df[mevcut_kolonlar]

# ─── Temizlik ─────────────────────────────────────────────────────────────────
# Strateji:
#   Train/Val : NaN olan satırı tamamen at.
#               Her ülke tüm feature'ların dolu olduğu yıldan başlar.
#   Test      : 2024-2025'te ek feature'lar (owid vb.) henüz yayınlanmadı.
#               Ek feature'lar için her ülkenin son bilinen değeri (2023)
#               forward-fill ile taşınır. Temel feature'larda NaN varsa o satır düşer.

ek_feature_cols = [c for c in EK_FEATURE_LISTESI if c in df.columns]
temel_feature_cols = [c for c in TEMEL_FEATURES if c in df.columns]

# Train + Val: tüm feature'larda NaN olan satırı at
train_val_mask = df['year'] <= VAL_SON
df_trainval = df[train_val_mask].copy()
onceki = len(df_trainval)
df_trainval = df_trainval.dropna(
    subset=[c for c in TUM_FEATURES if c in df.columns]
).reset_index(drop=True)
print(f"  Train/Val NaN temizligi: {onceki} -> {len(df_trainval)} satir")

# Test: ek feature'ları ülke bazında forward-fill et (son bilinen 2023 değeri)
df_test = df[df['year'] >= TEST_ILK].copy()
df_test = df_test.sort_values(['country', 'year'])
df_test[ek_feature_cols] = df_test.groupby('country')[ek_feature_cols].ffill()
# Temel feature'larda NaN kalan satırı at
df_test = df_test.dropna(subset=temel_feature_cols).reset_index(drop=True)
print(f"  Test satir sayisi       : {len(df_test)} (ek featurelar ffill)")

# Birleştir
df = pd.concat([df_trainval, df_test], ignore_index=True)

# ─── Ülkeye Özgü Split ────────────────────────────────────────────────────────
# Her ülke kendi veri başlangıcından 2020'ye kadar train'e girer.
# 1990-1999 arasındaki ek feature sütunları NaN olabilir — bu normal.
# XGBoost NaN ile çalışabilir. LSTM/RF için imputation Stage 2'de yapılır.

train_df = df[df['year'] <= TRAIN_SON].copy()
val_df   = df[(df['year'] >= VAL_ILK) & (df['year'] <= VAL_SON)].copy()
test_df  = df[(df['year'] >= TEST_ILK) & (df['year'] <= TEST_SON)].copy()

# ─── İstatistik Özeti ─────────────────────────────────────────────────────────
print("\nÜlkeye özgü veri istatistikleri (Train):")
ulke_train = train_df.groupby('country')['year'].agg(['min','max','count'])
ulke_train.columns = ['ilk_yil','son_yil','satir_sayisi']
print(f"  Ortalama eğitim süresi: {ulke_train['satir_sayisi'].mean():.1f} yıl")
print(f"  Min eğitim süresi     : {ulke_train['satir_sayisi'].min()} yıl "
      f"({ulke_train[ulke_train['satir_sayisi']==ulke_train['satir_sayisi'].min()].index[0]})")
print(f"  Max eğitim süresi     : {ulke_train['satir_sayisi'].max()} yıl "
      f"({ulke_train[ulke_train['satir_sayisi']==ulke_train['satir_sayisi'].max()].index[0]})")

print(f"\nEk feature doluluk oranları (Train):")
for feat in EK_FEATURE_LISTESI:
    if feat in train_df.columns:
        doluluk = train_df[feat].notna().sum() / len(train_df) * 100
        print(f"  {feat:<35}: %{doluluk:.1f} dolu")

print(f"\nSplit özeti:")
print(f"  Train : {len(train_df):>6} satır | {train_df['country'].nunique()} ülke | "
      f"{train_df['year'].min()}-{train_df['year'].max()}")
print(f"  Val   : {len(val_df):>6} satır | {val_df['country'].nunique()} ülke | "
      f"{val_df['year'].min()}-{val_df['year'].max()}")
print(f"  Test  : {len(test_df):>6} satır | {test_df['country'].nunique()} ülke | "
      f"{test_df['year'].min()}-{test_df['year'].max()}")

# ─── Kaydet ───────────────────────────────────────────────────────────────────
train_df.to_csv(os.path.join(CIKTI_DIR, 'train.csv'), index=False)
val_df.to_csv(  os.path.join(CIKTI_DIR, 'val.csv'),   index=False)
test_df.to_csv( os.path.join(CIKTI_DIR, 'test.csv'),  index=False)

# Feature listesini de kaydet (Stage 1 ve 2 buradan okur)
feature_config = {
    'hedef'          : 'electricity_demand',
    'temel_features' : [f for f in TEMEL_FEATURES if f != 'electricity_demand'],
    'ek_features'    : [f for f in EK_FEATURE_LISTESI if f in df.columns],
    'tum_features'   : [f for f in TUM_FEATURES
                        if f != 'electricity_demand' and f in df.columns],
}
pd.DataFrame(dict([(k, pd.Series(v)) for k,v in feature_config.items()])).to_csv(
    os.path.join(CIKTI_DIR, 'feature_listesi.csv'), index=False
)

print(f"\nKaydedilen dosyalar:")
print(f"  islenmis/train.csv")
print(f"  islenmis/val.csv")
print(f"  islenmis/test.csv")
print(f"  islenmis/feature_listesi.csv")
print(f"\nToplam feature sayisi: {len(feature_config['tum_features'])}")
print(f"  {feature_config['tum_features']}")
