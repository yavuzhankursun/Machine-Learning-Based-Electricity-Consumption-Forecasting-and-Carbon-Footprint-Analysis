"""
Feature Engineering - DATA LEAKAGE YOK

Her (ulke, yil=t) satiri icin YALNIZCA t-1 ve oncesi kullanilir:

  lag_1          = elec(t-1)                         -> bir yil oncesi gercek deger
  lag_2          = elec(t-2)                         -> iki yil oncesi
  lag_3          = elec(t-3)                         -> uc yil oncesi
  rolling_mean_3 = mean(t-1, t-2, t-3)               -> son 3 yilin ortalamasi
  rolling_std_3  = std(t-1, t-2, t-3)                -> volatilite
  yoy_growth     = (elec(t-1) - elec(t-2)) / elec(t-2)  -> ONCEKI yilin buyume orani

  NEDEN shift(1) yapilir yoy_growth icin?
  Yanlis: pct_change()          -> (t - t-1) / t-1   HEDEFI (t) kullanir! LEAKAGE!
  Dogru:  pct_change().shift(1) -> (t-1 - t-2) / t-2  Sadece gecmisi kullanir. TEMIZ.

  year           = t                                  -> dogrusal trend
  country_encoded = LabelEncoder(iso_code)            -> ulke kimligi

Temporal Split (veri sizintisi onleme):
  TRAIN  : 1993 - 2015   (23 yil x 150 ulke = 3450 ornek)
  VAL    : 2016 - 2018   (3 yil  x 150 ulke =  450 ornek) -> early stopping + model secimi
  TEST   : 2019 - 2022   (4 yil  x 150 ulke =  600 ornek) -> final degerlendirme
  FORECAST: 2023 - 2028  (tahmin edilecek, gercek veri yok)

  Kural: TEST verisine egitim sirasinda DOKUNULMAZ.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

TRAIN_END  = 2015
VAL_START  = 2016
VAL_END    = 2018
TEST_START = 2019
TEST_END   = 2022


def create_features(df: pd.DataFrame):
    """
    Lag ve istatistik feature'lari olustur.
    Tum feature'lar sadece gecmis bilgi kullanir (no leakage).
    """
    df = df.sort_values(['iso_code', 'year']).copy()
    grp = df.groupby('iso_code')['elec_per_capita']

    # Lag features: t-1, t-2, t-3
    df['lag_1'] = grp.shift(1)
    df['lag_2'] = grp.shift(2)
    df['lag_3'] = grp.shift(3)

    # Rolling mean/std: shift(1) ile once gecmise kaydir -> rolling al
    # Boylece rolling_mean_3 = mean(t-1, t-2, t-3), t dahil degil
    df['rolling_mean_3'] = grp.transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).mean()
    )
    df['rolling_std_3'] = grp.transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).std()
    ).fillna(0)

    # YoY growth: shift(1) ile ONCEKI yilin buyume orani
    # pct_change().shift(1) -> (t-1 - t-2) / t-2  [t kullanilmaz!]
    df['yoy_growth'] = grp.transform(
        lambda x: x.pct_change().shift(1)
    ).fillna(0).replace([np.inf, -np.inf], 0)

    # Country encoding
    le = LabelEncoder()
    df['country_encoded'] = le.fit_transform(df['iso_code'])

    # lag_1/lag_2/lag_3 NaN olan satirlari at (ilk uc yil)
    df = df.dropna(subset=['lag_1', 'lag_2', 'lag_3']).reset_index(drop=True)

    return df, le


def get_feature_cols():
    """
    Feature listesi.
    Siralama onemli: ilk 3 LSTM icin sekans (eskiden yeniye: lag_3, lag_2, lag_1).
    """
    return [
        'lag_3', 'lag_2', 'lag_1',        # sekans (t-3, t-2, t-1)
        'rolling_mean_3', 'rolling_std_3', # istatistiksel ozet
        'yoy_growth',                       # momentum (onceki yil)
        'year',                             # dogrusal trend
        'country_encoded',                  # ulke kimligi
    ]


def temporal_split(df: pd.DataFrame):
    """Zamansal veri bolme. Test verisine egitimde dokunulmaz."""
    train = df[df['year'] <= TRAIN_END].copy()
    val   = df[(df['year'] >= VAL_START)  & (df['year'] <= VAL_END)].copy()
    test  = df[(df['year'] >= TEST_START) & (df['year'] <= TEST_END)].copy()

    print(f"Temporal Split:")
    print(f"  TRAIN  : {train['year'].min()}-{train['year'].max()} "
          f"-> {len(train)} satir ({train['iso_code'].nunique()} ulke)")
    print(f"  VAL    : {val['year'].min()}-{val['year'].max()} "
          f"-> {len(val)} satir")
    print(f"  TEST   : {test['year'].min()}-{test['year'].max()} "
          f"-> {len(test)} satir")
    return train, val, test


def prepare_data(df_raw: pd.DataFrame) -> dict:
    """Tam pipeline: feature olustur, split yap, numpy dizileri hazirla."""
    df, le = create_features(df_raw)
    feat   = get_feature_cols()
    target = 'elec_per_capita'

    train, val, test = temporal_split(df)

    def to_arrays(split_df):
        X = split_df[feat].values.astype(float)
        y = split_df[target].values.astype(float)
        return X, y

    X_train, y_train = to_arrays(train)
    X_val,   y_val   = to_arrays(val)
    X_test,  y_test  = to_arrays(test)

    print(f"\n  Feature sayisi : {len(feat)}")
    print(f"  Feature'lar    : {feat}")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val'  : X_val,   'y_val'  : y_val,
        'X_test' : X_test,  'y_test' : y_test,
        'train_df': train, 'val_df': val, 'test_df': test,
        'full_df' : df,
        'feature_cols': feat,
        'target_col'  : target,
        'label_encoder': le,
    }
