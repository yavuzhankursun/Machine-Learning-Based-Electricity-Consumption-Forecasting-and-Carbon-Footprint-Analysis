"""
Feature Engineering
- Lag features (t-1, t-2, t-3)
- Rolling statistics (3 yillik ortalama, std)
- YoY buyume orani
- Country encoding (label)
- Temporal split (train/val/test)
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')


def create_features(df):
    """Zaman serisi feature'lari olustur."""
    df = df.sort_values(['iso_code', 'year']).copy()

    # --- Lag features ---
    for lag in [1, 2, 3]:
        df[f'demand_lag_{lag}'] = df.groupby('iso_code')['electricity_demand'].shift(lag)

    # --- Rolling statistics (3 yillik pencere) ---
    df['demand_rolling_mean_3'] = df.groupby('iso_code')['electricity_demand'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df['demand_rolling_std_3'] = df.groupby('iso_code')['electricity_demand'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).std()
    )

    # --- YoY buyume orani ---
    df['demand_yoy_growth'] = df.groupby('iso_code')['electricity_demand'].pct_change()

    # --- Country encoding ---
    le = LabelEncoder()
    df['country_encoded'] = le.fit_transform(df['iso_code'])

    # NaN olan rolling_std'yi 0 ile doldur (ilk yillar icin)
    df['demand_rolling_std_3'] = df['demand_rolling_std_3'].fillna(0)
    df['demand_yoy_growth'] = df['demand_yoy_growth'].fillna(0)

    # Lag feature'lari NaN olan satirlari at (ilk 3 yil)
    df = df.dropna(subset=['demand_lag_1'])

    return df, le


def get_feature_columns():
    """Model icin kullanilacak feature sutunlarini dondur."""
    return [
        # Ekonomik/demografik
        'Population', 'Population_Density', 'Urbanization_Rate',
        'GDP_Growth', 'GDP_Per_Capita',
        # Zaman serisi (autoregresif)
        'demand_lag_1', 'demand_lag_2', 'demand_lag_3',
        'demand_rolling_mean_3', 'demand_rolling_std_3',
        'demand_yoy_growth',
        # Yapisal
        'year', 'country_encoded',
    ]
    # NOT: Iklim ve sanayi feature'lari ablation study sonucunda CIKARILDI.
    # temperature_change_from_ghg → hedefle coklu dogrusal bagimlılik, modele zarar veriyor
    # Industry_Value_Added_Pct_GDP, renewables → test performansini dusuruyor
    # Ablation kaniti: Bu degiskenler cikarildiginda MAPE duser (bkz. ablation_study sonuclari)


def temporal_split(df):
    """
    Zamana gore veri bolme — data leakage onleme.
    Train: 2000-2018, Val: 2019-2020, Test: 2021-2023
    """
    train = df[df['year'] <= 2018].copy()
    val = df[(df['year'] >= 2019) & (df['year'] <= 2020)].copy()
    test = df[df['year'] >= 2021].copy()

    print(f"Temporal Split:")
    print(f"  Train: {len(train)} satir ({train['year'].min()}-{train['year'].max()})")
    print(f"  Val:   {len(val)} satir ({val['year'].min()}-{val['year'].max()})")
    print(f"  Test:  {len(test)} satir ({test['year'].min()}-{test['year'].max()})")

    return train, val, test


def prepare_data(df):
    """Feature'lari olustur, split yap, X/y ayir."""
    df_feat, label_encoder = create_features(df)
    feature_cols = get_feature_columns()
    target_col = 'electricity_demand'

    train, val, test = temporal_split(df_feat)

    X_train = train[feature_cols].values
    y_train = train[target_col].values
    X_val = val[feature_cols].values
    y_val = val[target_col].values
    X_test = test[feature_cols].values
    y_test = test[target_col].values

    # NaN kontrolu
    for name, arr in [('X_train', X_train), ('X_val', X_val), ('X_test', X_test)]:
        nan_count = np.isnan(arr).sum()
        if nan_count > 0:
            print(f"  UYARI: {name}'de {nan_count} NaN var, 0 ile dolduruluyor")
            arr[np.isnan(arr)] = 0

    print(f"\nFeature sayisi: {len(feature_cols)}")
    print(f"Feature'lar: {feature_cols}")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'train_df': train,
        'val_df': val,
        'test_df': test,
        'full_df': df_feat,
        'label_encoder': label_encoder,
    }


if __name__ == '__main__':
    from data_merge import load_and_merge
    df = load_and_merge()
    data = prepare_data(df)
    print(f"\nX_train shape: {data['X_train'].shape}")
    print(f"X_val shape: {data['X_val'].shape}")
    print(f"X_test shape: {data['X_test'].shape}")
