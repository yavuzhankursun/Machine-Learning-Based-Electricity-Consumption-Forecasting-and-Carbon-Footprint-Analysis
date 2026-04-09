"""
Recursive Multi-Step Forecasting (2023-2028)

Model 1 yil ilerisi icin egitildi: f(lag_1, lag_2, lag_3, ...) = elec(t)
Gercek veri olmayan gelecek yillari tahmin etmek icin RECURSIVE kullanilir:

  2023 tahmini: lag_1=gercek(2022), lag_2=gercek(2021), lag_3=gercek(2020)
  2024 tahmini: lag_1=tahmin(2023), lag_2=gercek(2022),  lag_3=gercek(2021)
  2025 tahmini: lag_1=tahmin(2024), lag_2=tahmin(2023),  lag_3=gercek(2022)
  ...

HATA BIRIKIMI (error accumulation):
  Her adimda bir onceki tahminin hatasi birikir.
  2028 tahmini 2023 tahmininden cok daha belirsizdir.
  Grafikte belirsizlik bandi ile gosterilir.
"""
import numpy as np
import pandas as pd

FORECAST_YEARS = list(range(2023, 2029))


def _build_feature_vec(lag_1, lag_2, lag_3, year, country_encoded):
    """feature_engineering.get_feature_cols() ile ayni sirada."""
    rolling_mean_3 = np.mean([lag_1, lag_2, lag_3])
    rolling_std_3  = np.std([lag_1, lag_2, lag_3])
    yoy_growth = (lag_1 - lag_2) / lag_2 if lag_2 > 0 else 0.0
    return np.array([[lag_3, lag_2, lag_1,
                      rolling_mean_3, rolling_std_3,
                      yoy_growth, year, country_encoded]], dtype=float)


def _predict_one(model, X, model_type):
    if model_type == 'lstm':
        from model_lstm import split_inputs
        Xs, Xa = split_inputs(X)
        return float(model.predict([Xs, Xa], verbose=0).flatten()[0])
    return float(model.predict(X)[0])


def recursive_forecast(model, country_df: pd.DataFrame, country_encoded: int,
                       forecast_years=None, model_type='sklearn'):
    """
    Tek ulke icin recursive tahmin.

    Returns list of {'year': int, 'predicted': float}
    """
    if forecast_years is None:
        forecast_years = FORECAST_YEARS

    history = list(country_df['elec_per_capita'].values)
    results = []

    for year in forecast_years:
        lag_1 = history[-1]
        lag_2 = history[-2] if len(history) >= 2 else lag_1
        lag_3 = history[-3] if len(history) >= 3 else lag_2

        X = _build_feature_vec(lag_1, lag_2, lag_3, year, country_encoded)
        pred = _predict_one(model, X, model_type)
        pred = max(pred, 0.0)

        history.append(pred)
        results.append({'year': year, 'predicted': round(pred, 2)})

    return results


def forecast_all_countries(data: dict, result: dict,
                           forecast_years=None) -> pd.DataFrame:
    """
    Tum ulkeler icin 2023-2028 tahmini uret.

    Returns
    -------
    DataFrame: iso_code, country, year, predicted_elec_per_capita
    """
    if forecast_years is None:
        forecast_years = FORECAST_YEARS

    model      = result['model']
    model_type = 'lstm' if result['name'] == 'LSTM' else 'sklearn'
    df         = data['full_df']
    le         = data['label_encoder']

    rows = []
    for iso in df['iso_code'].unique():
        df_c            = df[df['iso_code'] == iso].sort_values('year')
        country_name    = df_c['country'].iloc[0]
        country_encoded = int(le.transform([iso])[0])

        preds = recursive_forecast(model, df_c, country_encoded,
                                   forecast_years, model_type)
        for p in preds:
            rows.append({
                'iso_code'                  : iso,
                'country'                   : country_name,
                'year'                      : p['year'],
                'predicted_elec_per_capita' : p['predicted'],
            })

    return pd.DataFrame(rows)


def build_country_history(data: dict, result: dict,
                          forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Her ulke icin tam zaman serisi olustur (train/val/test/forecast).
    Kullanim: per-country grafik olusturma.

    Returns
    -------
    DataFrame: iso_code, country, year, value, split
      split in {'train', 'val_actual', 'val_pred',
                'test_actual', 'test_pred', 'forecast'}
    """
    full_df  = data['full_df']
    val_df   = data['val_df']
    test_df  = data['test_df']
    feat     = data['feature_cols']
    model    = result['model']
    model_type = 'lstm' if result['name'] == 'LSTM' else 'sklearn'

    # Val ve test tahminleri ulke bazli indeksle
    def indexed_preds(split_df, preds_array):
        df_tmp = split_df[['iso_code', 'year']].copy()
        df_tmp['pred'] = preds_array
        return df_tmp.set_index(['iso_code', 'year'])['pred']

    val_preds_idx  = indexed_preds(val_df,  result['pred_val'])
    test_preds_idx = indexed_preds(test_df, result['pred_test'])

    rows = []
    for iso in full_df['iso_code'].unique():
        df_c  = full_df[full_df['iso_code'] == iso].sort_values('year')
        cname = df_c['country'].iloc[0]

        for _, row in df_c.iterrows():
            yr = row['year']
            v  = row['elec_per_capita']

            if yr <= 2015:
                split = 'train'
                rows.append({'iso_code': iso, 'country': cname,
                             'year': yr, 'value': v, 'split': split})
            elif yr <= 2018:
                rows.append({'iso_code': iso, 'country': cname,
                             'year': yr, 'value': v, 'split': 'val_actual'})
                try:
                    vp = val_preds_idx.loc[(iso, yr)]
                    rows.append({'iso_code': iso, 'country': cname,
                                 'year': yr, 'value': vp, 'split': 'val_pred'})
                except KeyError:
                    pass
            else:  # 2019-2022
                rows.append({'iso_code': iso, 'country': cname,
                             'year': yr, 'value': v, 'split': 'test_actual'})
                try:
                    tp = test_preds_idx.loc[(iso, yr)]
                    rows.append({'iso_code': iso, 'country': cname,
                                 'year': yr, 'value': tp, 'split': 'test_pred'})
                except KeyError:
                    pass

        # Gelecek tahminleri
        fc = forecast_df[forecast_df['iso_code'] == iso]
        for _, frow in fc.iterrows():
            rows.append({'iso_code': iso, 'country': cname,
                         'year': int(frow['year']),
                         'value': frow['predicted_elec_per_capita'],
                         'split': 'forecast'})

    return pd.DataFrame(rows)
