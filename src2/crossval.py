"""
Walk-Forward Cross-Validation

n_splits=5 fold, egitim verisi icinde (1993-2015).
Her fold icin:
  - Train R2  : modelin egitildigi veriye ne kadar uyuyor
  - Val R2    : hic gormedigi yillarda ne kadar basarili

Train R2 >> Val R2 ise overfitting var demektir.
Train R2 ~ Val R2 ise model genelliyor demektir.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


def compute_cv_preds(data: dict, result: dict, n_splits: int = 5):
    """
    Walk-forward CV calistir.

    Returns
    -------
    cv_preds  : DataFrame  [iso_code, year, fold, predicted, actual]
                Val parcasindaki tahminler (per-ulke grafik icin)

    fold_stats: DataFrame  [fold, val_years, train_r2, val_r2, train_n, val_n]
                Her fold'un ozet metrikleri (CV grafigi icin)
    """
    model_name = result['name']
    print(f"\n=== WALK-FORWARD CV ({n_splits} fold, model={model_name}) ===")

    X_tr  = data['X_train']
    y_tr  = data['y_train']
    tdf   = data['train_df'].reset_index(drop=True)

    cv_model_cls = 'xgb' if model_name == 'XGBoost' else 'rf'
    if model_name == 'LSTM':
        print("  LSTM icin RF kullaniliyor (hiz)...")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    pred_rows  = []
    stat_rows  = []

    for fold_idx, (tr_idx, va_idx) in enumerate(tscv.split(X_tr)):
        # --- Model egit ---
        if cv_model_cls == 'xgb' and 'best_params' in result:
            params = {k: v for k, v in result['best_params'].items()
                      if k not in ('early_stopping_rounds', 'eval_metric')}
            m = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
        else:
            m = RandomForestRegressor(n_estimators=150, max_depth=8,
                                      random_state=42, n_jobs=-1)

        m.fit(X_tr[tr_idx], y_tr[tr_idx])

        # --- Tahminler ---
        train_pred = m.predict(X_tr[tr_idx])
        val_pred   = m.predict(X_tr[va_idx])

        # --- R2 ---
        train_r2 = r2_score(y_tr[tr_idx], train_pred)
        val_r2   = r2_score(y_tr[va_idx], val_pred)

        # --- Val yil araligi ---
        fold_years = sorted(tdf.iloc[va_idx]['year'].unique())
        yr_str = f"{fold_years[0]}-{fold_years[-1]}" if fold_years else "?"

        print(f"  Fold {fold_idx+1}: val yillari={yr_str} | "
              f"Train R2={train_r2:.4f} | Val R2={val_r2:.4f}")

        # --- Val tahminlerini sakla (per-ulke grafigi icin) ---
        vrows = tdf.iloc[va_idx][['iso_code', 'year', 'elec_per_capita']].copy()
        vrows['predicted'] = val_pred
        vrows['fold']      = fold_idx
        vrows = vrows.rename(columns={'elec_per_capita': 'actual'})
        pred_rows.append(vrows)

        stat_rows.append({
            'fold'      : fold_idx,
            'val_years' : yr_str,
            'train_r2'  : round(train_r2, 4),
            'val_r2'    : round(val_r2,   4),
            'train_n'   : len(tr_idx),
            'val_n'     : len(va_idx),
        })

    cv_preds   = pd.concat(pred_rows, ignore_index=True)
    fold_stats = pd.DataFrame(stat_rows)

    print(f"\n  Ort. Val R2  : {fold_stats['val_r2'].mean():.4f}")
    print(f"  Ort. Train R2: {fold_stats['train_r2'].mean():.4f}")

    return cv_preds, fold_stats
